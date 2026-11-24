# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.2.0",
#   "huggingface_hub>=0.30.0",
#   "mlx>=0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-lm>=0.25.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-vlm>=0.8.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "torch>=2.4.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "safetensors>=0.4.0",
#   "scipy>=1.13.0",
#   "tokenizers>=0.21.0",
#   "transformers>=5.18.0",
# ]
# ///
"""10. Standalone v4.0 Deterministic Evaluation — Dev-45, Audited Holdout A & Prospective Blind Holdout B.

Strict provenance & scientific discipline:
  - Verifies 5-artifact SHA-256 integrity (`eo_search_lib.py`, `06_eval_rerankers.py`, `full_catalogue.parquet`,
    `full_embeddings_bf16.npz`, `full_bm25.duckdb`) against immutable `artifacts/frozen_v4_config.json`.
  - Programmatically asserts 0 overlapping record_ids across all 3 suites (Dev-45, Holdout A, Holdout B).
  - Complete unconfounded 2x2 Stage-1 factorization (`{BM25_PreFilter, Dense, HybridRRF} x {Raw, GenericCollapse}`)
    plus orthogonal `NaiveStrict + Collapse` (Top-50 retained) and all 4 Stage-2 configurations (including
    unconfounded fixed-k=50 `08b_JinaV35_Top50_Pure`).
  - Solo `batch_size=1` query encoding (bit-identical to CLI & script 07) with persisted real measured latencies
    and device-agnostic auto-selection (`--device auto|mlx|cuda|mps|cpu`).
  - Evaluates all 6 pre-registered hypotheses with paired bootstrap 95% CIs, exact sign-flip permutation p-values,
    Holm-Bonferroni FWER adjustment, Kish G_eff, and small-sample Student-t MDE80 across all 5 pre-registered metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DuckDBLexicalIndex,
    SearchFilters,
    apply_filters_mask,
    collapse_product_families,
    deterministic_topk,
    holm_bonferroni_adjust,
    load_dense_embedder,
    resolve_device,
    rrf_fuse,
)

_ev06_spec = importlib.util.spec_from_file_location("ev06", SCRIPT_DIR / "06_eval_rerankers.py")
_ev06 = importlib.util.module_from_spec(_ev06_spec)
assert _ev06_spec and _ev06_spec.loader
_ev06_spec.loader.exec_module(_ev06)

_b02_spec = importlib.util.spec_from_file_location("b02", SCRIPT_DIR / "02_build_policy_benchmark.py")
_b02 = importlib.util.module_from_spec(_b02_spec)
assert _b02_spec and _b02_spec.loader
_b02_spec.loader.exec_module(_b02)

load_reranker = _ev06.load_reranker
clear_device_cache = _ev06.clear_device_cache
format_reranker_doc = _ev06.format_reranker_doc
blend_rrf_and_rerank = _ev06.blend_rrf_and_rerank
EO_RERANK_INSTRUCTION = _ev06.EO_RERANK_INSTRUCTION
DEV_BENCHMARK_SPECS = _b02.BENCHMARK_SPECS


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_ranking_metrics(
    ranked_doc_indices: list[int],
    rid_by_idx: list[str],
    family_rid_sets: dict[str, set[str]],
) -> dict[str, float]:
    rid_to_fams: dict[str, set[str]] = {}
    for fid, rset in family_rid_sets.items():
        for rid in rset:
            rid_to_fams.setdefault(rid, set()).add(fid)

    n_fams = max(1, len(family_rid_sets))
    first_rel_rank = 0
    seen_fams: set[str] = set()
    dcg10 = 0.0
    dcg20 = 0.0
    fam_at = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    hit_at = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}

    for rank, doc_idx in enumerate(ranked_doc_indices[:50], start=1):
        fams = rid_to_fams.get(rid_by_idx[doc_idx])
        if fams:
            if first_rel_rank == 0:
                first_rel_rank = rank
            new_fams = fams - seen_fams
            if new_fams:
                seen_fams.update(new_fams)
                gain = float(len(new_fams))
                if rank <= 10:
                    dcg10 += gain / math.log2(rank + 1)
                if rank <= 20:
                    dcg20 += gain / math.log2(rank + 1)
        if rank in fam_at:
            fam_at[rank] = len(seen_fams)
            hit_at[rank] = 1.0 if len(seen_fams) > 0 else 0.0

    for k in (1, 5, 10, 20, 50):
        if len(ranked_doc_indices) < k:
            fam_at[k] = len(seen_fams)
            hit_at[k] = 1.0 if len(seen_fams) > 0 else 0.0

    idcg10 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 10) + 1))
    idcg20 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 20) + 1))
    return {
        "hit_1": hit_at[1],
        "hit_5": hit_at[5],
        "hit_10": hit_at[10],
        "hit_20": hit_at[20],
        "hit_50": hit_at[50],
        "fam_r_5": fam_at[5] / n_fams,
        "fam_r_10": fam_at[10] / n_fams,
        "fam_r_20": fam_at[20] / n_fams,
        "fam_r_50": fam_at[50] / n_fams,
        "mrr_10": (1.0 / first_rel_rank) if (0 < first_rel_rank <= 10) else 0.0,
        "fam_ndcg_10": dcg10 / idcg10 if idcg10 > 0 else 0.0,
        "fam_ndcg_20": dcg20 / idcg20 if idcg20 > 0 else 0.0,
    }


def cluster_bootstrap_and_perm(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    item_col: str = "item_id",
    cluster_col: str = "cluster_id",
    n_resamples: int = 10000,
    seed: int = 2026,
    n_hypotheses: int = 6,
) -> dict[str, float]:
    """Exact 1-to-1 paired cluster bootstrap 95% CI, permutation p-value, Kish G_eff & small-sample t MDE80."""
    a_sub = df_a[[item_col, cluster_col, metric]].copy()
    b_sub = df_b[[item_col, metric]].copy()
    a_sub[item_col] = a_sub[item_col].astype(str)
    b_sub[item_col] = b_sub[item_col].astype(str)
    a_sub[cluster_col] = a_sub[cluster_col].astype(str)

    m = a_sub.merge(b_sub, on=item_col, suffixes=("_a", "_b"), validate="one_to_one")
    assert len(m) == len(df_a) == len(df_b), f"Pairing mismatch: {len(m)} vs {len(df_a)}/{len(df_b)}"
    m["diff"] = m[f"{metric}_a"] - m[f"{metric}_b"]

    exact_diff = float(df_a[metric].mean() - df_b[metric].mean())
    grp = m.groupby(cluster_col, sort=True)["diff"].agg(["sum", "count"]).reset_index()
    c_sums = grp["sum"].to_numpy(dtype=np.float64)
    c_cnts = grp["count"].to_numpy(dtype=np.float64)
    n_c = len(c_sums)
    g_eff = float((c_cnts.sum() ** 2) / np.sum(c_cnts ** 2))
    obs = float(c_sums.sum() / c_cnts.sum())
    assert abs(obs - exact_diff) < 1e-12, f"Cluster mean mismatch: {obs} vs {exact_diff}"

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(n_resamples, n_c))
    boot_means = c_sums[idx].sum(axis=1) / c_cnts[idx].sum(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    se_boot = float(np.std(boot_means, ddof=1))

    df_eff = max(2.0, g_eff - 1.0)
    t_05 = float(student_t.ppf(1.0 - 0.05 / 2.0, df_eff))
    t_holm = float(student_t.ppf(1.0 - (0.05 / n_hypotheses) / 2.0, df_eff))
    t_80 = float(student_t.ppf(0.80, df_eff))
    mde80_alpha05 = (t_05 + t_80) * se_boot
    mde80_holm = (t_holm + t_80) * se_boot

    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n_c))
    perm_means = (signs * c_sums[None, :]).sum(axis=1) / c_cnts.sum()
    p_val = float((np.sum(np.abs(perm_means) >= abs(obs) - 1e-12) + 1) / (n_resamples + 1))
    return {
        "delta_pp": round(obs * 100.0, 2),
        "ci95_lo_pp": round(float(ci_lo) * 100.0, 2),
        "ci95_hi_pp": round(float(ci_hi) * 100.0, 2),
        "p_value": round(p_val, 5),
        "mde80_alpha05_pp": round(mde80_alpha05 * 100.0, 2),
        "mde80_holm6_pp": round(mde80_holm * 100.0, 2),
        "n_clusters": int(n_c),
        "g_eff": round(g_eff, 1),
    }


def resolve_dev_items_full(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Resolve Dev-45 benchmark specs on active canonical catalogue with strict disjointness assertion."""
    obs_re = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)
    active_mask = (~df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)) & (
        ~df["title"].fillna("").str.contains(obs_re, regex=True).to_numpy(dtype=bool, copy=True)
    )
    comb = df["title"].fillna("") + " || " + df["raw_id"].fillna("") + " || " + df["source_catalogue"].fillna("")
    items = []
    for sp in DEV_BENCHMARK_SPECS:
        fams_out = []
        for fam in sp["expected_families"]:
            hit = active_mask & comb.str.contains(re.compile(fam["regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                hit = hit & ~comb.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            idxs = np.flatnonzero(hit)
            assert len(idxs) > 0, f"Dev family 0 matches: {sp['query_id']} / {fam['family_id']}"
            fams_out.append({"family_id": fam["family_id"], "record_ids": df.iloc[idxs]["record_id"].tolist()})
        for i in range(len(fams_out)):
            for j in range(i + 1, len(fams_out)):
                inter = set(fams_out[i]["record_ids"]) & set(fams_out[j]["record_ids"])
                assert not inter, f"Dev overlap in {sp['query_id']}: {fams_out[i]['family_id']} vs {fams_out[j]['family_id']} ({len(inter)})"
        items.append({
            "query_id": sp["query_id"],
            "domain": sp.get("policy_framework", "Dev-45"),
            "query": sp["query"],
            "filters": sp.get("filters", {}),
            "expected_families": fams_out,
        })
    return items


def encode_queries_solo_timed(enc: Any, queries: list[str]) -> tuple[np.ndarray, list[float]]:
    """Encode queries one-by-one (`batch_size=1`) for exact CLI/07 parity & real per-query latency."""
    vecs_list = []
    ms_list = []
    for q in queries:
        t0 = time.perf_counter()
        qv = enc.encode_queries([q], max_length=256)[0]
        ms_list.append((time.perf_counter() - t0) * 1000.0)
        vecs_list.append(qv)
    return np.vstack(vecs_list), ms_list


def run_stage1_for_suite(
    items: list[dict[str, Any]],
    q_vecs: np.ndarray,
    q_enc_ms: list[float],
    df: pd.DataFrame,
    vecs: np.ndarray,
    generic_sigs: list[str],
    lex: DuckDBLexicalIndex,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run complete unconfounded 2x2 Stage-1 factorization + orthogonal NaiveStrict (Top-50) with measured timings."""
    s1 = cfg["stage1"]
    rrf_k = int(s1["rrf"]["k"])
    w_d = float(s1["rrf"]["w_dense"])
    w_b = float(s1["rrf"]["w_bm25"])
    cand_k = int(s1["rrf"]["candidate_k"])
    cos_th = float(s1["collapser"]["near_dup_cos"])
    sg = bool(s1["collapser"]["sensor_guard"])
    pool_k = int(cfg["stage2_frozen_systems"]["primary_best_quality"]["pool_k"])

    out: dict[str, list[Any]] = {
        "bm25_raw": [], "bm25_col": [],
        "dense_raw": [], "dense_col": [],
        "hybrid_raw": [], "hybrid_col50": [], "hybrid_strict_col50": [],
        "docs50": [],
        "ms_bm25_raw": [], "ms_bm25_col": [],
        "ms_dense_raw": [], "ms_dense_col": [],
        "ms_hybrid_raw": [], "ms_hybrid_col": [], "ms_hybrid_strict": [],
    }
    for qi, it in enumerate(items):
        qtxt = it["query"]
        qms = q_enc_ms[qi]
        flt_spec = dict(it.get("filters", {}))
        flt = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            provider_pool=it.get("provider_pool"),
            exclude_superseded=True,
            null_safe=True,
        )
        t_m0 = time.perf_counter()
        mask = apply_filters_mask(df, flt)
        m_ms = (time.perf_counter() - t_m0) * 1000.0

        t_d0 = time.perf_counter()
        d_res = deterministic_topk(vecs @ q_vecs[qi], k=cand_k, mask=mask)
        d_ms = (time.perf_counter() - t_d0) * 1000.0

        t_b0 = time.perf_counter()
        b_res = lex.search_bm25(qtxt, top_k=cand_k, eligible_mask=mask)
        b_ms = (time.perf_counter() - t_b0) * 1000.0

        t_bc0 = time.perf_counter()
        b_col = collapse_product_families(
            [(i, s, {}) for i, s in b_res], df, top_k=pool_k, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=sg
        )
        bc_ms = (time.perf_counter() - t_bc0) * 1000.0

        t_dc0 = time.perf_counter()
        d_col = collapse_product_families(
            [(i, s, {}) for i, s in d_res], df, top_k=pool_k, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=sg
        )
        dc_ms = (time.perf_counter() - t_dc0) * 1000.0

        t_f0 = time.perf_counter()
        fused = rrf_fuse(d_res, b_res, rrf_k=rrf_k, w_dense=w_d, w_bm25=w_b)
        f_ms = (time.perf_counter() - t_f0) * 1000.0

        t_c0 = time.perf_counter()
        col = collapse_product_families(
            fused, df, top_k=pool_k, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=sg
        )
        c_ms = (time.perf_counter() - t_c0) * 1000.0

        t_st0 = time.perf_counter()
        flt_st = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            provider_pool=it.get("provider_pool"),
            exclude_superseded=True,
            null_safe=False,
        )
        m_st = apply_filters_mask(df, flt_st)
        d_st = deterministic_topk(vecs @ q_vecs[qi], k=cand_k, mask=m_st)
        b_st = lex.search_bm25(qtxt, top_k=cand_k, eligible_mask=m_st)
        f_st = rrf_fuse(d_st, b_st, rrf_k=rrf_k, w_dense=w_d, w_bm25=w_b)
        c_st = collapse_product_families(
            f_st, df, top_k=pool_k, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=sg
        )
        st_ms = (time.perf_counter() - t_st0) * 1000.0

        c50 = [i for i, _, _ in col]
        out["bm25_raw"].append([i for i, _ in b_res[:pool_k]])
        out["bm25_col"].append([i for i, _, _ in b_col])
        out["dense_raw"].append([i for i, _ in d_res[:pool_k]])
        out["dense_col"].append([i for i, _, _ in d_col])
        out["hybrid_raw"].append([i for i, _, _ in fused[:pool_k]])
        out["hybrid_col50"].append(c50)
        out["hybrid_strict_col50"].append([i for i, _, _ in c_st])
        out["docs50"].append([format_reranker_doc(df.iloc[idx].to_dict(), max_chars=1200) for idx in c50])

        out["ms_bm25_raw"].append(m_ms + b_ms)
        out["ms_bm25_col"].append(m_ms + b_ms + bc_ms)
        out["ms_dense_raw"].append(qms + m_ms + d_ms)
        out["ms_dense_col"].append(qms + m_ms + d_ms + dc_ms)
        out["ms_hybrid_raw"].append(qms + m_ms + d_ms + b_ms + f_ms)
        out["ms_hybrid_col"].append(qms + m_ms + d_ms + b_ms + f_ms + c_ms)
        out["ms_hybrid_strict"].append(qms + st_ms)
    return out


PREREGISTERED_COMPS = [
    ("H1_Collapser: Hybrid+Col vs Hybrid_NoCol", "06_Stage1_HybridRRF_GenericCollapse", "05_Stage1_HybridRRF_NoCollapse"),
    ("H2_Fair_Hybrid vs BM25+Col", "06_Stage1_HybridRRF_GenericCollapse", "02_BM25_PreFilter_GenericCollapse"),
    ("H3_Fair_Hybrid vs Dense+Col", "06_Stage1_HybridRRF_GenericCollapse", "04_Dense_768d_GenericCollapse"),
    ("H4_JinaV35_Top50_Blend vs Stage1", "08_Frozen_Best_JinaV35_Top50_RRFBlend", "06_Stage1_HybridRRF_GenericCollapse"),
    ("H5_Qwen3_Top50_Blend vs Stage1", "10_Frozen_Apache2_Qwen3_Top50_RRFBlend", "06_Stage1_HybridRRF_GenericCollapse"),
    ("H6_JinaV35 vs Qwen3 (Top50 Blend)", "08_Frozen_Best_JinaV35_Top50_RRFBlend", "10_Frozen_Apache2_Qwen3_Top50_RRFBlend"),
]

PREREGISTERED_METRICS = ("fam_ndcg_10", "mrr_10", "fam_r_10", "hit_5", "hit_10")


def compute_sig_table(df_pq: pd.DataFrame, cluster_col: str = "cluster_id") -> pd.DataFrame:
    stat_rows = []
    for clab, sa, sb in PREREGISTERED_COMPS:
        da = df_pq[df_pq["system"] == sa]
        db = df_pq[df_pq["system"] == sb]
        rec: dict[str, Any] = {"comparison": clab}
        for met in PREREGISTERED_METRICS:
            st = cluster_bootstrap_and_perm(
                da, db, met, item_col="item_id", cluster_col=cluster_col, n_hypotheses=len(PREREGISTERED_COMPS)
            )
            rec[f"{met}_delta"] = st["delta_pp"]
            rec[f"{met}_ci95"] = f"[{st['ci95_lo_pp']:+.1f}, {st['ci95_hi_pp']:+.1f}]"
            rec[f"{met}_p"] = st["p_value"]
            if met == "fam_ndcg_10":
                rec["mde80_alpha05_pp"] = st["mde80_alpha05_pp"]
                rec["mde80_holm6_pp"] = st["mde80_holm6_pp"]
                rec["n_clusters"] = st["n_clusters"]
                rec["g_eff"] = st["g_eff"]
        stat_rows.append(rec)
    df_stats = pd.DataFrame(stat_rows)
    for met in PREREGISTERED_METRICS:
        df_stats[f"{met}_p_holm"] = holm_bonferroni_adjust(df_stats[f"{met}_p"].tolist())
    return df_stats


def summarize_and_save_suite(
    prefix: str,
    title: str,
    items: list[dict[str, Any]],
    sys_lists: dict[str, list[list[int]]],
    lat_map: dict[str, float],
    rid_by_idx: list[str],
    cluster_key: str = "query_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rankings = {
        sname: {
            str(it["query_id"]): [rid_by_idx[i] for i in ranked_all[qi]]
            for qi, it in enumerate(items)
        }
        for sname, ranked_all in sys_lists.items()
    }
    (ARTIFACTS_DIR / f"{prefix}_rankings.json").write_text(json.dumps(raw_rankings, indent=2), encoding="utf-8")

    rows = []
    for sname, ranked_all in sys_lists.items():
        for qi, it in enumerate(items):
            fam_sets = {f["family_id"]: set(f["record_ids"]) for f in it["expected_families"]}
            m = eval_ranking_metrics(ranked_all[qi], rid_by_idx, fam_sets)
            rows.append({
                "system": sname,
                "item_id": str(it["query_id"]),
                "cluster_id": str(it.get(cluster_key, it["query_id"])),
                "domain": str(it.get("domain", it.get("source_framework", "general"))),
                **m,
            })
    df_pq = pd.DataFrame(rows)
    df_pq.to_csv(ARTIFACTS_DIR / f"{prefix}_per_task.csv", index=False)

    mcols = ["hit_1", "hit_5", "hit_10", "hit_20", "hit_50", "fam_r_5", "fam_r_10", "fam_r_20", "fam_r_50", "mrr_10", "fam_ndcg_10", "fam_ndcg_20"]
    df_sum = (df_pq.groupby("system", sort=True)[mcols].mean() * 100.0).round(2).reset_index()
    df_sum["total_ms"] = df_sum["system"].map(lambda s: round(lat_map[s], 1))

    s1_name = "06_Stage1_HybridRRF_GenericCollapse"
    base_idx = df_pq[df_pq["system"] == s1_name].set_index("item_id")["fam_ndcg_10"]
    wtls = []
    for sname in df_sum["system"]:
        if sname == s1_name:
            wtls.append("-")
            continue
        cur_idx = df_pq[df_pq["system"] == sname].set_index("item_id")["fam_ndcg_10"]
        d = cur_idx - base_idx
        wtls.append(f"{int((d > 1e-4).sum())}W/{int((d.abs() <= 1e-4).sum())}T/{int((d < -1e-4).sum())}L")
    df_sum["wtl_vs_s1"] = wtls
    df_sum.to_csv(ARTIFACTS_DIR / f"{prefix}_summary.csv", index=False)

    df_stats = compute_sig_table(df_pq, cluster_col="cluster_id")
    df_stats.to_csv(ARTIFACTS_DIR / f"{prefix}_significance.csv", index=False)

    print("\n" + "=" * 108, flush=True)
    print(title, flush=True)
    print("=" * 108, flush=True)
    print(df_sum[["system", "hit_5", "hit_10", "fam_r_5", "fam_r_10", "fam_r_20", "fam_r_50", "mrr_10", "fam_ndcg_10", "total_ms", "wtl_vs_s1"]].to_string(index=False), flush=True)
    print(
        f"\nPre-Registered Primary Metric (`fam_ndcg_10`) — Paired Bootstrap 95% CIs, Holm p_holm & t-MDE80 "
        f"(G={df_stats['n_clusters'].iloc[0]}, G_eff={df_stats['g_eff'].iloc[0]}):",
        flush=True,
    )
    for _, r in df_stats.iterrows():
        print(
            f"  {r['comparison']:42s} | nDCG@10 {r['fam_ndcg_10_delta']:+5.2f} {r['fam_ndcg_10_ci95']} "
            f"(p={r['fam_ndcg_10_p']:.4f}, p_holm={r['fam_ndcg_10_p_holm']:.4f}, MDE_05={r['mde80_alpha05_pp']:.1f}pp, MDE_holm={r['mde80_holm6_pp']:.1f}pp)",
            flush=True,
        )
    return df_pq, df_sum, df_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone v4.0 Deterministic Evaluation")
    parser.add_argument("--device", default="auto", choices=["auto", "mlx", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    device = resolve_device(args.device)

    t_all0 = time.perf_counter()
    print("=" * 112, flush=True)
    print(f"10. v4.0-STANDALONE EVALUATION — DEV-45 (N=45), AUDITED HOLDOUT A (N=35) & PROSPECTIVE BLIND HOLDOUT B (N=60) [{device}]", flush=True)
    print("=" * 112, flush=True)

    cfg = json.loads((ARTIFACTS_DIR / "frozen_v4_config.json").read_text(encoding="utf-8"))
    expected_sha = cfg["code_and_index_sha256"]
    cat_pq = ARTIFACTS_DIR / "full_catalogue.parquet"
    if not cat_pq.exists():
        cat_pq = SCRIPT_DIR.parent / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
    check_paths = {
        "eo_search_lib.py": SCRIPT_DIR / "eo_search_lib.py",
        "06_eval_rerankers.py": SCRIPT_DIR / "06_eval_rerankers.py",
        "full_catalogue.parquet": cat_pq,
        "full_embeddings_bf16.npz": ARTIFACTS_DIR / "full_embeddings_bf16.npz",
        "full_bm25.duckdb": ARTIFACTS_DIR / "full_bm25.duckdb",
    }
    for k, p in check_paths.items():
        actual = sha256_file(p)
        assert actual == expected_sha[k], f"SHA-256 mismatch for {k}: {actual} vs {expected_sha[k]}"
    print(
        f"[OK] Verified {cfg['protocol_version']} 5-artifact SHA-256 integrity "
        f"(primary_metric={cfg['preregistered_primary_metric']}, near_dup_cos={cfg['stage1']['collapser']['near_dup_cos']}, "
        f"bm25_deterministic=True, sensor_guard=True).",
        flush=True,
    )

    df = pd.read_parquet(cat_pq)
    rid_by_idx = df["record_id"].tolist()
    z = np.load(ARTIFACTS_DIR / "full_embeddings_bf16.npz", allow_pickle=True)
    vecs = z["embeddings_f16"].astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    generic_sigs = [str(s) for s in z["family_sigs"].tolist()]
    assert not any(s.startswith("fam:") for s in generic_sigs), "Legacy fam: sig detected in npz!"
    lex = DuckDBLexicalIndex(str(ARTIFACTS_DIR / "full_bm25.duckdb"), read_only=True)

    dev_items = resolve_dev_items_full(df)
    ha_items = json.loads((ARTIFACTS_DIR / "holdout_a_external_benchmark.json").read_text(encoding="utf-8"))["queries"]
    hb_items = json.loads((ARTIFACTS_DIR / "holdout_b_blind_benchmark.json").read_text(encoding="utf-8"))["queries"]

    # Programmatic 3-way cross-suite disjointness verification
    dev_rids = {r for it in dev_items for f in it["expected_families"] for r in f["record_ids"]}
    ha_rids = {r for it in ha_items for f in it["expected_families"] for r in f["record_ids"]}
    hb_rids = {r for it in hb_items for f in it["expected_families"] for r in f["record_ids"]}
    assert dev_rids.isdisjoint(ha_rids), "Dev-45 ∩ Holdout A overlap!"
    assert dev_rids.isdisjoint(hb_rids), "Dev-45 ∩ Holdout B overlap!"
    assert ha_rids.isdisjoint(hb_rids), "Holdout A ∩ Holdout B overlap!"
    print(
        f"[OK] 3-way cross-suite disjointness PASSED: Dev-45 ({len(dev_rids)} rids), "
        f"Holdout A ({len(ha_rids)} rids), Blind Holdout B ({len(hb_rids)} rids) — 0 overlaps.",
        flush=True,
    )

    # Stage 0: Solo batch_size=1 query encoding
    enc, _, _ = load_dense_embedder(device=device)
    _ = enc.encode_queries(["warmup satellite query"], max_length=256)
    dev_qv, dev_qms = encode_queries_solo_timed(enc, [x["query"] for x in dev_items])
    ha_qv, ha_qms = encode_queries_solo_timed(enc, [x["query"] for x in ha_items])
    hb_qv, hb_qms = encode_queries_solo_timed(enc, [x["query"] for x in hb_items])
    del enc
    clear_device_cache(device)

    all_qms = dev_qms + ha_qms + hb_qms
    mean_q_ms = float(np.mean(all_qms))
    print(f"[Stage 0] Solo query encoding done across {len(all_qms)} queries (mean real latency={mean_q_ms:.2f} ms).", flush=True)

    # Stage 1 across all 3 standalone suites
    s1_dev = run_stage1_for_suite(dev_items, dev_qv, dev_qms, df, vecs, generic_sigs, lex, cfg)
    s1_ha = run_stage1_for_suite(ha_items, ha_qv, ha_qms, df, vecs, generic_sigs, lex, cfg)
    s1_hb = run_stage1_for_suite(hb_items, hb_qv, hb_qms, df, vecs, generic_sigs, lex, cfg)
    all_s1_ms = list(s1_dev["ms_hybrid_col"]) + list(s1_ha["ms_hybrid_col"]) + list(s1_hb["ms_hybrid_col"])
    mean_s1_ms = float(np.mean(all_s1_ms))
    print(
        f"[Stage 1] Done across Dev ({len(dev_items)}), HA ({len(ha_items)}), Blind HB ({len(hb_items)}) | "
        f"real mean Stage-1 latency={mean_s1_ms:.1f} ms",
        flush=True,
    )

    # SEQUENTIAL STAGE 2A: Load ONLY Jina v3.5, score all 3 suites, then free
    print(f"\n[Stage 2A] Loading ONLY Jina v3.5 ({device}, sequential lifecycle)...", flush=True)
    s2_best = cfg["stage2_frozen_systems"]["primary_best_quality"]
    s2_fast = cfg["stage2_frozen_systems"]["fast_sweet_spot"]
    fast_k = int(s2_fast["pool_k"])
    jina, _, _ = load_reranker("jina-v3.5", device=device, max_doc_length=int(s2_best["max_doc_length"]))
    _ = jina.score_candidates("warmup", ["warmup 1", "warmup 2"])

    def _run_jina_suite(label: str, items: list[dict[str, Any]], s1_pack: dict[str, Any]):
        b50_l, p50_l, p25_l, ms50_l, ms25_l = [], [], [], [], []
        for qi, it in enumerate(items):
            qtxt = it["query"]
            c50, d50 = s1_pack["hybrid_col50"][qi], s1_pack["docs50"][qi]
            t0 = time.perf_counter()
            sc50 = jina.score_candidates(qtxt, d50) if d50 else []
            ms50_l.append((time.perf_counter() - t0) * 1000.0)
            b50_l.append(
                blend_rrf_and_rerank(
                    c50, sc50,
                    rrf_k=int(s2_best["blend"]["rrf_k"]),
                    w_rrf=float(s2_best["blend"]["w_rrf"]),
                    w_rerank=float(s2_best["blend"]["w_rerank"]),
                ) if c50 else []
            )
            p50_l.append([c50[int(k)] for k in np.argsort(-np.asarray(sc50, dtype=np.float64), kind="stable")] if c50 else [])

            c25, d25 = c50[:fast_k], d50[:fast_k]
            t0 = time.perf_counter()
            sc25 = jina.score_candidates(qtxt, d25) if d25 else []
            ms25_l.append((time.perf_counter() - t0) * 1000.0)
            p25_l.append([c25[int(k)] for k in np.argsort(-np.asarray(sc25, dtype=np.float64), kind="stable")] if c25 else [])
            if (qi + 1) % 20 == 0 or (qi + 1) == len(items):
                print(f"  [Jina {label}] {qi+1:2d}/{len(items)} | Top-50={np.mean(ms50_l):.0f} ms | Top-25={np.mean(ms25_l):.0f} ms", flush=True)
        return b50_l, p50_l, p25_l, ms50_l, ms25_l

    j_dev_b50, j_dev_p50, j_dev_p25, j_dev_m50, j_dev_m25 = _run_jina_suite("Dev-45", dev_items, s1_dev)
    j_ha_b50, j_ha_p50, j_ha_p25, j_ha_m50, j_ha_m25 = _run_jina_suite("Holdout-A(Audited)", ha_items, s1_ha)
    j_hb_b50, j_hb_p50, j_hb_p25, j_hb_m50, j_hb_m25 = _run_jina_suite("Holdout-B(Blind)", hb_items, s1_hb)
    del jina
    clear_device_cache(device)

    # SEQUENTIAL STAGE 2B: Load ONLY Qwen3-Reranker-0.6B (Apache-2.0, log-odds margin + stable sort)
    print(f"\n[Stage 2B] Loading ONLY Qwen3-Reranker-0.6B ({device}, sequential lifecycle)...", flush=True)
    s2_qwen = cfg["stage2_frozen_systems"]["permissive_apache2"]
    qwen, _, _ = load_reranker("qwen3-0.6b", device=device, verify_exactness=(device == "mlx"))
    _ = qwen.score_candidates("warmup", ["warmup 1"], batch_size=1, max_length=128)

    def _run_qwen_suite(label: str, items: list[dict[str, Any]], s1_pack: dict[str, Any]):
        b50_l, ms50_l = [], []
        for qi, it in enumerate(items):
            qtxt = it["query"]
            c50, d50 = s1_pack["hybrid_col50"][qi], s1_pack["docs50"][qi]
            t0 = time.perf_counter()
            sc50 = qwen.score_candidates(qtxt, d50, batch_size=8, max_length=int(s2_qwen["max_length"])) if d50 else []
            ms50_l.append((time.perf_counter() - t0) * 1000.0)
            b50_l.append(
                blend_rrf_and_rerank(
                    c50, sc50,
                    rrf_k=int(s2_qwen["blend"]["rrf_k"]),
                    w_rrf=float(s2_qwen["blend"]["w_rrf"]),
                    w_rerank=float(s2_qwen["blend"]["w_rerank"]),
                ) if c50 else []
            )
            if (qi + 1) % 20 == 0 or (qi + 1) == len(items):
                print(f"  [Qwen3 {label}] {qi+1:2d}/{len(items)} | Top-50={np.mean(ms50_l):.0f} ms", flush=True)
        return b50_l, ms50_l

    q_dev_b50, q_dev_m50 = _run_qwen_suite("Dev-45", dev_items, s1_dev)
    q_ha_b50, q_ha_m50 = _run_qwen_suite("Holdout-A(Audited)", ha_items, s1_ha)
    q_hb_b50, q_hb_m50 = _run_qwen_suite("Holdout-B(Blind)", hb_items, s1_hb)
    del qwen
    clear_device_cache(device)

    def _pack_systems(s1_p, jb50, jp50, jp25, qb50):
        return {
            "01_BM25_PreFilter_Raw": s1_p["bm25_raw"],
            "02_BM25_PreFilter_GenericCollapse": s1_p["bm25_col"],
            "03_Dense_768d_Raw": s1_p["dense_raw"],
            "04_Dense_768d_GenericCollapse": s1_p["dense_col"],
            "05_Stage1_HybridRRF_NoCollapse": s1_p["hybrid_raw"],
            "06_Stage1_HybridRRF_GenericCollapse": s1_p["hybrid_col50"],
            "07_Stage1_HybridRRF_NaiveStrict_Collapse": s1_p["hybrid_strict_col50"],
            "08_Frozen_Best_JinaV35_Top50_RRFBlend": jb50,
            "08b_JinaV35_Top50_Pure": jp50,
            "09_Frozen_Fast_JinaV35_Top25_Pure": jp25,
            "10_Frozen_Apache2_Qwen3_Top50_RRFBlend": qb50,
        }

    def _pack_lats(s1_p, jm50, jm25, qm50):
        s1m = float(np.mean(s1_p["ms_hybrid_col"]))
        return {
            "01_BM25_PreFilter_Raw": float(np.mean(s1_p["ms_bm25_raw"])),
            "02_BM25_PreFilter_GenericCollapse": float(np.mean(s1_p["ms_bm25_col"])),
            "03_Dense_768d_Raw": float(np.mean(s1_p["ms_dense_raw"])),
            "04_Dense_768d_GenericCollapse": float(np.mean(s1_p["ms_dense_col"])),
            "05_Stage1_HybridRRF_NoCollapse": float(np.mean(s1_p["ms_hybrid_raw"])),
            "06_Stage1_HybridRRF_GenericCollapse": s1m,
            "07_Stage1_HybridRRF_NaiveStrict_Collapse": float(np.mean(s1_p["ms_hybrid_strict"])),
            "08_Frozen_Best_JinaV35_Top50_RRFBlend": s1m + float(np.mean(jm50)),
            "08b_JinaV35_Top50_Pure": s1m + float(np.mean(jm50)),
            "09_Frozen_Fast_JinaV35_Top25_Pure": s1m + float(np.mean(jm25)),
            "10_Frozen_Apache2_Qwen3_Top50_RRFBlend": s1m + float(np.mean(qm50)),
        }

    def _save_blend_ablation(suite_tag: str, df_pq: pd.DataFrame) -> dict[str, Any]:
        da = df_pq[df_pq["system"] == "08_Frozen_Best_JinaV35_Top50_RRFBlend"]
        db = df_pq[df_pq["system"] == "08b_JinaV35_Top50_Pure"]
        row_rec: dict[str, Any] = {"suite": suite_tag, "comparison": "JinaV35_Top50_RRFBlend vs JinaV35_Top50_Pure (Unconfounded Fixed k=50)"}
        for met in PREREGISTERED_METRICS:
            st = cluster_bootstrap_and_perm(da, db, met, cluster_col="cluster_id", n_hypotheses=1)
            row_rec[f"{met}_delta"] = st["delta_pp"]
            row_rec[f"{met}_ci95"] = f"[{st['ci95_lo_pp']:+.1f}, {st['ci95_hi_pp']:+.1f}]"
            row_rec[f"{met}_p"] = st["p_value"]
        return row_rec

    blend_rows = []
    dev_pq, _, _ = summarize_and_save_suite(
        "dev45_v4",
        "DEV-45 v4.0-STANDALONE (DEVELOPMENT SET — 45 QUERIES, UNCONFOUNDED 2x2) [%]",
        dev_items,
        _pack_systems(s1_dev, j_dev_b50, j_dev_p50, j_dev_p25, q_dev_b50),
        _pack_lats(s1_dev, j_dev_m50, j_dev_m25, q_dev_m50),
        rid_by_idx,
    )
    blend_rows.append(_save_blend_ablation("dev45_v4", dev_pq))

    ha_pq, _, _ = summarize_and_save_suite(
        "holdout_a_v4",
        "HOLDOUT A v4.0-STANDALONE (RETROSPECTIVE GT-AUDITED EXTERNAL DOMAINS — 35 QUERIES, 0 OVERLAPS) [%]",
        ha_items,
        _pack_systems(s1_ha, j_ha_b50, j_ha_p50, j_ha_p25, q_ha_b50),
        _pack_lats(s1_ha, j_ha_m50, j_ha_m25, q_ha_m50),
        rid_by_idx,
    )
    blend_rows.append(_save_blend_ablation("holdout_a_v4", ha_pq))

    hb_pq, _, _ = summarize_and_save_suite(
        "holdout_b_v4",
        "PROSPECTIVE SINGLE-SHOT BLIND HOLDOUT B v4.0-STANDALONE (60 DISJOINT UNSEEN EO QUERIES, 10 DOMAINS, G=G_eff=60) [%]",
        hb_items,
        _pack_systems(s1_hb, j_hb_b50, j_hb_p50, j_hb_p25, q_hb_b50),
        _pack_lats(s1_hb, j_hb_m50, j_hb_m25, q_hb_m50),
        rid_by_idx,
    )
    blend_rows.append(_save_blend_ablation("holdout_b_v4", hb_pq))

    # Domain-stratified summary on Blind Holdout B
    hb_dom = (
        hb_pq[hb_pq["system"].isin([
            "02_BM25_PreFilter_GenericCollapse",
            "04_Dense_768d_GenericCollapse",
            "06_Stage1_HybridRRF_GenericCollapse",
            "08_Frozen_Best_JinaV35_Top50_RRFBlend",
            "10_Frozen_Apache2_Qwen3_Top50_RRFBlend",
        ])]
        .groupby(["domain", "system"], sort=True)[["hit_5", "hit_10", "fam_r_10", "mrr_10", "fam_ndcg_10"]]
        .mean() * 100.0
    ).round(2).reset_index()
    hb_dom.to_csv(ARTIFACTS_DIR / "holdout_b_v4_by_domain.csv", index=False)

    pd.DataFrame(blend_rows).to_csv(ARTIFACTS_DIR / "v4_unconfounded_top50_blend_vs_pure.csv", index=False)
    print("\n--- SUPPLEMENTARY UNCONFOUNDED TOP-50 BLEND vs PURE ABLATION (FIXED k=50) ---", flush=True)
    print(pd.DataFrame(blend_rows)[["suite", "fam_ndcg_10_delta", "fam_ndcg_10_ci95", "fam_ndcg_10_p", "mrr_10_delta"]].to_string(index=False), flush=True)

    completed_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rt_manifest = {
        "protocol_version": cfg["protocol_version"],
        "device": device,
        "frozen_at_utc": cfg["frozen_at_utc"],
        "completed_at_utc": completed_utc,
        "strictly_prospective_timestamp_order": cfg["frozen_at_utc"] < completed_utc,
        "total_queries_encoded_solo": len(all_qms),
        "single_query_encode_ms_mean": round(mean_q_ms, 2),
        "single_query_encode_ms_p50": round(float(np.median(all_qms)), 2),
        "single_query_encode_ms_p95": round(float(np.percentile(all_qms, 95)), 2),
        "stage1_mean_ms_all_queries": round(mean_s1_ms, 2),
        "stage2_jina50_mean_ms_all_queries": round(float(np.mean(j_dev_m50 + j_ha_m50 + j_hb_m50)), 1),
        "stage2_jina25_mean_ms_all_queries": round(float(np.mean(j_dev_m25 + j_ha_m25 + j_hb_m25)), 1),
        "stage2_qwen50_mean_ms_all_queries": round(float(np.mean(q_dev_m50 + q_ha_m50 + q_hb_m50)), 1),
        "suite_sizes": {
            "dev45": len(dev_items),
            "holdout_a_audited": len(ha_items),
            "holdout_b_prospective_blind": len(hb_items),
        },
        "total_wall_minutes": round((time.perf_counter() - t_all0) / 60.0, 2),
    }
    (ARTIFACTS_DIR / "v4_runtime_manifest.json").write_text(json.dumps(rt_manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] v4.0-standalone full suite finished in {rt_manifest['total_wall_minutes']:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
