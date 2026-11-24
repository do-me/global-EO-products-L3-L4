# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
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
"""11. Reproducible Scientific Audits for v4.0-standalone (Zero Orphan Claims, Zero Drift).

Generates persistent, reproducible code + JSON/CSV artifacts for every auxiliary technical claim:
  0. DuckDB BM25 bit-determinism audit across all 45 Dev-45 queries x 2 filter modes (NullSafe & NaiveStrict).
  1. Dense encoder exactness audit (on Apple Silicon MLX: fast text-only path vs stock `mlx_vlm` for BOTH
     `L <= 512` and `L > 513` sliding-window mask branches, plus `batch=1` vs `batch=16` parity).
  2. Jina v3.5 causal listwise position-bias audit (Top-50 forward vs 180 deg reversed across 15 Dev-45 queries).
  3. Cross-family false-collapse audit (`near_dup_cos` + `sensor_guard=True` on title/raw_id/platform_sensor)
     using the exact `fused[:scan_limit]` (scan_limit=200) denominator across all 3 standalone suites
     (`dev45`, `holdout_a`, `holdout_b_blind`).
  4. True NESTED Leave-One-Framework-Out (LOFO-CV) on Dev-45 — exact 3-tier tie-breaker matching `07`,
     asserting 0.000000 pp drift vs `10`, reporting nested 3-preset `{08,09,10}`, 4-config `{08,08b,09,10}`,
     and fixed-primary Jina Top-50 Blend.
  5. Empirical Dev-45 `NullSafe + Stage-2` vs `NaiveStrict + Stage-2` comparison (`all_45` & `grid_filtered_13`).
  6. Live CLI `--json` stdout purity, `--bm25-only` fallback, & interactive Stage-1/Stage-2 latency persistence.
  7. Cross-backend MLX vs PyTorch parity audit across all 3 neural models (`embeddinggemma-2`,
     `jina-reranker-v3.5`, `Qwen3-Reranker-0.6B`), confirming numerical & rank-order equivalence across hardware.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_MODEL_BF16,
    DEFAULT_PT_EMBED_MODEL,
    DuckDBLexicalIndex,
    SearchFilters,
    apply_filters_mask,
    build_eo_document,
    collapse_product_families,
    compute_file_sha256,
    deterministic_topk,
    ensure_search_indexes,
    filter_benchmark_family_mask,
    load_dense_embedder,
    resolve_device,
    rrf_fuse,
    verify_index_alignment,
)

_ev06_spec = importlib.util.spec_from_file_location("ev06", SCRIPT_DIR / "06_eval_rerankers.py")
_ev06 = importlib.util.module_from_spec(_ev06_spec)
assert _ev06_spec and _ev06_spec.loader
_ev06_spec.loader.exec_module(_ev06)

load_reranker = _ev06.load_reranker
clear_device_cache = _ev06.clear_device_cache
EO_RERANK_INSTRUCTION = _ev06.EO_RERANK_INSTRUCTION
format_reranker_doc = _ev06.format_reranker_doc
blend_rrf_and_rerank = _ev06.blend_rrf_and_rerank
DEV_BENCHMARK_SPECS = json.loads((ARTIFACTS_DIR / "policy_retrieval_benchmark.json").read_text(encoding="utf-8"))["queries"]


def map_framework_group(fw: str) -> str:
    fl = fw.lower()
    if "biodiversity strategy" in fl:
        return "F1_EU_BDS_2030"
    if "urban climate adaptation" in fl:
        return "F2_KCEO_Urban_Adaptation"
    if "global stocktake" in fl or "paris agreement" in fl:
        return "F3_UNFCCC_GST"
    if "sdg" in fl:
        return "F4_UN_SDGs"
    if "kunming-montreal" in fl or "gbf" in fl:
        return "F5_CBD_GBF_A2"
    return "F6_USGEO_SNWG"


def eval_stage1_metrics(
    ranked_doc_indices: list[int], rid_by_idx: list[str], family_rid_sets: dict[str, set[str]]
) -> dict[str, float]:
    rid_to_fams: dict[str, set[str]] = {}
    for fid, rset in family_rid_sets.items():
        for rid in rset:
            rid_to_fams.setdefault(rid, set()).add(fid)
    n_fams = max(1, len(family_rid_sets))
    seen: set[str] = set()
    dcg10 = 0.0
    first_rel = 0
    fam50 = 0
    for rank, doc_idx in enumerate(ranked_doc_indices[:50], start=1):
        fams = rid_to_fams.get(rid_by_idx[doc_idx])
        if fams:
            if first_rel == 0:
                first_rel = rank
            new_f = fams - seen
            if new_f:
                seen.update(new_f)
                if rank <= 10:
                    dcg10 += float(len(new_f)) / math.log2(rank + 1)
        if rank == 50:
            fam50 = len(seen)
    if len(ranked_doc_indices) < 50:
        fam50 = len(seen)
    idcg10 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 10) + 1))
    return {
        "fam_ndcg_10": dcg10 / idcg10 if idcg10 > 0 else 0.0,
        "fam_r_50": fam50 / n_fams,
        "mrr_10": (1.0 / first_rel) if (0 < first_rel <= 10) else 0.0,
    }


def audit_0_bm25_determinism(df: pd.DataFrame, lex: DuckDBLexicalIndex, cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify DuckDB BM25 query determinism, content SHA reproducibility, and permutation rejection."""
    print("\n--- [AUDIT 0] DuckDB BM25 Determinism, Content SHA & Record-ID Alignment Audit ---")
    ns_diffs = 0
    st_diffs = 0
    for sp in DEV_BENCHMARK_SPECS:
        fs = sp.get("filters", {})
        for mode_ns in (True, False):
            flt = SearchFilters(
                geo_scope=fs.get("geo_scope"),
                max_grid_m=fs.get("max_grid_m"),
                strict_eo_only=bool(fs.get("strict_eo_only", False)),
                exclude_superseded=True,
                null_safe=mode_ns,
            )
            mask = apply_filters_mask(df, flt)
            r1 = [i for i, _ in lex.search_bm25(sp["query"], top_k=300, eligible_mask=mask)]
            r2 = [i for i, _ in lex.search_bm25(sp["query"], top_k=300, eligible_mask=mask)]
            if r1 != r2:
                if mode_ns:
                    ns_diffs += 1
                else:
                    st_diffs += 1
    assert ns_diffs == 0 and st_diffs == 0, f"BM25 non-determinism detected: ns={ns_diffs}, st={st_diffs}"

    # Verify deterministic bm25_content_sha256 across fresh independent DuckDB builds + locked manifest
    content_sha_main = lex.compute_content_sha256()
    locked_sha = cfg.get("locked_artifact_sha256", cfg["code_and_index_sha256"]).get("bm25_content_sha256", "")
    if locked_sha:
        assert content_sha_main == locked_sha, f"bm25_content_sha256 mismatch: {content_sha_main} vs {locked_sha}"
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_db1 = Path(tmpd) / "rebuild1.duckdb"
        tmp_db2 = Path(tmpd) / "rebuild2.duckdb"
        for dbp in (tmp_db1, tmp_db2):
            b = DuckDBLexicalIndex(str(dbp))
            b.build_from_dataframe(df)
            b.con.close()
        r_b1 = DuckDBLexicalIndex(str(tmp_db1), read_only=True)
        r_b2 = DuckDBLexicalIndex(str(tmp_db2), read_only=True)
        csha1 = r_b1.compute_content_sha256()
        csha2 = r_b2.compute_content_sha256()
        r_b1.con.close()
        r_b2.con.close()
    assert csha1 == csha2 == content_sha_main, f"Rebuild bm25_content_sha256 drifted: {csha1} vs {csha2} vs {content_sha_main}"

    # Verify same_length_different_order permutation raises RuntimeError in verify_index_alignment()
    perm_idx = [1, 0] + list(range(2, len(df)))
    df_perm = df.iloc[perm_idx].reset_index(drop=True)
    perm_rejected = False
    try:
        verify_index_alignment(df_perm, lex_index=lex)
    except RuntimeError:
        perm_rejected = True
    assert perm_rejected, "verify_index_alignment failed to reject same_length_different_order DataFrame!"

    # Verify same record_ids with mutated metadata raises RuntimeError in verify_index_alignment()
    df_mut = df.copy()
    df_mut.loc[df_mut.index[0], "title"] = str(df_mut.loc[df_mut.index[0], "title"]) + " [MUTATED]"
    meta_mut_rejected = False
    try:
        verify_index_alignment(df_mut, lex_index=lex)
    except RuntimeError:
        meta_mut_rejected = True
    assert meta_mut_rejected, "verify_index_alignment failed to reject same-IDs mutated-metadata DataFrame!"

    # Verify ensure_search_indexes(..., allow_download=False) accepts equivalent local BM25 rebuild by semantic content
    with tempfile.TemporaryDirectory() as tmpd2:
        tmp_art = Path(tmpd2)
        man_copy = json.loads((ARTIFACTS_DIR / "full_index_manifest.json").read_text(encoding="utf-8"))
        # Force file-byte SHA mismatch on full_bm25.duckdb while preserving exact semantic content hashes
        man_copy["sha256"]["full_bm25.duckdb"] = "0" * 64
        (tmp_art / "full_index_manifest.json").write_text(json.dumps(man_copy, indent=2), encoding="utf-8")
        cat_pq = ARTIFACTS_DIR / "full_catalogue.parquet"
        cat_sha_tmp = compute_file_sha256(cat_pq) if cat_pq.exists() else ""
        b_loc = DuckDBLexicalIndex(str(tmp_art / "full_bm25.duckdb"))
        b_loc.build_from_dataframe(df, catalogue_sha256=cat_sha_tmp)
        b_loc.con.close()
        ensure_search_indexes(tmp_art, require_dense=False, allow_download=False)
        local_semantic_ok = True

    # Verify Verified hard-limits filters: min_end_year=2035 & documented output grid exclusion of OTD / GCMD bands
    m_2035 = apply_filters_mask(df, SearchFilters(min_end_year=2035, hard_limits=True))
    ey_2035 = pd.to_numeric(df.loc[m_2035, "temporal_end_year"], errors="coerce")
    hard_2035_ok = bool(ey_2035.notna().all() and (ey_2035 >= 2035).all())
    assert hard_2035_ok, f"Hard min_end_year=2035 admitted invalid rows: {len(ey_2035)}"

    m_grid12k = apply_filters_mask(df, SearchFilters(geo_scope="global", max_grid_m=12000, hard_limits=True))
    sub_grid12k = df.loc[m_grid12k]
    otd_excluded = "12_ceos_cgms_wgclimate_ecv::WGCLIMATE_CDR_11956" not in set(sub_grid12k["record_id"])
    gcmd_excluded = "gcmd_coarse_resolution_band" not in set(sub_grid12k["adjudicated_spatial_resolution_source"].fillna(""))
    ecv_excluded = "ecv_inventory_horizontal_resolution" not in set(sub_grid12k["adjudicated_spatial_resolution_source"].fillna(""))
    hard_grid_ok = bool(otd_excluded and gcmd_excluded and ecv_excluded)
    assert hard_grid_ok, f"Hard grid filter failed: otd={otd_excluded}, gcmd={gcmd_excluded}, ecv={ecv_excluded}"

    # Verify Annual NLCD 6 distinct delivered variables & GLC_FCS30D annual vs five-year do not collapse
    nlcd_idxs = df.index[df["raw_id"].fillna("").str.contains("ANNUAL_NLCD/", regex=False)].tolist()
    nlcd_ranked = [(int(i), 1.0 - 0.01 * k, {}) for k, i in enumerate(nlcd_idxs)]
    nlcd_col = collapse_product_families(nlcd_ranked, df, top_k=20, near_dup_cos=0.94, sensor_guard=True)
    nlcd_distinct_ok = len(nlcd_col) == len(nlcd_idxs) and len(nlcd_idxs) >= 6
    assert nlcd_distinct_ok, f"Annual NLCD collapsed distinct variables: {len(nlcd_col)} vs {len(nlcd_idxs)}"

    glc_idxs = df.index[df["raw_id"].fillna("").str.contains("GLC-FCS30D/", regex=False)].tolist()
    glc_ranked = [(int(i), 1.0 - 0.01 * k, {}) for k, i in enumerate(glc_idxs)]
    glc_col = collapse_product_families(glc_ranked, df, top_k=20, near_dup_cos=0.94, sensor_guard=True)
    glc_distinct_ok = len(glc_col) == len(glc_idxs) and len(glc_idxs) >= 2
    assert glc_distinct_ok, f"GLC_FCS30D collapsed annual & five-year variants: {len(glc_col)} vs {len(glc_idxs)}"

    out = {
        "n_queries_checked": len(DEV_BENCHMARK_SPECS),
        "nullsafe_top300_diffs": ns_diffs,
        "naivestrict_top300_diffs": st_diffs,
        "total_paired_checks": len(DEV_BENCHMARK_SPECS) * 2,
        "bm25_content_sha256": content_sha_main,
        "rebuild_content_sha_identical": True,
        "same_length_different_order_rejected": perm_rejected,
        "same_ids_changed_metadata_rejected": meta_mut_rejected,
        "local_rebuild_semantic_preflight_accepted": local_semantic_ok,
        "verified_hard_limits_min_end_year_2035_zero_violations": hard_2035_ok,
        "verified_hard_limits_grid_excludes_otd_and_gcmd": hard_grid_ok,
        "annual_nlcd_distinct_variables_preserved": nlcd_distinct_ok,
        "glc_fcs30d_cadence_variants_preserved": glc_distinct_ok,
        "exact_determinism_passed": True,
    }
    (ARTIFACTS_DIR / "audit_bm25_determinism.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"  [OK] Top-300 BM25 diffs: NullSafe={ns_diffs}/45 | NaiveStrict={st_diffs}/45 | "
        f"rebuild content SHA identical={content_sha_main[:12]}... | perm/meta rejected=True | "
        f"local semantic preflight=True | hard-limits & variable/cadence grouping=PASS"
    )
    return out


def audit_1_dense_exactness(df: pd.DataFrame, device: str) -> dict[str, Any]:
    """Verify fast MLX text-only path vs stock mlx_vlm (L<=512 & L>513) and batch=1 vs batch=16 parity."""
    print(f"\n--- [AUDIT 1] Dense Encoder Exactness & Batch Parity ({device}) ---")
    enc, _, _ = load_dense_embedder(device=device)
    sample_docs = [build_eo_document(df.iloc[i].to_dict()) for i in range(0, 160, 10)]
    if device == "mlx":
        import mlx.core as mx

        fast_b16 = enc.encode_texts(sample_docs, batch_size=16, max_length=512, sort_by_length=False, use_compile=False, async_pipe=False)
        fast_b1 = np.vstack([
            enc.encode_texts([d], batch_size=1, max_length=512, sort_by_length=False, use_compile=False, async_pipe=False)
            for d in sample_docs
        ])
        cos_b1_b16 = np.sum(fast_b1 * fast_b16, axis=1)
        max_diff_b1_b16 = float(np.max(np.abs(fast_b1 - fast_b16)))

        fast_fn = enc._get_forward_fn(False)
        stock_diffs_512 = []
        for d in sample_docs[:4]:
            toks = enc.tokenizer(d, truncation=True, max_length=512, return_tensors="np")
            ids = mx.array(toks["input_ids"].astype(np.int32))
            mask = mx.array(toks["attention_mask"].astype(np.int32))
            out_stock = enc.model(input_ids=ids, attention_mask=mask)
            e_stock = out_stock.text_embeds if hasattr(out_stock, "text_embeds") else out_stock
            e_fast = fast_fn(ids, mask)
            mx.eval(e_stock, e_fast)
            stock_diffs_512.append(float(np.max(np.abs(np.asarray(e_stock, dtype=np.float32) - np.asarray(e_fast, dtype=np.float32)))))

        long_docs = []
        for i in range(len(df)):
            doc = build_eo_document(df.iloc[i].to_dict())
            toks = enc.tokenizer(doc, truncation=True, max_length=640, return_tensors="np")
            if toks["input_ids"].shape[1] >= 580:
                long_docs.append((doc, int(toks["input_ids"].shape[1])))
                if len(long_docs) >= 2:
                    break
        assert len(long_docs) >= 2, "Could not find 2 corpus docs with >580 tokens for L>513 sliding-window audit"
        stock_diffs_gt513 = []
        long_lens = []
        for doc, seq_len in long_docs:
            toks = enc.tokenizer(doc, truncation=True, max_length=640, return_tensors="np")
            ids = mx.array(toks["input_ids"].astype(np.int32))
            mask = mx.array(toks["attention_mask"].astype(np.int32))
            out_stock = enc.model(input_ids=ids, attention_mask=mask)
            e_stock = out_stock.text_embeds if hasattr(out_stock, "text_embeds") else out_stock
            e_fast = fast_fn(ids, mask)
            mx.eval(e_stock, e_fast)
            stock_diffs_gt513.append(float(np.max(np.abs(np.asarray(e_stock, dtype=np.float32) - np.asarray(e_fast, dtype=np.float32)))))
            long_lens.append(seq_len)

        out = {
            "device": device,
            "model": DEFAULT_MODEL_BF16,
            "n_docs_checked_le512": len(stock_diffs_512),
            "max_abs_diff_stock_vs_fast_le512": round(float(max(stock_diffs_512)), 7),
            "n_docs_checked_gt513_sliding_window": len(stock_diffs_gt513),
            "gt513_token_lengths": long_lens,
            "max_abs_diff_stock_vs_fast_gt513": round(float(max(stock_diffs_gt513)), 7),
            "batch1_vs_batch16_max_abs_diff": round(max_diff_b1_b16, 6),
            "batch1_vs_batch16_min_cosine": round(float(np.min(cos_b1_b16)), 6),
            "batch1_vs_batch16_mean_cosine": round(float(np.mean(cos_b1_b16)), 6),
        }
    else:
        fast_b16 = enc.encode_texts(sample_docs, batch_size=16, max_length=512, sort_by_length=False)
        fast_b1 = np.vstack([enc.encode_texts([d], batch_size=1, max_length=512, sort_by_length=False) for d in sample_docs])
        cos_b1_b16 = np.sum(fast_b1 * fast_b16, axis=1)
        out = {
            "device": device,
            "model": DEFAULT_PT_EMBED_MODEL,
            "batch1_vs_batch16_max_abs_diff": round(float(np.max(np.abs(fast_b1 - fast_b16))), 6),
            "batch1_vs_batch16_min_cosine": round(float(np.min(cos_b1_b16)), 6),
            "batch1_vs_batch16_mean_cosine": round(float(np.mean(cos_b1_b16)), 6),
        }

    del enc
    clear_device_cache(device)
    (ARTIFACTS_DIR / "audit_mlx_bit_exactness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  [OK] {json.dumps(out)}")
    return out


def audit_2_jina_position_bias(df: pd.DataFrame, cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Causal listwise position-bias audit: Top-50 forward vs 180 deg reversed across 15 diverse Dev-45 queries."""
    print(f"\n--- [AUDIT 2] Jina v3.5 Causal Listwise Position-Bias Audit (N=15, Top-50 Fwd vs Rev, {device}) ---")
    rankings = json.loads((ARTIFACTS_DIR / "dev45_v4_rankings.json").read_text(encoding="utf-8"))
    s1_key = "06_Stage1_HybridRRF_GenericCollapse"
    rid_to_idx = {r: i for i, r in enumerate(df["record_id"].tolist())}

    jina, _, _ = load_reranker("jina-v3.5", device=device, max_doc_length=int(cfg["stage2_frozen_systems"]["primary_best_quality"]["max_doc_length"]))
    _ = jina.score_candidates("warmup", ["warmup 1", "warmup 2"])

    rows = []
    for sp in DEV_BENCHMARK_SPECS[::3][:15]:
        qid = sp["query_id"]
        qtxt = sp["query"]
        rids = rankings[s1_key][qid][:50]
        docs = [format_reranker_doc(df.iloc[rid_to_idx[r]].to_dict(), max_chars=1200) for r in rids]
        sc_fwd = np.asarray(jina.score_candidates(qtxt, docs), dtype=np.float64)
        sc_rev_raw = np.asarray(jina.score_candidates(qtxt, list(reversed(docs))), dtype=np.float64)
        sc_rev = sc_rev_raw[::-1]

        rho, _ = spearmanr(sc_fwd, sc_rev)
        top5_fwd = set(np.argsort(-sc_fwd, kind="stable")[:5].tolist())
        top5_rev = set(np.argsort(-sc_rev, kind="stable")[:5].tolist())
        top1_agree = int(np.argsort(-sc_fwd, kind="stable")[0] == np.argsort(-sc_rev, kind="stable")[0])
        first25_minus_last25_fwd = float(sc_fwd[:25].mean() - sc_fwd[25:].mean())
        first25_minus_last25_rev = float(sc_rev[:25].mean() - sc_rev[25:].mean())
        recency_bias = 0.5 * (first25_minus_last25_rev - first25_minus_last25_fwd)
        rows.append({
            "query_id": qid,
            "n_docs": len(docs),
            "spearman_rho": round(float(rho), 4),
            "top5_overlap_pct": round(len(top5_fwd & top5_rev) / 5.0 * 100.0, 1),
            "top1_exact_agree": top1_agree,
            "fwd_first25_minus_last25": round(first25_minus_last25_fwd, 5),
            "rev_first25_minus_last25": round(first25_minus_last25_rev, 5),
            "causal_recency_boost": round(recency_bias, 5),
        })

    del jina
    clear_device_cache(device)

    df_pb = pd.DataFrame(rows)
    df_pb.to_csv(ARTIFACTS_DIR / "audit_jina_position_bias.csv", index=False)
    summary = {
        "n_queries": len(df_pb),
        "mean_spearman_rho": round(float(df_pb["spearman_rho"].mean()), 4),
        "median_spearman_rho": round(float(df_pb["spearman_rho"].median()), 4),
        "min_spearman_rho": round(float(df_pb["spearman_rho"].min()), 4),
        "mean_top5_overlap_pct": round(float(df_pb["top5_overlap_pct"].mean()), 1),
        "top1_exact_agreement_pct": round(float(df_pb["top1_exact_agree"].mean() * 100.0), 1),
        "mean_causal_recency_boost": round(float(df_pb["causal_recency_boost"].mean()), 5),
    }
    (ARTIFACTS_DIR / "audit_jina_position_bias.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"  Mean Spearman rho={summary['mean_spearman_rho']} (min={summary['min_spearman_rho']}) | "
        f"Top-5 overlap={summary['mean_top5_overlap_pct']}% | Top-1 agree={summary['top1_exact_agreement_pct']}% | "
        f"Causal recency boost={summary['mean_causal_recency_boost']:+.5f}"
    )
    return summary


def audit_3_false_collapse(
    df: pd.DataFrame,
    vecs: np.ndarray,
    generic_sigs: list[str],
    lex: DuckDBLexicalIndex,
    cfg: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Audit cross-family GT false collapses using exact `fused[:scan_limit]` (scan_limit=200) across all 3 suites."""
    print("\n--- [AUDIT 3] Cross-Family False-Collapse Audit (Exact scan_limit=200 Denominator, All 3 Suites) ---")
    cos_th = float(cfg["stage1"]["collapser"]["near_dup_cos"])
    sg = bool(cfg["stage1"]["collapser"]["sensor_guard"])
    top_k = int(cfg["stage2_frozen_systems"]["primary_best_quality"]["pool_k"])
    scan_limit = max(top_k * 3, 200)
    rid_by_idx = df["record_id"].tolist()
    titles = df["title"].tolist()

    dev_items = []
    for sp in DEV_BENCHMARK_SPECS:
        fams_out = [{"family_id": fam["family_id"], "record_ids": fam["record_ids"]} for fam in sp["expected_families"]]
        dev_items.append({"query_id": sp["query_id"], "query": sp["query"], "filters": sp.get("filters", {}), "expected_families": fams_out})

    ha_items = json.loads((ARTIFACTS_DIR / "holdout_a_external_benchmark.json").read_text(encoding="utf-8"))["queries"]
    hb_items = json.loads((ARTIFACTS_DIR / "holdout_b_blind_benchmark.json").read_text(encoding="utf-8"))["queries"]

    enc, _, _ = load_dense_embedder(device=device)
    _ = enc.encode_queries(["warmup satellite query"], max_length=256)
    dev_qvecs = np.vstack([enc.encode_queries([x["query"]], max_length=256)[0] for x in dev_items])

    report = {}
    for sname, items, pre_qv in [
        ("dev45", dev_items, dev_qvecs),
        ("holdout_a", ha_items, None),
        ("holdout_b_blind", hb_items, None),
    ]:
        gt_in_scan200 = 0
        gt_in_top50_or_swallowed = 0
        gt_fams_swallowed = []
        for qi, it in enumerate(items):
            qv = pre_qv[qi] if pre_qv is not None else enc.encode_queries([it["query"]], max_length=256)[0]
            flt_spec = it.get("filters", {})
            flt = SearchFilters(
                geo_scope=flt_spec.get("geo_scope"),
                max_grid_m=flt_spec.get("max_grid_m"),
                strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
                provider_pool=it.get("provider_pool"),
                exclude_superseded=True,
                null_safe=True,
            )
            mask = apply_filters_mask(df, flt)
            d_res = deterministic_topk(vecs @ qv, k=300, mask=mask)
            b_res = lex.search_bm25(it["query"], top_k=300, eligible_mask=mask)
            fused = rrf_fuse(d_res, b_res, rrf_k=60, w_dense=1.0, w_bm25=0.6)
            col = collapse_product_families(
                fused, df, top_k=top_k, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=sg, query_text=it["query"]
            )

            scanned_rids = {rid_by_idx[i] for i, _, _ in fused[:scan_limit]}
            kept_rids = {rid_by_idx[i] for i, _, _ in col}
            item_id = it["query_id"]
            for fam in it["expected_families"]:
                fset = set(fam["record_ids"])
                if not (fset & scanned_rids):
                    continue
                gt_in_scan200 += 1
                if fset & kept_rids:
                    gt_in_top50_or_swallowed += 1
                    continue
                swallowed_by = None
                for doc_idx, _, meta in col:
                    for sib in meta.get("collapsed_siblings", []):
                        if sib["record_id"] in fset:
                            s_idx = int(sib["doc_idx"])
                            same_gstem = generic_sigs[doc_idx] == generic_sigs[s_idx]
                            cos_sim = float(np.dot(vecs[doc_idx], vecs[s_idx]))
                            swallowed_by = {
                                "kept_title": titles[doc_idx],
                                "kept_rid": rid_by_idx[doc_idx],
                                "swallowed_rid": sib["record_id"],
                                "swallowed_title": titles[s_idx],
                                "collapse_mechanism": "gstem" if same_gstem else "cosine",
                                "cosine_sim": round(cos_sim, 4),
                            }
                            break
                    if swallowed_by:
                        break
                if swallowed_by:
                    gt_in_top50_or_swallowed += 1
                    gt_fams_swallowed.append({"item_id": item_id, "family_id": fam["family_id"], **swallowed_by})

        report[sname] = {
            "near_dup_cos": cos_th,
            "sensor_guard": sg,
            "scan_limit": scan_limit,
            "gt_families_present_in_scan200": gt_in_scan200,
            "gt_families_reached_top50_or_swallowed": gt_in_top50_or_swallowed,
            "gt_families_falsely_swallowed": len(gt_fams_swallowed),
            "false_collapse_rate_scan200_pct": round(len(gt_fams_swallowed) / max(1, gt_in_scan200) * 100.0, 2),
            "false_collapse_rate_active_top50_pct": round(len(gt_fams_swallowed) / max(1, gt_in_top50_or_swallowed) * 100.0, 2),
            "cases": gt_fams_swallowed,
        }
        print(
            f"  [{sname:18s}] Swallowed: {len(gt_fams_swallowed)}/{gt_in_scan200} scan200 "
            f"({report[sname]['false_collapse_rate_scan200_pct']}%) | "
            f"{len(gt_fams_swallowed)}/{gt_in_top50_or_swallowed} active-top50 ({report[sname]['false_collapse_rate_active_top50_pct']}%)"
        )

    del enc
    clear_device_cache(device)
    (ARTIFACTS_DIR / "audit_false_collapse.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, dev_items, dev_qvecs


def audit_4_nested_lofo_cv(
    df: pd.DataFrame,
    vecs: np.ndarray,
    generic_sigs: list[str],
    lex: DuckDBLexicalIndex,
    dev_items: list[dict[str, Any]],
    dev_qvecs: np.ndarray,
) -> None:
    """TRUE NESTED LOFO-CV on Dev-45: exact 3-tier tie-breaker matching `07` & 0.000000 pp drift assertion vs `10`."""
    print("\n--- [AUDIT 4] TRUE NESTED Leave-One-Framework-Out (LOFO-CV) on Dev-45 (Zero-Drift Verified) ---")
    rid_by_idx = df["record_id"].tolist()
    fw_map = {sp["query_id"]: map_framework_group(sp["policy_framework"]) for sp in DEV_BENCHMARK_SPECS}

    cos_grid = [1.0, 0.98, 0.97, 0.96, 0.94]
    s1_grid_rows = []
    for qi, it in enumerate(dev_items):
        flt_spec = it.get("filters", {})
        flt = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            exclude_superseded=True,
            null_safe=True,
        )
        mask = apply_filters_mask(df, flt)
        d_res = deterministic_topk(vecs @ dev_qvecs[qi], k=300, mask=mask)
        b_res = lex.search_bm25(it["query"], top_k=300, eligible_mask=mask)
        fused = rrf_fuse(d_res, b_res, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        fam_sets = {f["family_id"]: set(f["record_ids"]) for f in it["expected_families"]}
        for cos_th in cos_grid:
            col = collapse_product_families(
                fused, df, top_k=50, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=True, query_text=it["query"]
            )
            ms = eval_stage1_metrics([i for i, _, _ in col], rid_by_idx, fam_sets)
            s1_grid_rows.append({
                "item_id": it["query_id"],
                "framework_group": fw_map[it["query_id"]],
                "near_dup_cos": cos_th,
                "s1_ndcg10": ms["fam_ndcg_10"],
                "s1_fam_r50": ms["fam_r_50"],
            })
    df_s1g = pd.DataFrame(s1_grid_rows)

    df_pq = pd.read_csv(ARTIFACTS_DIR / "dev45_v4_per_task.csv")
    df_pq["framework_group"] = df_pq["item_id"].map(fw_map)
    s1_name = "06_Stage1_HybridRRF_GenericCollapse"
    primary_s2 = "08_Frozen_Best_JinaV35_Top50_RRFBlend"
    cand_systems = [
        "08_Frozen_Best_JinaV35_Top50_RRFBlend",
        "09_Frozen_Fast_JinaV35_Top25_Pure",
        "10_Frozen_Apache2_Qwen3_Top50_RRFBlend",
    ]
    rows = []
    for held_fw in sorted(df_pq["framework_group"].unique()):
        train_s1 = df_s1g[df_s1g["framework_group"] != held_fw]
        s1_means = train_s1.groupby("near_dup_cos")[["s1_ndcg10", "s1_fam_r50"]].mean().reset_index()
        nested_cos = float(
            s1_means.sort_values(["s1_ndcg10", "s1_fam_r50", "near_dup_cos"], ascending=[False, False, False]).iloc[0]["near_dup_cos"]
        )
        assert nested_cos == 0.94, f"Fold {held_fw} selected cos={nested_cos} != 0.94"
        test_s1_nested = float(df_s1g[(df_s1g["framework_group"] == held_fw) & (df_s1g["near_dup_cos"] == nested_cos)]["s1_ndcg10"].mean())

        train_df = df_pq[df_pq["framework_group"] != held_fw]
        test_df = df_pq[df_pq["framework_group"] == held_fw]
        base_t = test_df[test_df["system"] == s1_name]
        s1_from_10 = float(base_t["fam_ndcg_10"].mean())
        assert abs(test_s1_nested - s1_from_10) < 1e-12, (
            f"Determinism drift on {held_fw}: 11={test_s1_nested:.6f} vs 10={s1_from_10:.6f}"
        )

        s2_train = train_df[train_df["system"].isin(cand_systems)].groupby("system")[["fam_ndcg_10", "mrr_10"]].mean().reset_index()
        winner = str(s2_train.sort_values(["fam_ndcg_10", "mrr_10", "system"], ascending=[False, False, True]).iloc[0]["system"])
        win_t = test_df[test_df["system"] == winner]
        prim_t = test_df[test_df["system"] == primary_s2]

        cand_with_08b = cand_systems + ["08b_JinaV35_Top50_Pure"]
        s2_train_4 = train_df[train_df["system"].isin(cand_with_08b)].groupby("system")[["fam_ndcg_10", "mrr_10"]].mean().reset_index()
        winner_4 = str(s2_train_4.sort_values(["fam_ndcg_10", "mrr_10", "system"], ascending=[False, False, True]).iloc[0]["system"])
        win4_t = test_df[test_df["system"] == winner_4]

        rows.append({
            "held_out_framework": held_fw,
            "n_queries": len(base_t),
            "nested_s1_cos_selected": nested_cos,
            "nested_s1_ndcg10": round(s1_from_10 * 100.0, 2),
            "nested_s2_selected": winner,
            "lofo_s2_ndcg10": round(float(win_t["fam_ndcg_10"].mean()) * 100.0, 2),
            "delta_ndcg10": round((float(win_t["fam_ndcg_10"].mean()) - s1_from_10) * 100.0, 2),
            "nested_s2_with_08b_selected": winner_4,
            "lofo_s2_with_08b_ndcg10": round(float(win4_t["fam_ndcg_10"].mean()) * 100.0, 2),
            "lofo_s2_with_08b_delta": round((float(win4_t["fam_ndcg_10"].mean()) - s1_from_10) * 100.0, 2),
            "fixed_jina50_ndcg10": round(float(prim_t["fam_ndcg_10"].mean()) * 100.0, 2),
            "fixed_jina50_delta": round((float(prim_t["fam_ndcg_10"].mean()) - s1_from_10) * 100.0, 2),
            "lofo_mrr10": round(float(win_t["mrr_10"].mean()) * 100.0, 2),
            "lofo_fam_r10": round(float(win_t["fam_r_10"].mean()) * 100.0, 2),
        })
    df_lofo = pd.DataFrame(rows)
    df_lofo.to_csv(ARTIFACTS_DIR / "dev45_v4_nested_lofo_cv.csv", index=False)
    print("  [OK] Exact 0.000000 pp drift assertion PASSED across all 6 folds vs 10_run_blind_holdouts.py")
    print(df_lofo.to_string(index=False))


def audit_5_dev45_filter_stage2(
    df: pd.DataFrame,
    dev_items: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Empirical Dev-45 test of NullSafe + Stage-2 vs NaiveStrict + Stage-2 (`all_45` & `grid_filtered_13`)."""
    print(f"\n--- [AUDIT 5] Empirical Dev-45 NullSafe vs NaiveStrict + Stage-2 Reranking ({device}) ---")
    rid_by_idx = df["record_id"].tolist()
    rid_to_idx = {r: i for i, r in enumerate(rid_by_idx)}
    rankings = json.loads((ARTIFACTS_DIR / "dev45_v4_rankings.json").read_text(encoding="utf-8"))
    df_pq = pd.read_csv(ARTIFACTS_DIR / "dev45_v4_per_task.csv")
    s2_best = cfg["stage2_frozen_systems"]["primary_best_quality"]
    s2_qwen = cfg["stage2_frozen_systems"]["permissive_apache2"]

    st_c50, st_d50 = [], []
    for it in dev_items:
        rids = rankings["07_Stage1_HybridRRF_NaiveStrict_Collapse"][it["query_id"]][:50]
        c50 = [rid_to_idx[r] for r in rids]
        st_c50.append(c50)
        st_d50.append([format_reranker_doc(df.iloc[idx].to_dict(), max_chars=1200) for idx in c50])

    jina, _, _ = load_reranker("jina-v3.5", device=device, max_doc_length=int(s2_best["max_doc_length"]))
    _ = jina.score_candidates("warmup", ["warmup 1", "warmup 2"])
    st_jina_ranked = []
    for qi, it in enumerate(dev_items):
        sc = jina.score_candidates(it["query"], st_d50[qi]) if st_d50[qi] else []
        st_jina_ranked.append(
            blend_rrf_and_rerank(st_c50[qi], sc, rrf_k=int(s2_best["blend"]["rrf_k"]), w_rrf=float(s2_best["blend"]["w_rrf"]), w_rerank=float(s2_best["blend"]["w_rerank"]))
            if st_c50[qi] else []
        )
    del jina
    clear_device_cache(device)

    qwen, _, _ = load_reranker("qwen3-0.6b", device=device, verify_exactness=False)
    _ = qwen.score_candidates("warmup", ["warmup 1"], batch_size=1, max_length=128)
    st_qwen_ranked = []
    for qi, it in enumerate(dev_items):
        sc = qwen.score_candidates(it["query"], st_d50[qi], batch_size=8, max_length=int(s2_qwen["max_length"])) if st_d50[qi] else []
        st_qwen_ranked.append(
            blend_rrf_and_rerank(st_c50[qi], sc, rrf_k=int(s2_qwen["blend"]["rrf_k"]), w_rrf=float(s2_qwen["blend"]["w_rrf"]), w_rerank=float(s2_qwen["blend"]["w_rerank"]))
            if st_c50[qi] else []
        )
    del qwen
    clear_device_cache(device)

    grid_qids = {it["query_id"] for it in dev_items if it.get("filters", {}).get("max_grid_m") is not None}
    st_rows = []
    for qi, it in enumerate(dev_items):
        fsets = {f["family_id"]: set(f["record_ids"]) for f in it["expected_families"]}
        for label, rk in [("NaiveStrict_Jina50_Blend", st_jina_ranked[qi]), ("NaiveStrict_Qwen3_50_Blend", st_qwen_ranked[qi])]:
            ms = eval_stage1_metrics(rk, rid_by_idx, fsets)
            st_rows.append({"system": label, "item_id": it["query_id"], "is_grid": it["query_id"] in grid_qids, **ms})
    df_st = pd.DataFrame(st_rows)

    out: dict[str, Any] = {}
    for sl_name, qmask_fn in [("all_45", lambda s: s == s), ("grid_filtered_13", lambda s: s.isin(grid_qids))]:
        sl_out = {}
        for label, sys_col in [
            ("NullSafe_Stage1", "06_Stage1_HybridRRF_GenericCollapse"),
            ("NaiveStrict_Stage1", "07_Stage1_HybridRRF_NaiveStrict_Collapse"),
            ("NullSafe_Jina50_Blend", "08_Frozen_Best_JinaV35_Top50_RRFBlend"),
            ("NullSafe_Qwen3_50_Blend", "10_Frozen_Apache2_Qwen3_Top50_RRFBlend"),
        ]:
            sub = df_pq[(df_pq["system"] == sys_col) & qmask_fn(df_pq["item_id"])]
            sl_out[label] = {
                "fam_ndcg_10": round(float(sub["fam_ndcg_10"].mean()) * 100.0, 2),
                "mrr_10": round(float(sub["mrr_10"].mean()) * 100.0, 2),
                "fam_r_50": round(float(sub["fam_r_50"].mean()) * 100.0, 2),
            }
        for label in ("NaiveStrict_Jina50_Blend", "NaiveStrict_Qwen3_50_Blend"):
            sub = df_st[(df_st["system"] == label) & qmask_fn(df_st["item_id"])]
            sl_out[label] = {
                "fam_ndcg_10": round(float(sub["fam_ndcg_10"].mean()) * 100.0, 2),
                "mrr_10": round(float(sub["mrr_10"].mean()) * 100.0, 2),
                "fam_r_50": round(float(sub["fam_r_50"].mean()) * 100.0, 2),
            }
        out[sl_name] = sl_out
        print(f"  [{sl_name}]")
        for k, v in sl_out.items():
            print(f"    {k:26s}: nDCG@10={v['fam_ndcg_10']:5.2f}% | MRR@10={v['mrr_10']:5.2f}% | FamR@50={v['fam_r_50']:5.2f}%")

    (ARTIFACTS_DIR / "audit_dev45_filter_stage2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def audit_6_cli_latency(device: str) -> dict[str, Any]:
    """Verify pure JSON stdout, BM25-only fallback, and interactive CLI Stage-1/Stage-2 latencies."""
    print(f"\n--- [AUDIT 6] Live CLI `--json` Purity & Interactive Latency Audit (`04_search_cli.py`, {device}) ---")
    cli_path = str(SCRIPT_DIR / "04_search_cli.py")
    probes = [
        (
            "bm25_only_zero_gpu",
            [sys.executable, cli_path, "--query", "Global flood hazard mapping", "--bm25-only", "--top-k", "3", "--json"],
        ),
        (
            "pool_filtered_stage1_none",
            [sys.executable, cli_path, "--device", device, "--query", "Global flood hazard mapping", "--pool", "copernicus_services", "--rerank", "none", "--top-k", "3", "--json"],
        ),
        (
            "global_grid_filtered_stage1_none",
            [sys.executable, cli_path, "--device", device, "--query", "Global 30 meter multitemporal inland surface water occurrence, seasonality, and dynamic surface water extent", "--geo", "global", "--max-grid-m", "300", "--rerank", "none", "--top-k", "10", "--json"],
        ),
        (
            "pool_filtered_jina_pool25",
            [sys.executable, cli_path, "--device", device, "--query", "Global flood hazard mapping", "--pool", "copernicus_services", "--rerank", "jina-v3.5", "--rerank-pool", "25", "--top-k", "3", "--json"],
        ),
        (
            "pool_filtered_jina_pool50_default",
            [sys.executable, cli_path, "--device", device, "--query", "Global flood hazard mapping", "--pool", "copernicus_services", "--rerank", "jina-v3.5", "--rerank-pool", "50", "--top-k", "3", "--json"],
        ),
        (
            "pool_filtered_qwen3_pool50",
            [sys.executable, cli_path, "--device", device, "--query", "Global flood hazard mapping", "--pool", "copernicus_services", "--rerank", "qwen3-0.6b", "--rerank-pool", "50", "--top-k", "3", "--json"],
        ),
    ]
    out: dict[str, Any] = {}
    for label, cmd in probes:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(res.stdout)
        out[label] = {
            "json_valid": True,
            "device": payload.get("device"),
            "eligible_after_filter": int(payload["eligible_after_filter"]),
            "load_seconds": float(payload["load_seconds"]),
            "stage1_ms": float(payload["stage1_ms"]),
            "rerank_ms": float(payload["rerank_ms"]),
            "n_results": len(payload["results"]),
            "top1_title": payload["results"][0]["title"] if payload["results"] else None,
        }
        print(f"  [{label:34s}] JSON_OK | dev={out[label]['device']} | s1={out[label]['stage1_ms']:.1f} ms | rr={out[label]['rerank_ms']:.1f} ms | top1={str(out[label]['top1_title'])[:45]}")
    (ARTIFACTS_DIR / "audit_cli_latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def audit_7_backend_parity(device: str) -> dict[str, Any]:
    """Verify cross-backend MLX vs PyTorch parity on all 3 neural models (EmbeddingGemma 2, Jina v3.5, Qwen3-0.6B)."""
    print("\n--- [AUDIT 7] Cross-Backend Device-Agnostic Parity Audit (MLX vs PyTorch) ---")
    test_queries = [
        "Global 30 meter inland surface water occurrence and seasonality from satellite optical imagery",
        "Global daily sea surface temperature climate data record from satellite thermal radiometers",
        "Circumpolar boreal forest aboveground biomass density mapped from spaceborne lidar",
    ]
    long_padding = (
        " Multi-decadal radiometric inter-calibration across Landsat 5 TM, Landsat 7 ETM+, Landsat 8 OLI, "
        "and Sentinel-2 MSI ensures consistent surface water recurrence, seasonal persistence, transition classes, "
        "maximum water extent, and monthly observation history across inland rivers, reservoirs, lakes, floodplains, "
        "estuaries, and coastal wetlands with automated cloud, cloud-shadow, terrain-shadow, and ice masking."
    ) * 12
    test_docs = [
        "Title: JRC Global Surface Water Mapping Layers (v1.4) | Source: Earth Engine | Spatial: global (30 m) | Summary: Maps of the location and temporal distribution of surface water from 1984 to 2021 derived from Landsat." + long_padding,
        "Title: ESA Soil Moisture Climate Change Initiative (CCI) | Source: CEDA | Spatial: global (25000 m) | Summary: Global multi-satellite active and passive microwave surface soil moisture.",
        "Title: OPERA Dynamic Surface Water Extent from Harmonized Landsat Sentinel-2 | Source: NASA Earthdata | Spatial: global (30 m) | Summary: Near-global 30 m gridded surface water extent and inundation mapping.",
        "Title: NOAA Coral Reef Temperature Anomaly Database (CoRTAD) | Source: NCEI | Spatial: global (4000 m) | Summary: Global 4 km sea surface temperature and coral thermal stress metrics.",
    ]

    if device != "mlx":
        out = {
            "primary_device": device,
            "cross_backend_comparison_ran": False,
            "note": "MLX backend only available on Apple Silicon macOS; PyTorch backend verified directly.",
        }
        (ARTIFACTS_DIR / "audit_backend_parity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  [INFO] Non-Apple-Silicon host ({device}); skipped MLX-vs-PyTorch cross-comparison.")
        return out

    # 1. Measure MLX embeddings & reranker scores directly
    enc_mlx, _, _ = load_dense_embedder(device="mlx")
    emb_mlx = np.vstack([enc_mlx.encode_queries([q], max_length=256)[0] for q in test_queries])
    del enc_mlx
    clear_device_cache("mlx")

    jina_mlx, _, _ = load_reranker("jina-v3.5", device="mlx", max_doc_length=420)
    sc_jina_mlx = [float(x) for x in jina_mlx.score_candidates(test_queries[0], test_docs)]
    del jina_mlx
    clear_device_cache("mlx")

    qwen_mlx, _, _ = load_reranker("qwen3-0.6b", device="mlx", verify_exactness=False)
    sc_qwen_mlx = [float(x) for x in qwen_mlx.score_candidates(test_queries[0], test_docs, batch_size=4, max_length=512)]
    del qwen_mlx
    clear_device_cache("mlx")

    # 2. Run PyTorch CPU via isolated `uv run --with torch` subprocess so MLX env stays lean
    uv_bin = shutil.which("uv")
    assert uv_bin, "uv executable not found for cross-backend PyTorch parity check"
    pt_code = f"""
import importlib.util, json, sys
import numpy as np
sys.path.insert(0, {str(SCRIPT_DIR)!r})
from eo_search_lib import load_dense_embedder
spec = importlib.util.spec_from_file_location("ev06", {str(SCRIPT_DIR / "06_eval_rerankers.py")!r})
ev06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev06)

qs = {test_queries!r}
docs = {test_docs!r}

enc, _, _ = load_dense_embedder(device="cpu")
emb = np.vstack([enc.encode_queries([q], max_length=256)[0] for q in qs]).tolist()
del enc

jina, _, _ = ev06.load_reranker("jina-v3.5", device="cpu", max_doc_length=420)
sc_j = [float(x) for x in jina.score_candidates(qs[0], docs)]
del jina

qwen, _, _ = ev06.load_reranker("qwen3-0.6b", device="cpu", verify_exactness=False)
sc_q = [float(x) for x in qwen.score_candidates(qs[0], docs, batch_size=4, max_length=512)]
del qwen

print(json.dumps({{"emb": emb, "jina": sc_j, "qwen": sc_q}}))
"""
    cmd = [
        uv_bin, "run", "--no-project",
        "--with", "torch>=2.4.0",
        "--with", "transformers>=5.18.0",
        "--with", "huggingface_hub>=0.30.0",
        "--with", "safetensors>=0.4.0",
        "--with", "tokenizers>=0.21.0",
        "--with", "numpy>=1.26.0",
        "--with", "pandas>=2.2.0",
        "--with", "pyarrow>=18.0.0",
        "--with", "duckdb>=1.5.0",
        "python3", "-c", pt_code,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Last non-empty line is JSON payload
    pt_payload = json.loads([ln for ln in res.stdout.splitlines() if ln.strip()][-1])
    emb_pt = np.asarray(pt_payload["emb"], dtype=np.float32)
    sc_jina_pt = [float(x) for x in pt_payload["jina"]]
    sc_qwen_pt = [float(x) for x in pt_payload["qwen"]]

    cosines = [float(np.dot(emb_mlx[i], emb_pt[i])) for i in range(len(test_queries))]
    jina_rank_mlx = [int(i) for i in np.argsort(-np.asarray(sc_jina_mlx), kind="stable")]
    jina_rank_pt = [int(i) for i in np.argsort(-np.asarray(sc_jina_pt), kind="stable")]
    qwen_rank_mlx = [int(i) for i in np.argsort(-np.asarray(sc_qwen_mlx), kind="stable")]
    qwen_rank_pt = [int(i) for i in np.argsort(-np.asarray(sc_qwen_pt), kind="stable")]

    assert min(cosines) >= 0.999, f"Embedding parity below 0.999: {cosines}"
    assert jina_rank_mlx == jina_rank_pt, f"Jina rank mismatch: {jina_rank_mlx} vs {jina_rank_pt}"
    assert qwen_rank_mlx == qwen_rank_pt, f"Qwen3 rank mismatch: {qwen_rank_mlx} vs {qwen_rank_pt}"

    out = {
        "cross_backend_comparison_ran": True,
        "backends_compared": ["mlx (Apple Silicon BF16)", "pytorch (CPU BF16)"],
        "includes_gt512_token_truncated_doc": True,
        "embeddinggemma2": {
            "mlx_model": DEFAULT_MODEL_BF16,
            "pytorch_model": DEFAULT_PT_EMBED_MODEL,
            "n_queries": len(test_queries),
            "per_query_cosine": [round(c, 6) for c in cosines],
            "min_cosine": round(min(cosines), 6),
            "mean_cosine": round(float(np.mean(cosines)), 6),
        },
        "jina_reranker_v3_5": {
            "mlx_scores": [round(x, 5) for x in sc_jina_mlx],
            "pytorch_scores": [round(x, 5) for x in sc_jina_pt],
            "max_abs_score_diff": round(float(np.max(np.abs(np.asarray(sc_jina_mlx) - np.asarray(sc_jina_pt)))), 5),
            "rank_order_mlx": jina_rank_mlx,
            "rank_order_pytorch": jina_rank_pt,
            "exact_rank_agreement": jina_rank_mlx == jina_rank_pt,
        },
        "qwen3_reranker_0_6b": {
            "mlx_scores": [round(x, 5) for x in sc_qwen_mlx],
            "pytorch_scores": [round(x, 5) for x in sc_qwen_pt],
            "max_abs_score_diff": round(float(np.max(np.abs(np.asarray(sc_qwen_mlx) - np.asarray(sc_qwen_pt)))), 5),
            "rank_order_mlx": qwen_rank_mlx,
            "rank_order_pytorch": qwen_rank_pt,
            "exact_rank_agreement": qwen_rank_mlx == qwen_rank_pt,
        },
    }
    (ARTIFACTS_DIR / "audit_backend_parity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"  [OK] EmbeddingGemma-2 min_cos={out['embeddinggemma2']['min_cosine']:.6f} | "
        f"Jina max_diff={out['jina_reranker_v3_5']['max_abs_score_diff']:.5f} (rank_agree={out['jina_reranker_v3_5']['exact_rank_agreement']}) | "
        f"Qwen3 max_diff={out['qwen3_reranker_0_6b']['max_abs_score_diff']:.5f} (rank_agree={out['qwen3_reranker_0_6b']['exact_rank_agreement']})"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Scientific Audits for v4.0-standalone")
    parser.add_argument("--device", default="auto", choices=["auto", "mlx", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    device = resolve_device(args.device)

    t0 = time.perf_counter()
    ensure_search_indexes(ARTIFACTS_DIR, require_dense=True, allow_download=True)
    cfg = json.loads((ARTIFACTS_DIR / "frozen_v4_config.json").read_text(encoding="utf-8"))
    cat_pq = ARTIFACTS_DIR / "full_catalogue.parquet"
    if not cat_pq.exists():
        cat_pq = SCRIPT_DIR.parent / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
    df = pd.read_parquet(cat_pq)
    z = np.load(ARTIFACTS_DIR / "full_embeddings_bf16.npz", allow_pickle=True)
    vecs = z["embeddings_f16"].astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    generic_sigs = [str(s) for s in z["family_sigs"].tolist()]
    lex = DuckDBLexicalIndex(str(ARTIFACTS_DIR / "full_bm25.duckdb"), read_only=True)
    verify_index_alignment(df, lex_index=lex, emb_npz=z, catalogue_path=cat_pq)

    audit_0_bm25_determinism(df, lex, cfg)
    audit_1_dense_exactness(df, device)
    audit_2_jina_position_bias(df, cfg, device)
    _, dev_items, dev_qvecs = audit_3_false_collapse(df, vecs, generic_sigs, lex, cfg, device)
    audit_4_nested_lofo_cv(df, vecs, generic_sigs, lex, dev_items, dev_qvecs)
    audit_5_dev45_filter_stage2(df, dev_items, cfg, device)
    lex.con.close()
    audit_6_cli_latency(device)
    audit_7_backend_parity(device)
    print(f"\n[DONE] All 8 reproducible {cfg['protocol_version']} audits completed in {(time.perf_counter() - t0):.1f} s.")


if __name__ == "__main__":
    main()
