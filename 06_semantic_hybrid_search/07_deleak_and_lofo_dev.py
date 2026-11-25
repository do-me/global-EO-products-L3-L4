# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "mlx>=0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-vlm>=0.8.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "torch>=2.4.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "transformers>=5.18.0",
# ]
# ///
"""07. Strictly-on-Dev-45 Collapser Selection, Orthogonal Filter Ablation & v4.2-standalone Freeze.

Executes strictly on the 45-query Development set (ZERO access to Holdout A or Retrospectively Audited Holdout B):
  1. Uses solo `batch_size=1` query encoding (bit-identical to `10_run_blind_holdouts.py` & `04_search_cli.py`),
     with device-agnostic auto-selection (`--device auto|mlx|cuda|mps|cpu`).
  2. Sweeps generic `near_dup_cos` with `sensor_guard=True` (`1.0, 0.98, 0.97, 0.96, 0.94`) on Dev-45,
     breaking ties by `(fam_ndcg_10 desc, fam_r_50 desc, near_dup_cos desc)`.
  3. Persists the orthogonal `NullSafe` vs `NaiveStrict` full-catalogue ablation (including `FamR@50` and
     the 13 grid-filtered queries) at the exact selected `near_dup_cos` -> `artifacts/dev45_filter_ablation_v4.csv`.
  4. Locks immutable `artifacts/frozen_v4_config.json` with artifact SHA-256 hashes (including
     `eo_search_lib.py`, `06_eval_rerankers.py`, locked JSON benchmarks, and locked HuggingFace model revisions).
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

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_MODEL_BF16,
    DEFAULT_PT_EMBED_MODEL,
    DuckDBLexicalIndex,
    SearchFilters,
    apply_filters_mask,
    collapse_product_families,
    compute_catalogue_content_sha256,
    compute_record_ids_sha256,
    deterministic_topk,
    filter_benchmark_family_mask,
    load_dense_embedder,
    resolve_device,
    resolve_hf_model_revision,
    rrf_fuse,
    verify_index_alignment,
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clear_cache(device: str) -> None:
    if device == "mlx":
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
    elif device in ("cuda", "mps"):
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass


def eval_ranking(
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
    fam_at = {5: 0, 10: 0, 20: 0, 50: 0}
    hit_at = {5: 0.0, 10: 0.0}

    for rank, doc_idx in enumerate(ranked_doc_indices[:50], start=1):
        fams = rid_to_fams.get(rid_by_idx[doc_idx])
        if fams:
            if first_rel_rank == 0:
                first_rel_rank = rank
            new_fams = fams - seen_fams
            if new_fams:
                seen_fams.update(new_fams)
                if rank <= 10:
                    dcg10 += float(len(new_fams)) / math.log2(rank + 1)
        if rank in fam_at:
            fam_at[rank] = len(seen_fams)
        if rank in hit_at:
            hit_at[rank] = 1.0 if len(seen_fams) > 0 else 0.0

    for k in (5, 10, 20, 50):
        if len(ranked_doc_indices) < k:
            fam_at[k] = len(seen_fams)
        if k in hit_at and len(ranked_doc_indices) < k:
            hit_at[k] = 1.0 if len(seen_fams) > 0 else 0.0

    idcg10 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 10) + 1))
    return {
        "hit_5": hit_at[5],
        "hit_10": hit_at[10],
        "fam_r_5": fam_at[5] / n_fams,
        "fam_r_10": fam_at[10] / n_fams,
        "fam_r_20": fam_at[20] / n_fams,
        "fam_r_50": fam_at[50] / n_fams,
        "mrr_10": (1.0 / first_rel_rank) if (0 < first_rel_rank <= 10) else 0.0,
        "fam_ndcg_10": dcg10 / idcg10 if idcg10 > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-45 Collapser Selection & v4.0-standalone Pre-Registration")
    parser.add_argument("--device", default="auto", choices=["auto", "mlx", "cuda", "mps", "cpu"])
    args = parser.parse_args()
    device = resolve_device(args.device)

    print("=" * 94)
    print(f"07. DEV-45 GENERIC COLLAPSER SELECTION, ORTHOGONAL FILTER ABLATION & v4.0 PRE-REGISTRATION ({device})")
    print("=" * 94)

    cat_pq = ARTIFACTS_DIR / "full_catalogue.parquet"
    if not cat_pq.exists():
        cat_pq = SCRIPT_DIR.parent / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
    df = pd.read_parquet(cat_pq)
    rid_by_idx = df["record_id"].tolist()
    rid_to_pos = {r: i for i, r in enumerate(rid_by_idx)}
    z = np.load(ARTIFACTS_DIR / "full_embeddings_bf16.npz", allow_pickle=True)
    vecs = z["embeddings_f16"].astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    generic_sigs = [str(s) for s in z["family_sigs"].tolist()]
    assert not any(s.startswith("fam:") for s in generic_sigs), "Legacy fam: sig detected!"
    lex = DuckDBLexicalIndex(str(ARTIFACTS_DIR / "full_bm25.duckdb"), read_only=True)
    verify_index_alignment(df, lex_index=lex, emb_npz=z, catalogue_path=cat_pq)
    bm25_content_sha = lex.compute_content_sha256()
    rids_sha = compute_record_ids_sha256(rid_by_idx)
    cat_content_sha = compute_catalogue_content_sha256(df)

    dev_bench_path = ARTIFACTS_DIR / "policy_retrieval_benchmark.json"
    dev_queries = json.loads(dev_bench_path.read_text(encoding="utf-8"))["queries"]
    gt_by_q = []
    for sp in dev_queries:
        fam_rids = {}
        for fam in sp["expected_families"]:
            rlist = fam.get("record_ids")
            assert rlist and len(rlist) > 0, f"0 locked record_ids for {sp['query_id']} / {fam['family_id']}"
            fam_rids[fam["family_id"]] = set(rlist)
        gt_by_q.append(fam_rids)

    # Solo batch_size=1 query encoding (bit-identical with 10_run_blind_holdouts.py & 04_search_cli.py)
    enc, _, resolved_embed_model = load_dense_embedder(device=device)
    _ = enc.encode_queries(["warmup satellite query"], max_length=256)
    q_vecs = np.vstack([enc.encode_queries([sp["query"]], max_length=256)[0] for sp in dev_queries])
    del enc
    _clear_cache(device)

    # 1. Generic Collapser + Sensor-Guard Sweep strictly on Dev-45
    print("\n--- [1] GENERIC COLLAPSER + SENSOR-GUARD SWEEP (STRICTLY DEV-45, SOLO QUERY ENCODING) ---")
    collapser_variants = [
        ("01_No_Collapse", None, 1.0, False),
        ("02_Generic_StemOnly", generic_sigs, 1.0, True),
        ("03_Generic_SensorGuard_cos0.98", generic_sigs, 0.98, True),
        ("04_Generic_SensorGuard_cos0.97", generic_sigs, 0.97, True),
        ("05_Generic_SensorGuard_cos0.96", generic_sigs, 0.96, True),
        ("06_Generic_SensorGuard_cos0.94", generic_sigs, 0.94, True),
    ]
    col_rows = []
    for qi, sp in enumerate(dev_queries):
        flt_spec = sp.get("filters", {})
        flt_ns = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            exclude_superseded=True,
            null_safe=True,
        )
        mask = apply_filters_mask(df, flt_ns)
        for fid, rset in gt_by_q[qi].items():
            assert any(mask[rid_to_pos[r]] for r in rset if r in rid_to_pos), f"Unreachable Dev family: {sp['query_id']} / {fid}"

        d_res = deterministic_topk(vecs @ q_vecs[qi], k=300, mask=mask)
        b_res = lex.search_bm25(sp["query"], top_k=300, eligible_mask=mask)
        fused = rrf_fuse(d_res, b_res, rrf_k=60, w_dense=1.0, w_bm25=0.6)

        for vname, sigs, cos_th, sg in collapser_variants:
            if sigs is None:
                idxs = [i for i, _, _ in fused[:50]]
            else:
                collapsed = collapse_product_families(
                    fused, df, top_k=50, family_sigs=sigs, vectors=vecs, near_dup_cos=cos_th, sensor_guard=sg, query_text=sp["query"]
                )
                idxs = [i for i, _, _ in collapsed]
            m = eval_ranking(idxs, rid_by_idx, gt_by_q[qi])
            col_rows.append({"variant": vname, "near_dup_cos": cos_th, "query_id": sp["query_id"], **m})

    df_col = pd.DataFrame(col_rows)
    mcols = ["hit_5", "hit_10", "fam_r_5", "fam_r_10", "fam_r_20", "fam_r_50", "mrr_10", "fam_ndcg_10"]
    col_sum = (df_col.groupby(["variant", "near_dup_cos"], sort=True)[mcols].mean() * 100.0).round(2).reset_index()
    print(col_sum.to_string(index=False))
    col_sum.to_csv(ARTIFACTS_DIR / "deleak_collapser_audit.csv", index=False)

    # Explicit 3-tier tie-breaker: fam_ndcg_10 desc -> fam_r_50 desc -> near_dup_cos desc (higher = safer)
    gen_only = col_sum[col_sum["variant"] != "01_No_Collapse"]
    best_gen_row = gen_only.sort_values(["fam_ndcg_10", "fam_r_50", "near_dup_cos"], ascending=[False, False, False]).iloc[0]
    chosen_cos = float(best_gen_row["near_dup_cos"])
    print(f"  -> Selected 100% generic collapser on Dev-45: {best_gen_row['variant']} (near_dup_cos={chosen_cos}, sensor_guard=True)")

    # 2. Orthogonal NullSafe vs NaiveStrict Filter Ablation AT EXACT SELECTED near_dup_cos (Top-50)
    filt_rows = []
    for qi, sp in enumerate(dev_queries):
        flt_spec = sp.get("filters", {})
        has_grid = flt_spec.get("max_grid_m") is not None
        for mode_name, ns_flag in [("NullSafe_Collapse", True), ("NaiveStrict_Collapse", False)]:
            flt = SearchFilters(
                geo_scope=flt_spec.get("geo_scope"),
                max_grid_m=flt_spec.get("max_grid_m"),
                strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
                exclude_superseded=True,
                null_safe=ns_flag,
            )
            m_f = apply_filters_mask(df, flt)
            d_f = deterministic_topk(vecs @ q_vecs[qi], k=300, mask=m_f)
            b_f = lex.search_bm25(sp["query"], top_k=300, eligible_mask=m_f)
            f_f = rrf_fuse(d_f, b_f, rrf_k=60, w_dense=1.0, w_bm25=0.6)
            c_f = collapse_product_families(
                f_f, df, top_k=50, family_sigs=generic_sigs, vectors=vecs, near_dup_cos=chosen_cos, sensor_guard=True, query_text=sp["query"]
            )
            filt_rows.append({
                "mode": mode_name,
                "near_dup_cos": chosen_cos,
                "query_id": sp["query_id"],
                "has_grid_filter": has_grid,
                **eval_ranking([i for i, _, _ in c_f], rid_by_idx, gt_by_q[qi]),
            })
    df_filt = pd.DataFrame(filt_rows)
    filt_sum_all = (df_filt.groupby("mode")[mcols].mean() * 100.0).round(2).reset_index()
    filt_sum_all.insert(0, "slice", "all_45")
    filt_sum_grid = (df_filt[df_filt["has_grid_filter"]].groupby("mode")[mcols].mean() * 100.0).round(2).reset_index()
    filt_sum_grid.insert(0, "slice", "grid_filtered_13")
    df_filt_sum = pd.concat([filt_sum_all, filt_sum_grid], ignore_index=True)
    df_filt_sum.to_csv(ARTIFACTS_DIR / "dev45_filter_ablation_v4.csv", index=False)
    print(f"\n--- [2] ORTHOGONAL NULL-SAFE vs NAIVE-STRICT ABLATION ON DEV-45 (cos={chosen_cos}, grid_tol=1.25 BOTH) ---")
    print(df_filt_sum.to_string(index=False))

    # 3. Freeze Immutable v4.3-standalone Manifest (locking all benchmark JSONs, code, semantic index hashes & all MLX + PyTorch model revisions)
    _PINNED_FALLBACK_REVS = {
        DEFAULT_MODEL_BF16: "1a4ffddb7905d3f63486748deabe091a01fb6201",
        DEFAULT_PT_EMBED_MODEL: "f9972ac22e696a105c482202646979e33e4b5a1e",
        "jinaai/jina-reranker-v3.5-mlx": "3dd4ac901ccdcac85abe3815df0a0aaaf44e4a21",
        "jinaai/jina-reranker-v3.5": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    }
    locked_model_revisions = {}
    for mid, fb_rev in _PINNED_FALLBACK_REVS.items():
        got = resolve_hf_model_revision(mid)
        locked_model_revisions[mid] = got if (got and got != "unresolved") else fb_rev
    locked_hashes = {
        "eo_search_lib.py": sha256_file(SCRIPT_DIR / "eo_search_lib.py"),
        "02_build_policy_benchmark.py": sha256_file(SCRIPT_DIR / "02_build_policy_benchmark.py"),
        "06_eval_rerankers.py": sha256_file(SCRIPT_DIR / "06_eval_rerankers.py"),
        "10_run_blind_holdouts.py": sha256_file(SCRIPT_DIR / "10_run_blind_holdouts.py"),
        "full_catalogue.parquet": sha256_file(cat_pq),
        "catalogue_content_sha256": cat_content_sha,
        "full_embeddings_bf16.npz": sha256_file(ARTIFACTS_DIR / "full_embeddings_bf16.npz"),
        "bm25_content_sha256": bm25_content_sha,
        "record_ids_sha256": rids_sha,
        "policy_retrieval_benchmark.json": sha256_file(dev_bench_path),
        "holdout_a_external_benchmark.json": sha256_file(ARTIFACTS_DIR / "holdout_a_external_benchmark.json"),
        "holdout_b_blind_benchmark.json": sha256_file(ARTIFACTS_DIR / "holdout_b_blind_benchmark.json"),
    }
    frozen_v4 = {
        "protocol_version": "v4.3-standalone",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_used_for_freeze": device,
        "rule": (
            "IMMUTABLE v4.3-standalone SPEC (Device-Agnostic MLX/CUDA/CPU + Cold-Cache Snapshot-Pinned Replay + "
            "record_id- & content-hash-bound DuckDB BM25 + Variable/Layer-Role/Cadence/Grid/Geo-Aware Collapser + "
            "Verified Hard-Limits Mode + Locked Benchmark JSONs & Model Revisions). "
            "Strictly self-contained standalone catalogue evaluation: Dev-45 (development, N=45), "
            "Holdout A (retrospectively audited external domains, N=35), and "
            "Holdout B (retrospectively audited 60-query cross-domain holdout, disjoint from Dev-45 & Holdout A, G=G_eff=60)."
        ),
        "benchmark_provenance": {
            "dev45": "development_tuning_set_45_queries_locked_json",
            "holdout_a": "retrospectively_audited_external_domains_35_queries_locked_json",
            "holdout_b": "retrospectively_audited_60_queries_disjoint_from_dev45_and_holdout_a_locked_json",
        },
        "preregistered_primary_metric": "fam_ndcg_10",
        "preregistered_secondary_metrics": ["mrr_10", "fam_r_10", "hit_5", "hit_10"],
        "preregistered_primary_hypotheses": [
            "H1_Collapser: HybridRRF+GenericCollapse vs HybridRRF_NoCollapse",
            "H2_Hybrid_vs_BM25_Fair: HybridRRF+GenericCollapse vs BM25_PreFilter+GenericCollapse",
            "H3_Hybrid_vs_Dense_Fair: HybridRRF+GenericCollapse vs Dense+GenericCollapse",
            "H4_Jina_vs_Stage1: JinaV35_Top50_RRFBlend vs HybridRRF+GenericCollapse",
            "H5_Qwen3_vs_Stage1: Qwen3_Top50_RRFBlend vs HybridRRF+GenericCollapse",
            "H6_Jina_vs_Qwen3: JinaV35_Top50_RRFBlend vs Qwen3_Top50_RRFBlend",
        ],
        "locked_artifact_sha256": locked_hashes,
        "code_and_index_sha256": locked_hashes,
        "locked_model_revisions": locked_model_revisions,
        "stage1": {
            "dense_model_mlx": DEFAULT_MODEL_BF16,
            "dense_model_pytorch": DEFAULT_PT_EMBED_MODEL,
            "dim": 768,
            "max_length": 512,
            "chunking": False,
            "query_encode_batch_size": 1,
            "bm25_prefilter_sql": True,
            "bm25_record_id_keyed": True,
            "bm25_deterministic_threads1_round10": True,
            "bm25_weights": {"title": 2.5, "keywords": 1.5, "body": 1.0, "k1": 1.2, "b": 0.75},
            "rrf": {"k": 60, "w_dense": 1.0, "w_bm25": 0.6, "candidate_k": 300},
            "collapser": {
                "use_known_patterns": False,
                "generic_stem_plus_cos": True,
                "near_dup_cos": chosen_cos,
                "sensor_guard": True,
                "sensor_guard_on_gstem": True,
                "cadence_grid_geo_doi_guards": True,
                "query_variant_promotion": True,
                "sensor_guard_fields": ["title", "raw_id", "platform_sensor_raw"],
            },
            "null_safe_filter": True,
            "orthogonal_grid_tolerance": 1.25,
            "deterministic_tie_breaking": True,
        },
        "stage2_frozen_systems": {
            "primary_best_quality": {
                "model_mlx": "jinaai/jina-reranker-v3.5-mlx",
                "model_pytorch": "jinaai/jina-reranker-v3.5",
                "license": "CC BY-NC 4.0",
                "pool_k": 50,
                "max_doc_length": 420,
                "blend": {"rrf_k": 30, "w_rrf": 0.4, "w_rerank": 1.0},
            },
            "ablation_top50_pure": {
                "model_mlx": "jinaai/jina-reranker-v3.5-mlx",
                "model_pytorch": "jinaai/jina-reranker-v3.5",
                "pool_k": 50,
                "max_doc_length": 420,
                "blend": None,
            },
            "fast_sweet_spot": {
                "model_mlx": "jinaai/jina-reranker-v3.5-mlx",
                "model_pytorch": "jinaai/jina-reranker-v3.5",
                "pool_k": 25,
                "max_doc_length": 420,
                "blend": None,
            },
            "permissive_apache2": {
                "model_mlx": "Qwen/Qwen3-Reranker-0.6B",
                "model_pytorch": "Qwen/Qwen3-Reranker-0.6B",
                "license": "Apache-2.0",
                "score_mode": "log_odds_margin_stable_sort",
                "pool_k": 50,
                "max_length": 512,
                "blend": {"rrf_k": 30, "w_rrf": 0.4, "w_rerank": 1.0},
            },
        },
    }
    out_cfg = ARTIFACTS_DIR / "frozen_v4_config.json"
    out_cfg.write_text(json.dumps(frozen_v4, indent=2), encoding="utf-8")

    # Sync code & benchmark hashes in full_index_manifest.json
    man_path = ARTIFACTS_DIR / "full_index_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for idx_k in ("full_catalogue.parquet", "full_embeddings_bf16.npz", "bm25_content_sha256", "record_ids_sha256"):
            assert man["sha256"][idx_k] == locked_hashes[idx_k], f"Index drift on {idx_k}"
        man["sha256"]["eo_search_lib.py"] = locked_hashes["eo_search_lib.py"]
        man["sha256"]["06_eval_rerankers.py"] = locked_hashes["06_eval_rerankers.py"]
        man["updated_at_utc"] = frozen_v4["frozen_at_utc"]
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    lex.con.close()
    print(f"\n[FROZEN] Wrote immutable v4.3-standalone artifact SHA-256 manifest -> {out_cfg}")


if __name__ == "__main__":
    main()
