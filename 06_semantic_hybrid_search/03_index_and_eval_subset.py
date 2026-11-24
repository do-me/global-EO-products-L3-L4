# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.2.0",
#   "mlx>=0.32.3",
#   "mlx-vlm>=0.8.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "transformers>=5.18.0",
# ]
# ///
"""Index the 3,500-record challenging subset and run the full SOTA ablation & Matryoshka evaluation.

Compares:
  1. Naive substring ILIKE
  2. DuckDB multi-field BM25 (EO-normalized)
  3. Dense Raw (EmbeddingGemma-2 BF16, uncleaned concat, no official prompts)
  4. Dense EO-Structured Single-Vector (EmbeddingGemma-2 BF16 + official prompts)
  5. Dense EO-Structured Chunked MaxSim (512-tok / 384-stride)
  6. Hybrid RRF (Unfiltered)
  7. Hybrid RRF + Naive Strict Filter (drops NULLs!)
  8. Hybrid RRF + Null-Safe Filter
  9. Full SOTA: Hybrid RRF + Null-Safe Filter + Product-Family Mirror Collapse (768d)
 10. Matryoshka Sensitivity on Full SOTA: 512d, 256d, 128d, and 768d-fp16 storage!
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
SUBSET_PQ = ARTIFACTS_DIR / "subset_catalogue.parquet"
BENCH_JSON = ARTIFACTS_DIR / "policy_retrieval_benchmark.json"
MLX_BENCH_JSON = ARTIFACTS_DIR / "mlx_throughput_benchmark.json"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_MODEL_BF16,
    DuckDBLexicalIndex,
    MLXEmbeddingGemma2,
    SearchFilters,
    apply_filters_mask,
    build_eo_document,
    build_raw_document,
    collapse_product_families,
    compute_family_signature,
    matryoshka_truncate,
    maxsim_chunk_scores,
    rrf_fuse,
)


def evaluate_ranking(
    ranked_doc_indices: list[int],
    rid_by_idx: list[str],
    expected_families: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute Hit@K, Family-Recall@K, MRR@10, and Family-nDCG@10/20 (dedup-aware!).

    Family-nDCG@K awards gain=1 ONLY the first time each distinct expected product family
    appears in the ranking — preventing mirror/edition spam from inflating nDCG!
    """
    rid_to_fams: dict[str, set[str]] = {}
    for fam in expected_families:
        fid = fam["family_id"]
        for rid in fam["subset_record_ids"]:
            rid_to_fams.setdefault(rid, set()).add(fid)

    n_fams = max(1, len(expected_families))
    first_rel_rank = 0
    seen_fams: set[str] = set()
    dcg10 = 0.0
    dcg20 = 0.0

    fam_counts_at = {5: 0, 10: 0, 20: 0, 50: 0}
    hit_at = {5: 0.0, 10: 0.0, 20: 0.0}

    for rank, doc_idx in enumerate(ranked_doc_indices[:50], start=1):
        rid = rid_by_idx[doc_idx]
        fams = rid_to_fams.get(rid)
        if fams:
            if first_rel_rank == 0:
                first_rel_rank = rank
            new_fams = fams - seen_fams
            if new_fams:
                # Award unit gain per newly discovered distinct policy product family
                seen_fams.update(new_fams)
                gain = float(len(new_fams))
                if rank <= 10:
                    dcg10 += gain / math.log2(rank + 1)
                if rank <= 20:
                    dcg20 += gain / math.log2(rank + 1)
        if rank in fam_counts_at:
            fam_counts_at[rank] = len(seen_fams)
        if rank in hit_at:
            hit_at[rank] = 1.0 if len(seen_fams) > 0 else 0.0

    # Handle lists shorter than 50
    for k in (5, 10, 20, 50):
        if len(ranked_doc_indices) < k:
            fam_counts_at[k] = len(seen_fams)
        if k in hit_at and len(ranked_doc_indices) < k:
            hit_at[k] = 1.0 if len(seen_fams) > 0 else 0.0

    idcg10 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 10) + 1))
    idcg20 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 20) + 1))
    mrr10 = (1.0 / first_rel_rank) if (0 < first_rel_rank <= 10) else 0.0

    return {
        "hit_5": hit_at[5],
        "hit_10": hit_at[10],
        "hit_20": hit_at[20],
        "fam_r_5": fam_counts_at[5] / n_fams,
        "fam_r_10": fam_counts_at[10] / n_fams,
        "fam_r_20": fam_counts_at[20] / n_fams,
        "fam_r_50": fam_counts_at[50] / n_fams,
        "mrr_10": mrr10,
        "fam_ndcg_10": dcg10 / idcg10 if idcg10 > 0 else 0.0,
        "fam_ndcg_20": dcg20 / idcg20 if idcg20 > 0 else 0.0,
        "missed_families": sorted(set(f["family_id"] for f in expected_families) - seen_fams),
    }


def make_search_filters(spec_filters: dict[str, Any], null_safe: bool) -> SearchFilters:
    return SearchFilters(
        geo_scope=spec_filters.get("geo_scope"),
        max_grid_m=spec_filters.get("max_grid_m"),
        strict_eo_only=bool(spec_filters.get("strict_eo_only", False)),
        exclude_superseded=True,
        null_safe=null_safe,
    )


def topk_from_scores(scores: np.ndarray, k: int = 250, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    if mask is not None:
        s = np.where(mask, scores, -1e9)
    else:
        s = scores
    valid_n = int(np.sum(s > -1e8))
    if valid_n == 0:
        return []
    take = min(k, valid_n)
    part = np.argpartition(-s, take - 1)[:take]
    order = part[np.argsort(-s[part])]
    return [(int(i), float(s[i])) for i in order]


def main() -> None:
    print("=" * 88)
    print("03. SUBSET INDEXING & SOTA HYBRID RETRIEVAL ABLATION (45 POLICY QUERIES / 3,500 RECORDS)")
    print("=" * 88)

    df = pd.read_parquet(SUBSET_PQ)
    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    queries = bench["queries"]
    n_docs = len(df)
    rid_by_idx = df["record_id"].tolist()
    rows = df.to_dict(orient="records")

    # Use empirically optimal Apple M3 Max MLX strategy (bf16, bs=16, 8 fixed buckets, compiled + async_eval -> ~100.8 doc/s)
    batch_size = 16
    fixed_buckets = [64, 128, 192, 256, 320, 384, 448, 512]
    print(f"Using winning MLX BF16 strategy: batch_size={batch_size}, fixed_buckets={fixed_buckets}, compiled=True, async_pipe=True")

    # Precompute product-family signatures for mirror collapsing
    family_sigs = [
        compute_family_signature(
            str(r.get("title", "")),
            str(r.get("raw_id", "")),
            str(r.get("canonical_product_key", "")),
        )
        for r in rows
    ]

    # 1. Build persistent DuckDB BM25 index on subset
    bm25_db = ARTIFACTS_DIR / "subset_bm25.duckdb"
    if bm25_db.exists():
        bm25_db.unlink()
    lex_index = DuckDBLexicalIndex(str(bm25_db))
    t_bm25 = lex_index.build_from_dataframe(df)
    print(f"[1/4] Built DuckDB multi-field BM25 index ({n_docs:,} docs) in {t_bm25:.2f}s -> {bm25_db.name}")

    # 2. Load MLX EmbeddingGemma-2 BF16 & encode subset (Raw, EO-Structured Single, EO-Structured Chunked)
    enc = MLXEmbeddingGemma2(DEFAULT_MODEL_BF16)
    print(f"[2/4] Loaded {DEFAULT_MODEL_BF16} in {enc.load_seconds:.2f}s")

    eo_docs = [build_eo_document(r) for r in rows]
    raw_docs = [build_raw_document(r) for r in rows]

    eo_vecs, eo_st = enc.encode_texts(
        eo_docs, batch_size=batch_size, max_length=512, sort_by_length=True,
        prefetch=True, async_pipe=True, fixed_buckets=fixed_buckets, use_compile=True, return_stats=True,
    )
    print(
        f"  EO-Structured Single-Vec: {n_docs:,} docs in {eo_st['total_seconds']:.2f}s "
        f"({eo_st['docs_per_sec']:.1f} doc/s, pad={eo_st['padding_overhead_pct']}%, mem={eo_st['peak_mem_gb']:.2f} GB)"
    )

    raw_vecs, raw_st = enc.encode_texts(
        raw_docs, batch_size=batch_size, max_length=512, sort_by_length=True,
        prefetch=True, async_pipe=True, fixed_buckets=fixed_buckets, use_compile=True, return_stats=True,
    )
    print(
        f"  Raw-Concat Baseline     : {n_docs:,} docs in {raw_st['total_seconds']:.2f}s "
        f"({raw_st['docs_per_sec']:.1f} doc/s)"
    )

    t_ch0 = time.perf_counter()
    chunk_vecs, chunk_doc_idx = enc.encode_documents_chunked(
        rows, batch_size=batch_size, max_length=512, stride=384, max_chunks=4,
    )
    t_ch = time.perf_counter() - t_ch0
    print(
        f"  EO-Structured Chunked   : {len(chunk_vecs):,} chunks ({len(chunk_vecs)/n_docs:.3f}x docs) in {t_ch:.2f}s"
    )

    # Persist EO-Structured embeddings (float16) + metadata index for CLI search
    emb_path = ARTIFACTS_DIR / "subset_embeddings_bf16.npz"
    np.savez_compressed(
        emb_path,
        embeddings_f16=eo_vecs.astype(np.float16),
        record_ids=np.array(rid_by_idx, dtype=object),
        family_sigs=np.array(family_sigs, dtype=object),
    )
    print(f"[3/4] Saved float16 subset embeddings ({emb_path.stat().st_size / (1024**2):.2f} MB) -> {emb_path.name}")

    # Precompute fp16-roundtrip and Matryoshka vectors for ablation
    eo_vecs_f16_rt = eo_vecs.astype(np.float16).astype(np.float32)
    eo_vecs_f16_rt /= np.maximum(np.linalg.norm(eo_vecs_f16_rt, axis=1, keepdims=True), 1e-12)
    matryo_docs = {d: matryoshka_truncate(eo_vecs, d) for d in (512, 256, 128)}

    # Encode all 45 queries (official prompt & raw)
    q_texts = [q["query"] for q in queries]
    t_q0 = time.perf_counter()
    q_vecs_eo = enc.encode_queries(q_texts, max_length=256)
    t_q_all = time.perf_counter() - t_q0
    q_vecs_raw = enc.encode_texts(q_texts, batch_size=32, max_length=256, sort_by_length=False, prefetch=False)
    matryo_qs = {d: matryoshka_truncate(q_vecs_eo, d) for d in (512, 256, 128)}

    # Measure realistic live single-query encode latency (mean over 10 sequential queries)
    single_q_times = []
    for qt in q_texts[:10]:
        ts = time.perf_counter()
        _ = enc.encode_queries([qt], max_length=256)
        single_q_times.append((time.perf_counter() - ts) * 1000.0)
    mean_single_q_encode_ms = float(np.mean(single_q_times))
    print(f"[4/4] Encoded {len(queries)} queries (batch={t_q_all*1000:.1f} ms; single-query MLX latency={mean_single_q_encode_ms:.1f} ms)")

    # -----------------------------------------------------------------------
    # Run Full 13-Variant Ablation Matrix across all 45 queries
    # -----------------------------------------------------------------------
    systems = [
        "01_Naive_Substring_ILIKE",
        "02_BM25_EO_Normalized",
        "03_Dense_Raw_NoPrompt",
        "04_Dense_EO_SingleVec_768d",
        "05_Dense_EO_Chunked_MaxSim",
        "06_Hybrid_RRF_Unfiltered",
        "07_Hybrid_RRF_NaiveStrictFilter",
        "08_Hybrid_RRF_NullSafeFilter",
        "09_SOTA_Hybrid_RRF_NullSafe_Collapse_768d",
        "10_SOTA_Matryoshka_512d",
        "11_SOTA_Matryoshka_256d",
        "12_SOTA_Matryoshka_128d",
        "13_SOTA_FP16_Storage_768d",
    ]

    per_query_rows: list[dict[str, Any]] = []
    sota_diagnostics: list[dict[str, Any]] = []
    latencies_ms: dict[str, list[float]] = {s: [] for s in systems}

    for qi, q in enumerate(queries):
        qtxt = q["query"]
        fams = q["expected_families"]
        flt_spec = q.get("filters", {})
        has_grid_or_geo = ("max_grid_m" in flt_spec) or ("geo_scope" in flt_spec)

        mask_null_safe = apply_filters_mask(df, make_search_filters(flt_spec, null_safe=True))
        mask_naive_strict = apply_filters_mask(df, make_search_filters(flt_spec, null_safe=False))

        # 1. Naive ILIKE
        t0 = time.perf_counter()
        res_like = lex_index.search_naive_like(qtxt, top_k=100)
        latencies_ms["01_Naive_Substring_ILIKE"].append((time.perf_counter() - t0) * 1000.0)

        # 2. BM25
        t0 = time.perf_counter()
        res_bm25 = lex_index.search_bm25(qtxt, top_k=250)
        t_bm25_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms["02_BM25_EO_Normalized"].append(t_bm25_ms)

        # 3. Dense Raw
        t0 = time.perf_counter()
        sc_raw = raw_vecs @ q_vecs_raw[qi]
        res_d_raw = topk_from_scores(sc_raw, k=250)
        latencies_ms["03_Dense_Raw_NoPrompt"].append(mean_single_q_encode_ms + (time.perf_counter() - t0) * 1000.0)

        # 4. Dense EO SingleVec 768d
        t0 = time.perf_counter()
        sc_eo = eo_vecs @ q_vecs_eo[qi]
        res_d_eo = topk_from_scores(sc_eo, k=250)
        t_d_eo_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms["04_Dense_EO_SingleVec_768d"].append(mean_single_q_encode_ms + t_d_eo_ms)

        # 5. Dense EO Chunked MaxSim
        t0 = time.perf_counter()
        sc_ch = maxsim_chunk_scores(q_vecs_eo[qi], chunk_vecs, chunk_doc_idx, n_docs)
        res_d_ch = topk_from_scores(sc_ch, k=250)
        latencies_ms["05_Dense_EO_Chunked_MaxSim"].append(mean_single_q_encode_ms + (time.perf_counter() - t0) * 1000.0)

        # 6. Hybrid RRF Unfiltered
        t0 = time.perf_counter()
        fused_unf = rrf_fuse(res_d_eo, res_bm25, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        t_rrf_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms["06_Hybrid_RRF_Unfiltered"].append(mean_single_q_encode_ms + t_d_eo_ms + t_bm25_ms + t_rrf_ms)

        # 7. Hybrid RRF + Naive Strict Filter (filters before ranking)
        t0 = time.perf_counter()
        d_strict = topk_from_scores(sc_eo, k=250, mask=mask_naive_strict)
        b_strict = [(idx, sc) for idx, sc in res_bm25 if mask_naive_strict[idx]]
        fused_strict = rrf_fuse(d_strict, b_strict, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        latencies_ms["07_Hybrid_RRF_NaiveStrictFilter"].append(
            mean_single_q_encode_ms + t_bm25_ms + (time.perf_counter() - t0) * 1000.0
        )

        # 8. Hybrid RRF + Null-Safe Filter
        t0 = time.perf_counter()
        d_ns = topk_from_scores(sc_eo, k=250, mask=mask_null_safe)
        b_ns = [(idx, sc) for idx, sc in res_bm25 if mask_null_safe[idx]]
        fused_ns = rrf_fuse(d_ns, b_ns, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        t_ns_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms["08_Hybrid_RRF_NullSafeFilter"].append(mean_single_q_encode_ms + t_bm25_ms + t_ns_ms)

        # 9. Full SOTA: Hybrid RRF + Null-Safe Filter + Product-Family Mirror Collapse (768d)
        t0 = time.perf_counter()
        sota_768 = collapse_product_families(fused_ns, df, top_k=50, family_sigs=family_sigs, vectors=eo_vecs)
        t_col_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms["09_SOTA_Hybrid_RRF_NullSafe_Collapse_768d"].append(
            mean_single_q_encode_ms + t_bm25_ms + t_ns_ms + t_col_ms
        )

        # 10-12. Matryoshka 512d / 256d / 128d on Full SOTA
        matryo_results: dict[int, list[tuple[int, float, dict[str, Any]]]] = {}
        for d in (512, 256, 128):
            t0 = time.perf_counter()
            sc_m = matryo_docs[d] @ matryo_qs[d][qi]
            d_m = topk_from_scores(sc_m, k=250, mask=mask_null_safe)
            f_m = rrf_fuse(d_m, b_ns, rrf_k=60, w_dense=1.0, w_bm25=0.6)
            matryo_results[d] = collapse_product_families(
                f_m, df, top_k=50, family_sigs=family_sigs, vectors=matryo_docs[d]
            )
            latencies_ms[f"{ {512: '10', 256: '11', 128: '12'}[d] }_SOTA_Matryoshka_{d}d"].append(
                mean_single_q_encode_ms + t_bm25_ms + (time.perf_counter() - t0) * 1000.0
            )

        # 13. FP16 storage roundtrip check (768d)
        t0 = time.perf_counter()
        sc_f16 = eo_vecs_f16_rt @ q_vecs_eo[qi]
        d_f16 = topk_from_scores(sc_f16, k=250, mask=mask_null_safe)
        f_f16 = rrf_fuse(d_f16, b_ns, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        sota_f16 = collapse_product_families(
            f_f16, df, top_k=50, family_sigs=family_sigs, vectors=eo_vecs_f16_rt
        )
        latencies_ms["13_SOTA_FP16_Storage_768d"].append(
            mean_single_q_encode_ms + t_bm25_ms + (time.perf_counter() - t0) * 1000.0
        )

        ranked_map: dict[str, list[int]] = {
            "01_Naive_Substring_ILIKE": [i for i, _ in res_like],
            "02_BM25_EO_Normalized": [i for i, _ in res_bm25],
            "03_Dense_Raw_NoPrompt": [i for i, _ in res_d_raw],
            "04_Dense_EO_SingleVec_768d": [i for i, _ in res_d_eo],
            "05_Dense_EO_Chunked_MaxSim": [i for i, _ in res_d_ch],
            "06_Hybrid_RRF_Unfiltered": [i for i, _, _ in fused_unf],
            "07_Hybrid_RRF_NaiveStrictFilter": [i for i, _, _ in fused_strict],
            "08_Hybrid_RRF_NullSafeFilter": [i for i, _, _ in fused_ns],
            "09_SOTA_Hybrid_RRF_NullSafe_Collapse_768d": [i for i, _, _ in sota_768],
            "10_SOTA_Matryoshka_512d": [i for i, _, _ in matryo_results[512]],
            "11_SOTA_Matryoshka_256d": [i for i, _, _ in matryo_results[256]],
            "12_SOTA_Matryoshka_128d": [i for i, _, _ in matryo_results[128]],
            "13_SOTA_FP16_Storage_768d": [i for i, _, _ in sota_f16],
        }

        for sys_name, r_idxs in ranked_map.items():
            m = evaluate_ranking(r_idxs, rid_by_idx, fams)
            per_query_rows.append({
                "system": sys_name,
                "query_id": q["query_id"],
                "policy_framework": q["policy_framework"],
                "has_grid_or_geo_filter": has_grid_or_geo,
                "has_max_grid_filter": "max_grid_m" in flt_spec,
                "n_expected_families": q["n_expected_families"],
                **{k: v for k, v in m.items() if k != "missed_families"},
                "missed_families": "; ".join(m["missed_families"]),
            })
            if sys_name == "09_SOTA_Hybrid_RRF_NullSafe_Collapse_768d":
                top5_titles = [
                    {
                        "rank": rk + 1,
                        "record_id": rid_by_idx[idx],
                        "title": str(df.iloc[idx]["title"])[:90],
                        "source": str(df.iloc[idx]["source_catalogue"]),
                        "siblings_collapsed": meta.get("sibling_count", 0),
                    }
                    for rk, (idx, _sc, meta) in enumerate(sota_768[:5])
                ]
                sota_diagnostics.append({
                    "query_id": q["query_id"],
                    "query": qtxt,
                    "filters": flt_spec,
                    "fam_r_10": round(m["fam_r_10"], 3),
                    "fam_r_20": round(m["fam_r_20"], 3),
                    "mrr_10": round(m["mrr_10"], 3),
                    "missed_families_in_top50": m["missed_families"],
                    "top5_returned": top5_titles,
                })

    df_pq = pd.DataFrame(per_query_rows)
    df_pq.to_csv(ARTIFACTS_DIR / "subset_eval_per_query.csv", index=False)

    # Aggregate full 45-query summary
    metric_cols = ["hit_5", "hit_10", "hit_20", "fam_r_5", "fam_r_10", "fam_r_20", "fam_r_50", "mrr_10", "fam_ndcg_10", "fam_ndcg_20"]
    summary = (
        df_pq.groupby("system", sort=True)[metric_cols]
        .mean()
        .reset_index()
    )
    summary["mean_query_ms"] = summary["system"].map(lambda s: round(float(np.mean(latencies_ms[s])), 1))
    for c in metric_cols:
        summary[c] = (summary[c] * 100.0).round(2)

    # Separate slice: queries WITH spatial-resolution filter (`max_grid_m`), highlighting Naive vs Null-Safe!
    grid_slice = (
        df_pq[df_pq["has_max_grid_filter"]]
        .groupby("system", sort=True)[["hit_10", "fam_r_10", "fam_r_20", "mrr_10", "fam_ndcg_10"]]
        .mean()
        .reset_index()
    )
    for c in ["hit_10", "fam_r_10", "fam_r_20", "mrr_10", "fam_ndcg_10"]:
        grid_slice[c] = (grid_slice[c] * 100.0).round(2)

    summary.to_csv(ARTIFACTS_DIR / "subset_eval_summary.csv", index=False)
    grid_slice.to_csv(ARTIFACTS_DIR / "subset_eval_grid_filter_slice.csv", index=False)

    # Save error analysis diagnostics (hardest queries first)
    sota_diagnostics.sort(key=lambda d: (d["fam_r_10"], d["mrr_10"]))
    (ARTIFACTS_DIR / "sota_error_analysis.json").write_text(
        json.dumps(sota_diagnostics, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("ABLATION SUMMARY — ALL 45 POLICY QUERIES (VALUES IN %)")
    print("=" * 88)
    disp = summary[["system", "hit_5", "hit_10", "fam_r_5", "fam_r_10", "fam_r_20", "mrr_10", "fam_ndcg_10", "mean_query_ms"]]
    print(disp.to_string(index=False))

    n_grid_q = int(df_pq[df_pq["system"] == systems[0]]["has_max_grid_filter"].sum())
    print(f"\n--- SLICE: {n_grid_q} QUERIES WITH EXPLICIT SPATIAL RESOLUTION FILTER (max_grid_m) ---")
    focus = grid_slice[grid_slice["system"].isin([
        "06_Hybrid_RRF_Unfiltered",
        "07_Hybrid_RRF_NaiveStrictFilter",
        "08_Hybrid_RRF_NullSafeFilter",
        "09_SOTA_Hybrid_RRF_NullSafe_Collapse_768d",
    ])]
    print(focus.to_string(index=False))

    print("\n--- HARDEST 6 QUERIES FOR SOTA PIPELINE (ERROR ANALYSIS) ---")
    for d in sota_diagnostics[:6]:
        print(f"  {d['query_id']} | FamR@10={d['fam_r_10']*100:.0f}% | FamR@20={d['fam_r_20']*100:.0f}% | MRR@10={d['mrr_10']:.2f} | Missed@50={d['missed_families_in_top50']}")
        for t in d["top5_returned"][:3]:
            print(f"      #{t['rank']} [{t['source']}] {t['title']} (+{t['siblings_collapsed']} sib)")

    print("\nSaved evaluation artifacts in:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
