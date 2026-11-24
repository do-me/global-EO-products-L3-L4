# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "huggingface_hub>=0.30.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "safetensors>=0.4.0",
#   "tokenizers>=0.21.0",
#   "transformers>=5.18.0",
#   "mlx>=0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-vlm>=0.8.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "torch>=2.4.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""Full 39,474-record device-agnostic indexing (MLX on Apple Silicon, PyTorch on CUDA/CPU) + DuckDB BM25.

Builds:
  - artifacts/full_catalogue.parquet   (relative symlink/copy of canonical 39,474 parquet)
  - artifacts/full_bm25.duckdb         (DuckDB native multi-field EO-normalized BM25 index; ~35 s)
  - artifacts/full_embeddings_bf16.npz (float16 39,474 x 768 unit-normalized EmbeddingGemma 2 vectors + IDs + family_sigs)
  - artifacts/full_index_manifest.json (timing, throughput, memory, integrity & SHA-256 verification)
  - artifacts/full_eval_summary.csv    (45-query policy benchmark evaluated against all 39,474 products)

Pass `--device auto|mlx|cuda|cpu` to choose the compute backend, or `--bm25-only` (~35 s) for lexical-only setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CANONICAL_PARQUET = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_INDEX_RELEASE_TAG,
    DEFAULT_MODEL_BF16,
    DuckDBLexicalIndex,
    SearchFilters,
    apply_filters_mask,
    build_eo_document,
    collapse_product_families,
    compute_catalogue_content_sha256,
    compute_family_signature,
    compute_record_ids_sha256,
    deterministic_topk,
    filter_benchmark_family_mask,
    load_dense_embedder,
    matryoshka_truncate,
    resolve_hf_model_revision,
    rrf_fuse,
    verify_index_alignment,
)

# Import benchmark specs cleanly so full-catalogue evaluation uses identical ground-truth definitions
import importlib.util

_b02_spec = importlib.util.spec_from_file_location("b02", SCRIPT_DIR / "02_build_policy_benchmark.py")
_b02 = importlib.util.module_from_spec(_b02_spec)
assert _b02_spec and _b02_spec.loader
_b02_spec.loader.exec_module(_b02)
BENCHMARK_SPECS = _b02.BENCHMARK_SPECS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topk_from_scores(scores: np.ndarray, k: int = 300, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    return deterministic_topk(scores, k=k, mask=mask)


def eval_ranking_full(
    ranked_doc_indices: list[int],
    rid_by_idx: list[str],
    family_rid_sets: dict[str, set[str]],
) -> dict[str, Any]:
    rid_to_fams: dict[str, set[str]] = {}
    for fid, rset in family_rid_sets.items():
        for rid in rset:
            rid_to_fams.setdefault(rid, set()).add(fid)

    n_fams = max(1, len(family_rid_sets))
    first_rel_rank = 0
    seen_fams: set[str] = set()
    dcg10 = 0.0
    dcg20 = 0.0
    fam_counts_at = {5: 0, 10: 0, 20: 0, 50: 0}
    hit_at = {5: 0.0, 10: 0.0, 20: 0.0}

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
        if rank in fam_counts_at:
            fam_counts_at[rank] = len(seen_fams)
        if rank in hit_at:
            hit_at[rank] = 1.0 if len(seen_fams) > 0 else 0.0

    for k in (5, 10, 20, 50):
        if len(ranked_doc_indices) < k:
            fam_counts_at[k] = len(seen_fams)
        if k in hit_at and len(ranked_doc_indices) < k:
            hit_at[k] = 1.0 if len(seen_fams) > 0 else 0.0

    idcg10 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 10) + 1))
    idcg20 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 20) + 1))
    return {
        "hit_5": hit_at[5],
        "hit_10": hit_at[10],
        "hit_20": hit_at[20],
        "fam_r_5": fam_counts_at[5] / n_fams,
        "fam_r_10": fam_counts_at[10] / n_fams,
        "fam_r_20": fam_counts_at[20] / n_fams,
        "fam_r_50": fam_counts_at[50] / n_fams,
        "mrr_10": (1.0 / first_rel_rank) if (0 < first_rel_rank <= 10) else 0.0,
        "fam_ndcg_10": dcg10 / idcg10 if idcg10 > 0 else 0.0,
        "fam_ndcg_20": dcg20 / idcg20 if idcg20 > 0 else 0.0,
        "missed_families": sorted(set(family_rid_sets.keys()) - seen_fams),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full 39,474-record BM25 + EmbeddingGemma-2 dense indexes.")
    parser.add_argument(
        "--device",
        choices=["auto", "mlx", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute backend (default: auto = MLX on Apple Silicon, CUDA if available, else CPU PyTorch).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for dense embedding pass (default: 16).",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only the cross-platform DuckDB BM25 index (~35 s) without dense embeddings.",
    )
    args = parser.parse_args()

    t_all0 = time.perf_counter()
    print("=" * 88, flush=True)
    print(f"05. FULL-CATALOGUE (39,474 RECORDS) HYBRID INDEXING (device={args.device}) + DUCKDB", flush=True)
    print("=" * 88, flush=True)

    df = pd.read_parquet(CANONICAL_PARQUET)
    n_total = len(df)
    rid_by_idx = df["record_id"].tolist()
    print(f"Loaded {n_total:,} canonical records from {CANONICAL_PARQUET.name}", flush=True)

    # Link (relative) or copy canonical parquet into artifacts/full_catalogue.parquet
    full_pq = ARTIFACTS_DIR / "full_catalogue.parquet"
    if full_pq.exists() or full_pq.is_symlink():
        full_pq.unlink()
    try:
        full_pq.symlink_to(os.path.relpath(CANONICAL_PARQUET, ARTIFACTS_DIR))
    except OSError:
        shutil.copy2(CANONICAL_PARQUET, full_pq)

    # 1. Build EO-structured documents & family signatures
    cat_sha = sha256_file(full_pq)
    t0 = time.perf_counter()
    rows = df.to_dict(orient="records")
    eo_docs = [build_eo_document(r) for r in rows] if not args.bm25_only else []
    family_sigs = [
        compute_family_signature(
            str(r.get("title", "")),
            str(r.get("raw_id", "")),
            str(r.get("canonical_product_key", "")),
            cadence_class=str(r.get("temporal_cadence_class") or ""),
            grid_meters=r.get("adjudicated_output_grid_meters"),
            geo_class=str(r.get("geographic_coverage_class") or ""),
        )
        for r in rows
    ] if not args.bm25_only else []
    t_doc_build = time.perf_counter() - t0
    if not args.bm25_only:
        print(f"[1/4] Built {n_total:,} EO-structured docs & family signatures in {t_doc_build:.2f}s", flush=True)

    # 2. Build full DuckDB native multi-field BM25 index
    bm25_db = ARTIFACTS_DIR / "full_bm25.duckdb"
    if bm25_db.exists():
        bm25_db.unlink()
    lex_index = DuckDBLexicalIndex(str(bm25_db))
    t_bm25 = lex_index.build_from_dataframe(df, catalogue_sha256=cat_sha)
    lex_index.con.close()
    print(
        f"[2/4] Built DuckDB multi-field BM25 index ({n_total:,} docs) in {t_bm25:.2f}s "
        f"({bm25_db.stat().st_size / (1024**2):.1f} MB) -> {bm25_db.name}",
        flush=True,
    )
    if args.bm25_only:
        print(f"Done (--bm25-only) in {time.perf_counter() - t_all0:.2f}s.", flush=True)
        return
    # Reopen read-only (enforcing SET threads=1) and bind ordered record_ids cleanly
    lex_index = DuckDBLexicalIndex(str(bm25_db), read_only=True)
    lex_index.verify_against_dataframe(df, catalogue_sha256=cat_sha)

    # 3. Run full 39,474-record EmbeddingGemma 2 inference (auto: MLX -> CUDA -> CPU)
    enc, resolved_device, resolved_model_id = load_dense_embedder(device=args.device)
    print(
        f"[3/4] Loaded {resolved_model_id} on {resolved_device} in {enc.load_seconds:.2f}s; starting full pass...",
        flush=True,
    )

    batch_size = int(args.batch_size)
    fixed_buckets = [64, 128, 192, 256, 320, 384, 448, 512]
    vecs, st = enc.encode_texts(
        eo_docs,
        batch_size=batch_size,
        max_length=512,
        sort_by_length=True,
        prefetch=True,
        async_pipe=True,
        fixed_buckets=fixed_buckets,
        use_compile=True,
        clear_cache_every=64,
        progress_every=250,
        return_stats=True,
    )

    # Integrity checks (strict zero NaN/Inf, all unit norm)
    assert vecs.shape == (n_total, 768), f"Unexpected shape: {vecs.shape}"
    assert np.isfinite(vecs).all(), "Non-finite embedding values detected!"
    norms = np.linalg.norm(vecs, axis=1)
    assert float(norms.min()) > 0.99 and float(norms.max()) < 1.01, f"Bad norms: {norms.min()}..{norms.max()}"

    emb_path = ARTIFACTS_DIR / "full_embeddings_bf16.npz"
    t_save0 = time.perf_counter()
    np.savez_compressed(
        emb_path,
        embeddings_f16=vecs.astype(np.float16),
        record_ids=np.array(rid_by_idx, dtype=object),
        family_sigs=np.array(family_sigs, dtype=object),
        catalogue_sha256=np.array(cat_sha, dtype=object),
    )
    with np.load(emb_path, allow_pickle=True) as emb_chk:
        verify_index_alignment(df, lex_index=lex_index, emb_npz=emb_chk, catalogue_path=full_pq)
    t_save = time.perf_counter() - t_save0
    emb_mb = emb_path.stat().st_size / (1024**2)
    print(
        f"[3/4] DONE full {resolved_device.upper()} pass: {n_total:,} docs in {st['total_seconds']:.1f}s ({st['total_seconds']/60:.2f} min) "
        f"| {st['docs_per_sec']:.1f} doc/s | {st['real_tok_per_sec']:.0f} tok/s | peak={st['peak_mem_gb']:.2f} GB",
        flush=True,
    )
    print(f"      Saved {emb_path.name} ({emb_mb:.2f} MB in {t_save:.2f}s) | norms=[{norms.min():.6f}, {norms.max():.6f}]", flush=True)

    # 4. Evaluate all 45 zero-leakage policy queries against the FULL 39,474-record catalogue!
    print("\n[4/4] Evaluating all 45 policy queries against ALL 39,474 canonical products...", flush=True)
    obs_re = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)
    active_mask = (~df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)) & (
        ~df["title"].fillna("").str.contains(obs_re, regex=True).to_numpy(dtype=bool, copy=True)
    )
    combined = (
        df["title"].fillna("")
        + " || "
        + df["raw_id"].fillna("")
        + " || "
        + df["source_catalogue"].fillna("")
    )

    # Prepare fp16 roundtrip normalized vectors (exact production storage representation)
    vecs_f16_rt = vecs.astype(np.float16).astype(np.float32)
    vecs_f16_rt /= np.maximum(np.linalg.norm(vecs_f16_rt, axis=1, keepdims=True), 1e-12)
    vecs_512 = matryoshka_truncate(vecs_f16_rt, 512)
    vecs_256 = matryoshka_truncate(vecs_f16_rt, 256)

    q_texts = [sp["query"] for sp in BENCHMARK_SPECS]
    # Solo batch_size=1 query encoding for exact cross-script parity with 04/06/07/10/11
    _ = enc.encode_queries(["warmup satellite query"], max_length=256)
    sq_ms = []
    qv_list = []
    for qt in q_texts:
        ts = time.perf_counter()
        qv_list.append(enc.encode_queries([qt], max_length=256)[0])
        sq_ms.append((time.perf_counter() - ts) * 1000.0)
    q_vecs = np.vstack(qv_list)
    q_vecs_512 = matryoshka_truncate(q_vecs, 512)
    q_vecs_256 = matryoshka_truncate(q_vecs, 256)
    mean_q_enc_ms = float(np.mean(sq_ms))

    systems = [
        "01_BM25_Full39k",
        "02_Dense_EO_Full39k_768d",
        "03_Hybrid_RRF_Unfiltered_Full39k",
        "04_Hybrid_RRF_NaiveStrictFilter_Full39k",
        "05_Hybrid_RRF_NullSafeFilter_Full39k",
        "06_SOTA_Hybrid_RRF_NullSafe_Collapse_768d",
        "07_SOTA_Hybrid_RRF_NullSafe_Collapse_512d",
        "08_SOTA_Hybrid_RRF_NullSafe_Collapse_256d",
    ]
    lat_ms: dict[str, list[float]] = {s: [] for s in systems}
    per_query_rows: list[dict[str, Any]] = []
    full_diagnostics: list[dict[str, Any]] = []

    for qi, sp in enumerate(BENCHMARK_SPECS):
        qtxt = sp["query"]
        flt_spec = sp.get("filters", {})

        # Resolve exact full-catalogue relevant record_ids per family
        fam_rids: dict[str, set[str]] = {}
        for fam in sp["expected_families"]:
            pat = re.compile(fam["regex"], re.IGNORECASE)
            hit = active_mask & combined.str.contains(pat, regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                hit = hit & ~combined.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            hit = filter_benchmark_family_mask(df, hit, flt_spec, query_text=qtxt)
            idxs = np.flatnonzero(hit)
            if len(idxs) > 0:
                fam_rids[fam["family_id"]] = set(df.iloc[idxs]["record_id"].tolist())

        # Note: use ontology_mode="eo_and_hybrid" if strict_eo_only was requested?
        # Keep exact identical filter semantics to script 03 for clean apples-to-apples comparison!
        flt_ns = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            exclude_superseded=True,
            null_safe=True,
        )
        flt_strict = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            exclude_superseded=True,
            null_safe=False,
        )
        mask_ns = apply_filters_mask(df, flt_ns)
        mask_strict = apply_filters_mask(df, flt_strict)

        t0 = time.perf_counter()
        res_bm25 = lex_index.search_bm25(qtxt, top_k=300)
        t_bm = (time.perf_counter() - t0) * 1000.0
        lat_ms["01_BM25_Full39k"].append(t_bm)

        t0 = time.perf_counter()
        sc_768 = vecs_f16_rt @ q_vecs[qi]
        res_d768 = topk_from_scores(sc_768, k=300)
        t_d = (time.perf_counter() - t0) * 1000.0
        lat_ms["02_Dense_EO_Full39k_768d"].append(mean_q_enc_ms + t_d)

        t0 = time.perf_counter()
        f_unf = rrf_fuse(res_d768, res_bm25, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        lat_ms["03_Hybrid_RRF_Unfiltered_Full39k"].append(mean_q_enc_ms + t_d + t_bm + (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        d_st = topk_from_scores(sc_768, k=300, mask=mask_strict)
        b_st = lex_index.search_bm25(qtxt, top_k=300, eligible_mask=mask_strict)
        f_st = rrf_fuse(d_st, b_st, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        lat_ms["04_Hybrid_RRF_NaiveStrictFilter_Full39k"].append(mean_q_enc_ms + (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        d_ns = topk_from_scores(sc_768, k=300, mask=mask_ns)
        b_ns = lex_index.search_bm25(qtxt, top_k=300, eligible_mask=mask_ns)
        f_ns = rrf_fuse(d_ns, b_ns, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        t_ns = (time.perf_counter() - t0) * 1000.0
        lat_ms["05_Hybrid_RRF_NullSafeFilter_Full39k"].append(mean_q_enc_ms + t_ns)

        t0 = time.perf_counter()
        sota_768 = collapse_product_families(f_ns, df, top_k=50, family_sigs=family_sigs, vectors=vecs_f16_rt, near_dup_cos=0.94, sensor_guard=True, query_text=qtxt)
        t_col = (time.perf_counter() - t0) * 1000.0
        lat_ms["06_SOTA_Hybrid_RRF_NullSafe_Collapse_768d"].append(mean_q_enc_ms + t_ns + t_col)

        t0 = time.perf_counter()
        sc_512 = vecs_512 @ q_vecs_512[qi]
        f_512 = rrf_fuse(topk_from_scores(sc_512, k=300, mask=mask_ns), b_ns, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        sota_512 = collapse_product_families(f_512, df, top_k=50, family_sigs=family_sigs, vectors=vecs_512, near_dup_cos=0.94, sensor_guard=True, query_text=qtxt)
        lat_ms["07_SOTA_Hybrid_RRF_NullSafe_Collapse_512d"].append(mean_q_enc_ms + (time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        sc_256 = vecs_256 @ q_vecs_256[qi]
        f_256 = rrf_fuse(topk_from_scores(sc_256, k=300, mask=mask_ns), b_ns, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        sota_256 = collapse_product_families(f_256, df, top_k=50, family_sigs=family_sigs, vectors=vecs_256, near_dup_cos=0.94, sensor_guard=True, query_text=qtxt)
        lat_ms["08_SOTA_Hybrid_RRF_NullSafe_Collapse_256d"].append(mean_q_enc_ms + (time.perf_counter() - t0) * 1000.0)

        ranked_map = {
            "01_BM25_Full39k": [i for i, _ in res_bm25],
            "02_Dense_EO_Full39k_768d": [i for i, _ in res_d768],
            "03_Hybrid_RRF_Unfiltered_Full39k": [i for i, _, _ in f_unf],
            "04_Hybrid_RRF_NaiveStrictFilter_Full39k": [i for i, _, _ in f_st],
            "05_Hybrid_RRF_NullSafeFilter_Full39k": [i for i, _, _ in f_ns],
            "06_SOTA_Hybrid_RRF_NullSafe_Collapse_768d": [i for i, _, _ in sota_768],
            "07_SOTA_Hybrid_RRF_NullSafe_Collapse_512d": [i for i, _, _ in sota_512],
            "08_SOTA_Hybrid_RRF_NullSafe_Collapse_256d": [i for i, _, _ in sota_256],
        }

        for sname, r_idxs in ranked_map.items():
            m = eval_ranking_full(r_idxs, rid_by_idx, fam_rids)
            per_query_rows.append({
                "system": sname,
                "query_id": sp["query_id"],
                "has_max_grid_filter": "max_grid_m" in flt_spec,
                "n_expected_families": len(fam_rids),
                **{k: v for k, v in m.items() if k != "missed_families"},
                "missed_families": "; ".join(m["missed_families"]),
            })
            if sname == "06_SOTA_Hybrid_RRF_NullSafe_Collapse_768d":
                full_diagnostics.append({
                    "query_id": sp["query_id"],
                    "fam_r_10": round(m["fam_r_10"], 3),
                    "fam_r_20": round(m["fam_r_20"], 3),
                    "mrr_10": round(m["mrr_10"], 3),
                    "missed_families_in_top50": m["missed_families"],
                    "top5_returned": [
                        {
                            "rank": rk + 1,
                            "record_id": rid_by_idx[idx],
                            "title": str(df.iloc[idx]["title"])[:90],
                            "source": str(df.iloc[idx]["source_catalogue"]),
                            "siblings_collapsed": meta.get("sibling_count", 0),
                        }
                        for rk, (idx, _sc, meta) in enumerate(sota_768[:5])
                    ],
                })

    df_pq = pd.DataFrame(per_query_rows)
    df_pq.to_csv(ARTIFACTS_DIR / "full_eval_per_query.csv", index=False)

    mcols = ["hit_5", "hit_10", "hit_20", "fam_r_5", "fam_r_10", "fam_r_20", "fam_r_50", "mrr_10", "fam_ndcg_10", "fam_ndcg_20"]
    summary = df_pq.groupby("system", sort=True)[mcols].mean().reset_index()
    summary["mean_query_ms"] = summary["system"].map(lambda s: round(float(np.mean(lat_ms[s])), 1))
    for c in mcols:
        summary[c] = (summary[c] * 100.0).round(2)
    summary.to_csv(ARTIFACTS_DIR / "full_eval_summary.csv", index=False)

    full_diagnostics.sort(key=lambda d: (d["fam_r_10"], d["mrr_10"]))
    (ARTIFACTS_DIR / "full_error_analysis.json").write_text(json.dumps(full_diagnostics, indent=2), encoding="utf-8")

    bm25_content_sha = lex_index.compute_content_sha256()
    record_ids_sha = compute_record_ids_sha256(rid_by_idx)
    cat_content_sha = compute_catalogue_content_sha256(df)
    model_rev = resolve_hf_model_revision(resolved_model_id)
    lex_index.con.close()
    manifest = {
        "corpus_records": n_total,
        "device": resolved_device,
        "model_id": resolved_model_id,
        "model_revision": model_rev,
        "release_tag": DEFAULT_INDEX_RELEASE_TAG,
        "batch_size": batch_size,
        "fixed_buckets": fixed_buckets,
        "max_length": 512,
        "doc_build_seconds": round(t_doc_build, 2),
        "bm25_build_seconds": round(t_bm25, 2),
        "bm25_db_mb": round(bm25_db.stat().st_size / (1024**2), 2),
        "embed_stats": st,
        "mlx_stats": st,
        "embeddings_npz_mb": round(emb_mb, 2),
        "catalogue_parquet_mb": round(full_pq.stat().st_size / (1024**2), 2),
        "norm_min": round(float(norms.min()), 6),
        "norm_max": round(float(norms.max()), 6),
        "total_wall_seconds": round(time.perf_counter() - t_all0, 2),
        "single_query_encode_ms": round(mean_q_enc_ms, 2),
        "family_sigs_type": "100pct_generic_zero_hardcoding_var_role_cadence_grid_geo",
        "unique_generic_family_signatures": int(len(set(family_sigs))),
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalogue_content_sha256": cat_content_sha,
        "record_ids_sha256": record_ids_sha,
        "bm25_content_sha256": bm25_content_sha,
        "sha256": {
            "eo_search_lib.py": sha256_file(SCRIPT_DIR / "eo_search_lib.py"),
            "06_eval_rerankers.py": sha256_file(SCRIPT_DIR / "06_eval_rerankers.py"),
            "full_catalogue.parquet": cat_sha,
            "catalogue_content_sha256": cat_content_sha,
            "full_embeddings_bf16.npz": sha256_file(emb_path),
            "full_bm25.duckdb": sha256_file(bm25_db),
            "bm25_content_sha256": bm25_content_sha,
            "record_ids_sha256": record_ids_sha,
        },
    }
    (ARTIFACTS_DIR / "full_index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 88, flush=True)
    print("FULL 39,474-CATALOGUE ABLATION SUMMARY — ALL 45 POLICY QUERIES (VALUES IN %)", flush=True)
    print("=" * 88, flush=True)
    disp = summary[["system", "hit_5", "hit_10", "fam_r_5", "fam_r_10", "fam_r_20", "fam_r_50", "mrr_10", "fam_ndcg_10", "mean_query_ms"]]
    print(disp.to_string(index=False), flush=True)
    print(f"\nTotal end-to-end wall time: {manifest['total_wall_seconds']:.1f}s ({manifest['total_wall_seconds']/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
