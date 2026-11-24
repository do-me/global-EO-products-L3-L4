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
#   "mlx-lm>=0.25.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-vlm>=0.8.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "torch>=2.4.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""Device-agnostic CLI search client for the Global Harmonized L3/L4 EO Product Catalogue.

Auto-selects the best available backend (`--device auto`):
  - Apple Silicon (macOS arm64): native MLX (`mlx-community/embeddinggemma-2-bf16`, `jinaai/jina-reranker-v3.5-mlx`, `Qwen/Qwen3-Reranker-0.6B`)
  - NVIDIA GPU (CUDA): PyTorch BF16/FP32 (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`)
  - CPU-only fallback: PyTorch CPU (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`)

Supports two execution modes:
  1. Standalone local mode (default): loads local indexes directly and searches all 39,474 canonical products.
  2. Zero-cold-start client mode (`--server http://127.0.0.1:8080`): sends queries to a running
     `04_search_server.py` instance so models stay warm across requests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
CANONICAL_PARQUET = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DuckDBLexicalIndex,
    SearchFilters,
    apply_filters_mask,
    collapse_product_families,
    compute_generic_family_signature,
    compute_unknown_metadata_counts,
    deterministic_topk,
    ensure_search_indexes,
    format_resolution_human,
    load_dense_embedder,
    matryoshka_truncate,
    rrf_fuse,
    verify_index_alignment,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Device-Agnostic Hybrid Semantic + BM25 + Reranker EO Product Search")
    p.add_argument("--query", "-q", required=True, help="Natural-language policy or EO product query.")
    p.add_argument(
        "--server",
        default=None,
        help="Optional URL of a running 04_search_server.py instance (e.g. http://127.0.0.1:8080) for zero-cold-start queries.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "mlx", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute backend (default: auto = MLX on Apple Silicon, CUDA if available, else CPU PyTorch).",
    )
    p.add_argument(
        "--index-prefix",
        default="full",
        choices=["full", "subset"],
        help="Which local index to search (default: full = all 39,474 canonical products).",
    )
    p.add_argument(
        "--geo",
        choices=["global", "europe"],
        default=None,
        help="Geographic scope filter (null-safe by default; 'global' includes global, quasi-global, and hemispheric/pantropical).",
    )
    p.add_argument(
        "--exclude-local",
        action="store_true",
        help="Exclude confirmed regional_or_local datasets while keeping global, hemispheric, and european-scale products.",
    )
    p.add_argument("--max-grid-m", type=float, default=None, help="Maximum spatial grid size in meters (null-safe by default).")
    p.add_argument(
        "--eo-scope",
        choices=["all", "eo_and_hybrid", "strict_eo"],
        default="all",
        help="Ontology filter: 'all' (default), 'eo_and_hybrid' (EO+reanalysis+EO-hybrids), or 'strict_eo'.",
    )
    p.add_argument("--strict-eo", action="store_true", help="Alias for --eo-scope strict_eo.")
    p.add_argument("--pool", choices=["copernicus_services", "eu_non_copernicus", "non_eu"], default=None)
    p.add_argument("--level", nargs="+", choices=["L3", "L4"], default=None, help="Allowed processing levels (null-safe).")
    p.add_argument("--min-end-year", type=int, default=None)
    p.add_argument("--top-k", "-k", type=int, default=10)
    p.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=768, help="Matryoshka embedding dimension.")
    p.add_argument("--w-dense", type=float, default=1.0)
    p.add_argument("--w-bm25", type=float, default=0.6)
    p.add_argument(
        "--bm25-only",
        action="store_true",
        help="Run cross-platform DuckDB BM25 lexical search only (no embeddings or reranker required).",
    )
    p.add_argument(
        "--rerank",
        choices=["none", "jina-v3.5", "qwen3-0.6b"],
        default="jina-v3.5",
        help="Stage-2 reranker (default: 'jina-v3.5' [CC BY-NC 4.0]; 'qwen3-0.6b' [Apache-2.0]; 'none' [Stage-1 hybrid]).",
    )
    p.add_argument(
        "--rerank-pool",
        type=int,
        default=50,
        help="Collapsed Stage-1 families passed to reranker (default: 50; 25 for fast reranking).",
    )
    p.add_argument("--rerank-pure", "--no-blend", dest="rerank_pure", action="store_true", help="Use pure reranker order without Stage-1 RRF blend.")
    p.add_argument("--no-collapse", action="store_true", help="Disable product-family mirror/edition collapsing.")
    p.add_argument("--naive-filter", "--strict-filter", dest="naive_filter", action="store_true", help="Drop NULL metadata fields strictly under orthogonal 1.25x grid tolerance.")
    p.add_argument(
        "--hard-limits",
        action="store_true",
        help="Verified mode: drop unknown metadata (NULLs), enforce exact 1.0x grid cutoff, and restrict --geo global to true global footprints.",
    )
    p.add_argument("--rebuild-index", action="store_true", help="Rebuild local BM25/dense index from canonical parquet before searching.")
    p.add_argument("--no-download-index", action="store_true", help="Do not download missing/stale release indexes from GitHub Releases.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON on stdout.")
    return p.parse_args()


def _resolve_catalogue_path(index_prefix: str) -> Path:
    local_pq = ARTIFACTS_DIR / f"{index_prefix}_catalogue.parquet"
    if local_pq.exists():
        return local_pq
    if index_prefix == "full" and CANONICAL_PARQUET.exists():
        return CANONICAL_PARQUET
    raise FileNotFoundError(
        f"Missing catalogue parquet: {local_pq}\n"
        "Run 05_index_full_catalogue.py (for 'full') or 02_build_policy_benchmark.py (for 'subset')."
    )


def _query_server(args: argparse.Namespace) -> dict[str, Any]:
    base = args.server.rstrip("/")
    if args.bm25_only:
        mode = "bm25"
    elif args.rerank == "none":
        mode = "hybrid"
    elif args.rerank == "qwen3-0.6b":
        mode = "qwen3-0.6b"
    elif args.rerank_pool <= 25 and args.rerank_pure:
        mode = "jina-v3.5-fast"
    else:
        mode = "jina-v3.5"

    body = {
        "query": args.query,
        "mode": mode,
        "rerank_pool": args.rerank_pool,
        "rerank_pure": args.rerank_pure,
        "top_k": args.top_k,
        "pool": args.pool,
        "geo": args.geo,
        "exclude_local": args.exclude_local,
        "max_grid_m": args.max_grid_m,
        "eo_scope": "strict_eo" if args.strict_eo else args.eo_scope,
        "levels": args.level,
        "min_end_year": args.min_end_year,
        "null_safe": not args.naive_filter,
        "hard_limits": args.hard_limits,
        "collapse": not args.no_collapse,
        "dim": args.dim,
        "w_dense": args.w_dense,
        "w_bm25": args.w_bm25,
    }
    req = urllib.request.Request(
        f"{base}/api/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Could not reach search server at {base} ({exc}).\n"
            "Start it first with: uv run 06_semantic_hybrid_search/04_search_server.py"
        ) from exc


def _apply_stage2_rerank(
    args: argparse.Namespace,
    df: pd.DataFrame,
    stage1_items: list[tuple[int, float, dict[str, Any]]],
    resolved_device: str,
) -> tuple[list[tuple[int, float, dict[str, Any]]], float]:
    spec = importlib.util.spec_from_file_location("ev06", SCRIPT_DIR / "06_eval_rerankers.py")
    ev06 = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(ev06)

    docs = [ev06.format_reranker_doc(df.iloc[idx].to_dict(), max_chars=1200) for idx, _, _ in stage1_items]
    rr, _, _ = ev06.load_reranker(args.rerank, device=resolved_device, max_doc_length=420, verify_exactness=False)
    t0 = time.perf_counter()
    if args.rerank == "jina-v3.5":
        scores = rr.score_candidates(args.query, docs)
    else:
        scores = rr.score_candidates(args.query, docs, batch_size=8, max_length=512)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    rerank_order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    rerank_rank = {int(pos): rk + 1 for rk, pos in enumerate(rerank_order)}

    reordered = []
    for s1_pos, (doc_idx, _rrf_sc, meta) in enumerate(stage1_items):
        m2 = dict(meta)
        m2["rerank_score"] = round(float(scores[s1_pos]), 5)
        m2["rerank_rank"] = rerank_rank[s1_pos]
        m2["stage1_rank"] = s1_pos + 1
        if args.rerank_pure:
            final_sc = float(scores[s1_pos])
        else:
            final_sc = 0.4 / (30.0 + s1_pos + 1) + 1.0 / (30.0 + rerank_rank[s1_pos])
        reordered.append((doc_idx, final_sc, m2, s1_pos))

    reordered.sort(key=lambda x: (-x[1], x[3]))
    return [(d, s, m) for d, s, m, _ in reordered[: args.top_k]], dt_ms


def _print_human(payload: dict[str, Any]) -> None:
    print(f"\nQuery     : {payload['query']}")
    print(
        f"Index     : {payload.get('index_prefix', 'server')} ({payload['corpus_size']:,} docs | "
        f"{payload['eligible_after_filter']:,} eligible | device={payload.get('device', 'server')} | "
        f"mode={payload.get('mode_used', payload.get('reranker'))} | null_safe={payload['null_safe_filter']} | "
        f"hard_limits={payload.get('hard_limits', False)})"
    )
    if payload.get("hard_limits") and payload.get("unknown_metadata_counts"):
        u = payload["unknown_metadata_counts"]
        print(
            f"Unknowns  : excluded unknown metadata (level={u.get('processing_level', 0):,}, "
            f"grid={u.get('grid_meters', 0):,}, geo={u.get('geographic_coverage', 0):,}, "
            f"cadence={u.get('cadence', 0):,})"
        )
    startup_s = payload.get("load_seconds", 0.0)
    print(
        f"Timing    : stage1={payload['stage1_ms']:.1f} ms | rerank={payload['rerank_ms']:.1f} ms "
        f"(startup={startup_s:.2f} s)\n"
    )
    for r in payload["results"]:
        sib_n = r.get("collapsed_siblings", 0)
        if isinstance(sib_n, list):
            sib_n = len(sib_n)
        sib_tag = f" [+{sib_n} mirrors/editions]" if sib_n else ""
        rr_tag = (
            f" | rerank={r['rerank_score']} (#{r['rerank_rank']}, s1=#{r['stage1_rank']})"
            if r.get("rerank_score") is not None
            else ""
        )
        meta_line = (
            f"src={r['source_catalogue']} | lvl={r.get('processing_level') or 'null'} | "
            f"geo={r.get('geographic_coverage') or 'null'} | grid={r.get('grid') or 'null'} | "
            f"dense={r.get('dense_score')} (#{r.get('dense_rank')}) | bm25=#{r.get('bm25_rank')}{rr_tag}"
        )
        print(f" {r['rank']:2d}. {r['title']}{sib_tag}")
        print(f"     {meta_line}")
        print(f"     id={r['record_id']}")


def main() -> None:
    args = parse_args()

    if args.server:
        payload = _query_server(args)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_human(payload)
        return

    pq_path = _resolve_catalogue_path(args.index_prefix)
    emb_path = ARTIFACTS_DIR / f"{args.index_prefix}_embeddings_bf16.npz"
    db_path = ARTIFACTS_DIR / f"{args.index_prefix}_bm25.duckdb"

    if args.index_prefix == "full":
        if args.rebuild_index:
            cmd = [sys.executable, str(SCRIPT_DIR / "05_index_full_catalogue.py")]
            if args.bm25_only:
                cmd.append("--bm25-only")
            subprocess.run(cmd, check=True)
        else:
            ensure_search_indexes(
                ARTIFACTS_DIR,
                require_dense=not args.bm25_only,
                allow_download=not args.no_download_index,
            )

    if not db_path.exists():
        raise FileNotFoundError(
            f"Missing BM25 index: {db_path}\n"
            "Rebuild locally (~35 s on any OS) with:\n"
            "  uv run 06_semantic_hybrid_search/05_index_full_catalogue.py --bm25-only"
        )

    use_bm25_only = args.bm25_only or (not emb_path.exists())
    if use_bm25_only and not args.bm25_only:
        print(
            "[info] Dense embeddings file missing; running in BM25 lexical mode. "
            "Run `uv run 06_semantic_hybrid_search/05_index_full_catalogue.py` to build dense embeddings.",
            file=sys.stderr,
        )

    t0 = time.perf_counter()
    df = pd.read_parquet(pq_path)
    vecs: np.ndarray | None = None
    enc: Any = None
    z: Any = None
    resolved_device = "bm25"
    if not use_bm25_only:
        try:
            enc, resolved_device, _ = load_dense_embedder(device=args.device)
            z = np.load(emb_path, allow_pickle=True)
            vecs = z["embeddings_f16"].astype(np.float32)
            vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
            if "family_sigs" in z:
                family_sigs = [str(s) for s in z["family_sigs"].tolist()]
            else:
                family_sigs = [
                    compute_generic_family_signature(
                        str(t),
                        str(c),
                        cadence_class=str(cad or ""),
                        grid_meters=gm,
                        geo_class=str(geo or ""),
                    )
                    for t, c, cad, gm, geo in zip(
                        df["title"],
                        df["canonical_product_key"],
                        df.get("temporal_cadence_class", [""] * len(df)),
                        df.get("adjudicated_output_grid_meters", [None] * len(df)),
                        df.get("geographic_coverage_class", [""] * len(df)),
                    )
                ]
        except RuntimeError as exc:
            if args.device != "auto":
                raise
            print(f"[info] Falling back to BM25 lexical mode ({exc}).", file=sys.stderr)
            use_bm25_only = True
            family_sigs = [
                compute_generic_family_signature(
                    str(t),
                    str(c),
                    cadence_class=str(cad or ""),
                    grid_meters=gm,
                    geo_class=str(geo or ""),
                )
                for t, c, cad, gm, geo in zip(
                    df["title"],
                    df["canonical_product_key"],
                    df.get("temporal_cadence_class", [""] * len(df)),
                    df.get("adjudicated_output_grid_meters", [None] * len(df)),
                    df.get("geographic_coverage_class", [""] * len(df)),
                )
            ]
    else:
        family_sigs = [
            compute_generic_family_signature(
                str(t),
                str(c),
                cadence_class=str(cad or ""),
                grid_meters=gm,
                geo_class=str(geo or ""),
            )
            for t, c, cad, gm, geo in zip(
                df["title"],
                df["canonical_product_key"],
                df.get("temporal_cadence_class", [""] * len(df)),
                df.get("adjudicated_output_grid_meters", [None] * len(df)),
                df.get("geographic_coverage_class", [""] * len(df)),
            )
        ]

    lex = DuckDBLexicalIndex(str(db_path), read_only=True)
    verify_index_alignment(
        df,
        lex_index=lex,
        emb_npz=z if (not use_bm25_only and z is not None) else None,
        catalogue_path=pq_path,
    )
    unknown_counts = compute_unknown_metadata_counts(df, active_only=True)
    t_load = time.perf_counter() - t0

    t_q0 = time.perf_counter()
    eff_null_safe = False if args.hard_limits else (not args.naive_filter)
    flt = SearchFilters(
        geo_scope=args.geo,
        exclude_local=args.exclude_local,
        max_grid_m=args.max_grid_m,
        processing_levels=args.level,
        strict_eo_only=args.strict_eo,
        ontology_mode="strict_eo" if args.strict_eo else args.eo_scope,
        exclude_superseded=True,
        provider_pool=args.pool,
        min_end_year=args.min_end_year,
        null_safe=eff_null_safe,
        hard_limits=args.hard_limits,
    )
    mask = apply_filters_mask(df, flt)
    n_eligible = int(mask.sum())

    bm25_ranked = lex.search_bm25(args.query, top_k=300, eligible_mask=mask)
    if use_bm25_only:
        fused = [
            (doc_idx, float(sc), {"bm25_rank": rk, "bm25_score": round(float(sc), 5)})
            for rk, (doc_idx, sc) in enumerate(bm25_ranked, start=1)
        ]
        effective_rerank = "none"
    else:
        assert enc is not None and vecs is not None
        q_vec = enc.encode_queries([args.query], max_length=256)
        if args.dim < 768:
            vecs = matryoshka_truncate(vecs, args.dim)
            q_vec = matryoshka_truncate(q_vec, args.dim)
        dense_ranked = deterministic_topk(vecs @ q_vec[0], k=300, mask=mask)
        fused = rrf_fuse(dense_ranked, bm25_ranked, rrf_k=60, w_dense=args.w_dense, w_bm25=args.w_bm25)
        effective_rerank = args.rerank

    stage1_k = max(args.top_k, args.rerank_pool) if effective_rerank != "none" else args.top_k
    if args.no_collapse:
        stage1_final = fused[:stage1_k]
    else:
        stage1_final = collapse_product_families(
            fused,
            df,
            top_k=stage1_k,
            family_sigs=family_sigs,
            vectors=vecs,
            near_dup_cos=0.94 if vecs is not None else 1.0,
            sensor_guard=True,
            query_text=args.query,
        )
    t_stage1_ms = (time.perf_counter() - t_q0) * 1000.0

    rerank_ms = 0.0
    if effective_rerank != "none" and stage1_final:
        reranked, rerank_ms = _apply_stage2_rerank(args, df, stage1_final[: args.rerank_pool], resolved_device)
        final = reranked[: args.top_k]
        if args.top_k > len(final) and len(stage1_final) > args.rerank_pool:
            final = final + stage1_final[args.rerank_pool : args.top_k]
    else:
        final = stage1_final[: args.top_k]

    results: list[dict[str, Any]] = []
    for rank, (doc_idx, score_val, meta) in enumerate(final, start=1):
        row = df.iloc[doc_idx]
        grid_h = format_resolution_human(
            row.get("adjudicated_output_grid_meters"),
            row.get("adjudicated_spatial_grid_value"),
            row.get("adjudicated_spatial_grid_units"),
        )
        results.append({
            "rank": rank,
            "record_id": str(row["record_id"]),
            "title": str(row["title"]),
            "source_catalogue": str(row["source_catalogue"]),
            "provider_pool": str(row.get("provider_pool", "")),
            "copernicus_service": None if pd.isna(row.get("copernicus_service")) else str(row.get("copernicus_service")),
            "processing_level": None if pd.isna(row.get("processing_level")) else str(row.get("processing_level")),
            "geographic_coverage": None if pd.isna(row.get("geographic_coverage_class")) else str(row.get("geographic_coverage_class")),
            "grid": grid_h or None,
            "cadence": None if pd.isna(row.get("temporal_cadence_class")) else str(row.get("temporal_cadence_class")),
            "strict_eo_c1_eligible": bool(row.get("strict_eo_c1_eligible", False)),
            "score": round(float(score_val), 5),
            "dense_score": meta.get("dense_score"),
            "dense_rank": meta.get("dense_rank"),
            "bm25_rank": meta.get("bm25_rank"),
            "stage1_rank": meta.get("stage1_rank"),
            "rerank_score": meta.get("rerank_score"),
            "rerank_rank": meta.get("rerank_rank"),
            "collapsed_siblings": meta.get("sibling_count", 0),
            "access_url": None if pd.isna(row.get("access_url")) else str(row.get("access_url")),
        })

    payload = {
        "query": args.query,
        "index_prefix": args.index_prefix,
        "device": resolved_device,
        "corpus_size": len(df),
        "eligible_after_filter": n_eligible,
        "reranker": effective_rerank,
        "mode_used": "bm25" if use_bm25_only else effective_rerank,
        "null_safe_filter": eff_null_safe,
        "hard_limits": args.hard_limits,
        "unknown_metadata_counts": unknown_counts,
        "dim": args.dim,
        "load_seconds": round(t_load, 2),
        "stage1_ms": round(t_stage1_ms, 1),
        "rerank_ms": round(rerank_ms, 1),
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_human(payload)


if __name__ == "__main__":
    main()
