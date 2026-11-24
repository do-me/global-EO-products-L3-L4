# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "fastapi>=0.115.0",
#   "huggingface_hub>=0.30.0",
#   "numpy>=1.26.0",
#   "openpyxl>=3.1.5",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "safetensors>=0.4.0",
#   "tokenizers>=0.21.0",
#   "transformers>=5.18.0",
#   "uvicorn>=0.32.0",
#   "mlx>=0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-lm>=0.25.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-vlm>=0.8.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "torch>=2.4.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""Device-agnostic local server-client host & web UI for Hybrid Semantic EO Product Search.

Auto-selects the best compute backend (`--device auto`):
  - Apple Silicon (macOS arm64): MLX (`mlx-community/embeddinggemma-2-bf16`, `jinaai/jina-reranker-v3.5-mlx`, `Qwen/Qwen3-Reranker-0.6B`)
  - NVIDIA GPU (CUDA): PyTorch BF16/FP32 (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`)
  - CPU-only hosts: PyTorch CPU (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`)

Serves a minimal local web interface matching the static catalogue viewer style plus a
JSON/Export REST API (`/api/status`, `/api/search`, `/api/product/{record_id}`, `/api/export`)
usable from browsers, curl, or zero-cold-start CLI calls (`04_search_cli.py --server ...`).

Run:
  uv run 06_semantic_hybrid_search/04_search_server.py
  uv run 06_semantic_hybrid_search/04_search_server.py --port 8080 --open
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import platform
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
WEB_DIR = SCRIPT_DIR / "web"
CANONICAL_PARQUET = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DuckDBLexicalIndex,
    SearchFilters,
    apply_filters_mask,
    clean_description,
    collapse_product_families,
    compute_file_sha256,
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

VALID_MODES = ("jina-v3.5", "jina-v3.5-fast", "qwen3-0.6b", "hybrid", "dense", "bm25")
EXPORT_COLUMNS = [
    "rank",
    "record_id",
    "raw_id",
    "title",
    "producing_institution",
    "source_catalogue",
    "provider_pool",
    "copernicus_service",
    "processing_level",
    "uses_sentinel",
    "sentinel_missions",
    "grid",
    "grid_m",
    "cadence",
    "geographic_coverage",
    "t0",
    "t1",
    "operational_maturity_class",
    "strict_eo_c1_eligible",
    "score",
    "stage1_rank",
    "dense_score",
    "dense_rank",
    "bm25_rank",
    "rerank_score",
    "rerank_rank",
    "collapsed_siblings",
    "doi",
    "access_url",
]


def _clean_val(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean_val(x) for x in v]
    if pd.isna(v):
        return None
    return str(v)


def _extract_bbox(row: pd.Series) -> tuple[Optional[list[float]], bool]:
    w = _clean_val(row.get("bbox_west"))
    s = _clean_val(row.get("bbox_south"))
    e = _clean_val(row.get("bbox_east"))
    n = _clean_val(row.get("bbox_north"))
    if None in (w, s, e, n):
        return None, False
    w_f, s_f, e_f, n_f = float(w), float(s), float(e), float(n)
    if w_f > 180.0 and e_f > 180.0:
        w_f -= 360.0
        e_f -= 360.0
    elif e_f > 180.0 and (e_f - w_f) >= 359.5:
        w_f, e_f = -180.0, 180.0
    elif e_f > 180.0:
        e_f = ((e_f + 180.0) % 360.0) - 180.0
    s_f = max(-90.0, min(90.0, s_f))
    n_f = max(-90.0, min(90.0, n_f))
    antimeridian = bool(_clean_val(row.get("bbox_crosses_antimeridian"))) or (w_f > e_f)
    return [round(w_f, 4), round(s_f, 4), round(e_f, 4), round(n_f, 4)], antimeridian


def _make_snippet(desc: Any, max_len: int = 240) -> str:
    txt = clean_description(desc, max_chars=max_len + 80)
    if len(txt) <= max_len:
        return txt
    cut = txt[:max_len].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}..."


def _sanitize_excel_cell(val: Any) -> Any:
    if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + val
    return val


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["jina-v3.5", "jina-v3.5-fast", "qwen3-0.6b", "hybrid", "dense", "bm25"] = "jina-v3.5"
    pool: Optional[Literal["copernicus_services", "eu_non_copernicus", "non_eu"]] = None
    pools: Optional[list[Literal["copernicus_services", "eu_non_copernicus", "non_eu"]]] = None
    sources: Optional[list[str]] = None
    geo: Optional[Literal["global", "europe"]] = None
    exclude_local: bool = False
    max_grid_m: Optional[float] = Field(default=None, gt=0)
    eo_scope: Literal["all", "eo_and_hybrid", "strict_eo"] = "all"
    levels: Optional[list[Literal["L3", "L4"]]] = None
    min_end_year: Optional[int] = Field(default=None, ge=1950, le=2035)
    null_safe: bool = True
    hard_limits: bool = False
    collapse: bool = True
    rerank_pool: int = Field(default=50, ge=5, le=100)
    rerank_pure: bool = False
    top_k: int = Field(default=20, ge=1, le=100)
    dim: Literal[768, 512, 256, 128] = 768
    w_dense: float = Field(default=1.0, ge=0.0, le=5.0)
    w_bm25: float = Field(default=0.6, ge=0.0, le=5.0)


class ExportRequest(BaseModel):
    format: Literal["csv", "json", "xlsx", "parquet"]
    results: list[dict[str, Any]] = Field(..., min_length=1, max_length=500)
    metadata: Optional[dict[str, Any]] = None


class SearchEngine:
    """Device-agnostic local search engine managing Parquet metadata, DuckDB BM25, and MLX/CUDA/CPU models."""

    def __init__(
        self,
        requested_mode: str = "auto",
        requested_device: str = "auto",
        auto_build_bm25: bool = True,
        preload_reranker: str = "none",
        rebuild_index: bool = False,
        allow_download_index: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self.load_started = time.perf_counter()

        pq_path = ARTIFACTS_DIR / "full_catalogue.parquet"
        if not pq_path.exists():
            pq_path = CANONICAL_PARQUET
        if not pq_path.exists():
            raise FileNotFoundError(f"Canonical catalogue parquet not found at {pq_path}")

        self.df = pd.read_parquet(pq_path)
        self.corpus_size = len(self.df)
        self.rid_to_idx: dict[str, int] = {str(r): i for i, r in enumerate(self.df["record_id"].tolist())}
        self.unknown_metadata_counts = compute_unknown_metadata_counts(self.df, active_only=True)

        obs_re = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)
        active_mask = (~self.df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)) & (
            ~self.df["title"].fillna("").str.contains(obs_re, regex=True).to_numpy(dtype=bool, copy=True)
        )
        self.active_size = int(active_mask.sum())
        self.pool_counts = {
            str(k): int(v)
            for k, v in self.df.loc[active_mask, "provider_pool"].value_counts().items()
        }
        raw_src_counts = {
            str(k): int(v)
            for k, v in self.df.loc[active_mask, "source_catalogue"].value_counts().items()
            if pd.notna(k) and str(k).strip()
        }

        def _src_num_key(name: str) -> tuple[int, str]:
            m = re.match(r"^(\d+)", name)
            return (int(m.group(1)) if m else 9999, name)

        self.source_counts = {k: raw_src_counts[k] for k in sorted(raw_src_counts, key=_src_num_key)}

        bm25_path = ARTIFACTS_DIR / "full_bm25.duckdb"
        emb_path = ARTIFACTS_DIR / "full_embeddings_bf16.npz"
        want_hybrid = requested_mode in ("auto", "hybrid")

        if rebuild_index:
            cmd = [sys.executable, str(SCRIPT_DIR / "05_index_full_catalogue.py")]
            if requested_mode == "bm25":
                cmd.append("--bm25-only")
            subprocess.run(cmd, check=True)
        else:
            if allow_download_index:
                try:
                    ensure_search_indexes(
                        ARTIFACTS_DIR,
                        require_dense=want_hybrid,
                        allow_download=True,
                    )
                except Exception as exc:
                    if requested_mode == "hybrid":
                        raise
                    print(f"[server] Release index download skipped ({exc}); trying local artifacts.", file=sys.stderr, flush=True)
            else:
                ensure_search_indexes(
                    ARTIFACTS_DIR,
                    require_dense=False,
                    allow_download=False,
                )

        if not bm25_path.exists():
            if not auto_build_bm25:
                raise FileNotFoundError(
                    f"Missing {bm25_path}. Run `uv run 06_semantic_hybrid_search/05_index_full_catalogue.py --bm25-only`."
                )
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[server] Building {bm25_path.name} from {pq_path.name} (~35 s one-time setup)...", flush=True)
            builder = DuckDBLexicalIndex(str(bm25_path))
            dt_bm = builder.build_from_dataframe(self.df, catalogue_sha256=compute_file_sha256(pq_path))
            builder.con.close()
            print(f"[server] Built {bm25_path.name} in {dt_bm:.1f} s.", flush=True)

        self.lex = DuckDBLexicalIndex(str(bm25_path), read_only=True)

        self.hybrid_available = False
        self.resolved_device = "bm25"
        self.embed_model_id: Optional[str] = None
        self.fallback_reason: Optional[str] = None
        self.vecs: Optional[np.ndarray] = None
        self.enc: Any = None
        self._ev06_mod: Any = None
        self._reranker_name: Optional[str] = None
        self._reranker_obj: Any = None

        self.family_sigs: Optional[list[str]] = None

        if want_hybrid and emb_path.exists():
            try:
                enc, resolved_device, embed_model_id = load_dense_embedder(device=requested_device)
                z = np.load(emb_path, allow_pickle=True)
                verify_index_alignment(self.df, lex_index=self.lex, emb_npz=z, catalogue_path=pq_path)
                vecs = z["embeddings_f16"].astype(np.float32)
                vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
                self.vecs = vecs
                self.enc = enc
                self.resolved_device = resolved_device
                self.embed_model_id = embed_model_id
                if "family_sigs" in z:
                    self.family_sigs = [str(s) for s in z["family_sigs"].tolist()]
                self.hybrid_available = True
                if preload_reranker in ("jina-v3.5", "qwen3-0.6b"):
                    self._ensure_reranker(preload_reranker)
            except Exception as exc:
                if requested_mode == "hybrid" or requested_device != "auto":
                    raise
                verify_index_alignment(self.df, lex_index=self.lex, catalogue_path=pq_path)
                self.fallback_reason = f"Running in BM25 lexical mode (neural backend unavailable: {exc})."
        else:
            if requested_mode == "hybrid":
                raise RuntimeError(
                    "Hybrid mode requires artifacts/full_embeddings_bf16.npz. "
                    "Build it with: uv run 06_semantic_hybrid_search/05_index_full_catalogue.py"
                )
            verify_index_alignment(self.df, lex_index=self.lex, catalogue_path=pq_path)
            if requested_mode == "bm25":
                self.fallback_reason = "Started explicitly in BM25 lexical mode (--mode bm25)."
            else:
                self.fallback_reason = (
                    "Running in BM25 lexical mode because artifacts/full_embeddings_bf16.npz is not built yet. "
                    "Run `uv run 06_semantic_hybrid_search/05_index_full_catalogue.py` to build embeddings (MLX/CUDA/CPU)."
                )

        self.load_seconds = round(time.perf_counter() - self.load_started, 2)

    def _load_ev06(self) -> Any:
        if self._ev06_mod is None:
            spec = importlib.util.spec_from_file_location("ev06", SCRIPT_DIR / "06_eval_rerankers.py")
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            self._ev06_mod = mod
        return self._ev06_mod

    def _ensure_reranker(self, target: str) -> tuple[Any, float]:
        if self._reranker_name == target and self._reranker_obj is not None:
            return self._reranker_obj, 0.0

        ev06 = self._load_ev06()
        t0 = time.perf_counter()
        if self._reranker_obj is not None:
            self._reranker_obj = None
            self._reranker_name = None
            ev06.clear_device_cache(self.resolved_device)

        obj, _, _ = ev06.load_reranker(target, device=self.resolved_device, max_doc_length=420, verify_exactness=False)
        if self.resolved_device == "mlx":
            if target == "jina-v3.5":
                _ = obj.score_candidates("warmup", ["warmup 1", "warmup 2"])
            else:
                _ = obj.score_candidates("warmup", ["warmup 1"], batch_size=1, max_length=128)

        self._reranker_name = target
        self._reranker_obj = obj
        return obj, (time.perf_counter() - t0) * 1000.0

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "corpus_size": self.corpus_size,
            "active_size": self.active_size,
            "pool_counts": self.pool_counts,
            "source_counts": self.source_counts,
            "hybrid_available": self.hybrid_available,
            "device": self.resolved_device,
            "embed_model_id": self.embed_model_id,
            "default_mode": "jina-v3.5" if self.hybrid_available else "bm25",
            "available_modes": list(VALID_MODES) if self.hybrid_available else ["bm25"],
            "loaded_reranker": self._reranker_name,
            "fallback_reason": self.fallback_reason,
            "unknown_metadata_counts": self.unknown_metadata_counts,
            "startup_seconds": self.load_seconds,
            "platform": f"{sys.platform}/{platform.machine()}",
        }

    def get_product(self, record_id: str) -> dict[str, Any]:
        idx = self.rid_to_idx.get(record_id)
        if idx is None:
            raise KeyError(record_id)
        row = self.df.iloc[idx]
        rec = {str(col): _clean_val(row[col]) for col in self.df.columns}
        bbox, antimeridian = _extract_bbox(row)
        rec["bbox"] = bbox
        rec["antimeridian"] = antimeridian
        rec["grid_human"] = format_resolution_human(
            row.get("adjudicated_output_grid_meters"),
            row.get("adjudicated_spatial_grid_value"),
            row.get("adjudicated_spatial_grid_units"),
        ) or None
        return rec

    def search(self, req: SearchRequest) -> dict[str, Any]:
        query = req.query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        mode = req.mode
        if not self.hybrid_available:
            mode = "bm25"

        with self._lock:
            t_q0 = time.perf_counter()
            effective_pools = [p for p in (req.pools or []) if p] or ([req.pool] if req.pool else [])
            effective_sources = [s for s in (req.sources or []) if s]
            flt = SearchFilters(
                geo_scope=req.geo,
                exclude_local=req.exclude_local,
                max_grid_m=req.max_grid_m,
                processing_levels=req.levels,
                strict_eo_only=(req.eo_scope == "strict_eo"),
                ontology_mode=req.eo_scope,
                exclude_superseded=True,
                provider_pool=effective_pools[0] if len(effective_pools) == 1 else None,
                min_end_year=req.min_end_year,
                null_safe=False if req.hard_limits else req.null_safe,
                hard_limits=req.hard_limits,
            )
            mask = apply_filters_mask(self.df, flt)
            if len(effective_pools) > 1 and len(effective_pools) < 3:
                mask = mask & self.df["provider_pool"].isin(effective_pools).to_numpy(dtype=bool)
            if effective_sources:
                mask = mask & self.df["source_catalogue"].isin(effective_sources).to_numpy(dtype=bool)
            n_eligible = int(mask.sum())

            vecs_for_collapse: Optional[np.ndarray] = None
            if mode == "bm25":
                bm25_ranked = self.lex.search_bm25(query, top_k=300, eligible_mask=mask)
                fused = [
                    (doc_idx, float(sc), {"bm25_rank": rk, "bm25_score": round(float(sc), 5)})
                    for rk, (doc_idx, sc) in enumerate(bm25_ranked, start=1)
                ]
            elif mode == "dense":
                assert self.enc is not None and self.vecs is not None
                q_vec = self.enc.encode_queries([query], max_length=256)
                v_mat = matryoshka_truncate(self.vecs, req.dim) if req.dim < 768 else self.vecs
                if req.dim < 768:
                    q_vec = matryoshka_truncate(q_vec, req.dim)
                dense_ranked = deterministic_topk(v_mat @ q_vec[0], k=300, mask=mask)
                fused = [
                    (doc_idx, float(sc), {"dense_rank": rk, "dense_score": round(float(sc), 5)})
                    for rk, (doc_idx, sc) in enumerate(dense_ranked, start=1)
                ]
                vecs_for_collapse = self.vecs
            else:
                assert self.enc is not None and self.vecs is not None
                q_vec = self.enc.encode_queries([query], max_length=256)
                v_mat = matryoshka_truncate(self.vecs, req.dim) if req.dim < 768 else self.vecs
                if req.dim < 768:
                    q_vec = matryoshka_truncate(q_vec, req.dim)
                dense_ranked = deterministic_topk(v_mat @ q_vec[0], k=300, mask=mask)
                bm25_ranked = self.lex.search_bm25(query, top_k=300, eligible_mask=mask)
                fused = rrf_fuse(dense_ranked, bm25_ranked, rrf_k=60, w_dense=req.w_dense, w_bm25=req.w_bm25)
                vecs_for_collapse = self.vecs

            reranker_key: Optional[str] = None
            effective_pool = req.rerank_pool
            effective_pure = req.rerank_pure
            if mode == "jina-v3.5":
                reranker_key = "jina-v3.5"
            elif mode == "jina-v3.5-fast":
                reranker_key = "jina-v3.5"
                effective_pool = min(effective_pool, 25)
                effective_pure = True
            elif mode == "qwen3-0.6b":
                reranker_key = "qwen3-0.6b"

            stage1_k = max(req.top_k, effective_pool) if reranker_key is not None else req.top_k
            if req.collapse:
                stage1_final = collapse_product_families(
                    fused,
                    self.df,
                    top_k=stage1_k,
                    family_sigs=self.family_sigs,
                    vectors=vecs_for_collapse,
                    near_dup_cos=0.94 if vecs_for_collapse is not None else 1.0,
                    sensor_guard=True,
                    query_text=query,
                )
            else:
                stage1_final = fused[:stage1_k]

            stage1_ms = (time.perf_counter() - t_q0) * 1000.0
            rerank_ms = 0.0
            rerank_load_ms = 0.0

            if reranker_key is not None and stage1_final:
                rr_obj, rerank_load_ms = self._ensure_reranker(reranker_key)
                ev06 = self._load_ev06()
                pool_items = stage1_final[:effective_pool]
                docs = [ev06.format_reranker_doc(self.df.iloc[idx].to_dict(), max_chars=1200) for idx, _, _ in pool_items]
                t_r0 = time.perf_counter()
                if reranker_key == "jina-v3.5":
                    scores = rr_obj.score_candidates(query, docs)
                else:
                    scores = rr_obj.score_candidates(query, docs, batch_size=8, max_length=512)
                rerank_ms = (time.perf_counter() - t_r0) * 1000.0

                rerank_order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
                rerank_rank = {int(pos): rk + 1 for rk, pos in enumerate(rerank_order)}
                reordered = []
                for s1_pos, (doc_idx, _s1_sc, meta) in enumerate(pool_items):
                    m2 = dict(meta)
                    m2["rerank_score"] = round(float(scores[s1_pos]), 5)
                    m2["rerank_rank"] = rerank_rank[s1_pos]
                    m2["stage1_rank"] = s1_pos + 1
                    if effective_pure:
                        final_sc = float(scores[s1_pos])
                    else:
                        final_sc = 0.4 / (30.0 + s1_pos + 1) + 1.0 / (30.0 + rerank_rank[s1_pos])
                    reordered.append((doc_idx, final_sc, m2, s1_pos))
                reordered.sort(key=lambda x: (-x[1], x[3]))
                final_items = [(d, s, m) for d, s, m, _ in reordered[: req.top_k]]
                if req.top_k > len(final_items) and len(stage1_final) > effective_pool:
                    tail = stage1_final[effective_pool : req.top_k]
                    for pos, (d, s, m) in enumerate(tail, start=effective_pool + 1):
                        final_items.append((d, s, {**m, "stage1_rank": pos}))
            else:
                final_items = [
                    (d, s, {**m, "stage1_rank": pos})
                    for pos, (d, s, m) in enumerate(stage1_final[: req.top_k], start=1)
                ]

        results: list[dict[str, Any]] = []
        total_collapsed = 0
        for rank, (doc_idx, score_val, meta) in enumerate(final_items, start=1):
            row = self.df.iloc[doc_idx]
            grid_h = format_resolution_human(
                row.get("adjudicated_output_grid_meters"),
                row.get("adjudicated_spatial_grid_value"),
                row.get("adjudicated_spatial_grid_units"),
            )
            bbox, antimeridian = _extract_bbox(row)
            sibs_raw = meta.get("collapsed_siblings") or []
            sib_count = int(meta.get("sibling_count", len(sibs_raw)))
            total_collapsed += sib_count
            sib_items = []
            for s in sibs_raw[:15]:
                s_idx = int(s["doc_idx"])
                s_row = self.df.iloc[s_idx]
                sib_items.append({
                    "record_id": str(s_row["record_id"]),
                    "title": str(s_row["title"]),
                    "source_catalogue": str(s_row["source_catalogue"]),
                    "provider_pool": str(s_row.get("provider_pool", "")),
                })

            results.append({
                "rank": rank,
                "record_id": str(row["record_id"]),
                "raw_id": _clean_val(row.get("raw_id")),
                "title": str(row["title"]),
                "snippet": _make_snippet(row.get("description")),
                "producing_institution": _clean_val(row.get("producing_institution")),
                "source_catalogue": str(row["source_catalogue"]),
                "provider_pool": str(row.get("provider_pool", "")),
                "copernicus_service": _clean_val(row.get("copernicus_service")),
                "processing_level": _clean_val(row.get("processing_level")),
                "uses_sentinel": bool(row.get("uses_sentinel_data", False)) if not pd.isna(row.get("uses_sentinel_data")) else False,
                "sentinel_missions": _clean_val(row.get("sentinel_missions")),
                "sentinel_lineage_status": _clean_val(row.get("sentinel_lineage_status")),
                "geographic_coverage": _clean_val(row.get("geographic_coverage_class")),
                "grid": grid_h or None,
                "grid_m": _clean_val(row.get("adjudicated_output_grid_meters")),
                "cadence": _clean_val(row.get("temporal_cadence_class")),
                "t0": _clean_val(row.get("temporal_start_year")),
                "t1": _clean_val(row.get("temporal_end_year")),
                "operational_maturity_class": _clean_val(row.get("operational_maturity_class")),
                "strict_eo_c1_eligible": bool(row.get("strict_eo_c1_eligible", False)),
                "doi": _clean_val(row.get("doi")),
                "access_url": _clean_val(row.get("access_url")),
                "bbox": bbox,
                "antimeridian": antimeridian,
                "score": round(float(score_val), 5),
                "dense_score": meta.get("dense_score"),
                "dense_rank": meta.get("dense_rank"),
                "bm25_rank": meta.get("bm25_rank"),
                "stage1_rank": meta.get("stage1_rank"),
                "rerank_score": meta.get("rerank_score"),
                "rerank_rank": meta.get("rerank_rank"),
                "collapsed_siblings": sib_count,
                "collapsed_items": sib_items,
            })

        return {
            "query": query,
            "index_prefix": "full",
            "device": self.resolved_device,
            "corpus_size": self.corpus_size,
            "active_size": self.active_size,
            "eligible_after_filter": n_eligible,
            "mode_used": mode,
            "reranker": reranker_key or "none",
            "rerank_pool": effective_pool if reranker_key else 0,
            "rerank_pure": effective_pure if reranker_key else False,
            "null_safe_filter": False if req.hard_limits else req.null_safe,
            "hard_limits": req.hard_limits,
            "unknown_metadata_counts": self.unknown_metadata_counts,
            "collapsed": req.collapse,
            "total_collapsed_in_results": total_collapsed,
            "dim": req.dim,
            "load_seconds": 0.0,
            "stage1_ms": round(stage1_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "rerank_load_ms": round(rerank_load_ms, 1),
            "total_ms": round(stage1_ms + rerank_ms, 1),
            "results": results,
        }


def _export_rows_bytes(req: ExportRequest) -> tuple[bytes, str, str]:
    n = len(req.results)
    flat_rows = [{col: r.get(col) for col in EXPORT_COLUMNS} for r in req.results]
    df_out = pd.DataFrame(flat_rows, columns=EXPORT_COLUMNS)

    if req.format == "csv":
        df_safe = df_out.map(_sanitize_excel_cell)
        data = df_safe.to_csv(index=False).encode("utf-8")
        return data, "text/csv; charset=utf-8", f"eo_hybrid_search_top{n}.csv"

    if req.format == "json":
        payload = {**(req.metadata or {}), "results": req.results}
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        return data, "application/json; charset=utf-8", f"eo_hybrid_search_top{n}.json"

    if req.format == "xlsx":
        df_safe = df_out.map(_sanitize_excel_cell)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_safe.to_excel(writer, index=False, sheet_name="Search Results")
        return (
            buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"eo_hybrid_search_top{n}.xlsx",
        )

    buf = io.BytesIO()
    df_out.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue(), "application/vnd.apache.parquet", f"eo_hybrid_search_top{n}.parquet"


def build_app(engine: SearchEngine) -> FastAPI:
    app = FastAPI(title="Global L3/L4 EO Product Catalogue — Hybrid Search API", version="4.0")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/app.js")
    async def app_js():
        return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")

    @app.get("/api/status")
    async def api_status():
        return engine.status()

    @app.post("/api/search")
    async def api_search_post(req: SearchRequest):
        # Run on the main thread so MLX Metal streams & DuckDB keep exact thread affinity.
        try:
            return engine.search(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/search")
    async def api_search_get(
        q: str = Query(..., min_length=1, max_length=2000),
        mode: Literal["jina-v3.5", "jina-v3.5-fast", "qwen3-0.6b", "hybrid", "dense", "bm25"] = "jina-v3.5",
        pool: Optional[Literal["copernicus_services", "eu_non_copernicus", "non_eu"]] = None,
        pools: Optional[list[Literal["copernicus_services", "eu_non_copernicus", "non_eu"]]] = Query(default=None),
        sources: Optional[list[str]] = Query(default=None),
        geo: Optional[Literal["global", "europe"]] = None,
        exclude_local: bool = False,
        max_grid_m: Optional[float] = Query(default=None, gt=0),
        eo_scope: Literal["all", "eo_and_hybrid", "strict_eo"] = "all",
        level: Optional[list[Literal["L3", "L4"]]] = Query(default=None),
        min_end_year: Optional[int] = Query(default=None, ge=1950, le=2035),
        null_safe: bool = True,
        hard_limits: bool = False,
        collapse: bool = True,
        rerank_pool: int = Query(default=50, ge=5, le=100),
        rerank_pure: bool = False,
        top_k: int = Query(default=20, ge=1, le=100),
    ):
        req = SearchRequest(
            query=q,
            mode=mode,
            pool=pool,
            pools=pools,
            sources=sources,
            geo=geo,
            exclude_local=exclude_local,
            max_grid_m=max_grid_m,
            eo_scope=eo_scope,
            levels=level,
            min_end_year=min_end_year,
            null_safe=null_safe,
            hard_limits=hard_limits,
            collapse=collapse,
            rerank_pool=rerank_pool,
            rerank_pure=rerank_pure,
            top_k=top_k,
        )
        try:
            return engine.search(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export")
    async def api_export(req: ExportRequest):
        data, media_type, filename = _export_rows_bytes(req)
        return Response(
            content=data,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/product/{record_id:path}")
    async def api_product(record_id: str):
        try:
            return JSONResponse(engine.get_product(record_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Product not found: {record_id}") from exc

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start the local Hybrid Semantic EO Product Search server & web UI.")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080).")
    p.add_argument(
        "--device",
        choices=["auto", "mlx", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute backend (default: auto = MLX on Apple Silicon, CUDA if available, else CPU PyTorch).",
    )
    p.add_argument(
        "--mode",
        choices=["auto", "hybrid", "bm25"],
        default="auto",
        help="Search engine mode (default: auto = hybrid when embeddings exist, else BM25).",
    )
    p.add_argument(
        "--bm25-only",
        action="store_true",
        help="Shorthand for --mode bm25 (lexical-only mode requiring zero neural weights).",
    )
    p.add_argument(
        "--preload-reranker",
        choices=["none", "jina-v3.5", "qwen3-0.6b"],
        default="none",
        help="Optionally warm-load a Stage-2 reranker at startup instead of on the first reranked query.",
    )
    p.add_argument(
        "--no-auto-build-bm25",
        action="store_true",
        help="Fail instead of automatically building artifacts/full_bm25.duckdb if missing.",
    )
    p.add_argument("--rebuild-index", action="store_true", help="Rebuild local BM25/dense index before starting server.")
    p.add_argument("--no-download-index", action="store_true", help="Do not download missing/stale release indexes from GitHub Releases.")
    p.add_argument("--open", action="store_true", help="Open the UI in the default browser after startup.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    engine = SearchEngine(
        requested_mode="bm25" if args.bm25_only else args.mode,
        requested_device=args.device,
        auto_build_bm25=not args.no_auto_build_bm25,
        preload_reranker=args.preload_reranker,
        rebuild_index=args.rebuild_index,
        allow_download_index=not args.no_download_index,
    )
    app = build_app(engine)
    url = f"http://{args.host}:{args.port}"
    mode_label = (
        f"Hybrid ({engine.resolved_device.upper()}) + DuckDB BM25"
        if engine.hybrid_available
        else "DuckDB BM25 (lexical)"
    )
    print(
        f"[server] Ready in {engine.load_seconds:.2f} s | {engine.active_size:,} active / {engine.corpus_size:,} total | {mode_label}",
        flush=True,
    )
    if engine.fallback_reason:
        print(f"[server] Note: {engine.fallback_reason}", flush=True)
    print(f"[server] Web UI & API: {url}", flush=True)
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
