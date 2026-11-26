#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
# ]
# ///
"""
06_semantic_hybrid_search/export_browser_index.py
=================================================
Exports compact, CORS-free, static-hostable browser search artifacts from the
authoritative catalogue Parquet + `v4.3.0-indexes` (`full_bm25.duckdb` and
`full_embeddings_bf16.npz`, auto-downloaded from GitHub Releases if absent):

  1. `bm25_index.bin.gz`   (~4 MB)  Lossless 3-field DuckDB FTS inverted index
                                    (delta-varint doc IDs + field bitmasks + exact
                                    stopwords + field lengths + avg_field_lens).
  2. `emb_512d_i8_p0.bin.gz` / `p1.bin.gz` (~7.4 MB each)
                                    512-d Matryoshka L2-normalized per-row
                                    symmetric int8 document embeddings.
  3. `search_meta.json.gz` (~1.2 MB) Precomputed exact filter flags, family
                                    signatures, sensor/variable bitmasks, and
                                    active pool/source counts for 100% parity
                                    with Python `eo_search_lib.py`.
  4. `search_manifest.json` (~2 KB) Alignment hashes, file sizes, and metadata.

Run locally:
    uv run 06_semantic_hybrid_search/export_browser_index.py --out-dir dist/data/search
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
CANON_PATH = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_INDEX_RELEASE_TAG,
    DuckDBLexicalIndex,
    _DELIVERED_VAR_PATTERNS,
    _OBSOLETE_TITLE_RE,
    _SENSOR_CONFLICT_PATTERNS,
    _VERIFIED_OUTPUT_GRID_SOURCES,
    _extract_product_layer_and_var,
    _extract_sensor_set,
    compute_catalogue_content_sha256,
    compute_file_sha256,
    compute_generic_family_signature,
    compute_record_ids_sha256,
    compute_unknown_metadata_counts,
    ensure_search_indexes,
    matryoshka_truncate,
    verify_index_alignment,
)

BM25_MAGIC = 0x424D3235  # "BM25"
EMB8_MAGIC = 0x454D4238  # "EMB8"

_EO_AND_HYBRID_ONTOLOGIES = {
    "eo_thematic_or_biophysical",
    "eo_reanalysis_or_model",
    "eo_constrained_hybrid",
    "eo_ard_surface_reflectance_mosaic",
}

_SENSOR_BIT = {name: (1 << i) for i, (name, _) in enumerate(_SENSOR_CONFLICT_PATTERNS)}
_VTAG_BIT = {name: (1 << i) for i, (name, _) in enumerate(_DELIVERED_VAR_PATTERNS)}
assert len(_SENSOR_BIT) <= 30, "Sensor bitmask must fit in signed 32-bit JS int"
assert len(_VTAG_BIT) <= 30, "Variable bitmask must fit in signed 32-bit JS int"


def _write_varint(buf: bytearray, val: int) -> None:
    while val >= 0x80:
        buf.append((val & 0x7F) | 0x80)
        val >>= 7
    buf.append(val)


def _write_gz_deterministic(path: Path, raw: bytes, compresslevel: int = 6) -> None:
    """Write deterministic gzip (mtime=0) so CI rebuilds are reproducible."""
    with path.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, compresslevel=compresslevel, mtime=0) as gz:
            gz.write(raw)


def export_bm25_binary(bm25_db_path: Path, out_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    con = duckdb.connect(str(bm25_db_path), read_only=True)
    try:
        stats_row = con.execute("SELECT num_docs, avg_field_lens FROM fts_main_eo_docs.stats").fetchone()
        assert stats_row is not None
        n_docs = int(stats_row[0])
        avg_field_lens = [float(x) for x in stats_row[1]]
        assert len(avg_field_lens) == 3

        # Verify docid == doc_idx alignment
        misaligned = con.execute(
            "SELECT count(*) FROM fts_main_eo_docs.docs d JOIN eo_docs e ON d.name = e.record_id WHERE d.docid != e.doc_idx"
        ).fetchone()[0]
        if misaligned != 0:
            raise RuntimeError(f"DuckDB FTS docid != doc_idx for {misaligned} rows")

        # Field lengths per doc (n_docs x 3 x uint16)
        doc_rows = con.execute(
            "SELECT docid, field_lens[1], field_lens[2], field_lens[3] FROM fts_main_eo_docs.docs ORDER BY docid ASC"
        ).fetchall()
        assert len(doc_rows) == n_docs
        field_lens = np.zeros((n_docs, 3), dtype=np.uint16)
        for docid, l0, l1, l2 in doc_rows:
            i = int(docid)
            field_lens[i, 0] = int(l0 or 0)
            field_lens[i, 1] = int(l1 or 0)
            field_lens[i, 2] = int(l2 or 0)

        # Exact DuckDB stopwords
        stopwords = [
            str(r[0]) for r in con.execute("SELECT sw FROM fts_main_eo_docs.stopwords ORDER BY sw ASC").fetchall()
        ]
        sw_bytes = "\n".join(stopwords).encode("utf-8")

        # Vocabulary ordered alphabetically by term -> remap old termid to new contiguous id
        dict_rows = con.execute(
            "SELECT termid, term, df FROM fts_main_eo_docs.dict ORDER BY term ASC"
        ).fetchall()
        n_terms = len(dict_rows)
        old_to_new: dict[int, int] = {}
        terms_list: list[str] = []
        term_df = np.zeros(n_terms, dtype=np.uint32)
        for new_id, (old_id, term, df_val) in enumerate(dict_rows):
            old_to_new[int(old_id)] = new_id
            terms_list.append(str(term))
            term_df[new_id] = int(df_val)
        vocab_bytes = "\n".join(terms_list).encode("utf-8")

        # Aggregate (termid, docid, tf0, tf1, tf2) in DuckDB directly — super fast!
        postings_rows = con.execute(
            """
            SELECT d.term,
                   t.docid,
                   SUM(CASE WHEN t.fieldid = 0 THEN 1 ELSE 0 END) AS tf0,
                   SUM(CASE WHEN t.fieldid = 1 THEN 1 ELSE 0 END) AS tf1,
                   SUM(CASE WHEN t.fieldid = 2 THEN 1 ELSE 0 END) AS tf2
            FROM fts_main_eo_docs.terms t
            JOIN fts_main_eo_docs.dict d ON d.termid = t.termid
            GROUP BY d.term, t.docid
            ORDER BY d.term ASC, t.docid ASC
            """
        ).fetchall()
    finally:
        con.close()

    term_to_id = {t: i for i, t in enumerate(terms_list)}
    term_offsets = np.zeros(n_terms + 1, dtype=np.uint32)
    postings_buf = bytearray()

    cur_tid = -1
    prev_doc = 0
    for term_s, docid, tf0, tf1, tf2 in postings_rows:
        tid = term_to_id[str(term_s)]
        if tid != cur_tid:
            for fill_t in range(cur_tid + 1, tid + 1):
                term_offsets[fill_t] = len(postings_buf)
            cur_tid = tid
            prev_doc = 0
        did = int(docid)
        delta = did - prev_doc
        prev_doc = did
        _write_varint(postings_buf, delta)

        t0_i = int(tf0)
        t1_i = int(tf1)
        t2_i = int(tf2)
        flags = 0
        if t0_i > 0:
            flags |= 0x01
            if t0_i > 1:
                flags |= 0x08
        if t1_i > 0:
            flags |= 0x02
            if t1_i > 1:
                flags |= 0x10
        if t2_i > 0:
            flags |= 0x04
            if t2_i > 1:
                flags |= 0x20
        postings_buf.append(flags)
        if t0_i > 1:
            _write_varint(postings_buf, t0_i)
        if t1_i > 1:
            _write_varint(postings_buf, t1_i)
        if t2_i > 1:
            _write_varint(postings_buf, t2_i)

    for fill_t in range(cur_tid + 1, n_terms + 1):
        term_offsets[fill_t] = len(postings_buf)

    # Assemble binary payload
    out = bytearray()
    out.extend(
        struct.pack(
            "<IIII3d",
            BM25_MAGIC,
            1,
            n_docs,
            n_terms,
            avg_field_lens[0],
            avg_field_lens[1],
            avg_field_lens[2],
        )
    )
    out.extend(field_lens.astype("<u2").tobytes())
    out.extend(struct.pack("<I", len(sw_bytes)))
    out.extend(sw_bytes)
    out.extend(struct.pack("<I", len(vocab_bytes)))
    out.extend(vocab_bytes)
    out.extend(term_df.astype("<u4").tobytes())
    out.extend(term_offsets.astype("<u4").tobytes())
    out.extend(postings_buf)

    _write_gz_deterministic(out_path, bytes(out))
    elapsed = time.perf_counter() - t0
    gz_bytes = out_path.stat().st_size
    print(
        f"[export-browser] BM25 index: {n_terms:,} terms, {len(postings_rows):,} postings -> "
        f"{len(out)/1e6:.2f} MB raw / {gz_bytes/1e6:.2f} MB gz ({elapsed:.2f}s)",
        flush=True,
    )
    return {
        "file": out_path.name,
        "raw_bytes": len(out),
        "gz_bytes": gz_bytes,
        "sha256": compute_file_sha256(out_path),
        "n_terms": n_terms,
        "n_postings": len(postings_rows),
        "n_stopwords": len(stopwords),
    }


def export_embeddings_int8(
    emb_npz_path: Path,
    out_dir: Path,
    dim: int = 512,
    chunk_rows: int = 20000,
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    npz = np.load(emb_npz_path, allow_pickle=True)
    if "embeddings_f16" in npz:
        vecs_f32 = npz["embeddings_f16"].astype(np.float32)
    elif "embeddings_bf16_u16" in npz:
        vecs_u16 = npz["embeddings_bf16_u16"]
        vecs_f32 = (vecs_u16.astype(np.uint32) << 16).view(np.float32)
    else:
        raise KeyError(f"Expected embeddings_f16 or embeddings_bf16_u16 in {emb_npz_path}, found {npz.files}")
    vecs_mrl = matryoshka_truncate(vecs_f32, dim).astype(np.float32, copy=False)
    n_docs, actual_dim = vecs_mrl.shape
    assert actual_dim == dim

    # Symmetric per-row int8 quantization
    max_abs = np.max(np.abs(vecs_mrl), axis=1)
    scales = np.maximum(max_abs, 1e-12).astype(np.float32) / np.float32(127.0)
    quant = np.clip(np.rint(vecs_mrl / scales[:, None]), -127.0, 127.0).astype(np.int8)

    parts_meta: list[dict[str, Any]] = []
    for p_idx, start in enumerate(range(0, n_docs, chunk_rows)):
        end = min(n_docs, start + chunk_rows)
        n_rows = end - start
        raw = bytearray()
        raw.extend(struct.pack("<IIIII", EMB8_MAGIC, 1, start, n_rows, dim))
        raw.extend(scales[start:end].astype("<f4").tobytes())
        raw.extend(quant[start:end].tobytes())
        fname = f"emb_{dim}d_i8_p{p_idx}.bin.gz"
        out_path = out_dir / fname
        _write_gz_deterministic(out_path, bytes(raw))
        gz_bytes = out_path.stat().st_size
        parts_meta.append(
            {
                "file": fname,
                "start_row": start,
                "n_rows": n_rows,
                "dim": dim,
                "raw_bytes": len(raw),
                "gz_bytes": gz_bytes,
                "sha256": compute_file_sha256(out_path),
            }
        )
    elapsed = time.perf_counter() - t0
    tot_gz = sum(p["gz_bytes"] for p in parts_meta)
    tot_raw = sum(p["raw_bytes"] for p in parts_meta)
    print(
        f"[export-browser] Dense {dim}-d int8 ({len(parts_meta)} parts): "
        f"{tot_raw/1e6:.2f} MB raw / {tot_gz/1e6:.2f} MB gz ({elapsed:.2f}s)",
        flush=True,
    )
    return parts_meta


def export_search_meta(df: pd.DataFrame, out_path: Path) -> tuple[dict[str, Any], int]:
    t0 = time.perf_counter()
    n_docs = len(df)
    rids = [str(r) for r in df["record_id"]]
    titles = df["title"].fillna("").astype(str).tolist()
    raw_ids = df["raw_id"].fillna("").astype(str).tolist() if "raw_id" in df.columns else [""] * n_docs
    ckeys = df["canonical_product_key"].fillna("").astype(str).tolist() if "canonical_product_key" in df.columns else [""] * n_docs
    cads = df["temporal_cadence_class"].fillna("").astype(str).tolist() if "temporal_cadence_class" in df.columns else [""] * n_docs
    geos = df["geographic_coverage_class"].fillna("").astype(str).tolist() if "geographic_coverage_class" in df.columns else [""] * n_docs
    psens = df["platform_sensor_raw"].fillna("").astype(str).tolist() if "platform_sensor_raw" in df.columns else [""] * n_docs
    onts = df["product_ontology_class"].fillna("").astype(str).tolist() if "product_ontology_class" in df.columns else [""] * n_docs
    grid_srcs = (
        df["adjudicated_spatial_resolution_source"].fillna("").astype(str).tolist()
        if "adjudicated_spatial_resolution_source" in df.columns
        else [""] * n_docs
    )
    grid_vals_raw = (
        pd.to_numeric(df["adjudicated_output_grid_meters"], errors="coerce").to_numpy(dtype=np.float64)
        if "adjudicated_output_grid_meters" in df.columns
        else np.full(n_docs, np.nan)
    )
    sup_col = (
        df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool)
        if "is_superseded_edition" in df.columns
        else np.zeros(n_docs, dtype=bool)
    )
    obs_col = df["title"].fillna("").str.contains(_OBSOLETE_TITLE_RE, regex=True).to_numpy(dtype=bool)
    conflict_col = (
        df["spatial_resolution_conflict"].fillna(False).to_numpy(dtype=bool)
        if "spatial_resolution_conflict" in df.columns
        else np.zeros(n_docs, dtype=bool)
    )
    ongoing_col = (
        df["is_ongoing_archive"].fillna(False).to_numpy(dtype=bool)
        if "is_ongoing_archive" in df.columns
        else np.zeros(n_docs, dtype=bool)
    )
    strict_col = (
        df["strict_eo_c1_eligible"].fillna(False).to_numpy(dtype=bool)
        if "strict_eo_c1_eligible" in df.columns
        else np.zeros(n_docs, dtype=bool)
    )
    pools = df["provider_pool"].fillna("").astype(str).tolist()
    sources = df["source_catalogue"].fillna("").astype(str).tolist()

    flags: list[int] = []
    adj_grid_m: list[float | None] = []
    sig_map: dict[str, int] = {}
    sig_table: list[str] = []
    sig_ids: list[int] = []
    role_map: dict[str, int] = {}
    role_table: list[str] = []
    role_ids: list[int] = []
    cad_map: dict[str, int] = {}
    cad_table: list[str] = []
    cad_ids: list[int] = []
    vtag_masks: list[int] = []
    sensor_masks: list[int] = []

    active_pool_counts: dict[str, int] = {}
    active_source_counts: dict[str, int] = {}
    active_docs = 0

    for i in range(n_docs):
        is_sup = bool(sup_col[i] or obs_col[i])
        if not is_sup:
            active_docs += 1
            if pools[i]:
                active_pool_counts[pools[i]] = active_pool_counts.get(pools[i], 0) + 1
            if sources[i]:
                active_source_counts[sources[i]] = active_source_counts.get(sources[i], 0) + 1

        grid_ver = grid_srcs[i] in _VERIFIED_OUTPUT_GRID_SOURCES
        sp_conf = bool(conflict_col[i])
        is_ong = bool(ongoing_col[i])
        is_strict = bool(strict_col[i])
        is_eohyb = onts[i] in _EO_AND_HYBRID_ONTOLOGIES

        f_val = (
            (1 if is_sup else 0)
            | (2 if grid_ver else 0)
            | (4 if sp_conf else 0)
            | (8 if is_ong else 0)
            | (16 if is_strict else 0)
            | (32 if is_eohyb else 0)
        )
        flags.append(f_val)

        gv = float(grid_vals_raw[i])
        if math.isfinite(gv):
            adj_grid_m.append(round(gv, 4))
            gv_opt: float | None = gv
        else:
            adj_grid_m.append(None)
            gv_opt = None

        role, vtags, eff_cad = _extract_product_layer_and_var(titles[i], raw_ids[i], cads[i])
        sig = compute_generic_family_signature(titles[i], ckeys[i], eff_cad, gv_opt, geos[i])

        sid = sig_map.get(sig)
        if sid is None:
            sid = len(sig_table)
            sig_map[sig] = sid
            sig_table.append(sig)
        sig_ids.append(sid)

        rid_i = role_map.get(role)
        if rid_i is None:
            rid_i = len(role_table)
            role_map[role] = rid_i
            role_table.append(role)
        role_ids.append(rid_i)

        cid = cad_map.get(eff_cad)
        if cid is None:
            cid = len(cad_table)
            cad_map[eff_cad] = cid
            cad_table.append(eff_cad)
        cad_ids.append(cid)

        vm = 0
        for vt in vtags:
            vm |= _VTAG_BIT[vt]
        vtag_masks.append(vm)

        sm = 0
        for sn in _extract_sensor_set(titles[i], raw_ids[i], psens[i]):
            sm |= _SENSOR_BIT[sn]
        sensor_masks.append(sm)

    unknown_counts = compute_unknown_metadata_counts(df, active_only=True)
    payload = {
        "n_docs": n_docs,
        "active_docs": active_docs,
        "record_ids_sha256": compute_record_ids_sha256(rids),
        "unknown_metadata_counts": unknown_counts,
        "pool_counts": active_pool_counts,
        "source_counts": active_source_counts,
        "flags": flags,
        "adj_grid_m": adj_grid_m,
        "sig_table": sig_table,
        "sig_ids": sig_ids,
        "role_table": role_table,
        "role_ids": role_ids,
        "cad_table": cad_table,
        "cad_ids": cad_ids,
        "vtag_masks": vtag_masks,
        "sensor_masks": sensor_masks,
    }
    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    _write_gz_deterministic(out_path, raw_bytes)
    gz_bytes = out_path.stat().st_size
    elapsed = time.perf_counter() - t0
    print(
        f"[export-browser] Search metadata: {n_docs:,} docs ({active_docs:,} active), "
        f"{len(sig_table):,} family sigs -> {len(raw_bytes)/1e6:.2f} MB raw / {gz_bytes/1e6:.2f} MB gz ({elapsed:.2f}s)",
        flush=True,
    )
    return (
        {
            "file": out_path.name,
            "raw_bytes": len(raw_bytes),
            "gz_bytes": gz_bytes,
            "sha256": compute_file_sha256(out_path),
            "n_family_sigs": len(sig_table),
            "unknown_metadata_counts": unknown_counts,
            "pool_counts": active_pool_counts,
            "source_counts": active_source_counts,
        },
        active_docs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact browser search artifacts for static GitHub Pages.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "web_search" / "public" / "data" / "search",
        help="Destination directory for browser search index files.",
    )
    parser.add_argument("--dim", type=int, default=512, choices=(128, 256, 512, 768), help="Matryoshka embedding dim.")
    parser.add_argument("--chunk-rows", type=int, default=20000, help="Rows per int8 embedding shard.")
    parser.add_argument("--release-tag", default=DEFAULT_INDEX_RELEASE_TAG, help="GitHub Release tag for search indexes.")
    parser.add_argument("--no-download", action="store_true", help="Fail if local indexes are missing instead of downloading.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    assets = ensure_search_indexes(
        ARTIFACTS_DIR,
        require_dense=True,
        allow_download=not args.no_download,
        release_tag=args.release_tag,
    )

    df = pd.read_parquet(CANON_PATH)
    lex = DuckDBLexicalIndex(str(assets["bm25"]), read_only=True)
    npz = np.load(assets["embeddings"], allow_pickle=True)
    try:
        align_info = verify_index_alignment(df, lex_index=lex, emb_npz=npz, catalogue_path=CANON_PATH)
        lex_meta = lex.get_meta()
    finally:
        lex.con.close()
        npz.close()

    bm25_info = export_bm25_binary(assets["bm25"], args.out_dir / "bm25_index.bin.gz")
    emb_parts = export_embeddings_int8(assets["embeddings"], args.out_dir, dim=args.dim, chunk_rows=args.chunk_rows)
    meta_info, active_docs = export_search_meta(df, args.out_dir / "search_meta.json.gz")

    manifest = {
        "schema_version": "browser-search-v1",
        "release_tag": args.release_tag,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_docs": len(df),
        "active_docs": active_docs,
        "emb_dim": args.dim,
        "models": {
            "embedder": "onnx-community/embeddinggemma-2-ONNX",
            "embedder_dtype": "q4",
            "reranker": "onnx-community/Qwen3-Reranker-0.6B-ONNX",
            "reranker_dtype": "q4",
        },
        "alignment": {
            "record_ids_sha256": align_info["record_ids_sha256"],
            "catalogue_sha256": align_info["catalogue_sha256"],
            "catalogue_content_sha256": compute_catalogue_content_sha256(df),
            "bm25_content_sha256": lex_meta.get("bm25_content_sha256", ""),
        },
        "bm25": bm25_info,
        "embeddings": emb_parts,
        "search_meta": {
            "file": meta_info["file"],
            "raw_bytes": meta_info["raw_bytes"],
            "gz_bytes": meta_info["gz_bytes"],
            "sha256": meta_info["sha256"],
            "n_family_sigs": meta_info["n_family_sigs"],
        },
        "pool_counts": meta_info["pool_counts"],
        "source_counts": meta_info["source_counts"],
        "unknown_metadata_counts": meta_info["unknown_metadata_counts"],
    }
    manifest_path = args.out_dir / "search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total_gz = bm25_info["gz_bytes"] + meta_info["gz_bytes"] + sum(p["gz_bytes"] for p in emb_parts)
    print(
        f"[export-browser] DONE -> {args.out_dir} "
        f"(total compressed transfer: BM25+meta {(bm25_info['gz_bytes']+meta_info['gz_bytes'])/1e6:.2f} MB, "
        f"embeddings {sum(p['gz_bytes'] for p in emb_parts)/1e6:.2f} MB, all {total_gz/1e6:.2f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
