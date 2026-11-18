#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.0",
#   "pyarrow>=15.0.0",
# ]
# ///
"""
docs/build_viewer_data.py
=========================
Exports the canonical L3/L4 catalogue table (`03_harmonization/data/eo_l3_l4_canonical_products.parquet`)
into fast, lightweight static assets under `docs/data/` for `docs/index.html`:
  - `docs/data/index_meta.json`: Summary metadata, lookup enum dictionaries, and chunk manifest.
  - `docs/data/index_part_*.json`: Sharded compact arrays covering all 38,342 canonical products (~4-5 MB each) for parallel loading.
  - `docs/data/details/chunk_*.json`: 600-row detail shards (~0.5-1.5 MB each) fetched on demand when opening a product drawer.

Run via:
    uv run docs/build_viewer_data.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
CANON_PATH = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
OUT_DIR = DOCS_DIR / "data"
DETAILS_DIR = OUT_DIR / "details"

INDEX_CHUNK_SIZE = 10000
DETAIL_CHUNK_SIZE = 600

ENUM_COLS = [
    "source_catalogue",
    "producing_institution",
    "hosting_platform",
    "provider_pool",
    "provider_jurisdiction_tier",
    "copernicus_service",
    "processing_level",
    "product_ontology_class",
    "product_output_category",
    "sentinel_missions",
    "sentinel_lineage_status",
    "temporal_cadence_class",
    "operational_maturity_class",
    "geographic_coverage_class",
    "observation_domain_class",
    "observation_domain_mask",
    "license",
]

DETAIL_COLS = [
    "record_id",
    "description",
    "keywords",
    "platform_sensor_raw",
    "asset_type_raw",
    "canonical_product_key",
    "all_source_catalogues",
    "processing_level_raw",
    "processing_levels_available",
    "processing_level_provenance",
    "processing_level_conflict",
    "processing_level_conflict_reason",
    "product_level_subtype",
    "spatial_resolution_raw",
    "spatial_resolution_meters",
    "spatial_footprint_meters",
    "title_token_resolution_meters",
    "adjudicated_output_grid_meters",
    "adjudicated_spatial_grid_value",
    "adjudicated_spatial_grid_units",
    "adjudicated_spatial_grid_axes",
    "adjudicated_spatial_resolution_source",
    "adjudication_source_citation",
    "spatial_resolution_conflict",
    "spatial_resolution_conflict_reason",
    "prose_spatial_candidate_raw",
    "prose_spatial_role",
    "prose_spatial_evidence_sentence",
    "level_variant_specifications",
    "temporal_resolution_raw",
    "update_frequency_raw",
    "temporal_output_cadences",
    "temporal_cadence_by_epoch",
    "temporal_aggregation_window_raw",
    "temporal_start_raw",
    "temporal_end_raw",
    "temporal_span_provenance",
    "prose_cadence_role",
    "prose_cadence_evidence_sentence",
    "geographic_coverage_provenance",
    "doi_source",
    "doi_relation_type",
    "related_dois",
    "is_superseded_edition",
    "superseded_by_record",
]


def _clean_scalar(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        val = val.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
    return val


def _clean_text_snippet(val: Any, max_len: int) -> tuple[str, bool]:
    s = _clean_scalar(val)
    if not s:
        return "", False
    txt = str(s).strip()
    if len(txt) <= max_len:
        return txt, False
    cut = txt[:max_len].rsplit(" ", 1)[0]
    if len(cut) < max_len // 2:
        cut = txt[:max_len]
    return cut.rstrip(" ,;.") + "...", True


def main() -> None:
    df = pd.read_parquet(CANON_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.json"):
        stale.unlink()
    for stale in DETAILS_DIR.glob("*.json"):
        stale.unlink()

    enums: dict[str, list[str]] = {}
    enum_maps: dict[str, dict[str, int]] = {}
    for col in ENUM_COLS:
        vc = df[col].dropna().astype(str).value_counts()
        vals = sorted(vc.index.tolist()) if len(vc) <= 30 else vc.index.tolist()
        enums[col] = vals
        enum_maps[col] = {v: i for i, v in enumerate(vals)}

    def _e(col: str, val: Any) -> int:
        v = _clean_scalar(val)
        if v is None:
            return -1
        return enum_maps[col].get(str(v), -1)

    all_rows: list[list[Any]] = []
    detail_buckets: list[list[dict[str, Any]]] = []

    for idx, rec in enumerate(df.to_dict(orient="records")):
        rid = str(rec["record_id"])
        raw_id = str(_clean_scalar(rec.get("raw_id")) or "")
        src = str(rec["source_catalogue"])
        title = str(_clean_scalar(rec.get("title")) or "")
        desc_snip, desc_trunc = _clean_text_snippet(rec.get("description"), 180)
        kw_snip, _ = _clean_text_snippet(rec.get("keywords"), 90)

        grid_m = _clean_scalar(rec.get("adjudicated_output_grid_meters"))
        if grid_m is None:
            grid_m = _clean_scalar(rec.get("spatial_resolution_meters"))
        if grid_m is not None:
            grid_m = round(float(grid_m), 2)

        t0 = _clean_scalar(rec.get("temporal_start_year"))
        t1 = _clean_scalar(rec.get("temporal_end_year"))
        t0 = int(t0) if t0 is not None else None
        t1 = int(t1) if t1 is not None else None

        bw = _clean_scalar(rec.get("bbox_west"))
        bs = _clean_scalar(rec.get("bbox_south"))
        be = _clean_scalar(rec.get("bbox_east"))
        bn = _clean_scalar(rec.get("bbox_north"))
        bbox = (
            [round(float(bw), 2), round(float(bs), 2), round(float(be), 2), round(float(bn), 2)]
            if all(x is not None for x in (bw, bs, be, bn))
            else None
        )

        is_ong = _clean_scalar(rec.get("is_ongoing_archive"))
        is_ong_int = -1 if is_ong is None else (1 if bool(is_ong) else 0)

        # Store empty string for record_id if exactly f"{src}::{raw_id}" to eliminate 1.5 MB duplication
        rid_override = "" if rid == f"{src}::{raw_id}" else rid

        row = [
            idx,                                                            # 0: row_idx (detail lookup)
            rid_override,                                                   # 1: rid_override
            raw_id,                                                         # 2: raw_id
            title,                                                          # 3: title
            desc_snip,                                                      # 4: desc_snippet
            1 if desc_trunc else 0,                                         # 5: desc_truncated
            kw_snip,                                                        # 6: kw_snippet
            _e("source_catalogue", src),                                    # 7
            _e("producing_institution", rec.get("producing_institution")),  # 8
            _e("hosting_platform", rec.get("hosting_platform")),            # 9
            _e("provider_pool", rec.get("provider_pool")),                  # 10
            _e("provider_jurisdiction_tier", rec.get("provider_jurisdiction_tier")),  # 11
            _e("copernicus_service", rec.get("copernicus_service")),        # 12
            _e("processing_level", rec.get("processing_level")),            # 13
            _e("product_ontology_class", rec.get("product_ontology_class")),# 14
            _e("product_output_category", rec.get("product_output_category")), # 15
            1 if bool(_clean_scalar(rec.get("strict_eo_c1_eligible"))) else 0,  # 16
            1 if bool(_clean_scalar(rec.get("uses_sentinel_data"))) else 0,     # 17
            _e("sentinel_missions", rec.get("sentinel_missions")),          # 18
            _e("sentinel_lineage_status", rec.get("sentinel_lineage_status")), # 19
            grid_m,                                                         # 20: grid_m
            _e("temporal_cadence_class", rec.get("temporal_cadence_class")),# 21
            t0,                                                             # 22: t0
            t1,                                                             # 23: t1
            is_ong_int,                                                     # 24: is_ongoing
            _e("operational_maturity_class", rec.get("operational_maturity_class")), # 25
            _e("geographic_coverage_class", rec.get("geographic_coverage_class")),   # 26
            _e("observation_domain_class", rec.get("observation_domain_class")),     # 27
            _e("observation_domain_mask", rec.get("observation_domain_mask")),       # 28
            _clean_scalar(rec.get("doi")),                                  # 29: doi
            _e("license", rec.get("license")),                              # 30
            _clean_scalar(rec.get("access_url")),                           # 31: access_url
            bbox,                                                           # 32: bbox
            1 if bool(_clean_scalar(rec.get("bbox_crosses_antimeridian"))) else 0,   # 33
            int(_clean_scalar(rec.get("cross_catalogue_record_count")) or 1),        # 34
        ]
        all_rows.append(row)

        bucket_idx = idx // DETAIL_CHUNK_SIZE
        while len(detail_buckets) <= bucket_idx:
            detail_buckets.append([])
        det: dict[str, Any] = {}
        for col in DETAIL_COLS:
            val = _clean_scalar(rec.get(col))
            if val is not None and val is not False:
                det[col] = val
        detail_buckets[bucket_idx].append(det)

    fields = [
        "idx", "rid_override", "raw_id", "title", "desc_snippet", "desc_truncated", "kw_snippet",
        "source_catalogue", "producing_institution", "hosting_platform",
        "provider_pool", "provider_jurisdiction_tier", "copernicus_service",
        "processing_level", "product_ontology_class", "product_output_category",
        "strict_eo", "uses_sentinel", "sentinel_missions", "sentinel_lineage_status",
        "grid_m", "temporal_cadence_class", "t0", "t1", "is_ongoing",
        "operational_maturity_class", "geographic_coverage_class",
        "observation_domain_class", "observation_domain_mask",
        "doi", "license", "access_url", "bbox", "antimeridian", "cluster_count",
    ]

    index_parts: list[str] = []
    max_part_mb = 0.0
    for p_idx, start in enumerate(range(0, len(all_rows), INDEX_CHUNK_SIZE)):
        part_name = f"index_part_{p_idx:02d}.json"
        part_path = OUT_DIR / part_name
        part_rows = all_rows[start:start + INDEX_CHUNK_SIZE]
        part_path.write_text(json.dumps(part_rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        mb = part_path.stat().st_size / (1024 * 1024)
        max_part_mb = max(max_part_mb, mb)
        index_parts.append(part_name)

    meta_payload = {
        "meta": {
            "catalogue_title": "Global Harmonized L3/L4 Earth Observation Product Catalogue",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_canonical": len(df),
            "total_harmonized": 38974,
            "strict_eo_count": int(df["strict_eo_c1_eligible"].sum()),
            "sentinel_count": int(df["uses_sentinel_data"].sum()),
            "doi_count": int(df["doi"].notna().sum()),
            "sources_count": int(df["source_catalogue"].nunique()),
            "detail_chunk_size": DETAIL_CHUNK_SIZE,
            "index_parts": index_parts,
        },
        "fields": fields,
        "enums": enums,
    }
    meta_path = OUT_DIR / "index_meta.json"
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    max_det_mb = 0.0
    for b_idx, bucket in enumerate(detail_buckets):
        p = DETAILS_DIR / f"chunk_{b_idx:03d}.json"
        p.write_text(json.dumps(bucket, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        max_det_mb = max(max_det_mb, p.stat().st_size / (1024 * 1024))

    print(
        f"[build_viewer_data] Built {len(index_parts)} index parts (max {max_part_mb:.2f} MB) + "
        f"{len(detail_buckets)} detail chunks (max {max_det_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()
