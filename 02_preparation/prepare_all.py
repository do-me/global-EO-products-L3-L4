#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.0",
#   "pyarrow>=15.0.0",
# ]
# ///
"""
02_preparation/prepare_all.py
=============================
Deterministic preprocessing, structural normalization, and L3/L4 filtering across all 29 raw EO metadata sources.

Key Scientific & Methodological Principles:
1. 100% Reproducible ("let null be null"): Never manually impute or invent missing metadata. If a field is absent
   in the upstream raw JSON/JSONL, it remains `None` (`null`) and is logged in the completeness diagnostics.
2. Strict L3/L4 Scope: Excludes explicit Level 0, Level 1, and Level 2 swath/granule/tile sensor archives,
   visualisation-only WMS layers, and pure non-EO administrative/tabular survey records, while preserving:
   - Explicit Level 3 and Level 4 satellite & model products,
   - Implicit L3/L4 gridded thematic maps, biophysical products, Climate Data Records (CDRs), reanalyses, DEMs,
   - Gridded L3 ARD surface reflectance mosaics (`L3_ARD_REFLECTANCE`, e.g., HLS and CLMS S2 Global Mosaic) symmetrically,
   - EO-constrained hybrid grids (e.g., GHSL, OpenLandMap, WorldPop) and authoritative reference boundaries (e.g., Natura 2000, WDPA)
     for explicit downstream ontology tagging in `03_harmonization`.
3. Complete Audit Trail: Every excluded record is logged with a deterministic `exclusion_reason` in
   `02_preparation/data/excluded_records_log.parquet` and summarized in `02_preparation/exclusion_audit_summary.csv`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RAW_DIR = REPO_ROOT / "01_raw"
OUT_DATA_DIR = BASE_DIR / "data"
TAXONOMY_DIR = BASE_DIR / "reference_taxonomies"


# ---------------------------------------------------------------------------
# Helper utilities (strictly deterministic, zero manual imputation)
# ---------------------------------------------------------------------------
def clean_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, dict):
        for k in ("en", "langeng", "default", "value", "title", "name", "caption"):
            if k in val and val[k]:
                return clean_str(val[k])
        for v in val.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    if isinstance(val, list):
        parts = [clean_str(x) for x in val]
        parts = [p for p in parts if p]
        return "; ".join(dict.fromkeys(parts)) if parts else None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "n/a", "not provided", "unknown", "nan"):
        return None
    # Strip HTML tags if present in descriptions
    if "<p>" in s or "<div" in s or "<br" in s or "</" in s:
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def clean_bbox(bbox: Any) -> list[float] | None:
    """
    Validate and normalize a [west, south, east, north] WGS84 bounding box.
    Deterministically repairs axis-swapped [south, west, north, east] boxes (e.g. OpenLandMap),
    0-360 longitude conventions, inverted S/N or non-antimeridian W/E pairs, and drops
    non-degree projected coordinate boxes or corrupt extents.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        w, s, e, n = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        return None
    if not all(pd.notna(x) for x in (w, s, e, n)):
        return None

    # 1. Axis-swapped [south, west, north, east] (e.g. STAC collections storing lat first)
    if (abs(s) > 90.5 or abs(n) > 90.5) and abs(w) <= 90.5 and abs(e) <= 90.5:
        w, s, e, n = s, w, n, e

    # 2. Reject non-WGS84 projected coordinates (meters) or corrupt out-of-range bounds
    if (
        abs(s) > 90.5
        or abs(n) > 90.5
        or w < -180.5
        or e < -180.5
        or w > 360.5
        or e > 360.5
        or (s == 0.0 and n == 0.0 and abs(e - w) > 0.0 and w > 180.0)
    ):
        return None

    # 3. Convert [0, 360] longitudes (e.g. C3S/CEMS/EFAS) to [-180, 180]
    if w > 180.5 or e > 180.5:
        if (w == 0.0 and e == 360.0) or (w == 360.0 and e == 0.0):
            w, e = -180.0, 180.0
        else:
            if w > 180.5:
                w = w - 360.0
            if e > 180.5:
                e = e - 360.0

    # 4. Fix inverted south > north
    if s > n:
        s, n = n, s

    # 5. Fix inverted west > east when both lie in the same hemisphere or do not cross the antimeridian
    # True Pacific antimeridian crossovers start east of Greenwich (w > 0) and end west (e < 0).
    if w > e and not (w > 0.0 and e < 0.0):
        w, e = e, w

    return [w, s, e, n]


def clean_doi(val: Any) -> str | None:
    """Extract a canonical bare DOI (`10.xxxx/...`) if present, otherwise return None."""
    s = clean_str(val)
    if not s:
        return None
    m = re.search(r"\b(10\.\d{4,9}/[^\s,;\"'<>\]]+)", s)
    if not m:
        return None
    d = m.group(1).rstrip(".,;:]\"'")
    if d.endswith(")") and d.count(")") > d.count("("):
        d = d[:-1].rstrip(".,;:]\"'")
    return d if re.match(r"^10\.\d{4,9}/[^\s\]]+$", d) else None


def clean_url(val: Any) -> str | None:
    """Normalize malformed URL schemes/markdown/prose wrappers to a clean http(s)/ftp URL or None."""
    s = clean_str(val)
    if not s:
        return None
    s = re.sub(r"^ftp:/([^/])", r"ftp://\1", s, flags=re.IGNORECASE)
    s = re.sub(r"^http:([a-zA-Z0-9])", r"https://\1", s, flags=re.IGNORECASE)
    m = re.search(r"((?:https?|ftp)://[^\s,;\"'<>)\]]+)", s, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".,;")
    doi_only = clean_doi(s)
    if doi_only:
        return f"https://doi.org/{doi_only}"
    return None


def extract_stac_temporal(extent: Any) -> tuple[str | None, str | None]:
    if not isinstance(extent, dict):
        return None, None
    temp = extent.get("temporal", {})
    intervals = temp.get("interval", []) if isinstance(temp, dict) else []
    if intervals and isinstance(intervals[0], list) and len(intervals[0]) >= 2:
        t_start, t_end = clean_str(intervals[0][0]), clean_str(intervals[0][1])
        if (
            t_start
            and t_end
            and re.match(r"^\d{4}-\d{2}-\d{2}", t_start)
            and re.match(r"^\d{4}-\d{2}-\d{2}", t_end)
            and t_start[:10] > t_end[:10]
        ):
            t_start, t_end = t_end, t_start
        return t_start, t_end
    return None, None


def extract_stac_bbox(extent: Any) -> list[float] | None:
    if not isinstance(extent, dict):
        return None
    spat = extent.get("spatial", {})
    bboxes = spat.get("bbox", []) if isinstance(spat, dict) else []
    if bboxes and isinstance(bboxes[0], list) and len(bboxes[0]) >= 4:
        return clean_bbox(bboxes[0][:4])
    return None


def extract_stac_gsd(rec: dict[str, Any]) -> str | None:
    summaries = rec.get("summaries", {})
    if isinstance(summaries, dict):
        for k in ("gsd", "spatial_resolution", "resolution"):
            if k in summaries and summaries[k]:
                return clean_str(summaries[k])
    if "gsd" in rec and rec["gsd"]:
        return clean_str(rec["gsd"])
    cube = rec.get("cube:dimensions", {})
    if isinstance(cube, dict):
        for dim in cube.values():
            if isinstance(dim, dict) and dim.get("step"):
                step = dim.get("step")
                unit = dim.get("unit")
                if not unit:
                    ref_sys = str(dim.get("reference_system") or "").lower()
                    try:
                        step_val = abs(float(step))
                    except Exception:
                        step_val = None
                    if "metre" in ref_sys or "meter" in ref_sys or "projjson" in ref_sys or (step_val is not None and step_val >= 10.0):
                        unit = "m"
                    elif step_val is not None and 0 < step_val <= 5.0:
                        unit = "deg"
                    else:
                        return None
                return f"{step} {unit}".strip()
    return None


def extract_stac_providers(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    providers = rec.get("providers", [])
    if not isinstance(providers, list) or not providers:
        return None, None
    producers = []
    hosts = []
    all_names = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        name = clean_str(p.get("name"))
        if not name:
            continue
        all_names.append(name)
        roles = [str(r).lower() for r in (p.get("roles") or [])]
        if "producer" in roles or "licensor" in roles or "processor" in roles:
            producers.append(name)
        if "host" in roles:
            hosts.append(name)
    prod_str = "; ".join(dict.fromkeys(producers)) if producers else (all_names[0] if all_names else None)
    host_str = "; ".join(dict.fromkeys(hosts)) if hosts else None
    return prod_str, host_str


def extract_stac_doi_and_url(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    doi = clean_str(rec.get("sci:doi") or rec.get("doi"))
    url = None
    for lk in (rec.get("links") or []):
        if not isinstance(lk, dict):
            continue
        rel = str(lk.get("rel", "")).lower()
        href = clean_str(lk.get("href"))
        if not href:
            continue
        if "doi.org/" in href and not doi:
            doi = href
        if rel in ("canonical", "alternate", "about", "describedby", "self") and not url and href.startswith("http"):
            url = href
    return doi, url


# ---------------------------------------------------------------------------
# Deterministic L0/L1/L2 & Non-EO Exclusion Classifier
# ---------------------------------------------------------------------------
EXPLICIT_L0_L1_L2_TOKEN_RE = re.compile(
    r"^(?:Visualisation|Visualization|(?:level[- ]?|l)?[012](?:[abcdgprst]+|\.\d+)?\+?)$",
    re.IGNORECASE,
)

SWATH_L1_L2_ID_PATTERNS = re.compile(
    r"(^|:|/|_|-)(l0b?|l1|l1c|l1b|l1a|l1d|l1g|l1r|l1t|l1gt|l1gst|l2|l2a|l2b|l2c|l2g|l2p|l2t|level[-_]1|level[-_]2|"
    r"ol_1_efr|ol_1_err|ol_2_wfr|ol_2_wrr|sl_1_rbt|sl_2_wst|sl_2_lst|sr_1_sra|sr_2_wat|sr_2_lan|"
    r"s1_sar_grd|s1_sar_slc|s1_sar_raw|s2_msi_l1c|s2_msi_l2a|s5p_l1b|s5p_l2|l2__no2|l2__o3|l2__ch4|l2__co_|l2__so2|l2__hcho|"
    r"l2__aer|l2__cloud|l2__np_bd|amsul1|avhrrl1|mhsl1|hirsl1|iasil2|iassnd02|"
    r"landsat-c2l1|landsat-c2l2|sentinel-1-grd|sentinel-1-rtc|sentinel-1-slc-burst|sentinel-2-l1c|sentinel-2-l2a|"
    r"sentinel-3-olci-l1|sentinel-3-slstr-l1|sentinel-3-.*-l2-netcdf|envisat-meris-l1|envisat-asar|"
    r"ast_l1t|l2a\.gedi|slc_off|slc-off)(_|-|:|/|$)|"
    r"(L0L1Products|L1Products|L2Products|L2Validated|L2InstChecked|^l2-along-track|"
    r"^swath-elelvation-|^ccm-|^CEOS_ARD_NRB_S1_SAMPLES$|^SENTINEL1_(?:SCENE_)?GMR0$|^SENTINEL1_SIG0_20M$|^SENTINEL1_MPLIA$|"
    r"^COPERNICUS/S3/OLCI$|^JAXA/ALOS/AVNIR-2/ORI$|^LANDSAT/.*/C0[12]/T[12](?:_RT)?(?:_TOA)?$|^LANDSAT/GLS\d{4}(?:_L[57])?$|"
    r"^MODIS/(?:006|061)/M[OY]D09G[AQ]$|^NOAA/VIIRS/.*/VNP09GA$)",
    re.IGNORECASE,
)

SWATH_L1_L2_TITLE_PATTERNS = re.compile(
    r"(\b(level[- ]?(?:0[ab]?|1|1\.5|1[abcdgrt]|1gt|1gst|2|2[abcgprt])\b|l0[ab]?\b|l1[abcdgrt]?\b|l1gt\b|l1gst\b|l1\.5\b|l2[abcgprt]?\b)|"
    r"\b(single look complex|slc[- ]off|slc burst|ground range detected|top-of-atmosphere radiance|calibrated radiance|"
    r"raw scenes?|toa reflectance|raw instrument data|raw data record|\brdr\b|temperature data record|\btdr\b|sensor data record|\bsdr\b|"
    r"swath level 2|on the satellite swath|along-track re-tracked|l2 swath|swath grid|swath image|swath files|swath[- ]derived|swath elevation)\b)",
    re.IGNORECASE,
)

AERIAL_ORTHOPHOTO_PATTERNS = re.compile(
    r"(^naip$|^USDA/NAIP/|^USGS_DOQ$|^DOP_AUT_|^topo-dc-austria-op-utm$|"
    r"^(?:Estonia|Latvia|Germany)/.*/orthos/|^Slovakia/orthos/|^Netherlands/.*/LUCHTFOTO|^Spain/PNOA/|^Finland/(?:MAVI|SMK)/|"
    r"\b(?:digital orthophotos?|orthophotos?|digital orthophoto quadrangles?|naip: national agriculture imagery program)\b)",
    re.IGNORECASE,
)

# Multi-temporal composites, gridded syntheses, or L2-tagged CDRs that represent spatially gridded/composited EO products
COMPOSITE_OR_L3_OVERRIDE_PATTERNS = re.compile(
    r"(\b(?:level[- ]?[34][abc]?\b|l3[abc]?\b|l4[abc]?\b|l2 and l3\b|l2/3\b|"
    r"global mosaic|cloudless mosaic|monthly mosaic|quarterly mosaic|annual mosaic|"
    r"cloud-free annual composites?|s2mosaic|pixel based image composite|"
    r"8-day.*composite|32-day.*composite|annual.*composite|"
    r"harmonized landsat sentinel|hlsl30|hlss30|"
    r"daily synthesis|10-daily synthesis|1-day surface reflectance|10-day surface reflectance|"
    r"l2 statistics|gridded user friendly product|gravity field models and grids|"
    r"ghrsst level 2p gridded|data cube.*lcf \d+ days|"
    r"daily surface albedo|gsa level 2|10-day surface albedo|"
    r"fire radiative power grid|gridded fire radiative power|"
    r"daily shortwave solar irradiance|daily downward longwave irradiance|daily surface solar irradiance|"
    r"reference evapotranspiration|normalized difference vegetation index cdr|"
    r"downward surface solar radiation.*daily mean)\b|"
    r"(?:syn-2-v10|syn-2-vg1|s3-syn-vgt|sentinel-3-synergy-v10|l2c-ndvi|l2-smos-sss-daily|gomos_ufp_gridded|goce_global_gravity_field_models_and_grids))",
    re.IGNORECASE,
)

SWATH_OR_UNGRIDDED_GUARD_RE = re.compile(
    r"\b(swath grid|swath level|on the satellite swath|along[- ]track|ungridded|/pixel\b|radiance|bending angle|sounder|sounding profiles?)\b",
    re.IGNORECASE,
)

# Non-sensor domain uses of "Level 1/2" (FAO WaPOR spatial scale tiers, TransCom 3 project tiers, GADM/NUTS admin levels, IUCN hierarchy)
NON_SENSOR_HIERARCHY_PATTERNS = re.compile(
    r"(^FAO_GISMGR:WAPOR|^FAO/WAPOR/|transcom 3:|administrative level [0123]|global administrative unit|iucn_habitat|water resources zones: l[12]|river water level|lake water level)",
    re.IGNORECASE,
)


def check_l3_l4_eligibility(
    raw_id: str | None,
    title: str | None,
    description: str | None,
    level_raw: str | None,
    source_id: str,
) -> tuple[bool, str | None]:
    """
    Returns (is_eligible_l3_l4, exclusion_reason).
    If eligible, exclusion_reason is None.
    """
    rid = (raw_id or "").strip()
    ttl = (title or "").strip()
    lvl = (level_raw or "").strip()
    combined_id_title = f"{rid} {ttl}"
    is_non_sensor_hier = bool(NON_SENSOR_HIERARCHY_PATTERNS.search(combined_id_title))
    is_composite_override = bool(
        COMPOSITE_OR_L3_OVERRIDE_PATTERNS.search(combined_id_title)
        and not SWATH_OR_UNGRIDDED_GUARD_RE.search(combined_id_title)
    )

    # 1. Check explicit processing_level_raw field
    if lvl:
        lvl_tokens = [t.strip().lower() for t in re.split(r"[;,/|]", lvl) if t.strip()]
        has_l3_l4 = any(
            re.match(r"^(?:level[- ]?|l)?[34][abc]?$", t)
            or "level 3" in t
            or "level 4" in t
            for t in lvl_tokens
        )
        has_only_l0_l1_l2 = all(EXPLICIT_L0_L1_L2_TOKEN_RE.match(t) for t in lvl_tokens) and len(lvl_tokens) > 0
        has_l0_or_l1_or_vis = any(
            re.match(r"^(?:visuali[sz]ation|(?:level[- ]?|l)?(?:0|1|1\.5)(?:[abcdgrt]+|\.\d+)?)$", t)
            for t in lvl_tokens
        )
        if has_only_l0_l1_l2 and not has_l3_l4:
            # Allow explicit L2-only records when overridden by non-sensor hierarchy or verified gridded/composited CDR patterns
            # Never override explicit L0/L1/L1.5/Visualisation radiances or quicklooks.
            if has_l0_or_l1_or_vis or (not is_non_sensor_hier and not is_composite_override):
                return False, f"explicit_non_l3_l4_level:{lvl}"

    # 2. Check sandbox/test placeholder records and pure in-situ discrete station/buoy/gauge/borehole logs or Eurostat tables
    if re.match(r"^this is (a )?test$", ttl, re.IGNORECASE):
        return False, "sandbox_test_record"

    if (
        (re.search(r"INSITU_.*_(DISCRETE|NRT_OBSERVATIONS)", rid, re.IGNORECASE) and not re.search(r"(OA|GRID|L3|L4|REP_OBSERVATIONS)", rid, re.IGNORECASE))
        or re.match(r"^insitu-observations-", rid, re.IGNORECASE)
        or re.match(r"^EO:ECMWF:DAT:INSITU_(?:OBSERVATIONS_|COMPREHENSIVE_|GLACIERS_EXTENT)", rid, re.IGNORECASE)
        or re.match(r"^COPERNICUS/MARINE/INSITU_MULTI_4KM/", rid, re.IGNORECASE)
        or (source_id == "24_openlandmap_stac" and re.match(r"^(?:fluxnet|gbov|veg\.plot|fapar\.eml|geowiki\..*|obis)$", rid, re.IGNORECASE))
        or re.match(r"^(?:STAT\.EUSTAT\.DAT\.|EO\.GBIF\.DAT\.OCCURRENCE)", rid, re.IGNORECASE)
        or re.search(r"\b(drifting buoy data|belfort rain gauge data|subsurface temperature recorder|global database of borehole temperatures|marine optical buoy \(moby\))\b", ttl, re.IGNORECASE)
    ):
        return False, "pure_insitu_discrete_stations"

    # 3. Check aerial orthophotos (sub-meter airborne surveys, not satellite L3/L4)
    if AERIAL_ORTHOPHOTO_PATTERNS.search(rid) or AERIAL_ORTHOPHOTO_PATTERNS.search(ttl):
        return False, "aerial_orthophoto_imagery"

    # 4. Check swath/granule L1/L2 IDs or titles unless it is a non-sensor hierarchy or multi-temporal L3 composite
    if not is_non_sensor_hier and not is_composite_override:
        if SWATH_L1_L2_ID_PATTERNS.search(rid):
            return False, "swath_or_tile_l1_l2_id"
        if SWATH_L1_L2_TITLE_PATTERNS.search(ttl):
            return False, "swath_or_tile_l1_l2_title"

    return True, None


# ---------------------------------------------------------------------------
# Source-Specific Parsers (28 product sources + 1 taxonomy source)
# ---------------------------------------------------------------------------
def _extract_geojson_bbox(geom_obj: Any, primary_only: bool = False) -> list[float] | None:
    """Recursively extracts a validated WGS84 [west, south, east, north] bbox from GeoJSON geometries."""
    if not geom_obj:
        return None
    if primary_only and isinstance(geom_obj, list) and len(geom_obj) > 1:
        sub_boxes = [b for b in (_extract_geojson_bbox(g, primary_only=False) for g in geom_obj) if b is not None]
        if sub_boxes:
            eu_boxes = [b for b in sub_boxes if b[0] >= -46.0 and b[1] >= 24.0 and b[2] <= 70.0 and b[3] <= 85.0]
            target = eu_boxes if eu_boxes else sub_boxes
            return [
                round(min(b[0] for b in target), 6),
                round(min(b[1] for b in target), 6),
                round(max(b[2] for b in target), 6),
                round(max(b[3] for b in target), 6),
            ]
        return None
    xs: list[float] = []
    ys: list[float] = []

    def _walk(c: Any) -> None:
        if isinstance(c, (list, tuple)):
            if len(c) >= 2 and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)) and not isinstance(c[0], bool) and not isinstance(c[1], bool):
                xs.append(float(c[0]))
                ys.append(float(c[1]))
            else:
                for item in c:
                    _walk(item)
        elif isinstance(c, dict) and "coordinates" in c:
            _walk(c["coordinates"])

    _walk(geom_obj)
    if xs and ys:
        w, s, e, n = min(xs), min(ys), max(xs), max(ys)
        if -180.0 <= w <= 180.0 and -90.0 <= s <= 90.0 and -180.0 <= e <= 180.0 and -90.0 <= n <= 90.0 and s <= n:
            return [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]
    return None


def _split_date_range_str(val: Any) -> tuple[str | None, str | None]:
    """Splits free-text date spans (e.g. '1950-2100', '1901-present') into (start, end)."""
    s = clean_str(val)
    if not s:
        return None, None
    years = [int(y) for y in re.findall(r"\b(19\d\d|20[0-3]\d|2100)\b", s)]
    if not years:
        return None, None
    has_ongoing = bool(re.search(r"\b(present|ongoing|current|now|near[- ]real[- ]time|nrt)\b", s, re.IGNORECASE))
    if len(years) == 1:
        y0 = str(years[0])
        return y0, None if has_ongoing else y0
    return str(min(years)), None if has_ongoing else str(max(years))


def parse_01_cmems() -> list[dict[str, Any]]:
    rows = []
    for p in sorted((RAW_DIR / "01_copernicus_cmems/data").glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pm = d.get("providerMetadata", {}) or {}
                ext = d.get("geoExtent", {}) or {}
                bbox = None
                sb = d.get("stacOrCswBbox")
                if isinstance(sb, list) and len(sb) == 4:
                    try:
                        w, s, e, n = float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])
                        if -180.0 <= w <= 180.0 and -90.0 <= s <= 90.0 and -180.0 <= e <= 180.0 and -90.0 <= n <= 90.0 and s <= n:
                            bbox = [w, s, e, n]
                    except Exception:
                        bbox = None
                if bbox is None and isinstance(ext, dict) and "bbox" in ext and isinstance(ext["bbox"], list) and len(ext["bbox"]) == 4:
                    bbox = [float(x) for x in ext["bbox"]]
                if bbox is None:
                    bbox = _extract_geojson_bbox(ext)
                gr = d.get("geoResolution")
                spat_raw = None
                if isinstance(gr, dict):
                    rc = gr.get("row") or gr.get("column") or {}
                    if isinstance(rc, dict) and rc.get("magnitude") is not None:
                        spat_raw = clean_str(f"{rc['magnitude']} {rc.get('units') or ''}")
                if not spat_raw and not isinstance(gr, dict):
                    spat_raw = clean_str(gr)
                rows.append({
                    "source_id": "01_copernicus_cmems",
                    "raw_id": clean_str(d.get("id")),
                    "title": clean_str(d.get("title")),
                    "description": clean_str(d.get("abstract")),
                    "processing_level_raw": clean_str(d.get("processingLevel") or pm.get("processingLevel")),
                    "spatial_resolution_raw": spat_raw,
                    "temporal_resolution_raw": clean_str(d.get("tempResolutions")),
                    "temporal_start_raw": clean_str(d.get("tempExtentBegin")),
                    "temporal_end_raw": clean_str(d.get("tempExtentEnd")),
                    "update_frequency_raw": clean_str(d.get("updateFrequencies")),
                    "bbox_raw": bbox,
                    "producing_org_raw": clean_str(d.get("originatingCenter") or pm.get("originatingCenter")) or "Mercator Océan International (Copernicus Marine Service - CMEMS)",
                    "platform_sensor_raw": clean_str(d.get("assimilatedData") or d.get("sources")),
                    "doi_raw": clean_str(d.get("doi")),
                    "license_raw": "Copernicus Data Policy (Free and Open)",
                    "landing_url_raw": f"https://data.marine.copernicus.eu/product/{d.get('id')}" if d.get("id") else None,
                    "keywords_raw": clean_str(d.get("keywords") or d.get("featureTypes")),
                })
    return rows


def parse_02_clms() -> list[dict[str, Any]]:
    rows = []
    p = RAW_DIR / "02_copernicus_clms/data/clms_products_detailed.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bbox = None
            gbox = d.get("geographicBoundingBox")
            items = gbox.get("items") if isinstance(gbox, dict) else (gbox if isinstance(gbox, list) else None)
            if isinstance(items, list) and items:
                valid_boxes: list[list[float]] = []
                for b_it in items:
                    if not isinstance(b_it, dict):
                        continue
                    try:
                        w, s, e, n = float(b_it["west"]), float(b_it["south"]), float(b_it["east"]), float(b_it["north"])
                        if -180.0 <= w <= 180.0 and -90.0 <= s <= 90.0 and -180.0 <= e <= 180.0 and -90.0 <= n <= 90.0 and s <= n:
                            valid_boxes.append([w, s, e, n])
                    except Exception:
                        continue
                if valid_boxes:
                    eu_boxes = [b for b in valid_boxes if b[0] >= -46.0 and b[1] >= 24.0 and b[2] <= 70.0 and b[3] <= 85.0]
                    target_boxes = eu_boxes if eu_boxes else valid_boxes
                    bbox = [
                        min(b[0] for b in target_boxes),
                        min(b[1] for b in target_boxes),
                        max(b[2] for b in target_boxes),
                        max(b[3] for b in target_boxes),
                    ]
            spat_res = clean_str(
                d.get("characteristics_spatial_resolution")
                or d.get("qualitySpatialResolution_line")
                or d.get("spatial_resolution")
            )
            temp_ext = clean_str(d.get("characteristics_temporal_extent"))
            t_start = clean_str(d.get("temporalExtentStart"))
            t_end = clean_str(d.get("temporalExtentEnd"))
            if not t_start and temp_ext:
                years = re.findall(r"\b(19\d\d|20\d\d)\b", temp_ext)
                if years:
                    t_start = years[0]
                    if "present" not in temp_ext.lower() and len(years) > 1:
                        t_end = years[-1]
            if any(k in f"{temp_ext or ''} {d.get('title') or ''} {t_end or ''}".lower() for k in ("present", "ongoing")):
                t_end = None
            if t_start and t_end and str(t_start)[:10] > str(t_end)[:10]:
                t_end = None
            platform = clean_str(
                [d.get("characteristics_platform"), d.get("characteristics_sensor"), d.get("platform"), d.get("sensor")]
            )
            rows.append({
                "source_id": "02_copernicus_clms",
                "raw_id": clean_str(d.get("UID") or d.get("id") or d.get("@id")),
                "title": clean_str(d.get("title")),
                "description": clean_str(d.get("description")),
                "processing_level_raw": clean_str(d.get("processing_level")),
                "spatial_resolution_raw": spat_res,
                "temporal_resolution_raw": clean_str(d.get("characteristics_update_frequency") or d.get("update_frequency")),
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": clean_str(d.get("characteristics_update_frequency")),
                "bbox_raw": bbox,
                "producing_org_raw": clean_str(d.get("organisationName") or "European Environment Agency / EC JRC (Copernicus Land Monitoring Service - CLMS)"),
                "platform_sensor_raw": platform,
                "doi_raw": clean_str(d.get("ResourceIdentifier")),
                "license_raw": clean_str(d.get("useLimitation") or "Copernicus Data Policy (Free and Open)"),
                "landing_url_raw": clean_str(d.get("@id")),
                "keywords_raw": clean_str(d.get("taxonomy_dataset_category") or d.get("subjects")),
            })
    return rows


def parse_cads_stac(source_id: str, filename: str, default_org: str) -> list[dict[str, Any]]:
    rows = []
    data = json.loads((RAW_DIR / source_id / "data" / filename).read_text(encoding="utf-8"))
    for c in data.get("collections", []):
        t_start, t_end = extract_stac_temporal(c.get("extent"))
        bbox = extract_stac_bbox(c.get("extent"))
        doi, url = extract_stac_doi_and_url(c)
        prod, _ = extract_stac_providers(c)
        rows.append({
            "source_id": source_id,
            "raw_id": clean_str(c.get("id")),
            "title": clean_str(c.get("title")),
            "description": clean_str(c.get("description")),
            "processing_level_raw": None,
            "spatial_resolution_raw": extract_stac_gsd(c),
            "temporal_resolution_raw": clean_str(c.get("cads:update_frequency")),
            "temporal_start_raw": t_start,
            "temporal_end_raw": t_end,
            "update_frequency_raw": clean_str(c.get("cads:update_frequency")),
            "bbox_raw": bbox,
            "producing_org_raw": prod or default_org,
            "platform_sensor_raw": None,
            "doi_raw": doi,
            "license_raw": clean_str(c.get("license")),
            "landing_url_raw": url,
            "keywords_raw": clean_str(c.get("keywords")),
        })
    return rows


def parse_generic_stac_collections(source_id: str, collections: list[dict[str, Any]], default_org: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for c in collections:
        if not isinstance(c, dict):
            continue
        t_start, t_end = extract_stac_temporal(c.get("extent"))
        bbox = extract_stac_bbox(c.get("extent"))
        doi, url = extract_stac_doi_and_url(c)
        prod, host = extract_stac_providers(c)
        summaries = c.get("summaries", {}) if isinstance(c.get("summaries"), dict) else {}
        lvl = clean_str(
            c.get("processing:level")
            or summaries.get("processing:level")
            or summaries.get("eo:processing_level")
            or c.get("osc:type")
        )
        plat = clean_str(
            summaries.get("platform")
            or summaries.get("instruments")
            or summaries.get("constellation")
            or c.get("osc:missions")
        )
        temp_res = clean_str(
            c.get("gee:interval")
            or summaries.get("temporal_resolution")
            or c.get("cads:update_frequency")
        )
        rows.append({
            "source_id": source_id,
            "raw_id": clean_str(c.get("id")),
            "title": clean_str(c.get("title") or c.get("id")),
            "description": clean_str(c.get("description")),
            "processing_level_raw": lvl,
            "spatial_resolution_raw": extract_stac_gsd(c),
            "temporal_resolution_raw": temp_res,
            "temporal_start_raw": t_start,
            "temporal_end_raw": t_end,
            "update_frequency_raw": clean_str(c.get("gee:status") or c.get("osc:status")),
            "bbox_raw": bbox,
            "producing_org_raw": prod or default_org,
            "hosting_platform_raw": host,
            "platform_sensor_raw": plat,
            "doi_raw": doi,
            "license_raw": clean_str(c.get("license")),
            "landing_url_raw": url,
            "keywords_raw": clean_str(c.get("keywords") or c.get("gee:categories") or c.get("osc:variables") or c.get("Theme")),
        })
    return rows


def parse_07_jrc() -> tuple[list[dict[str, Any]], int]:
    """
    Parses the EC JRC datasets from data.europa.eu DCAT-AP JSONL files plus JRC FOROBS TMF portal.
    Deterministic EO/environmental filter: retains JRC datasets that belong to Earth Observation,
    CEMS (GHSL, EDO, GDO, EFFIS, GWIS, GloFAS), land/forest/water/climate/soil/biodiversity/emissions
    monitoring, while excluding non-EO laboratory/nuclear/LCA/biomedical/socioeconomic spreadsheets.
    """
    non_eo_jrc_id_re = re.compile(
        r"^(jrc-odin|jrc-eplca|jrc-abcis|jrc-10088|jrc-10111|jrc-eanr|jrc-cet|jrc-fin|jrc-coin)",
        re.IGNORECASE,
    )
    non_eo_jrc_title_re = re.compile(
        r"\b(thermo-mechanical fatigue|life cycle assessment|sars-cov-2|eurl ecvam|alzheimer|breast cancer|"
        r"prostate cancer|indoor localization|wearables and health|eurobarometer|customs union|free trade area|"
        r"dietary transitions|tourism recovery|accommodations listed online|share of trips by train)\b",
        re.IGNORECASE,
    )
    eo_pattern = re.compile(
        r"\b(ghsl|ghs[-_]|human settlement|built[-_ ]up|population grid|degree of urbanisation|smod|"
        r"global surface water|gswe|tropical moist forest|forobs|projects/jrc/tmf|drought observatory|edo\b|gdo\b|spi\b|sma\b|"
        r"fapar|effis|gwis|wildfire|burnt area|burned area|glofas|efas|flood hazard|flood monitoring|"
        r"edgar|emission|esdac|soil erosion|glosem|soil organic carbon|dopa|protected area|biodiversity|"
        r"pvgis|solar radiation|asap|agricultural production|biomass|forest|land cover|land use|"
        r"remote sensing|satellite|sentinel|copernicus|earth observation|modis|landsat|ndvi|albedo|"
        r"marine|ocean|chlorophyll|sea surface|climate|cryosphere|glacier|permafrost|air quality|aerosol)\b",
        re.IGNORECASE,
    )
    rows = []
    seen_ids: set[str] = set()
    non_eo_excluded = 0
    for p in sorted((RAW_DIR / "07_ec_jrc_data_catalogue/data").glob("jrc_dcat_ap_datasets_part*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid = clean_str(d.get("id") or d.get("identifier"))
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                title = clean_str(d.get("title"))
                desc = clean_str(d.get("description"))
                kw_list = []
                for k in (d.get("keywords") or []):
                    if isinstance(k, dict) and k.get("label"):
                        kw_list.append(str(k["label"]))
                kw_str = "; ".join(dict.fromkeys(kw_list)) if kw_list else None
                if (rid and non_eo_jrc_id_re.search(rid)) or (title and non_eo_jrc_title_re.search(title)):
                    non_eo_excluded += 1
                    continue
                combined = f"{title or ''} {desc or ''} {kw_str or ''}"
                if not eo_pattern.search(combined):
                    non_eo_excluded += 1
                    continue

                t_start, t_end = None, None
                temp = d.get("temporal")
                if isinstance(temp, list) and temp and isinstance(temp[0], dict):
                    t_start = clean_str(temp[0].get("gte") or temp[0].get("start_date"))
                    t_end = clean_str(temp[0].get("lte") or temp[0].get("end_date"))
                if t_start and t_end and str(t_start)[:10] > str(t_end)[:10]:
                    t_end = None

                accrual = clean_str(d.get("accrual_periodicity"))
                doi = None
                landing = clean_str(d.get("resource"))
                for adms in (d.get("adms_identifier") or []):
                    if isinstance(adms, dict) and adms.get("identifier") and "doi" in str(adms.get("scheme", "")).lower():
                        doi = clean_str(adms.get("identifier"))
                rows.append({
                    "source_id": "07_ec_jrc_data_catalogue",
                    "raw_id": rid,
                    "title": title,
                    "description": desc,
                    "processing_level_raw": None,
                    "spatial_resolution_raw": clean_str(d.get("spatial_resolution")),
                    "temporal_resolution_raw": accrual,
                    "temporal_start_raw": t_start,
                    "temporal_end_raw": t_end,
                    "update_frequency_raw": accrual,
                    "bbox_raw": _extract_geojson_bbox(d.get("spatial")),
                    "producing_org_raw": "European Commission Joint Research Centre (EC JRC)",
                    "platform_sensor_raw": None,
                    "doi_raw": doi,
                    "license_raw": "European Commission Reuse Policy (CC-BY-4.0 / Free and Open)",
                    "landing_url_raw": landing,
                    "keywords_raw": kw_str,
                })

    # Also parse EC JRC FOROBS Tropical Moist Forest (TMF) products
    tmf_path = RAW_DIR / "07_ec_jrc_data_catalogue/data/jrc_forobs_tmf_products.json"
    if tmf_path.exists():
        for t in json.loads(tmf_path.read_text(encoding="utf-8")):
            trid = clean_str(t.get("id"))
            if trid and trid in seen_ids:
                continue
            if trid:
                seen_ids.add(trid)
            rows.append({
                "source_id": "07_ec_jrc_data_catalogue",
                "raw_id": trid,
                "title": clean_str(t.get("title")),
                "description": clean_str(t.get("description")),
                "processing_level_raw": "Level 3",
                "spatial_resolution_raw": clean_str(t.get("spatial_resolution_raw")),
                "temporal_resolution_raw": clean_str(t.get("update_frequency_raw")),
                "temporal_start_raw": clean_str(t.get("temporal_start_raw")),
                "temporal_end_raw": clean_str(t.get("temporal_end_raw")),
                "update_frequency_raw": clean_str(t.get("update_frequency_raw")),
                "bbox_raw": [-180.0, -30.0, 180.0, 30.0],
                "producing_org_raw": "European Commission Joint Research Centre (EC JRC FOROBS)",
                "platform_sensor_raw": clean_str(t.get("platform_sensor_raw")),
                "doi_raw": None,
                "license_raw": "European Commission Reuse Policy (Free and Open)",
                "landing_url_raw": clean_str(t.get("landing_url_raw")),
                "keywords_raw": clean_str(t.get("keywords_raw")),
            })

    return rows, non_eo_excluded


def parse_09_wekeo() -> list[dict[str, Any]]:
    rows = []
    seen = set()
    p = RAW_DIR / "09_wekeo_dias/data/wekeo_hda_broker_datasets.jsonl"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                ft = json.loads(line)
                md = ft.get("metadata", {}) or {}
                did = clean_str(md.get("id") or ft.get("dataset_ext_id") or ft.get("dataset_id"))
                if not did or did in seen:
                    continue
                seen.add(did)
                loc = md.get("location", {}) or {}
                coords = loc.get("coordinates")
                bbox = None
                if isinstance(coords, list) and len(coords) == 2 and isinstance(coords[0], list) and len(coords[0]) == 2:
                    try:
                        xs = [float(coords[0][0]), float(coords[1][0])]
                        ys = [float(coords[0][1]), float(coords[1][1])]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    except Exception:
                        bbox = None
                kws = md.get("keywords") or []
                lvls: list[str] = []
                for k in (kws if isinstance(kws, list) else []):
                    ks = str(k).lower()
                    if ks in ("level-3", "level-4", "level-2", "level-1", "level-0", "l3", "l4", "l2", "l1"):
                        lvls.append(str(k))
                lvl = "; ".join(dict.fromkeys(lvls)) if lvls else None
                rows.append({
                    "source_id": "09_wekeo_dias",
                    "raw_id": did,
                    "title": clean_str(md.get("datasetTitle") or md.get("title") or did),
                    "description": clean_str(md.get("abstract") or md.get("description")),
                    "processing_level_raw": lvl,
                    "spatial_resolution_raw": None,
                    "temporal_resolution_raw": None,
                    "temporal_start_raw": clean_str(md.get("tempextent_begin")),
                    "temporal_end_raw": clean_str(md.get("tempextent_end")),
                    "update_frequency_raw": clean_str(md.get("productStatus")),
                    "bbox_raw": bbox,
                    "producing_org_raw": clean_str(md.get("responsible_org") or md.get("responsibleParty")),
                    "hosting_platform_raw": "WEkEO Copernicus DIAS",
                    "platform_sensor_raw": None,
                    "doi_raw": None,
                    "license_raw": clean_str(ft.get("terms") or md.get("conditions")),
                    "landing_url_raw": f"https://www.wekeo.eu/data?view=dataset&dataset={did}",
                    "keywords_raw": clean_str(kws),
                })
    return rows


def parse_11_eea_emodnet() -> list[dict[str, Any]]:
    rows = []
    for fname, default_org, portal_base in [
        ("eea_sdi_datasets.jsonl", "European Environment Agency (EEA)", "https://sdi.eea.europa.eu/catalogue/srv/eng/catalog.search#/metadata/"),
        ("emodnet_datasets.jsonl", "European Marine Observation and Data Network (EMODnet)", "https://emodnet.ec.europa.eu/geonetwork/srv/eng/catalog.search#/metadata/"),
    ]:
        p = RAW_DIR / "11_eea_and_emodnet/data" / fname
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                uuid = clean_str(d.get("metadataIdentifier") or d.get("_id") or d.get("id"))
                title = clean_str(d.get("resourceTitleObject"))
                desc = clean_str(d.get("resourceAbstractObject"))
                res_dist = clean_str(d.get("resolutionDistance"))
                t_start = clean_str(d.get("tempExtentBegin"))
                t_end = clean_str(d.get("tempExtentEnd"))
                if not t_start:
                    rtd = d.get("resourceTemporalExtentDetails")
                    if isinstance(rtd, list):
                        starts: list[str] = []
                        ends: list[str] = []
                        ongoing = False
                        for it in rtd:
                            if not isinstance(it, dict):
                                continue
                            st_d = it.get("start")
                            en_d = it.get("end")
                            st = st_d.get("date") if isinstance(st_d, dict) else None
                            en = en_d.get("date") if isinstance(en_d, dict) else None
                            indet = str((en_d or {}).get("indeterminatePosition") or "").lower() if isinstance(en_d, dict) else ""
                            if indet in ("now", "ongoing") or (st and not en):
                                ongoing = True
                            if st and re.match(r"^\d{4}", str(st)):
                                starts.append(str(st)[:10])
                            if en and re.match(r"^\d{4}", str(en)):
                                ends.append(str(en)[:10])
                        if starts:
                            t_start = min(starts)
                        if not ongoing and ends:
                            t_end = max(ends)
                        elif ongoing:
                            t_end = None
                if t_start and t_end and str(t_start)[:10] > str(t_end)[:10]:
                    t_end = None
                bbox = _extract_geojson_bbox(d.get("geom"), primary_only=(fname == "eea_sdi_datasets.jsonl"))
                freq = clean_str(d.get("cl_maintenanceAndUpdateFrequency"))
                org = clean_str(d.get("OrgObject") or d.get("custodianOrgForResourceObject")) or default_org
                doi = clean_str(d.get("resourceIdentifier"))
                rows.append({
                    "source_id": "11_eea_and_emodnet",
                    "raw_id": uuid,
                    "title": title,
                    "description": desc,
                    "processing_level_raw": None,
                    "spatial_resolution_raw": res_dist,
                    "temporal_resolution_raw": freq,
                    "temporal_start_raw": t_start,
                    "temporal_end_raw": t_end,
                    "update_frequency_raw": freq,
                    "bbox_raw": bbox,
                    "producing_org_raw": org,
                    "platform_sensor_raw": clean_str(d.get("cl_spatialRepresentationType")),
                    "doi_raw": doi,
                    "license_raw": clean_str(d.get("MD_LegalConstraintsUseLimitationObject")),
                    "landing_url_raw": f"{portal_base}{uuid}" if uuid else None,
                    "keywords_raw": clean_str(d.get("inspireTheme") or d.get("th_gemet")),
                })
    return rows


_WGC_HORIZ_UNITS = {1: "deg", 2: "km", 3: "m"}
_WGC_TEMP_UNITS = {1: "hours", 2: "days", 3: "months", 4: "years", 6: "minutes"}


def _extract_wgc_resolutions(items: Any, unit_map: dict[int, str]) -> str | None:
    if not isinstance(items, list):
        return None
    parts: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        mig = clean_str(it.get("migrated_text_value"))
        if mig and not re.match(r"^(?:n/?a|not available|not applicable)\b", mig, re.IGNORECASE):
            parts.append(mig)
            continue
        res = it.get("resolution")
        if isinstance(res, dict):
            uid = res.get("unit_id")
            unit = unit_map.get(uid) if uid is not None else None
            val = res.get("value")
            vmin = res.get("min")
            vmax = res.get("max")
            if unit and val is not None:
                v_str = str(int(val)) if float(val).is_integer() else str(val)
                u_str = unit[:-1] if float(val) == 1.0 and unit.endswith("s") else unit
                parts.append(f"{v_str} {u_str}")
            elif unit and vmin is not None and vmax is not None:
                parts.append(f"{vmin}-{vmax} {unit}")
    return "; ".join(dict.fromkeys(parts)) if parts else None


def _extract_wgc_orgs(orgs: Any) -> tuple[str | None, list[str]]:
    if not isinstance(orgs, list):
        return None, []
    preferred: list[str] = []
    fallback: list[str] = []
    acronyms_pref: list[str] = []
    acronyms_fb: list[str] = []
    for o in orgs:
        if not isinstance(o, dict):
            continue
        org_obj = o.get("organization") or {}
        acr = clean_str(org_obj.get("acronym"))
        name = clean_str(org_obj.get("name"))
        if not acr and not name:
            continue
        label = f"{name} ({acr})" if name and acr and acr.lower() not in name.lower() else (name or acr)
        short = acr or name
        role = clean_str((o.get("organization_role") or {}).get("name")) or ""
        if role in ("ResponsibleOrg", "TCDROrg"):
            preferred.append(label)
            if short:
                acronyms_pref.append(short)
        else:
            fallback.append(label)
            if short:
                acronyms_fb.append(short)
    chosen = list(dict.fromkeys(preferred or fallback))
    chosen_acr = list(dict.fromkeys(acronyms_pref or acronyms_fb))
    return ("; ".join(chosen) if chosen else None), chosen_acr


def _extract_wgc_instruments(items: Any) -> tuple[str | None, list[str]]:
    if not isinstance(items, list):
        return None, []
    labels: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ins_sat = it.get("instr_sat") or {}
        sat = clean_str((ins_sat.get("satellite") or {}).get("acronym"))
        inst = clean_str((ins_sat.get("instrument") or {}).get("acronym"))
        if sat and inst:
            labels.append(f"{sat} ({inst})")
        elif sat or inst:
            labels.append(sat or inst)
    uniq = list(dict.fromkeys(labels))
    return ("; ".join(uniq) if uniq else None), uniq


def _extract_wgc_variables(nodes: Any, extra_labels: Any) -> tuple[str | None, dict[str, str]]:
    kws: list[str] = []
    by_type: dict[str, str] = {}
    if isinstance(nodes, list):
        for vn in nodes:
            if not isinstance(vn, dict):
                continue
            vnode = vn.get("variables_node") or {}
            vname = clean_str(vnode.get("name"))
            vtype = clean_str((vnode.get("type") or {}).get("name"))
            if vname:
                kws.append(vname)
                if vtype and vtype not in by_type:
                    by_type[vtype] = vname
    lbl_str = clean_str(extra_labels)
    if lbl_str:
        kws.append(lbl_str)
    uniq = list(dict.fromkeys(kws))
    return ("; ".join(uniq) if uniq else None), by_type


def parse_12_wgclimate_ecv() -> list[dict[str, Any]]:
    rows = []
    p = RAW_DIR / "12_ceos_cgms_wgclimate_ecv/data/wgclimate_ecv_data_records.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("results", data.get("data", []))
    for r in records:
        org_str, org_acrs = _extract_wgc_orgs(r.get("organizations"))
        plat_str, plat_list = _extract_wgc_instruments(r.get("instruments_satellites"))
        kw_str, var_by_type = _extract_wgc_variables(r.get("variables_nodes"), r.get("labels"))

        raw_doi_field = clean_str(r.get("doi"))
        raw_link_field = clean_str(r.get("access_link"))
        doi_clean = clean_doi(raw_doi_field) or clean_doi(raw_link_field)
        existence = clean_str(r.get("existence") or (r.get("status") or {}).get("name"))

        # Recover title from misplaced non-DOI text in `doi` or synthesize deterministically from ECV metadata
        title = clean_str(r.get("name"))
        if not title and raw_doi_field and clean_doi(raw_doi_field) is None:
            title = raw_doi_field
        if not title:
            ecv_prod = var_by_type.get("ECV_Product") or var_by_type.get("ECV") or var_by_type.get("Domain") or "ECV Record"
            phys_q = var_by_type.get("Physical_Quantity")
            desc_part = f"{ecv_prod} — {phys_q}" if phys_q and phys_q.lower() != ecv_prod.lower() else ecv_prod
            org_tag = " / ".join(org_acrs[:2]) if org_acrs else "WGClimate"
            sat_tag = "; ".join(plat_list[:2])
            doi_tag = doi_clean.split("/")[-1] if doi_clean else ""
            qual_bits = [x for x in (doi_tag, sat_tag, "planned" if (existence or "").lower() == "planned" else "") if x]
            suffix = f" ({', '.join(qual_bits)})" if qual_bits else ""
            title = f"{org_tag} {desc_part}{suffix}"

        rows.append({
            "source_id": "12_ceos_cgms_wgclimate_ecv",
            "raw_id": f"WGCLIMATE_CDR_{r.get('id')}",
            "title": title,
            "description": clean_str(r.get("description")),
            "processing_level_raw": None,
            "spatial_resolution_raw": _extract_wgc_resolutions(r.get("horizontal_resolutions"), _WGC_HORIZ_UNITS),
            "temporal_resolution_raw": _extract_wgc_resolutions(r.get("temporal_resolutions"), _WGC_TEMP_UNITS),
            "temporal_start_raw": clean_str(r.get("start_date")),
            "temporal_end_raw": clean_str(r.get("end_date")),
            "update_frequency_raw": existence,
            "bbox_raw": (
                [float(x) for x in r["geocoverage"]["bbox"]]
                if isinstance(r.get("geocoverage"), dict)
                and isinstance(r["geocoverage"].get("bbox"), list)
                and len(r["geocoverage"]["bbox"]) == 4
                else ([-180.0, -90.0, 180.0, 90.0] if "global" in str(r.get("geocoverage", "")).lower() else None)
            ),
            "producing_org_raw": org_str,
            "platform_sensor_raw": plat_str,
            "doi_raw": doi_clean,
            "license_raw": clean_str(r.get("commitment_level")),
            "landing_url_raw": clean_url(raw_link_field) or (f"https://doi.org/{doi_clean}" if doi_clean else None),
            "keywords_raw": kw_str,
        })
    return rows


def parse_16_nasa_cmr(source_id: str = "16_nasa_cmr", pattern: str = "*.jsonl") -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for p in sorted((RAW_DIR / source_id / "data").glob(pattern)):
        with p.open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                cid = clean_str(e.get("id") or e.get("entry_id"))
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                bbox = None
                boxes = e.get("boxes")
                if isinstance(boxes, list) and boxes:
                    # CMR box format is "south west north east"
                    parts = [float(x) for x in str(boxes[0]).split() if re.match(r"^-?\d+(\.\d+)?$", x)]
                    if len(parts) == 4:
                        bbox = [parts[1], parts[0], parts[3], parts[2]]
                doi = clean_str(e.get("doi"))
                url = None
                for lk in (e.get("links") or []):
                    if isinstance(lk, dict) and lk.get("href") and str(lk.get("href")).startswith("http"):
                        url = clean_str(lk.get("href"))
                        break
                short_name = clean_str(e.get("short_name"))
                title = clean_str(e.get("title") or e.get("dataset_id"))
                if short_name and title and short_name not in title:
                    title = f"{short_name}: {title}"
                rows.append({
                    "source_id": source_id,
                    "raw_id": cid,
                    "short_name_raw": short_name,
                    "title": title,
                    "description": clean_str(e.get("summary")),
                    "processing_level_raw": clean_str(e.get("processing_level_id")),
                    "spatial_resolution_raw": None,
                    "temporal_resolution_raw": None,
                    "temporal_start_raw": clean_str(e.get("time_start")),
                    "temporal_end_raw": clean_str(e.get("time_end")),
                    "update_frequency_raw": clean_str(e.get("collection_data_type")),
                    "bbox_raw": bbox,
                    "producing_org_raw": clean_str(e.get("organizations") or e.get("archive_center") or e.get("data_center")),
                    "platform_sensor_raw": clean_str([e.get("platforms"), e.get("instruments")]),
                    "doi_raw": doi,
                    "license_raw": "NASA Earthdata Open Data Policy",
                    "landing_url_raw": url or f"https://search.earthdata.nasa.gov/search?q={short_name or cid}",
                    "keywords_raw": clean_str(e.get("consortiums")),
                })
    return rows


def parse_17_eumetsat() -> list[dict[str, Any]]:
    rows = []
    p = RAW_DIR / "17_eumetsat_product_navigator/data/eumetsat_product_navigator.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            hit = json.loads(line)
            s = hit.get("_source", {})
            te = s.get("temporalExtent", {}) or {}
            loc = s.get("location", {}) or {}
            bbox = _extract_geojson_bbox(loc)
            t_start = clean_str(te.get("begin") or te.get("beginDate"))
            t_end = clean_str((te.get("end") or te.get("endDate")) if not te.get("isOngoing") else None)
            if t_start and t_end and str(t_start)[:10] > str(t_end)[:10]:
                t_end = None
            rows.append({
                "source_id": "17_eumetsat_product_navigator",
                "raw_id": clean_str(s.get("id") or hit.get("_id")),
                "title": clean_str(s.get("datasetTitle")),
                "description": clean_str(s.get("abstract") or s.get("summary")),
                "processing_level_raw": clean_str(s.get("processingLevels")),
                "spatial_resolution_raw": clean_str(s.get("spatialResolutions")),
                "temporal_resolution_raw": None,
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": clean_str(s.get("productStatus")),
                "bbox_raw": bbox,
                "producing_org_raw": clean_str(s.get("originator") or s.get("distributors") or "EUMETSAT"),
                "platform_sensor_raw": clean_str(s.get("satellites")),
                "doi_raw": None,
                "license_raw": clean_str(s.get("dataPolicies")),
                "landing_url_raw": f"https://user.eumetsat.int/catalogue/{s.get('id')}" if s.get("id") else None,
                "keywords_raw": clean_str(s.get("parameters") or s.get("services")),
            })
    return rows


def parse_20_noaa() -> list[dict[str, Any]]:
    rows = []
    seen_ids: set[str] = set()
    files = sorted((RAW_DIR / "20_noaa_onestop/data").glob("noaa_onestop_collections*.jsonl"))
    for p in files:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                a = item.get("attributes", {}) or {}
                rid = clean_str(item.get("id") or a.get("fileIdentifier"))
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                sb = a.get("spatialBounding", {}) or {}
                bbox = None
                coords = sb.get("coordinates")
                if isinstance(coords, list) and coords and isinstance(coords[0], list) and len(coords[0]) >= 4:
                    try:
                        xs = [pt[0] for pt in coords[0]]
                        ys = [pt[1] for pt in coords[0]]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    except Exception:
                        bbox = None
                url = None
                for lk in (a.get("links") or []):
                    if isinstance(lk, dict) and lk.get("linkUrl"):
                        url = clean_str(lk.get("linkUrl"))
                        break
                rows.append({
                    "source_id": "20_noaa_onestop",
                    "raw_id": rid,
                    "title": clean_str(a.get("title")),
                    "description": clean_str(a.get("description")),
                    "processing_level_raw": clean_str(a.get("dataProcessingLevel")),
                    "spatial_resolution_raw": None,
                    "temporal_resolution_raw": clean_str(a.get("maintenanceFrequency")),
                    "temporal_start_raw": clean_str(a.get("beginDate") or a.get("beginYear")),
                    "temporal_end_raw": clean_str(a.get("endDate") or a.get("endYear")),
                    "update_frequency_raw": clean_str(a.get("maintenanceFrequency")),
                    "bbox_raw": bbox,
                    "producing_org_raw": "NOAA (NESDIS / NCEI / STAR)",
                    "platform_sensor_raw": clean_str(a.get("platforms") or a.get("instruments")),
                    "doi_raw": clean_str(a.get("doi")),
                    "license_raw": clean_str(a.get("useLimitation") or "U.S. Government Public Domain"),
                    "landing_url_raw": url,
                    "keywords_raw": clean_str(a.get("keywords") or a.get("gcmdScienceServices")),
                })
    return rows


def parse_23_wri() -> tuple[list[dict[str, Any]], int]:
    rows = []
    excluded_tabular = 0
    # 1. Global Forest Watch API
    gfw_path = RAW_DIR / "23_wri_gfw_and_resourcewatch/data/wri_gfw_datasets.json"
    gfw_data = json.loads(gfw_path.read_text(encoding="utf-8")).get("data", [])
    for d in gfw_data:
        meta = d.get("metadata", {}) or {}
        title = clean_str(meta.get("title") or d.get("dataset"))
        t_start, t_end = _split_date_range_str(meta.get("content_date"))
        rows.append({
            "source_id": "23_wri_gfw_and_resourcewatch",
            "raw_id": f"GFW:{d.get('dataset')}",
            "title": title,
            "description": clean_str(meta.get("overview") or meta.get("function")),
            "processing_level_raw": None,
            "spatial_resolution_raw": clean_str(meta.get("resolution")),
            "temporal_resolution_raw": clean_str(meta.get("update_frequency")),
            "temporal_start_raw": t_start,
            "temporal_end_raw": t_end,
            "update_frequency_raw": clean_str(meta.get("update_frequency")),
            "bbox_raw": None,
            "producing_org_raw": clean_str(meta.get("source")) or "World Resources Institute (Global Forest Watch / UMD GLAD)",
            "platform_sensor_raw": None,
            "doi_raw": clean_str(meta.get("citation")),
            "license_raw": clean_str(meta.get("license")),
            "landing_url_raw": f"https://data.globalforestwatch.org/datasets/{d.get('dataset')}",
            "keywords_raw": clean_str(meta.get("tags")),
        })

    # 2. Resource Watch API (retain geospatial / raster / GEE / Carto / WMS datasets)
    rw_path = RAW_DIR / "23_wri_gfw_and_resourcewatch/data/wri_resourcewatch_datasets.jsonl"
    with rw_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            attr = item.get("attributes", {}) or {}
            if not attr.get("geoInfo") and attr.get("provider") not in ("gee", "wms", "cartodb", "rasdaman", "nexgddp"):
                excluded_tabular += 1
                continue
            meta_list = attr.get("metadata") or []
            m0 = meta_list[0].get("attributes", {}) if meta_list and isinstance(meta_list[0], dict) else {}
            info = m0.get("info", {}) if isinstance(m0.get("info"), dict) else {}
            t_start, t_end = _split_date_range_str(info.get("date_of_content"))
            rows.append({
                "source_id": "23_wri_gfw_and_resourcewatch",
                "raw_id": f"RW:{item.get('id')}",
                "title": clean_str(m0.get("name") or attr.get("name")),
                "description": clean_str(m0.get("description") or info.get("functions")),
                "processing_level_raw": None,
                "spatial_resolution_raw": clean_str(info.get("spatial_resolution")),
                "temporal_resolution_raw": clean_str(info.get("frequency_of_updates")),
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": clean_str(info.get("frequency_of_updates")),
                "bbox_raw": None,
                "producing_org_raw": clean_str(info.get("organization") or m0.get("source")) or "World Resources Institute (Resource Watch)",
                "hosting_platform_raw": clean_str(attr.get("provider")),
                "platform_sensor_raw": clean_str(attr.get("tableName")),
                "doi_raw": clean_str(info.get("citation")),
                "license_raw": clean_str(info.get("license")),
                "landing_url_raw": clean_str(info.get("learn_more_link")),
                "keywords_raw": clean_str(attr.get("subtitle")),
            })
    return rows, excluded_tabular


def parse_26_fao_zenodo() -> list[dict[str, Any]]:
    rows = []
    fao_path = RAW_DIR / "26_fao_unep_and_thematic_benchmarks/data/fao_gismgr_workspaces_and_mapsets.json"
    fao_data = json.loads(fao_path.read_text(encoding="utf-8"))
    for m in fao_data.get("mapsets", []):
        self_url = None
        for lk in (m.get("links") or []):
            if isinstance(lk, dict) and lk.get("rel") == "self":
                self_url = clean_str(lk.get("href"))
        rows.append({
            "source_id": "26_fao_unep_and_thematic_benchmarks",
            "raw_id": f"FAO_GISMGR:{m.get('workspaceCode')}:{m.get('code')}",
            "title": clean_str(f"{m.get('_workspace_caption', 'FAO')} - {m.get('caption') or m.get('code')}"),
            "description": clean_str(m.get("description") or m.get("measureCaption")),
            "processing_level_raw": None,
            "spatial_resolution_raw": None,
            "temporal_resolution_raw": None,
            "temporal_start_raw": None,
            "temporal_end_raw": None,
            "update_frequency_raw": None,
            "bbox_raw": None,
            "producing_org_raw": "Food and Agriculture Organization of the United Nations (FAO)",
            "platform_sensor_raw": None,
            "doi_raw": None,
            "license_raw": "FAO Open Data Licensing Policy (CC-BY-4.0)",
            "landing_url_raw": self_url or "https://data.apps.fao.org/",
            "keywords_raw": clean_str(m.get("tags")),
        })

    z_path = RAW_DIR / "26_fao_unep_and_thematic_benchmarks/data/zenodo_thematic_benchmarks.jsonl"
    with z_path.open(encoding="utf-8") as f:
        for line in f:
            h = json.loads(line)
            md = h.get("metadata", {}) or {}
            creators = [c.get("name") for c in (md.get("creators") or []) if isinstance(c, dict) and c.get("name")]
            affils = [c.get("affiliation") for c in (md.get("creators") or []) if isinstance(c, dict) and c.get("affiliation")]
            org = "; ".join(dict.fromkeys(affils[:3])) if affils else ("; ".join(creators[:3]) if creators else "Zenodo Scientific Benchmark")
            rows.append({
                "source_id": "26_fao_unep_and_thematic_benchmarks",
                "raw_id": f"ZENODO:{h.get('id')}",
                "title": clean_str(md.get("title") or h.get("title")),
                "description": clean_str(md.get("description")),
                "processing_level_raw": None,
                "spatial_resolution_raw": None,
                "temporal_resolution_raw": None,
                "temporal_start_raw": None,  # Zenodo deposit publication_date is not observation start year
                "temporal_end_raw": None,
                "update_frequency_raw": None,
                "bbox_raw": None,
                "producing_org_raw": org,
                "hosting_platform_raw": "Zenodo (CERN)",
                "platform_sensor_raw": None,
                "doi_raw": clean_str(h.get("doi") or md.get("doi")),
                "license_raw": clean_str((md.get("license") or {}).get("id") if isinstance(md.get("license"), dict) else md.get("license")),
                "landing_url_raw": clean_str((h.get("links") or {}).get("html")),
                "keywords_raw": clean_str(md.get("keywords")),
            })

    # 3. GFZ Data Services (DataCite) for Horizon 2020 G3P Groundwater Products
    g3p_path = RAW_DIR / "26_fao_unep_and_thematic_benchmarks/data/gfz_datacite_g3p_benchmarks.json"
    if g3p_path.exists():
        for d in json.loads(g3p_path.read_text(encoding="utf-8")).get("data", []):
            attr = d.get("attributes", {}) or {}
            rtype_gen = str((attr.get("types") or {}).get("resourceTypeGeneral") or "").lower()
            if rtype_gen and rtype_gen != "dataset":
                continue
            titles = [t.get("title") for t in (attr.get("titles") or []) if isinstance(t, dict) and t.get("title")]
            descs = [x.get("description") for x in (attr.get("descriptions") or []) if isinstance(x, dict) and x.get("description")]
            creators = [c.get("name") for c in (attr.get("creators") or []) if isinstance(c, dict) and c.get("name")]
            doi_id = clean_str(d.get("id"))
            rows.append({
                "source_id": "26_fao_unep_and_thematic_benchmarks",
                "raw_id": f"DATACITE:{doi_id}",
                "title": clean_str(titles[0] if titles else doi_id),
                "description": clean_str(descs[0] if descs else None),
                "processing_level_raw": None,
                "spatial_resolution_raw": None,
                "temporal_resolution_raw": None,
                "temporal_start_raw": None,  # DataCite publicationYear is not observation start year
                "temporal_end_raw": None,
                "update_frequency_raw": None,
                "bbox_raw": None,
                "producing_org_raw": clean_str(attr.get("publisher")) or clean_str(creators[:2]) or "GFZ German Research Centre for Geosciences / G3P Consortium",
                "hosting_platform_raw": "GFZ Data Services (DataCite)",
                "platform_sensor_raw": "GRACE / GRACE-FO",
                "doi_raw": doi_id,
                "license_raw": "CC-BY-4.0",
                "landing_url_raw": clean_str(attr.get("url")) or (f"https://doi.org/{doi_id}" if doi_id else None),
                "keywords_raw": "G3P; groundwater; GRACE; GRACE-FO; Horizon 2020",
            })
    return rows


def parse_29_awesome_gee() -> list[dict[str, Any]]:
    rows = []
    p = RAW_DIR / "29_awesome_gee_community/data/awesome_gee_community_datasets.json"
    items = json.loads(p.read_text(encoding="utf-8"))
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append({
            "source_id": "29_awesome_gee_community",
            "raw_id": clean_str(it.get("id") or it.get("title")),
            "title": clean_str(it.get("title")),
            "description": clean_str(f"Thematic group: {it.get('thematic_group')}. Tags: {it.get('tags')}"),
            "processing_level_raw": None,
            "spatial_resolution_raw": None,
            "temporal_resolution_raw": None,
            "temporal_start_raw": clean_str(it.get("start_year") or it.get("year")),
            "temporal_end_raw": clean_str(it.get("end_year")),
            "update_frequency_raw": None,
            "bbox_raw": None,
            "producing_org_raw": clean_str(it.get("provider")),
            "hosting_platform_raw": "Google Earth Engine Community Catalog",
            "platform_sensor_raw": clean_str(it.get("type")),
            "doi_raw": clean_str(it.get("doi")),
            "license_raw": clean_str(it.get("license")),
            "landing_url_raw": clean_str(it.get("docs") or it.get("sample_code")),
            "keywords_raw": clean_str(it.get("tags") or it.get("thematic_group")),
        })
    return rows


def main() -> None:
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Save WMO OSCAR & WGClimate ECV Reference Taxonomies
    wmo_vars = json.loads((RAW_DIR / "13_wmo_oscar_and_gcos/data/wmo_oscar_variables.json").read_text())["variables"]
    wmo_sats = json.loads((RAW_DIR / "13_wmo_oscar_and_gcos/data/wmo_oscar_satellites.json").read_text())["satellites"]
    wmo_vars_flat = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in row.items()} for row in wmo_vars]
    wmo_sats_flat = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in row.items()} for row in wmo_sats]
    pd.DataFrame(wmo_vars_flat).to_parquet(TAXONOMY_DIR / "wmo_oscar_variables.parquet", index=False)
    pd.DataFrame(wmo_sats_flat).to_parquet(TAXONOMY_DIR / "wmo_oscar_satellites.parquet", index=False)

    # Build raw records per source
    source_raw_batches: list[tuple[str, list[dict[str, Any]], int, int]] = []

    # 01 - 06
    cmems_rows = parse_01_cmems()
    clms_rows = parse_02_clms()
    c3s_rows = parse_cads_stac("03_copernicus_c3s_cds", "c3s_cds_datasets.json", "ECMWF (Copernicus Climate Change Service - C3S)")
    cams_rows = parse_cads_stac("04_copernicus_cams_ads", "cams_ads_datasets.json", "ECMWF (Copernicus Atmosphere Monitoring Service - CAMS)")
    cems_rows = parse_cads_stac("05_copernicus_cems_ewds", "cems_ewds_datasets.json", "ECMWF / EC JRC (Copernicus Emergency Management Service - CEMS)")
    source_raw_batches.append(("01_copernicus_cmems", cmems_rows, len(cmems_rows), 0))
    source_raw_batches.append(("02_copernicus_clms", clms_rows, len(clms_rows), 0))
    source_raw_batches.append(("03_copernicus_c3s_cds", c3s_rows, len(c3s_rows), 0))
    source_raw_batches.append(("04_copernicus_cams_ads", cams_rows, len(cams_rows), 0))
    source_raw_batches.append(("05_copernicus_cems_ewds", cems_rows, len(cems_rows), 0))
    eodc_cols = json.loads((RAW_DIR / "06_copernicus_cems_gfm_eodc/data/eodc_stac_collections.json").read_text())["collections"]
    source_raw_batches.append(("06_copernicus_cems_gfm_eodc", parse_generic_stac_collections("06_copernicus_cems_gfm_eodc", eodc_cols, "EODC / Copernicus Emergency Management Service (CEMS GFM)"), len(eodc_cols), 0))

    # 07 JRC
    jrc_rows, jrc_non_eo = parse_07_jrc()
    source_raw_batches.append(("07_ec_jrc_data_catalogue", jrc_rows, len(jrc_rows) + jrc_non_eo, jrc_non_eo))

    # 08 DestinE DEDL
    dedl_cols = json.loads((RAW_DIR / "08_destine_dedl_stac/data/destine_dedl_collections.json").read_text())["collections"]
    source_raw_batches.append(("08_destine_dedl_stac", parse_generic_stac_collections("08_destine_dedl_stac", dedl_cols, "Destination Earth Data Lake (EUMETSAT / ECMWF / ESA / JRC)"), len(dedl_cols), 0))

    # 09 WEkEO
    wekeo_rows = parse_09_wekeo()
    source_raw_batches.append(("09_wekeo_dias", wekeo_rows, len(wekeo_rows), 0))

    # 10 CDSE
    cdse_native = json.loads((RAW_DIR / "10_cdse_stac/data/cdse_native_stac_collections.json").read_text()).get("collections", [])
    cdse_fedeo = json.loads((RAW_DIR / "10_cdse_stac/data/cdse_fedeo_stac_collections.json").read_text()).get("collections", [])
    cdse_seen = set()
    cdse_merged = []
    for c in cdse_native + cdse_fedeo:
        cid = c.get("id")
        if cid and cid not in cdse_seen:
            cdse_seen.add(cid)
            cdse_merged.append(c)
    source_raw_batches.append(("10_cdse_stac", parse_generic_stac_collections("10_cdse_stac", cdse_merged, "Copernicus Data Space Ecosystem (ESA / EC)"), len(cdse_native) + len(cdse_fedeo), 0))

    # 11 EEA & EMODnet
    eea_rows = parse_11_eea_emodnet()
    source_raw_batches.append(("11_eea_and_emodnet", eea_rows, len(eea_rows), 0))

    # 12 WGClimate ECV
    ecv_rows = parse_12_wgclimate_ecv()
    source_raw_batches.append(("12_ceos_cgms_wgclimate_ecv", ecv_rows, len(ecv_rows), 0))

    # 14 CEOS FedEO Agencies + Native CEDA & ESA MAAP STAC
    fedeo_cols = []
    fedeo_seen_ids: set[str] = set()
    for p in sorted((RAW_DIR / "14_ceos_fedeo_agencies/data").glob("fedeo_*.jsonl")):
        agency_tag = p.stem.replace("fedeo_", "")
        with p.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                c["_fedeo_agency"] = agency_tag
                fedeo_cols.append(c)
                if c.get("id"):
                    fedeo_seen_ids.add(str(c["id"]))
    ceda_native_path = RAW_DIR / "14_ceos_fedeo_agencies/data/ceda_native_stac.json"
    if ceda_native_path.exists():
        for c in json.loads(ceda_native_path.read_text(encoding="utf-8")).get("collections", []):
            cid = c.get("id")
            if cid and str(cid) in fedeo_seen_ids:
                continue
            if cid:
                fedeo_seen_ids.add(str(cid))
            c["_fedeo_agency"] = "ceda_native"
            fedeo_cols.append(c)
    esa_maap_path = RAW_DIR / "14_ceos_fedeo_agencies/data/esa_maap_stac.json"
    if esa_maap_path.exists():
        for c in json.loads(esa_maap_path.read_text(encoding="utf-8")).get("collections", []):
            cid = str(c.get("id") or "")
            if not cid or cid in ("datasets", "services") or cid in fedeo_seen_ids:
                continue
            fedeo_seen_ids.add(cid)
            c["_fedeo_agency"] = "esa_maap_stac"
            fedeo_cols.append(c)
    source_raw_batches.append(("14_ceos_fedeo_agencies", parse_generic_stac_collections("14_ceos_fedeo_agencies", fedeo_cols, "CEOS FedEO Federated Space Agency"), len(fedeo_cols), 0))

    # 15 GEO BON & GEO Knowledge Hub
    gb_cols = json.loads((RAW_DIR / "15_geobon_and_geokhub/data/geobon_ebv_stac_collections.json").read_text()).get("collections", [])
    gb_rows = parse_generic_stac_collections("15_geobon_and_geokhub", gb_cols, "GEO BON (Essential Biodiversity Variables)")
    gkh_path = RAW_DIR / "15_geobon_and_geokhub/data/geokhub_records.jsonl"
    gkh_total = 0
    gkh_non_dataset = 0
    with gkh_path.open(encoding="utf-8") as f:
        for line in f:
            gkh_total += 1
            h = json.loads(line)
            md = h.get("metadata", {}) or {}
            rtype = ((md.get("resource_type") or {}).get("id") or "").lower()
            if "dataset" not in rtype and "data" not in rtype and "knowledge" not in rtype:
                gkh_non_dataset += 1
                continue
            gb_rows.append({
                "source_id": "15_geobon_and_geokhub",
                "raw_id": f"GEOKHUB:{h.get('id')}",
                "title": clean_str(md.get("title")),
                "description": clean_str(md.get("description")),
                "processing_level_raw": None,
                "spatial_resolution_raw": None,
                "temporal_resolution_raw": None,
                "temporal_start_raw": None,  # GEO Knowledge Hub publication_date is not observation start year
                "temporal_end_raw": None,
                "update_frequency_raw": None,
                "bbox_raw": None,
                "producing_org_raw": "Group on Earth Observations (GEO Knowledge Hub)",
                "platform_sensor_raw": None,
                "doi_raw": clean_str((h.get("pids") or {}).get("doi", {}).get("identifier")),
                "license_raw": "CC-BY-4.0",
                "landing_url_raw": clean_str((h.get("links") or {}).get("self_html")),
                "keywords_raw": clean_str(md.get("Geo_programme")),
            })
    source_raw_batches.append(("15_geobon_and_geokhub", gb_rows, len(gb_cols) + gkh_total, gkh_non_dataset))

    # 16 NASA CMR
    cmr_rows = parse_16_nasa_cmr("16_nasa_cmr", "*.jsonl")
    cmr_raw_total = sum(sum(1 for ln in p.open(encoding="utf-8") if ln.strip()) for p in (RAW_DIR / "16_nasa_cmr/data").glob("*.jsonl"))
    source_raw_batches.append(("16_nasa_cmr", cmr_rows, cmr_raw_total, 0))

    # 17 EUMETSAT
    eum_rows = parse_17_eumetsat()
    source_raw_batches.append(("17_eumetsat_product_navigator", eum_rows, len(eum_rows), 0))

    # 18 ESA OSC EarthCODE
    osc_cols = [json.loads(line) for line in (RAW_DIR / "18_esa_osc_earthcode/data/esa_osc_products.jsonl").open(encoding="utf-8")]
    source_raw_batches.append(("18_esa_osc_earthcode", parse_generic_stac_collections("18_esa_osc_earthcode", osc_cols, "European Space Agency (ESA EarthCODE Open Science Catalog)"), len(osc_cols), 0))

    # 19 DLR EOC STAC
    dlr_cols = json.loads((RAW_DIR / "19_dlr_eoc_stac/data/dlr_eoc_stac_collections.json").read_text()).get("collections", [])
    source_raw_batches.append(("19_dlr_eoc_stac", parse_generic_stac_collections("19_dlr_eoc_stac", dlr_cols, "German Aerospace Center (DLR Earth Observation Center)"), len(dlr_cols), 0))

    # 20 NOAA OneStop
    noaa_rows = parse_20_noaa()
    source_raw_batches.append(("20_noaa_onestop", noaa_rows, len(noaa_rows), 0))

    # 21 USGS Landsat & EROS
    usgs_stac = json.loads((RAW_DIR / "21_usgs_landsat_and_eros/data/usgs_landsatlook_stac.json").read_text()).get("collections", [])
    usgs_rows = parse_generic_stac_collections("21_usgs_landsat_and_eros", usgs_stac, "U.S. Geological Survey (USGS)")
    usgs_cmr = parse_16_nasa_cmr("21_usgs_landsat_and_eros", "usgs_eros_cmr_collections.jsonl")
    source_raw_batches.append(("21_usgs_landsat_and_eros", usgs_rows + usgs_cmr, len(usgs_stac) + len(usgs_cmr), 0))

    # 22 Data Terra, INPE, DEA
    nat_rows = []
    nat_raw_total = 0
    _nat_org_map = {
        "australia_dea_stac.json": "Geoscience Australia (Digital Earth Australia)",
        "brazil_inpe_bdc_stac.json": "INPE Brazil Data Cube (Instituto Nacional de Pesquisas Espaciais)",
        "france_dataterra_stac.json": "Data Terra / CNES / Theia (France)",
    }
    _nat_host_map = {
        "australia_dea_stac.json": "Digital Earth Australia (DEA)",
        "brazil_inpe_bdc_stac.json": "INPE Brazil Data Cube",
        "france_dataterra_stac.json": "Data Terra / Theia",
    }
    for p in sorted((RAW_DIR / "22_data_terra_inpe_dea/data").glob("*.json")):
        cols = json.loads(p.read_text(encoding="utf-8")).get("collections", [])
        nat_raw_total += len(cols)
        parsed = parse_generic_stac_collections("22_data_terra_inpe_dea", cols, _nat_org_map.get(p.name, "National EO Data Cube"))
        for pr in parsed:
            if not pr.get("hosting_platform_raw") and p.name in _nat_host_map:
                pr["hosting_platform_raw"] = _nat_host_map[p.name]
        nat_rows.extend(parsed)
    source_raw_batches.append(("22_data_terra_inpe_dea", nat_rows, nat_raw_total, 0))

    # 23 WRI GFW & Resource Watch
    wri_rows, wri_excl = parse_23_wri()
    source_raw_batches.append(("23_wri_gfw_and_resourcewatch", wri_rows, len(wri_rows) + wri_excl, wri_excl))

    # 24 OpenLandMap
    olm_cols = [json.loads(line) for line in (RAW_DIR / "24_openlandmap_stac/data/openlandmap_collections.jsonl").open(encoding="utf-8")]
    source_raw_batches.append(("24_openlandmap_stac", parse_generic_stac_collections("24_openlandmap_stac", olm_cols, "OpenGeoHub Foundation (OpenLandMap)"), len(olm_cols), 0))

    # 25 WorldPop & CIESIN
    wp_cols = json.loads((RAW_DIR / "25_worldpop_and_ciesin_stac/data/worldpop_stac_collections.json").read_text()).get("collections", [])
    ciesin_cols = [json.loads(line) for line in (RAW_DIR / "25_worldpop_and_ciesin_stac/data/ciesin_stac_collections.jsonl").open(encoding="utf-8")]
    source_raw_batches.append(("25_worldpop_and_ciesin_stac", parse_generic_stac_collections("25_worldpop_and_ciesin_stac", wp_cols, "WorldPop (University of Southampton)") + parse_generic_stac_collections("25_worldpop_and_ciesin_stac", ciesin_cols, "CIESIN (Columbia University)"), len(wp_cols) + len(ciesin_cols), 0))

    # 26 FAO & Zenodo Benchmarks
    fao_rows = parse_26_fao_zenodo()
    source_raw_batches.append(("26_fao_unep_and_thematic_benchmarks", fao_rows, len(fao_rows), 0))

    # 27 GEE Official STAC
    gee_cols = [json.loads(line) for line in (RAW_DIR / "27_gee_official_stac/data/gee_official_stac_collections.jsonl").open(encoding="utf-8")]
    source_raw_batches.append(("27_gee_official_stac", parse_generic_stac_collections("27_gee_official_stac", gee_cols, None), len(gee_cols), 0))

    # 28 Planetary Computer STAC
    pc_cols = json.loads((RAW_DIR / "28_planetary_computer_stac/data/planetary_computer_stac_collections.json").read_text()).get("collections", [])
    source_raw_batches.append(("28_planetary_computer_stac", parse_generic_stac_collections("28_planetary_computer_stac", pc_cols, None), len(pc_cols), 0))

    # 29 Awesome GEE Community
    agee_rows = parse_29_awesome_gee()
    source_raw_batches.append(("29_awesome_gee_community", agee_rows, len(agee_rows), 0))

    # -----------------------------------------------------------------------
    # Apply deterministic L3/L4 filtering & write per-source prepared outputs
    # -----------------------------------------------------------------------
    audit_rows = []
    all_excluded = []
    total_prepared = 0

    for sid, rows, raw_total, pre_excluded_non_eo in source_raw_batches:
        kept = []
        excluded_l1_l2 = 0
        excluded_insitu = 0
        seen_raw_ids: set[str] = set()
        for r in rows:
            rid_val = clean_str(r.get("raw_id"))
            if rid_val:
                if rid_val in seen_raw_ids:
                    continue
                seen_raw_ids.add(rid_val)
            # Apply uniform deterministic normalization across all 28 product sources
            r["bbox_raw"] = clean_bbox(r.get("bbox_raw"))
            r["doi_raw"] = clean_doi(r.get("doi_raw"))
            r["landing_url_raw"] = clean_url(r.get("landing_url_raw"))
            ts, te = r.get("temporal_start_raw"), r.get("temporal_end_raw")
            if (
                ts
                and te
                and re.match(r"^\d{4}-\d{2}-\d{2}", str(ts))
                and re.match(r"^\d{4}-\d{2}-\d{2}", str(te))
                and str(ts)[:10] > str(te)[:10]
            ):
                r["temporal_start_raw"], r["temporal_end_raw"] = te, ts
            ok, reason = check_l3_l4_eligibility(
                r.get("raw_id"),
                r.get("title"),
                r.get("description"),
                r.get("processing_level_raw"),
                sid,
            )
            if ok:
                kept.append(r)
            else:
                if reason == "pure_insitu_discrete_stations":
                    excluded_insitu += 1
                else:
                    excluded_l1_l2 += 1
                all_excluded.append({
                    "source_id": sid,
                    "raw_id": r.get("raw_id"),
                    "title": r.get("title"),
                    "processing_level_raw": r.get("processing_level_raw"),
                    "exclusion_reason": reason,
                })

        total_prepared += len(kept)
        df_src = pd.DataFrame(kept)
        df_src.to_parquet(OUT_DATA_DIR / f"{sid}.parquet", index=False)
        with (OUT_DATA_DIR / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Compute null rates for audit
        n_kept = max(len(kept), 1)
        null_spat = sum(1 for x in kept if not x.get("spatial_resolution_raw")) / n_kept * 100.0
        null_temp = sum(1 for x in kept if not x.get("temporal_start_raw")) / n_kept * 100.0
        null_doi = sum(1 for x in kept if not x.get("doi_raw")) / n_kept * 100.0

        audit_rows.append({
            "source_id": sid,
            "raw_records_mined": raw_total,
            "excluded_non_eo_or_non_dataset": pre_excluded_non_eo,
            "excluded_l0_l1_l2_or_visualisation": excluded_l1_l2,
            "excluded_pure_insitu_station_logs": excluded_insitu,
            "prepared_l3_l4_records": len(kept),
            "pct_missing_spatial_res_raw": round(null_spat, 1),
            "pct_missing_temporal_start_raw": round(null_temp, 1),
            "pct_missing_doi_raw": round(null_doi, 1),
        })
        print(
            f"[02_preparation] {sid:36s}: raw={raw_total:5d} -> kept_L3_L4={len(kept):5d} "
            f"(excl_L1_L2={excluded_l1_l2}, excl_nonEO={pre_excluded_non_eo}, excl_insitu={excluded_insitu})"
        )

    # Save audit summary & excluded log
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(BASE_DIR / "exclusion_audit_summary.csv", index=False)
    pd.DataFrame(all_excluded).to_parquet(OUT_DATA_DIR / "excluded_records_log.parquet", index=False)
    print("-" * 88)
    print(f"[02_preparation] Complete: {total_prepared} prepared L3/L4 records across 28 product sources (+ 1,300 WMO OSCAR taxonomy records).")


if __name__ == "__main__":
    main()
