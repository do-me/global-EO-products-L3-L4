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
Deterministic preprocessing, structural normalization, and L3/L4 filtering across all 32 raw EO metadata sources.

Key Scientific & Methodological Principles:
1. Missing Metadata Kept Null: Never manually impute or invent missing metadata. If a field is absent
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

from html.parser import HTMLParser
import html
import json
import math
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
def strict_json_sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf/NA/numpy scalars with strict RFC 8259 JSON values."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "tolist") and not isinstance(obj, (dict, list, tuple)):
        return strict_json_sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): strict_json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json_sanitize(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if hasattr(obj, "item"):
        return strict_json_sanitize(obj.item())
    return str(obj)


_KNOWN_HTML_TAGS_PATTERN = (
    r"p|div|br|ul|ol|li|dl|dt|dd|strong|em|b|i|u|s|strike|span|a|"
    r"h[1-6]|table|thead|tbody|tfoot|tr|td|th|caption|"
    r"sup|sub|blockquote|pre|code|hr|img|font|center|style|script|"
    r"abbr|cite|del|ins|mark|tt|small|figure|figcaption|details|summary|"
    r"section|article|aside|header|footer|nav|html|head|body|href|iframe|noscript"
)
_ESCAPED_HTML_TAG_RE = re.compile(
    rf"&lt;(/?(?:{_KNOWN_HTML_TAGS_PATTERN})\b(?:(?!(?:&lt;|&gt;)).)*?)&gt;",
    re.IGNORECASE | re.DOTALL,
)
_VALID_HTML_OR_COMMENT_RE = re.compile(
    rf"<!--[\s\S]*?-->|<![A-Za-z][^<>]*>|<\?[\s\S]*?\?>|"
    rf"</?(?:{_KNOWN_HTML_TAGS_PATTERN})(?=[\s/>])(?:\"[^<>\"]*\"|'[^<>']*'|[^<>])*?>",
    re.IGNORECASE,
)
_BLOCK_HTML_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "dl", "dt", "dd", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr", "section", "article",
    "header", "footer", "aside", "nav", "figure", "figcaption", "table", "thead", "tbody", "tfoot",
}
_INLINE_HTML_TAGS = {
    "sub", "sup", "b", "strong", "i", "em", "u", "s", "strike", "small",
    "code", "font", "tt", "abbr", "cite", "del", "ins", "mark", "span", "a", "href",
}


class _SafeHTMLTextExtractor(HTMLParser):
    """HTML-spec tokenizer that strips markup tags while preserving comparison expressions (<2010, <5m) and URL query '&'."""

    def __init__(self, preserve_newlines: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.preserve_newlines = preserve_newlines
        self.parts: list[str] = []

    def _emit_tag_break(self, tag: str) -> None:
        t = tag.lower()
        if t in _INLINE_HTML_TAGS:
            return
        if self.preserve_newlines and t in _BLOCK_HTML_TAGS:
            self.parts.append("\n")
        else:
            self.parts.append(" ")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._emit_tag_break(tag)

    def handle_endtag(self, tag: str) -> None:
        self._emit_tag_break(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._emit_tag_break(tag)

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _escape_non_html_lt(s: str) -> str:
    """Escape '<' unless it starts a well-formed known HTML tag or comment/declaration."""
    out_parts: list[str] = []
    pos = 0
    while True:
        idx = s.find("<", pos)
        if idx == -1:
            out_parts.append(s[pos:])
            break
        out_parts.append(s[pos:idx])
        m = _VALID_HTML_OR_COMMENT_RE.match(s, idx)
        if m:
            out_parts.append(m.group(0))
            pos = m.end()
        else:
            out_parts.append("&lt;")
            pos = idx + 1
    return "".join(out_parts)


def strip_html_safely(raw_text: str, preserve_newlines: bool = False) -> str:
    """Strip HTML markup BEFORE unescaping comparison entities such as &lt;2010 or &lt;5 m."""
    s = raw_text
    # Decode only double-encoded &amp;lt;tag&gt; wrappers if present, never numeric/comparison &lt;2010
    if "&amp;lt;" in s and re.search(
        rf"&amp;lt;/?(?:{_KNOWN_HTML_TAGS_PATTERN})\b",
        s,
        re.IGNORECASE,
    ):
        s = s.replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;")
    if "&lt;" in s:
        s = _ESCAPED_HTML_TAG_RE.sub(lambda m: f"<{html.unescape(m.group(1))}>", s)
    # Shield non-tag literal '<' (e.g. 0<HotSpot<1, <https://...>, <product_type>, <5m) so HTMLParser never drops text
    if "<" in s:
        s = _escape_non_html_lt(s)
    parser = _SafeHTMLTextExtractor(preserve_newlines=preserve_newlines)
    try:
        parser.feed(s)
        parser.close()
        out = "".join(parser.parts)
    except Exception:
        out = s
    for _ in range(2):
        if "&" in out:
            out = html.unescape(out)
    return out.replace("\xa0", " ")


def clean_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, dict):
        for k in ("en", "langeng", "default", "value", "title", "name", "caption"):
            if k in val and val[k]:
                return clean_str(val[k])
        for v in val.values():
            if isinstance(v, str) and v.strip():
                return clean_str(v)
        return None
    if isinstance(val, list):
        parts = [clean_str(x) for x in val]
        parts = [p for p in parts if p]
        return "; ".join(dict.fromkeys(parts)) if parts else None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "n/a", "not provided", "unknown", "nan"):
        return None
    s = strip_html_safely(s, preserve_newlines=False)
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


_IMAGE_PREVIEW_URL_RE = re.compile(
    r"\.(?:png|jpe?g|gif|bmp|webp|svg|ico|kmz)(?:[?#]|$)|/browse[/?#]",
    re.IGNORECASE,
)


def is_image_preview_url(url: str | None) -> bool:
    if not url:
        return False
    return bool(_IMAGE_PREVIEW_URL_RE.search(url))


def clean_url(val: Any) -> str | None:
    """Normalize malformed URL schemes/markdown/prose wrappers to a clean http(s)/ftp URL or None."""
    s = clean_str(val)
    if not s:
        return None
    s = re.sub(r"^ftp:/([^/])", r"ftp://\1", s, flags=re.IGNORECASE)
    s = re.sub(r"^http:([a-zA-Z0-9])", r"https://\1", s, flags=re.IGNORECASE)
    m = re.search(r"((?:https?|ftp)://[^\s,;\"'<>)\]]+)", s, re.IGNORECASE)
    if m:
        candidate = m.group(1).rstrip(".,;")
        if is_image_preview_url(candidate):
            return None
        return candidate
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


def extract_stac_doi_provenance(rec: dict[str, Any]) -> dict[str, str | None]:
    """
    Extract verified primary dataset DOI and non-primary related DOIs (inputs, journal papers)
    from a STAC Collection without promoting `describedby`/`related` citations into `doi`.
    """
    explicit_doi = clean_doi(rec.get("sci:doi") or rec.get("doi"))
    doi = explicit_doi
    doi_source: str | None = "stac_explicit_field" if doi else None
    doi_relation: str | None = "is_primary_dataset" if doi else None
    related_dois: list[str] = []
    preferred_url: str | None = None
    fallback_url: str | None = None

    for pub in (rec.get("sci:publications") or []):
        if isinstance(pub, dict):
            pd_c = clean_doi(pub.get("doi") or pub.get("citation"))
            if pd_c and pd_c != doi:
                related_dois.append(pd_c)

    for lk in (rec.get("links") or []):
        if not isinstance(lk, dict):
            continue
        rel = str(lk.get("rel", "")).lower()
        href = clean_str(lk.get("href"))
        if not href:
            continue
        if "doi.org/" in href:
            d_href = clean_doi(href)
            if d_href:
                if not doi and rel in ("cite-as", "canonical"):
                    doi = d_href
                    doi_source = f"stac_link_{rel}"
                    doi_relation = "is_primary_dataset"
                elif d_href != doi:
                    related_dois.append(d_href)
        if href.startswith("http"):
            if rel in ("canonical", "about", "alternate") and "httpAccept=" not in href and not preferred_url:
                preferred_url = clean_url(href)
            elif rel == "describedby" and "catalogue.ceda.ac.uk/uuid/" in href and not preferred_url:
                preferred_url = clean_url(href)
            elif rel in ("canonical", "alternate", "about", "describedby", "self") and not fallback_url:
                fallback_url = clean_url(href)

    rel_str = "; ".join(dict.fromkeys(related_dois)) if related_dois else None
    return {
        "doi": doi,
        "url": preferred_url or fallback_url,
        "doi_source": doi_source,
        "doi_relation": doi_relation,
        "related_dois": rel_str,
    }


def extract_stac_doi_and_url(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    info = extract_stac_doi_provenance(rec)
    return info["doi"], info["url"]


def _load_ceda_doi_crosswalk() -> dict[str, str]:
    cw_path = RAW_DIR / "14_ceos_fedeo_agencies/data/ceda_datacite_doi_crosswalk.json"
    if not cw_path.exists():
        return {}
    try:
        data = json.loads(cw_path.read_text(encoding="utf-8")).get("by_uuid", {})
        return {str(k).strip().lower(): str(v["doi"]).strip() for k, v in data.items() if isinstance(v, dict) and v.get("doi")}
    except Exception:
        return {}


_CEDA_DOI_CROSSWALK = _load_ceda_doi_crosswalk()


# ---------------------------------------------------------------------------
# Deterministic L0/L1/L2 & Non-EO Exclusion Classifier
# ---------------------------------------------------------------------------
EXPLICIT_L0_L1_L2_TOKEN_RE = re.compile(
    r"^(?:Visualisation|Visualization|(?:level[- ]?|l)?[012](?:[abcdgprst]+|\.\d+)?\+?(?:\s+data)?)$",
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
    r"^COPERNICUS/S3/OLCI$|^COPERNICUS/S2_CLOUD_PROBABILITY$|^JAXA/ALOS/AVNIR-2/ORI$|"
    r"^LANDSAT/.*/C0[12]/T[12](?:_RT)?(?:_TOA)?$|^LANDSAT/GLS\d{4}(?:_L[57])?$|"
    r"^C12205679(?:06|07|69|85|86)-USGS_LTA$|^(?:AMZ1|CB4A?)-(?:MUX|WFI)-L4-SR-1$|"
    r"^MODIS/(?:006|061)/M[OY]D09G[AQ]$|^NOAA/VIIRS/.*/VNP09GA$|"
    r"^DEAFRICA:(?:ls[5789]_(?:sr|st)|landsat_c2l2_ar|s1_rtc|s2_l2a(?:_c1)?|s3_(?:ol|olci|slstr)_(?:2|l2)_.*|"
    r"s5p_tropomi_l2_.*|maxar_morocco_earthquake.*|fc_ls|wofs_ls)$|"
    r"^DEPACIFIC:dep_ls_(?:fc|wofl)$)",
    re.IGNORECASE,
)

SWATH_L1_L2_TITLE_PATTERNS = re.compile(
    r"(\b(level[- ]?(?:0[ab]?|1|1\.5|1[abcdgrt]|1gt|1gst|2|2[abcgprt])\b|l0[ab]?\b|l1[abcdgrt]?\b|l1gt\b|l1gst\b|l1\.5\b|l2[abcgprt]?\b)|"
    r"\b(single look complex|slc[- ]off|slc burst|ground range detected|top-of-atmosphere radiance|top of atmosphere reflectance|calibrated radiance|"
    r"raw scenes?|toa reflectance|raw instrument data|raw data record|\brdr\b|"
    r"(?:brightness|antenna|microwave|radiometer|sensor)\s+temperature data record|\btdr\b|sensor data record|\bsdr\b|"
    r"swath level 2|on the satellite swath|along-track re-tracked|l2 swath|swath grid|swath image|swath files|swath[- ]derived|swath elevation)\b)",
    re.IGNORECASE,
)

AERIAL_ORTHOPHOTO_PATTERNS = re.compile(
    r"(^naip$|^USDA/NAIP/|^USGS_DOQ$|^DOP_AUT_|^topo-dc-austria-op-utm$|^napl-|^banff-national-park$|"
    r"^(?:Estonia|Latvia|Germany)/.*/orthos/|^Slovakia/orthos/|^Netherlands/.*/LUCHTFOTO|^Spain/PNOA/|^Finland/(?:MAVI|SMK)/|"
    r"\b(?:digital orthophotos?|orthophotos?|digital orthophoto quadrangles?|naip: national agriculture imagery program|"
    r"national air photo library|aerial orthophoto|aerial photography)\b)",
    re.IGNORECASE,
)

# Multi-temporal composites, gridded syntheses, or L2-tagged CDRs that represent spatially gridded/composited EO products
COMPOSITE_OR_L3_OVERRIDE_PATTERNS = re.compile(
    r"(\b(?:level[- ]?[34][abcgsup]?\b(?!(?:[-_ ]?sr)?\s*-\s*cloud optimized)|l3[abcgsup]?\b|l4[a-d]?\b(?![-_]?5\b)(?!-sr-1\b)|l2 and l3\b|l2/3\b|"
    r"global mosaic|cloudless mosaic|monthly mosaic|quarterly mosaic|annual mosaic|"
    r"cloud-free annual composites?|s2mosaic|pixel based image composite|"
    r"8-day.*composite|10-day.*composite|16-day.*composite|32-day.*composite|monthly.*composite|annual.*composite|"
    r"10-day composites of msg|composites of msg land surface temperature|"
    r"harmonized landsat sentinel|hlsl30|hlss30|gweldmo|gweldyr|global web-enabled landsat|"
    r"daily synthesis|10-daily synthesis|1-day surface reflectance|10-day surface reflectance|"
    r"l2 statistics|gridded user friendly product|gravity field models and grids|"
    r"ghrsst level 2p gridded|ghrsst l2p gridded|data cube.*lcf \d+ (?:days|months)|"
    r"daily surface albedo|gsa level 2|10-day surface albedo|"
    r"fire radiative power grid|gridded fire radiative power|"
    r"daily shortwave solar irradiance|daily downward longwave irradiance|daily surface solar irradiance|"
    r"reference evapotranspiration|normalized difference vegetation index.*(?:cdr|climate data record)|"
    r"downward surface solar radiation.*daily mean|"
    r"lpj[-_]eosim.*simulated.*(?:daily|monthly).*flux|simulated.*(?:methane|carbon).*flux|"
    r"gridded population of the world.*grid|cygnss.*level 2.*surface.*flux.*climate data record)\b|"
    r"(?:syn-2-v10|syn-2-vg1|s3-syn-vgt|s3_syn_2_vg1|sentinel-3-synergy-v10|l2c-ndvi|l2-smos-sss-daily|"
    r"jaxa\.jasmes_gcom-c\.sgli_standard\.l2-lst|"
    r"gomos_ufp_gridded|goce_global_gravity_field_models_and_grids|clms_toc_global_300m_daily))",
    re.IGNORECASE,
)

SWATH_OR_UNGRIDDED_GUARD_RE = re.compile(
    r"\b(swath grid|swath level|on the satellite swath|along[- ]track|ungridded|/pixel\b|radiance|bending angle|sounder|sounding profiles?)\b",
    re.IGNORECASE,
)

# Non-sensor domain uses of "Level 1/2" (FAO WaPOR spatial scale tiers, TransCom 3 project tiers, GADM/NUTS admin levels, IUCN hierarchy)
NON_SENSOR_HIERARCHY_PATTERNS = re.compile(
    r"(^FAO_GISMGR:WAPOR|^FAO/WAPOR/|^DEAFRICA:wapor_soil_moisture$|transcom 3:|administrative level [0123]|global administrative unit|iucn_habitat|water resources zones: l[12]|river water level|lake water level)",
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
    has_explicit_l3_l4 = False
    if lvl:
        lvl_tokens = [t.strip().lower() for t in re.split(r"[;,/|]", lvl) if t.strip()]
        has_l3_l4 = any(
            re.match(r"^(?:level[- ]?|l)?[34][abcgsup]?$", t)
            or "level 3" in t
            or "level 4" in t
            for t in lvl_tokens
        )
        has_only_l0_l1_l2 = all(EXPLICIT_L0_L1_L2_TOKEN_RE.match(t) for t in lvl_tokens) and len(lvl_tokens) > 0
        has_l0_or_l1_or_vis = any(
            re.match(r"^(?:visuali[sz]ation|(?:level[- ]?|l)?(?:0|1|1\.5)(?:[abcdgrt]+|\.\d+)?)$", t)
            for t in lvl_tokens
        )
        if has_l3_l4 and not has_l0_or_l1_or_vis:
            has_explicit_l3_l4 = True
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

    # 4. Check swath/granule L1/L2 IDs or titles unless it is a non-sensor hierarchy, multi-temporal L3 composite, or explicit provider L3/L4
    if not is_non_sensor_hier and not is_composite_override:
        if SWATH_L1_L2_ID_PATTERNS.search(rid) and not has_explicit_l3_l4:
            return False, "swath_or_tile_l1_l2_id"
        if SWATH_L1_L2_TITLE_PATTERNS.search(ttl):
            if not has_explicit_l3_l4 or re.search(r"\b(swath|level[- ]?[012]|l[012][abcgprst]?\b)\b", ttl, re.IGNORECASE):
                return False, "swath_or_tile_l1_l2_title"

    return True, None


# ---------------------------------------------------------------------------
# Source-Specific Parsers (31 product sources + 1 taxonomy source)
# ---------------------------------------------------------------------------
def _extract_geojson_bbox(geom_obj: Any, primary_only: bool = False) -> list[float] | None:
    """Recursively extracts a validated WGS84 [west, south, east, north] bbox from GeoJSON geometries,
    preserving antimeridian-wrapping bounds [west_pos, south, east_neg, north] for dateline-split MultiPolygons."""
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

    # Check if geom_obj is a MultiPolygon or list of polygons split at the +/-180 antimeridian
    geom_type = str(geom_obj.get("type", "")).lower() if isinstance(geom_obj, dict) else ""
    coords_root = geom_obj.get("coordinates") if isinstance(geom_obj, dict) else None
    if geom_type == "multipolygon" and isinstance(coords_root, list) and len(coords_root) >= 2:
        poly_boxes: list[tuple[float, float, float, float]] = []
        for poly in coords_root:
            pxs: list[float] = []
            pys: list[float] = []

            def _walk_poly(c: Any) -> None:
                if isinstance(c, (list, tuple)):
                    if len(c) >= 2 and isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)) and not isinstance(c[0], bool) and not isinstance(c[1], bool):
                        pxs.append(float(c[0]))
                        pys.append(float(c[1]))
                    else:
                        for it in c:
                            _walk_poly(it)

            _walk_poly(poly)
            if pxs and pys:
                poly_boxes.append((min(pxs), min(pys), max(pxs), max(pys)))

        if len(poly_boxes) >= 2:
            min_w = min(b[0] for b in poly_boxes)
            min_s = min(b[1] for b in poly_boxes)
            max_e = max(b[2] for b in poly_boxes)
            max_n = max(b[3] for b in poly_boxes)
            # If polygons touch both -180 and +180 but leave an interior longitude gap > 60 deg around Greenwich,
            # compute the antimeridian-wrapping bbox [west_pos, south, east_neg, north]
            neg_polys = [b for b in poly_boxes if b[2] <= 0.0]
            pos_polys = [b for b in poly_boxes if b[0] >= 0.0]
            if (
                min_w <= -179.0
                and max_e >= 179.0
                and neg_polys
                and pos_polys
                and len(neg_polys) + len(pos_polys) == len(poly_boxes)
            ):
                east_neg = max(b[2] for b in neg_polys)
                west_pos = min(b[0] for b in pos_polys)
                if west_pos - east_neg > 60.0 and -90.0 <= min_s <= 90.0 and -90.0 <= max_n <= 90.0 and min_s <= max_n:
                    return [round(west_pos, 6), round(min_s, 6), round(east_neg, 6), round(max_n, 6)]

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


_TEXT_CADENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(sub[- ]?hourly|\d+\s*min(?:ute)?s?|half[- ]?hourly|every\s+(?:15|30)\s*min)\b", re.I), "sub-hourly"),
    (re.compile(r"(?<!sub[- ])\b(hourly|1[- ]?hour(?:ly)?|3[- ]?hour(?:ly)?|6[- ]?hour(?:ly)?|every\s+[136]\s*hours?)\b", re.I), "hourly"),
    (re.compile(r"\b(bi[- ]?weekly|fortnightly|semi[- ]?monthly|twice[- ]?monthly)\b", re.I), "biweekly"),
    (re.compile(r"\b(10[- ]?da(?:y|ily|ys)|ten[- ]?da(?:y|ily|ys)|8[- ]?da(?:y|ily|ys)|16[- ]?da(?:y|ily|ys)|5[- ]?da(?:y|ily|ys)|dekad(?:al|e)?|three times a month)\b", re.I), "10-daily"),
    (re.compile(r"(?<!bi[- ])(?<!bi)\b(weekly|7[- ]?da(?:y|ily|ys))\b", re.I), "weekly"),
    (re.compile(r"(?<!\d[- ])(?<!ten[- ])(?<!\bmulti[- ])(?<!\bsub[- ])\b(daily|1[- ]?day|day[- ]?by[- ]?day|diurnal(?!\s+cycle))\b", re.I), "daily"),
    (re.compile(r"(?<!semi[- ])(?<!twice[- ])(?<!bi[- ])(?<!bi)\b(monthly|1[- ]?month|month[- ]?by[- ]?month)\b", re.I), "monthly"),
    (re.compile(r"\b(seasonal|quarterly|3[- ]?month)\b", re.I), "seasonal"),
    (re.compile(r"(?<!semi[- ])(?<!semi)(?<!bi[- ])(?<!bi)\b(annual(?:ly)?|yearly|1[- ]?year)\b", re.I), "annual"),
]

_NON_CADENCE_SEASONAL_PHRASE_RE = re.compile(
    r"\bseasonal\s+trajector(?:y|ies)\b|"
    r"\bseasonal\s+productivit(?:y|ies)(?=.*\b(?:yearly|annual)\b)|"
    r"\bseasonal\s+(?:cycle|variability|dynamics|amplitude|growth|decline|integral)\b|"
    r"\b(?:growing|cropping)\s+seasons?\b",
    re.IGNORECASE,
)

_CADENCE_EXCLUDE_CONTEXT_RE = re.compile(
    r"\b(?:no longer archived|no longer produced|previously\s+(?:provided|archived)|"
    r"aggregated\s+(?:\w+\s+){0,2}from|derived\s+(?:\w+\s+){0,2}from|"
    r"interpolated\s+from|instead\s+of|compared\s+(?:to|with)|"
    r"retrieved\s+over|retrieval\s+period)\b",
    re.IGNORECASE,
)

_CLAUSE_CADENCE_WORD = (
    r"(?:sub[- ]?hourly|hourly|[136][- ]?hour(?:ly)?|half[- ]?hourly|\d+[- ]?min|"
    r"daily|weekly|bi[- ]?weekly|fortnightly|semi[- ]?monthly|twice[- ]?monthly|"
    r"5[- ]?da(?:y|ily|ys)|7[- ]?da(?:y|ily|ys)|8[- ]?da(?:y|ily|ys)|"
    r"10[- ]?da(?:y|ily|ys)|ten[- ]?da(?:y|ily|ys)|16[- ]?da(?:y|ily|ys)|dekad(?:al|e)?|"
    r"monthly|quarterly|seasonal|annual(?:ly)?|yearly)"
)


def _scan_cadence_tokens(text_fragment: str) -> list[str]:
    clean_frag = _NON_CADENCE_SEASONAL_PHRASE_RE.sub(" ", text_fragment)
    matched: list[str] = []
    for pat, _ in _TEXT_CADENCE_PATTERNS:
        for m in pat.finditer(clean_frag):
            tok = m.group(0).strip()
            if tok and tok.lower() not in [x.lower() for x in matched]:
                matched.append(tok)
    return matched


def _extract_cadence_from_text(title: str | None, desc: str | None = None) -> str | None:
    """Extract explicit product temporal sampling/composite cadence from title (or explicit cadence clauses in description),
    preserving multi-cadence declarations (e.g. 'daily; monthly', 'biweekly; annual') while guarding product-name false
    positives ('Seasonal Trajectories'), substring leaks ('productivity' -> 'product'), and non-output input/archived notes."""
    if title:
        matched_t = _scan_cadence_tokens(title)
        if matched_t:
            return "; ".join(matched_t)
    if desc:
        # Tier 1: Explicit key-value metadata lines (e.g. "Temporal resolution: 8 day")
        for kv_m in re.finditer(r"\btemporal\s+(?:resolution|sampling|step|granularity|interval)\s*[:=]\s*([^\n.;]+)", desc, re.IGNORECASE):
            matched_kv = _scan_cadence_tokens(kv_m.group(1))
            if matched_kv:
                return "; ".join(matched_kv)

        # Tier 2: Explicit "temporal resolution" phrasing with preceding or following cadence (e.g. "daily temporal resolution")
        for tr_m in re.finditer(
            rf"(?:\b{_CLAUSE_CADENCE_WORD}\b[^.;]{{0,35}})?\btemporal\s+(?:resolution|sampling|step|aggregation)\b[^.;]{{0,80}}",
            desc,
            re.IGNORECASE,
        ):
            clause = tr_m.group(0)
            if _CADENCE_EXCLUDE_CONTEXT_RE.search(clause):
                continue
            matched_tr = _scan_cadence_tokens(clause)
            if matched_tr:
                return "; ".join(matched_tr)

        # Tier 3: Explicit output composite / grid / product clauses with strict word boundaries
        clause_re = re.compile(
            rf"\b{_CLAUSE_CADENCE_WORD}\b"
            rf"(?:\s*(?:and|or|,|/)\s*\b{_CLAUSE_CADENCE_WORD}\b){{0,3}}"
            rf"(?:\s+\w+){{0,2}}\s+\b(?:composites?|means?|averages?|grids?|products?|estimates?|fields?|time[- ]?series|maps?|rasters?)\b",
            re.IGNORECASE,
        )
        for clause_m in clause_re.finditer(desc):
            start_ctx = max(0, clause_m.start() - 45)
            end_ctx = min(len(desc), clause_m.end() + 40)
            window = desc[start_ctx:end_ctx]
            if _CADENCE_EXCLUDE_CONTEXT_RE.search(window):
                continue
            matched_c = _scan_cadence_tokens(clause_m.group(0))
            if matched_c:
                return "; ".join(matched_c)
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
                    r_d = gr.get("row") if isinstance(gr.get("row"), dict) else {}
                    c_d = gr.get("column") if isinstance(gr.get("column"), dict) else {}
                    r_mag = r_d.get("magnitude")
                    c_mag = c_d.get("magnitude")
                    units = r_d.get("units") or c_d.get("units") or ""
                    if r_mag is not None and c_mag is not None and str(r_mag) != str(c_mag):
                        spat_raw = clean_str(f"{r_mag} x {c_mag} {units}")
                    elif r_mag is not None or c_mag is not None:
                        mag = r_mag if r_mag is not None else c_mag
                        spat_raw = clean_str(f"{mag} {units}")
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
            title_str = clean_str(d.get("title"))
            desc_str = clean_str(d.get("description"))
            temp_res_clms = clean_str(d.get("characteristics_temporal_resolution") or d.get("temporal_resolution")) or _extract_cadence_from_text(title_str, desc_str)
            rows.append({
                "source_id": "02_copernicus_clms",
                "raw_id": clean_str(d.get("UID") or d.get("id") or d.get("@id")),
                "title": title_str,
                "description": desc_str,
                "processing_level_raw": clean_str(d.get("processing_level")),
                "spatial_resolution_raw": spat_res,
                "temporal_resolution_raw": temp_res_clms,
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": clean_str(d.get("characteristics_update_frequency") or d.get("update_frequency")),
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
        d_info = extract_stac_doi_provenance(c)
        doi, url = d_info["doi"], d_info["url"]
        prod, _ = extract_stac_providers(c)
        title_str = clean_str(c.get("title"))
        desc_str = clean_str(c.get("description"))
        lvl_raw = clean_str(c.get("processing:level"))
        if not lvl_raw and desc_str:
            lvls = list(dict.fromkeys(m.group(1) for m in re.finditer(r"\b(Level[- ]?[34][A-Z]?|L[34][A-Z]?)\b", desc_str)))
            if lvls:
                lvl_raw = "; ".join(lvls)
        temp_res = _extract_cadence_from_text(title_str, desc_str)
        rows.append({
            "source_id": source_id,
            "raw_id": clean_str(c.get("id")),
            "title": title_str,
            "description": desc_str,
            "processing_level_raw": lvl_raw,
            "spatial_resolution_raw": extract_stac_gsd(c),
            "temporal_resolution_raw": temp_res,
            "temporal_start_raw": t_start,
            "temporal_end_raw": t_end,
            "update_frequency_raw": clean_str(c.get("cads:update_frequency")),
            "bbox_raw": bbox,
            "producing_org_raw": prod or default_org,
            "platform_sensor_raw": None,
            "doi_raw": doi,
            "doi_source_raw": d_info["doi_source"],
            "doi_relation_type_raw": d_info["doi_relation"],
            "related_dois_raw": d_info["related_dois"],
            "license_raw": clean_str(c.get("license")),
            "landing_url_raw": url,
            "keywords_raw": clean_str(c.get("keywords")),
        })
    return rows


def _format_gee_interval(gi: Any) -> str | None:
    if isinstance(gi, list) and gi:
        gi = gi[0]
    if isinstance(gi, dict):
        iv = gi.get("interval")
        unit = clean_str(gi.get("unit"))
        if iv is not None and unit:
            return f"{iv} {unit}"
        if unit:
            return unit
        return None
    return clean_str(gi)


def parse_generic_stac_collections(source_id: str, collections: list[dict[str, Any]], default_org: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for c in collections:
        if not isinstance(c, dict):
            continue
        t_start, t_end = extract_stac_temporal(c.get("extent"))
        bbox = extract_stac_bbox(c.get("extent"))
        d_info = extract_stac_doi_provenance(c)
        doi, url = d_info["doi"], d_info["url"]
        doi_src, doi_rel = d_info["doi_source"], d_info["doi_relation"]
        rel_dois = d_info["related_dois"]
        cid = clean_str(c.get("id"))
        if source_id == "14_ceos_fedeo_agencies" and not doi and cid and cid.lower() in _CEDA_DOI_CROSSWALK:
            doi = _CEDA_DOI_CROSSWALK[cid.lower()]
            doi_src = "datacite_primary_ceda_uuid"
            doi_rel = "is_primary_dataset"
            if rel_dois:
                rel_parts = [x.strip() for x in rel_dois.split(";") if x.strip() and x.strip().lower() != doi.lower()]
                rel_dois = "; ".join(rel_parts) if rel_parts else None
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
        temp_res = (
            _format_gee_interval(c.get("gee:interval"))
            or clean_str(summaries.get("temporal_resolution") or c.get("temporal_resolution"))
        )
        rows.append({
            "source_id": source_id,
            "raw_id": cid,
            "title": clean_str(c.get("title") or c.get("id")),
            "description": clean_str(c.get("description")),
            "processing_level_raw": lvl,
            "spatial_resolution_raw": extract_stac_gsd(c),
            "temporal_resolution_raw": temp_res,
            "temporal_start_raw": t_start,
            "temporal_end_raw": t_end,
            "update_frequency_raw": clean_str(c.get("gee:status") or c.get("osc:status") or c.get("cads:update_frequency")),
            "bbox_raw": bbox,
            "producing_org_raw": prod or default_org,
            "hosting_platform_raw": host,
            "platform_sensor_raw": plat,
            "doi_raw": doi,
            "doi_source_raw": doi_src,
            "doi_relation_type_raw": doi_rel,
            "related_dois_raw": rel_dois,
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
                    "temporal_resolution_raw": _extract_cadence_from_text(title, desc),
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


def _map_eumetsat_transfer_frequency(freqs: list[str], text_hint: str = "") -> str | None:
    """Map unambiguous EUMETSAT/WEkEO digitalTransfers frequency counts to canonical cadence strings.
    Per-orbit/granule file counts (e.g. '14-15 (per day)', '~150 (per day)') return None."""
    if not freqs:
        return None
    hint_l = text_hint.lower()
    joined = " ; ".join(str(x).lower() for x in freqs if x)
    if re.search(r"\b1\s*\(\s*every\s+10\s+days\s*\)", joined):
        return "10-day"
    if re.search(r"\b1\s*\(\s*per\s+month\s*\)", joined):
        return "monthly"
    if re.search(r"\b1\s*\(\s*per\s+day\s*\)", joined) and not re.search(r"per\s+satellite", joined):
        return "daily"
    if re.search(r"\b96\s*\(\s*per\s+day\s*\)", joined) and re.search(r"\b(?:15[- ]?min(?:ute)?|msg|seviri|fci|mtg)\b", hint_l):
        return "15 min"
    if re.search(r"\b48\s*\(\s*per\s+day\s*\)", joined) and re.search(r"\b(?:30[- ]?min|half[- ]?hour|seviri|msg)\b", hint_l):
        return "30 min"
    if re.search(r"\b24\s*\(\s*per\s+day\s*\)", joined) and re.search(r"\b(?:hourly|1[- ]?hour)\b", hint_l):
        return "hourly"
    return None


def _extract_eumetsat_dt_freqs(dt_list: Any) -> list[str]:
    out: list[str] = []
    for dt in (dt_list if isinstance(dt_list, list) else []):
        if not isinstance(dt, dict):
            continue
        fmts = dt.get("formats") or dt.get("format") or []
        if isinstance(fmts, dict):
            fmts = [fmts]
        for fm in (fmts if isinstance(fmts, list) else []):
            if isinstance(fm, dict) and fm.get("frequency"):
                out.append(str(fm["frequency"]).strip())
    return out


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
                raw_plvl = clean_str(md.get("processingLevels") or md.get("processingLevel"))
                if raw_plvl:
                    lvls.append(raw_plvl)
                else:
                    for k in (kws if isinstance(kws, list) else []):
                        ks = str(k).lower()
                        if ks in ("level-3", "level-4", "level-2", "level-1", "level-0", "l3", "l4", "l2", "l1"):
                            lvls.append(str(k))
                lvl = "; ".join(dict.fromkeys(lvls)) if lvls else None

                # Recover structured DOI, platform/sensor, spatial resolution, and unambiguous transfer cadence
                doi_wk = None
                raw_doi_obj = md.get("doi")
                if isinstance(raw_doi_obj, list):
                    for item in raw_doi_obj:
                        if isinstance(item, dict) and (item.get("identifier") or item.get("id") or item.get("url")):
                            doi_wk = clean_str(item.get("identifier") or item.get("id") or item.get("url"))
                            break
                        elif isinstance(item, str):
                            doi_wk = clean_str(item)
                            break
                elif isinstance(raw_doi_obj, dict):
                    doi_wk = clean_str(raw_doi_obj.get("identifier") or raw_doi_obj.get("id") or raw_doi_obj.get("url"))
                else:
                    doi_wk = clean_str(raw_doi_obj)

                plat_parts: list[str] = []
                for pk in ("satellite", "instrument"):
                    pv = md.get(pk)
                    if isinstance(pv, list):
                        for elem in pv:
                            es = clean_str(elem.get("name") or elem.get("seriesName") if isinstance(elem, dict) else elem)
                            if es:
                                plat_parts.append(es)
                    elif isinstance(pv, dict):
                        es = clean_str(pv.get("name") or pv.get("seriesName"))
                        if es:
                            plat_parts.append(es)
                    elif pv:
                        es = clean_str(pv)
                        if es:
                            plat_parts.append(es)
                plat_wk = "; ".join(dict.fromkeys(plat_parts)) if plat_parts else None

                spat_wk = None
                sr_objs = md.get("spatialResolutionAsObjects")
                if isinstance(sr_objs, list) and sr_objs:
                    sr_labels = [clean_str(o.get("label")) for o in sr_objs if isinstance(o, dict) and o.get("label")]
                    spat_wk = "; ".join(dict.fromkeys(x for x in sr_labels if x)) or None
                elif isinstance(sr_objs, dict) and sr_objs.get("label"):
                    spat_wk = clean_str(sr_objs.get("label"))
                if not spat_wk and md.get("spatialResolution") is not None:
                    sr_val = clean_str(md.get("spatialResolution"))
                    sr_uom = clean_str(md.get("spatialResolutionUom"))
                    if sr_val and sr_uom and not any(u in sr_val.lower() for u in ("m", "km", "deg", "°")):
                        spat_wk = f"{sr_val} {sr_uom}"
                    elif sr_val and any(u in sr_val.lower() for u in ("m", "km", "deg", "°")):
                        spat_wk = sr_val

                ttl_wk = clean_str(md.get("datasetTitle") or md.get("title") or did)
                dsc_wk = clean_str(md.get("abstract") or md.get("description"))
                freqs_wk = _extract_eumetsat_dt_freqs(md.get("digitalTransfers"))
                if md.get("frequency"):
                    freqs_wk.append(str(md["frequency"]))
                temp_wk = _map_eumetsat_transfer_frequency(freqs_wk, f"{ttl_wk or ''} {dsc_wk or ''}")

                rows.append({
                    "source_id": "09_wekeo_dias",
                    "raw_id": did,
                    "title": ttl_wk,
                    "description": dsc_wk,
                    "processing_level_raw": lvl,
                    "spatial_resolution_raw": spat_wk,
                    "temporal_resolution_raw": temp_wk,
                    "temporal_start_raw": clean_str(md.get("tempextent_begin")),
                    "temporal_end_raw": clean_str(md.get("tempextent_end")),
                    "update_frequency_raw": clean_str(md.get("productStatus")),
                    "bbox_raw": bbox,
                    "producing_org_raw": clean_str(md.get("responsible_org") or md.get("responsibleParty")),
                    "hosting_platform_raw": "WEkEO Copernicus DIAS",
                    "platform_sensor_raw": plat_wk,
                    "doi_raw": doi_wk,
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
                    "temporal_resolution_raw": _extract_cadence_from_text(title, desc),
                    "temporal_start_raw": t_start,
                    "temporal_end_raw": t_end,
                    "update_frequency_raw": freq,
                    "bbox_raw": bbox,
                    "producing_org_raw": org,
                    "platform_sensor_raw": None,
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


def _extract_umm_spatial_resolution(umm: dict[str, Any], title: str | None = None) -> str | None:
    cand: str | None = None
    spat = umm.get("SpatialExtent")
    if isinstance(spat, dict):
        hsd = spat.get("HorizontalSpatialDomain")
        if isinstance(hsd, dict):
            rcs = hsd.get("ResolutionAndCoordinateSystem")
            if isinstance(rcs, dict):
                hdr = rcs.get("HorizontalDataResolution")
                if isinstance(hdr, dict):
                    for key in ("GriddedResolutions", "NonGriddedResolutions", "GenericResolutions", "VariesResolutions"):
                        items = hdr.get(key)
                        if isinstance(items, list):
                            for it in items:
                                if not isinstance(it, dict):
                                    continue
                                unit = clean_str(it.get("Unit"))
                                if not unit or unit.lower() in ("not provided", "na", "n/a"):
                                    continue
                                xdim = (
                                    it.get("XDimension")
                                    if it.get("XDimension") is not None
                                    else (it.get("MinimumXDimension") or it.get("MaximumXDimension"))
                                )
                                ydim = (
                                    it.get("YDimension")
                                    if it.get("YDimension") is not None
                                    else (it.get("MinimumYDimension") or it.get("MaximumYDimension"))
                                )
                                if xdim is not None and ydim is not None and str(xdim) != str(ydim):
                                    cand = f"{xdim} x {ydim} {unit}"
                                    break
                                dim = xdim if xdim is not None else ydim
                                if dim is not None:
                                    cand = f"{dim} {unit}"
                                    break
                        if cand:
                            break
    if not cand:
        for attr in (umm.get("AdditionalAttributes") or []):
            if isinstance(attr, dict) and attr.get("Value") is not None:
                aname = str(attr.get("Name") or "").lower()
                if "spatial_resolution" in aname or "horizontal_resolution" in aname or aname in ("resolution", "grid_spacing"):
                    val = clean_str(attr.get("Value"))
                    unit = clean_str(attr.get("ParameterUnitsOfMeasure"))
                    if val:
                        cand = f"{val} {unit}".strip() if unit else val
                        break
    return cand


def _gcmd_temporal_contradicts_title(tkw_str: str | None, title: str | None) -> bool:
    """Return True if fallback GCMD TemporalKeywords state sub-daily/hourly/daily scan rates on an explicitly Monthly/Annual titled product."""
    if not tkw_str or not title:
        return False
    tl = title.lower()
    kl = tkw_str.lower()
    if re.search(r"\b(monthly|annual|yearly|seasonal|quarterly)\b", tl) and not re.search(
        r"\b(daily|hourly|sub-daily|minute|second|3-hour|6-hour)\b", tl
    ):
        if re.search(r"\b(second|minute|hour|hourly|daily)\b", kl) and not re.search(r"\b(month|monthly|annual|year)\b", kl):
            return True
    return False


def _extract_umm_temporal_resolution(umm: dict[str, Any], title: str | None = None) -> str | None:
    for te in (umm.get("TemporalExtents") or []):
        if not isinstance(te, dict):
            continue
        tr = te.get("TemporalResolution")
        if isinstance(tr, dict):
            val = tr.get("Value")
            unit = clean_str(tr.get("Unit"))
            if val is not None and unit and unit.lower() not in ("varies", "constant", "not provided"):
                return f"{val} {unit}"
            if unit and unit.lower() not in ("varies", "constant", "not provided"):
                return unit
    tkws = [clean_str(k) for k in (umm.get("TemporalKeywords") or []) if clean_str(k)]
    tkws = [k for k in tkws if k and k.lower() not in ("not provided", "varies", "other")]
    if tkws:
        joined = "; ".join(dict.fromkeys(tkws))
        if _gcmd_temporal_contradicts_title(joined, title):
            return None
        return joined
    return None


def _is_mismatched_earthdata_catalog_url(u: str, short_name: str | None) -> bool:
    """Guard against upstream GES_DISC UMM copy-paste errors where earthdata.nasa.gov/data/catalog/<slug> points to a different dataset's short_name."""
    if "earthdata.nasa.gov/data/catalog/" not in u.lower() or not short_name:
        return False
    slug = u.lower().split("earthdata.nasa.gov/data/catalog/", 1)[1].strip("/")
    slug_norm = re.sub(r"[^a-z0-9]+", "", slug)
    sn_norm = re.sub(r"[^a-z0-9]+", "", short_name.lower())
    if len(sn_norm) >= 4 and sn_norm not in slug_norm:
        return True
    return False


def _select_cmr_landing_url(e: dict[str, Any], umm: dict[str, Any], doi: str | None, cid: str, short_name: str | None) -> str:
    # 1. Explicit dataset-specific DATA SET LANDING PAGE from UMM RelatedUrls (excluding cross-dataset slug mismatches)
    for ru in (umm.get("RelatedUrls") or []):
        if not isinstance(ru, dict):
            continue
        u = clean_url(ru.get("URL"))
        if not u or _is_mismatched_earthdata_catalog_url(u, short_name):
            continue
        utype = str(ru.get("Type") or "").upper()
        if utype == "DATA SET LANDING PAGE":
            return u
    # 2. DOI resolver URL if canonical DOI exists
    doi_clean = clean_doi(doi)
    if doi_clean:
        return f"https://doi.org/{doi_clean}"
    # 3. CollectionURL / Project Home Page from UMM RelatedUrls
    for ru in (umm.get("RelatedUrls") or []):
        if not isinstance(ru, dict):
            continue
        u = clean_url(ru.get("URL"))
        if not u or _is_mismatched_earthdata_catalog_url(u, short_name):
            continue
        utype = str(ru.get("Type") or "").upper()
        uct = str(ru.get("URLContentType") or "").upper()
        if utype == "COLLECTION URL" or uct == "COLLECTIONURL":
            return u
    # 3. Typed links from CMR Atom-JSON entry (excluding browse# and image URLs)
    for preferred_suffix in ("/data#", "/service#", "/metadata#", "/documentation#"):
        for lk in (e.get("links") or []):
            if not isinstance(lk, dict):
                continue
            rel = str(lk.get("rel") or "").lower()
            if rel.endswith(preferred_suffix):
                u = clean_url(lk.get("href"))
                if u:
                    return u
    # 4. Distribution or publication URL from UMM RelatedUrls (excluding visualizations/browse)
    for ru in (umm.get("RelatedUrls") or []):
        if not isinstance(ru, dict):
            continue
        utype = str(ru.get("Type") or "").upper()
        if "VISUALIZATION" in utype or "BROWSE" in utype:
            continue
        u = clean_url(ru.get("URL"))
        if u:
            return u
    # 5. Any remaining non-browse link in entry
    for lk in (e.get("links") or []):
        if not isinstance(lk, dict):
            continue
        rel = str(lk.get("rel") or "").lower()
        if "browse#" in rel:
            continue
        u = clean_url(lk.get("href"))
        if u:
            return u
    return f"https://search.earthdata.nasa.gov/search/granules?p={cid}"


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
                umm = e.get("_umm") if isinstance(e.get("_umm"), dict) else {}
                bbox = None
                boxes = e.get("boxes")
                if isinstance(boxes, list) and boxes:
                    # CMR box format is "south west north east"
                    parts = [float(x) for x in str(boxes[0]).split() if re.match(r"^-?\d+(\.\d+)?$", x)]
                    if len(parts) == 4:
                        bbox = [parts[1], parts[0], parts[3], parts[2]]
                if bbox is None and isinstance(umm.get("SpatialExtent"), dict):
                    brs = (
                        ((umm["SpatialExtent"].get("HorizontalSpatialDomain") or {}).get("Geometry") or {}).get("BoundingRectangles")
                    )
                    if isinstance(brs, list) and brs and isinstance(brs[0], dict):
                        try:
                            bbox = [
                                float(brs[0]["WestBoundingCoordinate"]),
                                float(brs[0]["SouthBoundingCoordinate"]),
                                float(brs[0]["EastBoundingCoordinate"]),
                                float(brs[0]["NorthBoundingCoordinate"]),
                            ]
                        except Exception:
                            bbox = None

                # Extract DOI from UMM.DOI, UMM.AdditionalAttributes, or Atom-JSON entry
                doi = clean_str(e.get("doi"))
                umm_doi = umm.get("DOI")
                if isinstance(umm_doi, dict) and umm_doi.get("DOI"):
                    doi = clean_str(umm_doi.get("DOI"))
                if not doi:
                    for attr in (umm.get("AdditionalAttributes") or []):
                        if isinstance(attr, dict) and "doi" in str(attr.get("Name") or "").lower() and attr.get("Value"):
                            doi = clean_str(attr.get("Value"))
                            if doi:
                                break

                short_name = clean_str(e.get("short_name") or umm.get("ShortName"))
                title = clean_str(e.get("title") or umm.get("EntryTitle") or e.get("dataset_id"))
                if short_name and title and short_name not in title:
                    title = f"{short_name}: {title}"

                lvl_raw = clean_str(e.get("processing_level_id") or (umm.get("ProcessingLevel") or {}).get("Id"))
                spat_raw = _extract_umm_spatial_resolution(umm, title)
                temp_raw = _extract_umm_temporal_resolution(umm, title)

                t_start = clean_str(e.get("time_start"))
                t_end = clean_str(e.get("time_end"))
                ends_at_present = False
                for te in (umm.get("TemporalExtents") or []):
                    if isinstance(te, dict) and te.get("EndsAtPresentFlag") is True:
                        ends_at_present = True
                if ends_at_present:
                    t_end = None

                coll_prog = clean_str(umm.get("CollectionProgress"))
                coll_dt = clean_str(e.get("collection_data_type") or umm.get("CollectionDataType"))
                # Do not label update_frequency_raw as CollectionProgress:ACTIVE when the collection has a closed historical end date <= 2024
                # or is a static single-epoch mask/DEM or future USAP grant period (CollectionProgress:ACTIVE in CMR denotes archive availability, not ongoing temporal updates)
                is_closed_or_static_cmr = bool(
                    (t_end and not ends_at_present and str(t_end)[:4] <= "2024")
                    or (t_start and str(t_start)[:4] >= "2025" and cid.endswith("-AMD_USAPDC"))
                    or re.search(
                        r"\b(srtm|land/sea static mask|landseamask|mod44w|static layers|proba-v)\b",
                        f"{short_name or ''} {title or ''}",
                        re.IGNORECASE,
                    )
                )
                prog_token = (
                    f"CollectionProgress:{coll_prog}"
                    if (coll_prog and not (coll_prog.upper() == "ACTIVE" and is_closed_or_static_cmr))
                    else None
                )
                upd_parts = [x for x in (prog_token, coll_dt) if x]
                upd_raw = "; ".join(upd_parts) if upd_parts else None

                # Platforms & Instruments from UMM or Atom-JSON
                plat_items: list[str] = []
                for pl in (umm.get("Platforms") or []):
                    if not isinstance(pl, dict):
                        continue
                    psn = clean_str(pl.get("ShortName"))
                    if psn and psn.upper() not in ("NOT APPLICABLE", "NOT PROVIDED"):
                        plat_items.append(psn)
                    for ins in (pl.get("Instruments") or []):
                        if isinstance(ins, dict):
                            isn = clean_str(ins.get("ShortName"))
                            if isn and isn.upper() not in ("NOT APPLICABLE", "NOT PROVIDED"):
                                plat_items.append(isn)
                plat_str = "; ".join(dict.fromkeys(plat_items)) if plat_items else clean_str([e.get("platforms"), e.get("instruments")])

                # ScienceKeywords from UMM
                skw_items: list[str] = []
                for sk in (umm.get("ScienceKeywords") or []):
                    if isinstance(sk, dict):
                        for k_lvl in ("Topic", "Term", "VariableLevel1", "VariableLevel2", "DetailedVariable"):
                            kv = clean_str(sk.get(k_lvl))
                            if kv and kv.upper() not in ("NOT APPLICABLE", "NOT PROVIDED"):
                                skw_items.append(kv)
                if e.get("consortiums"):
                    skw_items.append(str(clean_str(e.get("consortiums"))))
                kw_str = "; ".join(dict.fromkeys(skw_items)) if skw_items else None

                url = _select_cmr_landing_url(e, umm, doi, cid, short_name)
                doi_c = clean_doi(doi)
                cmr_rel_dois: list[str] = []
                for ru in (umm.get("RelatedUrls") or []):
                    if isinstance(ru, dict) and str(ru.get("Type") or "").upper() == "VIEW RELATED INFORMATION":
                        d_ru = clean_doi(ru.get("URL"))
                        if d_ru and d_ru != doi_c:
                            cmr_rel_dois.append(d_ru)

                rows.append({
                    "source_id": source_id,
                    "raw_id": cid,
                    "short_name_raw": short_name,
                    "title": title,
                    "description": clean_str(e.get("summary")),
                    "processing_level_raw": lvl_raw,
                    "spatial_resolution_raw": spat_raw,
                    "temporal_resolution_raw": temp_raw,
                    "temporal_start_raw": t_start,
                    "temporal_end_raw": t_end,
                    "update_frequency_raw": upd_raw,
                    "bbox_raw": bbox,
                    "producing_org_raw": clean_str(e.get("organizations") or e.get("archive_center") or e.get("data_center")),
                    "platform_sensor_raw": plat_str,
                    "doi_raw": doi,
                    "doi_source_raw": "nasa_cmr_umm_doi" if doi_c else None,
                    "doi_relation_type_raw": "is_primary_dataset" if doi_c else None,
                    "related_dois_raw": "; ".join(dict.fromkeys(cmr_rel_dois)) if cmr_rel_dois else None,
                    "license_raw": "NASA Earthdata Open Data Policy",
                    "landing_url_raw": url,
                    "keywords_raw": kw_str,
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
            doi_eum = None
            raw_doi_e = s.get("doi")
            if isinstance(raw_doi_e, dict):
                doi_eum = clean_str(raw_doi_e.get("id") or raw_doi_e.get("href"))
            elif raw_doi_e:
                doi_eum = clean_str(raw_doi_e)
            ttl_eum = clean_str(s.get("datasetTitle"))
            dsc_eum = clean_str(s.get("abstract") or s.get("summary"))
            temp_eum = _map_eumetsat_transfer_frequency(
                _extract_eumetsat_dt_freqs(s.get("digitalTransfers")),
                f"{ttl_eum or ''} {clean_str(s.get('summary')) or ''} {dsc_eum or ''}",
            )
            rows.append({
                "source_id": "17_eumetsat_product_navigator",
                "raw_id": clean_str(s.get("id") or hit.get("_id")),
                "title": ttl_eum,
                "description": dsc_eum,
                "processing_level_raw": clean_str(s.get("processingLevels")),
                "spatial_resolution_raw": clean_str(s.get("spatialResolutions")),
                "temporal_resolution_raw": temp_eum,
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": clean_str(s.get("productStatus")),
                "bbox_raw": bbox,
                "producing_org_raw": clean_str(s.get("originator") or s.get("distributors") or "EUMETSAT"),
                "platform_sensor_raw": clean_str(s.get("satellites")),
                "doi_raw": doi_eum,
                "license_raw": clean_str(s.get("dataPolicies")),
                "landing_url_raw": f"https://user.eumetsat.int/catalogue/{s.get('id')}" if s.get("id") else None,
                "keywords_raw": clean_str(s.get("parameters") or s.get("services")),
            })
    return rows


def _clean_gcmd_list(vals: Any) -> str | None:
    if not vals:
        return None
    items = vals if isinstance(vals, list) else [vals]
    kept: list[str] = []
    for it in items:
        s = clean_str(it)
        if not s:
            continue
        if s.lower() in ("not provided", "not applicable", "none", "unknown"):
            continue
        kept.append(s)
    return "; ".join(dict.fromkeys(kept)) if kept else None


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
                bbox = _extract_geojson_bbox(sb)

                # Extract DOI from doi attribute, links, crossReferences, or citeAsStatements
                doi = clean_doi(a.get("doi"))
                if not doi:
                    for lk in (a.get("links") or []):
                        if isinstance(lk, dict):
                            d_cand = clean_doi(lk.get("linkUrl") or lk.get("linkName") or lk.get("linkDescription"))
                            if d_cand:
                                doi = d_cand
                                break
                if not doi:
                    for cs in (a.get("citeAsStatements") or []):
                        d_cand = clean_doi(cs)
                        if d_cand:
                            doi = d_cand
                            break

                # Role-prioritized non-image landing URL
                url = None
                for pref_fn in ("information", "search", "download", "offlineaccess"):
                    for lk in (a.get("links") or []):
                        if not isinstance(lk, dict):
                            continue
                        fn = str(lk.get("linkFunction") or "").lower()
                        if fn == pref_fn:
                            u_cand = clean_url(lk.get("linkUrl"))
                            if u_cand:
                                url = u_cand
                                break
                    if url:
                        break
                if not url:
                    for lk in (a.get("links") or []):
                        if isinstance(lk, dict):
                            u_cand = clean_url(lk.get("linkUrl"))
                            if u_cand:
                                url = u_cand
                                break
                if not url and doi:
                    url = f"https://doi.org/{doi}"

                title_noaa = clean_str(a.get("title"))
                spat_raw = _clean_gcmd_list(a.get("gcmdHorizontalResolution"))
                temp_raw = _clean_gcmd_list(a.get("gcmdTemporalResolution"))
                if _gcmd_temporal_contradicts_title(temp_raw, title_noaa):
                    temp_raw = None
                plat_raw = _clean_gcmd_list((a.get("gcmdPlatforms") or a.get("platforms") or []) + (a.get("gcmdInstruments") or a.get("instruments") or []))
                org_raw = _clean_gcmd_list(a.get("gcmdDataCenters") or a.get("organizationNames")) or "NOAA (NESDIS / NCEI / STAR)"
                kw_raw = _clean_gcmd_list((a.get("gcmdScience") or []) + (a.get("keywords") or []) + (a.get("gcmdScienceServices") or []))

                rows.append({
                    "source_id": "20_noaa_onestop",
                    "raw_id": rid,
                    "title": title_noaa,
                    "description": clean_str(a.get("description")),
                    "processing_level_raw": clean_str(a.get("dataProcessingLevel")),
                    "spatial_resolution_raw": spat_raw,
                    "temporal_resolution_raw": temp_raw,
                    "temporal_start_raw": clean_str(a.get("beginDate") or a.get("beginYear")),
                    "temporal_end_raw": clean_str(a.get("endDate") or a.get("endYear")),
                    "update_frequency_raw": clean_str(a.get("maintenanceFrequency")),
                    "bbox_raw": bbox,
                    "producing_org_raw": org_raw,
                    "platform_sensor_raw": plat_raw,
                    "doi_raw": doi,
                    "license_raw": clean_str(a.get("useLimitation") or a.get("dataLicenses") or "U.S. Government Public Domain"),
                    "landing_url_raw": url,
                    "keywords_raw": kw_raw,
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
        desc = clean_str(meta.get("overview") or meta.get("function"))
        t_start, t_end = _split_date_range_str(meta.get("content_date"))
        rows.append({
            "source_id": "23_wri_gfw_and_resourcewatch",
            "raw_id": f"GFW:{d.get('dataset')}",
            "title": title,
            "description": desc,
            "processing_level_raw": None,
            "spatial_resolution_raw": clean_str(meta.get("resolution")),
            "temporal_resolution_raw": _extract_cadence_from_text(title, desc),
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
            title_rw = clean_str(m0.get("name") or attr.get("name"))
            desc_rw = clean_str(m0.get("description") or info.get("functions"))
            rows.append({
                "source_id": "23_wri_gfw_and_resourcewatch",
                "raw_id": f"RW:{item.get('id')}",
                "title": title_rw,
                "description": desc_rw,
                "processing_level_raw": None,
                "spatial_resolution_raw": clean_str(info.get("spatial_resolution")),
                "temporal_resolution_raw": _extract_cadence_from_text(title_rw, desc_rw),
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": clean_str(info.get("frequency_of_updates")),
                "bbox_raw": None,
                "producing_org_raw": clean_str(info.get("organization") or m0.get("source")) or "World Resources Institute (Resource Watch)",
                "hosting_platform_raw": clean_str(attr.get("provider")),
                "platform_sensor_raw": None,
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
        title_fao = clean_str(f"{m.get('_workspace_caption', 'FAO')} - {m.get('caption') or m.get('code')}")
        desc_fao = clean_str(m.get("description") or m.get("measureCaption"))
        rows.append({
            "source_id": "26_fao_unep_and_thematic_benchmarks",
            "raw_id": f"FAO_GISMGR:{m.get('workspaceCode')}:{m.get('code')}",
            "title": title_fao,
            "description": desc_fao,
            "processing_level_raw": None,
            "spatial_resolution_raw": None,
            "temporal_resolution_raw": _extract_cadence_from_text(title_fao, desc_fao),
            "temporal_start_raw": None,
            "temporal_end_raw": None,
            "update_frequency_raw": None,
            "bbox_raw": m.get("bbox") if isinstance(m.get("bbox"), list) else None,
            "producing_org_raw": "Food and Agriculture Organization of the United Nations (FAO)",
            "platform_sensor_raw": None,
            "doi_raw": None,
            "license_raw": "FAO Open Data Licensing Policy (CC-BY-4.0)",
            "landing_url_raw": self_url or "https://data.apps.fao.org/",
            "keywords_raw": clean_str(m.get("tags")),
        })

    seen_zenodo_ids: set[str] = set()
    seen_zenodo_dois: set[str] = set()
    z_path = RAW_DIR / "26_fao_unep_and_thematic_benchmarks/data/zenodo_thematic_benchmarks.jsonl"
    with z_path.open(encoding="utf-8") as f:
        for line in f:
            h = json.loads(line)
            md = h.get("metadata", {}) or {}
            z_id_str = str(h.get("id") or "")
            if z_id_str:
                seen_zenodo_ids.add(z_id_str)
            doi_z = clean_str(h.get("doi") or md.get("doi"))
            if doi_z:
                seen_zenodo_dois.add(doi_z.lower())
            creators = [c.get("name") for c in (md.get("creators") or []) if isinstance(c, dict) and c.get("name")]
            affils = [c.get("affiliation") for c in (md.get("creators") or []) if isinstance(c, dict) and c.get("affiliation")]
            if z_id_str == "10212472":
                org = "UCLouvain / University of Liège / Forschungszentrum Jülich (ESA TROPOSIF)"
            else:
                org = "; ".join(dict.fromkeys(affils[:3])) if affils else ("; ".join(creators[:3]) if creators else "Zenodo Scientific Benchmark")
            title_z = clean_str(md.get("title") or h.get("title"))
            desc_z = clean_str(md.get("description"))
            links_z = h.get("links") or {}
            landing_z = clean_str(
                links_z.get("self_html")
                or links_z.get("html")
                or links_z.get("latest_html")
                or links_z.get("doi")
                or links_z.get("self_doi_html")
                or (f"https://zenodo.org/records/{z_id_str}" if z_id_str else None)
            )
            rel_z: list[str] = []
            doi_z_clean = clean_doi(doi_z)
            for ri in (md.get("related_identifiers") or []):
                if isinstance(ri, dict) and ri.get("identifier"):
                    d_ri = clean_doi(ri.get("identifier"))
                    if d_ri and d_ri != doi_z_clean:
                        rel_z.append(d_ri)
            rows.append({
                "source_id": "26_fao_unep_and_thematic_benchmarks",
                "raw_id": f"ZENODO:{h.get('id')}",
                "title": title_z,
                "description": desc_z,
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
                "doi_raw": doi_z,
                "doi_source_raw": "zenodo_deposit_doi" if doi_z_clean else None,
                "doi_relation_type_raw": "is_primary_dataset" if doi_z_clean else None,
                "related_dois_raw": "; ".join(dict.fromkeys(rel_z)) if rel_z else None,
                "license_raw": clean_str((md.get("license") or {}).get("id") if isinstance(md.get("license"), dict) else md.get("license")),
                "landing_url_raw": landing_z,
                "keywords_raw": clean_str(md.get("keywords")),
                "_file_keys": [str(fn.get("key") or "") for fn in (h.get("files") or []) if isinstance(fn, dict) and fn.get("key")],
                "_deposit_pub_date": clean_str(md.get("publication_date") or h.get("created")),
            })

    # 3. DataCite Benchmark Records (G3P, FLUXCOM-X-BASE, LCSPP/LCSIF, J-OFURO3/DIAS, CoSEB/ScienceDB, Theia, PANGAEA)
    seen_dc_dois: set[str] = set()
    g3p_path = RAW_DIR / "26_fao_unep_and_thematic_benchmarks/data/gfz_datacite_g3p_benchmarks.json"
    if g3p_path.exists():
        for d in json.loads(g3p_path.read_text(encoding="utf-8")).get("data", []):
            attr = d.get("attributes", {}) or {}
            rtype_gen = str((attr.get("types") or {}).get("resourceTypeGeneral") or "").lower()
            if rtype_gen and rtype_gen not in ("dataset", "collection"):
                continue
            titles = [t.get("title") for t in (attr.get("titles") or []) if isinstance(t, dict) and t.get("title")]
            descs = [x.get("description") for x in (attr.get("descriptions") or []) if isinstance(x, dict) and x.get("description")]
            creators = [c.get("name") for c in (attr.get("creators") or []) if isinstance(c, dict) and c.get("name")]
            subjects = [s.get("subject") for s in (attr.get("subjects") or []) if isinstance(s, dict) and s.get("subject")]
            doi_id = clean_str(d.get("id"))
            title_dc = clean_str(titles[0] if titles else doi_id)
            desc_dc = clean_str(descs[0] if descs else None)
            publisher_dc = clean_str(attr.get("publisher"))
            doi_low = (doi_id or "").lower()
            # Filter noisy DataCite records (data-paper texts or purely local non-EO biological surveys)
            if title_dc and re.search(r"\bdata\s+paper\b", title_dc, re.IGNORECASE):
                continue

            if doi_low in seen_zenodo_dois or (
                doi_low.startswith("10.5281/zenodo.")
                and doi_low.split("10.5281/zenodo.")[-1] in seen_zenodo_ids
            ):
                continue
            if doi_low:
                seen_dc_dois.add(doi_low)

            dc_boxes: list[tuple[float, float, float, float]] = []
            for gl in (attr.get("geoLocations") or []):
                if isinstance(gl, dict) and isinstance(gl.get("geoLocationBox"), dict):
                    gb = gl["geoLocationBox"]
                    try:
                        w_b = float(gb["westBoundLongitude"])
                        s_b = float(gb["southBoundLatitude"])
                        e_b = float(gb["eastBoundLongitude"])
                        n_b = float(gb["northBoundLatitude"])
                        # Fix known DataCite inverted full-globe longitudes (west=180, east=-180)
                        if w_b == 180.0 and e_b == -180.0:
                            w_b, e_b = -180.0, 180.0
                        if s_b > n_b:
                            s_b, n_b = n_b, s_b
                        dc_boxes.append((w_b, s_b, e_b, n_b))
                    except Exception:
                        pass
            bbox_dc = None
            if len(dc_boxes) == 1:
                bbox_dc = list(dc_boxes[0])
            elif len(dc_boxes) > 1:
                if all(wb <= eb for wb, _, eb, _ in dc_boxes):
                    bbox_dc = [
                        min(wb for wb, _, _, _ in dc_boxes),
                        min(sb for _, sb, _, _ in dc_boxes),
                        max(eb for _, _, eb, _ in dc_boxes),
                        max(nb for _, _, _, nb in dc_boxes),
                    ]
                else:
                    bbox_dc = list(dc_boxes[0])

            dias_md = d.get("_dias_metadata") if isinstance(d.get("_dias_metadata"), dict) else {}
            spat_dc = clean_str(dias_md.get("spatial_resolution_raw"))
            temp_dc = clean_str(dias_md.get("temporal_resolution_raw"))
            ts_dc = clean_str(dias_md.get("temporal_start_raw"))
            te_dc = clean_str(dias_md.get("temporal_end_raw"))
            if not ts_dc or not te_dc:
                for dt_item in (attr.get("dates") or []):
                    if isinstance(dt_item, dict) and str(dt_item.get("dateType") or "").lower() == "collected":
                        d_str = str(dt_item.get("date") or "").strip()
                        if "/" in d_str:
                            p1, p2 = [clean_str(x) for x in d_str.split("/", 1)]
                            ts_dc = ts_dc or p1
                            te_dc = te_dc or p2
                            break
            rel_dc: list[str] = []
            for ri in (attr.get("relatedIdentifiers") or []):
                if isinstance(ri, dict) and ri.get("relatedIdentifier"):
                    d_ri = clean_doi(ri.get("relatedIdentifier"))
                    if d_ri and d_ri.lower() != doi_low:
                        rel_dc.append(d_ri)

            if doi_low.startswith("10.5880/"):
                org_dc = publisher_dc or clean_str(creators[:2]) or "GFZ German Research Centre for Geosciences / G3P Consortium"
                host_dc = "GFZ Data Services (DataCite)"
                plat_dc = "GRACE / GRACE-FO"
                kw_dc = clean_str(subjects) or "G3P; groundwater; GRACE; GRACE-FO; Horizon 2020"
            elif doi_low.startswith("10.18160/"):
                org_dc = "Max Planck Institute for Biogeochemistry / ICOS Carbon Portal (FLUXCOM-X)"
                host_dc = "ICOS Carbon Portal (DataCite)"
                plat_dc = "MODIS; FLUXNET; ICOS"
                kw_dc = clean_str(subjects) or "FLUXCOM-X-BASE; gross primary production; GPP; NEE; evapotranspiration; carbon flux; energy flux"
            elif doi_low.startswith("10.15454/"):
                org_dc = "INRAE / CESBIO (Institut National de Recherche pour l'Agriculture, l'Alimentation et l'Environnement)"
                host_dc = "Recherche Data Gouv / Portail Data INRAE (DataCite)"
                plat_dc = "SMOS"
                kw_dc = clean_str(subjects) or "SMOS-IC; soil moisture; vegetation optical depth; L-VOD; global"
            elif doi_low == "10.1594/pangaea.911944":
                org_dc = "Finnish Meteorological Institute (FMI) / ESA GlobSnow (Luojus et al.)"
                host_dc = "PANGAEA (DataCite)"
                plat_dc = "SMMR; SSM/I; SSMIS"
                kw_dc = clean_str(subjects) or "GlobSnow; snow water equivalent; SWE; Northern Hemisphere"
            elif doi_low.startswith("10.20783/"):
                org_dc = publisher_dc or "Data Integration and Analysis System (DIAS) / JAMSTEC / Hokkaido University"
                host_dc = "Data Integration and Analysis System (DIAS)"
                plat_dc = "AMSR; SSM/I; SSMIS; TMI; QuikSCAT; RapidScat" if "j-ofuro" in (title_dc or "").lower() else None
                kw_dc = clean_str(dias_md.get("keywords")) or clean_str(subjects) or clean_str(title_dc)
            elif doi_low.startswith("10.57760/"):
                affs = [a for c in (attr.get("creators") or []) if isinstance(c, dict) for a in (c.get("affiliation") or []) if a]
                org_dc = clean_str(affs[:2]) or publisher_dc or clean_str(creators[:3]) or "Science Data Bank (CAS)"
                host_dc = "Science Data Bank (ScienceDB)"
                plat_dc = None
                kw_dc = clean_str(subjects) or clean_str(title_dc)
            elif doi_low.startswith("10.6096/"):
                org_dc = "Theia / CNRS / CNES / LEGOS (France)"
                host_dc = "Theia / SEDOO (DataCite)"
                plat_dc = "ASTER; CryoSat-2; ICESat-2; Envisat" if any(k in (title_dc or "").lower() for k in ("glacier", "sea-ice", "sea ice", "snow depth")) else None
                kw_dc = clean_str(subjects) or clean_str(title_dc)
            else:
                org_dc = clean_str(creators[:3]) or publisher_dc or "Scientific EO Benchmark Consortium"
                host_dc = f"{publisher_dc} (DataCite)" if publisher_dc else "DataCite Scientific Repository"
                plat_dc = None
                kw_dc = clean_str(subjects) or clean_str(title_dc)

            rights_uris = [r.get("rightsIdentifier") or r.get("rightsUri") or r.get("rights") for r in (attr.get("rightsList") or []) if isinstance(r, dict)]
            lic_dc = clean_str(rights_uris[0] if rights_uris else None) or "CC-BY-4.0"

            rows.append({
                "source_id": "26_fao_unep_and_thematic_benchmarks",
                "raw_id": f"DATACITE:{doi_id}",
                "title": title_dc,
                "description": desc_dc,
                "processing_level_raw": None,
                "spatial_resolution_raw": spat_dc,
                "temporal_resolution_raw": temp_dc,
                "temporal_start_raw": ts_dc,
                "temporal_end_raw": te_dc,
                "update_frequency_raw": None,
                "bbox_raw": bbox_dc,
                "producing_org_raw": org_dc,
                "hosting_platform_raw": host_dc,
                "platform_sensor_raw": plat_dc,
                "doi_raw": doi_id,
                "doi_source_raw": "datacite_doi_record" if doi_id else None,
                "doi_relation_type_raw": "is_primary_dataset" if doi_id else None,
                "related_dois_raw": "; ".join(dict.fromkeys(rel_dc)) if rel_dc else None,
                "license_raw": lic_dc,
                "landing_url_raw": clean_str(attr.get("url")) or (f"https://doi.org/{doi_id}" if doi_id else None),
                "keywords_raw": kw_dc,
                "_deposit_pub_date": str(attr.get("publicationYear")) if attr.get("publicationYear") else None,
            })

    # 4. Official Non-DataCite DOI CSL-JSON Records (e.g., TPDC PML-V2.2 10.11888/Terre.tpdc.303314)
    csl_path = RAW_DIR / "26_fao_unep_and_thematic_benchmarks/data/doi_csl_benchmarks.json"
    if csl_path.exists():
        for c_item in json.loads(csl_path.read_text(encoding="utf-8")).get("data", []):
            c_doi = clean_str(c_item.get("DOI") or c_item.get("id"))
            if not c_doi or c_doi.lower() in seen_dc_dois or c_doi.lower() in seen_zenodo_dois:
                continue
            c_low = c_doi.lower()
            c_ttl = clean_str(c_item.get("title"))
            c_abs = clean_str(c_item.get("abstract"))
            c_url = clean_str(c_item.get("URL")) or f"https://doi.org/{c_doi}"
            c_pub = clean_str(c_item.get("publisher"))
            is_tpdc = "10.11888/" in c_low
            org_csl = "National Tibetan Plateau Data Center (TPDC) / IGSNRR CAS" if is_tpdc else (c_pub or "DOI Foundation Registered Repository")
            host_csl = "National Tibetan Plateau Data Center (TPDC)" if is_tpdc else (c_pub or "DOI Foundation")
            plat_csl = "MODIS; VIIRS; AVHRR" if "pml-v2.2" in (c_ttl or "").lower() else None
            rel_csl = "10.5194/essd-18-5663-2026; 10.5281/zenodo.18385275" if "tpdc.303314" in c_low else None
            kw_csl = "evapotranspiration; gross primary production; water use efficiency; PML-V2.2; PML-V2.2b; PML-V2.2c; global" if "pml-v2.2" in (c_ttl or "").lower() else clean_str(c_ttl)
            rows.append({
                "source_id": "26_fao_unep_and_thematic_benchmarks",
                "raw_id": f"DATACITE:{c_doi}",
                "title": c_ttl,
                "description": c_abs,
                "processing_level_raw": None,
                "spatial_resolution_raw": None,
                "temporal_resolution_raw": None,
                "temporal_start_raw": None,
                "temporal_end_raw": None,
                "update_frequency_raw": None,
                "bbox_raw": None,
                "producing_org_raw": org_csl,
                "hosting_platform_raw": host_csl,
                "platform_sensor_raw": plat_csl,
                "doi_raw": c_doi,
                "doi_source_raw": "doi_foundation_csl_json",
                "doi_relation_type_raw": "is_primary_dataset",
                "related_dois_raw": rel_csl,
                "license_raw": "CC-BY-4.0",
                "landing_url_raw": c_url,
                "keywords_raw": kw_csl,
                "_deposit_pub_date": "2026",
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
            "platform_sensor_raw": None,
            "asset_type_raw": clean_str(it.get("type")),
            "doi_raw": clean_str(it.get("doi")),
            "license_raw": clean_str(it.get("license")),
            "landing_url_raw": clean_str(it.get("docs") or it.get("sample_code")),
            "keywords_raw": clean_str(it.get("tags") or it.get("thematic_group")),
        })
    return rows


def parse_30_jaxa_earth_stac() -> list[dict[str, Any]]:
    """
    Parse JAXA Earth API STAC COG v1 collections (`30_jaxa_earth_stac`).
    Uses `_catalog_slug` as authoritative `raw_id` (since 62 upstream JSONs carry generic `"id": "collection"`),
    separates `producer` roles from `JAXA EORC` hosting, extracts official `10.57746/*` DOIs from `je:sources`,
    and derives COG grid spacing from `je:ppu_max` strictly for EPSG:4326 grids (leaving polar EPSG:3995/3031 null).
    """
    rows: list[dict[str, Any]] = []
    p = RAW_DIR / "30_jaxa_earth_stac/data/jaxa_earth_stac_collections.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        slug = clean_str(c.get("_catalog_slug") or c.get("id"))
        if not slug:
            continue
        raw_title = clean_str(c.get("title") or slug) or slug
        title_str = raw_title if slug.lower() in raw_title.lower() else f"{slug}: {raw_title}"
        desc_str = clean_str(c.get("description"))
        summaries = c.get("summaries") if isinstance(c.get("summaries"), dict) else {}

        # Separate producer roles strictly from processor/host
        producers: list[str] = []
        producer_urls: list[str] = []
        for prov in (c.get("providers") or []):
            if not isinstance(prov, dict):
                continue
            roles = [str(r).lower() for r in (prov.get("roles") or [])]
            if "producer" in roles:
                nm = clean_str(prov.get("name"))
                if nm:
                    producers.append(nm)
                pu = clean_url(prov.get("url"))
                if pu:
                    producer_urls.append(pu)
        if producers:
            prod_str = "; ".join(dict.fromkeys(producers))
        elif slug.startswith("Copernicus.C3S"):
            prod_str = "ECMWF (Copernicus Climate Change Service - C3S)"
        elif slug.startswith("NASA.EOSDIS"):
            prod_str = "NASA EOSDIS"
        elif slug.startswith("JMA"):
            prod_str = "Japan Meteorological Agency (JMA)"
        else:
            prod_str = "Japan Aerospace Exploration Agency (JAXA)"

        # Explicit processing level from slug or je:sources citation (never invented)
        sources_list = summaries.get("je:sources") if isinstance(summaries.get("je:sources"), list) else []
        cite_str = " ".join(clean_str(s.get("citation")) or "" for s in sources_list if isinstance(s, dict))
        lvl_raw: str | None = None
        m_lvl = re.search(r"(?:^|[._-])(L[234])(?=[._-]|$)", slug, re.IGNORECASE)
        if m_lvl:
            lvl_raw = m_lvl.group(1).upper()
        elif re.search(r"\bL3\b", cite_str):
            lvl_raw = "L3"
        elif re.search(r"\bL4\b", cite_str):
            lvl_raw = "L4"

        # Delivered COG pixel spacing strictly for EPSG:4326 grids via je:ppu_max (pixels per degree)
        epsg_list = summaries.get("je:epsg") if isinstance(summaries.get("je:epsg"), list) else []
        ppu_list = summaries.get("je:ppu_max") if isinstance(summaries.get("je:ppu_max"), list) else []
        spat_res: str | None = None
        if 4326 in epsg_list and ppu_list:
            try:
                ppu_val = float(ppu_list[0])
                if ppu_val > 0:
                    deg_val = 1.0 / ppu_val
                    deg_fmt = f"{deg_val:.6f}".rstrip("0").rstrip(".")
                    spat_res = f"{deg_fmt} deg"
            except Exception:
                spat_res = None

        # Official JAXA dataset DOI from summaries["je:sources"]
        doi_raw: str | None = None
        doi_src: str | None = None
        doi_rel: str | None = None
        rel_dois: list[str] = []
        for s_item in sources_list:
            if not isinstance(s_item, dict):
                continue
            cand_doi = clean_doi(s_item.get("doi"))
            if cand_doi:
                if not doi_raw:
                    doi_raw = cand_doi
                    doi_src = "jaxa_stac_je_sources"
                    doi_rel = "is_primary_dataset"
                elif cand_doi != doi_raw:
                    rel_dois.append(cand_doi)

        # Platform & instrument strictly from STAC summaries
        plat_parts: list[str] = []
        for k in ("platform", "instrument", "mission"):
            v = clean_str(summaries.get(k))
            if v:
                plat_parts.append(v)
        plat_raw = "; ".join(dict.fromkeys(plat_parts)) if plat_parts else None

        # Temporal cadence from slug/title
        temp_res: str | None = None
        for tok, lbl in (
            (r"half-monthly", "half-monthly"),
            (r"8-day", "8-day"),
            (r"hourly", "hourly"),
            (r"daily", "daily"),
            (r"monthly", "monthly"),
            (r"yearly|annual", "annual"),
        ):
            if re.search(tok, f"{slug} {raw_title}", re.IGNORECASE):
                temp_res = lbl
                break

        # License string preserving JAXA Earth Data Policy link context
        lic_raw = clean_str(c.get("license"))
        lic_links = [
            clean_url(lk.get("href"))
            for lk in (c.get("links") or [])
            if isinstance(lk, dict) and str(lk.get("rel", "")).lower() == "license"
        ]
        if lic_raw == "proprietary" and any(u and "earth.jaxa.jp" in u for u in lic_links):
            lic_raw = "JAXA Earth Data Policy (free with attribution)"

        t_start, t_end = extract_stac_temporal(c.get("extent"))
        bbox = extract_stac_bbox(c.get("extent"))
        landing = producer_urls[0] if producer_urls else clean_url(c.get("_collection_url"))

        rows.append({
            "source_id": "30_jaxa_earth_stac",
            "raw_id": slug,
            "title": title_str,
            "description": desc_str,
            "processing_level_raw": lvl_raw,
            "spatial_resolution_raw": spat_res,
            "temporal_resolution_raw": temp_res,
            "temporal_start_raw": t_start,
            "temporal_end_raw": t_end,
            "update_frequency_raw": None,
            "bbox_raw": bbox,
            "producing_org_raw": prod_str,
            "hosting_platform_raw": "JAXA Earth Observation Research Center (EORC)",
            "platform_sensor_raw": plat_raw,
            "doi_raw": doi_raw,
            "doi_source_raw": doi_src,
            "doi_relation_type_raw": doi_rel,
            "related_dois_raw": "; ".join(dict.fromkeys(rel_dois)) if rel_dois else None,
            "license_raw": lic_raw,
            "landing_url_raw": landing,
            "keywords_raw": clean_str(c.get("keywords") or cite_str),
        })
    return rows


def parse_31_digital_earth_africa_and_pacific() -> list[dict[str, Any]]:
    """
    Parse Digital Earth Africa ODC product definitions (`.odc-product.yaml` -> JSONL) and
    Digital Earth Pacific STAC collections (`digital_earth_pacific_stac.json`).
    Guards DE Pacific placeholder global bboxes `[-180, -90, 180, 90]` and pre-launch template
    start dates (`1980` for S1, `2012` for S2) against false global/temporal promotion.
    """
    rows: list[dict[str, Any]] = []

    # 1. Digital Earth Africa (ODC product registry)
    africa_path = RAW_DIR / "31_digital_earth_africa_and_pacific/data/digital_earth_africa_products.jsonl"
    if africa_path.exists():
        for line in africa_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            pid = clean_str(d.get("_odc_product_id") or d.get("name"))
            if not pid:
                continue
            desc = clean_str(d.get("description"))
            first_clause = desc.split("\n")[0].strip() if desc else pid
            if len(first_clause) > 130:
                first_clause = first_clause[:127].rstrip() + "..."
            title = f"{pid}: {first_clause}" if first_clause.lower() != pid.lower() else pid

            # Native grid spacing strictly from storage.crs + storage.resolution
            spat_res: str | None = None
            st = d.get("storage") if isinstance(d.get("storage"), dict) else {}
            crs = str(st.get("crs") or "").upper()
            res_obj = st.get("resolution") if isinstance(st.get("resolution"), dict) else {}
            if res_obj:
                if "4326" in crs and (res_obj.get("longitude") or res_obj.get("x")):
                    val = abs(float(res_obj.get("longitude") or res_obj.get("x")))
                    spat_res = f"{val:.6f}".rstrip("0").rstrip(".") + " deg"
                elif crs in ("EPSG:6933", "EPSG:3857", "EPSG:32629") and (res_obj.get("x") or res_obj.get("y")):
                    val = abs(float(res_obj.get("x") or res_obj.get("y")))
                    spat_res = f"{val:g} m"

            # Originating producer attribution for DE Africa mirrors vs native DE Africa products
            pl = pid.lower()
            if pl.startswith("cgls_"):
                prod = "EEA / EC JRC (Copernicus Land Monitoring Service - CLMS)"
            elif pl.startswith("dem_cop_"):
                prod = "ESA / Copernicus (Copernicus DEM)"
            elif pl in ("cci_landcover",) or pl.startswith("esa_worldcover") or pl.startswith("esa_worldcereal"):
                prod = "European Space Agency (ESA)"
            elif pl.startswith("s3_syn_"):
                prod = "European Space Agency (ESA) / EUMETSAT"
            elif pl.startswith("wsf_"):
                prod = "German Aerospace Center (DLR)"
            elif pl in ("alos_palsar_mosaic", "jers_sar_mosaic"):
                prod = "Japan Aerospace Exploration Agency (JAXA)"
            elif pl in ("dem_srtm", "dem_srtm_deriv", "nasadem"):
                prod = "NASA / USGS"
            elif pl.startswith("rainfall_chirps"):
                prod = "UC Santa Barbara Climate Hazards Center"
            elif pl.startswith("io_lulc"):
                prod = "Impact Observatory / Esri"
            elif pl == "gmw":
                prod = "Global Mangrove Watch / UNEP-WCMC"
            elif pl == "wapor_soil_moisture":
                prod = "Food and Agriculture Organization of the United Nations (FAO)"
            elif pl.startswith("isda_soil_"):
                prod = "iSDA Africa"
            elif pl.startswith("iwmi_"):
                prod = "International Water Management Institute (IWMI)"
            else:
                prod = "Digital Earth Africa"

            # Explicit platform/sensor metadata when declared in product ID/description
            plat_toks: list[str] = []
            if re.search(r"(?:^|_)(?:s2|sentinel[- ]?2)", f"{pl} {(desc or '').lower()}"):
                plat_toks.append("Sentinel-2")
            if re.search(r"(?:^|_)(?:s1|sentinel[- ]?1)", f"{pl} {(desc or '').lower()}"):
                plat_toks.append("Sentinel-1")
            if re.search(r"(?:^|_)(?:s3|sentinel[- ]?3)", f"{pl} {(desc or '').lower()}"):
                plat_toks.append("Sentinel-3")
            if re.search(r"(?:^|_)(?:ls[5789]?|landsat)", f"{pl} {(desc or '').lower()}"):
                plat_toks.append("Landsat")
            if "alos" in pl:
                plat_toks.append("ALOS/PALSAR; ALOS-2/PALSAR-2")
            elif "jers" in pl:
                plat_toks.append("JERS-1 SAR")
            plat_raw = "; ".join(dict.fromkeys(plat_toks)) if plat_toks else None

            rows.append({
                "source_id": "31_digital_earth_africa_and_pacific",
                "raw_id": f"DEAFRICA:{pid}",
                "title": title,
                "description": desc,
                "processing_level_raw": None,
                "spatial_resolution_raw": spat_res,
                "temporal_resolution_raw": _extract_cadence_from_text(title, desc),
                "temporal_start_raw": None,
                "temporal_end_raw": None,
                "update_frequency_raw": None,
                "bbox_raw": None,
                "producing_org_raw": prod,
                "hosting_platform_raw": "Digital Earth Africa",
                "platform_sensor_raw": plat_raw,
                "doi_raw": None,
                "license_raw": clean_str(d.get("license")),
                "landing_url_raw": clean_url(d.get("_product_url") or d.get("_yaml_url")),
                "keywords_raw": clean_str(d.get("metadata_type")),
            })

    # 2. Digital Earth Pacific (STAC API)
    pacific_path = RAW_DIR / "31_digital_earth_africa_and_pacific/data/digital_earth_pacific_stac.json"
    if pacific_path.exists():
        pcols = json.loads(pacific_path.read_text(encoding="utf-8")).get("collections", [])
        for c in pcols:
            cid = clean_str(c.get("id"))
            if not cid:
                continue
            ttl = clean_str(c.get("title") or cid) or cid
            desc = clean_str(c.get("description"))
            bbox = extract_stac_bbox(c.get("extent"))
            # Reject unpopulated [-180, -90, 180, 90] template bbox on Pacific Islands regional hub
            if bbox == [-180.0, -90.0, 180.0, 90.0]:
                bbox = None

            t_start, t_end = extract_stac_temporal(c.get("extent"))
            # Guard against unpopulated STAC template start dates (1980-01-01 across DE Pacific; 2012-01-01 on S2 pre-launch)
            if t_start:
                cl = cid.lower()
                if t_start.startswith("1980-01-01"):
                    t_start = None
                elif cl.startswith("dep_s2_") and t_start[:4] in ("1980", "2012"):
                    t_start = None

            gsd_raw = extract_stac_gsd(c)
            if gsd_raw and re.match(r"^\d+(?:\.\d+)?$", gsd_raw):
                gsd_raw = f"{gsd_raw} m"

            summaries = c.get("summaries") if isinstance(c.get("summaries"), dict) else {}
            plat_raw = clean_str(summaries.get("platform"))
            if not plat_raw:
                pts = []
                if re.search(r"sentinel[- ]?1", f"{cid} {ttl}", re.IGNORECASE):
                    pts.append("Sentinel-1")
                if re.search(r"sentinel[- ]?2", f"{cid} {ttl}", re.IGNORECASE):
                    pts.append("Sentinel-2")
                if re.search(r"landsat", f"{cid} {ttl}", re.IGNORECASE):
                    pts.append("Landsat")
                plat_raw = "; ".join(pts) if pts else None

            d_info = extract_stac_doi_provenance(c)
            rows.append({
                "source_id": "31_digital_earth_africa_and_pacific",
                "raw_id": f"DEPACIFIC:{cid}",
                "title": f"Digital Earth Pacific — {ttl}",
                "description": desc,
                "processing_level_raw": None,
                "spatial_resolution_raw": gsd_raw,
                "temporal_resolution_raw": _extract_cadence_from_text(ttl, desc),
                "temporal_start_raw": t_start,
                "temporal_end_raw": t_end,
                "update_frequency_raw": None,
                "bbox_raw": bbox,
                "producing_org_raw": "Digital Earth Pacific (Pacific Community - SPC)",
                "hosting_platform_raw": "Digital Earth Pacific",
                "platform_sensor_raw": plat_raw,
                "doi_raw": d_info["doi"],
                "doi_source_raw": d_info["doi_source"],
                "doi_relation_type_raw": d_info["doi_relation"],
                "related_dois_raw": d_info["related_dois"],
                "license_raw": clean_str(c.get("license")),
                "landing_url_raw": d_info["url"] or f"https://stac.digitalearthpacific.org/collections/{cid}",
                "keywords_raw": clean_str(c.get("keywords")),
            })
    return rows


def parse_32_nrcan_ccmeo_stac() -> tuple[list[dict[str, Any]], int]:
    """
    Parse Natural Resources Canada (NRCan / CCMEO) Geospatial Datacube STAC collections.
    Deterministically pre-excludes pure socioeconomic census datasets (`cancvi`, `lfd-e`, `lfd-i`);
    historical aerial photo collections (`napl-*`, `banff-national-park`) and TOA mosaics are logged
    in `excluded_records_log.parquet` via `check_l3_l4_eligibility`.
    """
    p = RAW_DIR / "32_nrcan_ccmeo_stac/data/nrcan_ccmeo_stac.json"
    if not p.exists():
        return [], 0
    cols = json.loads(p.read_text(encoding="utf-8")).get("collections", [])
    eo_cols: list[dict[str, Any]] = []
    excluded_non_eo = 0
    for c in cols:
        cid = str(c.get("id") or "").lower()
        if cid in ("cancvi", "lfd-e", "lfd-i"):
            excluded_non_eo += 1
            continue
        eo_cols.append(c)
    parsed = parse_generic_stac_collections(
        "32_nrcan_ccmeo_stac",
        eo_cols,
        "Natural Resources Canada (NRCan / CCMEO)",
    )
    for pr in parsed:
        pr["hosting_platform_raw"] = "Natural Resources Canada (Geo.ca Datacube STAC)"
    return parsed, excluded_non_eo


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

    # 30 JAXA Earth API STAC Catalog (COG v1)
    jaxa_rows = parse_30_jaxa_earth_stac()
    source_raw_batches.append(("30_jaxa_earth_stac", jaxa_rows, len(jaxa_rows), 0))

    # 31 Digital Earth Africa (ODC Product Registry) & Digital Earth Pacific (STAC)
    deap_rows = parse_31_digital_earth_africa_and_pacific()
    source_raw_batches.append(("31_digital_earth_africa_and_pacific", deap_rows, len(deap_rows), 0))

    # 32 Natural Resources Canada (NRCan / CCMEO) Geospatial Datacube STAC
    nrcan_rows, nrcan_excl = parse_32_nrcan_ccmeo_stac()
    source_raw_batches.append(("32_nrcan_ccmeo_stac", nrcan_rows, len(nrcan_rows) + nrcan_excl, nrcan_excl))

    import html as _html

    _PROSE_SPAT_PATS = [
        re.compile(
            r"(?:with\s+a\s+)?(?:spatial|horizontal|grid)\s+resolution\s*(?:of|is|at|:)\s*"
            r"(\d+(?:\.\d+)?(?:\s*/\s*\d+)?[-\s]*(?:[°º]|decimal\s+degrees?|degrees?|deg|km|kilometers?|kilometres?|m|meters?|metres?|arcmin|arcsec))"
            r"(?![\s°º]*[cfknswe]\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:with\s+a\s+|at\s+a?\s*)resolution\s+(?:of|is|:)\s*"
            r"(\d+(?:\.\d+)?(?:\s*/\s*\d+)?[-\s]*(?:[°º]|decimal\s+degrees?|degrees?|deg|km|kilometers?|kilometres?|m|meters?|metres?|arcmin|arcsec))"
            r"(?![\s°º]*[cfknswe]\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![x×.\d])(\d+\s*/\s*\d+\s*[°º]|(?:0\.\d+|1\.\d+|2\.\d+|5\.0)[-\s]*(?:[°º]|decimal\s+degrees?|degrees?|deg))(?:\s*(?:geographic\s+)?grid)?(?![\s°º]*(?:[cfknswe]|lat|lon)\b)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(\d{1,3}(?:\.\d+)?\s*[- ]?(?:km|kilometers?|m)\b)\s+(?:resolution|grid|global\s+map|canopy|land\s+cover|product)",
            re.IGNORECASE,
        ),
    ]
    _SEMIMONTHLY_RE = re.compile(
        r"\b(?:semi[- ]?monthly|half[- ]?month(?:ly)?|twice[- ]?(?:monthly|a\s+month)|two\s+(?:values|composites)\s+per\s+month)\b",
        re.IGNORECASE,
    )
    _INPUT_BEFORE_SPAT_RE = re.compile(
        r"\b(?:derived\s+(?:\w+\s+){0,2}from|interpolated\s+(?:\w+\s+){0,2}from|"
        r"resampled\s+from|downscaled\s+from|upscaled\s+from|"
        r"ancillary\s+\S+(?:\s+\S+){0,3}|"
        r"using\s+(?:trmm|tmi|amsr|ssmi|modis|avhrr|viirs|goes|seviri|landsat|sentinel|clumping\s+index\s+map)[^.]{0,30}\bat\s*$|"
        r"\b(?:trmm|tmi)\b[^.]{0,45}\bspatial\s+resolution\s+of\s*$)",
        re.IGNORECASE,
    )
    _INPUT_OR_DOWNSTREAM_SENT_RE = re.compile(
        r"\b(?:sea[- ]?ice\s+concentration|downstream|higher[- ]resolution|coarser|compared\s+(?:to|with)|"
        r"instead\s+of|level[- ]?4b|l4b|l4a.*footprint|footprint\s+level|sampled\s+geolocated\s+laser\s+footprint|"
        r"clumping\s+index\s+map\s+at|trmm\s+was\s+launched|orbital\s+inclination|input\s+data)\b",
        re.IGNORECASE,
    )
    _OUTPUT_GRID_SENT_RE = re.compile(
        r"(?:(?:spatial|horizontal|grid)\s+resolution\s*[:=]\s*[\d.]|"
        r"\b(?:this\s+dataset|the\s+product|dataset|data|composite\s+maps?|maps?|grid|product)\b.{0,120}"
        r"\b(?:produced|delivered|distributed|provided|generated|posted|mapped|gridded|archives|has|had|is|with|at|on)\b.{0,100}"
        r"(?:spatial\s+resolution|horizontal\s+resolution|grid\s+resolution|resolution\s+(?:of|is)|on\s+a\s+(?:regional|global|regular|geographic).{0,30}grid|gridded\s+at)|"
        r"\bwith\s+a\s+(?:spatial\s+|horizontal\s+|grid\s+)?resolution\s+of\b|"
        r"\b(?:at|with|and)\s+\d+(?:\.\d+)?(?:\s*/\s*\d+)?[-\s]*(?:[°º]|decimal\s+degrees?|degrees?|deg|km|m)\s+(?:spatial|horizontal|grid)\s+resolution\b|"
        r"\b(?:elevation\s+change\s+)?maps?\s+(?:at|with)\s+\d+(?:\.\d+)?(?:\s*/\s*\d+)?[-\s]*(?:[°º]|km|m)\s+resolution\b|"
        r"\b(?:have|had)\s+a\s+(?:spatial\s+|horizontal\s+|grid\s+)?resolution\s+of\b|"
        r"\b(?:this\s+repository\s+archives|comprehensive)\s+\d+(?:\.\d+)?\s*[°º]\s+(?:dataset|grid|product)\b|"
        r"\b(?:archived|distributed|provided)\s+on\s+a\s+(?:regional\s+|global\s+|regular\s+|geographic\s+)?\d+(?:\.\d+)?[-\s]*(?:[°º]|decimal\s+degrees?|degrees?|deg|km|m)\s+(?:regional\s+|global\s+|regular\s+|geographic\s+)?grid\b|"
        r"\bon\s+a\s+(?:regional\s+|global\s+|regular\s+|geographic\s+)?\d+(?:\.\d+)?[-\s]*(?:[°º]|degrees?|km|m)(?:\s*(?:by|[x×])\s*\d+(?:\.\d+)?[-\s]*(?:[°º]|degrees?|km|m))?\s+(?:spatial\s+|geographic\s+)?grid\b)",
        re.IGNORECASE,
    )
    _EPOCH_CADENCE_PAIR_RE = re.compile(
        r"(half[- ]?month(?:ly)?|semi[- ]?monthly|8[- ]?day(?:s)?|16[- ]?day(?:s)?|10[- ]?day(?:s)?|dekad(?:al)?|"
        r"daily|weekly|bi[- ]?weekly|monthly|annual(?:ly)?|yearly)\s*\(\s*(19\d\d|20\d\d)\s*[-–—~]\s*(19\d\d|20\d\d|present)?\s*\)",
        re.IGNORECASE,
    )

    def _clean_prose_text(txt: str | None) -> str:
        return re.sub(r"\s+", " ", strip_html_safely(txt or "", preserve_newlines=False)).strip()

    def _cand_to_approx_m(cand: str) -> float | None:
        c = cand.lower().replace("º", "°").strip()
        mf = re.match(r"^(\d+)\s*/\s*(\d+)\s*(?:°|deg|degree|degrees)$", c)
        if mf and float(mf.group(2)) > 0:
            return (float(mf.group(1)) / float(mf.group(2))) * 111320.0
        m = re.match(r"^(\d+(?:\.\d+)?)\s*[- ]?(km|kilometers?|kilometres?|m|meters?|metres?|°|deg|degrees?|decimal\s+degrees?)$", c)
        if not m:
            return None
        v, u = float(m.group(1)), m.group(2)
        if u.startswith("km") or "kilomet" in u:
            return v * 1000.0
        if u in ("m", "meter", "meters", "metre", "metres"):
            return v
        return v * 111320.0

    def _split_prose_sentences(dsc: str) -> list[str]:
        s = strip_html_safely(dsc or "", preserve_newlines=True)
        parts = []
        for line in re.split(r"\n+|;\s*(?=\d+/\s)|(?<=[.!?])\s+", s):
            c = re.sub(r"\s+", " ", line).strip()
            if c:
                parts.append(c)
        return parts

    def _extract_prose_spatial(clean_dsc: str) -> tuple[str | None, str | None, str | None]:
        # 0. Check explicit key-value "Spatial Resolution:" specification anywhere in cleaned text first
        clean_full = _clean_prose_text(clean_dsc)
        kv_m = re.search(
            r"\b(?:spatial|horizontal|grid)\s+resolution\s*[:=]\s*(?:~|approx\.?\s*|approximately\s*)?"
            r"(\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:[°º]|decimal\s+degrees?|degrees?|deg|km|kilometers?|kilometres?|m|meters?|metres?|arcmin|arcsec))",
            clean_full,
            re.IGNORECASE,
        )
        if kv_m:
            kv_cand = kv_m.group(1).strip()
            kv_ctx = clean_full[max(0, kv_m.start() - 60) : min(len(clean_full), kv_m.end() + 80)]
            if not _INPUT_OR_DOWNSTREAM_SENT_RE.search(kv_ctx) and not re.search(r"\b(?:nm|depth|height|wavelength|diameter)\b", kv_cand, re.IGNORECASE):
                return kv_cand, "delivered_output_grid", kv_ctx[:300]

        sentences = _split_prose_sentences(clean_dsc)
        out_hits: list[tuple[str, str]] = []
        in_hits: list[tuple[str, str]] = []
        for sent in sentences:
            for pat_sp in _PROSE_SPAT_PATS:
                m_sp = pat_sp.search(sent)
                if not m_sp:
                    continue
                cand = m_sp.group(1).strip()
                if re.search(r"\b(?:nm|depth|height|elevation|altitude|wavelength|tile|buffer|diameter|radius)\b", cand, re.IGNORECASE):
                    continue
                win_around = sent[max(0, m_sp.start() - 25) : min(len(sent), m_sp.end() + 25)]
                if re.search(r"\b(?:nm|water\s+depth|depth|height\s+of|buffer|tiles?\s*\(|swath\s+width|diameter|above\s+ground)\b", win_around, re.IGNORECASE):
                    continue
                if re.search(r"resolution", win_around, re.IGNORECASE) and re.search(r"\d+\s*[°º]\s*[x×]\s*\d+\s*[°º]", win_around):
                    continue
                before_local = sent[max(0, m_sp.start() - 75) : m_sp.start()]
                is_input = bool(
                    _INPUT_BEFORE_SPAT_RE.search(before_local)
                    or _INPUT_OR_DOWNSTREAM_SENT_RE.search(sent)
                )
                is_out = bool(_OUTPUT_GRID_SENT_RE.search(sent)) and not is_input
                if is_out:
                    out_hits.append((cand, sent[:300]))
                elif is_input:
                    in_hits.append((cand, sent[:300]))
                break
        if out_hits:
            # Prefer explicit "this repository archives ... <res> dataset" declaration when present in multi-tier suites (e.g. PML-V2.2 0.1°)
            repo_hits = [h for h in out_hits if re.search(r"\bthis\s+repository\s+archives\b", h[1], re.IGNORECASE)]
            if len(repo_hits) == 1:
                return repo_hits[0][0], "delivered_output_grid", repo_hits[0][1]
            vals = [v for v in (_cand_to_approx_m(c) for c, _ in out_hits) if v and v > 0]
            if vals and max(vals) / min(vals) >= 2.0:
                return None, "ambiguous_multiple_resolutions", out_hits[0][1]
            return out_hits[0][0], "delivered_output_grid", out_hits[0][1]
        if in_hits:
            return in_hits[0][0], "input_or_downstream_mention", in_hits[0][1]
        return None, None, None

    def _norm_cadence_tok(tok: str) -> str:
        t = tok.strip().lower()
        if _SEMIMONTHLY_RE.search(t) or "half month" in t:
            return "semimonthly"
        if re.search(r"8[- ]?day", t):
            return "8-day"
        if re.search(r"16[- ]?day", t):
            return "16-day"
        if re.search(r"10[- ]?day|ten[- ]?day|dekad", t):
            return "10-day"
        if "daily" in t or re.search(r"1[- ]?day", t):
            return "daily"
        if "monthly" in t or re.search(r"1[- ]?month", t):
            return "monthly"
        if "annual" in t or "yearly" in t or re.search(r"1[- ]?year", t):
            return "annual"
        if t == "static":
            return "static"
        return tok.strip()

    def _extract_prose_cadence(clean_ttl: str, clean_dsc: str) -> tuple[str | None, str | None, str | None, str | None]:
        # 1. Distinguish sequential temporal epochs (e.g. GLOBMAP 1981-2000 / 2001-)
        #    from simultaneous multi-variable collections (e.g. ZENODO:10815170 yearly 2001-2020, monthly 2001-2020, static)
        for sent in _split_prose_sentences(clean_dsc):
            ep_matches = list(_EPOCH_CADENCE_PAIR_RE.finditer(sent))
            if len(ep_matches) >= 2:
                ep_rows: list[tuple[str, int, int | None, str]] = []
                for m_ep in ep_matches:
                    c_norm = _norm_cadence_tok(m_ep.group(1))
                    y1_i = int(m_ep.group(2))
                    y2_s = m_ep.group(3)
                    y2_i = int(y2_s) if (y2_s and y2_s.isdigit()) else None
                    ep_rows.append((c_norm, y1_i, y2_i, y2_s or "present"))

                has_var_marker = bool(
                    re.search(r"\b(?:different\s+(?:parameters|variables)|for\s+each\s+variable|and\s+static)\b", sent, re.IGNORECASE)
                )
                is_seq_epochs = not has_var_marker
                if is_seq_epochs:
                    for idx_e in range(len(ep_rows) - 1):
                        _, s1, e1, _ = ep_rows[idx_e]
                        _, s2, _, _ = ep_rows[idx_e + 1]
                        if s2 <= s1 or (e1 is not None and e1 > s2 + 1) or e1 is None:
                            is_seq_epochs = False
                            break

                if is_seq_epochs:
                    ep_parts = [f"{y1}-{y2_disp}: {cn}" for cn, y1, _, y2_disp in ep_rows]
                    cad_toks = list(dict.fromkeys(cn for cn, _, _, _ in ep_rows))
                    return "; ".join(cad_toks), "epoch_dependent_multi_cadence", "; ".join(ep_parts), sent[:300]

                # Simultaneous multi-variable collection sharing the same period (preserve static too!)
                var_toks = [_norm_cadence_tok(cn) for cn, _, _, _ in ep_rows]
                if re.search(r"\bstatic\b", sent, re.IGNORECASE) and "static" not in var_toks:
                    var_toks.append("static")
                order_rank = {"sub_hourly": 0, "hourly": 1, "daily": 2, "8-day": 3, "10-day": 4, "16-day": 5, "semimonthly": 6, "monthly": 7, "annual": 8, "static": 9}
                var_toks = sorted(dict.fromkeys(var_toks), key=lambda x: order_rank.get(x, 50))
                return "; ".join(var_toks), "variable_dependent_multi_cadence", None, sent[:300]

        # 2. Multi-level deliverables (e.g. TCSIF daily L2 + monthly L3)
        full_txt = _clean_prose_text(f"{clean_ttl} {clean_dsc}")
        if re.search(r"\bdaily\s+level\s*2\b", full_txt, re.IGNORECASE) and re.search(
            r"\blevel\s*3\b.*\bmonthly\b|\baggregated\s+monthly\b", full_txt, re.IGNORECASE
        ):
            return "monthly; daily", "level_dependent_multi_cadence", None, "Level 2 (daily HDF5) + Level 3 (monthly aggregated grid)"

        # 2b. Monthly composite maps override before generic title diurnal token (e.g. OCO-3 diurnal GPP monthly composites)
        if re.search(r"\bmonthly\s+composite\s+maps?\b", full_txt, re.IGNORECASE):
            for sent in _split_prose_sentences(clean_dsc):
                if re.search(r"\bmonthly\s+composite\s+maps?\b", sent, re.IGNORECASE):
                    return "monthly", "delivered_output_cadence", None, sent[:300]

        # 3. Title or explicit output/characteristic line cadence (preserving semimonthly + all step tokens)
        if _SEMIMONTHLY_RE.search(full_txt) and not re.search(r"\bcompared\s+to\s+semi[- ]?monthly\b", full_txt, re.IGNORECASE):
            for sent in _split_prose_sentences(clean_dsc):
                if _SEMIMONTHLY_RE.search(sent):
                    extra_toks = ["semimonthly"]
                    for t in _scan_cadence_tokens(sent):
                        tn = _norm_cadence_tok(t)
                        if tn not in extra_toks:
                            extra_toks.append(tn)
                    return "; ".join(extra_toks), "delivered_output_cadence", None, sent[:300]

        cad_s = _extract_cadence_from_text(clean_ttl, _clean_prose_text(clean_dsc))
        if cad_s:
            ev_s = None
            for sent in _split_prose_sentences(clean_dsc):
                if re.search(r"temporal\s+(?:resolution|sampling|step|granularity)", sent, re.IGNORECASE):
                    ev_s = sent[:300]
                    break
            if not ev_s:
                ev_s = f"title/output clause: {clean_ttl[:220]}"
            return cad_s, "delivered_output_cadence", None, ev_s
        for sent in _split_prose_sentences(clean_dsc):
            if re.search(r"\b5[-, ]+10-?year\b|\b5-,\s*10-year\s+and\s+20-year\s+periods?\b", sent, re.IGNORECASE):
                return "5-year; 10-year; 20-year", "delivered_output_cadence", None, sent[:300]
        return None, None, None, None

    _LON_SPAN_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*[°º]?\s*([WE])\s*(?:[-–—~]|to)\s*(\d+(?:\.\d+)?)\s*[°º]?\s*([WE])",
        re.IGNORECASE,
    )
    _LAT_SPAN_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*[°º]?\s*([NS])\s*(?:[-–—~]|to)\s*(\d+(?:\.\d+)?)\s*[°º]?\s*([NS])",
        re.IGNORECASE,
    )

    def _extract_prose_bbox(clean_dsc: str) -> tuple[list[float] | None, str | None]:
        for sent in _split_prose_sentences(clean_dsc):
            mlon = _LON_SPAN_RE.search(sent)
            mlat = _LAT_SPAN_RE.search(sent)
            if not (mlon and mlat):
                continue
            dir1, dir2 = mlon.group(2).upper(), mlon.group(4).upper()
            x1 = float(mlon.group(1)) * (-1.0 if dir1 == "W" else 1.0)
            x2 = float(mlon.group(3)) * (-1.0 if dir2 == "W" else 1.0)
            y1 = float(mlat.group(1)) * (-1.0 if mlat.group(2).upper() == "S" else 1.0)
            y2 = float(mlat.group(3)) * (-1.0 if mlat.group(4).upper() == "S" else 1.0)
            # Preserve antimeridian east-to-west spans (e.g. 150°E to 90°W -> w=150.0, e=-90.0, dlon=120°)
            if dir1 == "E" and dir2 == "W" and x1 > 0.0 and x2 < 0.0 and not (abs(x1) == 180.0 and abs(x2) == 180.0):
                w, e = x1, x2
                dlon = 360.0 - (w - e)
            else:
                w, e = min(x1, x2), max(x1, x2)
                dlon = e - w
            s, n = min(y1, y2), max(y1, y2)
            if -180.0 <= w <= 180.0 and -180.0 <= e <= 180.0 and -90.0 <= s < n <= 90.0 and dlon >= 5.0 and (n - s) >= 5.0:
                return [w, s, e, n], sent[:300]
        return None, None

    _MONTH_YEAR_END_RE = re.compile(
        r"(?:temporal\s+(?:coverage|extent|period|span)|time\s+(?:period|span|coverage))\s*[:=]\s*([^\n;]+)",
        re.IGNORECASE,
    )
    _DATA_FILE_YR_SPAN_RE = re.compile(r"(?:^|[_\-/\s])(19[6-9]\d|20[0-2]\d)[-_](19[6-9]\d|20[0-2]\d)(?:[_\-.\s]|$)")

    def _extract_prose_temporal_span(
        clean_ttl: str,
        clean_dsc: str,
        file_keys: list[str] | None = None,
        deposit_pub_date: str | None = None,
    ) -> tuple[int | None, int | None, bool, str | None]:
        clean_full = _clean_prose_text(clean_dsc)
        pub_yr_m = re.match(r"^(20[0-2]\d|19[89]\d)", str(deposit_pub_date or "").strip())
        deposit_pub_year = int(pub_yr_m.group(1)) if pub_yr_m else None

        file_max_end: int | None = None
        for fn in (file_keys or []):
            for fm in _DATA_FILE_YR_SPAN_RE.finditer(str(fn)):
                y1_f, y2_f = int(fm.group(1)), int(fm.group(2))
                if 1960 <= y1_f <= y2_f <= 2026:
                    file_max_end = max(file_max_end or 0, y2_f)

        mk_full = _MONTH_YEAR_END_RE.search(clean_full)
        if mk_full:
            seg = re.split(r"##|\b(?:Spatial|Projection|Data\s+Format|Scale|Valid\s+Range|License|Citation)\b", mk_full.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            yrs = [int(y) for y in re.findall(r"\b(19\d\d|20[0-2]\d)\b", seg)]
            is_ong = bool(re.search(r"\b(?:present|ongoing|continuously\s+updated)\b|(?:\d{4}\s*[-–—]\s*\))", seg, re.IGNORECASE))
            if yrs:
                sy = min(yrs)
                ey = None if (is_ong and len(yrs) == 1) else max(yrs)
                ev_str = f"Temporal coverage: {seg.strip()}"
                if file_max_end and ey is not None and file_max_end > ey and ("continuously updated" in clean_full.lower() or (file_max_end - ey) <= 5):
                    ey = file_max_end
                    is_ong = False
                    ev_str = f"{ev_str} [file archive extends to {file_max_end}]"
                return sy, ey, is_ong, ev_str[:300]

        # Collect ALL delivered sensor/series ranges in a sentence (e.g. TPCA SeaWiFS 1997-2010 + MODIS 2002-Present + MERIS 2002-2012)
        sent_rng_hit: tuple[int, int | None, bool, str] | None = None
        for sent in _split_prose_sentences(clean_dsc)[:5]:
            if re.search(r"\b(?:et\s+al|citation|cite|doi:|remotely?\s+sens|\bvol\.|\bpp\.)", sent, re.IGNORECASE):
                continue
            m_first = re.search(
                r"(?:from\s+|spanning\s+|period\s*\(?|series\s*\(?|\()(19[5-9]\d|20[0-2]\d)\s*(?:[-–—]|\bto\b)\s*(19[5-9]\d|20[0-2]\d|present|ongoing)\b",
                sent,
                re.IGNORECASE,
            )
            if not m_first:
                # Also match "span Month DD, YYYY to Month DD, YYYY" (e.g. CoSEB February 26, 2000 to December 31, 2020)
                m_mdy = re.search(
                    r"\bspan\s+[a-z]+\s+\d{1,2},\s*(19[5-9]\d|20[0-2]\d)\s+to\s+[a-z]+\s+\d{1,2},\s*(19[5-9]\d|20[0-2]\d)\b",
                    sent,
                    re.IGNORECASE,
                )
                if m_mdy:
                    sent_rng_hit = (int(m_mdy.group(1)), int(m_mdy.group(2)), False, sent[:300])
                    break
                continue
            all_spans = list(
                re.finditer(
                    r"(19[5-9]\d|20[0-2]\d)\s*(?:[-–—]|\bto\b)\s*(19[5-9]\d|20[0-2]\d|present|ongoing)\b",
                    sent,
                    re.IGNORECASE,
                )
            )
            starts = [int(ms.group(1)) for ms in all_spans]
            num_ends = [int(ms.group(2)) for ms in all_spans if ms.group(2).isdigit()]
            has_present_tok = any(ms.group(2).lower() in ("present", "ongoing") for ms in all_spans) or "continuously updated" in sent.lower()
            sy = min(starts)
            if has_present_tok and (len(all_spans) >= 2 or deposit_pub_year) and "continuously updated" not in sent.lower():
                # Resolve relative "Present" in static archival multi-sensor deposits cleanly via deposit publication year (never stuck at first sensor's 2010!)
                cands = list(num_ends)
                if deposit_pub_year and deposit_pub_year >= sy:
                    cands.append(deposit_pub_year)
                ey = max(cands) if cands else None
                is_ong = False
            elif has_present_tok:
                ey = None
                is_ong = True
            else:
                ey = max(num_ends) if num_ends else None
                is_ong = False
            sent_rng_hit = (sy, ey, is_ong, sent[:300])
            break

        # Non-citation title ranges
        ttl_no_cite = re.sub(r"\bet\s+al\.?", "", clean_ttl)
        m_ttl_rng = re.search(r"\b(19[5-9]\d|20[0-2]\d)\s*(?:[-–—]|to)\s*(19[5-9]\d|20[0-2]\d|present|ongoing)\b", ttl_no_cite, re.IGNORECASE)
        if m_ttl_rng:
            sy = int(m_ttl_rng.group(1))
            rhs = m_ttl_rng.group(2).lower()
            if rhs in ("present", "ongoing"):
                return sy, None, True, clean_ttl[:260]
            ey_t = int(rhs)
            ev_t = clean_ttl[:260]
            if file_max_end and file_max_end > ey_t and (file_max_end - ey_t) <= 5:
                ey_t = file_max_end
                ev_t = f"{ev_t} [file archive extends to {file_max_end}]"[:300]
            return sy, ey_t, False, ev_t
        if sent_rng_hit:
            sy_h, ey_h, ong_h, ev_h = sent_rng_hit
            if file_max_end and ey_h is not None and file_max_end > ey_h and (file_max_end - ey_h) <= 5:
                ey_h = file_max_end
                ong_h = False
                ev_h = f"{ev_h} [file archive extends to {file_max_end}]"[:300]
            return sy_h, ey_h, ong_h, ev_h
        m_ttl_since = re.search(r"\bsince\s+(19[5-9]\d|20[0-2]\d)\b", clean_ttl, re.IGNORECASE)
        if m_ttl_since:
            return int(m_ttl_since.group(1)), file_max_end, False if file_max_end else True, clean_ttl[:260]
        m_ttl_for = re.search(r"\b(?:for|in)\s+(20[0-2]\d)\b", clean_ttl, re.IGNORECASE)
        if m_ttl_for:
            yr = int(m_ttl_for.group(1))
            return yr, file_max_end or yr, False, clean_ttl[:260]
        return None, None, False, None

    def _extract_prose_metadata_with_provenance(rec: dict[str, Any]) -> None:
        """Extract separated prose-inferred metadata + evidence without mutating native upstream fields."""
        clean_ttl = _clean_prose_text(clean_str(rec.get("title")))
        raw_dsc = clean_str(rec.get("description")) or ""
        file_keys = rec.pop("_file_keys", None)
        deposit_pub_date = rec.pop("_deposit_pub_date", None)

        sp_cand, sp_role, sp_ev = _extract_prose_spatial(raw_dsc)
        rec["prose_spatial_candidate_raw"] = sp_cand
        rec["prose_spatial_role"] = sp_role
        rec["prose_spatial_evidence_sentence"] = sp_ev

        cad_cand, cad_role, cad_epoch, cad_ev = _extract_prose_cadence(clean_ttl, raw_dsc)
        rec["prose_cadence_raw"] = cad_cand
        rec["prose_cadence_role"] = cad_role
        rec["prose_cadence_by_epoch"] = cad_epoch
        rec["prose_cadence_evidence_sentence"] = cad_ev

        p_bbox, p_bbox_ev = _extract_prose_bbox(raw_dsc)
        rec["prose_bbox_raw"] = p_bbox
        rec["prose_bbox_evidence_sentence"] = p_bbox_ev

        p_sy, p_ey, p_ong, p_temp_ev = _extract_prose_temporal_span(
            clean_ttl, raw_dsc, file_keys=file_keys, deposit_pub_date=deposit_pub_date
        )
        rec["prose_temporal_start_year"] = p_sy
        rec["prose_temporal_end_year"] = p_ey
        rec["prose_temporal_is_ongoing"] = p_ong
        rec["prose_temporal_evidence_sentence"] = p_temp_ev

        if "asset_type_raw" not in rec:
            rec["asset_type_raw"] = None
        if "doi_source_raw" not in rec:
            rec["doi_source_raw"] = None
        if "doi_relation_type_raw" not in rec:
            rec["doi_relation_type_raw"] = None
        if "related_dois_raw" not in rec:
            rec["related_dois_raw"] = None

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
            # Apply uniform deterministic normalization across all 31 product sources (native fields preserved!)
            _extract_prose_metadata_with_provenance(r)
            r["bbox_raw"] = clean_bbox(r.get("bbox_raw"))
            r["doi_raw"] = clean_doi(r.get("doi_raw"))
            if r["doi_raw"] and not r.get("doi_source_raw"):
                r["doi_source_raw"] = "upstream_api_doi_field"
            if r["doi_raw"] and not r.get("doi_relation_type_raw"):
                r["doi_relation_type_raw"] = "is_primary_dataset"
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
        for old_j in OUT_DATA_DIR.glob(f"{sid}*.jsonl"):
            old_j.unlink()
        if len(kept) <= 6000:
            with (OUT_DATA_DIR / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
                for item in kept:
                    f.write(json.dumps(strict_json_sanitize(item), ensure_ascii=False, allow_nan=False) + "\n")
        else:
            jsonl_chunk = 5000
            for c_idx in range(0, len(kept), jsonl_chunk):
                c_part = (c_idx // jsonl_chunk) + 1
                with (OUT_DATA_DIR / f"{sid}_part{c_part:02d}.jsonl").open("w", encoding="utf-8") as f:
                    for item in kept[c_idx : c_idx + jsonl_chunk]:
                        f.write(json.dumps(strict_json_sanitize(item), ensure_ascii=False, allow_nan=False) + "\n")

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
    print(f"[02_preparation] Complete: {total_prepared} prepared L3/L4 records across {len(source_raw_batches)} product sources (+ 1,300 WMO OSCAR taxonomy records).")


if __name__ == "__main__":
    main()
