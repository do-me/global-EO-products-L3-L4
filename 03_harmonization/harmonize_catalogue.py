#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.0",
#   "pyarrow>=15.0.0",
# ]
# ///
"""
03_harmonization/harmonize_catalogue.py
=======================================
Harmonizes all 28 prepared L3/L4 EO product sources into a unified catalogue schema supporting
geophysical/thematic, spatial, temporal, geographic, and operational-maturity discovery.

Strict Reproducibility Guarantee:
- Zero manual imputation ("let null be null"): if a metadata attribute is absent in the upstream source,
  the corresponding harmonized attribute remains `None` (`null`).
- Separates `producing_institution` from `hosting_platform`.
- Classifies every record into `provider_pool` (`copernicus_services`, `eu_non_copernicus`, `non_eu`)
  strictly by producing institution, plus an explicit `is_un_or_international_body` flag.
- Flags `uses_sentinel_data` and `sentinel_missions` (`S1`, `S2`, `S3`, `S5P`, `S6`) across all pools.
- Symmetrically tags `product_level_subtype` (`L3_THEMATIC_BIOPHYSICAL`, `L4_MODEL_REANALYSIS_SYNTHESIS`,
  `L3_ARD_REFLECTANCE`) and `product_ontology_class` (strict EO vs. hybrid, gridded in-situ, inventory,
  or reference geometry).
- Distinguishes `operational_ongoing_series` from `completed_or_mission_bounded_release` and
  `static_or_single_epoch` releases.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PREP_DATA_DIR = REPO_ROOT / "02_preparation" / "data"
OUT_DATA_DIR = BASE_DIR / "data"

# Authority priority rank (lower = higher primary producer authority)
SOURCE_AUTHORITY_RANK: dict[str, int] = {
    "01_copernicus_cmems": 1,
    "02_copernicus_clms": 1,
    "03_copernicus_c3s_cds": 1,
    "04_copernicus_cams_ads": 1,
    "05_copernicus_cems_ewds": 1,
    "06_copernicus_cems_gfm_eodc": 1,
    "07_ec_jrc_data_catalogue": 2,
    "08_destine_dedl_stac": 3,
    "09_wekeo_dias": 4,
    "10_cdse_stac": 4,
    "11_eea_and_emodnet": 2,
    "12_ceos_cgms_wgclimate_ecv": 2,
    "14_ceos_fedeo_agencies": 2,
    "15_geobon_and_geokhub": 3,
    "16_nasa_cmr": 1,
    "17_eumetsat_product_navigator": 1,
    "18_esa_osc_earthcode": 2,
    "19_dlr_eoc_stac": 1,
    "20_noaa_onestop": 1,
    "21_usgs_landsat_and_eros": 1,
    "22_data_terra_inpe_dea": 2,
    "23_wri_gfw_and_resourcewatch": 3,
    "24_openlandmap_stac": 1,
    "25_worldpop_and_ciesin_stac": 1,
    "26_fao_unep_and_thematic_benchmarks": 2,
    "27_gee_official_stac": 5,
    "28_planetary_computer_stac": 5,
    "29_awesome_gee_community": 6,
}


# ---------------------------------------------------------------------------
# 1. Spatial Resolution Parser (strictly from spatial_resolution_raw)
# ---------------------------------------------------------------------------
def parse_resolution_meters(res_str: str | None) -> float | None:
    """
    Strictly parses a resolution string into meters.
    Returns None if `res_str` is None, contains no parseable numeric resolution,
    or exceeds physical planetary bounds (> 600 km).
    Conversions:
      - 1 degree latitude/longitude at equator ~ 111,320 m
      - 1 arc-minute ~ 1,855.3 m
      - 1 arc-second ~ 30.92 m
      - 1 km = 1,000 m
    """
    if not res_str:
        return None
    s = str(res_str).lower().replace("&deg;", "°").strip()
    # Direct numeric string (e.g. from STAC summaries.gsd like "10" or "30.0")
    if re.match(r"^\d+(\.\d+)?$", s):
        val = float(s)
        return val if 0 < val <= 600_000.0 else None

    candidates: list[float] = []
    # 1. Explicit fractional degree notations: e.g. "1°/112", "1/336°", "5°/112", "1/24°", "1/8th degree", "1/4 deg"
    frac_deg_pat = re.compile(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:(°)|\s*(deg|degree|degrees))?\s*/\s*(\d+(?:\.\d+)?)\s*(?:(th)\b)?\s*(?:(deg|degree|degrees)\b|(°)(?!\s*[cfknswe]\b))?"
    )
    for m in frac_deg_pat.finditer(s):
        num = float(m.group(1))
        den = float(m.group(4))
        has_deg = bool(m.group(2) or m.group(3) or m.group(6) or m.group(7))
        if has_deg and num > 0 and den > 0:
            val_m = round((num / den) * 111320.0, 2)
            if 0 < val_m <= 600_000.0:
                candidates.append(val_m)

    # 1b. Explicit anisotropic X x Y grid spacing (e.g. "0.625 x 0.5 Decimal Degrees", "0.5 x 0.625 degree")
    xy_pat = re.compile(
        r"(?<![\d.])(\.\d+|\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?)?\s*[x×]\s*(\.\d+|\d+(?:\.\d+)?)\s*(?:(km|kilometers?|kilometres?|m|meters?|metres?|(?:decimal\s+)?(?:deg|degrees?)|arcsec|arc-seconds?|arcseconds?|arcmin|arc-minutes?|arcminutes?)\b|(°)(?!\s*[cfknswe]\b))"
    )
    for m in xy_pat.finditer(s):
        v1, v2 = float(m.group(1)), float(m.group(2))
        unit = (m.group(3) or m.group(4) or "").strip()
        for val in (v1, v2):
            if val <= 0 or not unit:
                continue
            if unit in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
                val_m = val * 1000.0
            elif unit in ("m", "meter", "meters", "metre", "metres"):
                val_m = val
            elif unit in ("deg", "degree", "degrees", "°") or unit.startswith("decimal deg"):
                val_m = round(val * 111320.0, 2)
            elif unit in ("arcsec", "arc-second", "arc-seconds", "arcsecond", "arcseconds"):
                val_m = round(val * 30.92, 2)
            elif unit in ("arcmin", "arc-minute", "arc-minutes", "arcminute", "arcminutes"):
                val_m = round(val * 1855.3, 2)
            else:
                continue
            if 0 < val_m <= 600_000.0:
                candidates.append(val_m)

    # 2. Standard decimal/integer value + unit (excluding numerator/denominator numbers adjacent to '/')
    pattern = re.compile(
        r"(?<![\d.])(?<!\d/)(?<!\d\s/)(\.\d+|\d+(?:\.\d+)?)\s*(?:(km|kilometers?|kilometres?|m|meters?|metres?|(?:decimal\s+)?(?:deg|degrees?)|arcsec|arc-seconds?|arcseconds?|arcmin|arc-minutes?|arcminutes?)\b|(°)(?!\s*[cfknswe]\b))(?![\s°]*\/)"
    )
    for m in pattern.finditer(s):
        val = float(m.group(1))
        unit = (m.group(2) or m.group(3) or "").strip()
        if val <= 0 or not unit:
            continue
        if unit in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
            val_m = val * 1000.0
        elif unit in ("m", "meter", "meters", "metre", "metres"):
            val_m = val
        elif unit in ("deg", "degree", "degrees", "°") or unit.startswith("decimal deg"):
            val_m = round(val * 111320.0, 2)
        elif unit in ("arcsec", "arc-second", "arc-seconds", "arcsecond", "arcseconds"):
            val_m = round(val * 30.92, 2)
        elif unit in ("arcmin", "arc-minute", "arc-minutes", "arcminute", "arcminutes"):
            val_m = round(val * 1855.3, 2)
        else:
            continue
        if 0 < val_m <= 600_000.0:
            candidates.append(val_m)
    if candidates:
        return min(candidates)
    return None


def parse_title_token_resolution_meters(raw_id: str | None, title: str | None) -> float | None:
    """Extracts horizontal pixel resolution tokens from title/ID after masking vertical depths and non-GSD codes."""
    txt = f"{raw_id or ''} {title or ''}"
    txt = re.sub(r"sensor depth:\s*[\d.]+\s*(?:m|meters?|metres?)\b", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bat\s+(?:the\s+surface\s+and\s+)?\d+(?:\.\d+)?\s*(?:m|meters?|metres?)\b", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\b\d+(?:\.\d+)?\s*(?:m|meters?|metres?)\s+(?:above|below|depth|height|altitude|elevation|(?:surface\s+)?winds?)\b", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\b\d+(?:\.\d+)?\s*(?:m|meters?)\s+urban\s+and\s+", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bin\s+\d+(?:\.\d+)?\s*m\.?\s*transect\b", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\b\d+\s*M\s+[\w\s]*Mosaic\b", " ", txt)
    txt = re.sub(r"mosaic-cbers4[a-z]*-[a-z-]+-\d+m\b", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\bdegree\s+of\s+[a-z\s]+?\([a-z]+\)\s*\d{4}\b", " ", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\b[a-zA-Z0-9_]*[a-zA-Z]\d+(?:\.\d+)?(?:m|km|deg)\b", " ", txt, flags=re.IGNORECASE)
    return parse_resolution_meters(txt)


# ---------------------------------------------------------------------------
# 2. Temporal Extent & Cadence Parser
# ---------------------------------------------------------------------------
def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.search(r"\b(19\d\d|20\d\d|2100)\b", str(date_str))
    return int(m.group(1)) if m else None


_MULTI_DAY_RE = re.compile(
    r"\b(?:8[- ]?da(?:y|ily|ys)|10[- ]?da(?:y|ily|ys)|16[- ]?da(?:y|ily|ys)|5[- ]?da(?:y|ily|ys)|"
    r"7[- ]?da(?:y|ily|ys)|15[- ]?da(?:y|ily|ys)|20[- ]?da(?:y|ily|ys)|"
    r"dekad(?:al|e|s)?|three times a month|weekly|bi[- ]?weekly|fortnightly)\b|"
    r"\b(?:[2-9]|1\d|2\d)\s*days?\b",
    re.IGNORECASE,
)

_SUB_DAILY_OR_DAILY_RE = re.compile(
    r"\b(?:sub[- ]?hourly|hourly|1[- ]?hour(?:ly)?|3[- ]?hour(?:ly)?|6[- ]?hour(?:ly)?|12[- ]?hour(?:ly)?|"
    r"every\s+[136]\s*hours?|sub[- ]?daily|diurnal(?!\s+cycle)|1\s*days?|p1d|pt\d+[hm])\b|"
    r"(?<!\d[- ])(?<!\d)(?<!\bmulti[- ])\bdaily\b|"
    r"\b\d+(?:\.\d+)?\s*(?:hours?|minutes?|seconds?)\b",
    re.IGNORECASE,
)

_MONTHLY_OR_SEASONAL_RE = re.compile(
    r"\b(?:monthly|1\s*months?|p1m|p3m|quarterly|seasonal|3[- ]?month(?:ly)?|30\s*days?|1?\d\s*months?)\b",
    re.IGNORECASE,
)

_MULTI_YEAR_RE = re.compile(
    r"\b(?:[2-9][- ]?year(?:ly|s)?|multi[- ]?year|decadal|\d{2,}\s*years?)\b",
    re.IGNORECASE,
)

_ANNUAL_RE = re.compile(
    r"(?<!semi)(?<!semi-)(?<!bi)(?<!bi-)\b(?:annual(?:ly)?|yearly|1\s*years?|p1y)\b",
    re.IGNORECASE,
)


def _match_cadence_classes(txt: str) -> list[str]:
    classes: list[str] = []
    if _MULTI_DAY_RE.search(txt):
        classes.append("multi_day_or_dekadal")
    if _SUB_DAILY_OR_DAILY_RE.search(re.sub(r"\b\d+[- ]?da(?:y|ily|ys)\b", " ", txt, flags=re.I)):
        classes.append("sub_daily_or_daily")
    if _MONTHLY_OR_SEASONAL_RE.search(txt):
        classes.append("monthly_or_seasonal")
    if _MULTI_YEAR_RE.search(txt):
        classes.append("multi_year_periodic")
    if _ANNUAL_RE.search(re.sub(r"\b\d+[- ]?year(?:ly|s)?\b", " ", txt, flags=re.I)):
        classes.append("annual")
    if re.search(r"\b(?:static|single[- ]epoch|one[- ]off|once)\b", txt, re.I):
        classes.append("static_or_single_epoch")
    return classes


def classify_cadence(
    temp_res_raw: str | None,
    update_freq_raw: str | None,
    title: str | None,
    start_year: int | None,
    end_year: int | None,
    desc: str | None = None,
) -> str | None:
    """Classify product temporal support/sampling cadence strictly separately from archival publication frequency.
    Accepts None (null) when temporal_resolution_raw and title contain multiple distinct cadences or lack explicit cadence tokens."""
    tr = (temp_res_raw or "").strip().lower()
    title_l = (title or "").strip().lower()
    uf = (update_freq_raw or "").strip().lower()

    # 1. Explicit single-cadence token in title takes priority (resolving multi-cadence collection tags when title is specific)
    title_classes = _match_cadence_classes(title_l)
    if len(title_classes) == 1:
        return title_classes[0]
    if "multi_day_or_dekadal" in title_classes and "monthly_or_seasonal" not in title_classes:
        return "multi_day_or_dekadal"

    # 2. Evaluate explicit temporal_resolution_raw (returning None if it lists multiple distinct cadence classes without title disambiguation)
    if tr:
        tr_classes = _match_cadence_classes(tr)
        if len(tr_classes) == 1:
            return tr_classes[0]
        if len(tr_classes) > 1:
            return None

    # 3. Explicit static / single-epoch indicators in update_freq_raw ONLY when not a multi-year time series
    if any(k in uf for k in ("static", "single-epoch", "one-off", "once")) and not (
        start_year is not None and end_year is not None and end_year > start_year
    ):
        return "static_or_single_epoch"

    return None


# ---------------------------------------------------------------------------
# 3. Geographic Coverage Classifier (from bbox_raw; null stays null)
# ---------------------------------------------------------------------------
_EXPLICIT_EUROPE_SCOPE_RE = re.compile(
    r"(?:,\s*Europe\b|\bPan-European\b|\bEEA39\b|\bin Europe(?:'s)?\s+Seas\b|\balong European coastlines\b|"
    r"\bEuropean soil database\b|\bfor Europe\b|\bacross Europe\b|\b Baltic Sea\b|\bMediterranean Sea\b|\bBlack Sea\b)",
    re.IGNORECASE,
)

_EXPLICIT_REGIONAL_SCOPE_RE = re.compile(
    r"\b(?:regional\b|north america\b|contiguous u\.?s\.?|conterminous united states|\bconus\b|"
    r"united states and alaska|alaska|hawaii|puerto rico|for africa\b|in africa\b|sub-saharan africa|"
    r"south america|central america|southeast asia|east asia|south asia|middle east|"
    r"australia|new zealand|great lakes|gulf of mexico|chesapeake bay)\b",
    re.IGNORECASE,
)


def classify_geographic_coverage(
    bbox: Any,
    title: str | None = None,
    desc: str | None = None,
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, None, None, None, None
    try:
        w, s, e, n = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        return None, None, None, None, None
    # Antimeridian wrap-around spans east -> west across 180 deg
    if w > e:
        dlon = 360.0 - (w - e)
    else:
        dlon = abs(e - w)
    dlat = abs(n - s)

    title_str = title or ""
    has_global_in_title = bool(re.search(r"\b(?:global|world|worldwide|quasi-global|near-global)\b", title_str, re.IGNORECASE))

    # Guard against nominal [-180, -90, 180, 90] provider bounding boxes on explicitly regional products
    if not has_global_in_title:
        if _EXPLICIT_EUROPE_SCOPE_RE.search(title_str):
            return w, s, e, n, "europe_regional"
        if _EXPLICIT_REGIONAL_SCOPE_RE.search(title_str):
            return w, s, e, n, "regional_or_local"
        if desc and re.search(r"\b(?:across the contiguous U\.?S\.?\s*\(CONUS\)|for the conterminous United States)\b", desc, re.IGNORECASE):
            return w, s, e, n, "regional_or_local"

    if dlon >= 300.0 and dlat >= 120.0:
        cov = "global"
    elif dlon >= 300.0 and dlat >= 90.0:
        cov = "quasi_global"
    elif (dlon >= 300.0 and dlat >= 40.0) or (dlon >= 120.0 and dlat >= 120.0):
        cov = "hemispheric_or_pantropical"
    elif not (w > e) and -65.0 <= w <= 45.0 and 10.0 <= s <= 75.0 and -10.0 <= e <= 85.0 and 35.0 <= n <= 90.0:
        cov = "europe_regional"
    elif dlon < 300.0 and title and _EXPLICIT_EUROPE_SCOPE_RE.search(title):
        cov = "europe_regional"
    else:
        cov = "regional_or_local"
    return w, s, e, n, cov


# ---------------------------------------------------------------------------
# 4. Producing Institution vs Hosting Platform, Provider Pool & Jurisdiction Tier
# ---------------------------------------------------------------------------
COPERNICUS_JRC_CEMS_RE = re.compile(
    r"\b(ghsl|ghs[-_]|global human settlement|degree of urbanisation|smod|edo\b|gdo\b|"
    r"european drought observatory|global drought observatory|"
    r"effis|gwis|european forest fire information system|global wildfire information system|"
    r"glofas|efas|flood hazard|global flood|copernicus emergency|copernicus ems|\bcems\b|rapid mapping|\bemsr\d*\b)\b",
    re.IGNORECASE,
)

COPERNICUS_CLMS_JRC_RE = re.compile(
    r"\b(copernicus land|\bclms\b|global land service|\bcgls\b)\b",
    re.IGNORECASE,
)

COPERNICUS_EEA_CLMS_RE = re.compile(
    r"\b(copernicus land|\bclms\b|corine land cover|\bclc\s*(?:1990|2000|2006|2012|2018)\b|"
    r"high resolution layer|\bhrl\b|imperviousness|impervious built-up|share of built-up|"
    r"tree cover density|dominant leaf type|forest type|grassland 20|grassland change|grassland vegetation|grassland mowing|"
    r"water and wetness|small woody features|small landscape features|urban atlas|riparian zones|coastal zones|"
    r"\beu-dem\b|\beu-hydro\b|high resolution vegetation phenology|medium resolution vegetation phenology|\bhr-vpp\b|"
    r"high-resolution snow|fractional snow cover|persistent snow|river and lake ice|biophysical parameters)\b",
    re.IGNORECASE,
)

# JRC datasets mentioning Copernicus in prose descriptions/projects but NOT Copernicus service outputs
NON_COPERNICUS_JRC_EXEMPT_RE = re.compile(r"\b(ardeco|lucas)\b", re.IGNORECASE)

# Explicit third-country national agencies (always non_eu even when mirrored or co-cited in federated gateways)
THIRD_COUNTRY_PRODUCER_RE = re.compile(
    r"\b(nasa|jpl|podaac|po\.daac|nsidc|lpdaac|ges_disc|ornl|ornl_cloud|laads|asdc|cmr|"
    r"usgs|noaa|nesdis|ncei|usda|usfs|usepa|columbia university|ciesin|"
    r"inpe|cptec|brazil data cube|jaxa|g-portal|isro|mosdac|nrsc|bhoonidhi|"
    r"nrcan|natural resources canada|ccmeo|canadian space agency|eccc|csiro|"
    r"geoscience australia|digital earth australia|conae|sansa|nrsa|cbers|"
    r"\bcas\b|\bigsnrr\b|pml_v2|\busap\b|\busap-dc\b|\bnsf\b)\b",
    re.IGNORECASE,
)

# Formal Copernicus Participating / Associated States under Reg. (EU) 2021/696:
# EEA states (Norway, Iceland) and United Kingdom (Copernicus Association effective 2024-01-01)
COPERNICUS_ASSOCIATED_STATE_RE = re.compile(
    r"\b(university of southampton|worldpop|university of bristol|fabdem|"
    r"uk centre for ecology|\bukceh\b|met office|hadley centre|plymouth marine laboratory|plymouth marine|"
    r"national oceanography centre|\bnoc\b|british geological survey|\bbgs\b|"
    r"british antarctic survey|\bbas\b|\bceda\b|\bstfc\b|\bnerc\b|\bukri\b|\bdefra\b|\bnceo\b|"
    r"university of reading|university of leicester|university of edinburgh|university of leeds|"
    r"university of exeter|university of oxford|university of cambridge|"
    r"norwegian meteorological|met norway|\bnersc\b|nansen environmental|\bnilu\b|"
    r"\bnibio\b|\bimr\b|\bsintef\b|\bniva\b|icelandic meteorological|\bvedur\b)\b",
    re.IGNORECASE,
)

# Switzerland: ESA/EUMETSAT/ECMWF Member State NOT associated to Copernicus -> legal third country (non_eu)
CH_ESA_NON_COPERNICUS_RE = re.compile(
    r"\b(eth z[uü]rich|\bethz\b|crowther lab|\bwsl\b|envidat|\bepfl\b|\bwgms\b|"
    r"world glacier monitoring|meteoswiss|swisstopo|federal office of topography|"
    r"university of zurich|university of bern)\b",
    re.IGNORECASE,
)

# European Space / Meteorological IGOs (non-Copernicus activities of ESA, EUMETSAT, ECMWF)
EUROPEAN_IGO_RE = re.compile(
    r"\b(european space agency|\besa\b|esa/esrin|esa[-_ ]cci|esa worldcover|esa worldcereal|"
    r"eumetsat|cm saf|osi saf|osisaf|lsa saf|ac saf|h saf|rom saf|nwc saf|nwpsaf|"
    r"\becmwf\b|destination earth|destine)\b",
    re.IGNORECASE,
)

EU_PRODUCER_RE = re.compile(
    r"\b(joint research centre|european commission|european environment agency|eurostat|"
    r"emodnet|german aerospace|centre national d'etudes spatiales|data terra|theia|dinamis|terrascope|"
    r"opengeohub|openlandmap|isric|soilgrids|wageningen|ghent university|ugent|"
    r"vrije universiteit amsterdam|miralles, diego|\bgleam4\b|global land evaporation amsterdam model|"
    r"ifremer|mercator oc[eé]an|helmholtz|alfred wegener institute|potsdam institute for climate impact|"
    r"tu wien|deltares|technical university of denmark|deutscher wetterdienst|"
    r"university of bremen|goethe university|max planck|collecte localisation satellites)\b",
    re.IGNORECASE,
)

# Short agency/institute acronyms matched only against explicit producing_institution strings (prevents homograph leaks in titles)
EU27_ACRONYM_RE = re.compile(
    r"\b(copernicus|ec-jrc|jrc|eea|dlr|cnes|vito|rbins|knmi|smhi|dmi|fmi|"
    r"ispra|ingv|csic|awi|gfz|pik|iiasa|eodc|dtu|bgr|boku|itc|dwd|"
    r"inrae|cnrs|ird|cesbio|m[eé]t[eé]o[- ]france|ignf|enveo|cls|cmcc|mercator)\b",
    re.IGNORECASE,
)

UN_INTL_PRODUCER_RE = re.compile(
    r"\b(food and agriculture organization|\bfao\b|wapor|gaez|gsoc|"
    r"united nations|\bunep\b|wcmc|protected planet|wdpa|wmo\b|world meteorological|"
    r"geo bon|geobon|group on earth observations|cgiar|ciat|cifor|icraf|iwmi|"
    r"world bank|world bank group|\bundp\b|\bunsd\b|\bunhcr\b)\b",
    re.IGNORECASE,
)


def classify_eu_non_copernicus_tier(
    combined: str,
    prod_str: str,
    source_id: str,
) -> tuple[str, str]:
    """Sub-classify non-Copernicus European records by institutional/jurisdiction tier."""
    # Check ESA / EUMETSAT / ECMWF IGO identity before national institute partners (e.g. ESA CCI hosted at CEDA)
    if EUROPEAN_IGO_RE.search(prod_str) or EUROPEAN_IGO_RE.search(combined) or source_id in ("17_eumetsat_product_navigator", "18_esa_osc_earthcode"):
        # If an associated-state national agency is explicit without ESA/EUMETSAT/ECMWF in prod_str, keep national tier
        if COPERNICUS_ASSOCIATED_STATE_RE.search(prod_str) and not EUROPEAN_IGO_RE.search(prod_str) and "climate change initiative" not in combined.lower() and "_cci" not in combined.lower():
            return "eu_non_copernicus", "copernicus_associated_state"
        return "eu_non_copernicus", "european_space_met_igo"
    if (EU_PRODUCER_RE.search(prod_str) or EU27_ACRONYM_RE.search(prod_str)) and not COPERNICUS_ASSOCIATED_STATE_RE.search(prod_str):
        return "eu_non_copernicus", "eu27_national_or_eu_agency"
    if COPERNICUS_ASSOCIATED_STATE_RE.search(prod_str) or COPERNICUS_ASSOCIATED_STATE_RE.search(combined):
        return "eu_non_copernicus", "copernicus_associated_state"
    return "eu_non_copernicus", "eu27_national_or_eu_agency"


def classify_producer_and_pool(
    source_id: str,
    raw_id: str | None,
    title: str | None,
    desc: str | None,
    org_raw: str | None,
    host_raw: str | None,
    kw_raw: str | None = None,
) -> tuple[str | None, str | None, str, str | None, bool, str]:
    """
    Returns:
      (producing_institution, hosting_platform, provider_pool, copernicus_service,
       is_un_or_international_body, provider_jurisdiction_tier)
    where provider_pool in {"copernicus_services", "eu_non_copernicus", "non_eu"}.
    """
    rid = (raw_id or "").strip()
    ttl = (title or "").strip()
    prod_str = (org_raw or "").strip()
    combined = f"{rid} {ttl} {prod_str}"
    is_un = bool(UN_INTL_PRODUCER_RE.search(combined)) or source_id == "15_geobon_and_geokhub"

    # Default hosting platform by catalogue source
    default_hosts = {
        "01_copernicus_cmems": "Copernicus Marine Data Store (MDS)",
        "02_copernicus_clms": "Copernicus Land Monitoring Service Portal",
        "03_copernicus_c3s_cds": "Copernicus Climate Data Store (CDS)",
        "04_copernicus_cams_ads": "Copernicus Atmosphere Data Store (ADS)",
        "05_copernicus_cems_ewds": "Copernicus Early Warning Data Store (EWDS)",
        "06_copernicus_cems_gfm_eodc": "EODC STAC",
        "07_ec_jrc_data_catalogue": "EC JRC Data Catalogue / EU Open Data Portal",
        "08_destine_dedl_stac": "Destination Earth Data Lake (DEDL)",
        "09_wekeo_dias": "WEkEO Copernicus DIAS",
        "10_cdse_stac": "Copernicus Data Space Ecosystem (CDSE)",
        "11_eea_and_emodnet": "EEA SDI / EMODnet GeoNetwork",
        "12_ceos_cgms_wgclimate_ecv": "CEOS/CGMS WGClimate ECV Inventory",
        "14_ceos_fedeo_agencies": "CEOS FedEO Gateway",
        "15_geobon_and_geokhub": "GEO BON / GEO Knowledge Hub",
        "16_nasa_cmr": "NASA Earthdata CMR",
        "17_eumetsat_product_navigator": "EUMETSAT Product Navigator",
        "18_esa_osc_earthcode": "ESA EarthCODE Open Science Catalog",
        "19_dlr_eoc_stac": "DLR EOC Geoservice STAC",
        "20_noaa_onestop": "NOAA OneStop",
        "21_usgs_landsat_and_eros": "USGS LandsatLook / EROS",
        "22_data_terra_inpe_dea": "National EO STAC (Data Terra / DEA / INPE)",
        "23_wri_gfw_and_resourcewatch": "WRI Global Forest Watch / Resource Watch",
        "24_openlandmap_stac": "OpenLandMap Wasabi S3 STAC",
        "25_worldpop_and_ciesin_stac": "WorldPop / CIESIN STAC",
        "26_fao_unep_and_thematic_benchmarks": "FAO GISMGR / Zenodo",
        "27_gee_official_stac": "Google Earth Engine",
        "28_planetary_computer_stac": "Microsoft Planetary Computer",
        "29_awesome_gee_community": "Google Earth Engine Community Catalog",
    }
    host = host_raw or default_hosts.get(source_id)
    producer = org_raw

    # 1. Primary Copernicus Thematic Services (01-05)
    if source_id == "01_copernicus_cmems":
        return producer or "Mercator Océan International (CMEMS)", host, "copernicus_services", "CMEMS", False, "copernicus_service"
    if source_id == "02_copernicus_clms":
        return producer or "EEA / EC JRC (CLMS)", host, "copernicus_services", "CLMS", False, "copernicus_service"
    if source_id == "03_copernicus_c3s_cds":
        return producer or "ECMWF (C3S)", host, "copernicus_services", "C3S", False, "copernicus_service"
    if source_id == "04_copernicus_cams_ads":
        return producer or "ECMWF (CAMS)", host, "copernicus_services", "CAMS", False, "copernicus_service"
    if source_id == "05_copernicus_cems_ewds":
        return producer or "ECMWF / EC JRC (CEMS)", host, "copernicus_services", "CEMS", False, "copernicus_service"

    # 2. EODC (06): CEMS GFM, COP_DEM, CLMS, C3S mirrors vs other EODC/TU Wien/BOKU research collections
    if source_id == "06_copernicus_cems_gfm_eodc":
        if rid.upper() == "GFM" or "global flood monitoring" in ttl.lower():
            return "EC JRC / EODC Consortium (Copernicus Emergency Management Service - CEMS GFM)", host, "copernicus_services", "CEMS", False, "copernicus_service"
        if "COP_DEM" in rid.upper() or "copernicus dem" in ttl.lower():
            return "European Space Agency / Airbus (Copernicus DEM)", host, "copernicus_services", "COP_DEM", False, "copernicus_service"
        if rid in ("CGLS_SSM_1KM", "CORINE_LAND_COVER", "clms-st", "clms-vpp"):
            return "EEA / EC JRC (CLMS)", host, "copernicus_services", "CLMS", False, "copernicus_service"
        if rid == "era5-land-datacube":
            return "ECMWF (C3S)", host, "copernicus_services", "C3S", False, "copernicus_service"
        if producer == "EODC / Copernicus Emergency Management Service (CEMS GFM)":
            producer = "Earth Observation Data Centre (EODC)"
        return producer or "EODC / TU Wien", host, "eu_non_copernicus", None, False, "eu27_national_or_eu_agency"

    # 3. EC JRC (07): Distinguish JRC CEMS/CLMS products from standalone JRC scientific flagships (TMF, GSWE, EDGAR, ARDECO, LUCAS, DOPA, ESDAC)
    # Only scan rid + ttl + keywords (never full prose description) so background mentions of CEMS/Copernicus do not misclassify ARDECO/LUCAS.
    if source_id == "07_ec_jrc_data_catalogue":
        jrc_head = f"{rid} {ttl} {kw_raw or ''}"
        if not NON_COPERNICUS_JRC_EXEMPT_RE.search(f"{rid} {ttl}"):
            if COPERNICUS_JRC_CEMS_RE.search(jrc_head):
                return "European Commission Joint Research Centre (EC JRC - CEMS)", host, "copernicus_services", "CEMS", False, "copernicus_service"
            if COPERNICUS_CLMS_JRC_RE.search(jrc_head):
                return "European Commission Joint Research Centre (EC JRC - CLMS)", host, "copernicus_services", "CLMS", False, "copernicus_service"
        return "European Commission Joint Research Centre (EC JRC)", host, "eu_non_copernicus", None, False, "eu27_national_or_eu_agency"

    # 3b. National hubs in 22_data_terra_inpe_dea & Brazilian INPE SAMeT in 14_ceos_fedeo_agencies:
    # Classify before cross-platform C3S/ESA checks so co-listed forcing providers (e.g. C3S ERA5) never hijack INPE/DEA products.
    if source_id == "22_data_terra_inpe_dea":
        hub_text = f"{rid} {prod_str} {host_raw or ''}"
        if re.search(r"\b(geoscience australia|digital earth australia|\bdea\b|^ga_|inpe|brazil)\b", hub_text, re.IGNORECASE):
            return producer, host, "non_eu", None, False, "third_country_non_european"
        return producer or "Data Terra / CNES / Theia (France)", host, "eu_non_copernicus", None, False, "eu27_national_or_eu_agency"

    if source_id == "14_ceos_fedeo_agencies" and re.match(r"^samet_", rid, re.IGNORECASE):
        return producer or "INPE / CPTEC (Brazil)", host, "non_eu", None, False, "third_country_non_european"

    # 4. Cross-platform Copernicus Service detection (DEDL, WEkEO, CDSE, GEE, PC, EEA, FedEO, OLM)
    if re.search(r"(^EO[:.]MO[:.]DAT[:.]|^cmems_|^COPERNICUS/MARINE|Copernicus Marine|CMEMS)", combined, re.IGNORECASE):
        return producer or "Mercator Océan International (CMEMS)", host, "copernicus_services", "CMEMS", False, "copernicus_service"
    if re.search(r"(^EO[:.]CLMS[:.]DAT[:.]|^EO[:.]EEA[:.]DAT[:.](?:CLMS|HRL|CORINE|CLC|URBAN-ATLAS|COASTAL|RIPARIAN|EU-HYDRO)|^EO[:.]JRC[:.]DAT[:.]CLMS|^COPERNICUS/Landcover|^COPERNICUS/CORINE|\bland\.cover_copernicus\b|\bfapar_proba\.v\b|Copernicus (?:Global )?Land|\bCLMS\b|\bCGLS\b|Copernicus Urban Atlas|Copernicus Imperviousness)", combined, re.IGNORECASE):
        return producer or "EEA / EC JRC (CLMS)", host, "copernicus_services", "CLMS", False, "copernicus_service"
    if re.search(r"(^EO[:.]ECMWF[:.]DAT[:.]CAMS|^ECMWF/CAMS|Copernicus Atmosphere|\bCAMS\b)", combined, re.IGNORECASE):
        return producer or "ECMWF (CAMS)", host, "copernicus_services", "CAMS", False, "copernicus_service"
    if re.search(r"(^EO[:.]ECMWF[:.]DAT[:.](?:CEMS_|GLOFAS|EFAS|FIRE_)|^EO[:.]GHSL[:.]|^JRC/GHSL|\bpop\.count_ghs\.jrc\b|\bGHSL\b|GHS-POP|GHS-BUILT|GHS-SMOD|human settlements? grid|Copernicus Emergency|\bCEMS\b|GloFAS|EFAS)", combined, re.IGNORECASE):
        return producer or "EC JRC / ECMWF (CEMS)", host, "copernicus_services", "CEMS", False, "copernicus_service"
    if re.search(r"(^EO[:.]DEM[:.]DAT[:.]COP-DEM|COP-DEM|^COPERNICUS/DEM|cop-dem-glo|\bdsm_glo30\b|Copernicus DEM|\bCopernicus Digital Elevation Model\b)", combined, re.IGNORECASE):
        return producer or "ESA / Copernicus (Copernicus DEM)", host, "copernicus_services", "COP_DEM", False, "copernicus_service"
    if re.match(r"^EO[:.]ECMWF[:.]DAT[:.]", rid, re.IGNORECASE):
        if re.search(r"(?:\.D1\.|\.DT_|DESTINE|Digital Twin)", f"{rid} {ttl}", re.IGNORECASE):
            return producer or "ECMWF (Destination Earth)", host, "eu_non_copernicus", None, False, "european_space_met_igo"
        return producer or "ECMWF (C3S)", host, "copernicus_services", "C3S", False, "copernicus_service"
    if re.search(r"(^ECMWF/ERA5|\bera5-pds\b|\bera5_repack\b|ce-ag-era5|terrascope-sis-agrometeorological-indicators|\bAgERA5\b|\bERA5[- ]Land\b|\bERA5 Heat\b|Copernicus Climate Change|\bC3S\b)", combined, re.IGNORECASE):
        return producer or "ECMWF (C3S)", host, "copernicus_services", "C3S", False, "copernicus_service"

    # Upstream non-EU mirrors on EU platforms (CDSE MODIS, OpenLandMap third-party global datasets, DestinE GSW/GBIF)
    if source_id == "10_cdse_stac" and re.match(r"^modis-", rid, re.IGNORECASE):
        return "NASA / USGS LP DAAC", host, "non_eu", None, False, "third_country_non_european"
    if source_id == "24_openlandmap_stac":
        if re.match(r"^(?:lc_glad\.|landsat_glad\.)", rid, re.IGNORECASE):
            return "University of Maryland (GLAD)", host, "non_eu", None, False, "third_country_non_european"
        if re.match(r"^(?:lc_glc\.fcs30d|wilderness_li2022|fapar_essd\.lstm|lc_mcd12q1v061)", rid, re.IGNORECASE):
            return producer or "Non-EU Partner (OpenLandMap Mirror)", host, "non_eu", None, False, "third_country_non_european"
        return producer, host, "eu_non_copernicus", None, False, "eu27_national_or_eu_agency"
    if source_id == "08_destine_dedl_stac":
        if re.match(r"^EO\.GSW\.DAT\.", rid, re.IGNORECASE):
            return "European Commission Joint Research Centre (EC JRC)", host, "eu_non_copernicus", None, False, "eu27_national_or_eu_agency"
        if re.match(r"^EO\.GBIF\.", rid, re.IGNORECASE):
            return producer, host, "non_eu", None, False, "un_or_international_body" if is_un else "third_country_non_european"

    # 5. Dedicated European Non-Copernicus Catalogues (preserving CLMS datasets distributed via EEA SDI and Swiss/IGO/Associated distinctions)
    if source_id == "11_eea_and_emodnet":
        is_emodnet = bool(
            re.search(r"emodnet|marine", prod_str, re.IGNORECASE)
            or re.search(r"emodnet", host_raw or "", re.IGNORECASE)
        )
        if not is_emodnet and (
            COPERNICUS_EEA_CLMS_RE.search(f"{ttl} {prod_str} {kw_raw or ''}")
            or re.search(r"\b(copernicus land monitoring service|\bclms\b)\b", desc or "", re.IGNORECASE)
        ):
            return "European Environment Agency (Copernicus Land Monitoring Service - CLMS)", host, "copernicus_services", "CLMS", False, "copernicus_service"

    if source_id == "17_eumetsat_product_navigator" and prod_str in (
        "National Oceanic and Atmospheric Administration",
        "National Aeronautics and Space Administration",
    ):
        return producer, host, "non_eu", None, False, "third_country_non_european"

    if source_id in ("11_eea_and_emodnet", "17_eumetsat_product_navigator", "18_esa_osc_earthcode", "19_dlr_eoc_stac"):
        if CH_ESA_NON_COPERNICUS_RE.search(prod_str) and not EUROPEAN_IGO_RE.search(prod_str):
            return producer, host, "non_eu", None, False, "european_esa_non_copernicus_state"
        pool, tier = classify_eu_non_copernicus_tier(combined, prod_str, source_id)
        return producer, host, pool, None, False, tier

    if source_id == "14_ceos_fedeo_agencies":
        # Check explicit third-country agencies first (NASA, JPL, ASF, ORNL, USGS, INPE, JAXA, ISRO, NRCan)
        # while allowing explicit European-hosted Third Party Mission archives (e.g. ESA/ESRIN, VITO, DLR, CNES) without NASA/JPL/USGS co-lead
        has_third = bool(THIRD_COUNTRY_PRODUCER_RE.search(prod_str) or (not prod_str and THIRD_COUNTRY_PRODUCER_RE.search(combined)))
        has_eu_host_archive = bool(re.search(r"\b(vito|rbins|esa/esrin|dlr|cnes|terrascope)\b", prod_str, re.IGNORECASE))
        has_non_eu_primary = bool(re.search(r"\b(nasa|jpl|asf|ornl|cmr|inpe|jaxa|g-portal|isro|mosdac|nrsc|bhoonidhi|nrcan|usgs)\b", prod_str, re.IGNORECASE))
        if has_third and not (has_eu_host_archive and not has_non_eu_primary):
            return producer, host, "non_eu", None, False, "third_country_non_european"
        if CH_ESA_NON_COPERNICUS_RE.search(prod_str) and not EUROPEAN_IGO_RE.search(prod_str):
            return producer, host, "non_eu", None, False, "european_esa_non_copernicus_state"
        pool, tier = classify_eu_non_copernicus_tier(combined, prod_str, source_id)
        return producer, host, pool, None, False, tier

    if source_id == "25_worldpop_and_ciesin_stac":
        if "ciesin" in combined.lower() or "columbia" in combined.lower() or "gpw" in combined.lower():
            return producer or "CIESIN (Columbia University)", host, "non_eu", None, False, "third_country_non_european"
        # WorldPop (University of Southampton, UK) -> formal Copernicus Associated State effective 2024-01-01
        return producer or "WorldPop (University of Southampton)", host, "eu_non_copernicus", None, False, "copernicus_associated_state"

    # 6. UN / International Bodies & Multilateral Consortia (e.g. FAO, UNEP, WMO, GEO BON, World Bank)
    if is_un:
        return producer, host, "non_eu", None, True, "un_or_international_body"

    # 7. Swiss Institutions (ESA/EUMETSAT/ECMWF member, NOT Copernicus Participating/Associated State)
    if CH_ESA_NON_COPERNICUS_RE.search(combined) and not EUROPEAN_IGO_RE.search(prod_str):
        return producer, host, "non_eu", None, False, "european_esa_non_copernicus_state"

    if source_id == "16_nasa_cmr":
        if re.search(r"\bEUMETSAT\b", prod_str, re.IGNORECASE) and re.search(r"\bproduced by\s+OSI\s*SAF\b", ttl, re.IGNORECASE):
            return "EUMETSAT Ocean and Sea Ice SAF (OSI SAF)", host, "eu_non_copernicus", None, False, "european_space_met_igo"
        if prod_str == "GPCC" and ttl.startswith("DWD-GPCC:"):
            return "Deutscher Wetterdienst (DWD - GPCC)", host, "eu_non_copernicus", None, False, "eu27_national_or_eu_agency"

    # 8. Third-Country Primary Producers (prevents NASA/USGS/NOAA/Canada records mentioning ECMWF or campaign partners from leaking to EU)
    if THIRD_COUNTRY_PRODUCER_RE.search(prod_str) and not (
        (EUROPEAN_IGO_RE.search(prod_str) or EU_PRODUCER_RE.search(prod_str))
        and not re.search(r"^(?:NASA|NOAA|USGS|ORNL|NSIDC|LPDAAC|GES_DISC)", prod_str, re.IGNORECASE)
    ):
        return producer, host, "non_eu", None, False, "third_country_non_european"

    # 9. General European Producer Classification (EU-27, ESA/EUMETSAT/ECMWF, UK/NO/IS Copernicus Associated States)
    if (
        EUROPEAN_IGO_RE.search(combined)
        or COPERNICUS_ASSOCIATED_STATE_RE.search(combined)
        or EU_PRODUCER_RE.search(combined)
        or EU27_ACRONYM_RE.search(prod_str)
        or re.search(r"^(?:ESA|JRC|DLR|OpenLandMap|ISRIC|COPERNICUS)/|esa-worldcover|esa-cci", f"{rid} {ttl}", re.IGNORECASE)
    ):
        pool, tier = classify_eu_non_copernicus_tier(combined, prod_str, source_id)
        return producer, host, pool, None, False, tier

    return producer, host, "non_eu", None, False, "third_country_non_european"


# ---------------------------------------------------------------------------
# 5. Sentinel Mission Dependency Detector
# ---------------------------------------------------------------------------
S1_RE = re.compile(
    r"\b(sentinel[-_ ]?1[abc]?|s1[abc]?[-_ ]sar|c-band sar|egms|european ground motion|"
    r"(?:copernicus|cems|eodc)\s+global flood monitoring|hr-s&i)\b",
    re.IGNORECASE,
)
S2_RE = re.compile(
    r"\b(sentinel[-_ ]?2[abc]?|s2[abc]?[-_ ]msi|glad[-_ ]?s2|hlss30|harmonized landsat[- ]sentinel|dynamic[-_ ]world|worldcover|worldcereal|"
    r"ghs-built-s|hr-vpp|high resolution vegetation phenology|clcplus backbone|clc\+\s*backbone|vlcc|hr-s&i)\b",
    re.IGNORECASE,
)
S3_RE = re.compile(
    r"\b(sentinel[-_ ]?3[abc]?|olci|slstr|sral)\b|burnt area\s+20(?:1[89]|2\d)-present\s*\(raster 300\s*m\)",
    re.IGNORECASE,
)
S5P_RE = re.compile(r"\b(sentinel[-_ ]?5p|s5p|tropomi)\b", re.IGNORECASE)
S6_RE = re.compile(r"\b(sentinel[-_ ]?6[ab]?|poseidon-4)\b", re.IGNORECASE)

_SENTINEL_LAUNCH_YEAR = {
    "S1": 2014,
    "S2": 2015,
    "S3": 2016,
    "S5P": 2017,
    "S6": 2020,
    "Sentinel_Unspecified": 2014,
}


_CONTEXTUAL_NON_SENTINEL_DESC_RE = re.compile(
    r"(?:complementary with[^.;]*?\bsentinel[-_ ]?[12356][abp]?\b[^.;]*|"
    r"provides data to[^.;]*?\bsentinel[-_ ]?[12356][abp]?\b[^.;]*|"
    r"the data target for meris and olci[^.;]*|"
    r"other sensors running the same algorithm include olci[^.;]*|"
    r"detectable across some european sentinel platforms[^.;]*|"
    r"taking advantage of the availability of sentinel[-_ ]?2 data,\s*whereas earlier layers[^.;]*|"
    r"transition to sentinel[-_ ]?2 imagery in 2018[^.;]*|"
    r"tropospheric monitoring instrument\s*\(\s*tropomi\s*\)[^.;]*|"
    r"harmonized landsat and sentinel-2\s*\(\s*hls\s*\)\s*project[^.;]*|"
    r"sentinel-2a and sentinel-2b[^.;]*?landsat 8 and landsat 9[^.;]*)",
    re.IGNORECASE,
)


def detect_sentinel_usage(
    raw_id: str | None,
    title: str | None,
    desc: str | None,
    plat_raw: str | None,
    kw_raw: str | None,
    end_year: int | None = None,
    is_ongoing: bool | None = None,
) -> tuple[bool, str | None]:
    rid_str = raw_id or ""
    ttl_str = title or ""
    if re.search(r"\b(hurricane and severe storm sentinel|hs3[a-z0-9_-]*)\b", f"{rid_str} {ttl_str}", re.IGNORECASE):
        return False, None

    # Check whether the title itself pins a single pre-2014 reference year/range (ignoring EEA publication month-year suffixes like ', Feb. 2024')
    ttl_no_pub = re.sub(
        r",\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*\s+20[12]\d\b",
        "",
        ttl_str,
        flags=re.IGNORECASE,
    )
    ttl_years = [int(y) for y in re.findall(r"(?<![\d_])(19[7-9]\d|20\d\d|2100)(?![\d_])", ttl_no_pub)]
    title_has_post2014_year = any(y >= 2014 for y in ttl_years)
    title_pinned_pre2014 = bool(
        ttl_years
        and max(ttl_years) < 2014
        and not re.search(
            r"\b(ongoing|present|nrt|near[- ]real[- ]time|multi[- ]?year|multitemporal|sentinel)\b",
            ttl_str,
            re.IGNORECASE,
        )
    )
    if title_pinned_pre2014:
        return False, None

    is_retrospective_fusion = bool(
        re.search(r"\b(biomass_cci|cci\s+biomass|gedi[-_ ]?fusion|wsci|gfm|worldcover)\b", f"{rid_str} {ttl_str}", re.IGNORECASE)
    )
    bounded_end = (
        end_year
        if (end_year is not None and is_ongoing is not True and not title_has_post2014_year and not is_retrospective_fusion)
        else None
    )
    if bounded_end is not None and bounded_end < 2014:
        return False, None

    head_blob = f"{rid_str} {ttl_str} {plat_raw or ''} {kw_raw or ''}"

    # Strip known non-input contextual/comparison clauses from description rather than suppressing description via platform_sensor_raw
    desc_clean = _CONTEXTUAL_NON_SENTINEL_DESC_RE.sub(" ", desc or "")
    # Explicit Landsat-only HLS component (HLSL30 / L30HLS) or ENVISAT-only v1 LWQ reprocessing (2002-2012)
    if (
        re.search(r"\bhlsl30\b", f"{rid_str} {ttl_str}", re.IGNORECASE)
        or ("l30hls" in (desc or "").lower() and "s30hls" not in (desc or "").lower())
        or re.search(r"^clms_lwq-reproc_global_300m_10daily_v1_", rid_str, re.IGNORECASE)
    ):
        desc_clean = ""

    blob = f"{head_blob} {desc_clean}"

    missions: list[str] = []
    if S1_RE.search(blob):
        missions.append("S1")
    if S2_RE.search(blob):
        missions.append("S2")
    if S3_RE.search(blob):
        missions.append("S3")
    if S5P_RE.search(blob):
        missions.append("S5P")
    if S6_RE.search(blob):
        missions.append("S6")
    if not missions and (
        re.search(r"\bsentinel\b", head_blob, re.IGNORECASE)
        or (
            re.search(r"\b(copernicus sentinel|sentinel-[12356]|sentinel [12356])\b", desc_clean, re.IGNORECASE)
            and "emodnet digital bathymetry" not in ttl_str.lower()
        )
    ):
        missions.append("Sentinel_Unspecified")

    if bounded_end is not None:
        missions = [m for m in missions if bounded_end >= _SENTINEL_LAUNCH_YEAR.get(m, 2014)]

    if missions:
        return True, "; ".join(missions)
    return False, None


# ---------------------------------------------------------------------------
# 6. Product Level, Subtype & EO Ontology Classifier
# ---------------------------------------------------------------------------
ARD_REFLECTANCE_RE = re.compile(
    r"\b(surface reflectance|boa reflectance|bottom[- ]of[- ]atmosphere|global mosaic|cloudless mosaic|"
    r"sentinel-2.*mosaic|s2gm|hlsl30|hlss30|harmonized landsat.*sentinel|mod09|myd09|vnp09|"
    r"8-day.*composite|32-day.*composite|annual.*composite|pixel based image composite|nbar)\b",
    re.IGNORECASE,
)

L4_MODEL_RE = re.compile(
    r"\b(level[- ]?4\b|\bl4[a-d]?\b|reanalysis|era5|merra-2|jra-55|glofas|efas|forecast|hindcast|"
    r"(?<!for data )assimilation|numerical model|lpj[-_]eosim|simulated\s+global\s+gridded|"
    r"multiyear_phy|multiyear_bgc|MultiYear_wav|analysisforecast|"
    r"primary productivity|mod17|myd17|vnp17|gpp\b|gross primary product(?:ion|ivity)|net primary product(?:ion|ivity)|"
    r"evapotranspiration|mod16|wapor|gleam4?|global land evaporation amsterdam|"
    r"carbon flux|biomass density|gedi04_[a-d]|gedi l4[a-d]|cci biomass|globbiomass|fwi|fire weather index|"
    r"drought indicator|cdi\b|sma\b|spi\b|soil organic carbon|soilgrids)\b",
    re.IGNORECASE,
)

HYBRID_EO_CONSTRAINED_RE = re.compile(
    r"\b(ghs[-_ ]?pop|worldpop|landscan|hrsl|openlandmap|soilgrids|isric|esdac|glosem|"
    r"global soil organic carbon|gsocmap|gaez|grid3)\b",
    re.IGNORECASE,
)

NON_EO_INVENTORY_RE = re.compile(
    r"(^FAO_GISMGR:GLEAM3:|SOCIO-ECONOMIC-FORCING|"
    r"\b(?:gridded population of the world|gpwv[34]|edgar|emissions database for global atmospheric research|"
    r"faostat|gdp per capita|gridded gdp|gross domestic product|ardeco|human development index|nighttime lights socioeconomic|census|"
    r"unemployment rate|employment in agriculture|forestry employment|tenure security|national law|"
    r"eurobarometer|harmonised lucas in-situ|cruise ports|suprema|agmip|policy relevant modelling|"
    r"treaty texts|environmental treaty|hazardous waste site|population projections.*sres|"
    r"global livestock environmental assessment model)\b)",
    re.IGNORECASE,
)

IN_SITU_GRIDDED_RE = re.compile(
    r"(^insitu[-_]|^EO[:.]MO[:.]DAT[:.]INSITU_|^EO[:.]ECMWF[:.]DAT[:.]INSITU_GRIDDED_|^INSITU_GLO_PHY_|"
    r"^npp_[a-z0-9_-]+:\s*npp\b|"
    r"\b(?:derived from in[- ]situ observations|global ocean in situ|lucas\s+copernicus|"
    r"integrated global radiosonde archive|\bradiosonde\b|argo float|java ocean atlas|"
    r"oceanographic profile|benthic macrofauna|multibeam survey|hydrographic survey|coastal survey|"
    r"polarstern cruise|discrete.*stations|towed behind.*ship|in[- ]situ.*measurements|"
    r"ctd casts?|collected (?:by|from|using) (?:ctd|xbt)|using [\w\s,]*(?:ctd|xbt)\b|xbt casts?|"
    r"sensor depth:\s*[\d.]+\s*m|wave and tide recorder|temperature and conductivity recorder|"
    r"coral reef early warning system.*buoy|sst\)?\s+buoy|mooring\s+tao|from\s+moorings?\b|"
    r"gppdi_npp|bigfoot gpp surfaces|"
    r"fluxnet|eddy covariance|airborne lidar|airborne radar|uavsar|airsar|drone imagery|\bsuas\b|"
    r"aircraft campaign|shipboard)\b)",
    re.IGNORECASE,
)

REFERENCE_GEOMETRY_RE = re.compile(
    r"\b(wdpa|wdoecm|protected planet|world database on protected areas|natura 2000|natura2000|"
    r"global administrative unit|gaul|gadm|hydrosheds|hydrobasins|hydrorivers|hydrolakes|eu-hydro)\b",
    re.IGNORECASE,
)


def classify_level_and_ontology(
    source_id: str,
    raw_id: str | None,
    title: str | None,
    desc: str | None,
    lvl_raw: str | None,
    kw_raw: str | None,
) -> tuple[str, str, str, bool]:
    """
    Returns:
      (processing_level, product_level_subtype, product_ontology_class, is_ard_reflectance_only)
    """
    rid = (raw_id or "").strip()
    ttl = (title or "").strip()
    lvl = (lvl_raw or "").strip().lower()
    head = f"{rid} {ttl} {kw_raw or ''}"
    full = f"{head} {desc or ''}"

    # 1. Check ARD Surface Reflectance Mosaic (Question 2 symmetric category; exclude explicit Level-4 or vegetation-index cubes)
    if ARD_REFLECTANCE_RE.search(head) and not re.search(
        r"\b(level[- ]?4\b|l4[a-d]?\b|land cover|burned area|flood|biomass|leaf area|ndvi|evi|fapar|albedo|vegetation index)\b",
        head,
        re.IGNORECASE,
    ):
        return "L3", "L3_ARD_REFLECTANCE", "eo_ard_surface_reflectance_mosaic", True

    # 2. Check Level 4 vs Level 3 (preserving L4 for multi-level L3/L4 collections such as C3S satellite-sea-surface-temperature so L4 queries discover them)
    is_explicit_l4 = bool(re.search(r"\b(?:level[- ]?|l)?4[a-d]?\b", lvl))
    is_explicit_l3 = (
        bool(re.search(r"^(?:level[- ]?|l)?3[abcgs]?\b", lvl))
        and not bool(re.search(r"4", lvl))
        and not bool(re.search(
            r"\b(level[- ]?4|l4[a-d]?|reanalysis|era5|merra-2|jra-55|glofas|efas|mod16|mod17|myd17|vnp17|gross primary product|net primary product|gedi04_[a-d]|gedi l4[a-d]|soilgrids)\b",
            head,
            re.IGNORECASE,
        ))
    )
    is_cads_model_or_reanalysis = (
        source_id in ("04_copernicus_cams_ads", "05_copernicus_cems_ewds")
        or (
            source_id == "03_copernicus_c3s_cds"
            and not re.match(r"^(?:satellite|insitu-gridded)-", rid, re.IGNORECASE)
        )
        or bool(
            source_id == "03_copernicus_c3s_cds"
            and re.match(
                r"^satellite-(?:sea-surface-temperature-ensemble|ist-sst-global|ice-sheet-mass-balance)",
                rid,
                re.IGNORECASE,
            )
        )
    )
    if is_explicit_l4 or is_cads_model_or_reanalysis or (not is_explicit_l3 and L4_MODEL_RE.search(head)):
        p_level = "L4"
        subtype = "L4_MODEL_REANALYSIS_SYNTHESIS"
    else:
        p_level = "L3"
        subtype = "L3_THEMATIC_BIOPHYSICAL"

    # 3. Classify EO vs. hybrid/in-situ/inventory/reference ontology
    if REFERENCE_GEOMETRY_RE.search(head):
        ont = "authoritative_reference_geometry"
    elif IN_SITU_GRIDDED_RE.search(f"{rid} {ttl}"):
        ont = "in_situ_gridded_interpolation"
    elif NON_EO_INVENTORY_RE.search(head):
        ont = "non_eo_inventory_or_census"
    elif HYBRID_EO_CONSTRAINED_RE.search(head) or source_id in ("24_openlandmap_stac", "25_worldpop_and_ciesin_stac"):
        ont = "eo_constrained_hybrid"
    elif p_level == "L4":
        ont = "eo_reanalysis_or_model"
    else:
        ont = "eo_thematic_or_biophysical"

    return p_level, subtype, ont, False


# ---------------------------------------------------------------------------
# 7. Operational Maturity Classifier
# ---------------------------------------------------------------------------
COMPLETED_MISSION_OR_STATIC_RE = re.compile(
    r"\b(gedi04_b|gedi l4b|worldcover|wsf[-_ ]?2015|wsf[-_ ]?2019|wsf[-_ ]?evolution|wsf[-_ ]?3d|wsf3d|wsf_evo|"
    r"world settlement footprint \(wsf\) (?:2015|2019|evolution|3d)|world settlement footprint (?:2015|2019|evolution)|"
    r"srtm|nasadem|tandem-x.*dem|cop-dem|alos.*dsm|meris|envisat|ers-1|ers-2|avhrr pathfinder|"
    r"globcover|glc2000|corine land cover 1990|corine land cover 2000|corine land cover 2006|"
    r"corine land cover 2012|corine land cover 2018|land cover 2020|lcm-10|tcd-10|"
    r"seawifs|proba-v|gomos|sciamachy|mipas|aatsr|quikscat|aquarius|adeos|cryosat-1|goce)\b|\btrmm\b(?!.*gpm)|\bsmmr\b(?!.*ssmi)",
    re.IGNORECASE,
)


FUTURE_PROJECTION_OR_STATIC_RELEASE_RE = re.compile(
    r"\b(scenariomip|ssp[1-5][-_.]|rcp[-_ ]?[2468]\.?[05]|sres[_ -]?[ab][12]|projections?-cmip|"
    r"cmip[56].*decadal.*prototype|climate projection|urban expansion|puexpans|white_paper|white paper|"
    r"luisa|ref20\d\d|reference scenario|medium-term outlook|r202[345]a|p202[345]a)\b",
    re.IGNORECASE,
)


def classify_operational_maturity(
    source_id: str,
    raw_id: str | None,
    title: str | None,
    update_freq_raw: str | None,
    cadence_class: str | None,
    start_year: int | None,
    end_year: int | None,
    temporal_end_raw: str | None,
    kw_raw: str | None = None,
) -> tuple[str, bool | None]:
    """
    Returns (operational_maturity_class, is_ongoing_archive).
    Distinguishes operational ongoing series from planned/pre-operational entries,
    deprecated records, completed or mission-bounded releases (GEDI L4B, WorldCover,
    WSF, static DEMs, single-year maps, future scenario/demographic projections, or archives ended <= 2024),
    and research benchmark releases whose temporal timestamp is only a publication date.
    """
    head = f"{raw_id or ''} {title or ''}"
    uf = (update_freq_raw or "").strip().lower()

    if uf == "planned" or "collectionprogress:planned" in uf:
        return "planned_or_preoperational", False

    if (
        "collectionprogress:deprecated" in uf
        or any(k in uf for k in ("deprecated", "superseded", "obsolete", "retired"))
        or any(k in head.lower() for k in ("**superseded**", "[deprecated]", "[archived]", "version superseded", "dismissed)"))
    ):
        return "deprecated_or_superseded", False

    is_jrc_single_event = (
        source_id == "07_ec_jrc_data_catalogue"
        and bool(
            re.search(r"\(\d{4}-\d{2}-\d{2}\)\s*$", title or "")
            or re.search(r"\b(emsr\d+|emsn\d+|rapid mapping|risk and recovery)\b", f"{title or ''} {kw_raw or ''}", re.IGNORECASE)
        )
    )
    if is_jrc_single_event:
        return "completed_or_mission_bounded_release", False

    has_ongoing_uf = any(
        k in uf
        for k in (
            "collectionprogress:active",
            "continual",
            "ongoing",
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "annual",
            "yearly",
            "update_cont",
            "regular",
            "near real time",
            "near_real_time",
        )
    )
    has_non_ongoing_uf = any(
        k in uf
        for k in (
            "collectionprogress:complete",
            "completed",
            "archived",
            "not updated",
            "no updates",
            "notplanned",
            "not_planned",
            "once",
            "static",
            "one-off",
            "single",
        )
    )

    is_future_proj = bool(
        FUTURE_PROJECTION_OR_STATIC_RELEASE_RE.search(head)
        or (source_id == "25_worldpop_and_ciesin_stac" and end_year is not None and end_year >= 2025)
        or (end_year is not None and end_year >= 2028)
    )

    is_explicit_present_series = bool(
        re.search(r"\b(19\d\d|20[0-2]\d)\s*[-–—]\s*(?:present|ongoing)\b", head, re.IGNORECASE)
        and not is_future_proj
        and "[archived]" not in head.lower()
    )

    # Determine is_ongoing_archive strictly from raw dates and explicit update status
    if is_future_proj:
        is_ongoing = False
    elif has_non_ongoing_uf or "[archived]" in head.lower() or COMPLETED_MISSION_OR_STATIC_RE.search(head):
        is_ongoing = False
    elif is_explicit_present_series:
        is_ongoing = True
    elif end_year is not None:
        is_ongoing = end_year >= 2025
    elif start_year is None and end_year is None:
        if has_ongoing_uf and source_id not in ("15_geobon_and_geokhub", "18_esa_osc_earthcode", "26_fao_unep_and_thematic_benchmarks", "29_awesome_gee_community"):
            is_ongoing = True
        else:
            is_ongoing = None
    elif source_id in ("15_geobon_and_geokhub", "18_esa_osc_earthcode", "26_fao_unep_and_thematic_benchmarks", "29_awesome_gee_community") and not has_ongoing_uf:
        # Release/publication dates alone without explicit ongoing update cadence do not imply an ongoing operational series
        is_ongoing = False
    else:
        is_ongoing = True

    if is_future_proj:
        if re.search(r"\b(prototype|white_paper|white paper|benchmark|scenariomip)\b", head, re.IGNORECASE) or source_id in ("15_geobon_and_geokhub", "18_esa_osc_earthcode", "26_fao_unep_and_thematic_benchmarks"):
            return "research_or_benchmark_release", False
        return "completed_or_mission_bounded_release", False

    if COMPLETED_MISSION_OR_STATIC_RE.search(head) or "[archived]" in head.lower() or has_non_ongoing_uf:
        return "completed_or_mission_bounded_release", False

    if is_explicit_present_series:
        return "operational_ongoing_series", True

    if cadence_class == "static_or_single_epoch":
        return "completed_or_mission_bounded_release", False

    if is_ongoing is False and end_year is not None and end_year <= 2024:
        return "completed_or_mission_bounded_release", False

    if source_id in ("15_geobon_and_geokhub", "18_esa_osc_earthcode", "26_fao_unep_and_thematic_benchmarks", "29_awesome_gee_community") and is_ongoing is not True:
        return "research_or_benchmark_release", is_ongoing

    if is_ongoing is False:
        return "completed_or_mission_bounded_release", False

    if is_ongoing is None:
        return "unknown", None

    return "operational_ongoing_series", True


# ---------------------------------------------------------------------------
# 8. Canonical Product Key for Cross-Catalogue Deduplication
# ---------------------------------------------------------------------------
def make_title_slug(title: str | None, raw_id: str | None) -> str:
    base = (title or raw_id or "").lower()
    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    tokens = [t for t in base.split() if len(t) >= 1]
    return "_".join(tokens) if tokens else hashlib.md5(str(raw_id).encode()).hexdigest()[:12]


def clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    m = re.search(r"\b(10\.\d{4,9}/[^\s,;\"'<>\]]+)", doi.strip().lower())
    if not m:
        return None
    d_clean = m.group(1).rstrip(".,;:]\"'")
    if d_clean.endswith(")") and d_clean.count(")") > d_clean.count("("):
        d_clean = d_clean[:-1].rstrip(".,;:]\"'")
    return d_clean if re.match(r"^10\.\d{4,9}/[^\s\]]+$", d_clean) else None


def build_canonical_key(doi: str | None, title: str | None, raw_id: str | None, pool: str) -> str:
    d_clean = clean_doi(doi)
    if d_clean:
        return f"doi:{d_clean}"
    slug = make_title_slug(title, raw_id)
    return f"{pool}:{slug}"


def norm_none(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (list, tuple, dict)):
        return val
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


def main() -> None:
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    harmonized_rows: list[dict[str, Any]] = []

    for p in sorted(PREP_DATA_DIR.glob("*.parquet")):
        if p.name == "excluded_records_log.parquet":
            continue
        sid = p.stem
        df_prep = pd.read_parquet(p)
        records = df_prep.to_dict(orient="records")
        for idx, r_raw in enumerate(records):
            r = {k: norm_none(v) for k, v in r_raw.items()}
            raw_id = r.get("raw_id")
            title = r.get("title")
            desc = r.get("description")
            lvl_raw = r.get("processing_level_raw")
            spat_raw = r.get("spatial_resolution_raw")
            temp_res_raw = r.get("temporal_resolution_raw")
            t_start_raw = r.get("temporal_start_raw")
            t_end_raw = r.get("temporal_end_raw")
            uf_raw = r.get("update_frequency_raw")
            bbox_raw = r.get("bbox_raw")
            if hasattr(bbox_raw, "tolist"):
                bbox_raw = bbox_raw.tolist()
            org_raw = r.get("producing_org_raw")
            host_raw = r.get("hosting_platform_raw")
            plat_raw = r.get("platform_sensor_raw")
            doi_raw = r.get("doi_raw")
            lic_raw = r.get("license_raw")
            url_raw = r.get("landing_url_raw")
            kw_raw = r.get("keywords_raw")

            # 1. Spatial resolution (strictly from spat_raw; accept null when upstream UMM unit conflicts >10x with title token)
            spat_m = parse_resolution_meters(spat_raw)
            title_spat_m = parse_title_token_resolution_meters(raw_id, title)
            if spat_m is not None and title_spat_m is not None:
                res_ratio = spat_m / title_spat_m
                if res_ratio > 10.0 or res_ratio < 0.1:
                    spat_m = None

            # 2. Temporal years & cadence (strictly from raw dates; accept null when raw date is a future placeholder or conflicts with explicit title bounds)
            start_yr = parse_year(t_start_raw)
            end_yr = parse_year(t_end_raw)
            if start_yr is not None and start_yr > 2026:
                start_yr = None
            if t_start_raw and str(t_start_raw).startswith("1970-01-01") and "1970" not in (title or ""):
                start_yr = None
            title_yrs = [int(y) for y in re.findall(r"(?<![\d_])(19[5-9]\d|20\d\d|2100)(?![\d_])", title or "")]
            has_present_in_title = bool(re.search(r"\b(present|ongoing|current|now|near[- ]real[- ]time|nrt)\b", title or "", re.IGNORECASE))
            if title_yrs:
                if has_present_in_title:
                    m_pres = re.search(r"\b(19[5-9]\d|20[0-2]\d)\s*[-–—]\s*(?:present|ongoing)\b", title or "", re.IGNORECASE)
                    if m_pres and start_yr is not None and start_yr > int(m_pres.group(1)) + 1:
                        start_yr = None
                else:
                    if start_yr is not None and start_yr > max(title_yrs) + 1:
                        start_yr = None
                        if end_yr is not None and end_yr > max(title_yrs) + 1:
                            end_yr = None
                    if end_yr is not None and end_yr < min(title_yrs):
                        start_yr = None
                        end_yr = None
            if start_yr is not None and end_yr is not None and start_yr > end_yr:
                end_yr = None
            cadence_cls = classify_cadence(temp_res_raw, uf_raw, title, start_yr, end_yr, desc)

            # 3. Geographic coverage
            bw, bs, be, bn, geo_cls = classify_geographic_coverage(bbox_raw, title, desc)

            # 4. Producer vs Host, Provider Pool & Institutional Jurisdiction Tier
            prod_inst, host_plat, pool, cop_srv, is_un, tier = classify_producer_and_pool(
                sid, raw_id, title, desc, org_raw, host_raw, kw_raw
            )

            # 5. Operational maturity (computed before Sentinel detector so bounded end_year & is_ongoing are available)
            mat_cls, is_ongoing = classify_operational_maturity(
                sid, raw_id, title, uf_raw, cadence_cls, start_yr, end_yr, t_end_raw, kw_raw
            )

            # 6. Sentinel data dependency (with launch-year and explicit non-Sentinel platform guards)
            uses_sent, sent_missions = detect_sentinel_usage(
                raw_id, title, desc, plat_raw, kw_raw, end_yr, is_ongoing
            )

            # 7. Processing level, subtype & EO ontology
            p_lvl, p_subtype, ont_cls, is_ard = classify_level_and_ontology(
                sid, raw_id, title, desc, lvl_raw, kw_raw
            )
            if (
                sid == "07_ec_jrc_data_catalogue"
                and mat_cls == "completed_or_mission_bounded_release"
                and cadence_cls is None
                and (
                    re.search(r"\(\d{4}-\d{2}-\d{2}\)\s*$", title or "")
                    or re.search(r"\b(emsr\d+|emsn\d+|rapid mapping|risk and recovery)\b", f"{title or ''} {kw_raw or ''}", re.IGNORECASE)
                )
            ):
                cadence_cls = "static_or_single_epoch"

            c_key = build_canonical_key(doi_raw, title, raw_id, pool)
            rec_uid = f"{sid}::{raw_id or idx}"

            harmonized_rows.append({
                "record_id": rec_uid,
                "canonical_product_key": c_key,
                "source_catalogue": sid,
                "source_authority_rank": SOURCE_AUTHORITY_RANK.get(sid, 5),
                "raw_id": raw_id,
                "title": title,
                "description": desc,
                "producing_institution": prod_inst,
                "hosting_platform": host_plat,
                "provider_pool": pool,
                "provider_jurisdiction_tier": tier,
                "copernicus_service": cop_srv,
                "is_un_or_international_body": is_un,
                "processing_level_raw": lvl_raw,
                "processing_level": p_lvl,
                "product_level_subtype": p_subtype,
                "product_ontology_class": ont_cls,
                "is_ard_reflectance_only": is_ard,
                "strict_eo_c1_eligible": ont_cls in ("eo_thematic_or_biophysical", "eo_reanalysis_or_model", "eo_ard_surface_reflectance_mosaic"),
                "uses_sentinel_data": uses_sent,
                "sentinel_missions": sent_missions,
                "spatial_resolution_raw": spat_raw,
                "spatial_resolution_meters": spat_m,
                "title_token_resolution_meters": title_spat_m,
                "temporal_resolution_raw": temp_res_raw,
                "update_frequency_raw": uf_raw,
                "temporal_cadence_class": cadence_cls,
                "temporal_start_raw": t_start_raw,
                "temporal_end_raw": t_end_raw,
                "temporal_start_year": start_yr,
                "temporal_end_year": end_yr,
                "is_ongoing_archive": is_ongoing,
                "operational_maturity_class": mat_cls,
                "bbox_west": bw,
                "bbox_south": bs,
                "bbox_east": be,
                "bbox_north": bn,
                "geographic_coverage_class": geo_cls,
                "platform_sensor_raw": plat_raw,
                "doi": doi_raw,
                "license": lic_raw,
                "access_url": url_raw,
                "keywords": kw_raw,
            })

    # Disambiguate shared collection/paper DOIs & prevent distinct-record collisions within any single catalogue
    doi_title_slugs: dict[tuple[str, str], set[str]] = {}
    for row in harmonized_rows:
        ck = row["canonical_product_key"]
        if ck.startswith("doi:"):
            doi_title_slugs.setdefault((row["source_catalogue"], ck), set()).add(
                make_title_slug(row["title"], row["raw_id"])
            )
    shared_dois = {k for k, v in doi_title_slugs.items() if len(v) > 1}
    for row in harmonized_rows:
        ck = row["canonical_product_key"]
        if (row["source_catalogue"], ck) in shared_dois:
            row["canonical_product_key"] = f"{ck}::{make_title_slug(row['title'], row['raw_id'])}"

    # Guarantee zero within-catalogue distinct-raw_id collisions (e.g. WGClimate multi-variable entries with same title+DOI)
    src_key_counts: dict[tuple[str, str], int] = {}
    for row in harmonized_rows:
        sk = (row["source_catalogue"], row["canonical_product_key"])
        src_key_counts[sk] = src_key_counts.get(sk, 0) + 1
    for row in harmonized_rows:
        sk = (row["source_catalogue"], row["canonical_product_key"])
        if src_key_counts[sk] > 1:
            row["canonical_product_key"] = f"{row['canonical_product_key']}::{row['raw_id'] or row['record_id']}"

    df_all = pd.DataFrame(harmonized_rows)
    df_all = df_all.sort_values(["source_authority_rank", "source_catalogue", "raw_id"])

    # Save full harmonized catalogue
    df_all.to_parquet(OUT_DATA_DIR / "eo_l3_l4_all_catalogue_records.parquet", index=False)

    # Also save chunked JSONL (<20 MB per file)
    chunk_size = 4000
    for old in OUT_DATA_DIR.glob("eo_l3_l4_all_catalogue_records_part*.jsonl"):
        old.unlink()
    recs_all = df_all.to_dict(orient="records")
    for idx in range(0, len(recs_all), chunk_size):
        chunk = recs_all[idx : idx + chunk_size]
        part = (idx // chunk_size) + 1
        with (OUT_DATA_DIR / f"eo_l3_l4_all_catalogue_records_part{part:02d}.jsonl").open("w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Build deduplicated canonical products table (prioritizing highest-authority primary source)
    # Sort so primary source with earliest start_year (preventing cloud truncation!) comes first
    df_sorted = df_all.sort_values(
        by=["source_authority_rank", "temporal_start_year"],
        ascending=[True, True],
        na_position="last",
    )
    agg_catalogues = (
        df_sorted.groupby("canonical_product_key")["source_catalogue"]
        .apply(lambda s: "; ".join(dict.fromkeys(s)))
        .rename("all_source_catalogues")
    )
    agg_counts = (
        df_sorted.groupby("canonical_product_key")["record_id"]
        .count()
        .rename("cross_catalogue_record_count")
    )
    df_canon = df_sorted.drop_duplicates(subset=["canonical_product_key"], keep="first").copy()
    df_canon = df_canon.merge(agg_catalogues, on="canonical_product_key", how="left")
    df_canon = df_canon.merge(agg_counts, on="canonical_product_key", how="left")
    df_canon.to_parquet(OUT_DATA_DIR / "eo_l3_l4_canonical_products.parquet", index=False)

    # Export pool and completeness summaries
    pool_summary = (
        df_canon.groupby(["provider_pool", "processing_level", "uses_sentinel_data"], dropna=False)
        .size()
        .reset_index(name="canonical_product_count")
    )
    pool_summary.to_csv(BASE_DIR / "provider_pool_and_sentinel_summary.csv", index=False)

    completeness = []
    for col in list(df_all.columns):
        non_null = int(df_all[col].notna().sum())
        null_cnt = int(df_all[col].isna().sum())
        completeness.append({
            "column": col,
            "non_null_records": non_null,
            "null_records": null_cnt,
            "completeness_pct": round(non_null / len(df_all) * 100.0, 2),
        })
    pd.DataFrame(completeness).to_csv(BASE_DIR / "schema_completeness_report.csv", index=False)

    print(f"[03_harmonization] Total harmonized L3/L4 catalogue records: {len(df_all)}")
    print(f"[03_harmonization] Unique canonical L3/L4 products:         {len(df_canon)}")
    print("\nCanonical Products by Provider Pool:")
    print(df_canon["provider_pool"].value_counts().to_string())
    print("\nCanonical Products by Provider Jurisdiction Tier:")
    print(df_canon["provider_jurisdiction_tier"].value_counts().to_string())
    print("\nSentinel Data Dependency Outside Copernicus Services:")
    ext = df_canon[df_canon["provider_pool"] != "copernicus_services"]
    print(ext.groupby(["provider_pool", "uses_sentinel_data"]).size().to_string())


if __name__ == "__main__":
    main()
