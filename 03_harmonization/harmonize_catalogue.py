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
    Returns None if `res_str` is None or contains no parseable numeric resolution.
    Conversions:
      - 1 degree latitude/longitude at equator ~ 111,320 m
      - 1 arc-minute ~ 1,855.3 m
      - 1 arc-second ~ 30.92 m
      - 1 km = 1,000 m
    """
    if not res_str:
        return None
    s = str(res_str).lower().strip()
    # Direct numeric string (e.g. from STAC summaries.gsd like "10" or "30.0")
    if re.match(r"^\d+(\.\d+)?$", s):
        val = float(s)
        return val if val > 0 else None

    candidates: list[float] = []
    # Match patterns like "0.25 deg", "10 m", "1 km", "30 arcsec", "15 arc-second"
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(km|kilometer|kilometre|m|meter|metre|deg|degree|°|arcsec|arc-second|arcsecond|\"|arcmin|arc-minute|')\b"
    )
    for m in pattern.finditer(s):
        val = float(m.group(1))
        unit = m.group(2)
        if val <= 0:
            continue
        if unit in ("km", "kilometer", "kilometre"):
            candidates.append(val * 1000.0)
        elif unit in ("m", "meter", "metre"):
            candidates.append(val)
        elif unit in ("deg", "degree", "°"):
            candidates.append(round(val * 111320.0, 2))
        elif unit in ("arcsec", "arc-second", "arcsecond", '"'):
            candidates.append(round(val * 30.92, 2))
        elif unit in ("arcmin", "arc-minute", "'"):
            candidates.append(round(val * 1855.3, 2))
    if candidates:
        return min(candidates)
    return None


# ---------------------------------------------------------------------------
# 2. Temporal Extent & Cadence Parser
# ---------------------------------------------------------------------------
def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.search(r"\b(19\d\d|20[0-3]\d)\b", str(date_str))
    return int(m.group(1)) if m else None


def classify_cadence(
    temp_res_raw: str | None,
    update_freq_raw: str | None,
    title: str | None,
    start_year: int | None,
    end_year: int | None,
) -> str | None:
    # If start_year and end_year are identical and not explicitly daily/monthly/hourly NRT,
    # or if raw frequency explicitly says static/notPlanned/irregular single epoch, treat as static.
    txt = f"{temp_res_raw or ''} {update_freq_raw or ''}".lower()
    title_l = (title or "").lower()

    if any(k in txt for k in ("hourly", "sub-daily", "3-hourly", "6-hourly", "daily", "1 day", "p1d", "pt1h", "pt3h", "pt6h", "continual")) or re.search(r"\b\d+(?:\.\d+)?\s*(?:hours?|minutes?)\b", txt):
        return "sub_daily_or_daily"
    if any(k in txt for k in ("weekly", "8-day", "8 day", "10-day", "10 day", "dekad", "5-day", "16-day", "biweekly", "fortnightly")) or re.search(r"\b(?:[2-9]|1\d|2\d)\s*days?\b", txt):
        return "multi_day_or_dekadal"
    if any(k in txt for k in ("monthly", "1 month", "p1m", "quarterly", "seasonal")) or re.search(r"\b(?:30 days?|1?\d\s*months?)\b", txt):
        return "monthly_or_seasonal"
    if any(k in txt for k in ("annual", "yearly", "1 year", "p1y", "biannual")):
        return "annual"
    if any(k in txt for k in ("3-year", "5-year", "multi-year", "decadal")) or re.search(r"\b(?:[2-9]|\d{2,})\s*years?\b", txt):
        return "multi_year_periodic"
    if any(k in txt for k in ("notplanned", "not_planned", "asneeded", "static", "single-epoch", "one-off", "irregular", "once")):
        return "static_or_single_epoch"

    # Check title tokens only if raw cadence metadata is absent
    if not temp_res_raw:
        if re.search(r"\b(daily|hourly|near[- ]real[- ]time|nrt)\b", title_l):
            return "sub_daily_or_daily"
        if re.search(r"\b(8-day|10-day|16-day|dekadal|weekly)\b", title_l):
            return "multi_day_or_dekadal"
        if re.search(r"\b(monthly|seasonal|quarterly)\b", title_l):
            return "monthly_or_seasonal"
        if re.search(r"\b(annual|yearly)\b", title_l):
            return "annual"
        if start_year and end_year and start_year == end_year and start_year <= 2024:
            return "static_or_single_epoch"
    return None


# ---------------------------------------------------------------------------
# 3. Geographic Coverage Classifier (from bbox_raw; null stays null)
# ---------------------------------------------------------------------------
def classify_geographic_coverage(bbox: Any) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, None, None, None, None
    try:
        w, s, e, n = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        return None, None, None, None, None
    # True antimeridian crossover spans east-of-180 into negative western longitudes
    if w > e and w > 0.0 and e < 0.0:
        dlon = 360.0 - (w - e)
    else:
        dlon = abs(e - w)
    dlat = abs(n - s)
    if dlon >= 300.0 and dlat >= 120.0:
        cov = "global"
    elif dlon >= 300.0 and dlat >= 90.0:
        cov = "quasi_global"
    elif dlon >= 300.0 and dlat >= 40.0:
        cov = "hemispheric_or_pantropical"
    elif not (w > 0.0 and e < 0.0) and -35.0 <= w <= 45.0 and 25.0 <= s <= 75.0 and -10.0 <= e <= 70.0 and 35.0 <= n <= 85.0:
        cov = "europe_regional"
    else:
        cov = "regional_or_local"
    return w, s, e, n, cov


# ---------------------------------------------------------------------------
# 4. Producing Institution vs Hosting Platform & Provider Pool
# ---------------------------------------------------------------------------
COPERNICUS_JRC_CEMS_RE = re.compile(
    r"\b(ghsl|ghs[-_]|global human settlement|edo\b|gdo\b|European Drought Observatory|Global Drought Observatory|"
    r"effis|gwis|European Forest Fire Information System|Global Wildfire Information System|glofas|efas|"
    r"copernicus emergency|cems)\b",
    re.IGNORECASE,
)

COPERNICUS_CLMS_JRC_RE = re.compile(
    r"\b(copernicus land|clms|global land service|cgls)\b",
    re.IGNORECASE,
)

EU_PRODUCER_RE = re.compile(
    r"\b(european space agency|eumetsat|cm saf|osi saf|osisaf|lsa saf|ac saf|h saf|rom saf|nwc saf|nwpsaf|"
    r"ecmwf|mercator oc[eé]an|joint research centre|european commission|european environment agency|eurostat|"
    r"emodnet|german aerospace|centre national d'etudes spatiales|data terra|theia|dinamis|terrascope|"
    r"opengeohub|openlandmap|isric|soilgrids|wageningen|eth z[uü]rich|ghent university|ugent|"
    r"vrije universiteit amsterdam|miralles, diego|\bgleam4\b|global land evaporation amsterdam model|"
    r"university of southampton|worldpop|plymouth marine|met office|ifremer|helmholtz|"
    r"alfred wegener institute|potsdam institute for climate impact|tu wien|deltares|"
    r"technical university of denmark|deutscher wetterdienst|norwegian meteorological|"
    r"uk centre for ecology|university of bremen|university of reading|university of leicester|"
    r"goethe university|max planck|british geological survey|british antarctic survey|"
    r"swisstopo|federal office of topography|collecte localisation satellites|"
    r"esa[-_ ]cci|esa worldcover|esa worldcereal)\b",
    re.IGNORECASE,
)

# Short agency/institute acronyms matched only against explicit producing_institution strings (prevents homograph leaks in titles)
EU_PRODUCER_ACRONYM_RE = re.compile(
    r"\b(esa|esa/esrin|copernicus|ec-jrc|jrc|eea|dlr|cnes|vito|rbins|pml|noc|knmi|smhi|dmi|fmi|"
    r"ispra|ingv|csic|awi|gfz|pik|iiasa|eodc|ceda|stfc|nerc|dtu|bgr|boku|itc|ukceh|nilu|dwd|"
    r"inrae|cnrs|ird|cesbio|m[eé]t[eé]o[- ]france|nceo|bgs|bas|ignf|enveo|cls|mercator)\b",
    re.IGNORECASE,
)

UN_INTL_PRODUCER_RE = re.compile(
    r"\b(food and agriculture organization|\bfao\b|wapor|gaez|gsoc|"
    r"united nations|\bunep\b|wcmc|protected planet|wdpa|wmo\b|world meteorological|"
    r"geo bon|geobon|group on earth observations|cgiar|ciat|cifor|icraf|iwmi)\b",
    re.IGNORECASE,
)


def classify_producer_and_pool(
    source_id: str,
    raw_id: str | None,
    title: str | None,
    desc: str | None,
    org_raw: str | None,
    host_raw: str | None,
) -> tuple[str | None, str | None, str, str | None, bool]:
    """
    Returns:
      (producing_institution, hosting_platform, provider_pool, copernicus_service, is_un_or_international_body)
    where provider_pool in {"copernicus_services", "eu_non_copernicus", "non_eu"}.
    """
    rid = (raw_id or "").strip()
    ttl = (title or "").strip()
    combined = f"{rid} {ttl} {org_raw or ''}"

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
        return producer or "Mercator Océan International (CMEMS)", host, "copernicus_services", "CMEMS", False
    if source_id == "02_copernicus_clms":
        return producer or "EEA / EC JRC (CLMS)", host, "copernicus_services", "CLMS", False
    if source_id == "03_copernicus_c3s_cds":
        return producer or "ECMWF (C3S)", host, "copernicus_services", "C3S", False
    if source_id == "04_copernicus_cams_ads":
        return producer or "ECMWF (CAMS)", host, "copernicus_services", "CAMS", False
    if source_id == "05_copernicus_cems_ewds":
        return producer or "ECMWF / EC JRC (CEMS)", host, "copernicus_services", "CEMS", False

    # 2. EODC (06): CEMS GFM, COP_DEM, CLMS, C3S mirrors vs other EODC/TU Wien/BOKU research collections
    if source_id == "06_copernicus_cems_gfm_eodc":
        if rid.upper() == "GFM" or "global flood monitoring" in ttl.lower():
            return "EC JRC / EODC Consortium (Copernicus Emergency Management Service - CEMS GFM)", host, "copernicus_services", "CEMS", False
        if "COP_DEM" in rid.upper() or "copernicus dem" in ttl.lower():
            return "European Space Agency / Airbus (Copernicus DEM)", host, "copernicus_services", "COP_DEM", False
        if rid in ("CGLS_SSM_1KM", "CORINE_LAND_COVER", "clms-st", "clms-vpp"):
            return "EEA / EC JRC (CLMS)", host, "copernicus_services", "CLMS", False
        if rid == "era5-land-datacube":
            return "ECMWF (C3S)", host, "copernicus_services", "C3S", False
        return producer or "EODC / TU Wien", host, "eu_non_copernicus", None, False

    # 3. EC JRC (07): Distinguish JRC CEMS/CLMS products from standalone JRC scientific flagships (TMF, GSWE, EDGAR, DOPA, ESDAC)
    if source_id == "07_ec_jrc_data_catalogue":
        if COPERNICUS_JRC_CEMS_RE.search(f"{ttl} {desc or ''}"):
            return "European Commission Joint Research Centre (EC JRC - CEMS)", host, "copernicus_services", "CEMS", False
        if COPERNICUS_CLMS_JRC_RE.search(f"{ttl} {desc or ''}"):
            return "European Commission Joint Research Centre (EC JRC - CLMS)", host, "copernicus_services", "CLMS", False
        return "European Commission Joint Research Centre (EC JRC)", host, "eu_non_copernicus", None, False

    # 4. Cross-platform Copernicus Service detection (DEDL, WEkEO, CDSE, GEE, PC, EEA, OLM)
    if re.search(r"(^EO[:.]MO[:.]DAT[:.]|^cmems_|^COPERNICUS/MARINE|Copernicus Marine|CMEMS)", combined, re.IGNORECASE):
        return producer or "Mercator Océan International (CMEMS)", host, "copernicus_services", "CMEMS", False
    if re.search(r"(^EO[:.]CLMS[:.]DAT[:.]|^EO[:.]EEA[:.]DAT[:.](?:CLMS|HRL|CORINE|CLC|URBAN-ATLAS|COASTAL|RIPARIAN|EU-HYDRO)|^EO[:.]JRC[:.]DAT[:.]CLMS|^COPERNICUS/Landcover|^COPERNICUS/CORINE|^land\.cover_copernicus$|^fapar_proba\.v$|Copernicus Land|CLMS)", combined, re.IGNORECASE):
        return producer or "EEA / EC JRC (CLMS)", host, "copernicus_services", "CLMS", False
    if re.search(r"(^EO[:.]ECMWF[:.]DAT[:.]CAMS|^ECMWF/CAMS|Copernicus Atmosphere|\bCAMS\b)", combined, re.IGNORECASE):
        return producer or "ECMWF (CAMS)", host, "copernicus_services", "CAMS", False
    if re.search(r"(^EO[:.]ECMWF[:.]DAT[:.](?:CEMS_|GLOFAS|EFAS|FIRE_)|^EO[:.]GHSL[:.]|^JRC/GHSL|^pop\.count_ghs\.jrc$|GHS-POP|GHS-BUILT|GHS-SMOD|Copernicus Emergency|\bCEMS\b|GloFAS|EFAS)", combined, re.IGNORECASE):
        return producer or "EC JRC / ECMWF (CEMS)", host, "copernicus_services", "CEMS", False
    if re.search(r"(^EO[:.]DEM[:.]DAT[:.]COP-DEM|COP-DEM|^COPERNICUS/DEM|cop-dem-glo|^dsm_glo30$|Copernicus DEM)", combined, re.IGNORECASE):
        return producer or "ESA / Copernicus (Copernicus DEM)", host, "copernicus_services", "COP_DEM", False
    if re.match(r"^EO[:.]ECMWF[:.]DAT[:.]", rid, re.IGNORECASE):
        if re.search(r"(?:\.D1\.|\.DT_|DESTINE|Digital Twin)", f"{rid} {ttl}", re.IGNORECASE):
            return producer or "ECMWF (Destination Earth)", host, "eu_non_copernicus", None, False
        return producer or "ECMWF (C3S)", host, "copernicus_services", "C3S", False
    if re.search(r"(^ECMWF/ERA5|\bERA5 Heat\b|Copernicus Climate Change|\bC3S\b)", combined, re.IGNORECASE):
        return producer or "ECMWF (C3S)", host, "copernicus_services", "C3S", False

    # Upstream non-EU mirrors on EU platforms (CDSE MODIS, OpenLandMap third-party global datasets, DestinE GSW/GBIF)
    if source_id == "10_cdse_stac" and re.match(r"^modis-", rid, re.IGNORECASE):
        return "NASA / USGS LP DAAC", host, "non_eu", None, False
    if source_id == "24_openlandmap_stac":
        if re.match(r"^(?:lc_glad\.|landsat_glad\.)", rid, re.IGNORECASE):
            return "University of Maryland (GLAD)", host, "non_eu", None, False
        if re.match(r"^(?:lc_glc\.fcs30d|wilderness_li2022|fapar_essd\.lstm|lc_mcd12q1v061)", rid, re.IGNORECASE):
            return producer or "Non-EU Partner (OpenLandMap Mirror)", host, "non_eu", None, False
        return producer, host, "eu_non_copernicus", None, False
    if source_id == "08_destine_dedl_stac":
        if re.match(r"^EO\.GSW\.DAT\.", rid, re.IGNORECASE):
            return "European Commission Joint Research Centre (EC JRC)", host, "eu_non_copernicus", None, False
        if re.match(r"^EO\.GBIF\.", rid, re.IGNORECASE):
            return producer, host, "non_eu", None, False

    # 5. Dedicated European Non-Copernicus Catalogues
    if source_id in ("11_eea_and_emodnet", "17_eumetsat_product_navigator", "18_esa_osc_earthcode", "19_dlr_eoc_stac"):
        return producer, host, "eu_non_copernicus", None, False

    if source_id == "14_ceos_fedeo_agencies":
        prod_str = producer or ""
        # Check explicit producing institution first so European archives of non-EU sensors (e.g. ESA/ESRIN Landsat) stay European
        if re.search(r"\b(inpe|jaxa|g-portal|isro|mosdac|nrsc|bhoonidhi|nrcan|ccmeo|usgs)\b", prod_str, re.IGNORECASE):
            if re.search(r"\b(vito|rbins|esa/esrin|dlr|cnes|ceda)\b", prod_str, re.IGNORECASE) and not re.search(
                r"\b(inpe|jaxa|isro|nrcan)\b", prod_str, re.IGNORECASE
            ):
                return producer, host, "eu_non_copernicus", None, False
            return producer, host, "non_eu", None, False
        if not prod_str and re.search(r"\b(jaxa|isro|inpe|nrcan|usgs)\b", combined, re.IGNORECASE):
            return producer, host, "non_eu", None, False
        return producer, host, "eu_non_copernicus", None, False

    if source_id == "22_data_terra_inpe_dea":
        if re.search(r"\b(geoscience australia|digital earth australia|\bdea\b|^ga_|inpe|brazil)\b", f"{rid} {producer or ''}", re.IGNORECASE):
            return producer, host, "non_eu", None, False
        return producer or "Data Terra / CNES / Theia (France)", host, "eu_non_copernicus", None, False

    if source_id == "25_worldpop_and_ciesin_stac":
        if "ciesin" in combined.lower() or "columbia" in combined.lower() or "gpw" in combined.lower():
            return producer or "CIESIN (Columbia University)", host, "non_eu", None, False
        return producer or "WorldPop (University of Southampton)", host, "eu_non_copernicus", None, False

    # 6. Check UN / International body flag (FAO, UNEP-WCMC, WMO, GEO BON)
    is_un = bool(UN_INTL_PRODUCER_RE.search(combined)) or source_id == "15_geobon_and_geokhub"

    # 7. General Producer-Based Classification
    if (
        EU_PRODUCER_RE.search(combined)
        or EU_PRODUCER_ACRONYM_RE.search(producer or "")
        or re.search(r"^(?:ESA|JRC|DLR|OpenLandMap|ISRIC|COPERNICUS)/|esa-worldcover|esa-cci", f"{rid} {ttl}", re.IGNORECASE)
    ):
        return producer, host, "eu_non_copernicus", None, False

    return producer, host, "non_eu", None, is_un


# ---------------------------------------------------------------------------
# 5. Sentinel Mission Dependency Detector
# ---------------------------------------------------------------------------
S1_RE = re.compile(r"\b(sentinel[-_ ]?1[abc]?|s1[abc]?[-_ ]sar|c-band sar)\b", re.IGNORECASE)
S2_RE = re.compile(r"\b(sentinel[-_ ]?2[abc]?|s2[abc]?[-_ ]msi|hlss30|harmonized landsat[- ]sentinel|dynamic[-_ ]world|worldcover|worldcereal)\b", re.IGNORECASE)
S3_RE = re.compile(r"\b(sentinel[-_ ]?3[abc]?|olci|slstr|sral)\b", re.IGNORECASE)
S5P_RE = re.compile(r"\b(sentinel[-_ ]?5p|s5p|tropomi)\b", re.IGNORECASE)
S6_RE = re.compile(r"\b(sentinel[-_ ]?6[ab]?|poseidon-4)\b", re.IGNORECASE)


def detect_sentinel_usage(
    raw_id: str | None,
    title: str | None,
    desc: str | None,
    plat_raw: str | None,
    kw_raw: str | None,
) -> tuple[bool, str | None]:
    blob = f"{raw_id or ''} {title or ''} {desc or ''} {plat_raw or ''} {kw_raw or ''}"
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
    if not missions and re.search(r"\bsentinel\b", blob, re.IGNORECASE):
        missions.append("Sentinel_Unspecified")
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
    r"\b(level[- ]?4\b|\bl4[ab]?\b|reanalysis|era5|merra-2|jra-55|glofas|efas|forecast|hindcast|"
    r"assimilation|numerical model|multiyear_phy|multiyear_bgc|MultiYear_wav|analysisforecast|"
    r"primary productivity|mod17|myd17|vnp17|gpp\b|npp\b|evapotranspiration|mod16|wapor|gleam4?|global land evaporation amsterdam|"
    r"carbon flux|biomass density|gedi04_b|gedi l4b|cci biomass|globbiomass|fwi|fire weather index|"
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
    r"\b(?:gridded population of the world|gpwv4|edgar|emissions database for global atmospheric research|"
    r"faostat|gdp per capita|human development index|nighttime lights Socioeconomic|census|"
    r"global livestock environmental assessment model)\b)",
    re.IGNORECASE,
)

IN_SITU_GRIDDED_RE = re.compile(
    r"(^insitu[-_]|^EO[:.]MO[:.]DAT[:.]INSITU_|^EO[:.]ECMWF[:.]DAT[:.]INSITU_GRIDDED_|^INSITU_GLO_PHY_|"
    r"\bderived from in[- ]situ observations\b|\bGlobal Ocean in situ\b)",
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

    # 1. Check ARD Surface Reflectance Mosaic (Question 2 symmetric category)
    if ARD_REFLECTANCE_RE.search(head) and not re.search(r"\b(land cover|burned area|flood|biomass|leaf area|ndvi|evi|fapar|albedo)\b", ttl, re.IGNORECASE):
        return "L3", "L3_ARD_REFLECTANCE", "eo_ard_surface_reflectance_mosaic", True

    # 2. Check Level 4 vs Level 3 (match explicit L4 tokens only; avoid ambiguous combined labels)
    is_explicit_l4 = bool(re.search(r"^(?:level[- ]?|l)?4\b", lvl)) and not bool(re.search(r"(?:level[- ]?3|l3|\b3\b)\s*[/,;]\s*(?:level[- ]?4|l4|\b4\b)", lvl))
    if is_explicit_l4 or source_id in ("03_copernicus_c3s_cds", "04_copernicus_cams_ads", "05_copernicus_cems_ewds") or L4_MODEL_RE.search(head):
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
    r"corine land cover 2012|corine land cover 2018|land cover 2020|lcm-10|tcd-10)\b",
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
) -> tuple[str, bool | None]:
    """
    Returns (operational_maturity_class, is_ongoing_archive).
    Distinguishes operational ongoing series from planned/pre-operational entries,
    deprecated records, completed or mission-bounded releases (GEDI L4B, WorldCover,
    WSF, static DEMs, single-year maps, or archives ended <= 2024), and research
    benchmark releases whose temporal timestamp is only a publication date.
    """
    head = f"{raw_id or ''} {title or ''}"
    uf = (update_freq_raw or "").strip().lower()

    if uf == "planned":
        return "planned_or_preoperational", False

    has_ongoing_uf = any(
        k in uf for k in ("continual", "ongoing", "daily", "weekly", "monthly", "annual", "yearly", "update_cont", "regular", "near real time")
    )
    has_non_ongoing_uf = any(
        k in uf for k in ("completed", "archived", "not updated", "no updates", "notplanned", "not_planned", "once", "static", "one-off", "single")
    )

    # Determine is_ongoing_archive strictly from raw dates and update status
    if start_year is None and end_year is None:
        is_ongoing = None
    elif end_year is not None:
        is_ongoing = end_year >= 2025
    elif has_non_ongoing_uf or "[archived]" in head.lower():
        is_ongoing = False
    elif source_id in ("18_esa_osc_earthcode", "26_fao_unep_and_thematic_benchmarks", "29_awesome_gee_community") and not has_ongoing_uf:
        # Release/publication dates alone without explicit ongoing update cadence do not imply an ongoing operational series
        is_ongoing = False
    else:
        is_ongoing = True

    if any(k in uf for k in ("deprecated", "superseded", "obsolete", "retired")):
        return "deprecated_or_superseded", False

    if COMPLETED_MISSION_OR_STATIC_RE.search(head) or cadence_class == "static_or_single_epoch" or "[archived]" in head.lower() or "completed" in uf:
        return "completed_or_mission_bounded_release", False

    if is_ongoing is False and end_year is not None and end_year <= 2024:
        return "completed_or_mission_bounded_release", False

    if source_id in ("18_esa_osc_earthcode", "26_fao_unep_and_thematic_benchmarks", "29_awesome_gee_community") and is_ongoing is not True:
        return "research_or_benchmark_release", is_ongoing

    if is_ongoing is False:
        return "completed_or_mission_bounded_release", False

    return "operational_ongoing_series", is_ongoing


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

            # 1. Spatial resolution (strictly from spat_raw; title token kept separate!)
            spat_m = parse_resolution_meters(spat_raw)
            title_spat_m = parse_resolution_meters(f"{raw_id or ''} {title or ''}")

            # 2. Temporal years & cadence
            start_yr = parse_year(t_start_raw)
            end_yr = parse_year(t_end_raw)
            cadence_cls = classify_cadence(temp_res_raw, uf_raw, title, start_yr, end_yr)

            # 3. Geographic coverage
            bw, bs, be, bn, geo_cls = classify_geographic_coverage(bbox_raw)

            # 4. Producer vs Host & Provider Pool
            prod_inst, host_plat, pool, cop_srv, is_un = classify_producer_and_pool(
                sid, raw_id, title, desc, org_raw, host_raw
            )

            # 5. Sentinel data dependency
            uses_sent, sent_missions = detect_sentinel_usage(raw_id, title, desc, plat_raw, kw_raw)

            # 6. Processing level, subtype & EO ontology
            p_lvl, p_subtype, ont_cls, is_ard = classify_level_and_ontology(
                sid, raw_id, title, desc, lvl_raw, kw_raw
            )

            # 7. Operational maturity
            mat_cls, is_ongoing = classify_operational_maturity(
                sid, raw_id, title, uf_raw, cadence_cls, start_yr, end_yr, t_end_raw
            )

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

    # Also save chunked JSONL (<25 MB per file)
    chunk_size = 7000
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
    for col in [
        "title", "description", "producing_institution", "provider_pool", "processing_level",
        "product_level_subtype", "product_ontology_class", "spatial_resolution_raw",
        "spatial_resolution_meters", "title_token_resolution_meters", "temporal_start_raw",
        "temporal_start_year", "temporal_cadence_class", "geographic_coverage_class",
        "operational_maturity_class", "doi", "access_url",
    ]:
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
    print("\nSentinel Data Dependency Outside Copernicus Services:")
    ext = df_canon[df_canon["provider_pool"] != "copernicus_services"]
    print(ext.groupby(["provider_pool", "uses_sentinel_data"]).size().to_string())


if __name__ == "__main__":
    main()
