# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
# ]
# ///
"""Build Audited External Domain Set (Holdout A): 35 Disjoint External Policy / GCOS ECV / USGEO EOA / NASA SNWG Queries.

Provenance & scientific honesty note:
  1. ZERO thematic overlap and ZERO record_id overlap with Dev-45 (`02_build_policy_benchmark.py`).
  2. Grounded in external authoritative registries (GCOS ECV / WGClimate, USGEO EOA 2023,
     NASA SNWG, WMO OSCAR, unused KCEO Urban Adaptation & UN-Habitat EO Toolkit rows).
  3. Programmatic anti-leakage check: ZERO product brand names, mission/sensor codes, or shortnames
     permitted in query text.
  4. Classified transparently as a RETROSPECTIVE GT-AUDITED external domain benchmark (not a
     prospective blind holdout) after uniform system-blind regex hardening fixed underscore word-boundary
     bugs, spelling variants (sulfur/sulphur), false positives, and symmetric official EO product omissions
     affecting Stage-1, Jina, and Qwen3 equally. Prospective single-shot blindness is carried by
     Prospective Blind Holdout B (`09_build_blind_holdout_b.py`), built strictly from unseen active
     catalogue families disjoint from both Dev-45 and Holdout A.
"""

from __future__ import annotations

import json
import re
import sys
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
from eo_search_lib import filter_benchmark_family_mask  # noqa: E402


HOLDOUT_A_SPECS: list[dict[str, Any]] = [
    {
        "query_id": "HA01_ECV_GLOBAL_PRECIPITATION",
        "source_framework": "GCOS ECV / USGEO EOA 2023 (Climate)",
        "policy_ref": "Atmosphere ECV — Global gridded multi-satellite and gauge-merged precipitation",
        "query": "Global gridded daily and monthly precipitation estimates merged from multi-satellite microwave, infrared, and rain gauge observations",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "gpm_imerg", "regex": r"(?<![a-z0-9])imerg(?![a-z0-9])|gpm.*precipitation", "exclude_regex": r"(?<![a-z0-9])chirps(?![a-z0-9])|(?<![a-z0-9])gpcp(?![a-z0-9])|\bcpc\b"},
            {"family_id": "chirps", "regex": r"(?<![a-z0-9])chirps(?![a-z0-9])"},
            {"family_id": "cpc_gpcp_precip", "regex": r"\bcpc\s+global\s+unified.*precipitation|(?<![a-z0-9])gpcp(?![a-z0-9]).*precip", "exclude_regex": r"(?<![a-z0-9])imerg(?![a-z0-9])|(?<![a-z0-9])chirps(?![a-z0-9])"},
        ],
    },
    {
        "query_id": "HA02_KCEO_DROUGHT_INDICES",
        "source_framework": "EC KCEO Urban Adaptation / C3S Climate Indicators",
        "policy_ref": "Climate hazard — Multi-decadal gridded meteorological drought indices (SPI/SPEI)",
        "query": "Long-term gridded monthly meteorological drought indicators and standardized precipitation drought indices derived from climate reanalysis",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "era5_drought", "regex": r"\bdrought\s+indices\b.*era5|\bspi\b.*drought|\bspei\b|standardized\s+precipitation\s+index"},
        ],
    },
    {
        "query_id": "HA03_EOA_METEOROLOGICAL_FORCING",
        "source_framework": "USGEO EOA 2023 (Climate & Hydrology)",
        "policy_ref": "Hydrological modeling — Gridded bias-corrected surface meteorological forcing datasets",
        "query": "Gridded surface meteorological forcing data for land surface and hydrological impact models including air temperature, humidity, wind, and radiation",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "wfdei", "regex": r"\bwfdei\b"},
            {"family_id": "nldas_forcing", "regex": r"\bnldas\b.*forcing"},
        ],
    },
    {
        "query_id": "HA04_ECV_GLACIER_INVENTORY",
        "source_framework": "GCOS ECV / USGEO EOA 2023 (Cryosphere)",
        "policy_ref": "Cryosphere ECV — Global land ice and mountain glacier outlines and area change",
        "query": "Global satellite inventory of mountain glacier outlines, land ice extent, and multi-temporal glacier area change",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "glims_rgi", "regex": r"\bglims\b|randolph\s+glacier", "exclude_regex": r"\bglaciers?[_\s]cci\b|esa\s+glaciers"},
            {"family_id": "esa_cci_glaciers", "regex": r"\bglaciers?[_\s]cci\b|esa\s+glaciers\s+climate\s+change"},
        ],
    },
    {
        "query_id": "HA05_EOA_POLAR_ICE_SHEET_DEM",
        "source_framework": "GCOS ECV / USGEO EOA 2023 (Cryosphere)",
        "policy_ref": "Cryosphere ECV — Greenland and Antarctic continental ice sheet digital elevation topography",
        "query": "High-resolution polar digital elevation models of the Greenland and Antarctic continental ice sheets derived from satellite radar and altimetry",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "gimp_dem", "regex": r"\bgimp\b.*digital\s+elevation|greenland\s+ice\s+mapping\s+project"},
            {"family_id": "ramp_dem", "regex": r"\bramp\b.*antarctic.*elevation|radarsat\s+antarctic\s+mapping\s+project"},
        ],
    },
    {
        "query_id": "HA06_ECV_SEA_ICE_CONCENTRATION",
        "source_framework": "GCOS ECV / USGEO EOA 2023 (Cryosphere)",
        "policy_ref": "Ocean & Cryosphere ECV — Daily global passive microwave sea ice concentration climate data record",
        "query": "Multi-decadal daily polar sea ice concentration and sea ice extent climate data record derived from satellite passive microwave radiometry",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "osisaf_sic_cdr", "regex": r"sea\s+ice\s+concentration.*(?:osi|eumetsat|amsr)", "exclude_regex": r"\bnsidc\b|\bcci\b|\bnoaa\b"},
            {"family_id": "nsidc_cci_sic", "regex": r"sea\s+ice\s+concentration.*(?:nsidc|cci|noaa)", "exclude_regex": r"\bosi[- ]?saf\b|eumetsat"},
        ],
    },
    {
        "query_id": "HA07_SNWG_SEA_ICE_FREEBOARD",
        "source_framework": "NASA SNWG / GCOS Cryosphere ECV",
        "policy_ref": "Cryosphere ECV — Satellite altimetry polar sea ice freeboard and sea ice thickness",
        "query": "Satellite laser and radar altimetry measurements of polar sea ice freeboard height and sea ice thickness",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "icesat2_atl10", "regex": r"\batl10\b|sea\s+ice\s+freeboard"},
            {"family_id": "cryosat_smos_sit", "regex": r"sea\s+ice\s+thickness.*(?:cryosat|smos|cci|cs2smos)"},
        ],
    },
    {
        "query_id": "HA08_ECV_AEROSOL_OPTICAL_DEPTH",
        "source_framework": "GCOS ECV / USGEO EOA 2023 (Atmosphere)",
        "policy_ref": "Atmosphere ECV — Global satellite aerosol optical depth climate data record and reanalysis",
        "query": "Global multi-year satellite atmospheric aerosol optical depth and total aerosol loading climate data record",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "avhrr_cdr_aod", "regex": r"aerosol\s+optical\s+depth.*avhrr|avhrr.*aerosol\s+optical"},
            {"family_id": "esa_cci_aerosol", "regex": r"\baerosol[_\s]cci\b|esa\s+aerosol\s+climate\s+change"},
            {"family_id": "modis_maiac_merra_aod", "regex": r"aerosol\s+optical\s+depth.*(?:modis|viirs|maiac|merra|mcd19|mod08)"},
        ],
    },
    {
        "query_id": "HA09_ECV_ATMOSPHERIC_OZONE",
        "source_framework": "GCOS ECV / Montreal Protocol (WMO OSCAR)",
        "policy_ref": "Atmosphere ECV — Global total column and vertical profile atmospheric ozone",
        "query": "Global satellite measurements of total column ozone and stratospheric and tropospheric ozone profiles for ozone layer monitoring",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "esa_cci_ozone", "regex": r"\bozone[_\s]cci\b|esa\s+ozone\s+climate\s+change"},
            {"family_id": "total_col_ozone", "regex": r"total\s+(?:column\s+)?ozone", "exclude_regex": r"\bozone[_\s]cci\b|esa\s+ozone\s+climate\s+change"},
        ],
    },
    {
        "query_id": "HA10_SNWG_TROPOSPHERIC_FORMALDEHYDE",
        "source_framework": "NASA SNWG / Air Quality Precursors",
        "policy_ref": "Atmospheric Chemistry — Gridded satellite tropospheric formaldehyde total column",
        "query": "Gridded Level 3 satellite measurements of atmospheric tropospheric formaldehyde column density as a volatile organic compound air quality precursor",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "tempo_hcho", "regex": r"\btempo_hcho\b|tempo.*formaldehyde"},
            {"family_id": "tropomi_omi_hcho", "regex": r"formaldehyde.*(?:column|tropospheric|l3|omi|tropomi|sentinel-5)", "exclude_regex": r"\btempo\b|\btempo_hcho\b"},
        ],
    },
    {
        "query_id": "HA11_SNWG_CLOUD_PROPERTIES",
        "source_framework": "GCOS ECV / NASA SNWG (Atmosphere)",
        "policy_ref": "Atmosphere ECV — Satellite gridded cloud cover fraction, cloud pressure, and optical properties",
        "query": "Global and geostationary Level 3 gridded satellite cloud cover fraction, cloud-top pressure, and cloud optical property climate data records",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "esa_cci_cloud", "regex": r"(?<![a-z0-9])cloud[_\s]cci(?![a-z0-9])|esa\s+cloud\s+climate\s+change"},
            {"family_id": "tempo_cldo4", "regex": r"(?<![a-z0-9])tempo_cldo4(?![a-z0-9])|tempo\s+gridded\s+cloud\s+fraction"},
            {
                "family_id": "modis_cdr_clouds",
                "regex": r"cloud\s+(?:fraction|properties|cover).*(?:cdr|l3|level\s*3|climate|isccp|patmos|modis)|modis.*cloud\s+(?:fraction|properties|cover)",
                "exclude_regex": r"(?<![a-z0-9])cloud[_\s]cci(?![a-z0-9])|esa\s+cloud\s+climate\s+change|(?<![a-z0-9])tempo(?![a-z0-9])",
            },
        ],
    },
    {
        "query_id": "HA12_EOA_RADIATION_BUDGET",
        "source_framework": "GCOS ECV / USGEO EOA 2023 (Climate)",
        "policy_ref": "Earth Radiation Budget ECV — Top-of-atmosphere and surface radiative fluxes",
        "query": "Global energy-balanced top-of-atmosphere and surface shortwave and longwave radiative energy budget fluxes from satellite radiometers",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "ceres_ebaf", "regex": r"\bceres\b.*ebaf"},
            {"family_id": "clara_radiation", "regex": r"\bclara\b.*radiation|surface\s+radiation\s+budget"},
        ],
    },
    {
        "query_id": "HA13_EOA_LAND_SURFACE_EMISSIVITY",
        "source_framework": "USGEO EOA 2023 (Climate & Land)",
        "policy_ref": "Thermal Infrared Land Property — High-resolution global land surface spectral emissivity",
        "query": "Global 100 meter multispectral thermal infrared land surface emissivity database derived from satellite thermal observations",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "aster_ged", "regex": r"\baster\s+global\s+emissivity\b|\bag100\b"},
        ],
    },
    {
        "query_id": "HA14_ECV_LAND_SURFACE_TEMPERATURE",
        "source_framework": "GCOS ECV / WMO OSCAR (Land)",
        "policy_ref": "Terrestrial ECV — Global satellite skin Land Surface Temperature (LST, NOT ocean SST)",
        "query": "Global satellite-derived terrestrial land surface skin temperature daily and monthly climate time series from thermal infrared sensors",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "modis_lst", "regex": r"(?<![a-z0-9])(?:mod|myd)11[ac]\d(?![a-z0-9])|(?<![a-z0-9])mod21(?![a-z0-9])|(?<![a-z0-9])modis(?![a-z0-9]).*land\s+surface\s+temperature", "exclude_regex": r"(?<![a-z0-9])lst[_\s]cci(?![a-z0-9])|climate\s+change\s+initiative|geesebal|evapotranspiration"},
            {"family_id": "esa_cci_lst", "regex": r"\blst[_\s]cci\b|esa\s+land\s+surface\s+temperature\s+climate\s+change"},
            {"family_id": "clms_lst", "regex": r"land\s+surface\s+temperature.*(?:5\s*km|copernicus|hourly|10-daily|clms|sentinel-3)", "exclude_regex": r"\b(?:mod|myd)11|\bmod21\b|modis|\blst[_\s]cci\b|climate\s+change\s+initiative"},
        ],
    },
    {
        "query_id": "HA15_EOA_WILDFIRE_FUEL_MODELS",
        "source_framework": "USGEO EOA 2023 (Agriculture & Forestry)",
        "policy_ref": "Wildfire Risk — Spatially explicit surface fire behavior fuel models",
        "query": "30 meter spatial fire behavior fuel model classification maps representing surface wildfire fuel loads and spread potential",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "landfire_fbfm", "regex": r"\bfire\s+behavior\s+fuel\s+models\b|\bfbfm(?:40|13)\b"},
        ],
    },
    {
        "query_id": "HA16_EOA_SMOKE_PLUMES_ACTIVE_FIRE",
        "source_framework": "USGEO EOA 2023 / GCOS ECV (Fire)",
        "policy_ref": "Wildfire Emissions — Satellite active fire detections, fire radiative power, and wildfire smoke plumes",
        "query": "Near-real-time satellite active fire thermal anomaly detections, fire radiative power, and wildfire smoke plume extent mapping",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "hms_smoke", "regex": r"\bsmoke\s+plumes\b"},
            {"family_id": "active_fire_frp", "regex": r"\bfire\s+radiative\s+power\b|\bactive\s+fires?\b.*(?:modis|viirs|gfas|global|gwis|wildfire)|thermal\s+anomal(?:y|ies).*fire"},
        ],
    },
    {
        "query_id": "HA17_SNWG_LAND_SURFACE_DISTURBANCE",
        "source_framework": "NASA SNWG / USGEO EOA 2023",
        "policy_ref": "Ecosystem Disturbance — 30 m satellite near-real-time vegetation loss and land surface disturbance alerts",
        "query": "30 meter satellite near-real-time and annual land surface vegetation disturbance alerts tracking abrupt land cover change",
        "filters": {"max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "opera_dist", "regex": r"(?<![a-z0-9])opera_l3_dist(?![a-z0-9])|land\s+surface\s+disturbance\s+(?:alert|annual)|global\s+integrated\s+disturbance\s+alerts"},
            {"family_id": "landfire_dist", "regex": r"\blandfire\s+annual\s+disturbance\b"},
        ],
    },
    {
        "query_id": "HA18_ECV_OCEAN_SURFACE_WINDS",
        "source_framework": "GCOS ECV / WMO OSCAR (Ocean)",
        "policy_ref": "Ocean Physics ECV — Global satellite ocean surface vector wind speed and wind stress",
        "query": "Global Level 3 and Level 4 gridded ocean surface vector wind speed, direction, and wind stress derived from satellite scatterometers",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "ocean_wind_scat_cmems", "regex": r"(?:global\s+ocean|sea\s+surface).*wind.*(?:l3|l4|reanalysis|forecast|scatterometer|ascat|ccmp|hourly|daily)|scatsat.*wind|quikscat.*wind|ascat.*wind"},
        ],
    },
    {
        "query_id": "HA19_ECV_OCEAN_SURFACE_WAVES",
        "source_framework": "GCOS ECV / WMO OSCAR (Ocean)",
        "policy_ref": "Ocean Physics ECV — Global significant wave height, wave period, and sea state reanalysis",
        "query": "Global gridded ocean surface wave analysis, hindcast, and satellite altimetry significant wave height sea state climate data record",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cmems_waves", "regex": r"global\s+ocean\s+waves"},
            {"family_id": "esa_cci_seastate", "regex": r"\bsea[_\s]state[_\s]cci\b|significant\s+wave\s+height"},
        ],
    },
    {
        "query_id": "HA20_ECV_OCEAN_SURFACE_CURRENTS",
        "source_framework": "GCOS ECV / WMO OSCAR (Ocean)",
        "policy_ref": "Ocean Physics ECV — Global geostrophic and Ekman ocean surface currents from satellite altimetry and winds",
        "query": "Global gridded ocean surface geostrophic and Ekman current velocity estimates derived from satellite altimetry and wind stress",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {
                "family_id": "oscar_globcurrent",
                "regex": r"(?<![a-z0-9])globcurrent(?![a-z0-9])|(?<![a-z0-9])oscar(?:_l4_oc)?(?![a-z0-9]).*(?:ocean\s+surface\s+current|current|oc_)|geostrophic\s+current|sea\s+surface\s+currents?\s*\(geostrophic\)",
                "exclude_regex": r"WGCLIMATE_CDR_12374",
            },
        ],
    },
    {
        "query_id": "HA21_ECV_TERRESTRIAL_WATER_STORAGE",
        "source_framework": "GCOS ECV / WMO OSCAR (Hydrology)",
        "policy_ref": "Terrestrial Hydrology ECV — Global satellite gravimetry terrestrial water storage and groundwater mass anomalies",
        "query": "Global monthly satellite gravimetry measurements of terrestrial water storage anomalies, equivalent water thickness, and aquifer groundwater depletion",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "grace_tws", "regex": r"(?<![a-z0-9])grace(?![a-z0-9]).*(?:mascon|terrestrial\s+water\s+storage|gravity|groundwater|liquid\s+water\s+equivalent|water[- ]equivalent[- ]thickness)|tellus_grac.*lnd|\bterrestrial\s+water\s+storage\b.*(?:global|gridded|satellite|grace)|global\s+gravity-based\s+groundwater|(?<![a-z0-9])g3p(?![a-z0-9]).*groundwater"},
        ],
    },
    {
        "query_id": "HA22_ECV_PERMAFROST",
        "source_framework": "GCOS ECV / WMO OSCAR (Cryosphere)",
        "policy_ref": "Cryosphere ECV — Northern Hemisphere permafrost extent, ground temperature, and active layer thickness",
        "query": "Satellite-constrained circumpolar permafrost extent, mean annual ground temperature, and active layer thickness climate data record",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "esa_cci_permafrost", "regex": r"\bpermafrost[_\s]cci\b|esa\s+permafrost\s+climate\s+change|northern\s+hemisphere\s+permafrost"},
        ],
    },
    {
        "query_id": "HA23_ECV_SURFACE_ALBEDO",
        "source_framework": "GCOS ECV / WMO OSCAR (Land)",
        "policy_ref": "Terrestrial ECV — Global broadband shortwave land surface directional and bihemispherical albedo",
        "query": "Global satellite-derived land surface broadband shortwave black-sky and white-sky surface albedo climate time series",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "modis_albedo", "regex": r"\bmcd43a3\b|modis.*albedo"},
            {
                "family_id": "clms_cci_albedo",
                "regex": r"\bsurface\s+albedo\b.*(?:global|1\s*km|300\s*m|cci|copernicus|10-?daily|gridded\s+data\s+from\s+1981)",
                "exclude_regex": r"\bmcd43a3\b|modis",
            },
        ],
    },
    {
        "query_id": "HA24_ECV_ATMOSPHERIC_WATER_VAPOUR",
        "source_framework": "GCOS ECV / WMO OSCAR (Atmosphere)",
        "policy_ref": "Atmosphere ECV — Global total column atmospheric water vapour and precipitable water",
        "query": "Global satellite climate data record of total column atmospheric water vapour and integrated precipitable water",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "esa_cci_wv", "regex": r"water[_\s]vapou?r[_\s]cci|esa\s+water\s+vapou?r\s+climate\s+change"},
            {"family_id": "tcwv_tpw", "regex": r"total\s+column\s+water\s+vapou?r|precipitable\s+water", "exclude_regex": r"water[_\s]vapou?r[_\s]cci|esa\s+water\s+vapou?r\s+climate\s+change"},
        ],
    },
    {
        "query_id": "HA25_EOA_INTERTIDAL_CORAL_COASTAL",
        "source_framework": "USGEO EOA 2023 / CBD Marine Coastal Targets",
        "policy_ref": "Coastal Ecosystems — Intertidal tidal flat change, coral reef habitat mapping, and coastal land cover change",
        "query": "High-resolution satellite mapping of coastal intertidal tidal flats, benthic coral reef habitats, and coastal zone land cover change",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "noaa_ccap", "regex": r"\bc-cap\b.*land\s+cover"},
            {"family_id": "intertidal_change", "regex": r"\bintertidal\b.*(?:change|tidal\s+flat|mudflat|global)"},
            {
                "family_id": "coral_reef_atlas",
                "regex": r"\ballen\s+coral\s+atlas\b|\bcoral\s+atlas\b|global\s+distribution\s+of\s+coral\s+reefs",
                "exclude_regex": r"\bcortad\b|temperature\s+anomaly|thermal\s+stress|bleaching|coral\s+reef\s+watch|degree\s+heating|hotspots?|heat\s+stress|\bsst\b",
            },
        ],
    },
    {
        "query_id": "HA26_ECV_RIVER_LAKE_WATER_LEVEL",
        "source_framework": "GCOS ECV / EC KCEO Biodiversity (Freshwater)",
        "policy_ref": "Inland Hydrology ECV — Satellite radar altimetry river and lake water surface elevation time series",
        "query": "Global satellite radar altimetry measurements of inland river and lake water surface level height time series",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_water_level", "regex": r"\briver\s+water\s+levels?\b|\blake\s+water\s+levels?\b|\bwater\s+level\s+(?:rivers|lakes)\b|(?<![a-z0-9])wl_rivers"},
        ],
    },
    {
        "query_id": "HA27_OSCAR_SULFUR_DIOXIDE",
        "source_framework": "WMO OSCAR / Air Pollution & Volcanic Monitoring",
        "policy_ref": "Atmospheric Chemistry — Satellite total column sulfur dioxide from volcanic eruptions and anthropogenic sources",
        "query": "Global satellite Level 3 gridded measurements of atmospheric sulfur dioxide total vertical column from industrial and volcanic emissions",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "so2_col", "regex": r"(?<![a-z0-9])sul(?:f|ph)ur\s+dioxide(?![a-z0-9])|(?<![a-z0-9])so2(?![a-z0-9]).*(?:column|volcanic|l3|omi|tropomi|omps)"},
        ],
    },
    {
        "query_id": "HA28_ECV_CARBON_MONOXIDE",
        "source_framework": "GCOS ECV / WMO OSCAR (Atmosphere)",
        "policy_ref": "Atmospheric Chemistry — Global tropospheric carbon monoxide total column and mixing ratio",
        "query": "Global satellite retrievals of atmospheric tropospheric carbon monoxide total column concentrations for air quality and biomass burning tracking",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "co_col", "regex": r"\bcarbon\s+monoxide\b.*(?:column|mopitt|iasi|tropomi|cams|l3|mixing)"},
        ],
    },
    {
        "query_id": "HA29_KCEO_GLOBAL_BATHYMETRY",
        "source_framework": "EC KCEO Marine / UN Ocean Decade (Seabed 2030)",
        "policy_ref": "Marine Topography — Global and European gridded ocean seafloor bathymetry depth grids",
        "query": "Gridded global and European ocean seafloor bathymetry depth and marine seabed elevation models",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "gebco_bathymetry", "regex": r"(?<![a-z0-9])gebco(?![a-z0-9])"},
            {"family_id": "emodnet_bathymetry", "regex": r"\bemodnet\b.*bathymetry"},
        ],
    },
    {
        "query_id": "HA30_KCEO_SOLAR_IRRADIANCE_PV",
        "source_framework": "EC KCEO Urban Adaptation (Indicator U121 — Clean Energy)",
        "policy_ref": "Renewable Energy — Surface solar irradiance and photovoltaic solar power potential mapping",
        "query": "Gridded global surface downwelling solar irradiance and solar photovoltaic power generation potential derived from geostationary satellites",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {
                "family_id": "solar_irradiance_pv",
                "regex": r"\bglobal\s+solar\s+atlas\b|(?<![a-z0-9])pvgis(?![a-z0-9])|photovoltaic.*(?:potential|power|radiation|irradiance)|solar\s+radiation.*(?:cams|sarah|surface)",
                "exclude_regex": r"photovoltaics\s+inventory|water-surface\s+solar|building\s+stock|(?<![a-z0-9])dbsm(?![a-z0-9])|elevation\s+above\s+sea|temperature\s+at\s+2m|terrain\s+elevation|air\s+temperature",
            },
        ],
    },
    {
        "query_id": "HA31_EETT_VEGETATION_CONTINUOUS_FIELDS",
        "source_framework": "CEOS EETT / USGEO EOA 2023 (Vegetation Structure)",
        "policy_ref": "Ecosystem Cover — Annual global fractional percent tree cover, non-tree vegetation, and bare ground continuous fields",
        "query": "Annual 250 meter global fractional vegetation continuous fields representing percent tree cover, herbaceous vegetation, and bare ground",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "mod44b_vcf", "regex": r"\bmod44b\b|vegetation\s+continuous\s+fields"},
            {"family_id": "landfire_evc", "regex": r"\blandfire\s+evc\b|existing\s+vegetation\s+cover"},
        ],
    },
    {
        "query_id": "HA32_UNHABITAT_HUMAN_MODIFICATION",
        "source_framework": "UN-Habitat EO Toolkit / CBD Anthropogenic Pressure",
        "policy_ref": "Human Footprint — Cumulative global human modification of terrestrial ecosystems and land development threat",
        "query": "Global cumulative index of anthropogenic human modification of terrestrial lands and future land development threat pressure",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "sedac_hmts", "regex": r"global\s+human\s+modification"},
            {"family_id": "sedac_dti", "regex": r"development\s+threat\s+index"},
        ],
    },
    {
        "query_id": "HA33_EOA_TOPOGRAPHIC_DIVERSITY_INSOLATION",
        "source_framework": "USGEO EOA 2023 (Biodiversity & Terrain Microclimate)",
        "policy_ref": "Abiotic Microclimate — Multi-scale topographic position index, heat-insolation load, and terrain diversity",
        "query": "Multi-scale terrain topographic position index, continuous heat-insolation load, and abiotic topographic diversity derived from elevation models",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "ergo_topo_chili", "regex": r"\bsrtm\s+chili\b|\bmtpi\b|topographic\s+diversity"},
        ],
    },
    {
        "query_id": "HA34_SNWG_HARMONIZED_ARD_REFLECTANCE",
        "source_framework": "NASA SNWG / UNCCD GPG v2 (ARD Foundation)",
        "policy_ref": "Foundational ARD — 30 m harmonized cloud-free optical surface reflectance mosaics",
        "query": "30 meter harmonized multispectral optical surface reflectance analysis-ready data combining medium-resolution polar-orbiting imagers",
        "filters": {"max_grid_m": 100.0, "strict_eo_only": False},
        "expected_families": [
            {
                "family_id": "nasa_hls",
                "regex": r"\bhlsl30\b|\bhlss30\b|harmonized\s+landsat\s+sentinel",
                "exclude_regex": r"opera_l3_(?:dswx|dist)|dynamic\s+surface\s+water|disturbance",
            },
            {"family_id": "glad_ard", "regex": r"\bglad\b.*ard|cloud-free\s+reconstructed\s+landsat"},
        ],
    },
    {
        "query_id": "HA35_FAO_CROP_WATER_PRODUCTIVITY",
        "source_framework": "FAO Water Productivity / SDG 6.4.1",
        "policy_ref": "Agricultural Water Efficiency — Satellite evapotranspiration and biomass water productivity per drop",
        "query": "Satellite monitoring of agricultural crop water productivity, net biomass production per unit water consumed, and actual evapotranspiration",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "fao_wapor", "regex": r"\bwapor\b|water\s+productivity"},
        ],
    },
]

# Strict anti-leakage list: ZERO product brand names / codes / missions allowed in Holdout A queries!
FORBIDDEN_LEAKS_HA = [
    "imerg", "gpm", "chirps", "gpcp", "cpc", "era5", "wfdei", "nldas", "glims", "rgi",
    "gimp", "ramp", "osisaf", "osi saf", "nsidc", "icesat", "atl10", "cryosat", "smos",
    "avhrr", "modis", "viirs", "merra", "maiac", "tempo", "tropomi", "omi", "ceres",
    "ebaf", "clara", "aster", "ag100", "mod11", "myd11", "landfire", "fbfm", "hms",
    "gfas", "opera", "ascat", "ccmp", "cmems", "globcurrent", "oscar", "grace", "mcd43",
    "c-cap", "ccap", "g-realm", "grealm", "hydroweb", "mopitt", "iasi", "gebco", "emodnet",
    "sarah", "mod44b", "hmts", "sedac", "srtm", "chili", "mtpi", "hls", "hlsl30", "hlss30",
    "glad", "wapor", "grid3", "sentinel", "landsat", "copernicus", "esa cci", "clms", "cams",
]


def main() -> None:
    print("=" * 88)
    print("08. BUILD AUDITED EXTERNAL DOMAIN SET (HOLDOUT A — RETROSPECTIVE GT-AUDITED, 35 QUERIES)")
    print("=" * 88)

    # Anti-leakage verification
    leaks = []
    for sp in HOLDOUT_A_SPECS:
        ql = sp["query"].lower()
        for tok in FORBIDDEN_LEAKS_HA:
            if re.search(rf"\b{re.escape(tok)}\b", ql):
                leaks.append((sp["query_id"], tok))
    if leaks:
        raise ValueError(f"Product/mission name leakage in Holdout A queries: {leaks}")
    print(f"[OK] Anti-leakage check passed across all {len(HOLDOUT_A_SPECS)} Holdout A queries (0 leaks).")

    df = pd.read_parquet(
        CANONICAL_PARQUET,
        columns=["record_id", "title", "raw_id", "source_catalogue", "is_superseded_edition"],
    )
    obs_re = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)
    active_mask = (~df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)) & (
        ~df["title"].fillna("").str.contains(obs_re, regex=True).to_numpy(dtype=bool, copy=True)
    )
    df = df[active_mask].reset_index(drop=True)
    comb = df["title"].fillna("") + " || " + df["raw_id"].fillna("") + " || " + df["source_catalogue"].fillna("")

    resolved = []
    missing = []
    total_fams = 0
    for sp in HOLDOUT_A_SPECS:
        fams_out = []
        for fam in sp["expected_families"]:
            mask = comb.str.contains(re.compile(fam["regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                mask = mask & ~comb.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            mask = filter_benchmark_family_mask(df, mask, sp.get("filters"))
            idxs = np.flatnonzero(mask)
            if len(idxs) == 0:
                missing.append((sp["query_id"], fam["family_id"]))
                continue
            total_fams += 1
            fams_out.append({
                "family_id": fam["family_id"],
                "n_matches": int(len(idxs)),
                "sample_titles": df.iloc[idxs[:3]]["title"].tolist(),
                "record_ids": df.iloc[idxs]["record_id"].tolist(),
            })
        resolved.append({
            "query_id": sp["query_id"],
            "source_framework": sp["source_framework"],
            "policy_ref": sp["policy_ref"],
            "query": sp["query"],
            "filters": sp.get("filters", {}),
            "n_expected_families": len(fams_out),
            "expected_families": fams_out,
        })

    if missing:
        raise RuntimeError(f"Holdout A families matched 0 records: {missing}")

    # Strict peer-review guard: ZERO overlapping record_ids across distinct families in any query!
    overlaps = []
    for q in resolved:
        fams = q["expected_families"]
        for i in range(len(fams)):
            s1 = set(fams[i]["record_ids"])
            for j in range(i + 1, len(fams)):
                inter = s1 & set(fams[j]["record_ids"])
                if inter:
                    overlaps.append((q["query_id"], fams[i]["family_id"], fams[j]["family_id"], len(inter)))
    if overlaps:
        raise RuntimeError(f"Overlapping ground-truth families detected in Holdout A: {overlaps}")
    print("[OK] Strict family-disjointness assertion PASSED (0 overlapping record_ids across all 35 queries).")

    # Cross-suite disjointness assertion vs Dev-45 (pure AST read — zero heavy imports!)
    import ast

    b02_tree = ast.parse((SCRIPT_DIR / "02_build_policy_benchmark.py").read_text(encoding="utf-8"))
    dev_specs = None
    for node in b02_tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "BENCHMARK_SPECS":
            dev_specs = ast.literal_eval(node.value)
            break
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if getattr(tgt, "id", None) == "BENCHMARK_SPECS":
                    dev_specs = ast.literal_eval(node.value)
                    break
    assert dev_specs is not None, "Could not parse BENCHMARK_SPECS from 02_build_policy_benchmark.py"
    dev_rids: set[str] = set()
    for sp in dev_specs:
        for fam in sp["expected_families"]:
            m = comb.str.contains(re.compile(fam["regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                m = m & ~comb.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            m = filter_benchmark_family_mask(df, m, sp.get("filters"))
            dev_rids.update(df.iloc[np.flatnonzero(m)]["record_id"].tolist())
    ha_rids: set[str] = set()
    for q in resolved:
        for f in q["expected_families"]:
            ha_rids.update(f["record_ids"])
    cross_overlap = sorted(dev_rids & ha_rids)
    if cross_overlap:
        raise RuntimeError(f"Cross-suite Dev-45 ∩ Holdout A record_id overlap detected ({len(cross_overlap)}): {cross_overlap[:10]}")
    print("[OK] Cross-suite Dev-45 ∩ Holdout A assertion PASSED (0 overlapping record_ids!).")

    payload = {
        "benchmark_name": "Audited_External_Domain_Set_Holdout_A_v3",
        "blind_status": "retrospective_gt_audited",
        "n_queries": len(resolved),
        "n_total_families": total_fams,
        "disjoint_from_dev_45": True,
        "queries": resolved,
    }
    out_p = ARTIFACTS_DIR / "holdout_a_external_benchmark.json"
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] All {len(resolved)} queries and {total_fams}/{total_fams} disjoint product families resolved cleanly!")
    print(f"Saved -> {out_p}")


if __name__ == "__main__":
    main()
