# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
# ]
# ///
"""Build a 45-query Policy-Requirement-to-EO-Product retrieval benchmark & hard 3,500-record subset.

Every query is strictly grounded in downloaded external policy validation sources
(05_external_policy_validation/raw/) with ZERO product-name or acronym leakage into the query text.
Each query maps an abstract policy/treaty requirement (+ optional realistic spatial/scope/EO filters)
to verified canonical product families present in eo_l3_l4_canonical_products.parquet.
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
from eo_search_lib import compute_family_signature, filter_benchmark_family_mask  # noqa: E402


# ---------------------------------------------------------------------------
# 45 Formalized Policy-Requirement Queries (Zero Product-Name Leakage!)
# ---------------------------------------------------------------------------
# Each family spec has:
#   - family_id: canonical logical product family name (for Family-Recall@K!)
#   - regex: case-insensitive regex matched against (title + " " + raw_id + " " + source_catalogue)
#   - exclude_regex (optional): guards against false-positive title collisions
#   - source_filter (optional): restrict family match to specific source_catalogue substring(s)
#
# Anti-leakage check is enforced programmatically below against a forbidden list of product names!

BENCHMARK_SPECS: list[dict[str, Any]] = [
    # ==================== A. EU BIODIVERSITY STRATEGY 2030 (KCEO Annex 3) ====================
    {
        "query_id": "Q01_EU_SOIL_SEALING",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Indicator sdg_15_41 — Soil sealing index (<=30 m)",
        "query": "Soil sealing index and percentage of artificial impervious surface density and built-up change across Europe",
        "filters": {"geo_scope": "europe", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_hrl_imperviousness", "regex": r"imperviousness\s+(?:density|status|change|built)"},
            {"family_id": "clms_urban_atlas", "regex": r"\burban\s+atlas\b.*land\s+cover"},
            {"family_id": "ghsl_built", "regex": r"(?<![a-z0-9])ghs[-_\s]built(?:[-_\s][sc])?(?![a-z0-9])|global\s+human\s+settlement.*built[- ]up", "exclude_regex": r"ghs[-_\s]obat|open\s+buildings\s+and\s+terrain"},
        ],
    },
    {
        "query_id": "Q02_EU_URBAN_GREEN_CANOPY",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Target 13 — Urban green space and urban tree canopy cover in cities and towns",
        "query": "Urban green space extent and urban tree canopy cover density across European cities, towns, and suburbs",
        "filters": {"geo_scope": "europe", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_urban_atlas", "regex": r"\burban\s+atlas\b"},
            {"family_id": "clms_hrl_tree_cover", "regex": r"\btree\s+cover\s+density\b"},
            {"family_id": "clms_small_woody", "regex": r"\bsmall\s+woody\s+features\b"},
        ],
    },
    {
        "query_id": "Q03_EU_FOREST_PROTECTED_STRUCTURE",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Target 2 — Primary and old-growth forest protection, dominant leaf type and tree cover",
        "query": "High-resolution European forest type classification, dominant broadleaf and coniferous leaf type, and tree cover density change",
        "filters": {"geo_scope": "europe", "max_grid_m": 30.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_hrl_tree_cover", "regex": r"\btree\s+cover\s+density\b"},
            {"family_id": "clms_hrl_leaf_type", "regex": r"\bdominant\s+leaf\s+type\b|\bforest\s+type\b.*europe"},
        ],
    },
    {
        "query_id": "Q04_EU_HABITAT_PHENOLOGY",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Habitats Directive Annex I — Change in vegetation phenology and productivity",
        "query": "High-resolution seasonal vegetation phenology trajectories, start and end of growing season, and plant productivity dynamics in Europe",
        "filters": {"geo_scope": "europe", "max_grid_m": 30.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_hr_vpp", "regex": r"\bplant\s+phenology\s+index\b|\bvegetation\s+phenology\s+and\s+productivity\b|(?<![a-z0-9])hr-?vpp(?![a-z0-9])|(?:start-of-season|end-of-season|season\s+length|seasonal\s+productivity).*(?:europe|10\s*m|phenology|vpp)"},
        ],
    },
    {
        "query_id": "Q05_EU_RIVER_LAKE_ICE",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Target 4 — Restoration and monitoring of freshwater ice-covered ecosystems",
        "query": "High-resolution monitoring of seasonal river and lake ice extent across European inland waters",
        "filters": {"geo_scope": "europe", "max_grid_m": 30.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_rlie", "regex": r"\briver\s+and\s+lake\s+ice\s+extent\b|\brlie\b"},
        ],
    },
    {
        "query_id": "Q06_EU_SEABED_HABITATS",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Marine Target — Broad-scale seabed habitats, ecological corridors at sea, and seafloor integrity",
        "query": "Broad-scale predictive marine seabed habitat classification and seafloor benthic substrate mapping in European seas",
        "filters": {"geo_scope": "europe", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "emodnet_euseamap", "regex": r"\beuseamap\b"},
        ],
    },
    {
        "query_id": "Q07_EU_MARINE_3D_PHYSICS",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Indicator A1.2.1 — Marine protected area 3D ocean currents, temperature, salinity and sea ice",
        "query": "Three-dimensional global ocean physical analysis and forecast of ocean currents, potential temperature, salinity, and sea ice",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cmems_global_phy", "regex": r"global\s+ocean\s+physics\s+(?:analysis\s+and\s+forecast|reanalysis)"},
        ],
    },
    {
        "query_id": "Q08_EU_MARINE_BGC",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "MPA management effectiveness — Ocean dissolved oxygen, nutrients, acidity, phytoplankton and chlorophyll",
        "query": "Global ocean biogeochemistry analysis and forecast of dissolved oxygen, nitrate, phosphate, ocean acidity pH, phytoplankton, and chlorophyll",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cmems_global_bgc", "regex": r"global\s+ocean\s+biogeochemistry\s+(?:analysis\s+and\s+forecast|hindcast)"},
        ],
    },
    {
        "query_id": "Q09_EU_LAKE_WATER_QUALITY",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Freshwater water quality — Inland lake turbidity, trophic state index and phytoplankton",
        "query": "Satellite monitoring of inland lake water quality including turbidity, trophic state index, and suspended matter",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_lwq", "regex": r"\blake\s+water\s+quality\b"},
            {"family_id": "esa_cci_lakes", "regex": r"\besa\s+lakes\s+climate\s+change\s+initiative\b|\blakes_cci\b"},
        ],
    },
    {
        "query_id": "Q10_EU_WETLAND_RIPARIAN",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Freshwater connectivity — Wetland status, permanent and temporary water wetness, and riparian buffer zones",
        "query": "High-resolution European wetland extent, permanent and temporary water and wetness status, and riparian buffer zone land cover",
        "filters": {"geo_scope": "europe", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_water_wetness", "regex": r"\bwater\s+and\s+wetness\b"},
            {"family_id": "clms_riparian_zones", "regex": r"\briparian\s+zones\b"},
        ],
    },
    {
        "query_id": "Q11_EU_SMALL_WOODY_LANDSCAPE",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Agro-ecosystems — High-diversity landscape features, hedgerows and small woody elements",
        "query": "Mapping of linear hedgerows, scattered trees, and small woody landscape features across European agricultural areas",
        "filters": {"geo_scope": "europe", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_small_woody", "regex": r"\bsmall\s+woody\s+features\b"},
        ],
    },
    {
        "query_id": "Q12_EU_CROP_TYPE_DIVERSITY",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Agricultural biodiversity — High-resolution crop type classification across Europe",
        "query": "High-resolution 10 meter satellite crop type classification map across European Union agricultural parcels",
        "filters": {"geo_scope": "europe", "max_grid_m": 30.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "jrc_eucropmap", "regex": r"\beucropmap\b|\beu\s+crop\s+map\b"},
            {"family_id": "esa_worldcereal", "regex": r"\bworldcereal\b"},
        ],
    },
    {
        "query_id": "Q13_EU_PROTECTED_AREA_HABITAT_LC",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Indicator A1.1.2 — Land cover and land use inside European protected habitats and landscape connectivity",
        "query": "Detailed land cover and land use mapping for European protected habitat sites and continental ecosystem connectivity accounting",
        "filters": {"geo_scope": "europe", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_n2k", "regex": r"\bn2k\s+land\s+cover|\bnatura\s+2000\b.*land\s+cover"},
            {"family_id": "clms_corine", "regex": r"\bcorine\s+land\s+cover\b"},
            {"family_id": "clms_clc_plus", "regex": r"\bclc\+?\s*backbone\b|\bclcplus\b"},
        ],
    },
    {
        "query_id": "Q14_EU_WILDFIRE_BURNT_AREA",
        "policy_framework": "EU Biodiversity Strategy 2030 (EC KCEO Annex 3)",
        "policy_ref": "Forest disturbance — Wildfire burned area and forest fire events",
        "query": "Satellite-derived wildfire burned area extent and forest fire event mapping for ecosystem disturbance monitoring",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "cems_effis", "regex": r"\beffis\b|european\s+forest\s+fire\s+information\s+system"},
            {"family_id": "modis_mcd64", "regex": r"\bmcd64a1\b|burned\s+area\s+monthly\s+l3"},
            {"family_id": "esa_cci_fire", "regex": r"\bfire[_\s]cci\b|esa\s+fire\s+climate\s+change\s+initiative"},
        ],
    },

    # ==================== B. EC KCEO URBAN CLIMATE ADAPTATION ====================
    {
        "query_id": "Q15_URBAN_AIR_QUALITY_EUROPE",
        "policy_framework": "EC KCEO Urban Climate Adaptation Indicators",
        "policy_ref": "Air pollution — Urban atmospheric pollutant forecasts, ozone, nitrogen dioxide and PM2.5 in Europe",
        "query": "Operational European regional atmospheric air quality forecasts and reanalyses for particulate matter PM2.5, ozone, and nitrogen dioxide",
        "filters": {"geo_scope": "europe", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cams_eu_aq", "regex": r"\bcams\s+european\s+air\s+quality\b"},
            {"family_id": "sedac_pm25", "regex": r"\bglobal\s+annual\s+pm2\.?5\b"},
        ],
    },
    {
        "query_id": "Q16_URBAN_HEAT_ISLAND_EXPOSURE",
        "policy_framework": "EC KCEO Urban Climate Adaptation Indicators",
        "policy_ref": "Urban heat vulnerability — Urban heat island intensity, extreme daily urban heat exposure and local climate zones",
        "query": "Global urban heat island intensity, daily extreme urban heat exposure, and urban local climate zone classification for city climate adaptation",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "sedac_uhi", "regex": r"\bglobal\s+urban\s+heat\s+island\b"},
            {"family_id": "sedac_uhe", "regex": r"\bextreme\s+urban\s+heat\s+exposure\b"},
            {"family_id": "global_lcz", "regex": r"\blocal\s+climate\s+zones\b"},
        ],
    },
    {
        "query_id": "Q17_EUROPEAN_RIVER_FLOOD_FORECAST",
        "policy_framework": "EC KCEO Urban Climate Adaptation Indicators",
        "policy_ref": "Flood hazard — European river discharge forecasts, flood awareness and near-real-time flood inundation",
        "query": "Hydrological river discharge forecasts and satellite near-real-time flood inundation hazard monitoring across European river basins",
        "filters": {"geo_scope": "europe", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cems_efas", "regex": r"\befas\b|european\s+flood\s+awareness\s+system"},
            {"family_id": "cems_gfm", "regex": r"\bglobal\s+flood\s+monitoring\b|\bgfm\b.*flood", "exclude_regex": r"gpmtfmifld|\bgpm\b|16_nasa_cmr"},
        ],
    },
    {
        "query_id": "Q18_COASTAL_SEA_LEVEL_TREND",
        "policy_framework": "EC KCEO Urban Climate Adaptation Indicators",
        "policy_ref": "Coastal flood exposure — Satellite altimetry sea level anomaly and global mean sea level rise trend",
        "query": "Multi-mission satellite altimetry gridded sea level anomalies and long-term ocean sea surface height rise trends",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {
                "family_id": "cmems_c3s_sealevel",
                "regex": r"\bsea\s+level\b.*(?:anomal|gridded|reprocessed|altimet|trend|indicator)",
                "exclude_regex": r"\bcmip[56]\b|in[- ]?situ|tide\s+gauge|profile|\bncei\b|projection",
            },
        ],
    },
    {
        "query_id": "Q19_NIGHTTIME_LIGHTS_ELECTRIFICATION",
        "policy_framework": "EC KCEO Urban Climate Adaptation Indicators",
        "policy_ref": "Access to electricity — Satellite nighttime artificial lights and nocturnal radiance dynamics",
        "query": "Global satellite nighttime artificial light radiance and multitemporal nocturnal lights change for electrification and urban activity assessment",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "viirs_black_marble", "regex": r"\bblack\s+marble\b|\bvnp46a|\bnighttime\s+lights\b|\bchange\s+in\s+lights\b"},
        ],
    },

    # ==================== C. UNFCCC PARIS AGREEMENT GLOBAL STOCKTAKE (CEOS GST) ====================
    {
        "query_id": "Q20_GST_TOPDOWN_CO2_BUDGETS",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake (CEOS GST)",
        "policy_ref": "GHG Theme — Top-down atmospheric inversion CO2 net flux budgets",
        "query": "Satellite-constrained top-down atmospheric inversion estimates of annual carbon dioxide net surface fluxes and national CO2 budgets",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "oco2_mip_co2", "regex": r"\boco-?2\b.*mip.*(?:co2|co₂)\s+budget|\btop-down\s+(?:co2|co₂)\s+budget"},
            {"family_id": "oco2_geos_co2", "regex": r"\boco-?2\b.*geos.*(?:co2|co₂)|\bcolumn\s+(?:co2|co₂)\s+concentrations"},
        ],
    },
    {
        "query_id": "Q21_GST_TOPDOWN_CH4_BUDGETS",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake (CEOS GST)",
        "policy_ref": "GHG Theme — Top-down atmospheric inversion methane emissions and budgets",
        "query": "Satellite-derived top-down atmospheric inversion estimates of total, anthropogenic, and natural methane emissions and fluxes",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "gosat_ch4_budget", "regex": r"\bgosat\b.*methane|\btop-down.*methane\s+emissions"},
            {"family_id": "tm5_4dvar_ch4", "regex": r"\btm5-?4dvar\b.*(?:ch4|ch₄|methane)"},
        ],
    },
    {
        "query_id": "Q22_GST_FOSSIL_CO2_EMISSIONS",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake / US GHG Center",
        "policy_ref": "Anthropogenic GHG — Spatially explicit gridded fossil fuel carbon dioxide emissions",
        "query": "High-resolution spatially explicit gridded monthly and annual fossil fuel anthropogenic carbon dioxide emissions",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "odiac_ffco2", "regex": r"\bodiac\b.*fossil\s+fuel"},
            {"family_id": "vulcan_ffco2", "regex": r"\bvulcan\b.*fossil\s+fuel"},
        ],
    },
    {
        "query_id": "Q23_GST_GLOBAL_ABOVEGROUND_BIOMASS",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake (CEOS GST)",
        "policy_ref": "AFOLU — Global forest aboveground woody biomass density and terrestrial carbon stock",
        "query": "Global gridded forest aboveground woody biomass density and terrestrial live vegetation carbon stocks from spaceborne lidar and radar",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "gedi_l4b_biomass", "regex": r"\bgedi[_\s]l4b\b|gedi\s+l4b\s+gridded\s+aboveground"},
            {"family_id": "esa_cci_biomass", "regex": r"\bbiomass[_\s]cci\b|esa\s+biomass\s+climate\s+change\s+initiative"},
            {"family_id": "spawn_carbon_density", "regex": r"global\s+aboveground\s+and\s+belowground\s+biomass\s+carbon\s+density"},
            {"family_id": "jpl_xu_biomass", "regex": r"changes\s+in\s+global\s+terrestrial\s+live\s+biomass"},
        ],
    },
    {
        "query_id": "Q24_GST_FOREST_LOSS_CARBON_FLUX",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake (CEOS GST)",
        "policy_ref": "AFOLU Forests — Annual global tree cover loss, deforestation and forest carbon greenhouse gas flux",
        "query": "Annual 30 meter global tree cover loss, forest gain, deforestation disturbance, and forest greenhouse gas carbon emissions and removals",
        "filters": {"geo_scope": "global", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "hansen_gfc", "regex": r"\bhansen\s+global\s+forest\s+change\b"},
            {"family_id": "forest_carbon_flux", "regex": r"\bforest\s+carbon\s+(?:flux|stocks,\s+emissions)"},
        ],
    },
    {
        "query_id": "Q25_GST_MANGROVE_EXTENT",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake / SDG 6.6.1",
        "policy_ref": "AFOLU Blue Carbon — Annual global coastal mangrove forest extent and change",
        "query": "Global coastal mangrove forest habitat extent and multitemporal mangrove loss and gain monitoring for blue carbon accounting",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "global_mangrove_watch", "regex": r"\bglobal\s+mangrove\s+watch\b"},
        ],
    },
    {
        "query_id": "Q26_GST_GLOBAL_LAND_COVER_MULTIDECADAL",
        "policy_framework": "UNFCCC Paris Agreement Global Stocktake / UNCCD GPG",
        "policy_ref": "AFOLU Land Use — Multi-decadal consistent annual global land cover and land use change classification",
        "query": "Consistent multi-decadal annual global land cover classification and historical land use change maps for greenhouse gas land accounting",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "esa_cci_lc", "regex": r"land\s+cover\s+classification\s+gridded\s+maps\s+from\s+1992|\besa\s+cci.*land\s+cover\b|\besa\s+land\s+cover\s+climate\s+change\s+initiative\b|\besa\s+climate\s+change\s+initiative\s+land\s+cover\b"},
            {"family_id": "clms_lc100", "regex": r"\bland\s+cover\s+2015[-–]2019\b.*100\s*m"},
            {"family_id": "hilda_plus", "regex": r"(?<![a-z0-9])hilda\+?(?![a-z0-9])"},
            {"family_id": "esa_worldcover", "regex": r"(?<![a-z0-9])esa[_\s]worldcover(?![a-z0-9])"},
        ],
    },

    # ==================== D. UN SDG INDICATORS (CEOS SDG + UN-Habitat + UNCCD) ====================
    {
        "query_id": "Q27_SDG661_SURFACE_WATER_EXTENT",
        "policy_framework": "UN SDG 6.6.1 (CEOS SDG Support Sheet)",
        "policy_ref": "SDG 6.6.1 — Change in the extent of permanent and seasonal inland surface water ecosystems",
        "query": "Global 30 meter multitemporal inland surface water occurrence, seasonality, recurrence, and dynamic open surface water extent",
        "filters": {"geo_scope": "global", "max_grid_m": 300.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "jrc_gswe", "regex": r"\bglobal\s+surface\s+water\s+explorer\b|\bjrc\s+global\s+surface\s+water\b"},
            {"family_id": "opera_dswx", "regex": r"\bdynamic\s+surface\s+water\s+extent\b|\bopera_l3_dswx\b"},
            {"family_id": "clms_water_bodies", "regex": r"\bwater\s+bodies\s+20\d{2}"},
        ],
    },
    {
        "query_id": "Q28_SDG1131_URBAN_BUILTUP_EXTENT",
        "policy_framework": "UN SDG 11.3.1 (CEOS SDG Sheet & UN-Habitat EO Toolkit)",
        "policy_ref": "SDG 11.3.1 — Global multitemporal built-up human settlement footprint and urban expansion",
        "query": "Global high-resolution multitemporal human settlement built-up surface extent and urban land consumption footprint",
        "filters": {"geo_scope": "global", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "ghsl_built", "regex": r"(?<![a-z0-9])ghs[-_\s]built(?![a-z0-9])|global\s+human\s+settlement.*built[- ]up"},
            {"family_id": "dlr_wsf", "regex": r"\bworld\s+settlement\s+footprint\b"},
            {"family_id": "sedac_hbase_gmis", "regex": r"\bhuman\s+built-up\s+and\s+settlement\s+extent\b|\bman-made\s+impervious\s+surface\b"},
            {
                "family_id": "google_open_buildings",
                "regex": r"\bopen\s+buildings\b",
                "exclude_regex": r"ghs[-_]obat|open\s+buildings\s+attribute",
            },
        ],
    },
    {
        "query_id": "Q29_SDG1131_GRIDDED_POPULATION",
        "policy_framework": "UN SDG 11.3.1 (CEOS SDG Sheet & UN-Habitat EO Toolkit)",
        "policy_ref": "SDG 11.3.1 — Global spatially disaggregated gridded human population distribution and density",
        "query": "Global spatially explicit gridded human population count and population density distribution for urban growth rate estimation",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "ghsl_pop", "regex": r"(?<![a-z0-9])ghs[-_\s]pop(?![a-z0-9])|global\s+human\s+settlement.*population"},
            {"family_id": "worldpop", "regex": r"\bworldpop\b.*population"},
            {"family_id": "sedac_gpw", "regex": r"\bgridded\s+population\s+of\s+the\s+world\b"},
        ],
    },
    {
        "query_id": "Q30_SDG1411A_COASTAL_CHLOROPHYLL",
        "policy_framework": "UN SDG 14.1.1a (CEOS SDG Support Sheet)",
        "policy_ref": "SDG 14.1.1a — Index of coastal eutrophication via satellite ocean color chlorophyll-a",
        "query": "Global satellite ocean color chlorophyll-a concentration for monitoring marine phytoplankton blooms and coastal eutrophication",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "esa_oc_cci", "regex": r"\bocean[_\s]colou?r[_\s]cci\b|esa\s+ocean\s+colou?r\s+climate\s+change"},
            {"family_id": "viirs_chl_l3", "regex": r"\bviirs\b.*\blevel[- ]?3\b.*\bchlorophyll\b"},
            {"family_id": "olci_chl_l3", "regex": r"\bolci\b.*\blevel[- ]?3\b.*\bchlorophyll\b"},
        ],
    },
    {
        "query_id": "Q31_SDG1531_LAND_PRODUCTIVITY",
        "policy_framework": "UN SDG 15.3.1 (UNCCD GPG v2 & CEOS SDG Sheet)",
        "policy_ref": "SDG 15.3.1 Sub-indicator 2 — Land productivity dynamics and multi-year vegetation greenness NDVI trends",
        "query": "Global multi-year land productivity dynamics and normalized difference vegetation index time series for land degradation neutrality",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "jrc_lpd", "regex": r"\bland\s+productivity\s+dynamics\b"},
            {"family_id": "modis_mod13", "regex": r"\b(?:mod|myd)13[qa][12]\b"},
            {
                "family_id": "clms_ndvi",
                "regex": r"\bnormali[sz]ed\s+difference\s+vegetation\s+index\b.*(?:global|long\s+term)",
                "exclude_regex": r"\bavhrr\b|\bnoaa\b|\bjaxa\b|\bwapor\b|avh13|16_nasa_cmr|26_fao_unep",
            },
            {"family_id": "gimms_ndvi", "regex": r"(?<![a-z0-9])gimms(?![a-z0-9]).*(?:greenness|ndvi)|(?:greenness|ndvi).*(?<![a-z0-9])gimms(?![a-z0-9])"},
        ],
    },
    {
        "query_id": "Q32_SDG1531_SOIL_ORGANIC_CARBON",
        "policy_framework": "UN SDG 15.3.1 (UNCCD GPG v2 & KCEO Biodiversity)",
        "policy_ref": "SDG 15.3.1 Sub-indicator 3 — Global topsoil organic carbon stock (0-30 cm)",
        "query": "Global gridded topsoil organic carbon stocks and soil organic carbon content at 0 to 30 cm depth for land degradation reporting",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "isric_soilgrids", "regex": r"\bsoilgrids\b.*(?:organic\s+carbon|(?<![a-z0-9])ocs(?![a-z0-9])|(?<![a-z0-9])soc(?![a-z0-9]))"},
            {"family_id": "fao_gsoc", "regex": r"\bgsoc(?:seq)?\b|global\s+soil\s+organic\s+carbon"},
            {"family_id": "hwsd", "regex": r"\bharmonized\s+world\s+soil\s+database\b|\bhwsd\b"},
        ],
    },
    {
        "query_id": "Q33_SDG1162_URBAN_PM25_NO2",
        "policy_framework": "UN SDG 11.6.2 (UN-Habitat EO Toolkit & NASA SNWG)",
        "policy_ref": "SDG 11.6.2 — Ambient urban air quality fine particulate matter PM2.5 and tropospheric nitrogen dioxide",
        "query": "Satellite-derived ambient fine particulate matter PM2.5 concentrations and gridded tropospheric nitrogen dioxide columns for urban air pollution exposure",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "sedac_pm25", "regex": r"\bglobal\s+annual\s+pm2\.?5\b"},
            {"family_id": "omi_no2", "regex": r"\bomi/aura\s+tropospheric\s+nitrogen\s+dioxide\b|\bomi.*tropospheric.*no2"},
            {"family_id": "tempo_no2", "regex": r"\btempo_no2|\btempo\b.*(?:no2|nitrogen\s+dioxide)"},
        ],
    },

    # ==================== E. CBD GBF INDICATOR A.2 / SEEA (CEOS EETT) ====================
    {
        "query_id": "Q34_GBF_CANOPY_HEIGHT",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Ecosystem Structural Characteristic — Global vegetation canopy height from spaceborne lidar",
        "query": "Global gridded forest and vegetation canopy height structure derived from spaceborne laser altimetry",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "gedi_icesat2_height", "regex": r"global\s+(?:vegetation|forest\s+canopy)\s+height|canopy\s+height.*(?:gedi|icesat)"},
        ],
    },
    {
        "query_id": "Q35_GBF_GLOBAL_ELEVATION_DEM",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Abiotic Ecosystem Characteristic — Global 30 m digital elevation model and surface topography",
        "query": "Global 30 meter digital elevation model and terrain surface topography derived from radar and stereo satellite imagery",
        "filters": {"geo_scope": "global", "max_grid_m": 90.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "copernicus_dem", "regex": r"\bcopernicus\s+dem\b|(?<![a-z0-9])(?:cop-?dem|glo-?30|dsm_glo30)(?![a-z0-9])"},
            {"family_id": "nasadem_srtm", "regex": r"(?<![a-z0-9])nasadem(?![a-z0-9])|(?<![a-z0-9])srtm(?![a-z0-9]).*(?:dem|elevation|30m|1\s+arc)"},
            {"family_id": "aster_gdem", "regex": r"\baster\s+global\s+digital\s+elevation\b|(?<![a-z0-9])astgtm(?![a-z0-9])"},
            {"family_id": "alos_aw3d30", "regex": r"(?<![a-z0-9])aw3d30(?![a-z0-9])"},
            {"family_id": "tandemx_dem", "regex": r"\btandem-?x\b.*digital\s+elevation"},
        ],
    },
    {
        "query_id": "Q36_GBF_LAI_FAPAR",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Biophysical Canopy Density — Global Leaf Area Index (LAI) and Fraction of Absorbed PAR (fAPAR)",
        "query": "Global gridded terrestrial vegetation Leaf Area Index and Fraction of Absorbed Photosynthetically Active Radiation time series",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "modis_mod15", "regex": r"\b(?:mod|myd|mcd)15a2h?\b"},
            {
                "family_id": "clms_lai_fapar",
                "regex": r"\b(?:leaf\s+area\s+index|fraction\s+of\s+absorbed\s+photosynthetically\s+active\s+radiation)\b.*global",
                "exclude_regex": r"\b(?:mod|myd|mcd|vnp|vj1|vj2)15|modis|viirs|avhrr|muses|16_nasa_cmr|34_fao_geospatial|26_fao_unep",
            },
        ],
    },
    {
        "query_id": "Q37_GBF_TERRESTRIAL_GPP_NPP",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Ecosystem Function — Global terrestrial Gross Primary Productivity (GPP) and Net Primary Productivity (NPP)",
        "query": "Global satellite-derived terrestrial ecosystem Gross Primary Productivity and annual Net Primary Productivity of vegetation",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "modis_mod17", "regex": r"\b(?:mod|myd)17a[23]h?\b|gross\s+primary\s+productivity\s+8-day\s+l4"},
        ],
    },
    {
        "query_id": "Q38_GBF_SOLAR_INDUCED_FLUORESCENCE",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Photosynthetic Activity — Global satellite Solar-Induced Chlorophyll Fluorescence (SIF)",
        "query": "Global satellite measurements of terrestrial plant solar-induced chlorophyll fluorescence as a direct proxy for photosynthesis",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "harmonized_sif", "regex": r"\bsolar[- ]induced\s+(?:chlorophyll\s+)?fluorescence\b|\bsif[_\s]"},
        ],
    },
    {
        "query_id": "Q39_GBF_TERRESTRIAL_LEAF_CHLOROPHYLL",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Foliar Biochemical Trait — Global terrestrial vegetation leaf chlorophyll content (NOT ocean chlorophyll)",
        "query": "Global terrestrial plant foliar leaf chlorophyll content retrieved over land vegetation canopies from optical satellite reflectance",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "global_leaf_chl", "regex": r"\bleaf\s+chlorophyll\s+content\b"},
        ],
    },
    {
        "query_id": "Q40_GBF_SNOW_COVER_EXTENT",
        "policy_framework": "CBD Kunming-Montreal GBF Indicator A.2 (CEOS EETT)",
        "policy_ref": "Cryosphere Ecosystem Characteristic — Global snow cover extent and snow water equivalent",
        "query": "Global multi-decadal satellite snow cover extent and snow water equivalent climate data record",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "esa_cci_snow", "regex": r"\besa\s+snow\s+climate\s+change\s+initiative\b|\bsnow[_\s]cci\b"},
            {"family_id": "nsidc_snow_extent", "regex": r"\bease-grid\s+2\.0\s+weekly\s+snow\s+cover\b|\bmod10[ac]"},
        ],
    },

    # ==================== F. USGEO EOA 2023 & NASA SNWG OPERATIONAL REQUIREMENTS ====================
    {
        "query_id": "Q41_EOA_IRRIGATED_CROPLAND",
        "policy_framework": "USGEO EOA 2023 (Agriculture & Forestry)",
        "policy_ref": "Agricultural water use — Irrigated cropland extent and irrigated harvested area mapping",
        "query": "Satellite-based mapping of irrigated agricultural cropland extent and monthly irrigated versus rainfed harvested crop areas",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "lanid_us", "regex": r"\blanid-?us\b|landsat-based\s+irrigation"},
            {"family_id": "usgs_mirad", "regex": r"\bmirad\b|irrigated\s+agriculture\s+dataset\s+for\s+the\s+united\s+states"},
            {"family_id": "mirca2000", "regex": r"\bmirca2000\b|\bmirca-?os\b"},
        ],
    },
    {
        "query_id": "Q42_EOA_SOIL_MOISTURE_EVAPORATION",
        "policy_framework": "USGEO EOA 2023 (Climate & Hydrology)",
        "policy_ref": "Terrestrial water cycle — Global surface and root-zone soil moisture and land evapotranspiration",
        "query": "Global satellite-derived surface and root-zone soil moisture, soil water index, and terrestrial land evapotranspiration",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "gleam_evap_sm", "regex": r"\bgleam\d*(?:[-_\s]|$).*(?:evaporation|soil\s+moisture)"},
            {"family_id": "clms_swi", "regex": r"\bsoil\s+water\s+index\b"},
            {"family_id": "noaa_smops", "regex": r"\bsmops\b|soil\s+moisture\s+products\s+system"},
            {"family_id": "amsr2_smap_sm", "regex": r"\b(?:amsr2|smap)\b.*surface\s+soil\s+moisture|\bsmap.*l3.*soil\s+moisture"},
        ],
    },
    {
        "query_id": "Q43_EOA_SEA_SURFACE_TEMPERATURE",
        "policy_framework": "USGEO EOA 2023 (Climate)",
        "policy_ref": "Ocean Climate ECV — High-resolution global foundation sea surface temperature analysis",
        "query": "Global high-resolution multi-scale Level 4 foundation sea surface temperature analysis from merged microwave and infrared satellites",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "ghrsst_mur_sst", "regex": r"\bmur\b.*sea\s+surface\s+temperature|\bghrsst\s+level\s+4\s+mur\b"},
            {"family_id": "esa_cci_sst", "regex": r"\bsst[_\s]cci\b|sea\s+surface\s+temperature.*climate\s+change\s+initiative"},
        ],
    },
    {
        "query_id": "Q44_SNWG_SEA_SURFACE_SALINITY",
        "policy_framework": "NASA SNWG / GCOS Ocean ECV",
        "policy_ref": "Ocean salinity — Global satellite L-band microwave sea surface salinity mapped product",
        "query": "Global Level 3 gridded ocean sea surface salinity derived from L-band microwave satellite radiometry",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "smap_sss", "regex": r"\bsmap\b.*sea\s+surface\s+salinity"},
            {"family_id": "smos_sss", "regex": r"\bsmos\b.*sea\s+surface\s+salinity"},
        ],
    },
    {
        "query_id": "Q45_SNWG_SURFACE_DISPLACEMENT_INSAR",
        "policy_framework": "NASA SNWG / Copernicus Land Ground Motion",
        "policy_ref": "Geohazards & Subsidence — Satellite radar interferometry ground surface displacement and deformation",
        "query": "Satellite synthetic aperture radar interferometry measurements of ground surface displacement, land subsidence, and tectonic deformation",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "opera_disp_s1", "regex": r"\bopera\b.*surface\s+displacement|\bopera_l3_disp"},
            {"family_id": "clms_egms", "regex": r"\beuropean\s+ground\s+motion\s+service\b|\begms\b"},
        ],
    },
]

# Anti-leakage verification list: none of these concrete product brand names / codes may appear in query text!
FORBIDDEN_QUERY_LEAKS = [
    "corine", "clc+", "ghsl", "ghs-built", "ghs-pop", "wsf", "mod13", "mod15", "mod17",
    "mcd64", "gedi", "icesat", "oco-2", "oco2", "gosat", "tm5-4dvar", "odiac", "vulcan",
    "soilgrids", "gsoc", "hwsd", "euseamap", "worldcover", "worldcereal", "hansen",
    "gswe", "hr-vpp", "rlie", "cams", "efas", "effis", "gfm", "opera", "tempo",
    "lanid", "mirad", "mirca", "gleam", "smops", "ghrsst", "mur", "smap", "smos",
    "nasadem", "srtm", "aster", "aw3d30", "tandem-x", "hilda", "worldpop", "egms",
    "eucropmap", "black marble", "urban atlas", "natura 2000", "n2k", "cmems", "clms",
]


def verify_no_query_leakage(specs: list[dict[str, Any]]) -> None:
    leaks = []
    for sp in specs:
        ql = sp["query"].lower()
        for tok in FORBIDDEN_QUERY_LEAKS:
            if re.search(rf"\b{re.escape(tok)}\b", ql):
                leaks.append((sp["query_id"], tok, sp["query"]))
    if leaks:
        raise ValueError(f"Product-name leakage detected in benchmark queries: {leaks}")


def main() -> None:
    print("=" * 84)
    print("02. BUILD POLICY-REQUIREMENT RETRIEVAL BENCHMARK & CHALLENGING 3,500-RECORD SUBSET")
    print("=" * 84)

    verify_no_query_leakage(BENCHMARK_SPECS)
    print(f"[OK] Anti-leakage check passed across all {len(BENCHMARK_SPECS)} queries (0 product names leaked).")

    df = pd.read_parquet(CANONICAL_PARQUET)
    n_total = len(df)
    print(f"Loaded {n_total:,} canonical records from {CANONICAL_PARQUET.name}")

    # Combined searchable label + active-record filter (consistent with SearchFilters.exclude_superseded & 09)
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

    resolved_queries: list[dict[str, Any]] = []
    gt_indices_all: set[int] = set()
    missing_fams: list[tuple[str, str]] = []

    for sp in BENCHMARK_SPECS:
        q_fams = []
        q_all_rids: set[str] = set()
        for fam in sp["expected_families"]:
            pat = re.compile(fam["regex"], re.IGNORECASE)
            hit_mask = active_mask & combined.str.contains(pat, regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                ex_pat = re.compile(fam["exclude_regex"], re.IGNORECASE)
                hit_mask = hit_mask & ~combined.str.contains(ex_pat, regex=True).to_numpy(dtype=bool)
            hit_mask = filter_benchmark_family_mask(df, hit_mask, sp.get("filters"), query_text=sp["query"])
            idxs = np.flatnonzero(hit_mask)
            if len(idxs) == 0:
                missing_fams.append((sp["query_id"], fam["family_id"]))
                continue
            # Cap extremely wide multi-epoch families at 40 highest-authority canonical records for subset inclusion
            # while STILL marking ALL matching records in the subset as relevant for that family!
            rids = df.iloc[idxs]["record_id"].tolist()
            # Prioritize diverse sources + highest authority for subset inclusion if a family has >25 records
            if len(idxs) > 25:
                sub = df.iloc[idxs].sort_values(["source_authority_rank", "record_id"]).head(25)
                gt_indices_all.update(sub.index.tolist())
            else:
                gt_indices_all.update(idxs.tolist())

            q_all_rids.update(rids)
            q_fams.append({
                "family_id": fam["family_id"],
                "full_catalogue_matches": int(len(idxs)),
                "sample_titles": df.iloc[idxs[:3]]["title"].tolist(),
                "record_ids": rids,
            })

        resolved_queries.append({
            "query_id": sp["query_id"],
            "policy_framework": sp["policy_framework"],
            "policy_ref": sp["policy_ref"],
            "query": sp["query"],
            "filters": sp.get("filters", {}),
            "n_expected_families": len(q_fams),
            "expected_families": q_fams,
            "total_relevant_in_full_catalogue": len(q_all_rids),
        })

    if missing_fams:
        print(f"[WARN] Some expected families matched 0 records: {missing_fams}")
    else:
        print("[OK] Every single expected policy product family resolved cleanly in canonical catalogue!")

    total_fams = sum(q["n_expected_families"] for q in resolved_queries)
    print(f"Total benchmark queries: {len(resolved_queries)} | Total verified product families: {total_fams}")
    print(f"Unique ground-truth canonical records earmarked for subset: {len(gt_indices_all):,}")

    # -----------------------------------------------------------------------
    # Mine HARD confusable distractors (makes the subset a genuine stress test!)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(2026)
    hard_patterns = [
        # Ocean chlorophyll vs leaf chlorophyll; LST vs SST; regional US/AU vs Europe;
        # raw/inventory/in-situ; long Zenodo records; soil/forest/urban/water/GHG lookalikes
        r"\bchlorophyll\b",
        r"\bland\s+surface\s+temperature\b|\bsea\s+surface\s+temperature\b",
        r"\bsoil\s+moisture\b|\bsoil\s+organic\s+carbon\b|\bsoil\s+sealing\b",
        r"\bforest\b|\bbiomass\b|\bcanopy\b|\bphenology\b",
        r"\bmethane\b|\bcarbon\s+dioxide\b|\bnitrogen\s+dioxide\b|\bpm2\.?5\b",
        r"\bflood\b|\bwater\s+quality\b|\bwetland\b|\bmangrove\b|\bsea\s+level\b",
        r"\burban\b|\bsettlement\b|\bimpervious\b|\bpopulation\b|\bheat\s+island\b",
        r"\bland\s+cover\b|\bcropland\b|\birrigat|\belevation\b|\bsnow\s+cover\b|\bsalinity\b",
    ]
    hard_pool: set[int] = set()
    titles_lower = df["title"].fillna("").str.lower()
    for pat in hard_patterns:
        cand = np.flatnonzero(titles_lower.str.contains(pat, regex=True).to_numpy())
        cand = np.setdiff1d(cand, np.array(sorted(gt_indices_all), dtype=np.int64))
        if len(cand) > 0:
            take = min(140, len(cand))
            hard_pool.update(rng.choice(cand, size=take, replace=False).tolist())

    # Also include 150 longest-description records (stress-testing truncation/chunking & boilerplate!)
    desc_lens = df["description"].fillna("").str.len().to_numpy()
    longest_idx = np.argsort(-desc_lens)[:250]
    longest_cand = [int(i) for i in longest_idx if int(i) not in gt_indices_all][:150]
    hard_pool.update(longest_cand)

    print(f"Hard confusable & long-text distractor pool: {len(hard_pool):,} records")

    # Combine GT + hard distractors + stratified random background up to TARGET_SUBSET_N = 3,500
    TARGET_SUBSET_N = 3500
    chosen_set = set(gt_indices_all) | set(hard_pool)

    # Cap hard_pool if GT + hard_pool already exceeds 2,400 so we always leave >=1,100 stratified random docs
    if len(chosen_set) > 2300:
        hard_keep = rng.choice(
            sorted(hard_pool - gt_indices_all),
            size=max(500, 2300 - len(gt_indices_all)),
            replace=False,
        ).tolist()
        chosen_set = set(gt_indices_all) | set(hard_keep)

    remaining_needed = max(0, TARGET_SUBSET_N - len(chosen_set))
    if remaining_needed > 0:
        counts = df["source_catalogue"].value_counts()
        bg_chosen: list[int] = []
        for src, cnt in counts.items():
            idxs = np.flatnonzero((df["source_catalogue"] == src).to_numpy())
            avail = np.array([int(i) for i in idxs if int(i) not in chosen_set], dtype=np.int64)
            if len(avail) == 0:
                continue
            k = max(1, int(round(remaining_needed * cnt / n_total)))
            k = min(k, len(avail))
            bg_chosen.extend(rng.choice(avail, size=k, replace=False).tolist())
        chosen_set.update(bg_chosen)

    # Trim or pad cleanly to exact TARGET_SUBSET_N without ever dropping a GT record!
    if len(chosen_set) > TARGET_SUBSET_N:
        non_gt = sorted(chosen_set - gt_indices_all)
        keep_non_gt = rng.choice(non_gt, size=TARGET_SUBSET_N - len(gt_indices_all), replace=False).tolist()
        chosen_set = set(gt_indices_all) | set(keep_non_gt)
    elif len(chosen_set) < TARGET_SUBSET_N:
        pool_rest = np.setdiff1d(np.arange(n_total), np.array(sorted(chosen_set), dtype=np.int64))
        extra = rng.choice(pool_rest, size=TARGET_SUBSET_N - len(chosen_set), replace=False).tolist()
        chosen_set.update(extra)

    subset_indices = np.array(sorted(chosen_set), dtype=np.int64)
    df_sub = df.iloc[subset_indices].reset_index(drop=True)
    sub_rids = set(df_sub["record_id"].tolist())

    # Update each query's expected families with exact subset AND full-catalogue record_ids
    # (Keeping full `record_ids` inside the SHA-256-locked JSON guarantees 07/10/11 read locked JSON only.)
    for q in resolved_queries:
        for fam in q["expected_families"]:
            in_sub = [rid for rid in fam["record_ids"] if rid in sub_rids]
            fam["subset_record_ids"] = in_sub
            fam["subset_matches"] = len(in_sub)
        q["total_relevant_in_subset"] = len(
            {rid for fam in q["expected_families"] for rid in fam["subset_record_ids"]}
        )

    # Save subset parquet & benchmark JSON
    sub_pq = ARTIFACTS_DIR / "subset_catalogue.parquet"
    df_sub.to_parquet(sub_pq, index=False)

    bench_payload = {
        "benchmark_name": "EO_L3_L4_Policy_Requirement_Retrieval_Benchmark_v1",
        "n_queries": len(resolved_queries),
        "n_total_expected_families": total_fams,
        "subset_size": int(len(df_sub)),
        "subset_sources_covered": int(df_sub["source_catalogue"].nunique()),
        "subset_gt_records": int(len(gt_indices_all)),
        "subset_null_rates": {
            "adjudicated_output_grid_meters_null_pct": round(100.0 * float(df_sub["adjudicated_output_grid_meters"].isna().mean()), 1),
            "processing_level_null_pct": round(100.0 * float(df_sub["processing_level"].isna().mean()), 1),
            "temporal_cadence_class_null_pct": round(100.0 * float(df_sub["temporal_cadence_class"].isna().mean()), 1),
            "geographic_coverage_class_null_pct": round(100.0 * float(df_sub["geographic_coverage_class"].isna().mean()), 1),
        },
        "queries": resolved_queries,
    }
    bench_json = ARTIFACTS_DIR / "policy_retrieval_benchmark.json"
    bench_json.write_text(json.dumps(bench_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"SAVED SUBSET   : {sub_pq} ({len(df_sub):,} records across {df_sub['source_catalogue'].nunique()}/34 sources)")
    print(f"SAVED BENCHMARK: {bench_json} ({len(resolved_queries)} queries, {total_fams} verified families)")
    print("Subset null rates:", bench_payload["subset_null_rates"])
    # Sanity check: ensure every query has >=1 relevant record in subset
    zero_q = [q["query_id"] for q in resolved_queries if q["total_relevant_in_subset"] == 0]
    if zero_q:
        raise RuntimeError(f"Queries with 0 relevant subset records: {zero_q}")
    print("[OK] All 45 queries verified with >=1 relevant product family in subset!")
    print("=" * 84)


if __name__ == "__main__":
    main()
