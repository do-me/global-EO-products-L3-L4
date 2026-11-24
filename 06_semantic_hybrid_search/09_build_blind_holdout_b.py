# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
# ]
# ///
"""09. Build Prospective Single-Shot Blind Holdout B (60 Disjoint External EO Queries).

Standalone blind evaluation benchmark constructed purely from unseen active products
inside `03_harmonization/data/eo_l3_l4_canonical_products.parquet` with ZERO overlap
with either `Dev-45` (`02_build_policy_benchmark.py`) or `Holdout A`
(`08_build_holdout_a_external.py`):

  1. 60 thematically distinct queries spanning 10 Earth Observation domains
     (6 queries per domain; G = G_eff = 60 independent evaluation units).
  2. Strict cross-suite disjointness asserted programmatically:
       Holdout_B_rids ∩ (Dev45_rids ∪ Holdout_A_rids) == ∅
  3. Strict within-suite disjointness asserted programmatically:
       0 overlapping record_ids within any query or across any pair of Holdout B queries.
  4. Strict anti-leakage linter: 0 product brand names, mission/sensor codes, or
     provider shortnames permitted in query text.
  5. Filter reachability verified upfront against `SearchFilters(..., null_safe=True)`
     without running any retrieval model, preserving true prospective single-shot blindness.
"""

from __future__ import annotations

import ast
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
from eo_search_lib import SearchFilters, apply_filters_mask, filter_benchmark_family_mask  # noqa: E402


HOLDOUT_B_SPECS: list[dict[str, Any]] = [
    # =========================================================================
    # DOMAIN 1: Forests, Biomass, Canopy Structure & SAR Backscatter (HB01-HB06)
    # =========================================================================
    {
        "query_id": "HB01_BOREAL_ABOVEGROUND_BIOMASS",
        "domain": "Forests & Biomass",
        "source_framework": "CEOS AFOLU / High-Latitude Carbon Monitoring",
        "query": "Circumpolar boreal forest aboveground biomass density mapped at 30 meter resolution from spaceborne laser altimetry and optical imagery",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "boreal_agb_icesat2", "regex": r"circumpolar\s+boreal.*aboveground\s+biomass|boreal\s+aboveground\s+biomass"},
        ],
    },
    {
        "query_id": "HB02_LBAND_SAR_FOREST_MOSAIC",
        "domain": "Forests & Biomass",
        "source_framework": "GFOI / Global Radar Forest Monitoring",
        "query": "Global annual L-band synthetic aperture radar backscatter mosaics and radar-derived forest and non-forest classification maps",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "alos_palsar_fnf_mosaic", "regex": r"alos.*palsar.*(?:mosaic|forest\s*/?\s*non-?forest|fnf|yearly)|palsar-2.*(?:mosaic|forest\s*/?\s*non-?forest|fnf)"},
        ],
    },
    {
        "query_id": "HB03_CBAND_SAR_GLOBAL_BACKSCATTER",
        "domain": "Forests & Biomass",
        "source_framework": "CEOS SAR Cal/Val & Land Surface Physics",
        "query": "Global normalized C-band synthetic aperture radar land surface backscatter reference model derived from multi-year radar observations",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "s1_global_backscatter", "regex": r"sentinel-1\s+global\s+backscatter|global\s+sentinel-1.*backscatter"},
        ],
    },
    {
        "query_id": "HB04_AMAZON_DEFORESTATION_MONITORING",
        "domain": "Forests & Biomass",
        "source_framework": "UNFCCC REDD+ / Regional Tropical Forest Monitoring",
        "query": "Annual satellite monitoring of clear-cut tropical deforestation across the Brazilian Legal Amazon and tropical forest biomes",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "inpe_prodes", "regex": r"\bprodes\b"},
        ],
    },
    {
        "query_id": "HB05_OIL_PALM_PLANTATION_EXTENT",
        "domain": "Forests & Biomass",
        "source_framework": "EU Deforestation Regulation (Commodity Monitoring)",
        "query": "High-resolution satellite maps distinguishing industrial and smallholder closed-canopy oil palm plantations in Southeast Asia and the tropics",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "oil_palm_plantations", "regex": r"oil\s+palm\s+plantations?|industrial\s+(?:and|&)\s+smallholder.*oil\s+palm|smallholder\s+(?:and|&)\s+industrial.*oil\s+palm"},
        ],
    },
    {
        "query_id": "HB06_FOREST_TREE_SPECIES_COMPOSITION",
        "domain": "Forests & Biomass",
        "source_framework": "National Forest Inventories / Biodiversity Mapping",
        "query": "Multi-decadal satellite-derived leading forest tree species composition and species distribution maps",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "tree_species_maps", "regex": r"(?:long\s+term|leading)\s+tree\s+species"},
        ],
    },

    # =========================================================================
    # DOMAIN 2: Urban Morphology, Settlements & Building Footprints (HB07-HB12)
    # =========================================================================
    {
        "query_id": "HB07_GLOBAL_URBAN_CENTRE_DATABASE",
        "domain": "Urban & Settlements",
        "source_framework": "UN-Habitat / SDG 11 Urbanization Harmonization",
        "query": "Global multitemporal spatial database of delineated urban centres with multidimensional demographic, environmental, and socio-economic attributes",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "ghs_ucdb", "regex": r"\bghs[-_ ]ucdb\b|urban\s+centre\s+database", "exclude_regex": r"\bghs[-_ ]fua\b|functional\s+urban"},
        ],
    },
    {
        "query_id": "HB08_FUNCTIONAL_URBAN_AREAS",
        "domain": "Urban & Settlements",
        "source_framework": "OECD / EC Urban-Rural Typology",
        "query": "Global delineation of functional urban areas combining high-density urban centres with their surrounding commuting zones",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "ghs_fua", "regex": r"\bghs[-_ ]fua\b|functional\s+urban\s+areas?"},
        ],
    },
    {
        "query_id": "HB09_EUROPEAN_SETTLEMENT_MAP",
        "domain": "Urban & Settlements",
        "source_framework": "EU Territorial Cohesion / Urban Monitoring",
        "query": "Very high resolution European human settlement and built-up area spatial layer derived from satellite optical imagery",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "esm_europe", "regex": r"european\s+settlement\s+map|\besm\s+r20\d\d\b", "exclude_regex": r"building\s+footprints|\bdbsm\b"},
        ],
    },
    {
        "query_id": "HB10_GLOBAL_ARTIFICIAL_IMPERVIOUS_AREAS",
        "domain": "Urban & Settlements",
        "source_framework": "Global Urban Expansion & Hydrological Sealing",
        "query": "Annual 30 meter global artificial impervious surface area maps tracking multi-decadal urbanization and soil sealing",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "gaia_impervious", "regex": r"global\s+artificial\s+impervious\s+areas?|\bgaia\b.*impervious"},
        ],
    },
    {
        "query_id": "HB11_DEMOGRAPHIC_SETTLEMENT_EXTENTS",
        "domain": "Urban & Settlements",
        "source_framework": "Humanitarian Demography & Sub-National Planning",
        "query": "High-resolution gridded human settlement footprints and demographic population settlement layers for vulnerable sub-populations",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "hrsl_settlement", "regex": r"high\s+resolution\s+settlement\s+layer|\bhrsl\b"},
            {"family_id": "grid3_settlement", "regex": r"\bgrid3\b.*settlement"},
        ],
    },
    {
        "query_id": "HB12_INDIVIDUAL_BUILDING_FOOTPRINTS",
        "domain": "Urban & Settlements",
        "source_framework": "Urban Exposure & Building Stock Mapping",
        "query": "Continental and global individual building footprint polygon datasets extracted from very high resolution satellite imagery",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "building_footprints", "regex": r"microsoft\s+building\s+footprints|openbuildingmap.*building\s+footprints|individual\s+building\s+footprints"},
        ],
    },

    # =========================================================================
    # DOMAIN 3: Land Cover, Rangelands & Specialized Ecosystems (HB13-HB18)
    # =========================================================================
    {
        "query_id": "HB13_GLOBAL_HIGHRES_MULTICLASS_LC",
        "domain": "Land Cover & Ecosystems",
        "source_framework": "GEO Global Land Cover Harmonization",
        "query": "High-resolution 10 meter and 30 meter global multi-class land cover and land use classification maps derived from medium-resolution optical satellites",
        "filters": {"geo_scope": "global", "max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "globeland30", "regex": r"globeland\s*30"},
            {"family_id": "esri_io_lulc", "regex": r"\bio[-_]lulc\b|esri\s+10[- ]*meter\s+land\s+cover|impact\s+observatory.*land\s*cover"},
            {"family_id": "from_glc10", "regex": r"from[-_ ]?glc10|global\s+land\s+cover\s+10\b"},
            {"family_id": "glc_fcs30", "regex": r"glc[-_ ]?fcs30d?\b", "exclude_regex": r"city\s+and\s+town|gctb|impervious|mediterranean"},
        ],
    },
    {
        "query_id": "HB14_NORTH_AMERICA_NATIONAL_LC",
        "domain": "Land Cover & Ecosystems",
        "source_framework": "North American Land Monitoring",
        "query": "30 meter national and continental land cover and land cover change databases covering the United States, Canada, and North America",
        "filters": {"max_grid_m": 100.0, "strict_eo_only": True},
        "expected_families": [
            {"family_id": "nlcd_us", "regex": r"\bnlcd\b.*land\s+cover|national\s+land\s+cover\s+database", "exclude_regex": r"impervious|tree\s+canopy|shrub"},
            {"family_id": "nalcms_na", "regex": r"land\s+cover\s+of\s+north\s+america|\bnalcms\b"},
            {"family_id": "canada_lc", "regex": r"land\s+cover\s+of\s+canada"},
        ],
    },
    {
        "query_id": "HB15_GLOBAL_PASTURE_GRASSLAND_EXTENT",
        "domain": "Land Cover & Ecosystems",
        "source_framework": "FAO Rangelands & Grassland Productivity Monitoring",
        "query": "Global 30 meter annual grassland extent, managed pasture maps, and grassland gross primary productivity time series",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "global_pasture_watch", "regex": r"global\s+pasture\s+watch"},
        ],
    },
    {
        "query_id": "HB16_NORTHERN_PEATLAND_MAPPING",
        "domain": "Land Cover & Ecosystems",
        "source_framework": "Global Peatlands Initiative / Soil Carbon Protection",
        "query": "Regional satellite and geospatial maps of boreal and temperate peatland distribution and peatland land cover types",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "peatland_maps", "regex": r"peatland", "exclude_regex": r"methane|ch4|co2|flux|canada1water"},
        ],
    },
    {
        "query_id": "HB17_COASTAL_TIDAL_SALT_MARSH",
        "domain": "Land Cover & Ecosystems",
        "source_framework": "Ramsar Coastal Wetlands & Estuarine Habitat Monitoring",
        "query": "Satellite mapping of coastal tidal salt marsh extent, historical salt marsh change, relative marsh elevation, and marsh vegetation biomass",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "tidal_salt_marsh", "regex": r"tidal\s+marsh|salt\s*marsh", "exclude_regex": r"intertidal\s+change|blue\s+carbon|carbon"},
        ],
    },
    {
        "query_id": "HB18_SEAGRASS_AND_BLUE_CARBON",
        "domain": "Land Cover & Ecosystems",
        "source_framework": "CBD Coastal Blue Carbon & Marine Habitat Assessment",
        "query": "High-resolution coastal seagrass meadow extent maps and tidal wetland blue carbon organic carbon storage maps",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "seagrass_extent", "regex": r"seagrass", "exclude_regex": r"soil\s+organic\s+carbon|blue\s+carbon"},
            {"family_id": "blue_carbon_wetlands", "regex": r"blue\s+carbon", "exclude_regex": r"seagrass"},
        ],
    },

    # =========================================================================
    # DOMAIN 4: Greenhouse Gases & Terrestrial/Coastal Carbon Fluxes (HB19-HB24)
    # =========================================================================
    {
        "query_id": "HB19_METHANE_POINT_SOURCE_PLUMES",
        "domain": "Greenhouse Gases & Carbon",
        "source_framework": "Global Methane Pledge / Super-Emitter Detection",
        "query": "Satellite imaging spectrometer detection and mapping of anthropogenic methane point-source emission plume complexes",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "emit_ch4_plumes", "regex": r"\bemit\b.*methane|methane\s+point\s+source\s+plume"},
        ],
    },
    {
        "query_id": "HB20_GLOBAL_WETLAND_METHANE_FLUX",
        "domain": "Greenhouse Gases & Carbon",
        "source_framework": "Global Carbon Project / Methane Budget",
        "query": "Global gridded daily and monthly wetland methane emission fluxes and uncertainty ensembles for terrestrial methane budgets",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "lpj_eosim_ch4", "regex": r"lpj[-_ ]eosim.*methane|lpj[-_ ]eosim.*ch4"},
            {"family_id": "wetcharts_upch4", "regex": r"wetcharts|upch4|boreal\s+arctic\s+wetland\s+methane", "exclude_regex": r"lpj[-_ ]eosim"},
        ],
    },
    {
        "query_id": "HB21_GLOBAL_GHG_REANALYSIS",
        "domain": "Greenhouse Gases & Carbon",
        "source_framework": "WMO Global Greenhouse Gas Watch",
        "query": "Global multi-decadal atmospheric greenhouse gas reanalysis providing gridded carbon dioxide and methane concentrations",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cams_egg4_ghg", "regex": r"\bcams\b.*greenhouse\s+gas\s+reanalysis|\begg4\b"},
        ],
    },
    {
        "query_id": "HB22_ATMOSPHERIC_CARBON_INVERSIONS",
        "domain": "Greenhouse Gases & Carbon",
        "source_framework": "Continental Greenhouse Gas Inverse Modeling",
        "query": "Continental atmospheric inverse modeling vertical profiles and surface carbon dioxide flux footprints constrained by satellite observations",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "carbontracker_lagrange", "regex": r"carbontracker"},
        ],
    },
    {
        "query_id": "HB23_TERRESTRIAL_NEE_FIRE_CARBON_FLUX",
        "domain": "Greenhouse Gases & Carbon",
        "source_framework": "Terrestrial Biosphere Carbon Exchange",
        "query": "Gridded 3-hourly and monthly terrestrial net ecosystem carbon exchange, heterotrophic respiration, and fire fuel carbon emission fluxes",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "casa_gfed_carbon", "regex": r"casa[-_ ]gfed"},
        ],
    },
    {
        "query_id": "HB24_COASTAL_WETLAND_GHG_FLUX",
        "domain": "Greenhouse Gases & Carbon",
        "source_framework": "Coastal Ecosystem Greenhouse Gas Accounting",
        "query": "Gridded daily modeled carbon dioxide and methane exchange fluxes across subtropical and coastal wetlands",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "blueflux_coastal_ghg", "regex": r"\bblueflux\b|coastal\s+wetland.*(?:ghg|co2|methane)"},
        ],
    },

    # =========================================================================
    # DOMAIN 5: Atmospheric Composition, Aerosols, Clouds & Lightning (HB25-HB30)
    # =========================================================================
    {
        "query_id": "HB25_GLOBAL_ATMOSPHERIC_COMPOSITION_REANALYSIS",
        "domain": "Atmosphere & Air Quality",
        "source_framework": "WMO Global Atmosphere Watch / Operational Air Quality",
        "query": "Global gridded atmospheric composition reanalysis and operational chemical forecasts of aerosols and reactive trace gases",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "cams_eac4", "regex": r"\bcams\b.*global\s+reanalysis.*eac4"},
            {"family_id": "cams_atmo_fc", "regex": r"\bcams\b.*global\s+atmospheric\s+composition\s+forecasts?"},
        ],
    },
    {
        "query_id": "HB26_ATMOSPHERIC_AMMONIA_NH3",
        "domain": "Atmosphere & Air Quality",
        "source_framework": "Agricultural Nitrogen Emissions & Air Quality",
        "query": "Near-real-time satellite thermal infrared retrievals of atmospheric ammonia gas column concentrations from polar-orbiting sounders",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "iasi_nh3", "regex": r"\biasi\b.*ammonia|\bnh3\b.*ammonia"},
        ],
    },
    {
        "query_id": "HB27_ACTIVE_LIDAR_CLOUD_AEROSOL_PROFILES",
        "domain": "Atmosphere & Air Quality",
        "source_framework": "GEWEX / Vertical Cloud and Aerosol Climatology",
        "query": "Spaceborne active lidar Level 3 gridded vertical profiles of cloud properties and atmospheric particulate backscatter",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "calipso_lidar_l3", "regex": r"\bcalipso\b.*(?:l3|level\s*3|cloud|backscatter|aerosol|lidar)"},
        ],
    },
    {
        "query_id": "HB28_MULTIANGLE_GLOBAL_AEROSOL",
        "domain": "Atmosphere & Air Quality",
        "source_framework": "Multi-Angle Atmospheric Aerosol Remote Sensing",
        "query": "Global Level 3 gridded atmospheric aerosol microphysical and optical property retrievals from multi-angle satellite spectroradiometry",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "misr_aerosol_l3", "regex": r"\bmisr\b.*aerosol", "exclude_regex": r"aerosol\s+optical\s+depth"},
        ],
    },
    {
        "query_id": "HB29_ATMOSPHERIC_AEROSOL_RADIATION_REANALYSIS",
        "domain": "Atmosphere & Air Quality",
        "source_framework": "Global Atmospheric Reanalysis Diagnostics",
        "query": "Hourly and monthly assimilated atmospheric reanalysis diagnostics of aerosol species tendencies, radiation fluxes, and surface energy balance",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "merra2_diagnostics", "regex": r"\bmerra-?2\b.*(?:aerosol|radiation|surface\s+flux)", "exclude_regex": r"aerosol\s+optical\s+depth"},
        ],
    },
    {
        "query_id": "HB30_GLOBAL_LIGHTNING_CLIMATOLOGY",
        "domain": "Atmosphere & Air Quality",
        "source_framework": "WMO Severe Convection & Atmospheric Electricity",
        "query": "Quasi-global and global 0.1 degree and 0.5 degree satellite gridded climatology of lightning flash rate density and diurnal convective cycles",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "lis_lightning_clim", "regex": r"\blis\b.*lightning|gridded\s+lightning\s+climatology|lis/otd"},
        ],
    },

    # =========================================================================
    # DOMAIN 6: Climate Reanalysis, Downscaling & Bioclimatology (HB31-HB36)
    # =========================================================================
    {
        "query_id": "HB31_GLOBAL_ATMOSPHERIC_REANALYSIS_LEVELS",
        "domain": "Climate & Reanalysis",
        "source_framework": "WMO Global Climate Monitoring",
        "query": "Global hourly atmospheric climate reanalysis from 1940 to present providing meteorological variables on single surface levels and vertical pressure levels",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "era5_single_pressure", "regex": r"\bera5\b\s+(?:hourly|monthly)\s+data\s+on\s+(?:single|pressure)\s+levels", "exclude_regex": r"drought|era5-land"},
        ],
    },
    {
        "query_id": "HB32_GLOBAL_LAND_CLIMATE_REANALYSIS",
        "domain": "Climate & Reanalysis",
        "source_framework": "Terrestrial Hydrometeorology & Land Reanalysis",
        "query": "High-resolution global land-surface hourly and monthly climate reanalysis from 1950 to present for terrestrial energy and water cycle analysis",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "era5_land", "regex": r"\bera5-land\b.*(?:hourly|monthly)", "exclude_regex": r"drought"},
        ],
    },
    {
        "query_id": "HB33_REGIONAL_CLIMATE_DOWNSCALING",
        "domain": "Climate & Reanalysis",
        "source_framework": "WCRP Regional Climate Projections",
        "query": "Dynamically downscaled regional climate model projection ensembles on single levels for regional climate change impact assessment",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "cordex_rcm", "regex": r"\bcordex\b.*regional\s+climate"},
        ],
    },
    {
        "query_id": "HB34_GLOBAL_BIOCLIMATIC_WATER_BALANCE",
        "domain": "Climate & Reanalysis",
        "source_framework": "Ecological Niche & Global Bioclimatology",
        "query": "High-resolution global gridded terrestrial bioclimatic climatologies, monthly historical climate time series, and climatic water balance datasets",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "worldclim", "regex": r"\bworldclim\b"},
            {"family_id": "chelsa", "regex": r"\bchelsa\b"},
            {"family_id": "terraclimate", "regex": r"\bterraclimate\b"},
        ],
    },
    {
        "query_id": "HB35_NORTH_AMERICA_DAILY_1KM_WEATHER",
        "domain": "Climate & Reanalysis",
        "source_framework": "Continental Daily Gridded Surface Weather",
        "query": "Daily and annual 1 kilometer gridded surface weather and temperature summaries interpolated across North America",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "daymet_na", "regex": r"\bdaymet\b"},
        ],
    },
    {
        "query_id": "HB36_GLOBAL_ARIDITY_INDEX",
        "domain": "Climate & Reanalysis",
        "source_framework": "UNCCD Dryland Classification & Desertification",
        "query": "Global high-resolution monthly and annual climatic aridity index raster grids quantifying long-term moisture deficit",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "global_aridity_index", "regex": r"global\s+aridity\s+index"},
        ],
    },

    # =========================================================================
    # DOMAIN 7: Cryosphere & Polar Digital Elevation (HB37-HB42)
    # =========================================================================
    {
        "query_id": "HB37_GLACIER_MASS_CHANGE_BALANCE",
        "domain": "Cryosphere & Polar",
        "source_framework": "WMO / WGMS Global Glacier Mass Loss Assessment",
        "query": "Global gridded multi-decadal mountain glacier mass change and reconciled glacier mass balance intercomparison time series",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "glacier_mass_change", "regex": r"glacier\s+mass\s+(?:change|balance)", "exclude_regex": r"glims|randolph|glaciers?[_\s]cci"},
        ],
    },
    {
        "query_id": "HB38_ICE_SHEET_FLOW_VELOCITY",
        "domain": "Cryosphere & Polar",
        "source_framework": "GCOS Ice Sheet Dynamics & Sea Level Contribution",
        "query": "Satellite interferometric radar and optical feature-tracking surface ice flow velocity maps of the Greenland and Antarctic ice sheets",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "measures_ice_vel", "regex": r"measures.*ice\s+velocity|\bits_live\b.*velocity", "exclude_regex": r"climate\s+change\s+initiative|cci"},
            {"family_id": "esa_cci_ice_vel", "regex": r"greenland_ice_sheet_cci.*velocity|esa\s+greenland\s+ice\s+sheet.*velocity"},
        ],
    },
    {
        "query_id": "HB39_POLAR_SEA_ICE_DRIFT",
        "domain": "Cryosphere & Polar",
        "source_framework": "WMO Polar Sea Ice Dynamics & Navigation",
        "query": "Daily gridded polar sea ice drift and sea ice motion velocity vectors derived from satellite passive microwave and scatterometer tracking",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "sea_ice_drift", "regex": r"sea\s+ice\s+(?:drift|motion)"},
        ],
    },
    {
        "query_id": "HB40_ANTARCTIC_GROUNDING_LINE_ICE_SHELF",
        "domain": "Cryosphere & Polar",
        "source_framework": "Ice Sheet Stability & Grounding Line Monitoring",
        "query": "Satellite differential radar interferometry mapping of the Antarctic ice sheet grounding line and ice shelf dynamics",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "antarctic_grounding_line", "regex": r"antarctic\s+grounding\s+line|antarctic.*ice\s+shel(?:f|ves).*(?:velocity|height|thickness|melt)"},
        ],
    },
    {
        "query_id": "HB41_POLAR_HIGHRES_ARCTIC_DEM",
        "domain": "Cryosphere & Polar",
        "source_framework": "Polar Geospatial Topography",
        "query": "Very high resolution polar stereo digital surface elevation strips and mosaics covering the Arctic and circumpolar regions",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "arctic_dem", "regex": r"\barcticdem\b|digital\s+elevation\s+model\s+for\s+the\s+canadian\s+arctic"},
        ],
    },
    {
        "query_id": "HB42_COARSE_CONTINENTAL_DEM_HYDRO1K",
        "domain": "Cryosphere & Polar",
        "source_framework": "Global 1 km Continental Topographic Baseline",
        "query": "Global 30 arc-second 1 kilometer digital elevation model and hydrologically derived continental terrain derivatives",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "gtopo30_hydro1k", "regex": r"\bgtopo30\b"},
        ],
    },

    # =========================================================================
    # DOMAIN 8: Oceans, Marine Biogeochemistry & Maritime Activity (HB43-HB48)
    # =========================================================================
    {
        "query_id": "HB43_OCEAN_SURFACE_PCO2_AIRSEA_FLUX",
        "domain": "Oceans & Maritime",
        "source_framework": "Global Carbon Budget / Ocean Acidification Monitoring",
        "query": "Global observation-based monthly gridded sea surface carbon dioxide partial pressure and air-sea CO2 exchange flux products",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "socat_ocean_co2", "regex": r"surface\s+ocean\s+co2\s+atlas|\bsocat\b"},
            {"family_id": "gridded_pco2_flux", "regex": r"(?:sea|ocean)\s+surface\s+pco2.*air-?sea\s+co2|air-?sea\s+co2\s+flux.*pco2|fugact?i?ty\s+of\s+co2", "exclude_regex": r"\bsocat\b|surface\s+ocean\s+co2\s+atlas"},
        ],
    },
    {
        "query_id": "HB44_MARINE_HEATWAVES",
        "domain": "Oceans & Maritime",
        "source_framework": "Ocean Extreme Events & Marine Ecosystem Stress",
        "query": "Gridded marine heatwave index and extreme upper-ocean warming event datasets derived from ocean reanalysis and satellite temperature records",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "marine_heatwaves", "regex": r"marine\s+heatwave"},
        ],
    },
    {
        "query_id": "HB45_CORAL_REEF_THERMAL_STRESS_BLEACHING",
        "domain": "Oceans & Maritime",
        "source_framework": "ICRI / Global Coral Reef Bleaching Monitoring",
        "query": "Global 4 kilometer satellite sea surface temperature thermal stress anomalies, degree heating weeks, and coral reef bleaching databases",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "cortad_thermal", "regex": r"\bcortad\b|coral\s+reef\s+temperature\s+anomaly"},
            {"family_id": "coral_bleaching_crw", "regex": r"coral\s+bleaching\s+database|coral\s+reef\s+watch|degree\s+heating\s+week", "exclude_regex": r"\bcortad\b|coral\s+reef\s+temperature\s+anomaly"},
        ],
    },
    {
        "query_id": "HB46_OCEAN_PHYTOPLANKTON_CARBON_PFT",
        "domain": "Oceans & Maritime",
        "source_framework": "Marine Primary Production & Carbon Export",
        "query": "Global Level 4 satellite ocean color gridded estimates of marine phytoplankton carbon biomass and phytoplankton functional groups",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "phytoplankton_carbon_pft", "regex": r"phytoplankton\s+(?:carbon|functional\s+types?|size\s+classes?|groups?)"},
        ],
    },
    {
        "query_id": "HB47_GLOBAL_FISHING_EFFORT_AIS",
        "domain": "Oceans & Maritime",
        "source_framework": "SDG 14.4 / Maritime Fisheries Surveillance",
        "query": "Global satellite vessel automatic identification system tracking of apparent commercial fishing effort and illegal unreported fishing risk",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "gfw_fishing_effort", "regex": r"global\s+fishing\s+watch|apparent\s+fishing\s+effort"},
        ],
    },
    {
        "query_id": "HB48_EUROPEAN_MARITIME_VESSEL_DENSITY",
        "domain": "Oceans & Maritime",
        "source_framework": "EU Maritime Spatial Planning",
        "query": "European marine spatial planning maps of maritime vessel traffic density and shipping route density across European seas",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "emodnet_vessel_density", "regex": r"\bemodnet\b.*(?:vessel\s+density|route\s+density|human\s+activities)"},
        ],
    },

    # =========================================================================
    # DOMAIN 9: Hydrology, Rivers, Dams, Evapotranspiration & Soils (HB49-HB54)
    # =========================================================================
    {
        "query_id": "HB49_ENSEMBLE_SATELLITE_EVAPOTRANSPIRATION",
        "domain": "Hydrology & Soils",
        "source_framework": "Agricultural Irrigation & Basin Water Accounting",
        "query": "Ensemble field-scale monthly satellite evapotranspiration maps combining multiple thermal surface energy balance models for irrigated water management",
        "filters": {"strict_eo_only": True},
        "expected_families": [
            {"family_id": "openet_evapotranspiration", "regex": r"\bopenet\b.*evapotranspiration"},
        ],
    },
    {
        "query_id": "HB50_GLOBAL_RIVER_DISCHARGE_FLOOD_FORECAST",
        "domain": "Hydrology & Soils",
        "source_framework": "Global Operational Flood Early Warning",
        "query": "Global historical river discharge reanalysis and seasonal hydrological flood awareness reforecasts for basin-scale flood monitoring",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "glofas_discharge", "regex": r"\bglofas\b|global\s+flood\s+awareness"},
        ],
    },
    {
        "query_id": "HB51_GLOBAL_HYDROLOGICAL_BASINS_NETWORK",
        "domain": "Hydrology & Soils",
        "source_framework": "Global Watershed Delineation & Hydrography",
        "query": "Global hydrologically conditioned digital elevation derivatives, nested watershed drainage basins, and river network hydrography",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "hydrosheds_hydrobasins", "regex": r"\bhydrosheds\b|\bhydrobasins\b|\bhydrorivers\b"},
        ],
    },
    {
        "query_id": "HB52_GLOBAL_RESERVOIRS_AND_DAMS",
        "domain": "Hydrology & Soils",
        "source_framework": "Global Freshwater Impoundment & Dam Inventory",
        "query": "Global georeferenced spatial database of major artificial water reservoirs and dam locations with storage capacity attributes",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "grand_dams_reservoirs", "regex": r"global\s+reservoir\s+and\s+dam|\bgrandv1\b"},
        ],
    },
    {
        "query_id": "HB53_GLOBAL_RIVER_WIDTHS_AND_DISCHARGE",
        "domain": "Hydrology & Soils",
        "source_framework": "Surface Water Altimetry & River Geomorphology",
        "query": "Global satellite-derived river channel width vector databases and spaceborne altimetry river discharge reach datasets",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "grwl_river_widths", "regex": r"global\s+river\s+width|\bgrwl\b"},
            {"family_id": "swot_river_discharge", "regex": r"\bswot\b.*discharge|sword\s+of\s+science"},
        ],
    },
    {
        "query_id": "HB54_GLOBAL_SOIL_TEXTURE_PH_EROSION",
        "domain": "Hydrology & Soils",
        "source_framework": "Global Soil Physical Properties & Erosion Risk",
        "query": "Global gridded maps of soil sand and clay texture fractions, soil bulk density, soil pH, and rainfall-driven soil erosion risk",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "openlandmap_soil_phys", "regex": r"openlandmap.*(?:clay|sand|soil\s+ph|bulk\s+density|water\s+content)", "exclude_regex": r"soilgrids|organic\s+carbon"},
            {"family_id": "soil_erosion_maps", "regex": r"soil\s+erosion|rainfall\s+erosivity", "exclude_regex": r"openlandmap"},
        ],
    },

    # =========================================================================
    # DOMAIN 10: Productivity, Hazards, Energy, Biodiversity & Infrastructure (HB55-HB60)
    # =========================================================================
    {
        "query_id": "HB55_DRY_MATTER_PRODUCTIVITY_AND_VOD",
        "domain": "Productivity & Hazards",
        "source_framework": "Agricultural Biomass Productivity & Vegetation Water",
        "query": "Global 10-daily satellite dry matter productivity rasters and long-term microwave vegetation optical depth leaf moisture records",
        "filters": {"geo_scope": "global", "strict_eo_only": True},
        "expected_families": [
            {"family_id": "clms_dmp", "regex": r"dry\s+matter\s+productivity"},
            {"family_id": "microwave_vod", "regex": r"vegetation\s+optical\s+depth|live-fuel\s+moisture\s+content"},
        ],
    },
    {
        "query_id": "HB56_TROPICAL_CYCLONE_BEST_TRACKS",
        "domain": "Productivity & Hazards",
        "source_framework": "WMO Tropical Cyclone Risk & Climate Stewardship",
        "query": "Global historical archive of tropical cyclone and hurricane best tracks combining multi-agency storm position and intensity records",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "ibtracs_cyclones", "regex": r"\bibtracs\b|international\s+best\s+track\s+archive"},
        ],
    },
    {
        "query_id": "HB57_STORM_SURGE_AND_FLOOD_HAZARD",
        "domain": "Productivity & Hazards",
        "source_framework": "Sendai Framework / Coastal & River Flood Hazard",
        "query": "Global hindcasts and projections of coastal storm surge, extreme sea levels, and global flood hazard frequency and exposure maps",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "global_storm_surge", "regex": r"storm\s+surge|extreme\s+sea\s+level", "exclude_regex": r"flood\s+hazard"},
            {"family_id": "global_flood_hazard", "regex": r"(?:global|coastal)\s+flood\s+hazard", "exclude_regex": r"\bglofas\b|storm\s+surge"},
        ],
    },
    {
        "query_id": "HB58_SEISMIC_EARTHQUAKE_HAZARD",
        "domain": "Productivity & Hazards",
        "source_framework": "Global Natural Disaster Hazard Assessment",
        "query": "Global gridded earthquake seismic hazard frequency, peak ground acceleration distribution, and proportional mortality risk maps",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "sedac_earthquake_hazard", "regex": r"earthquake\s+hazard|seismic\s+hazard"},
        ],
    },
    {
        "query_id": "HB59_GLOBAL_WIND_ENERGY_AND_POWER_PLANTS",
        "domain": "Productivity & Hazards",
        "source_framework": "SDG 7 Renewable Energy Resource & Infrastructure",
        "query": "Global high-resolution wind energy resource atlas providing wind power density, capacity factor, air density, and global power plant inventories",
        "filters": {"geo_scope": "global", "strict_eo_only": False},
        "expected_families": [
            {"family_id": "global_wind_atlas", "regex": r"global\s+wind\s+atlas"},
            {"family_id": "global_power_plant_db", "regex": r"global\s+power\s+plant\s+database"},
        ],
    },
    {
        "query_id": "HB60_ECOREGIONS_AND_PROTECTED_AREAS",
        "domain": "Productivity & Hazards",
        "source_framework": "CBD Biodiversity Biogeography & Protected Area Networks",
        "query": "Global terrestrial and marine biogeographic ecoregion boundaries and European statutory protected conservation site rasters",
        "filters": {"strict_eo_only": False},
        "expected_families": [
            {"family_id": "terrestrial_marine_ecoregions", "regex": r"(?:terrestrial|marine)\s+ecoregions"},
            {"family_id": "natura2000_protected", "regex": r"\bnatura\s+2000\b"},
        ],
    },
]

# Strict anti-leakage blocklist: ZERO product brand names / missions / agencies / shortnames allowed!
FORBIDDEN_LEAKS_HB = [
    "icesat", "alos", "palsar", "jaxa", "sentinel", "prodes", "inpe", "deter",
    "ghs", "ucdb", "fua", "ghsl", "esm", "gaia", "hrsl", "grid3", "microsoft",
    "openbuildingmap", "dbsm", "globeland30", "esri", "impact observatory", "io_lulc",
    "hrlc", "cci", "esa", "nlcd", "nalcms", "nrcan", "pasture watch", "boreas", "nacp",
    "emit", "lpj", "eosim", "wetcharts", "upch4", "cams", "egg4", "eac4", "carbontracker",
    "oco-2", "oco2", "casa-gfed", "geos-carb", "blueflux", "iasi", "metop", "calipso",
    "caliop", "cloudsat", "misr", "merra", "lis", "otd", "era5", "cordex", "worldclim",
    "chelsa", "terraclimate", "daymet", "prism", "glambie", "measures", "its_live",
    "nsidc", "arcticdem", "rema", "gtopo30", "hydro1k", "socat", "cortad", "noaa",
    "modis", "viirs", "gfw", "global fishing watch", "emodnet", "openet", "eemetric",
    "glofas", "hydrosheds", "hydrobasins", "hydrorivers", "grand", "grwl", "swot",
    "sword", "openlandmap", "pesera", "luisa", "copernicus", "clms", "vodca",
    "ibtracs", "ncdc", "gssr", "ciesin", "sedac", "wri", "wwf", "natura 2000", "wdpa",
]


def _load_rids_from_specs(
    specs: list[dict[str, Any]], df: pd.DataFrame, comb: pd.Series
) -> set[str]:
    rids: set[str] = set()
    for sp in specs:
        for fam in sp["expected_families"]:
            m = comb.str.contains(re.compile(fam["regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                m &= ~comb.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            m = filter_benchmark_family_mask(df, m, sp.get("filters"))
            rids.update(df.iloc[np.flatnonzero(m)]["record_id"].tolist())
    return rids


def _extract_ast_list(script_path: Path, var_name: str) -> list[dict[str, Any]]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == var_name:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if getattr(tgt, "id", None) == var_name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not extract {var_name} from {script_path}")


def main() -> None:
    print("=" * 94)
    print("09. BUILD PROSPECTIVE SINGLE-SHOT BLIND HOLDOUT B (60 INDEPENDENT EO QUERIES)")
    print("=" * 94)

    # 1. Anti-leakage check
    leaks = []
    for sp in HOLDOUT_B_SPECS:
        ql = sp["query"].lower()
        for tok in FORBIDDEN_LEAKS_HB:
            if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", ql):
                leaks.append((sp["query_id"], tok))
    if leaks:
        raise ValueError(f"Product/mission/agency leakage detected in Holdout B queries: {leaks}")
    print(f"[OK] Anti-leakage check passed across all {len(HOLDOUT_B_SPECS)} Holdout B queries (0 leaks).")

    df_full = pd.read_parquet(CANONICAL_PARQUET)
    obs_re = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)
    active_mask = (~df_full["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)) & (
        ~df_full["title"].fillna("").str.contains(obs_re, regex=True).to_numpy(dtype=bool, copy=True)
    )
    df = df_full[active_mask].reset_index(drop=True)
    comb = df["title"].fillna("") + " || " + df["raw_id"].fillna("") + " || " + df["source_catalogue"].fillna("")
    rid_to_full_idx = {r: i for i, r in enumerate(df_full["record_id"].tolist())}

    resolved = []
    missing = []
    unreachable = []
    total_fams = 0
    for sp in HOLDOUT_B_SPECS:
        flt_spec = sp.get("filters", {})
        flt = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            exclude_superseded=True,
            null_safe=True,
        )
        full_mask = apply_filters_mask(df_full, flt)

        fams_out = []
        for fam in sp["expected_families"]:
            mask = comb.str.contains(re.compile(fam["regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                mask &= ~comb.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            mask = filter_benchmark_family_mask(df, mask, flt_spec)
            idxs = np.flatnonzero(mask)
            if len(idxs) == 0:
                missing.append((sp["query_id"], fam["family_id"]))
                continue
            rids = df.iloc[idxs]["record_id"].tolist()
            n_reachable = sum(1 for r in rids if full_mask[rid_to_full_idx[r]])
            if n_reachable == 0:
                unreachable.append((sp["query_id"], fam["family_id"], len(rids)))
            total_fams += 1
            fams_out.append({
                "family_id": fam["family_id"],
                "n_matches": int(len(idxs)),
                "n_reachable_under_filter": int(n_reachable),
                "sample_titles": df.iloc[idxs[:3]]["title"].tolist(),
                "record_ids": rids,
            })
        resolved.append({
            "query_id": sp["query_id"],
            "domain": sp["domain"],
            "source_framework": sp["source_framework"],
            "query": sp["query"],
            "filters": flt_spec,
            "n_expected_families": len(fams_out),
            "expected_families": fams_out,
        })

    if missing:
        raise RuntimeError(f"Holdout B families matched 0 active records: {missing}")
    if unreachable:
        raise RuntimeError(f"Holdout B families unreachable under SearchFilters: {unreachable}")
    print(f"[OK] All {len(resolved)} queries and {total_fams} families resolved and verified filter-reachable.")

    # 2. Strict within-query AND cross-query disjointness across ALL Holdout B families
    rid_owner: dict[str, tuple[str, str]] = {}
    intra_overlaps = []
    for q in resolved:
        for fam in q["expected_families"]:
            for r in fam["record_ids"]:
                if r in rid_owner:
                    intra_overlaps.append((rid_owner[r], (q["query_id"], fam["family_id"]), r))
                else:
                    rid_owner[r] = (q["query_id"], fam["family_id"])
    if intra_overlaps:
        raise RuntimeError(
            f"Overlapping record_ids within Holdout B ({len(intra_overlaps)}): {intra_overlaps[:8]}"
        )
    print(f"[OK] Strict within-suite disjointness PASSED (0 overlapping record_ids across all {total_fams} families / {len(rid_owner)} records).")

    # 3. Strict cross-suite disjointness vs Dev-45 AND Holdout A
    dev_specs = _extract_ast_list(SCRIPT_DIR / "02_build_policy_benchmark.py", "BENCHMARK_SPECS")
    ha_specs = _extract_ast_list(SCRIPT_DIR / "08_build_holdout_a_external.py", "HOLDOUT_A_SPECS")
    dev_rids = _load_rids_from_specs(dev_specs, df, comb)
    ha_rids = _load_rids_from_specs(ha_specs, df, comb)
    hb_rids = set(rid_owner.keys())

    ov_dev = sorted(hb_rids & dev_rids)
    ov_ha = sorted(hb_rids & ha_rids)
    if ov_dev or ov_ha:
        raise RuntimeError(
            f"Cross-suite overlap detected! vs Dev-45={len(ov_dev)} ({ov_dev[:5]}), vs Holdout A={len(ov_ha)} ({ov_ha[:5]})"
        )
    print(
        f"[OK] Cross-suite disjointness PASSED: Holdout B ({len(hb_rids)} rids) ∩ Dev-45 ({len(dev_rids)}) = 0, "
        f"∩ Holdout A ({len(ha_rids)}) = 0."
    )

    # Print compact domain summary
    domain_counts: dict[str, int] = {}
    for q in resolved:
        domain_counts[q["domain"]] = domain_counts.get(q["domain"], 0) + 1
    for dom, cnt in domain_counts.items():
        print(f"  - {dom:28s}: {cnt} queries")

    payload = {
        "benchmark_name": "Prospective_Single_Shot_Blind_Holdout_B_v4_standalone",
        "blind_status": "prospective_single_shot_blind",
        "n_queries": len(resolved),
        "n_total_families": total_fams,
        "n_unique_active_record_ids": len(hb_rids),
        "disjoint_from_dev_45": True,
        "disjoint_from_holdout_a": True,
        "disjoint_across_all_holdout_b_queries": True,
        "queries": resolved,
    }
    out_p = ARTIFACTS_DIR / "holdout_b_blind_benchmark.json"
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVED] -> {out_p}")


if __name__ == "__main__":
    main()
