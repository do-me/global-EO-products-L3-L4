#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.0",
#   "pyarrow>=15.0.0",
# ]
# ///
"""
05_external_policy_validation/03_validate_against_catalogue.py
==============================================================
Reproducibly checks every extracted EO product in
`results/extracted_policy_products.csv` against:
  1. `03_harmonization/data/eo_l3_l4_canonical_products.parquet` (canonical L3/L4 products)
  2. `03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet` (all harmonized L3/L4 records)
  3. `02_preparation/data/excluded_records_log.parquet` (logged L0-L2 / aerial / in-situ exclusions)

Uses a strict, auditable, tiered evidence protocol (zero loose fuzzy matches,
matching strictly on `raw_id`, `title`, and `doi` rather than noisy web URLs):
  - Tier 1 (`EXACT_DOI`)              : Exact normalized DOI match against `doi` (or `RELATED_DOI` on `related_dois`).
  - Tier 2 (`EXACT_ID_OR_SHORTNAME`)  : Exact `raw_id`, CMR `concept_id`, CMR `short_name`, or official product ID match.
  - Tier 3 (`PRODUCT_FAMILY_SIGNATURE`): Explicit, product-specific regex signature matched strictly against
                                         `raw_id | title | doi` with optional source/pool constraints.
  - Tier 4 (`EXCLUDED_IN_AUDIT_LOG`)  : Verifies whether out-of-scope L1/L2 items (e.g., CMR SNWG L2 collections)
                                         were deterministically excluded in `02_preparation/data/excluded_records_log.parquet`.

Outputs:
  - `results/validation_matches.csv`  : Complete item-by-item evidence table
  - `results/validation_summary.csv`  : Benchmark-by-benchmark recall summary (overall & in-scope L3/L4 EO)
  - `results/gap_analysis.csv`        : Every `IN_SCOPE_OPEN_L3_L4_EO` item not found in the catalogue with root-cause class

Run via:
    uv run 05_external_policy_validation/03_validate_against_catalogue.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RES_DIR = BASE_DIR / "results"

ALL_PATH = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_all_catalogue_records.parquet"
CAN_PATH = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
EXC_PATH = REPO_ROOT / "02_preparation" / "data" / "excluded_records_log.parquet"
EXT_PATH = RES_DIR / "extracted_policy_products.csv"


# Curated, strict product-family signatures (matched strictly against `raw_id | title | doi`):
# (item_match_regex, cat_title_id_doi_regex, optional_source_or_pool_filter, match_specificity_tier)
SIGNATURE_RULES: list[tuple[str, str, str | None, str]] = [
    # --- Paris Agreement / GHG / Carbon / Biomass ---
    (r"OCO-2.*MIP|oco2-mip", r"OCO-?2.*MIP|10\.48588/npf6-sw92", None, "EXACT_PRODUCT"),
    (r"OCO-2 GEOS|oco2geos", r"OCO-?2.*GEOS|OCO2_GEOS_L3CO2", None, "EXACT_PRODUCT"),
    (r"OCO-3.*SAM|oco3-co2-sams", r"OCO-?3.*(?:SAM|Snapshot Area Map)|OCO3_L2_Lite", None, "EXACT_PRODUCT"),
    (r"GOSAT.*Methane|gosat-based-ch4budget|GOSAT / TROPOMI.*Methane", r"GOSAT.*(?:CH4|Methane).*Budget|GOSAT-based Top-down.*Methane|Top-down.*Methane.*Budget.*GOSAT", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"GEDI.*L4B|GEDI_ISS_L4B|GEDI Biomass Map", r"GEDI.*L4B|GEDI04_B|GEDI.*Gridded Aboveground Biomass", None, "EXACT_PRODUCT"),
    (r"GEDI.*Vegetation Height|GEDI.*Canopy", r"GEDI.*(?:Canopy|Vegetation) Height|GEDI_ICESAT2_Global_Veg_Height", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"ESA CCI Biomass|CCI Biomass", r"Biomass[_\s-]*cci|CCI.*Biomass|ESACCI.*BIOMASS", None, "EXACT_PRODUCT"),
    (r"ICESat-2 Boreal.*Biomass", r"ICESat-2.*Boreal.*Biomass|Boreal_AGB_Density_ICESat2|C2756302505", None, "EXACT_PRODUCT"),
    (r"JPL 2020 Global.*Biomass", r"JPL.*2020.*Global.*Biomass|Xu.*2021.*Global.*Live.*Biomass|10\.5281/zenodo\.4161694", None, "EXACT_PRODUCT"),
    (r"NCEO Africa.*Biomass", r"NCEO.*Africa.*Biomass|10\.25392/leicester\.data\.15060270", None, "EXACT_PRODUCT"),
    (r"Spawn et al|Harmonized global maps of above and belowground biomass", r"Global_Maps_C_Density_2010|Global Aboveground and Belowground Biomass Carbon Density", None, "EXACT_PRODUCT"),
    (r"HILDA\+", r"HILDA\+|10\.1594/pangaea\.921846", None, "EXACT_PRODUCT"),
    (r"ODIAC", r"\bODIAC\b|10\.17595/20170411", None, "EXACT_PRODUCT"),
    (r"Vulcan.*CO2|vulcan-ffco2", r"\bVulcan\b.*(?:CO2|Fossil|Emissions)", None, "EXACT_PRODUCT"),
    (r"CarbonTracker-CH|ct-ch4", r"CarbonTracker[-_ ]?CH4|CarbonTracker.*Methane", None, "EXACT_PRODUCT"),
    (r"TM5-4DVar|tm54dvar", r"TM5[-_ ]?4DVAR", None, "EXACT_PRODUCT"),
    (r"ECCO-Darwin|eccodarwin", r"ECCO[-_ ]?Darwin", None, "EXACT_PRODUCT"),
    (r"LPJ-EOSIM|lpjeosim", r"LPJ[-_ ]?EOSIM", None, "EXACT_PRODUCT"),
    (r"LPJ-wsl|lpjwsl", r"LPJ[-_ ]?wsl|10\.5281/zenodo\.6964918", None, "EXACT_PRODUCT"),
    (r"MiCASA|micasa-carbonflux", r"\bMiCASA\b", None, "EXACT_PRODUCT"),
    (r"CASA-GFED|casagfed", r"CASA[-_ ]?GFED", None, "EXACT_PRODUCT"),
    (r"BlueFlux|blueflux", r"\bBlueFlux\b", None, "EXACT_PRODUCT"),
    (r"GRA2PES|GRA²PES|gra2pes", r"GRA2PES|GRA²PES|mds2-3520", None, "EXACT_PRODUCT"),
    (r"cms-mangrove-agb|Global Mangrove Aboveground Biomass", r"CMS.*Mangrove.*(?:Biomass|Canopy)|Global Mangrove.*(?:Biomass|Canopy Height)", None, "EXACT_PRODUCT"),
    (r"emit-ch4plume|EMIT Methane.*Plume", r"EMITL2BCH4PLM|emit-ch4plume", None, "EXACT_PRODUCT"),
    (r"goes-ch4plume|Geostationary.*Methane.*Oil and Gas", r"GOES.*(?:Methane|CH4).*Plume", None, "EXACT_PRODUCT"),
    (r"geos-co2-daygrid|GEOS Assimilated CO₂ Column", r"OCO-?2 GEOS.*assimilated CO2|GEOS.*CO2.*Column", None, "EXACT_PRODUCT"),
    (r"geos-co-ch4|geos-ghg|GEOS Assimilated (?:CH₄|CO₂, CH₄)", r"GEOS.*(?:CH4|Methane).*Column|GEOS-?CF.*CH4", None, "EXACT_PRODUCT"),
    (r"epa-ch4emission|Gridded.*EPA CH4|U\.S\. Gridded Anthropogenic Methane", r"Gridded.*EPA.*(?:CH4|Methane)|EPA.*Gridded.*Methane|10\.5281/zenodo\.8367082", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"OMI_tropospheric_no2|OMI/Aura Tropospheric Nitrogen Dioxide", r"OMI.*Tropospheric.*(?:NO2|Nitrogen Dioxide)|OMNO2d", None, "EXACT_PRODUCT"),
    (r"sedac-popdensity|Gridded World Population Density v4\.11", r"GPW.*v4\.11|CIESIN_SEDAC_GPWv4|Gridded Population of the World.*v4", None, "EXACT_PRODUCT"),

    # --- Forests / Land Cover / Agriculture / Fire / Wetlands ---
    (r"Global Forest Watch.*Tree Cover Loss|UMD Hansen|Global Forest Watch \(UMD", r"Hansen Global Forest Change|UMD/hansen/global_forest_change", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Forest Carbon Flux Maps", r"Forest.*(?:Greenhouse Gas|Carbon).*Flux|CMS_Forest_Carbon_Fluxes", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"ESA CCI Land Cover|ESA-CCI-LC|ESA CCI-LC", r"(?:ESA\s*)?CCI.*Land Cover|ESACCI-LC|satellite-land-cover", None, "EXACT_PRODUCT"),
    (r"ESA CCI High Resolution Land Cover|HRLC", r"High Resolution Land Cover.*CCI|High_Resolution_Land_Cover_cci", None, "EXACT_PRODUCT"),
    (r"Copernicus.*100\s*m.*Land Cover|CGLS-LC100|Copernicus Global Land Service LC100|Copernicus Global Land Cover", r"Land Cover 2015-2019 \(raster 100 m\)|cgls_landcover|CGLS[-_]LC100", None, "EXACT_PRODUCT"),
    (r"ESA WorldCover", r"ESA.*WorldCover|WorldCover.*10\s*m|ESA/WorldCover", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"WorldCereal", r"WorldCereal", None, "EXACT_PRODUCT"),
    (r"Global Mangrove Watch", r"Global Mangrove Watch|\bGMW\b.*Mangrove", None, "EXACT_PRODUCT"),
    (r"Sentinel-3 World Fire Atlas", r"Sentinel-3.*World Fire Atlas|World Fire Atlas.*SLSTR|\bS3WFA\b", None, "EXACT_PRODUCT"),
    (r"GlobeLand\s*30", r"GlobeLand\s*30|GlobeLand30", None, "EXACT_PRODUCT"),
    (r"Global Land Cover 10\b", r"FROM[-_ ]?GLC10|Global Land Cover 10\b", None, "EXACT_PRODUCT"),
    (r"Global Artificial Impervious Areas", r"Global Artificial Impervious|\bGAIA\b.*Impervious", None, "EXACT_PRODUCT"),
    (r"Global 30 m Impervious Surface Map", r"GISD30|10\.5281/zenodo\.3505079|Global 30\s*m Impervious", None, "EXACT_PRODUCT"),
    (r"Esri 10-Meter Land Cover|ESRI 2020 Global Land Cover", r"io-lulc|Esri 10-Meter Land Cover|Impact Observatory.*Land Cover", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Landcover 2050", r"Landcover 2050|Land Cover 2050", None, "EXACT_PRODUCT"),
    (r"Local Climate Zones", r"Local Climate Zones", None, "EXACT_PRODUCT"),
    (r"EU Crop Map", r"EU\s*Crop\s*Map|EUCROPMAP", None, "EXACT_PRODUCT"),
    (r"FAO Global Land Cover SHARE|GLC-Share", r"Dominant land cover type.*GLC[-_ ]?SHARE", None, "EXACT_PRODUCT"),
    (r"UMD GLAD Landsat.*ARD", r"landsat_glad\.swa\.ard2|GLAD.*Landsat.*ARD", None, "EXACT_PRODUCT"),
    (r"ALOS.*PALSAR Global Mosaic", r"alos[-_]palsar[-_]mosaic|ALOS.*PALSAR.*mosaic|JAXA/ALOS/PALSAR/YEARLY/SAR", "src:28_planetary_computer_stac|27_gee_official_stac", "EXACT_PRODUCT"),
    (r"ALOS-2 / Global Wetland Inundation|Global Mangrove.*Rosenqvist", r"Global Mangrove Watch|ALOS.*(?:Wetland|Inundation|Mangrove)", None, "RELATED_PROXY_OR_SUBSET"),

    # --- Water / Oceans / Lakes / Altimetry ---
    (r"Global Surface Water|GSWE|DB_GblSrfcWater_ESA", r"Global Surface Water|JRC/GSW", None, "EXACT_PRODUCT"),
    (r"DHI Global Wetland Map", r"DHI.*Wetland", None, "EXACT_PRODUCT"),
    (r"Lake Water Quality", r"Lake Water Quality", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"ESA CCI Lakes|CCI Lakes", r"ESA Lakes Climate Change Initiative|Lakes_cci|ESACCI-LAKES", None, "EXACT_PRODUCT"),
    (r"Hydroweb THEIA", r"Hydroweb", None, "EXACT_PRODUCT"),
    (r"DAHITI", r"\bDAHITI\b", None, "EXACT_PRODUCT"),
    (r"G-REALM|Global Reservoir and Lake Monitor", r"G-?REALM|Global Reservoir and Lake Monitor", None, "EXACT_PRODUCT"),
    (r"WorldWater", r"ESA.*WorldWater|\bWorldWater\b.*Surface Water", None, "EXACT_PRODUCT"),
    (r"SDG 14\.1\.1a Indicator Maps EEZ", r"SDG\s*14\.1\.1a.*EEZ|14\.1\.1a.*Chlorophyll", None, "EXACT_PRODUCT"),
    (r"(?:Ocean Colour CCI|OC-CCI).*Chlorophyll", r"(?:Ocean Colour Climate Change Initiative|Ocean_Colour_cci|OC[-_ ]?CCI).*chlorophyll", None, "EXACT_PRODUCT"),
    (r"Ocean Colour CCI|OC-CCI", r"Ocean Colour Climate Change Initiative|Ocean_Colour_cci|OC[-_ ]?CCI", None, "EXACT_PRODUCT"),
    (r"NOAA VIIRS.*Chlorophyll", r"VIIRS.*Chlorophyll", None, "EXACT_PRODUCT"),
    (r"EUSeaMap", r"EUSeaMap.*Broad-Scale Predictive Habitat", "src:11_eea_and_emodnet", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"GLOBAL_ANALYSIS_FORECAST_PHY_001_024", r"GLOBAL_ANALYSISFORECAST_PHY_001_024|GLOBAL_ANALYSIS_FORECAST_PHY_001_024", "src:01_copernicus_cmems", "EXACT_PRODUCT"),
    (r"Ocean Monitoring Indicators", r"OMI_|Ocean Monitoring Indicator", "src:01_copernicus_cmems", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Ocean Biogeochemistry Analysis and Forecast|biogeochemical global ocean analysis", r"Global Ocean Biogeochemistry Analysis and Forecast|GLOBAL_ANALYSISFORECAST_BGC_001_028", "src:01_copernicus_cmems", "EXACT_PRODUCT"),

    # --- Urban / Settlements / Population / Heat / Air Quality ---
    (r"Global Human Settlement-Built-Up|GHS-BUILT", r"GHS[-_ ]?BUILT[-_ ]?S\b|GHS[-_ ]?BUILT", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Human Settlement-Population", r"GHS[-_ ]?POP", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Human Settlement-Settlement Model", r"GHS[-_ ]?SMOD", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Human Settlement-Functional Urban Areas", r"GHS[-_ ]?FUA|GHS functional urban areas", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Urban Centre Database", r"GHS[-_ ]?UCDB|Urban Centre Database", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"GHSL Data Package 2023", r"GHS-(?:BUILT-S|BUILT-H|BUILT-V|BUILT-C|POP|SMOD).*R2023A", "src:07_ec_jrc_data_catalogue", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Human Settlement Layer|CEMS GHSL", r"GHSL|GHS[-_ ]?BUILT|GHS[-_ ]?POP|Global Human Settlement", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"World Settlement Footprint 2015", r"World Settlement Footprint.*2015|WSF2015", None, "EXACT_PRODUCT"),
    (r"World Settlement Footprint 2019", r"World Settlement Footprint.*2019|WSF2019", None, "EXACT_PRODUCT"),
    (r"World Settlement Footprint Evolution", r"World Settlement Footprint.*Evolution|WSF.*Evolution", None, "EXACT_PRODUCT"),
    (r"World Settlement Footprint \(WSF", r"World Settlement Footprint \(WSF\) (?:2015|2019|Evolution)", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"High Resolution Settlement Layer", r"High Resolution Settlement Layer.*Overall Population|\bhrslpop\b", None, "EXACT_PRODUCT"),
    (r"Human Built-up and Settlement Extent|HBASE", r"Human Built-up And Settlement Extent|\bHBASE\b|10\.7927/h4dn434s", None, "EXACT_PRODUCT"),
    (r"Global Man-made Impervious Surface", r"Global Man-made Impervious Surface|\bGMIS\b|10\.7927/h4p55kkf", None, "EXACT_PRODUCT"),
    (r"Global Grid of Probabilities of Urban Expansion to 2030", r"Probabilities of Urban Expansion to 2030|10\.7927/h4z899cg", None, "EXACT_PRODUCT"),
    (r"VIIRS Plus DMSP Change in Lights", r"VIIRS Plus DMSP Change in Lights|10\.7927/9ryj-6467", None, "EXACT_PRODUCT"),
    (r"Global Urban Heat Island", r"Global Urban Heat Island|10\.7927/h4h70crf", None, "EXACT_PRODUCT"),
    (r"Global Annual PM2\.5 Grids", r"Global Annual PM2\.5 Grids|10\.7927/h4zk5dqs", None, "EXACT_PRODUCT"),
    (r"Development Threat Index", r"Development Threat Index|10\.7927/61jv-th84", None, "EXACT_PRODUCT"),
    (r"Global Human Modification of Terrestrial Systems", r"Global Human Modification|10\.7927/edbc-3z60", None, "EXACT_PRODUCT"),
    (r"Urban-rural continuum", r"Urban-rural continuum|10\.6084/m9\.figshare\.12579572", None, "EXACT_PRODUCT"),
    (r"Global High Resolution Daily Extreme Urban Heat", r"Daily Extreme Urban Heat|10\.7927/fq7g-ny13", None, "EXACT_PRODUCT"),
    (r"WorldPop", r"WorldPop.*(?:Population|Global|100m|1km)|Global.*WorldPop|WorldPop/GP/", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Sub-Saharan Africa Settlement Extents", r"GRID3.*Settlement Extents", None, "EXACT_PRODUCT"),
    (r"Copernicus Urban Atlas|CLMS Urban Atlas", r"Urban Atlas Land Cover/Land Use", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Copernicus Imperviousness|HRL.*Imperviousness", r"Imperviousness Density", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"European Settlement Map", r"European Settlement Map", None, "EXACT_PRODUCT"),
    (r"Building Footprints USA, Canada, Uganda", r"Microsoft Building Footprints|ms-buildings", None, "EXACT_PRODUCT"),
    (r"Open Buildings", r"Open Buildings", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Urban Extents, Continental U\.S\.", r"Urban Extents from VIIRS|10\.7927/a49b-sm16", None, "EXACT_PRODUCT"),
    (r"LandScan", r"LandScan", None, "COMPATIBLE_EDITION_OR_COLLECTION"),

    # --- Copernicus CLMS / CEMS / C3S / CAMS (KCEO & CEOS) ---
    (r"Natura 2000 \(N2K\)", r"N2K Land Cover/Land Use", "src:02_copernicus_clms", "EXACT_PRODUCT"),
    (r"Hot Spot Monitoring", r"Hot Spot Monitoring", "src:02_copernicus_clms", "EXACT_PRODUCT"),
    (r"Corine Land Cover \(CLC\)", r"CORINE Land Cover 2018", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"CLC\+ Backbone", r"CLCplus Backbone", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"HRL.*Forest", r"Tree Cover Density|Dominant Leaf Type|Forest Type", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"HR-VPP|Vegetation Phenology and Productivity|Copernicus Land Surface Phenology", r"High-?Resolution Vegetation Phenology|Seasonal Trajectories.*10-daily", "pool:copernicus_services", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"RLIE S1\+S2", r"River and Lake Ice Extent", None, "EXACT_PRODUCT"),
    (r"Water and Wetness", r"Water and Wetness", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Riparian Zones", r"Riparian Zones", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Small Woody Features", r"Small Woody Features", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"CLMS Global Water Bodies", r"Water Bodies 2020-present|Water Bodies.*(?:100|300)\s*m", "src:02_copernicus_clms", "EXACT_PRODUCT"),
    (r"CLMS Global Water Level", r"(?:River|Lake) Water Level", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"CLMS.*FAPAR|Copernicus.*fAPAR", r"Fraction of Absorbed Photosynthetically Active Radiation", "src:02_copernicus_clms", "EXACT_PRODUCT"),
    (r"CLMS.*Soil Water Index", r"Soil Water Index", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"CLMS.*Leaf Area Index|Copernicus.*LAI", r"Leaf Area Index", "src:02_copernicus_clms", "EXACT_PRODUCT"),
    (r"Copernicus Global Land Service NDVI", r"Normali[sz]ed Difference Vegetation Index", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Pan-European Copernicus High Resolution Layers", r"Imperviousness Density|Tree Cover Density", "src:02_copernicus_clms", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"EFFIS Burnt Area", r"European Forest Fire Information System|EFFIS", "pool:copernicus_services", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"C3S Hydrology-Related Climate Impact", r"Hydrology.*climate.*indicator|sis-hydrology-meteorology", "src:03_copernicus_c3s_cds", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"C3S Urban Climate|ERA5 Products", r"reanalysis-era5-single-levels|reanalysis-era5-complete|ERA5.*single levels|sis-urban-climate", "src:03_copernicus_c3s_cds", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"CAMS Air Quality", r"CAMS European air quality|CAMS global atmospheric", "src:04_copernicus_cams_ads", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"EEA Floods Viewer|European Flood Hazard", r"European Flood Awareness System|EFAS|Flood.*Hazard", None, "RELATED_PROXY_OR_SUBSET"),

    # --- UNCCD / Soil / Productivity / Vegetation / Cryosphere ---
    (r"JRC Land Productivity Dynamics", r"Land Productivity Dynamics", None, "EXACT_PRODUCT"),
    (r"Trends\.Earth|MOD13Q1|MYD13Q1", r"MOD13Q1|MYD13Q1", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"SoilGrids", r"SoilGrids250m|SOILGRIDS 2\.0 - ISRIC|SoilGrids", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Harmonized World Soil Database|HWSD", r"Harmonized World Soil Database|\bHWSD\b", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Soil Organic Carbon Map|GSOCmap|\(GSOC", r"Global Soil Organic Carbon Map|GSOC\)", "src:34_fao_geospatial_catalog_ckan", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"MODIS Vegetation Continuous Fields", r"MOD44B|Vegetation Continuous Fields", None, "EXACT_PRODUCT"),
    (r"MODIS LAI|Myneni", r"MOD15A2H|MCD15A2H|MOD15A3H|MYD15A2H", None, "EXACT_PRODUCT"),
    (r"MODIS Vegetation Indices.*Phenology|Huete", r"MCD12Q2|MOD13Q1|MOD13A2", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"ESA CCI Snow", r"ESA Snow Climate Change Initiative|Snow_cci|ESACCI-SNOW", None, "EXACT_PRODUCT"),
    (r"Global Leaf Chlorophyll Content|GCOM-C SGLI Chlorophyll", r"Global.*Leaf Chlorophyll|Xu.*2022.*Chlorophyll|GCOM-C.*Leaf Chlorophyll", None, "EXACT_PRODUCT"),
    (r"MODIS / GIMMS fAPAR|Zhu et al", r"GIMMS.*(?:FPAR|fAPAR)|MOD15A2H|MCD15A2H", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"MODIS GPP / NPP|MOD17", r"MOD17A2H|MOD17A3|MYD17A2H", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Solar-Induced Fluorescence.*Joiner", r"SIF_SCIAMACHY_GOME2|OCO-?3.*SIF|GOSAT.*SIF|TROPOSIF|TCSIF|GOSIF", None, "COMPATIBLE_EDITION_OR_COLLECTION"),

    # --- DEMs & Topography ---
    (r"Copernicus DEM GLO-30", r"COP-?DEM.*GLO-?30|CopDEM.*30\s*m|GLO-30", None, "EXACT_PRODUCT"),
    (r"ASTER.*GDEM|ASTER.*Global Digital Elevation", r"ASTGTM|ASTER Global Digital Elevation", None, "EXACT_PRODUCT"),
    (r"ASTER.*Global Emissivity|DB_ASTER_GED", r"AG100|AG1KM|ASTER Global Emissivity", None, "EXACT_PRODUCT"),
    (r"SRTM.*CHILI", r"SRTM.*CHILI|SRTM_CHILI", None, "EXACT_PRODUCT"),
    (r"SRTM", r"SRTMGL1|SRTMGL3|NASADEM_HGT|SRTM Digital Elevation", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"TanDEM-X DEM", r"TanDEM-X.*Digital Elevation Model", None, "EXACT_PRODUCT"),
    (r"ALOS World 3D|AW3D30", r"AW3D30|ALOS World 3D", None, "EXACT_PRODUCT"),
    (r"National Elevation Dataset.*mTPI", r"US/mTPI|US NED mTPI", None, "EXACT_PRODUCT"),
    (r"National Elevation Dataset.*Topographic Diversity", r"US NED Topographic Diversity", None, "EXACT_PRODUCT"),
    (r"National Elevation Dataset \(NED\)|DB_NED_USGS", r"USGS/3DEP|National Elevation Dataset|3D Elevation Program", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Global Land Survey Digital Elevation Model|GLSDEM", r"GLSDEM|Global Land Survey Digital Elevation", None, "EXACT_PRODUCT"),
    (r"Greenland Ice Mapping Project.*DEM|DB_GIMP_DEM", r"NSIDC-0645|NSIDC-0715|Greenland Ice Mapping Project.*DEM", None, "EXACT_PRODUCT"),
    (r"Radarsat Antarctic Mapping Project.*DEM|DB_RAMP_DEM", r"NSIDC-0082|Radarsat Antarctic Mapping Project Digital Elevation", None, "EXACT_PRODUCT"),
    (r"GMTED2010", r"GMTED2010", None, "EXACT_PRODUCT"),
    (r"GTOPO30", r"USGS/GTOPO30|GTOPO30:\s*Global 30 Arc-Second Elevation", None, "EXACT_PRODUCT"),
    (r"Canadian Digital Elevation Model|DB_CDEM", r"Canadian Digital Elevation Model|\bcdem\b", None, "EXACT_PRODUCT"),
    (r"Global Land One-kilometer Base Elevation|DB_GLOBE_NCEI", r"Global Land One-kilometer Base Elevation", None, "EXACT_PRODUCT"),
    (r"Global Climate Model DEM", r"Global Climate Model DEM", None, "EXACT_PRODUCT"),

    # --- USGEO EOA 2023 Specific Products ---
    (r"OSISAF Global Daily Sea Ice|DB_SeaIce_METSAT_NCEI", r"Global Sea Ice Concentration Climate Data Record|OSI\s*SAF.*Sea Ice Concentration", None, "EXACT_PRODUCT"),
    (r"NLCD.*Tree Canopy", r"NLCD_RELEASES/.*/TCC|USFS Tree Canopy Cover", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"NLCD\s*2006|DB_NLCD2006", r"nlcd_2006|NLCD2006", None, "EXACT_PRODUCT"),
    (r"National Land Cover Data(?:base|set)|\bNLCD\b", r"Annual National Land Cover Database|NLCD_RELEASES/2021_REL/NLCD", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"LANDFIRE.*EVT|Existing Vegetation Type", r"LANDFIRE.*(?:EVT|Existing Vegetation Type)", None, "EXACT_PRODUCT"),
    (r"LANDFIRE.*ExistVegCover|Existing Vegetation Cover", r"LANDFIRE.*(?:EVC|Existing Vegetation Cover)", None, "EXACT_PRODUCT"),
    (r"LANDFIRE.*ExstVegHght|Existing Vegetation Height", r"LANDFIRE.*(?:EVH|Existing Vegetation Height)", None, "EXACT_PRODUCT"),
    (r"LANDFIRE.*40.*FBFM|40 Fire Behavior Fuel", r"LANDFIRE.*(?:FBFM40|40.*Fuel Model)", None, "EXACT_PRODUCT"),
    (r"LANDFIRE.*13.*FBFM|13 Fire Behavior Fuel", r"LANDFIRE.*(?:FBFM13|13 Anderson Fire Behavior)", None, "EXACT_PRODUCT"),
    (r"LANDFIRE.*Historical Disturbance", r"LANDFIRE.*Disturbance", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"NALCMS|North American Land Change Monitoring|North American Environmental Atlas - Land Cover", r"NALCMS|Land Cover of North America at 30 meters", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"LANID|LandID|Landsat-based Irrigation", r"\bLANID\b|Landsat-based Irrigation", None, "EXACT_PRODUCT"),
    (r"MIrAD-US|MODIS Irrigated Agriculture", r"\bMIrAD\b|MODIS Irrigated Agriculture", None, "EXACT_PRODUCT"),
    (r"eMODIS", r"\beMODIS\d*\b", None, "EXACT_PRODUCT"),
    (r"MERRA.*Aerosol|MERRAero|MERRA-AOT", r"MERRA-?2.*(?:aer|Aerosol)|MERRA.*Aerosol", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"MERRA.*Precip", r"MERRA-?2.*(?:flx|Surface Flux Diagnostics)", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"MERRA.*Temp", r"MERRA-?2.*(?:slv|Single-Level Diagnostics)", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"MERRA-Land|MERRA", r"MERRA-?2", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"GLIMS", r"NSIDC-0272|\bGLIMS\b", None, "EXACT_PRODUCT"),
    (r"NLDAS-2 Forcings", r"NLDAS.*Primary Forcing", None, "EXACT_PRODUCT"),
    (r"NLDAS Noah", r"NLDAS\s+Noah", None, "EXACT_PRODUCT"),
    (r"NLDAS MOSAIC", r"NLDAS\s+Mosaic", None, "EXACT_PRODUCT"),
    (r"Sea Ice and Snow Cover Extent.*NCEI|NSIDC.*Sea Ice Extent", r"NSIDC-0046|Sea Ice Index|G02135|NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice", "src:16_nasa_cmr|20_noaa_onestop", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"MIRCA2000", r"MIRCA2000", None, "EXACT_PRODUCT"),
    (r"GIMMS Leaf Area Index", r"GIMMS Leaf Area Index \(GIMMS LAI4g\)|Spatiotemporally consistent global dataset of the GIMMS Leaf Area Index", None, "EXACT_PRODUCT"),
    (r"GIMMS 3g|GIMMS_3g", r"GIMMS.*(?:3G|NDVI)", None, "EXACT_PRODUCT"),
    (r"GLEAM.*Soil Moisture", r"^GLEAM4$|GLEAM4-HR|GLEAM.*v[34].*(?:Evaporation|Soil Moisture)", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Copernicus Marine Environment Monitoring Service|DB_CMEMS_EU", r"GLOBAL_.*_001_", "src:01_copernicus_cmems", "RELATED_PROXY_OR_SUBSET"),
    (r"VIIRS Land Cover", r"VNP12|VIIRS.*(?:Land Cover|Surface Type)", None, "EXACT_PRODUCT"),
    (r"GAP Land Cover", r"USGS Gap Land Cover|USGS/GAP/CONUS|GAP/LANDFIRE", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"GridSat-GOES|DB_GVAR_NCEI", r"GridSat-GOES|10\.7289/V5HM56GM", "src:20_noaa_onestop", "EXACT_PRODUCT"),
    (r"GHRSST.*Level 4", r"GHRSST Level 4|MUR-JPL-L4", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"AVHRR Pathfinder.*Aerosol|DB_AerosolOD_AVHRR_NCEI", r"CDR.*AVHRR.*Aerosol Optical Thickness", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Pathfinder.*Sea Surface Temperature|DB_SST_AVHRR_NCEI", r"AVHRR Pathfinder.*Sea Surface Temperature", None, "EXACT_PRODUCT"),
    (r"Global Ocean Heat and Salt Content|Global Ocean Heat Content", r"Ocean Heat.*Content|World Ocean Atlas", None, "EXACT_PRODUCT"),
    (r"WFDEI|WATCH Forcing Data", r"\bWFDEI\b|WATCH Forcing Data", None, "EXACT_PRODUCT"),
    (r"CERES.*EBAF", r"CERES.*EBAF", None, "EXACT_PRODUCT"),
    (r"Global Land Cover Characteristics|DB_GLCC_USGS", r"USGS_GLCC|Global Land Cover Characterization", None, "EXACT_PRODUCT"),
    (r"VIIRS Vegetation Health Product|VIIRS-VVHP", r"VIIRS.*Vegetation Health", None, "EXACT_PRODUCT"),
    (r"SMOPS.*Soil Moisture", r"SMOPS", None, "EXACT_PRODUCT"),
    (r"Coastal Change Analysis Program|C-CAP", r"C-CAP.*Land Cover|Coastal Change Analysis Program", None, "EXACT_PRODUCT"),
    (r"Hazard Mapping System Fire and Smoke|HMS-FireSmoke", r"Hazard Mapping System.*Fire|RW:c667617a-44e8-4181-b96d-f99bbe73c331", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"^Global Land Cover Classification\b|DB_Gbl_Lnd_Cvr_Class", r"ISLSCP II.*Global Land Cover Classifications?|umd_landcover_xdeg", None, "EXACT_PRODUCT"),
    (r"CPC Global Gridded Precipitation", r"CPC Global Unified.*Precipitation", None, "EXACT_PRODUCT"),
    (r"CPC Global Gridded Temperature", r"CPC Global Unified Temperature", None, "EXACT_PRODUCT"),
    (r"nClimGrid", r"nClimGrid", None, "EXACT_PRODUCT"),
    (r"Bathymetry NCEI", r"ETOPO|NCEI.*Bathymetry|GEBCO", None, "RELATED_PROXY_OR_SUBSET"),
    (r"NOAA Above Ground Forest Biomass", r"NBCD2000|Above Ground Forest Biomass", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Historical Land-Cover Change.*NCEI", r"Historical Land-Cover Change|HYDE|LUH2", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"AMSR & SSMI Topographic Moisture|DB_Natl_Sfc_Moisture_Datasets", r"Topographic Moisture Potential|AMSR.*Soil Moisture", None, "RELATED_PROXY_OR_SUBSET"),
    (r"Irrigated Cropland \(Google Earth Engine\)|DB_Irrigated_Cropland_GEE", r"GFSAD|Global Food Security-Support Analysis Data", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"Landsat NDVI 10 Year Average", r"Landsat.*NDVI.*Composite", None, "RELATED_PROXY_OR_SUBSET"),
    (r"Land Cover Institute.*USGS", r"USGS_GLCC|LCMAP|NLCD", None, "RELATED_PROXY_OR_SUBSET"),
    (r"Forest Biomass Mapping in CA and OR", r"Forest Biomass.*(?:California|Oregon|CA and OR)", None, "EXACT_PRODUCT"),

    # --- NASA NSITE / SNWG Portfolio Suites ---
    (r"\bHLS Solutions\b", r"\bHLSL30\b|\bHLSS30\b", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bOPERA Near-Global DIST Product Suite\b", r"OPERA_L3_DIST", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bOPERA Near-Global DSWx Product Suite\b", r"OPERA_L3_DSWX", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bOPERA North America DISP Product Suite\b", r"OPERA_L3_DISP", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bFIRMS-2G\b", r"^FIRMS\b|Fire Information for Resource Management", None, "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bGlobal Cloud Composites from SatCORPS\b", r"SatCORPS", None, "RELATED_PROXY_OR_SUBSET"),
    (r"VIIRS_SNPP_DayNightBand_At_Sensor_Radiance|Black Marble Nighttime", r"VNP46A1|Black Marble", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bICESat-2/ATLAS Quick Look Products\b", r"ATL(?:07|08|09|10|13|20)QL", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bMultisensor Worldwide Ocean Winds \(MWOW\) Products\b", r"MWOW_L3", "src:16_nasa_cmr", "EXACT_PRODUCT"),
    (r"\bSea Surface Salinity with Sea Ice Mask\b", r"SMAP_RSS_L3_SSS", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bTEMPO NRT and Enhanced Products\b", r"TEMPO_(?:NO2|HCHO|CLDO4|O3TOT)_L3", "src:16_nasa_cmr", "COMPATIBLE_EDITION_OR_COLLECTION"),
    (r"\bWater Quality Products\b", r"OLCIS3[AB]_L[34]", "src:16_nasa_cmr", "RELATED_PROXY_OR_SUBSET"),
    (r"\bAir Quality Sensors and Forecasts\b", r"TEMPO_NO2_L3|GEOS-?CF", None, "RELATED_PROXY_OR_SUBSET"),
]


def _classify_gap(row: pd.Series) -> str:
    bid = str(row["benchmark_id"])
    name = str(row["product_name"]).lower()
    prov = str(row["provider_or_source"]).lower()
    if bid == "07_nasa_snwg_cmr":
        return "RECENT_OR_NRT_CMR_RELEASE"
    if bid == "08_us_ghg_center":
        return "US_GHG_CENTER_SPECIALIZED_OR_HOSTED_GAP"
    if any(k in name or k in prov for k in ("pangaea", "figshare", "leicester", "nceo", "world fire atlas", "oco-2", "gosat", "jpl", "dahiti", "g-realm", "dhi", "worldwater", "hydroweb", "esrioceans", "14.1.1a", "landcover 2050", "satcorps", "glc-share")):
        return "TRUE_GAP_SPECIALIZED_OR_NON_API_PORTAL"
    if any(k in name for k in ("landfire", "lanid", "mirad", "mirca", "wfdei", "hms", "hazard mapping", "ca and or", "glsdem", "epa ch4", "climate model dem")):
        return "NATIONAL_OR_REGIONAL_ARCHIVE_GAP"
    return "TRUE_COVERAGE_GAP_OR_UNINDEXED_PORTAL"


_OBSOLETE_TITLE_RE = re.compile(
    r"\b(?:OBSOLETE RELEASE|superseded|deprecated)\b|\[deprecated\]",
    re.IGNORECASE,
)

_AUX_LAYER_TITLE_RE = re.compile(
    r"\b(?:confidence\s+in\s+classification|quality\s+flag|static\s+layers?|"
    r"distance[- ]to[- ]second|uncertainty\s+layer|error\s+map|input\b)\b",
    re.IGNORECASE,
)


def _extract_requested_epoch_years(text: str) -> set[str]:
    """Extract explicitly requested single-epoch product years, ignoring citations and brand tokens."""
    cleaned = re.sub(r"\bet\s+al\.?,?\s*(?:19|20)\d{2}\b", " ", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:natura\s*2000|mirca\s*2000|gmted\s*2010|nbcd\s*2000|ecoregions\s*2017)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(?:19|20)\d{2}\s*[-–/]\s*(?:19|20)\d{2}\b", " ", cleaned)
    return set(re.findall(r"\b(19[789]\d|20[0-3]\d)\b", cleaned))


def classify_match_specificity_tier(
    pname: str,
    matched_can: pd.Series | None,
    match_method: str,
    rule_tier: str | None,
) -> str:
    """Assign a mutually exclusive, structural match-specificity tier."""
    if matched_can is None or match_method == "NOT_FOUND":
        return "NOT_FOUND"
    if match_method in ("EXACT_DOI", "EXACT_RAW_ID", "EXACT_CMR_SHORTNAME"):
        return "EXACT_PRODUCT"
    if match_method == "RELATED_DOI":
        return "RELATED_PROXY_OR_SUBSET"

    tier = rule_tier or "COMPATIBLE_EDITION_OR_COLLECTION"
    m_title = str(matched_can.get("title") or "")
    m_geo = str(matched_can.get("geographic_coverage_class") or "")
    m_is_dep = bool(matched_can.get("is_superseded_edition")) or bool(_OBSOLETE_TITLE_RE.search(m_title))

    # Auxiliary confidence, quality-flag, static-mask, or input layers cannot receive strict primary-product credit
    if _AUX_LAYER_TITLE_RE.search(m_title) and not _AUX_LAYER_TITLE_RE.search(pname):
        return "RELATED_PROXY_OR_SUBSET"
    if re.search(r"\bglobal\b", pname, re.IGNORECASE) and (
        m_geo == "regional_or_local" or re.search(r"\b(?:arctic|antarctic)\b", m_title, re.IGNORECASE)
    ):
        return "RELATED_PROXY_OR_SUBSET"
    if re.search(r"\balaska\b", pname, re.IGNORECASE) and not re.search(r"\balaska\b", m_title, re.IGNORECASE):
        return "RELATED_PROXY_OR_SUBSET"
    req_yrs = _extract_requested_epoch_years(pname)
    mat_yrs = _extract_requested_epoch_years(m_title)
    if tier == "EXACT_PRODUCT" and req_yrs and mat_yrs and req_yrs.isdisjoint(mat_yrs):
        return "COMPATIBLE_EDITION_OR_COLLECTION"
    if tier == "EXACT_PRODUCT" and m_is_dep:
        return "COMPATIBLE_EDITION_OR_COLLECTION"
    return tier


def main() -> None:
    ext = pd.read_csv(EXT_PATH)
    all_df = pd.read_parquet(ALL_PATH)
    can_df = pd.read_parquet(CAN_PATH)
    exc_df = pd.read_parquet(EXC_PATH)

    can_by_key = can_df.drop_duplicates(subset=["canonical_product_key"]).set_index("canonical_product_key")
    can_by_rid = can_df.set_index("record_id")

    all_df = all_df.copy()
    all_df["_doi_low"] = all_df["doi"].fillna("").astype(str).str.lower().str.strip()
    all_df["_rdoi_low"] = all_df["related_dois"].fillna("").astype(str).str.lower()
    all_df["_raw_low"] = all_df["raw_id"].fillna("").astype(str).str.lower().str.strip()
    # Strict metadata header ONLY (raw_id | title | doi) — never noisy web URLs!
    all_df["_head"] = (
        all_df["raw_id"].fillna("").astype(str)
        + " | "
        + all_df["title"].fillna("").astype(str)
        + " | "
        + all_df["doi"].fillna("").astype(str)
    )
    all_df["_is_dep"] = (
        all_df["operational_maturity_class"].eq("deprecated_or_superseded")
        | all_df["is_superseded_edition"].fillna(False)
        | all_df["title"].fillna("").str.contains(_OBSOLETE_TITLE_RE, regex=True)
    ).astype(int)

    all_df["_is_aux"] = all_df["title"].fillna("").str.contains(_AUX_LAYER_TITLE_RE, regex=True).astype(int)
    all_df["_is_non_global"] = (
        all_df["geographic_coverage_class"].fillna("").isin(["regional_or_local", "hemispheric_or_pantropical"])
        | all_df["title"].fillna("").str.contains(r"\b(?:arctic|antarctic)\b", case=False, regex=True)
    ).astype(int)

    doi_map: dict[str, list[int]] = {}
    raw_map: dict[str, list[int]] = {}
    for idx, (d_v, r_v) in enumerate(zip(all_df["_doi_low"], all_df["_raw_low"])):
        if d_v:
            doi_map.setdefault(d_v, []).append(idx)
        if r_v:
            raw_map.setdefault(r_v, []).append(idx)

    exc_raw_map: dict[str, tuple[str, str]] = {}
    for _, er in exc_df.iterrows():
        rid_e = str(er.get("raw_id") or "").lower().strip()
        if rid_e and rid_e not in exc_raw_map:
            exc_raw_map[rid_e] = (str(er.get("source_id") or ""), str(er.get("exclusion_reason") or ""))

    def _pick_best_canonical(row_indices: list[int], pname_hint: str = "") -> tuple[pd.Series, int]:
        sub = all_df.iloc[row_indices].copy()
        sort_cols = ["_is_dep"]
        if not _AUX_LAYER_TITLE_RE.search(pname_hint):
            sort_cols.append("_is_aux")
        if re.search(r"\bglobal\b", pname_hint, re.IGNORECASE):
            sort_cols.append("_is_non_global")
        req_yrs = _extract_requested_epoch_years(pname_hint)
        if req_yrs:
            sub["_yr_miss"] = [
                0 if any(y in str(h) for y in req_yrs) else 1
                for h in sub["_head"].tolist()
            ]
            sort_cols.append("_yr_miss")
        sort_cols.append("source_authority_rank")
        sub = sub.sort_values(sort_cols)
        ckeys = list(dict.fromkeys(sub["canonical_product_key"].tolist()))
        best_ckey = ckeys[0]
        if best_ckey in can_by_key.index:
            return can_by_key.loc[best_ckey], len(ckeys)
        rid0 = sub.iloc[0]["record_id"]
        if rid0 in can_by_rid.index:
            return can_by_rid.loc[rid0], len(ckeys)
        return sub.iloc[0], len(ckeys)

    results: list[dict[str, Any]] = []
    for _, r in ext.iterrows():
        bid = str(r["benchmark_id"])
        pname = str(r["product_name"]).strip()
        raw_id_ext = str(r["raw_identifier"] if pd.notna(r["raw_identifier"]) else "").strip()
        doi_ext = str(r["doi"] if pd.notna(r["doi"]) else "").lower().strip()
        scope = str(r["scope_class"])

        matched_can: pd.Series | None = None
        n_matches = 0
        match_method = "NOT_FOUND"
        rule_tier: str | None = None
        excl_hit_info = ""

        # --- Tier 1: Exact DOI ---
        if doi_ext:
            if doi_ext in doi_map:
                matched_can, n_matches = _pick_best_canonical(doi_map[doi_ext])
                match_method = "EXACT_DOI"
            else:
                rdoi_hits = all_df.index[all_df["_rdoi_low"].str.contains(re.escape(doi_ext), regex=True)].tolist()
                if rdoi_hits:
                    matched_can, n_matches = _pick_best_canonical(rdoi_hits)
                    match_method = "RELATED_DOI"

        # --- Tier 2: Exact ID / CMR concept_id / CMR short_name ---
        if matched_can is None and raw_id_ext:
            tokens = [t.strip() for t in raw_id_ext.split("|") if t.strip()]
            for tok in tokens:
                t_low = tok.lower()
                if t_low in raw_map:
                    matched_can, n_matches = _pick_best_canonical(raw_map[t_low])
                    match_method = "EXACT_RAW_ID"
                    break
            if matched_can is None and bid == "07_nasa_snwg_cmr" and len(tokens) >= 2:
                sn = tokens[1]
                cmr_mask = (all_df["source_catalogue"] == "16_nasa_cmr") & (
                    all_df["_head"].str.contains(r"\b" + re.escape(sn) + r"\b", case=False, regex=True)
                )
                idxs = all_df.index[cmr_mask].tolist()
                if idxs:
                    matched_can, n_matches = _pick_best_canonical(idxs)
                    match_method = "EXACT_CMR_SHORTNAME"

            for tok in tokens:
                t_low = tok.lower()
                if t_low in exc_raw_map:
                    src_e, rsn_e = exc_raw_map[t_low]
                    excl_hit_info = f"{src_e} ({rsn_e})"
                    break

        # --- Tier 3: Curated Product-Family Signature ---
        if matched_can is None:
            probe_str = f"{pname} | {raw_id_ext}" if raw_id_ext else pname
            for item_rx, cat_head_rx, constraint, r_tier in SIGNATURE_RULES:
                if re.search(item_rx, probe_str, re.I):
                    mask = all_df["_head"].str.contains(cat_head_rx, case=False, regex=True)
                    if constraint:
                        if constraint.startswith("pool:"):
                            mask = mask & (all_df["provider_pool"] == constraint.split(":", 1)[1])
                        elif constraint.startswith("src:"):
                            srcs = constraint.split(":", 1)[1].split("|")
                            mask = mask & (all_df["source_catalogue"].isin(srcs))
                    idxs = all_df.index[mask].tolist()
                    if idxs:
                        matched_can, n_matches = _pick_best_canonical(idxs, pname_hint=pname)
                        match_method = "PRODUCT_FAMILY_SIGNATURE"
                        rule_tier = r_tier
                    break

        specificity_tier = classify_match_specificity_tier(pname, matched_can, match_method, rule_tier)

        if scope == "IN_SCOPE_OPEN_L3_L4_EO":
            if matched_can is not None:
                status = "FOUND_IN_CATALOGUE"
            elif excl_hit_info:
                status = "EXCLUDED_IN_PREPARATION"
            else:
                status = "MISSING_FROM_CATALOGUE"
        else:
            if excl_hit_info:
                status = "OUT_OF_SCOPE_CORRECTLY_EXCLUDED_IN_LOG"
            elif matched_can is not None:
                status = "OUT_OF_SCOPE_PRESENT_IN_CATALOGUE"
            else:
                status = "OUT_OF_SCOPE_CORRECTLY_ABSENT"

        results.append({
            **r.to_dict(),
            "validation_status": status,
            "match_method": match_method,
            "match_specificity_tier": specificity_tier,
            "canonical_match_count": n_matches,
            "matched_record_id": str(matched_can["record_id"]) if matched_can is not None else "",
            "matched_source_catalogue": str(matched_can["source_catalogue"]) if matched_can is not None else "",
            "matched_provider_pool": str(matched_can["provider_pool"]) if matched_can is not None else "",
            "matched_title": str(matched_can["title"]) if matched_can is not None else "",
            "matched_doi": str(matched_can["doi"] if pd.notna(matched_can["doi"]) else "") if matched_can is not None else "",
            "excluded_log_evidence": excl_hit_info,
            "gap_category": _classify_gap(r) if status in ("MISSING_FROM_CATALOGUE", "EXCLUDED_IN_PREPARATION") else "",
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(RES_DIR / "validation_matches.csv", index=False)

    gaps = res_df[
        (res_df["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO")
        & (res_df["validation_status"] != "FOUND_IN_CATALOGUE")
    ].copy()
    gaps.to_csv(RES_DIR / "gap_analysis.csv", index=False)

    sum_rows = []
    for bid, grp in res_df.groupby("benchmark_id", sort=True):
        in_sc = grp[grp["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO"]
        found = in_sc[in_sc["validation_status"] == "FOUND_IN_CATALOGUE"]
        n_exact = int((in_sc["match_specificity_tier"] == "EXACT_PRODUCT").sum())
        n_compat = int((in_sc["match_specificity_tier"] == "COMPATIBLE_EDITION_OR_COLLECTION").sum())
        n_related = int((in_sc["match_specificity_tier"] == "RELATED_PROXY_OR_SUBSET").sum())
        oos = grp[grp["scope_class"] != "IN_SCOPE_OPEN_L3_L4_EO"]
        oos_excl_log = oos[oos["validation_status"] == "OUT_OF_SCOPE_CORRECTLY_EXCLUDED_IN_LOG"]
        strict_recall = (100.0 * (n_exact + n_compat) / len(in_sc)) if len(in_sc) else 0.0
        broad_recall = (100.0 * len(found) / len(in_sc)) if len(in_sc) else 0.0
        sum_rows.append({
            "benchmark_id": bid,
            "benchmark_name": grp["benchmark_name"].iloc[0],
            "policy_framework": grp["policy_framework"].iloc[0],
            "total_extracted": len(grp),
            "out_of_scope_count": len(oos),
            "oos_verified_in_excluded_log": len(oos_excl_log),
            "in_scope_l3_l4_count": len(in_sc),
            "exact_product_count": n_exact,
            "compatible_edition_count": n_compat,
            "related_proxy_count": n_related,
            "found_in_catalogue": len(found),
            "missing_from_catalogue": len(in_sc) - len(found),
            "strict_product_recall_pct": round(strict_recall, 1),
            "broad_family_recall_pct": round(broad_recall, 1),
            "in_scope_recall_pct": round(broad_recall, 1),
        })
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(RES_DIR / "validation_summary.csv", index=False)

    def _tier_stats(sub_df: pd.DataFrame) -> tuple[int, int, int, int, int]:
        n_tot = len(sub_df)
        n_ex = int((sub_df["match_specificity_tier"] == "EXACT_PRODUCT").sum())
        n_co = int((sub_df["match_specificity_tier"] == "COMPATIBLE_EDITION_OR_COLLECTION").sum())
        n_re = int((sub_df["match_specificity_tier"] == "RELATED_PROXY_OR_SUBSET").sum())
        n_f = int((sub_df["validation_status"] == "FOUND_IN_CATALOGUE").sum())
        return n_ex, n_co, n_re, n_f, n_tot

    in_all = res_df[res_df["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO"]
    indep_9 = in_all[in_all["benchmark_id"] != "07_nasa_snwg_cmr"]
    # Strictly independent: excludes BOTH ingested API benchmarks (07_nasa_snwg_cmr via Source 16 & 08_us_ghg_center via Source 33)
    indep_8 = in_all[~in_all["benchmark_id"].isin(["07_nasa_snwg_cmr", "08_us_ghg_center"])]

    print("=" * 126)
    print(f"EXTERNAL POLICY-TO-EO VALIDATION SUMMARY (strictly audited against {len(can_df):,} canonical / {len(all_df):,} harmonized L3/L4 records)")
    print("=" * 126)
    print(
        sum_df[[
            "benchmark_id", "in_scope_l3_l4_count",
            "exact_product_count", "compatible_edition_count", "related_proxy_count",
            "missing_from_catalogue", "strict_product_recall_pct", "broad_family_recall_pct",
        ]].to_string(index=False)
    )
    print("-" * 126)
    for label, sub in [
        ("ALL 10 TRACKS (incl. CMR SNWG & US GHG Center)", in_all),
        ("9 TRACKS (excl. CMR SNWG; incl. US GHG Center)", indep_9),
        ("STRICTLY INDEPENDENT 8 TRACKS (excl. 07 & 08) ", indep_8),
    ]:
        nex, nco, nre, nf, ntot = _tier_stats(sub)
        print(
            f"{label}: strict={nex+nco:3d}/{ntot:3d} ({100*(nex+nco)/ntot:5.1f}%) "
            f"[exact={nex}, compat={nco}] | broad={nf:3d}/{ntot:3d} ({100*nf/ntot:5.1f}%) "
            f"[+related={nre}, miss={ntot-nf}]"
        )

    eoa_in = in_all[in_all["benchmark_id"] == "06_usgeo_eoa_2023"].copy()
    eoa_in["kpso_num"] = pd.to_numeric(eoa_in["kpso_impact_count"], errors="coerce").fillna(0)
    k_tot = eoa_in["kpso_num"].sum()
    k_strict = eoa_in.loc[
        eoa_in["match_specificity_tier"].isin(["EXACT_PRODUCT", "COMPATIBLE_EDITION_OR_COLLECTION"]),
        "kpso_num",
    ].sum()
    k_broad = eoa_in.loc[eoa_in["validation_status"] == "FOUND_IN_CATALOGUE", "kpso_num"].sum()
    print(
        f"USGEO EOA 2023 KPSO-WEIGHTED RECALL           : "
        f"strict={int(k_strict)}/{int(k_tot)} ({100*k_strict/k_tot:.1f}%) | "
        f"broad={int(k_broad)}/{int(k_tot)} ({100*k_broad/k_tot:.1f}%)"
    )
    print(f"\nRemaining honest gaps written ({len(gaps)} rows) -> {RES_DIR / 'gap_analysis.csv'}")


if __name__ == "__main__":
    main()
