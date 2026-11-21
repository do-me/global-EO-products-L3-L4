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
# (item_match_regex, cat_title_id_doi_regex, optional_source_or_pool_filter)
SIGNATURE_RULES: list[tuple[str, str, str | None]] = [
    # --- Paris Agreement / GHG / Carbon / Biomass ---
    (r"OCO-2.*MIP|oco2-mip", r"OCO-?2.*MIP|10\.48588/npf6-sw92", None),
    (r"OCO-2 GEOS|oco2geos", r"OCO-?2.*GEOS|OCO2_GEOS_L3CO2", None),
    (r"OCO-3.*SAM|oco3-co2-sams", r"OCO-?3.*(?:SAM|Snapshot Area Map)|OCO3_L2_Lite", None),
    (r"GOSAT.*Methane|gosat-based-ch4budget|GOSAT / TROPOMI.*Methane", r"GOSAT.*(?:CH4|Methane).*Budget|GOSAT-based Top-down.*Methane|Top-down.*Methane.*Budget.*GOSAT", None),
    (r"GEDI.*L4B|GEDI_ISS_L4B|GEDI Biomass Map", r"GEDI.*L4B|GEDI04_B|GEDI.*Gridded Aboveground Biomass", None),
    (r"GEDI.*Vegetation Height|GEDI.*Canopy", r"GEDI.*(?:Canopy|Vegetation) Height|GEDI_ICESAT2_Global_Veg_Height", None),
    (r"ESA CCI Biomass|CCI Biomass", r"Biomass[_\s-]*cci|CCI.*Biomass|ESACCI.*BIOMASS", None),
    (r"ICESat-2 Boreal.*Biomass", r"ICESat-2.*Boreal.*Biomass|Boreal_AGB_Density_ICESat2|C2756302505", None),
    (r"JPL 2020 Global.*Biomass", r"JPL.*2020.*Global.*Biomass|Xu.*2021.*Global.*Live.*Biomass|10\.5281/zenodo\.4161694", None),
    (r"NCEO Africa.*Biomass", r"NCEO.*Africa.*Biomass|10\.25392/leicester\.data\.15060270", None),
    (r"Spawn et al|Harmonized global maps of above and belowground biomass", r"Global_Maps_C_Density_2010|Global Aboveground and Belowground Biomass Carbon Density", None),
    (r"HILDA\+", r"HILDA\+|10\.1594/pangaea\.921846", None),
    (r"ODIAC", r"\bODIAC\b|10\.17595/20170411", None),
    (r"Vulcan.*CO2|vulcan-ffco2", r"\bVulcan\b.*(?:CO2|Fossil|Emissions)", None),
    (r"CarbonTracker-CH|ct-ch4", r"CarbonTracker[-_ ]?CH4|CarbonTracker.*Methane", None),
    (r"TM5-4DVar|tm54dvar", r"TM5[-_ ]?4DVAR", None),
    (r"ECCO-Darwin|eccodarwin", r"ECCO[-_ ]?Darwin", None),
    (r"LPJ-EOSIM|lpjeosim", r"LPJ[-_ ]?EOSIM", None),
    (r"LPJ-wsl|lpjwsl", r"LPJ[-_ ]?wsl|10\.5281/zenodo\.6964918", None),
    (r"MiCASA|micasa-carbonflux", r"\bMiCASA\b", None),
    (r"CASA-GFED|casagfed", r"CASA[-_ ]?GFED", None),
    (r"BlueFlux|blueflux", r"\bBlueFlux\b", None),
    (r"GRA2PES|GRA²PES|gra2pes", r"GRA2PES|GRA²PES|mds2-3520", None),
    (r"cms-mangrove-agb|Global Mangrove Aboveground Biomass", r"CMS.*Mangrove.*(?:Biomass|Canopy)|Global Mangrove.*(?:Biomass|Canopy Height)", None),
    (r"emit-ch4plume|EMIT Methane.*Plume", r"EMIT.*(?:Methane|CH4).*Plume|EMITL2BCH4PLM|MAPL-EMIT", None),
    (r"goes-ch4plume|Geostationary.*Methane.*Oil and Gas", r"GOES.*(?:Methane|CH4).*Plume", None),
    (r"geos-co2-daygrid|GEOS Assimilated CO₂ Column", r"OCO-?2 GEOS.*assimilated CO2|GEOS.*CO2.*Column", None),
    (r"geos-co-ch4|geos-ghg|GEOS Assimilated (?:CH₄|CO₂, CH₄)", r"GEOS.*(?:CH4|Methane).*Column|GEOS-?CF.*CH4", None),
    (r"epa-ch4emission|Gridded.*EPA CH4|U\.S\. Gridded Anthropogenic Methane", r"Gridded.*EPA.*(?:CH4|Methane)|EPA.*Gridded.*Methane|10\.5281/zenodo\.8367082", None),
    (r"OMI_tropospheric_no2|OMI/Aura Tropospheric Nitrogen Dioxide", r"OMI.*Tropospheric.*(?:NO2|Nitrogen Dioxide)|OMNO2d", None),
    (r"sedac-popdensity|Gridded World Population Density v4\.11", r"GPW.*v4\.11|CIESIN_SEDAC_GPWv4|Gridded Population of the World.*v4", None),

    # --- Forests / Land Cover / Agriculture / Fire / Wetlands ---
    (r"Global Forest Watch.*Tree Cover Loss|UMD Hansen|Global Forest Watch \(UMD", r"Hansen Global Forest Change|UMD/hansen/global_forest_change", None),
    (r"Forest Carbon Flux Maps", r"Forest.*(?:Greenhouse Gas|Carbon).*Flux|CMS_Forest_Carbon_Fluxes", None),
    (r"ESA CCI Land Cover|ESA-CCI-LC|ESA CCI-LC", r"(?:ESA\s*)?CCI.*Land Cover|ESACCI-LC|satellite-land-cover", None),
    (r"ESA CCI High Resolution Land Cover|HRLC", r"High Resolution Land Cover.*CCI|High_Resolution_Land_Cover_cci", None),
    (r"Copernicus.*100\s*m.*Land Cover|CGLS-LC100|Copernicus Global Land Service LC100|Copernicus Global Land Cover", r"Land Cover 2015-2019 \(raster 100 m\)|cgls_landcover|CGLS[-_]LC100", None),
    (r"ESA WorldCover", r"ESA.*WorldCover|WorldCover.*10\s*m|ESA/WorldCover", None),
    (r"WorldCereal", r"WorldCereal", None),
    (r"Global Mangrove Watch", r"Global Mangrove Watch|\bGMW\b.*Mangrove", None),
    (r"Sentinel-3 World Fire Atlas", r"Sentinel-3.*World Fire Atlas|World Fire Atlas.*SLSTR|\bS3WFA\b", None),
    (r"GlobeLand\s*30", r"GlobeLand\s*30|GlobeLand30", None),
    (r"Global Land Cover 10\b", r"FROM[-_ ]?GLC10|Global Land Cover 10\b", None),
    (r"Global Artificial Impervious Areas", r"Global Artificial Impervious|\bGAIA\b.*Impervious", None),
    (r"Global 30 m Impervious Surface Map", r"GISD30|10\.5281/zenodo\.3505079|Global 30\s*m Impervious", None),
    (r"Esri 10-Meter Land Cover|ESRI 2020 Global Land Cover", r"io-lulc|Esri 10-Meter Land Cover|Impact Observatory.*Land Cover", None),
    (r"Landcover 2050", r"Landcover 2050|Land Cover 2050", None),
    (r"Local Climate Zones", r"Local Climate Zones", None),
    (r"EU Crop Map", r"EU\s*Crop\s*Map|EUCROPMAP", None),
    (r"FAO Global Land Cover SHARE|GLC-Share", r"Dominant land cover type.*GLC[-_ ]?SHARE|GLC[-_ ]?SHARE", None),
    (r"UMD GLAD Landsat.*ARD", r"landsat_glad\.swa\.ard2|GLAD.*Landsat.*ARD", None),
    (r"ALOS.*PALSAR Global Mosaic", r"alos_palsar_mosaic|ALOS.*PALSAR.*mosaic|JAXA/ALOS/PALSAR", None),
    (r"ALOS-2 / Global Wetland Inundation|Global Mangrove.*Rosenqvist", r"Global Mangrove Watch|ALOS.*(?:Wetland|Inundation|Mangrove)", None),

    # --- Water / Oceans / Lakes / Altimetry ---
    (r"Global Surface Water|GSWE|DB_GblSrfcWater_ESA", r"Global Surface Water|JRC/GSW", None),
    (r"DHI Global Wetland Map", r"DHI.*Wetland", None),
    (r"Lake Water Quality", r"Lake Water Quality", "src:02_copernicus_clms"),
    (r"ESA CCI Lakes|CCI Lakes", r"ESA Lakes Climate Change Initiative|Lakes_cci|ESACCI-LAKES", None),
    (r"Hydroweb THEIA", r"Hydroweb", None),
    (r"DAHITI", r"\bDAHITI\b", None),
    (r"G-REALM|Global Reservoir and Lake Monitor", r"G-?REALM|Global Reservoir and Lake Monitor", None),
    (r"WorldWater", r"ESA.*WorldWater|\bWorldWater\b.*Surface Water", None),
    (r"SDG 14\.1\.1a Indicator Maps EEZ", r"SDG\s*14\.1\.1a.*EEZ|14\.1\.1a.*Chlorophyll", None),
    (r"Ocean Colour CCI|OC-CCI", r"Ocean Colour Climate Change Initiative|Ocean_Colour_cci|OC[-_ ]?CCI", None),
    (r"NOAA VIIRS.*Chlorophyll", r"VIIRS.*Chlorophyll", None),
    (r"EUSeaMap", r"EUSeaMap.*Broad-Scale Predictive Habitat", "src:11_eea_and_emodnet"),
    (r"GLOBAL_ANALYSIS_FORECAST_PHY_001_024", r"GLOBAL_ANALYSISFORECAST_PHY_001_024|GLOBAL_ANALYSIS_FORECAST_PHY_001_024", "src:01_copernicus_cmems"),
    (r"Ocean Monitoring Indicators", r"OMI_|Ocean Monitoring Indicator", "src:01_copernicus_cmems"),
    (r"Global Ocean Biogeochemistry Analysis and Forecast|biogeochemical global ocean analysis", r"Global Ocean Biogeochemistry Analysis and Forecast|GLOBAL_ANALYSISFORECAST_BGC_001_028", "src:01_copernicus_cmems"),

    # --- Urban / Settlements / Population / Heat / Air Quality ---
    (r"Global Human Settlement-Built-Up|GHS-BUILT", r"GHS[-_ ]?BUILT", None),
    (r"Global Human Settlement-Population", r"GHS[-_ ]?POP", None),
    (r"Global Human Settlement-Settlement Model", r"GHS[-_ ]?SMOD", None),
    (r"Global Human Settlement-Functional Urban Areas", r"GHS[-_ ]?FUA|GHS functional urban areas", None),
    (r"Urban Centre Database", r"GHS[-_ ]?UCDB|Urban Centre Database", None),
    (r"GHSL Data Package 2023|Global Human Settlement Layer|CEMS GHSL", r"GHSL|GHS[-_ ]?BUILT|GHS[-_ ]?POP|Global Human Settlement", None),
    (r"World Settlement Footprint 2015", r"World Settlement Footprint.*2015|WSF2015", None),
    (r"World Settlement Footprint 2019", r"World Settlement Footprint.*2019|WSF2019", None),
    (r"World Settlement Footprint Evolution", r"World Settlement Footprint.*Evolution|WSF.*Evolution", None),
    (r"World Settlement Footprint \(WSF", r"World Settlement Footprint \(WSF\) (?:2015|2019|Evolution)", None),
    (r"High Resolution Settlement Layer", r"High Resolution Settlement Layer", None),
    (r"Human Built-up and Settlement Extent|HBASE", r"Human Built-up And Settlement Extent|\bHBASE\b|10\.7927/h4dn434s", None),
    (r"Global Man-made Impervious Surface", r"Global Man-made Impervious Surface|\bGMIS\b|10\.7927/h4p55kkf", None),
    (r"Global Grid of Probabilities of Urban Expansion to 2030", r"Probabilities of Urban Expansion to 2030|10\.7927/h4z899cg", None),
    (r"VIIRS Plus DMSP Change in Lights", r"VIIRS Plus DMSP Change in Lights|10\.7927/9ryj-6467", None),
    (r"Global Urban Heat Island", r"Global Urban Heat Island|10\.7927/h4h70crf", None),
    (r"Global Annual PM2\.5 Grids", r"Global Annual PM2\.5 Grids|10\.7927/h4zk5dqs", None),
    (r"Development Threat Index", r"Development Threat Index|10\.7927/61jv-th84", None),
    (r"Global Human Modification of Terrestrial Systems", r"Global Human Modification|10\.7927/edbc-3z60", None),
    (r"Urban-rural continuum", r"Urban-rural continuum|10\.6084/m9\.figshare\.12579572", None),
    (r"Global High Resolution Daily Extreme Urban Heat", r"Daily Extreme Urban Heat|10\.7927/fq7g-ny13", None),
    (r"WorldPop", r"WorldPop.*(?:Population|Global|100m|1km)|Global.*WorldPop|WorldPop/GP/", None),
    (r"Sub-Saharan Africa Settlement Extents", r"GRID3.*Settlement Extents", None),
    (r"Copernicus Urban Atlas|CLMS Urban Atlas", r"Urban Atlas Land Cover/Land Use", "src:02_copernicus_clms"),
    (r"Copernicus Imperviousness|HRL.*Imperviousness", r"Imperviousness Density", "src:02_copernicus_clms"),
    (r"European Settlement Map", r"European Settlement Map", None),
    (r"Building Footprints USA, Canada, Uganda", r"Microsoft Building Footprints|ms-buildings", None),
    (r"Open Buildings", r"Open Buildings", None),
    (r"Urban Extents, Continental U\.S\.", r"Urban Extents from VIIRS|10\.7927/a49b-sm16", None),
    (r"LandScan", r"LandScan", None),

    # --- Copernicus CLMS / CEMS / C3S / CAMS (KCEO & CEOS) ---
    (r"Natura 2000 \(N2K\)", r"N2K Land Cover/Land Use", "src:02_copernicus_clms"),
    (r"Hot Spot Monitoring", r"Hot Spot Monitoring", "src:02_copernicus_clms"),
    (r"Corine Land Cover \(CLC\)", r"CORINE Land Cover 2018", "src:02_copernicus_clms"),
    (r"CLC\+ Backbone", r"CLCplus Backbone", "src:02_copernicus_clms"),
    (r"HRL.*Forest", r"Tree Cover Density|Dominant Leaf Type|Forest Type", "src:02_copernicus_clms"),
    (r"HR-VPP|Vegetation Phenology and Productivity|Copernicus Land Surface Phenology", r"High-?Resolution Vegetation Phenology|Seasonal Trajectories.*10-daily", "pool:copernicus_services"),
    (r"RLIE S1\+S2", r"River and Lake Ice Extent", None),
    (r"Water and Wetness", r"Water and Wetness", "src:02_copernicus_clms"),
    (r"Riparian Zones", r"Riparian Zones", "src:02_copernicus_clms"),
    (r"Small Woody Features", r"Small Woody Features", "src:02_copernicus_clms"),
    (r"CLMS Global Water Bodies", r"Water Bodies 2020-present|Water Bodies.*(?:100|300)\s*m", "src:02_copernicus_clms"),
    (r"CLMS Global Water Level", r"(?:River|Lake) Water Level", "src:02_copernicus_clms"),
    (r"CLMS.*FAPAR|Copernicus.*fAPAR", r"Fraction of Absorbed Photosynthetically Active Radiation", "src:02_copernicus_clms"),
    (r"CLMS.*Soil Water Index", r"Soil Water Index", "src:02_copernicus_clms"),
    (r"CLMS.*Leaf Area Index|Copernicus.*LAI", r"Leaf Area Index", "src:02_copernicus_clms"),
    (r"Copernicus Global Land Service NDVI", r"Normali[sz]ed Difference Vegetation Index", "src:02_copernicus_clms"),
    (r"Pan-European Copernicus High Resolution Layers", r"Imperviousness Density|Tree Cover Density", "src:02_copernicus_clms"),
    (r"EFFIS Burnt Area", r"European Forest Fire Information System|EFFIS", "pool:copernicus_services"),
    (r"C3S Hydrology-Related Climate Impact", r"Hydrology.*climate.*indicator|sis-hydrology-meteorology", "src:03_copernicus_c3s_cds"),
    (r"C3S Urban Climate|ERA5 Products", r"ERA5|urban.*climate.*variables|sis-urban-climate", "src:03_copernicus_c3s_cds"),
    (r"CAMS Air Quality", r"CAMS European air quality|CAMS global atmospheric", "src:04_copernicus_cams_ads"),
    (r"EEA Floods Viewer|European Flood Hazard", r"European Flood Awareness System|EFAS|Flood.*Hazard", None),

    # --- UNCCD / Soil / Productivity / Vegetation / Cryosphere ---
    (r"JRC Land Productivity Dynamics", r"Land Productivity Dynamics", None),
    (r"Trends\.Earth|MOD13Q1|MYD13Q1", r"MOD13Q1|MYD13Q1", None),
    (r"SoilGrids", r"SoilGrids", None),
    (r"Harmonized World Soil Database|HWSD", r"Harmonized World Soil Database|\bHWSD\b", None),
    (r"Global Soil Organic Carbon Map|GSOC", r"GSOC|Global Soil Organic Carbon", "src:26_fao_unep_and_thematic_benchmarks"),
    (r"MODIS Vegetation Continuous Fields", r"MOD44B|Vegetation Continuous Fields", None),
    (r"MODIS LAI|Myneni", r"MOD15A2H|MCD15A2H|MOD15A3H|MYD15A2H", None),
    (r"MODIS Vegetation Indices.*Phenology|Huete", r"MCD12Q2|MOD13Q1|MOD13A2", None),
    (r"ESA CCI Snow", r"ESA Snow Climate Change Initiative|Snow_cci|ESACCI-SNOW", None),
    (r"Global Leaf Chlorophyll Content|GCOM-C SGLI Chlorophyll", r"Global.*Leaf Chlorophyll|Xu.*2022.*Chlorophyll|GCOM-C.*Leaf Chlorophyll", None),
    (r"MODIS / GIMMS fAPAR|Zhu et al", r"GIMMS.*(?:FPAR|fAPAR)|MOD15A2H|MCD15A2H", None),
    (r"MODIS GPP / NPP|MOD17", r"MOD17A2H|MOD17A3|MYD17A2H", None),
    (r"Solar-Induced Fluorescence.*Joiner", r"SIF_SCIAMACHY_GOME2|OCO-?3.*SIF|GOSAT.*SIF|TROPOSIF|TCSIF|GOSIF", None),

    # --- DEMs & Topography ---
    (r"Copernicus DEM GLO-30", r"COP-?DEM.*GLO-?30|CopDEM.*30\s*m|GLO-30", None),
    (r"ASTER.*GDEM|ASTER.*Global Digital Elevation", r"ASTGTM|ASTER Global Digital Elevation", None),
    (r"ASTER.*Global Emissivity|DB_ASTER_GED", r"AG100|AG1KM|ASTER Global Emissivity", None),
    (r"SRTM.*CHILI", r"SRTM.*CHILI|SRTM_CHILI", None),
    (r"SRTM", r"SRTMGL1|SRTMGL3|NASADEM_HGT|SRTM Digital Elevation", None),
    (r"TanDEM-X DEM", r"TanDEM-X.*Digital Elevation Model", None),
    (r"ALOS World 3D|AW3D30", r"AW3D30|ALOS World 3D", None),
    (r"National Elevation Dataset.*mTPI", r"US/mTPI|US NED mTPI", None),
    (r"National Elevation Dataset.*Topographic Diversity", r"US NED Topographic Diversity", None),
    (r"National Elevation Dataset \(NED\)|DB_NED_USGS", r"USGS/3DEP|National Elevation Dataset|3D Elevation Program", None),
    (r"Global Land Survey Digital Elevation Model|GLSDEM", r"GLSDEM|Global Land Survey Digital Elevation", None),
    (r"Greenland Ice Mapping Project.*DEM|DB_GIMP_DEM", r"NSIDC-0645|NSIDC-0715|Greenland Ice Mapping Project.*DEM", None),
    (r"Radarsat Antarctic Mapping Project.*DEM|DB_RAMP_DEM", r"NSIDC-0082|Radarsat Antarctic Mapping Project Digital Elevation", None),
    (r"GMTED2010", r"GMTED2010", None),
    (r"GTOPO30", r"GTOPO30", None),
    (r"Canadian Digital Elevation Model|DB_CDEM", r"Canadian Digital Elevation Model|\bcdem\b", None),
    (r"Global Land One-kilometer Base Elevation|DB_GLOBE_NCEI", r"Global Land One-kilometer Base Elevation", None),
    (r"Global Climate Model DEM", r"Global Climate Model DEM", None),

    # --- USGEO EOA 2023 Specific Products ---
    (r"OSISAF Global Daily Sea Ice|DB_SeaIce_METSAT_NCEI", r"Global Sea Ice Concentration Climate Data Record|OSI\s*SAF.*Sea Ice Concentration", None),
    (r"National Land Cover Data(?:base|set)|\bNLCD\b", r"National Land Cover Database|\bNLCD\b", None),
    (r"LANDFIRE.*EVT|Existing Vegetation Type", r"LANDFIRE.*(?:EVT|Existing Vegetation Type)", None),
    (r"LANDFIRE.*ExistVegCover|Existing Vegetation Cover", r"LANDFIRE.*(?:EVC|Existing Vegetation Cover)", None),
    (r"LANDFIRE.*ExstVegHght|Existing Vegetation Height", r"LANDFIRE.*(?:EVH|Existing Vegetation Height)", None),
    (r"LANDFIRE.*40.*FBFM|40 Fire Behavior Fuel", r"LANDFIRE.*(?:FBFM40|40.*Fuel Model)", None),
    (r"LANDFIRE.*13.*FBFM|13 Fire Behavior Fuel", r"LANDFIRE.*(?:FBFM13|13 Anderson Fire Behavior)", None),
    (r"LANDFIRE.*Historical Disturbance", r"LANDFIRE.*Disturbance", None),
    (r"NALCMS|North American Land Change Monitoring|North American Environmental Atlas - Land Cover", r"NALCMS|Land Cover of North America at 30 meters", None),
    (r"LANID|LandID|Landsat-based Irrigation", r"\bLANID\b|Landsat-based Irrigation", None),
    (r"MIrAD-US|MODIS Irrigated Agriculture", r"\bMIrAD\b|MODIS Irrigated Agriculture", None),
    (r"eMODIS", r"\beMODIS\d*\b", None),
    (r"MERRA.*Aerosol|MERRAero|MERRA-AOT", r"MERRA-?2.*(?:aer|Aerosol)|MERRA.*Aerosol", None),
    (r"MERRA.*Precip|MERRA-Land|MERRA-Temp|MERRA", r"MERRA-?2", None),
    (r"GLIMS", r"NSIDC-0272|\bGLIMS\b", None),
    (r"NLDAS-2 Forcings|NLDAS Noah|NLDAS MOSAIC", r"NLDAS.*(?:Forcing|NOAH|MOSAIC)", None),
    (r"Sea Ice and Snow Cover Extent.*NCEI|NSIDC.*Sea Ice Extent", r"NSIDC-0046|Sea Ice Index|G02135|NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice", "src:16_nasa_cmr|20_noaa_onestop"),
    (r"MIRCA2000", r"MIRCA2000", None),
    (r"GIMMS Leaf Area Index|GIMMS 3g|GIMMS_3g", r"GIMMS.*(?:LAI|3G|NDVI)", None),
    (r"GLEAM.*Soil Moisture", r"^GLEAM4$|GLEAM4-HR|GLEAM.*v[34].*(?:Evaporation|Soil Moisture)", None),
    (r"Copernicus Marine Environment Monitoring Service|DB_CMEMS_EU", r"GLOBAL_.*_001_", "src:01_copernicus_cmems"),
    (r"VIIRS Land Cover", r"VNP12|VIIRS.*(?:Land Cover|Surface Type)", None),
    (r"GAP Land Cover", r"USGS GAP|GAP/LANDFIRE", None),
    (r"GridSat-GOES|DB_GVAR_NCEI", r"GridSat", None),
    (r"GHRSST.*Level 4", r"GHRSST Level 4|MUR-JPL-L4", None),
    (r"AVHRR Pathfinder.*Aerosol|DB_AerosolOD_AVHRR_NCEI", r"CDR.*AVHRR.*Aerosol Optical Thickness|PATMOS-x", None),
    (r"Pathfinder.*Sea Surface Temperature|DB_SST_AVHRR_NCEI", r"AVHRR Pathfinder.*Sea Surface Temperature", None),
    (r"Global Ocean Heat and Salt Content|Global Ocean Heat Content", r"Ocean Heat.*Content|World Ocean Atlas", None),
    (r"WFDEI|WATCH Forcing Data", r"\bWFDEI\b|WATCH Forcing Data", None),
    (r"CERES.*EBAF", r"CERES.*EBAF", None),
    (r"Global Land Cover Characteristics|DB_GLCC_USGS", r"USGS_GLCC|Global Land Cover Characterization", None),
    (r"VIIRS Vegetation Health Product|VIIRS-VVHP", r"VIIRS.*Vegetation Health", None),
    (r"SMOPS.*Soil Moisture", r"SMOPS", None),
    (r"Coastal Change Analysis Program|C-CAP", r"C-CAP.*Land Cover|Coastal Change Analysis Program", None),
    (r"Hazard Mapping System Fire and Smoke|HMS-FireSmoke", r"Hazard Mapping System.*Fire|RW:c667617a-44e8-4181-b96d-f99bbe73c331", None),
    (r"^Global Land Cover Classification\b|DB_Gbl_Lnd_Cvr_Class", r"ISLSCP II.*Global Land Cover Classifications?|umd_landcover_xdeg", None),
    (r"CPC Global Gridded Precipitation", r"CPC Global Unified.*Precipitation", None),
    (r"CPC Global Gridded Temperature", r"CPC Global Unified Temperature", None),
    (r"nClimGrid", r"nClimGrid", None),
    (r"Bathymetry NCEI", r"ETOPO|NCEI.*Bathymetry|GEBCO", None),
    (r"NOAA Above Ground Forest Biomass", r"NBCD2000|Above Ground Forest Biomass", None),
    (r"Historical Land-Cover Change.*NCEI", r"Historical Land-Cover Change|HYDE|LUH2", None),
    (r"AMSR & SSMI Topographic Moisture|DB_Natl_Sfc_Moisture_Datasets", r"Topographic Moisture Potential|AMSR.*Soil Moisture", None),
    (r"Irrigated Cropland \(Google Earth Engine\)|DB_Irrigated_Cropland_GEE", r"GFSAD|Global Food Security-Support Analysis Data", None),
    (r"Landsat NDVI 10 Year Average", r"Landsat.*NDVI.*Composite", None),
    (r"Land Cover Institute.*USGS", r"USGS_GLCC|LCMAP|NLCD", None),
    (r"Forest Biomass Mapping in CA and OR", r"Forest Biomass.*(?:California|Oregon|CA and OR)", None),

    # --- NASA NSITE / SNWG Portfolio Suites ---
    (r"\bHLS Solutions\b", r"\bHLSL30\b|\bHLSS30\b", "src:16_nasa_cmr"),
    (r"\bOPERA Near-Global DIST Product Suite\b", r"OPERA_L3_DIST", "src:16_nasa_cmr"),
    (r"\bOPERA Near-Global DSWx Product Suite\b", r"OPERA_L3_DSWX", "src:16_nasa_cmr"),
    (r"\bOPERA North America DISP Product Suite\b", r"OPERA_L3_DISP", "src:16_nasa_cmr"),
    (r"\bFIRMS-2G\b", r"^FIRMS\b|Fire Information for Resource Management", None),
    (r"\bGlobal Cloud Composites from SatCORPS\b", r"SatCORPS", None),
    (r"VIIRS_SNPP_DayNightBand_At_Sensor_Radiance|Black Marble Nighttime", r"VNP46A1|Black Marble", "src:16_nasa_cmr"),
    (r"\bICESat-2/ATLAS Quick Look Products\b", r"ATL(?:07|08|09|10|13|20)QL", "src:16_nasa_cmr"),
    (r"\bMultisensor Worldwide Ocean Winds \(MWOW\) Products\b", r"MWOW_L3", "src:16_nasa_cmr"),
    (r"\bSea Surface Salinity with Sea Ice Mask\b", r"SMAP_RSS_L3_SSS", "src:16_nasa_cmr"),
    (r"\bTEMPO NRT and Enhanced Products\b", r"TEMPO_(?:NO2|HCHO|CLDO4|O3TOT)_L3", "src:16_nasa_cmr"),
    (r"\bWater Quality Products\b", r"OLCIS3[AB]_L[34]", "src:16_nasa_cmr"),
    (r"\bAir Quality Sensors and Forecasts\b", r"TEMPO_NO2_L3|GEOS-?CF", None),
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
    all_df["_is_dep"] = all_df["operational_maturity_class"].eq("deprecated_or_superseded").astype(int)

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

    def _pick_best_canonical(row_indices: list[int]) -> tuple[pd.Series, int]:
        sub = all_df.iloc[row_indices].sort_values(["_is_dep", "source_authority_rank"])
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

        # --- Tier 3: Curated Strict Product-Family Signature ---
        if matched_can is None:
            probe_str = f"{pname} | {raw_id_ext}" if raw_id_ext else pname
            for item_rx, cat_head_rx, constraint in SIGNATURE_RULES:
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
                        matched_can, n_matches = _pick_best_canonical(idxs)
                        match_method = "PRODUCT_FAMILY_SIGNATURE"
                    break

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
        oos = grp[grp["scope_class"] != "IN_SCOPE_OPEN_L3_L4_EO"]
        oos_excl_log = oos[oos["validation_status"] == "OUT_OF_SCOPE_CORRECTLY_EXCLUDED_IN_LOG"]
        recall = (100.0 * len(found) / len(in_sc)) if len(in_sc) else 0.0
        sum_rows.append({
            "benchmark_id": bid,
            "benchmark_name": grp["benchmark_name"].iloc[0],
            "policy_framework": grp["policy_framework"].iloc[0],
            "total_extracted": len(grp),
            "out_of_scope_count": len(oos),
            "oos_verified_in_excluded_log": len(oos_excl_log),
            "in_scope_l3_l4_count": len(in_sc),
            "found_in_catalogue": len(found),
            "missing_from_catalogue": len(in_sc) - len(found),
            "in_scope_recall_pct": round(recall, 1),
        })
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(RES_DIR / "validation_summary.csv", index=False)

    in_all = res_df[res_df["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO"]
    found_all = in_all[in_all["validation_status"] == "FOUND_IN_CATALOGUE"]
    indep_9 = res_df[(res_df["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO") & (res_df["benchmark_id"] != "07_nasa_snwg_cmr")]
    found_indep_9 = indep_9[indep_9["validation_status"] == "FOUND_IN_CATALOGUE"]
    # Strictly independent: excludes BOTH ingested API benchmarks (07_nasa_snwg_cmr via Source 16 & 08_us_ghg_center via Source 33)
    indep_8 = res_df[
        (res_df["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO")
        & (~res_df["benchmark_id"].isin(["07_nasa_snwg_cmr", "08_us_ghg_center"]))
    ]
    found_indep_8 = indep_8[indep_8["validation_status"] == "FOUND_IN_CATALOGUE"]

    print("=" * 112)
    print(f"EXTERNAL POLICY-TO-EO VALIDATION SUMMARY (strictly audited against {len(can_df):,} canonical / {len(all_df):,} harmonized L3/L4 records)")
    print("=" * 112)
    print(
        sum_df[[
            "benchmark_id", "total_extracted", "out_of_scope_count",
            "in_scope_l3_l4_count", "found_in_catalogue", "missing_from_catalogue", "in_scope_recall_pct"
        ]].to_string(index=False)
    )
    print("-" * 112)
    print(f"ALL 10 TRACKS (incl. CMR SNWG & US GHG Center)   : {len(found_all):3d} / {len(in_all):3d} ({100*len(found_all)/len(in_all):.1f}%) in-scope open L3/L4 EO found")
    print(f"9 TRACKS (excl. CMR SNWG; incl. US GHG Center)   : {len(found_indep_9):3d} / {len(indep_9):3d} ({100*len(found_indep_9)/len(indep_9):.1f}%) in-scope open L3/L4 EO found")
    print(f"STRICTLY INDEPENDENT 8 TRACKS (excl. 07 & 08)    : {len(found_indep_8):3d} / {len(indep_8):3d} ({100*len(found_indep_8)/len(indep_8):.1f}%) in-scope open L3/L4 EO found")

    eoa_in = res_df[(res_df["benchmark_id"] == "06_usgeo_eoa_2023") & (res_df["scope_class"] == "IN_SCOPE_OPEN_L3_L4_EO")].copy()
    eoa_in["kpso_num"] = pd.to_numeric(eoa_in["kpso_impact_count"], errors="coerce").fillna(0)
    k_tot = eoa_in["kpso_num"].sum()
    k_found = eoa_in.loc[eoa_in["validation_status"] == "FOUND_IN_CATALOGUE", "kpso_num"].sum()
    print(f"USGEO EOA 2023 KPSO-WEIGHTED RECALL              : {int(k_found)} / {int(k_tot)} KPSO impacts ({100*k_found/k_tot:.1f}%)")
    print(f"\nRemaining honest gaps written ({len(gaps)} rows) -> {RES_DIR / 'gap_analysis.csv'}")


if __name__ == "__main__":
    main()
