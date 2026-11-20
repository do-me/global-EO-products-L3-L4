#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2.0",
#   "pyarrow>=15.0.0",
# ]
# ///
"""
04_validation_and_review/validate_catalogue_benchmarks.py
=========================================================
Automated quality assurance & benchmark validation suite for the Global L3/L4 EO Product Catalogue.

Runs:
1. Internal QA & Provenance Suite (`25` structural, scientific, and regression assertions against `03_harmonization/data/`).
2. External Benchmark 1: Curated EU Policy & Operational Service Product Families (`86` benchmark families).
3. External Benchmark 2: GCOS / CEOS-CGMS WGClimate Essential Climate Variable (ECV) thematic coverage (`40` ECV groups).
4. External Benchmark 3: Specified Regression & Holdout Case Suite (`79` cases: `55` core + `24` holdout/regression).

Outputs:
- `04_validation_and_review/eu_policy_products_benchmark_audit.csv`
- `04_validation_and_review/gcos_ecv_domain_coverage.csv`
- `04_validation_and_review/independent_product_version_audit.csv`
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
HARM_DIR = REPO_ROOT / "03_harmonization" / "data"
PREP_DIR = REPO_ROOT / "02_preparation" / "data"
RAW_DIR = REPO_ROOT / "01_raw"


# ---------------------------------------------------------------------------
# Benchmark 1 Definitions: Curated EU Policy & Operational Service Product Families
# ---------------------------------------------------------------------------
EU_POLICY_PRODUCTS_BENCHMARK: list[tuple[str, str, str, str | None, str, str]] = [
    # (service_or_domain, product_name, scope_classification, source_prefix_hint, regex_pattern, notes)
    ("CLMS", "CLCplus Backbone (CLC+ Backbone, 10 m)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"clcplus backbone|clc\+\s*backbone", "2018, 2021, 2023 EU 10 m backbone products indexed in CLMS"),
    ("CLMS", "CLCplus Core (LULUCF instance, 100 m)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"clcplus core|clcplus", "CLCplus Core LULUCF / Core Engine products indexed in CLMS"),
    ("CLMS", "CORINE Land Cover (CLC 1990-2018; 2024 forthcoming)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"corine land cover", "CLC 1990-2018 & change layers indexed; CLC 2024 pending public release"),
    ("CLMS", "HRL Grasslands (GRA) & Grassland Mowing Events (GRAME)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"grassland", "HRL Grasslands & phenology/mowing related CLMS layers indexed"),
    ("CLMS", "HRL Small Woody Features (SWF) / Small Landscape Features", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"small woody features|small landscape features", "SWF 2015, 2018, 2021 (5 m / 100 m) indexed in CLMS"),
    ("CLMS", "HRL Croplands - Crop Types (CT) & Cropping Patterns (CP)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"crop types|cropping patterns|high resolution layer croplands|cropping seasons yearly|main crop emergence|main crop harvest|main crop duration|secondary crops|fallow land presence", "Crop Types & Cropping Patterns (9 pan-European 10 m layers: CT, Cropping Seasons, Emergence, Harvest, Duration, Secondary Crops, Fallow Land) indexed in CLMS"),
    ("CLMS", "HRL Imperviousness (IMD, IMC, Impervious Built-Up IBU)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"imperviousness|impervious built-up", "IMD, IMC, IBU 2006-2024 (10 m / 20 m / 100 m) indexed in CLMS"),
    ("CLMS", "HRL Tree Cover & Forests (TCD, FTY, DLT, Tree Cover Change)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"tree cover density|forest type|dominant leaf type|tree cover presence", "TCD, FTY, DLT & changes indexed in CLMS"),
    ("CLMS", "HRL Vegetated Land Cover Characteristics (VLCC)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"tree cover density|grassland|crop types|small woody", "HRL VLCC portfolio (Forest, Grassland, Cropland, SWF) indexed as component products"),
    ("CLMS", "HRL Water and Wetness (WAW)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"water and wetness", "WAW 2009-2018 & yearly series indexed in CLMS"),
    ("CLMS", "High Resolution Vegetation Phenology & Productivity (HR-VPP / PPI)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"vegetation phenology and productivity|plant phenology index|seasonal trajectories", "HR-VPP VPP Parameters, Seasonal Trajectories, VI, QFLAG indexed in CLMS"),
    ("CLMS", "Urban Atlas (LCLU, Building Heights, Street Tree Layer STL)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"urban atlas|street tree layer|building height", "Urban Atlas 2006/2012/2018/2021, STL & Building Heights indexed in CLMS"),
    ("CLMS", "Coastal Zones (CZ) LCLU", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"coastal zones", "Coastal Zones 2012/2018 LCLU & change indexed in CLMS"),
    ("CLMS", "Riparian Zones (RZ) LCLU", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"riparian zones", "Riparian Zones 2012/2018 LCLU & change indexed in CLMS"),
    ("CLMS", "Natura 2000 (N2K) LCLU", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"n2k|natura\s*2000", "N2K LCLU 2006/2012/2018 & change indexed in CLMS"),
    ("CLMS", "European Ground Motion Service (EGMS)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"ground motion|egms", "EGMS Basic, Calibrated, Ortho L3 displacement products indexed in CLMS"),
    ("CLMS", "Global Dynamic Land Cover (CGLS-LC100 & LCM-10 2020)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"land cover 2020|land cover 2015-2019|global dynamic land cover", "CGLS-LC100 & 10 m Global Land Cover 2020 indexed; 2020-2024 annual series pending public release"),
    ("CLMS", "Pan-Tropical Tree Cover Density 10m (TCD-10)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"tree cover density 2020.*pantropical", "Pan-tropical 10 m TCD 2020 indexed in CLMS"),
    ("CLMS", "High Resolution Hot Spot Monitoring (HSM)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"key landscapes for conservation|hot spot|hotspot", "CLMS Global Land Hot Spot Monitoring (KLC protected areas/land cover) indexed in CLMS"),
    ("CLMS", "Sentinel-2 Global Image Mosaic (S2GM)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"sentinel-2 global image mosaic|s2gm", "Tagged symmetrically as L3_ARD_REFLECTANCE mosaic"),
    ("CLMS", "Vegetation Indices (NDVI v3, EVI, etc.)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"normalised difference vegetation index|\bndvi\b", "NDVI 300m/1km v2 & v3 indexed in CLMS"),
    ("CLMS", "Biophysical Variables (FAPAR, GDMP, DMP, LAI, FCOVER, LSP)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"fraction of absorbed photosynthetically|dry matter productivity|leaf area index|phenology", "300m/1km biophysical ECV time series indexed in CLMS"),
    ("CLMS", "Vegetation - Burnt Area", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"burnt area", "Burnt Area 300m (v3, v4) indexed in CLMS"),
    ("CLMS", "Land Surface Temperature (LST)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"land surface temperature", "LST 3-5 km (hourly, 10-daily, v1 & v2) indexed in CLMS"),
    ("CLMS", "Surface Soil Moisture (SSM) & Soil Water Index (SWI)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"surface soil moisture|soil water index", "SSM 1km & SWI 1km/12.5km indexed in CLMS"),
    ("CLMS", "Evapotranspiration (ETA, 300 m)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"evapotranspiration", "CLMS Global Evapotranspiration (300 m) indexed in CLMS"),
    ("CLMS", "Water Bodies (100 m / 300 m)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"\bwater bodies\b", "Water Bodies 100 m & 300 m v1/v2 indexed in CLMS"),
    ("CLMS", "Lake Water Quality (LWQ, 100 m / 300 m)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"lake water quality", "LWQ 100 m / 300 m v1 & v2 (Chl-a, turbidity, TSM, TSI) indexed in CLMS"),
    ("CLMS", "Lake Surface Water Temperature (LSWT)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"lake surface water temperature", "LSWT 1 km global indexed in CLMS"),
    ("CLMS", "River and Lake Water Level (Altimetry)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"water level", "River Water Level & Lake Water Level 2002-present indexed in CLMS"),
    # CEMS & JRC
    ("CEMS", "GloFAS Global Flood Monitoring (GFM)", "IN_SCOPE_L3_L4_EO", "06_copernicus_cems_gfm_eodc", r"^GFM$|global flood monitoring", "Sentinel-1 operational 20 m flood monitoring indexed in EODC STAC"),
    ("CEMS", "Global Flood Awareness System (GloFAS)", "IN_SCOPE_L3_L4_EO", "05_copernicus_cems_ewds", r"glofas|river discharge and related forecasted", "GloFAS historical & seasonal river discharge datasets indexed in CEMS EWDS"),
    ("CEMS", "European Flood Awareness System (EFAS) & EFAS HERA", "IN_SCOPE_L3_L4_EO", "05_copernicus_cems_ewds", r"\befas\b|european flood awareness", "EFAS operational EWDS series & EFAS HERA reanalysis (in JRC DCAT-AP) indexed"),
    ("CEMS", "European / Global Drought Observatory (EDO / GDO): CDI, SPI, SPEI, RDrI-Agri, HCWI", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"drought observatory|\bgdo\b|\bedo\b|rdri-agri|combined drought", "Full EDO/GDO indicator suite (CDI, SPI, SPEI, RDrI-Agri, HCWI, etc.) indexed via JRC DCAT-AP"),
    ("CEMS", "EFFIS & GWIS (Burned Area, FWI, Fire Danger Forecast)", "IN_SCOPE_L3_L4_EO", "05_copernicus_cems_ewds", r"fire danger|fire weather index|\beffis\b|\bgwis\b", "EFFIS/GWIS FWI historical & seasonal datasets indexed in CEMS EWDS & JRC"),
    ("CEMS/JRC", "Global Human Settlement Layer (GHSL: GHS-BUILT-S, GHS-BUILT-V, GHS-POP, GHS-SMOD)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"ghs-built|ghs-pop|ghs-smod|global human settlement", "GHSL suite indexed across JRC DCAT-AP, DEDL (EO.GHSL), NASA SEDAC & GEE"),
    ("CEMS/JRC", "GHSL Exposure Mapping Component (EMC-BUILT R2025A)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"r2025a|emc-built|ghs-built-h arctic r2025a", "GHSL R2025A datasets (including Arctic exposure) indexed via JRC DCAT-AP"),
    ("JRC", "JRC Global Surface Water Explorer (GSWE, 30 m)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"global surface water explorer|jrc-gswe|JRC/GSW", "Indexed in JRC DCAT-AP and GEE STAC"),
    ("JRC", "JRC Tropical Moist Forest (TMF v2025, 30 m)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"tropical moist forest|forobs|projects/JRC/TMF", "Indexed via official JRC FOROBS TMF portal and GEE"),
    ("CEMS", "CEMS On-Demand Mapping: Rapid Mapping & Risk and Recovery (RRM)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"rapid mapping|risk and recovery|flood in |wildfire in ", "CEMS Rapid Mapping & RRM activations indexed via JRC DCAT-AP"),
    ("JRC", "Monitoring Agricultural Resources (MARS / ASAP Crop Monitoring)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"marsmet|crop monitoring|\bmars\b|\basap\b", "JRC MARS / ASAP agricultural monitoring products indexed in JRC & FAO GISMGR"),
    ("JRC", "European Soil Data Centre (ESDAC) / European Soil Database (ESDB)", "IN_SCOPE_L3_L4_EO", "07_ec_jrc_data_catalogue", r"jrc-esdac|\besdac\b|european soil database|glosem|soil erosion|soil organic carbon", "JRC ESDAC soil datasets indexed in JRC DCAT-AP & EEA SDI"),
    # C3S
    ("C3S", "ERA5 hourly data on single levels (1940-present)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"era5 hourly data on single levels", "Indexed natively in C3S CDS STAC"),
    ("C3S", "ERA5-Land hourly data (1950-present)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"era5-land", "Indexed natively in C3S CDS STAC"),
    ("C3S", "CERRA / CERRA-Land European Regional Reanalysis", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"\bcerra\b", "CERRA & CERRA-Land indexed in C3S CDS STAC"),
    ("C3S", "ECMWF Seasonal & Multi-System Forecasts (C3S / CDS)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"^seasonal-(?:monthly|original)-|seasonal forecast", "C3S seasonal & ECMWF forecast/reanalysis datasets indexed in CDS & DEDL"),
    ("C3S", "Pan-European Climate Database (PECD)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"pan-european climate database|energy related variables", "Indexed natively in C3S CDS STAC"),
    ("C3S", "CMIP6 climate projections", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"^projections-cmip6$|\bcmip6\b", "CMIP6 daily/monthly projections & NEX-GDDP-CMIP6 indexed in CDS, GEE & PC"),
    ("C3S", "Climate indicators for Europe (1940-2100)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"climate indicators for europe", "Indexed natively in C3S CDS STAC"),
    ("C3S", "Essential climate variables for assessment of climate variability", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"essential climate variables for assessment", "Indexed natively in C3S CDS STAC"),
    ("C3S", "Thermal comfort indices derived from ERA5 reanalysis (UTCI)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"thermal comfort indices", "Indexed natively in C3S CDS STAC"),
    ("C3S", "Agrometeorological & Agroclimatic indicators", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"agrometeorological|agroclimatic", "Indexed natively in C3S CDS STAC"),
    ("C3S", "Downscaled bioclimatic indicators for selected regions (1 km)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"downscaled bioclimatic", "Indexed natively in C3S CDS STAC"),
    ("C3S", "Global Crop Productivity Indicators (GCPI)", "IN_SCOPE_L3_L4_EO", "03_copernicus_c3s_cds", r"agrometeorological|crop productivity|wapor", "Covered under C3S agroclimatic / agrometeorological SIS & WaPOR productivity datasets"),
    # CAMS
    ("CAMS", "CAMS Global & Regional Emission Inventories (incl. CAMS-GLOB-SHIP)", "IN_SCOPE_L3_L4_EO", "04_copernicus_cams_ads", r"cams global emission|cams regional emissions|cams-glob-ship", "Indexed natively in CAMS ADS STAC"),
    ("CAMS", "CAMS Global Atmospheric Composition & GHG Forecasts", "IN_SCOPE_L3_L4_EO", "04_copernicus_cams_ads", r"cams global atmospheric composition forecasts|cams global greenhouse gas forecasts", "Indexed natively in CAMS ADS STAC"),
    ("CAMS", "CAMS Global Reanalysis (EAC4 / EGG4)", "IN_SCOPE_L3_L4_EO", "04_copernicus_cams_ads", r"cams global reanalysis|cams global greenhouse gas reanalysis|reanalysis-eac4|reanalysis-egg4", "EAC4 & EGG4 indexed natively in CAMS ADS STAC"),
    ("CAMS", "CAMS European Air Quality Forecasts & Reanalyses", "IN_SCOPE_L3_L4_EO", "04_copernicus_cams_ads", r"cams european air quality", "Forecasts & reanalyses indexed natively in CAMS ADS STAC"),
    ("CAMS", "CAMS Global Inversion-Optimised GHG Fluxes & Concentrations", "IN_SCOPE_L3_L4_EO", "04_copernicus_cams_ads", r"cams global inversion-optimised|greenhouse-gas-inversion", "CO2, CH4, N2O inversion fluxes indexed natively in CAMS ADS STAC"),
    ("CAMS", "CAMS Solar Radiation Time-Series / Services (CRS)", "IN_SCOPE_L3_L4_EO", "04_copernicus_cams_ads", r"cams solar radiation|cams gridded solar", "Solar radiation time series & gridded solar radiation indexed in CAMS ADS STAC"),
    ("CAMS", "CAMS Methane Hotspot Explorer (Interactive Web Viewer)", "OUT_OF_SCOPE_WEB_VIEWER_ONLY", "04_copernicus_cams_ads", r"methane hotspot explorer", "Interactive web visualisation portal; underlying Sentinel-5P/TROPOMI CH4 plumes hosted via ADS/CDSE"),
    ("CAMS", "CAMS Carbon Monitor (Daily Activity-Based CO2)", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", "04_copernicus_cams_ads", r"\bcarbon monitor\b", "Daily CO2 inventory derived from energy/transport/aviation statistics rather than downloadable satellite gridded API"),
    # CMEMS
    ("CMEMS", "Global & European Sea Physics Analysis/Forecast & Reanalysis", "IN_SCOPE_L3_L4_EO", "01_copernicus_cmems", r"GLOBAL_ANALYSISFORECAST_PHY|MULTIYEAR_PHY", "Indexed natively across all Global & European basin products in CMEMS"),
    ("CMEMS", "Global & European Sea Biogeochemistry Analysis/Forecast & Reanalysis", "IN_SCOPE_L3_L4_EO", "01_copernicus_cmems", r"GLOBAL_ANALYSISFORECAST_BGC|MULTIYEAR_BGC", "Indexed natively across all Global & European basin products in CMEMS"),
    ("CMEMS", "High-Resolution Ocean Colour (HR-OC 100 m)", "IN_SCOPE_L3_L4_EO", "01_copernicus_cmems", r"OCEANCOLOUR_.*_HR_L[34]", "100 m Sentinel-2/3 coastal HR-OC products indexed natively in CMEMS"),
    ("CMEMS", "Global/European Ocean Colour (Copernicus-GlobColour) & Chl-a trend", "IN_SCOPE_L3_L4_EO", "01_copernicus_cmems", r"OCEANCOLOUR_.*_BGC_L[34]", "Daily L3/L4 & multi-year reprocessed GlobColour & Chl-a OMI trends indexed in CMEMS"),
    ("CMEMS", "Ocean Monitoring Indicators (OMI) & Mean Sea Level Trend", "IN_SCOPE_L3_L4_EO", "01_copernicus_cmems", r"^OMI_|SEALEVEL_.*_OMI", "Global & regional Ocean Monitoring Indicators indexed natively in CMEMS"),
    ("CMEMS", "Global Coastal Satellite-Derived Bathymetry (SDB, 100 m)", "IN_SCOPE_L3_L4_EO", "01_copernicus_cmems", r"satellite derived bathymetry|BATHY", "Indexed natively in CMEMS"),
    ("CMEMS", "CMEMS In-Situ Near Real Time & Objective Analysis Observations", "OUT_OF_SCOPE_PURE_INSITU", "01_copernicus_cmems", r"^INSITU_", "Discrete NRT buoy/float stations excluded in 02_preparation; gridded OA (`INSITU_*_OA_*`) retained only with `strict_eo_c1_eligible=False`"),
    # COP-DEM, Security, EEA, EMODnet, Eurostat, External
    ("COP-DEM", "Copernicus DEM (GLO-30, GLO-90, EEA-10) & EU-DEM", "IN_SCOPE_L3_L4_EO", "08_destine_dedl_stac", r"cop-dem|copernicus dem|copernicus.*digital elevation model|\beu-dem\b", "Indexed across EODC, DEDL, WEkEO, CDSE & CLMS (EU-DEM)"),
    ("Security/EMSA", "Copernicus Security Services (CMS, CBSS, SESA) & EMSA CleanSeaNet", "OUT_OF_SCOPE_RESTRICTED_SECURITY_SERVICE", None, r"cleanseanet|maritime surveillance|border surveillance", "Security-restricted / non-public operational services without open public API metadata"),
    ("EEA", "EEA Water Exploitation Index plus (WEI+), WISE Flood Risk & EUCRA", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", "11_eea_and_emodnet", r"\bwei\+\b|\beucra\b|\bwise flood risk\b|european climate risk assessment", "Hydrological/statistical indices & directives reports; EMODnet/EEA SDI spatial layers present"),
    ("EMODnet", "EMODnet EUSeaMap, Seagrass, Bathymetry DTM & Vessel Density", "IN_SCOPE_L3_L4_EO", "11_eea_and_emodnet", r"euseamap|seagrass|bathymetry|vessel density", "Indexed via EMODnet GeoNetwork API (`11_eea_and_emodnet`)"),
    ("Eurostat/MS", "Eurostat (env_nwat, NUTS2 Pop, LUCAS) & Member State LPIS", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", None, r"\blucas\s+(?:copernicus|20\d{2}|survey|topsoil|harmonised)\b|\blpis\b|\beurostat\b|\benv_nwat\b", "Administrative/statistical tabular surveys and Member-State cadastral parcels (non-EO reference)"),
    ("External EO", "VIIRS Day/Night Band Nighttime Lights (VNP46)", "IN_SCOPE_L3_L4_EO", "16_nasa_cmr", r"VNP46|nighttime lights", "NASA Black Marble VNP46 L3 & NOAA VIIRS DNB indexed in NASA CMR & GEE"),
    ("External EO", "USGS Landsat Collection 2 Level-3 Science Products", "IN_SCOPE_L3_L4_EO", "21_usgs_landsat_and_eros", r"Landsat Collection 2|surface temperature", "USGS Landsat C2 L3 BA/DSWE/fSCA collections indexed (L1/L2 scenes excluded by L3/L4 rule)"),
    ("External EO", "Global Gravity-based Groundwater Product (G3P / GRACE TWS)", "IN_SCOPE_L3_L4_EO", None, r"DATACITE:10\.5880/g3p|gravity-based groundwater|groundwater storage change data from 2002", "Horizon 2020 G3P indexed via GFZ DataCite API + GRACE/C3S TWS in CDS & NASA CMR"),
    ("External EO", "FLUXCOM-X-BASE Global Carbon & Energy Fluxes (ICOS)", "IN_SCOPE_L3_L4_EO", "26_fao_unep_and_thematic_benchmarks", r"10\.18160/5NZG-JMJE|FLUXCOM-X-BASE", "Global gridded GPP, NEE, ET & energy fluxes indexed via ICOS Carbon Portal DataCite API"),
    ("External EO", "Global LCREF, LCSIF, and LCSPP (SIF & GPP Benchmarks)", "IN_SCOPE_L3_L4_EO", "26_fao_unep_and_thematic_benchmarks", r"\bLCSPP\b|\bLCSIF\b|10\.5281/zenodo\.13922379", "Global land-cover-based SIF and primary production products indexed via Zenodo/DataCite API"),
    ("External EO", "Global Web-Enabled Landsat Data (GWELD Monthly & Annual 30 m)", "IN_SCOPE_L3_L4_EO", "16_nasa_cmr", r"GWELDMO|GWELDYR|Global Web-Enabled Landsat Data", "Monthly and annual global 30 m Landsat composites retained via composite override"),
    ("External EO", "EUMETSAT MSG Land Surface Temperature Data Record (Level 3)", "IN_SCOPE_L3_L4_EO", "17_eumetsat_product_navigator", r"^EO:EUM:DAT:0088$|Land Surface Temperature Data Record - MSG", "Explicit EUMETSAT Level 3 MSG LST climate data record retained in both EUMETSAT & WEkEO"),
    ("External EO", "TROPOSIF Global Fluorescence Yield & TROPOMI SIF Benchmarks", "IN_SCOPE_L3_L4_EO", "26_fao_unep_and_thematic_benchmarks", r"10\.5281/zenodo\.10212472|\bTROPOSIF\b", "Global fluorescence yield based on TROPOSIF data normalized with NIRvP indexed via Zenodo/DataCite"),
    ("External EO", "SMOS-IC v2 Global Soil Moisture & L-VOD Data Record (INRAE/CESBIO)", "IN_SCOPE_L3_L4_EO", "26_fao_unep_and_thematic_benchmarks", r"10\.15454/tihhyp|\bSMOS-IC\b", "Global 0.25 deg daily SMOS-IC v2 soil moisture and L-band VOD indexed via INRAE DataCite"),
    ("External/Model", "EDGAR & CEDS Emission Inventories", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", "07_ec_jrc_data_catalogue", r"\bedgar\b|\bceds\b", "Inventory-based bottom-up emissions indexed under `non_eo_inventory_or_census` (`strict_eo_c1_eligible=False`)"),
    ("External/Non-EO", "EUBUCCO, OpenStreetMap, Building Stock Observatory, FloraVeg.EU, MeteoAlarm", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", None, r"\beubucco\b|\bopenstreetmap\b|\bfloraveg\b|\bmeteoalarm\b", "Non-EO building registries, crowd-sourced OSM vectors, botanical checklists, meteorological warnings"),
    ("CDSE L1/L2", "CDSE Sentinel Level-1 and Level-2 Operational Imagery", "OUT_OF_SCOPE_L1_L2_RAW", "10_cdse_stac", r"^sentinel-[1235]-l[12][abc]?\b", "Excluded in `02_preparation` (`excluded_records_log.parquet`) by L3/L4 eligibility filter"),
]


def run_internal_qa(df_all: pd.DataFrame, df_canon: pd.DataFrame) -> list[tuple[str, bool, str]]:
    qa: list[tuple[str, bool, str]] = []

    # 1. Zero duplicate raw_id within any source
    dup_sources = []
    for sid, grp in df_all.groupby("source_catalogue"):
        if len(grp) != grp["raw_id"].nunique():
            dup_sources.append(f"{sid} ({len(grp)} vs {grp['raw_id'].nunique()})")
    qa.append(("1. Zero duplicate raw_id within any source", len(dup_sources) == 0, f"Violations: {dup_sources or 'none'}"))

    # 2. Zero L0 / SLC-off / SLC-burst / Swath Grid / Test placeholder leakage in canonical
    leak_pat = r"(?:\bLevel[- ]?0[ab]?\b|\bSLC[- ]OFF\b|\bSLC Burst\b|\bSwath Grid\b|\bSwath Image Data\b|\bSwath Files\b|\bSwath elevation\b|^This is a test$)"
    leaks = df_canon[
        df_canon["title"].fillna("").str.contains(leak_pat, case=False, regex=True)
        & ~df_canon["title"].fillna("").str.contains(r"administrative level 0|gaul.*level 0", case=False, regex=True)
    ]
    qa.append(("2. Zero L0/Swath/SLC/Test leakage in canonical", len(leaks) == 0, f"Violations: {len(leaks)}"))

    # 3. Zero JRC non-EO nuclear/materials/LCA leakage
    jrc_bad = df_all[
        (df_all["source_catalogue"] == "07_ec_jrc_data_catalogue")
        & (
            df_all["raw_id"].fillna("").str.contains(r"^(?:jrc-odin|jrc-eplca|jrc-abcis|jrc-10088|jrc-10111)", case=False, regex=True)
            | df_all["title"].fillna("").str.contains(r"thermo-mechanical fatigue|eurl ecvam|sars-cov-2", case=False, regex=True)
        )
    ]
    qa.append(("3. Zero JRC non-EO lab/nuclear/LCA leakage", len(jrc_bad) == 0, f"Violations: {len(jrc_bad)}"))

    # 4. Zero unrealistic >600km degree-fraction, INSPIRE scale denominator, or >10x title-token resolution artifacts
    bad_res = df_all[df_all["spatial_resolution_meters"] > 600_000.0]
    eea_scale = df_all[
        (df_all["source_catalogue"] == "11_eea_and_emodnet")
        & (df_all["spatial_resolution_meters"] >= 500_000.0)
        & (~df_all["spatial_resolution_raw"].astype(str).str.contains("deg", case=False, na=False))
    ]
    both_res = df_all[df_all["spatial_resolution_meters"].notna() & df_all["title_token_resolution_meters"].notna()]
    res_ratio = both_res["spatial_resolution_meters"] / both_res["title_token_resolution_meters"]
    bad_title_res = both_res[(res_ratio >= 10.0) | (res_ratio <= 0.1)]
    qa.append(
        ("4. Zero implausible STAC/EEA scale or title-token resolution artifacts", len(bad_res) == 0 and len(eea_scale) == 0 and len(bad_title_res) == 0, f">600km: {len(bad_res)}, EEA_scale>=500km: {len(eea_scale)}, title_mismatch: {len(bad_title_res)}"),
    )

    # 5. Zero in-situ records marked strict_eo_c1_eligible == True
    bad_insitu = df_all[
        (
            df_all["raw_id"].fillna("").str.contains(r"^insitu[-_]|^EO:MO:DAT:INSITU_|^INSITU_", case=False, regex=True)
            | df_all["title"].fillna("").str.contains(r"\bderived from in[- ]situ observations\b|\bGlobal Ocean in situ\b|Sensor Depth:|\bcollected by XBT\b", case=False, regex=True)
        )
        & (df_all["strict_eo_c1_eligible"] == True)  # noqa: E712
    ]
    qa.append(("5. Zero in-situ records marked strict_eo_c1_eligible", len(bad_insitu) == 0, f"Violations: {len(bad_insitu)}"))

    # 6. Zero within-catalogue canonical_product_key collisions
    key_dups = df_all.groupby(["source_catalogue", "canonical_product_key"]).size()
    within_colls = key_dups[key_dups > 1]
    qa.append(("6. Zero within-catalogue canonical_product_key collisions", len(within_colls) == 0, f"Collisions: {len(within_colls)}"))

    # 7. Zero null or empty product titles
    null_titles = df_all[df_all["title"].isna() | (df_all["title"].astype(str).str.strip() == "")]
    qa.append(("7. Zero null/empty product titles", len(null_titles) == 0, f"Violations: {len(null_titles)}"))

    # 8. Valid WGS84 bounds, temporal ordering, and URL/DOI hygiene
    has_bbox = df_all[df_all[["bbox_west", "bbox_south", "bbox_east", "bbox_north"]].notna().all(axis=1)]
    bad_bbox = has_bbox[
        (has_bbox["bbox_west"] < -180.5)
        | (has_bbox["bbox_east"] > 180.5)
        | (has_bbox["bbox_south"] < -90.5)
        | (has_bbox["bbox_north"] > 90.5)
        | (has_bbox["bbox_south"] > has_bbox["bbox_north"])
    ]
    bad_urls = df_all[df_all["access_url"].notna() & ~df_all["access_url"].astype(str).str.match(r"^(?:https?|ftp)://")]
    bad_dois = df_all[df_all["doi"].notna() & ~df_all["doi"].astype(str).str.match(r"^10\.\d{4,9}/[^\s\]]+$")]
    ts = df_all["temporal_start_raw"].fillna("")
    te = df_all["temporal_end_raw"].fillna("")
    inv_dates = df_all[ts.str.match(r"^\d{4}-\d{2}-\d{2}") & te.str.match(r"^\d{4}-\d{2}-\d{2}") & (ts.str[:10] > te.str[:10])]
    qa.append((
        "8. Coordinate, temporal, DOI, and URL hygiene",
        len(bad_bbox) == 0 and len(bad_urls) == 0 and len(bad_dois) == 0 and len(inv_dates) == 0,
        f"bad_bbox={len(bad_bbox)}, bad_url={len(bad_urls)}, bad_doi={len(bad_dois)}, inv_dates={len(inv_dates)}",
    ))

    # 9. Producer-pool & temporal_resolution_raw integrity
    wgc_l2_leak = df_all[
        (df_all["source_catalogue"] == "12_ceos_cgms_wgclimate_ecv")
        & (df_all["temporal_resolution_raw"].astype(str).str.strip().str.upper() == "L2")
    ]
    cop_marine_noneu = df_all[
        df_all["raw_id"].astype(str).str.startswith("COPERNICUS/MARINE")
        & (df_all["provider_pool"] != "copernicus_services")
    ]
    fedeo_esa_noneu = df_all[
        (df_all["source_catalogue"] == "14_ceos_fedeo_agencies")
        & df_all["producing_institution"].astype(str).str.contains("ESA/ESRIN", case=False, na=False)
        & (df_all["provider_pool"] == "non_eu")
    ]
    fifra_eu = df_all[
        df_all["title"].astype(str).str.contains("ESA/FIFRA", case=False, na=False)
        & (df_all["provider_pool"] != "non_eu")
    ]
    qa.append((
        "9. Producer-pool & WGClimate attribute integrity",
        len(wgc_l2_leak) == 0 and len(cop_marine_noneu) == 0 and len(fedeo_esa_noneu) == 0 and len(fifra_eu) == 0,
        f"wgc_l2={len(wgc_l2_leak)}, cop_marine_non_eu={len(cop_marine_noneu)}, esa_esrin_non_eu={len(fedeo_esa_noneu)}, fifra_eu={len(fifra_eu)}",
    ))

    # 10. Leading-dot degree resolution & future-projection maturity guard
    dyocn_bad = df_all[
        df_all["title"].astype(str).str.contains("AE_DyOcn|AU_DyOcn", case=False, na=False)
        & (df_all["title_token_resolution_meters"] != 27830.0)
    ]
    fut_ongoing = df_all[(df_all["temporal_end_year"].fillna(0) >= 2028) & (df_all["is_ongoing_archive"] == True)]  # noqa: E712
    qa.append((
        "10. Leading-dot resolution & future-projection maturity",
        len(dyocn_bad) == 0 and len(fut_ongoing) == 0,
        f"dyocn_bad={len(dyocn_bad)}, ongoing_ge_2028={len(fut_ongoing)}",
    ))

    # 11. Jurisdiction tier invariants & ERA5/JRC/FedEO/INPE routing
    valid_tiers = {
        "copernicus_service",
        "eu27_national_or_eu_agency",
        "european_space_met_igo",
        "copernicus_associated_state",
        "european_esa_non_copernicus_state",
        "un_or_international_body",
        "third_country_non_european",
    }
    bad_tiers = df_all[~df_all["provider_jurisdiction_tier"].isin(valid_tiers)]
    era_bad = df_all[
        df_all["raw_id"].astype(str).str.contains(r"era5-pds|era5_repack|terrascope-sis-agrometeorological|ce-ag-era5", case=False, na=False)
        & (df_all["provider_pool"] != "copernicus_services")
    ]
    fedeo_noneu_bad = df_all[
        (df_all["source_catalogue"] == "14_ceos_fedeo_agencies")
        & df_all["raw_id"].astype(str).str.contains(r"samet_|opera-s1-rtc|mod11a2|ORNL_CLOUD", case=False, na=False)
        & (df_all["provider_pool"] != "non_eu")
    ]
    qa.append((
        "11. Jurisdiction tier & ERA5/FedEO institutional routing",
        len(bad_tiers) == 0 and len(era_bad) == 0 and len(fedeo_noneu_bad) == 0,
        f"bad_tiers={len(bad_tiers)}, era_non_cop={len(era_bad)}, fedeo_non_eu_leak={len(fedeo_noneu_bad)}",
    ))

    # 12. Zero socioeconomic/campaign/suborbital leaks into strict_eo_c1_eligible
    strict_leaks = df_all[
        (df_all["strict_eo_c1_eligible"] == True)  # noqa: E712
        & df_all["title"].fillna("").str.contains(
            r"\b(?:gridded gdp|unemployment rate|employment in agriculture|ardeco|lucas\s+copernicus|"
            r"integrated global radiosonde archive|argo float|polarstern cruise|airborne lidar|drone imagery)\b",
            case=False,
            regex=True,
        )
    ]
    qa.append((
        "12. Zero non-EO/campaign leaks in strict_eo_c1_eligible",
        len(strict_leaks) == 0,
        f"Violations: {len(strict_leaks)}",
    ))

    # 13. High-recall rescue of composited/gridded L2-tagged EO & SEDAC/WaPOR series
    has_wapor = (df_all["raw_id"].astype(str).str.startswith("FAO/WAPOR/")).sum() >= 12
    has_syn = (df_all["raw_id"].astype(str).str.contains("syn-2-v10|syn-2-vg1|s3-syn-vgt|synergy-v10", case=False, na=False)).sum() >= 6
    has_transcom = (df_all["title"].astype(str).str.contains("TransCom 3", case=False, na=False)).sum() >= 2
    has_gcom = (df_all["raw_id"].astype(str).str.contains("GCOM-C_SGLI_L2_statistics", na=False)).sum() >= 10
    qa.append((
        "13. Thematic EO reference series present in canonical",
        bool(has_wapor and has_syn and has_transcom and has_gcom),
        f"wapor={has_wapor}, s3_syn={has_syn}, transcom={has_transcom}, gcom_stats={has_gcom}",
    ))

    # 14. Zero Suomi-NPP L3-to-L4, PML/HS3 homograph, or >2026 start-year leaks
    id_title = df_all["raw_id"].astype(str) + " " + df_all["title"].astype(str)
    suomi_l4_cnt = int((id_title.str.contains("VIIRSN_L3|VNP43|OMPS_NPP.*L3", case=False, na=False) & (df_all["processing_level"] == "L4")).sum())
    pml_leak_cnt = int((id_title.str.contains("CAS/IGSNRR/PML", case=False, na=False) & (df_all["provider_pool"] != "non_eu")).sum())
    hs3_leak_cnt = int((df_all["title"].astype(str).str.contains("Hurricane and Severe Storm Sentinel", case=False, na=False) & (df_all["uses_sentinel_data"] == True)).sum())
    future_start_cnt = int((df_all["temporal_start_year"] > 2026).sum())
    qa.append((
        "14. Zero Suomi-NPP L3-to-L4, PML/HS3 homograph, or >2026 start-year leaks",
        suomi_l4_cnt == 0 and pml_leak_cnt == 0 and hs3_leak_cnt == 0 and future_start_cnt == 0,
        f"suomi_l4={suomi_l4_cnt}, pml_leak={pml_leak_cnt}, hs3_leak={hs3_leak_cnt}, future_start={future_start_cnt}",
    ))

    # 15. Deep provider metadata recovery (NOAA summary=False, CMR UMM-JSON) & non-image access URLs
    noaa_c = df_canon[df_canon["source_catalogue"] == "20_noaa_onestop"]
    cmr_c = df_canon[df_canon["source_catalogue"] == "16_nasa_cmr"]
    noaa_desc_ok = len(noaa_c) >= 12000 and float(noaa_c["description"].notna().mean()) >= 0.98
    cmr_umm_ok = (
        len(cmr_c) >= 5800
        and float(cmr_c["doi"].notna().mean()) >= 0.90
        and int(cmr_c["spatial_resolution_meters"].notna().sum()) >= 2300
        and int(cmr_c["temporal_resolution_raw"].notna().sum()) >= 1600
    )
    img_urls_cnt = int(
        df_all["access_url"]
        .fillna("")
        .str.contains(r"\.(?:png|jpe?g|gif|bmp|webp|svg|kmz)(?:[?#]|$)|/browse[/?#]", case=False, regex=True)
        .sum()
    )
    qa.append((
        "15. Deep NOAA/CMR metadata recovery & zero preview-image access_url",
        bool(noaa_desc_ok and cmr_umm_ok and img_urls_cnt == 0),
        f"noaa_desc={noaa_c['description'].notna().mean():.1%}, cmr_doi={cmr_c['doi'].notna().mean():.1%}, cmr_spat={int(cmr_c['spatial_resolution_meters'].notna().sum())}, img_urls={img_urls_cnt}",
    ))

    # 16. Cadence support, operational status & Sentinel lineage invariants
    multi_day_ttl = (
        df_all["title"]
        .fillna("")
        .str.contains(
            r"\b(?:8-day|10-day|10-daily|16-day|5-day|dekad|dekadal|weekly|biweekly)\b", case=False, regex=True
        )
        & df_all["temporal_aggregation_window_raw"].isna()
    )
    cadence_contra = int((multi_day_ttl & (df_all["temporal_cadence_class"] == "sub_daily_or_daily")).sum())
    op_null_ongoing = int(
        ((df_all["operational_maturity_class"] == "operational_ongoing_series") & df_all["is_ongoing_archive"].isna()).sum()
    )
    pre2014_sent = int(
        (
            (df_all["uses_sentinel_data"] == True)  # noqa: E712
            & (df_all["is_ongoing_archive"] != True)  # noqa: E712
            & (df_all["temporal_end_year"].fillna(9999) < 2014)
        ).sum()
    )
    meris_ilw_bad = int(
        (
            (df_all["raw_id"] == "C3433948164-OB_CLOUD")
            & ((df_all["geographic_coverage_class"] == "global") | (df_all["uses_sentinel_data"] == True))  # noqa: E712
        ).sum()
    )
    qa.append((
        "16. Zero cadence contradictions, null-ongoing maturity, or pre-2014 Sentinel tags",
        cadence_contra == 0 and op_null_ongoing == 0 and pre2014_sent == 0 and meris_ilw_bad == 0,
        f"cadence_contra={cadence_contra}, op_null={op_null_ongoing}, pre2014_sent={pre2014_sent}, meris_ilw_bad={meris_ilw_bad}",
    ))

    # 17. Structured GEE intervals, monthly/daily disambiguation, Sentinel positive lineage, LPJ-EOSIM L4 & MERRA-2 URL
    gee_cad_lit = int((df_all["temporal_resolution_raw"] == "cadence").sum())
    monthly_daily_contra = int(
        (
            df_all["title"].fillna("").str.contains(r"\bmonthly\b", case=False, regex=True)
            & ~df_all["title"].fillna("").str.contains(r"\bdaily\b|\bhourly\b", case=False, regex=True)
            & (df_all["temporal_cadence_class"] == "sub_daily_or_daily")
        ).sum()
    )
    title_l4_to_l3 = int(
        (
            df_all["title"].fillna("").str.contains(r"\b(?:L4[ab]?|level[- ]?4)\b", case=False, regex=True)
            & (df_all["processing_level"] == "L3")
        ).sum()
    )
    sent_pos_ids = {
        "01_copernicus_cmems::GLOBAL_MULTIYEAR_WAV_001_032",
        "01_copernicus_cmems::SEALEVEL_GLO_PHY_L3_MY_008_062",
        "01_copernicus_cmems::WAVE_GLO_PHY_SPC_L3_MY_014_006",
        "16_nasa_cmr::C3126327092-ORNL_CLOUD",
        "10_cdse_stac::clms_ndvi_global_300m_10daily_v1_cog",
    }
    sent_pos_sub = df_all[df_all["record_id"].isin(sent_pos_ids)]
    sent_pos_ok = len(sent_pos_sub) == len(sent_pos_ids) and bool(sent_pos_sub["uses_sentinel_data"].all())
    lpj_sub = df_all[df_all["title"].fillna("").str.contains("LPJ-EOSIM", case=False)]
    lpj_ok = len(lpj_sub) >= 4 and bool((lpj_sub["processing_level"] == "L4").all())
    merra2_sub = df_all[df_all["record_id"] == "16_nasa_cmr::C2531963170-GES_DISC"]
    merra2_ok = bool(
        len(merra2_sub) == 1
        and "10.5067/o8ax56do60mi" in str(merra2_sub.iloc[0]["access_url"]).lower()
        and "0.625 x 0.5" in str(merra2_sub.iloc[0]["spatial_resolution_raw"])
    )
    qa.append((
        "17. Structured GEE intervals, monthly cadence, Sentinel positive lineage & MERRA-2 DOI URL",
        bool(gee_cad_lit == 0 and monthly_daily_contra == 0 and title_l4_to_l3 == 0 and sent_pos_ok and lpj_ok and merra2_ok),
        f"gee_cad={gee_cad_lit}, mo_contra={monthly_daily_contra}, l4_to_l3={title_l4_to_l3}, sent_pos={sent_pos_ok}, lpj_l4={lpj_ok}, merra2={merra2_ok}",
    ))

    # 18. Deep scientific QA invariants (closed archives, European/polar geo scope, EU benchmark pool, strict-EO physical exclusions, citation-year Sentinel)
    op_closed = int(
        (
            (df_all["operational_maturity_class"] == "operational_ongoing_series")
            & (df_all["temporal_end_year"].fillna(9999) <= 2024)
        ).sum()
    )
    eu_nom_glob = int(
        (
            df_all["record_id"].str.contains("_europe_|^05_copernicus_cems_ewds::efas-", case=False, regex=True)
            & (df_all["geographic_coverage_class"] == "global")
        ).sum()
    )
    polar_bad = int(
        (
            df_all["raw_id"].isin([
                "satellite-sea-ice-thickness",
                "C3291000765-NSIDC_CPRD",
                "ANTARCTIC_OMI_SI_extent",
            ])
            & df_all["geographic_coverage_class"].isin(["global", "quasi_global"])
        ).sum()
    )
    eu_bench = df_all[df_all["raw_id"].isin(["ZENODO:2575599", "ZENODO:10212472", "DATACITE:10.1594/pangaea.911944"])]
    eu_bench_ok = len(eu_bench) == 3 and bool((eu_bench["provider_pool"] == "eu_non_copernicus").all())
    bicep_sub = df_all[df_all["raw_id"].isin([
        "export-production-bicep",
        "monthly-global-export-production-based-on-henson-et-al-2011-dataset-9-km-oc-cci-v42",
    ])]
    bicep_ok = len(bicep_sub) == 2 and bool(bicep_sub["uses_sentinel_data"].all())
    phys_bad = int(
        (
            df_all["title"].fillna("").str.contains(
                r"Archive of Geosample Data|NOAA Marine Environmental Buoy Database|based on in-situ raingauge observations|USGSPHOTOS|NASAPHOTOS",
                case=False,
                regex=True,
            )
            & (df_all["strict_eo_c1_eligible"] == True)  # noqa: E712
        ).sum()
    )
    single_epoch_my_bad = int(
        (
            df_all["raw_id"].isin(["C1711961296-LPCLOUD", "C2763261621-LPCLOUD", "C3543456364-LPCLOUD", "C2763264663-LPCLOUD", "C3206501379-NSIDC_CPRD", "C3297995634-NSIDC_CPRD"])
            & (df_all["temporal_cadence_class"] == "multi_year_periodic")
        ).sum()
    )
    qa.append((
        "18. Deep scientific invariants (closed-end maturity, EU/polar geo, EU benchmarks, physical strict-EO, BICEP S3, single-epoch windows)",
        bool(
            op_closed == 0
            and eu_nom_glob == 0
            and polar_bad == 0
            and eu_bench_ok
            and bicep_ok
            and phys_bad == 0
            and single_epoch_my_bad == 0
        ),
        f"op_closed={op_closed}, eu_glob={eu_nom_glob}, polar_bad={polar_bad}, eu_bench={eu_bench_ok}, bicep_s3={bicep_ok}, phys_bad={phys_bad}, epoch_my_bad={single_epoch_my_bad}",
    ))

    # 19. Provenance, HTML/URL safety, GEO disk, ARD vs output, L3+L4, Sentinel HLS & supersession invariants
    html_rem = int(df_all["description"].fillna("").str.contains(r"&(?:deg|times|ndash|nbsp|rarr);|</?(?:p|div|strong|ul|li)\b", regex=True).sum())
    zen_url_nulls = int(df_all[(df_all["source_catalogue"] == "26_fao_unep_and_thematic_benchmarks") & df_all["raw_id"].astype(str).str.startswith("ZENODO:")]["access_url"].isna().sum())
    tcsif_v1_ok = bool(df_all["doi"].fillna("").str.lower().str.contains("10.5281/zenodo.8242928").any())
    cbers_sub = df_all[df_all["raw_id"] == "CBERS4-MUX-2M-1"]
    cbers_ok = len(cbers_sub) >= 1 and bool((cbers_sub["spatial_resolution_meters"] == 20.0).all())
    abi_g16_sub = df_all[df_all["raw_id"].astype(str).str.contains("AERDB_M3_ABI_G16|AERDB_D3_AHI_H08", case=False) | df_all["title"].fillna("").str.contains("AERDB_M3_ABI_G16|AERDB_D3_AHI_H08", case=False)]
    abi_g16_glob = int((abi_g16_sub["geographic_coverage_class"] == "global").sum())
    abi_g16_dom_ok = len(abi_g16_sub) >= 2 and bool((abi_g16_sub["observation_domain_class"] == "geostationary_disk_or_sector").all())
    non_ard_leak = int((df_all["raw_id"].isin(["ZENODO:20683570", "ZENODO:13922379", "ZENODO:14568024", "C2617126679-POCLOUD", "MODIS/MCD43A4_006_BAI"]) & (df_all["is_ard_reflectance_only"] == True)).sum())  # noqa: E712
    mcd43_ard_ok = bool(df_all[df_all["raw_id"].isin(["C2218719731-LPCLOUD", "MODIS/061/MCD43A4"])]["is_ard_reflectance_only"].all())
    hlsl30_s2 = int((df_all["raw_id"].isin(["C2021957657-LPCLOUD", "NASA/HLS/HLSL30/v002", "hls2-l30"]) & (df_all["uses_sentinel_data"] == True)).sum())  # noqa: E712
    c3s_sst = df_all[(df_all["source_catalogue"] == "03_copernicus_c3s_cds") & (df_all["raw_id"] == "satellite-sea-surface-temperature")]
    c3s_sst_ok = len(c3s_sst) == 1 and c3s_sst.iloc[0]["processing_level"] == "L3; L4"
    c3s_alb_ongoing = int(((df_all["source_catalogue"] == "03_copernicus_c3s_cds") & (df_all["raw_id"].isin(["satellite-albedo", "satellite-lai-fapar"])) & (df_all["is_ongoing_archive"] == True)).sum())  # noqa: E712
    qa.append((
        "19. Provenance, HTML/URL safety, GEO disk, ARD NBAR, HLSL30, L3+L4 & maturity invariants",
        bool(
            html_rem == 0
            and zen_url_nulls == 0
            and tcsif_v1_ok
            and cbers_ok
            and abi_g16_glob == 0
            and abi_g16_dom_ok
            and non_ard_leak == 0
            and mcd43_ard_ok
            and hlsl30_s2 == 0
            and c3s_sst_ok
            and c3s_alb_ongoing == 0
        ),
        f"html={html_rem}, zen_null_url={zen_url_nulls}, tcsif_v1={tcsif_v1_ok}, cbers4={cbers_ok}, geo_glob={abi_g16_glob}, ard_leak={non_ard_leak}, hlsl30_s2={hlsl30_s2}, c3s_sst={c3s_sst_ok}",
    ))

    # 20. Deep regression & adjudication invariants (GEO global restore, MERRA-2 SNDRD, OC L3, semimonthly, variants, cited conflicts, op25, Sentinel split, ECV negative controls)
    gldas_omi = df_all[df_all["raw_id"].isin(["C1233767575-GES_DISC", "C1266136072-GES_DISC"])]
    sndrd_rows = df_all[df_all["raw_id"].astype(str).str.contains(r"^C3944483", regex=True)]
    merra2_geo_bad = int(
        (
            df_all["title"].fillna("").str.contains(r"\bmerra[-_ ]?2\b", case=False, regex=True)
            & (df_all["observation_domain_class"] == "geostationary_disk_or_sector")
        ).sum()
    )
    gldas_omi_sndrd_ok = (
        len(gldas_omi) == 2
        and bool((gldas_omi["geographic_coverage_class"] == "global").all())
        and len(sndrd_rows) >= 16
        and bool((sndrd_rows["geographic_coverage_class"] == "global").all())
        and merra2_geo_bad == 0
    )

    oc_rows = df_all[df_all["record_id"].isin(["03_copernicus_c3s_cds::satellite-ocean-colour", "09_wekeo_dias::EO:ECMWF:DAT:SATELLITE_OCEAN_COLOUR"])]
    c3s_unlab_sat = df_all[(df_all["source_catalogue"] == "03_copernicus_c3s_cds") & df_all["raw_id"].astype(str).str.startswith("satellite-") & df_all["processing_level"].isna()]
    oc_cited_l3_ok = (
        len(oc_rows) == 2
        and bool((oc_rows["processing_level"] == "L3").all())
        and bool((oc_rows["processing_level_provenance"] == "provider_documented_specification").all())
        and bool((oc_rows["adjudication_source_citation"] == "https://climate.copernicus.eu/ocean").all())
        and len(c3s_unlab_sat) >= 20
    )

    sm_rows = df_all[df_all["raw_id"].isin(["C2759076389-ORNL_CLOUD", "ZENODO:8253971"])]
    bimonthly_sm = int((df_all["raw_id"].astype(str).str.contains("bimonthly", case=False) & (df_all["temporal_output_cadences"] == "semimonthly")).sum())
    muses_r = df_all[df_all["raw_id"] == "ZENODO:7479028"]
    oco3_r = df_all[df_all["raw_id"] == "ZENODO:15382965"]
    semimonthly_prose_ok = (
        len(sm_rows) == 2
        and bool((sm_rows["temporal_output_cadences"] == "semimonthly").all())
        and bimonthly_sm == 0
        and len(muses_r) == 1
        and pd.isna(muses_r.iloc[0]["spatial_resolution_meters"])
        and float(muses_r.iloc[0]["adjudicated_output_grid_meters"]) == 5566.0
        and len(oco3_r) == 1
        and oco3_r.iloc[0]["temporal_cadence_class"] == "monthly_or_seasonal"
    )

    variants_path = HARM_DIR / "eo_l3_l4_product_variants.parquet"
    variants_ok = False
    if variants_path.exists():
        df_var = pd.read_parquet(variants_path)
        vtcsif = df_var[df_var["raw_id"] == "ZENODO:8242928"]
        vsst = df_var[df_var["raw_id"] == "satellite-sea-surface-temperature"]
        variants_ok = (
            len(df_var) >= 50
            and set(vtcsif["variant_level"]) >= {"L2", "L3"}
            and set(vsst["variant_level"]) >= {"L3", "L4"}
        )

    conf_rows = df_all[df_all["spatial_resolution_conflict"] == True]  # noqa: E712
    cited_conf = conf_rows[conf_rows["adjudication_source_citation"].notna()]
    unres_conf = conf_rows[conf_rows["adjudication_source_citation"].isna()]
    spatial_cite_ok = (
        len(cited_conf) >= 33
        and bool(conf_rows["spatial_resolution_meters"].isna().all())
        and bool(cited_conf["adjudicated_output_grid_meters"].notna().all())
        and bool(cited_conf["adjudicated_spatial_grid_units"].notna().all())
        and bool(cited_conf["adjudicated_spatial_grid_axes"].notna().all())
        and bool(cited_conf["adjudication_source_citation"].astype(str).str.startswith("https://doi.org/").all())
        and bool(unres_conf["adjudicated_output_grid_meters"].isna().all())
    )

    msat_rows = df_all[df_all["raw_id"].astype(str).str.startswith("projects/edf-methanesat-ee/assets/msat/")]
    closed_2025_rows = df_all[df_all["raw_id"].isin(["a49dc43a-d799-11f1-8257-02ffd92199af", "d4e9b21f-6991-48ba-9b32-d1b4a08d76f0"])]
    op25_ok = (
        len(msat_rows) >= 3
        and bool((msat_rows["operational_maturity_class"] == "unknown").all())
        and len(closed_2025_rows) == 2
        and bool((closed_2025_rows["operational_maturity_class"] == "completed_or_mission_bounded_release").all())
        and bool((closed_2025_rows["is_ongoing_archive"] == False).all())  # noqa: E712
    )

    sent_status_vc = df_all["sentinel_lineage_status"].value_counts().to_dict()
    gee29_sent_vc = df_all[df_all["source_catalogue"] == "29_awesome_gee_community"]["sentinel_lineage_status"].value_counts().to_dict()
    sentinel_split_ok = (
        sent_status_vc.get("no_sentinel_lineage", 0) == 0
        and sent_status_vc.get("explicit_non_sentinel", 0) >= 19000
        and sent_status_vc.get("unassessed_no_platform_metadata", 0) >= 16000
        and gee29_sent_vc.get("unassessed_no_platform_metadata", 0) >= 4200
    )

    # ECV cross-domain leak guards (synthetic + live negative controls)
    ecv_df_chk = audit_gcos_ecv(df_canon)
    ecv_40_ok = len(ecv_df_chk) == 40 and int((ecv_df_chk["canonical_catalogue_matches"] > 0).sum()) == 40 and int((ecv_df_chk["independent_non_wgclimate_matches"] > 0).sum()) == 40
    live_seaflux = df_canon[df_canon["raw_id"] == "ZENODO:8280457"]
    live_pace = df_canon[df_canon["raw_id"].isin(["C3620140148-OB_CLOUD", "C3620140152-OB_CLOUD", "C3620140344-OB_CLOUD", "C3620140363-OB_CLOUD"])]
    neg_sample = pd.concat(
        [
            pd.DataFrame([
                {"title": "SAFARI 2000 FAPAR PAR 1-km Grid", "raw_id": "SYNTH_FAPAR", "keywords": "par fapar", "description": "absorbed photosynthetically active radiation par", "product_output_category": "", "source_catalogue": "synthetic", "provider_pool": "non_eu"},
                {"title": "MOD16A2 Terrestrial Evapotranspiration & Latent Heat Flux", "raw_id": "SYNTH_MOD16", "keywords": "latent heat flux", "description": "terrestrial evapotranspiration mod16 global land", "product_output_category": "", "source_catalogue": "synthetic", "provider_pool": "non_eu"},
                {"title": "GLAS Specific Leaf Area and Biomass Map", "raw_id": "SYNTH_SLA", "keywords": "sla leaf", "description": "specific leaf area sla forest", "product_output_category": "", "source_catalogue": "synthetic", "provider_pool": "non_eu"},
            ]),
            live_seaflux[[c for c in df_canon.columns if c in ("title", "raw_id", "keywords", "description", "product_output_category", "source_catalogue", "provider_pool")]],
            live_pace[[c for c in df_canon.columns if c in ("title", "raw_id", "keywords", "description", "product_output_category", "source_catalogue", "provider_pool")]],
        ],
        ignore_index=True,
    )
    neg_ecv = audit_gcos_ecv(neg_sample).set_index("gcos_ecv_name")
    ecv_neg_ok = (
        ecv_40_ok
        and len(live_seaflux) == 1
        and len(live_pace) >= 1
        and int(neg_ecv.loc["Surface radiation budget", "canonical_catalogue_matches"]) == 0
        and int(neg_ecv.loc["Ocean-surface heat flux", "canonical_catalogue_matches"]) == 0
        and int(neg_ecv.loc["Sea level", "canonical_catalogue_matches"]) == 0
        and int(neg_ecv.loc["Plankton", "canonical_catalogue_matches"]) == 0
    )

    qa.append((
        "20. Geographic, cadence, variant, adjudication & ECV invariants",
        bool(gldas_omi_sndrd_ok and oc_cited_l3_ok and semimonthly_prose_ok and variants_ok and spatial_cite_ok and op25_ok and sentinel_split_ok and ecv_neg_ok),
        f"geo={gldas_omi_sndrd_ok}, oc_l3={oc_cited_l3_ok}, sm_prose={semimonthly_prose_ok}, var={variants_ok}, cite={spatial_cite_ok}, op25={op25_ok}, sent_split={sentinel_split_ok}, ecv_neg={ecv_neg_ok}",
    ))

    # 21. Verified-release regression baseline & Source 26 Zenodo pagination completeness
    baseline_path = BASE_DIR / "verified_releases_baseline.json"
    pag_audit_path = BASE_DIR.parent / "01_raw" / "26_fao_unep_and_thematic_benchmarks" / "data" / "zenodo_pagination_audit.json"
    baseline_ok = False
    pag_ok = False
    base_detail = "missing baseline files"
    if baseline_path.exists() and pag_audit_path.exists():
        base_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        pag_data = json.loads(pag_audit_path.read_text(encoding="utf-8"))
        zen_present = set(df_all[df_all["raw_id"].astype(str).str.startswith("ZENODO:")]["raw_id"].astype(str).str.replace("ZENODO:", "", regex=False))
        cf_missing = set(str(x) for x in base_data.get("zenodo_carryforward_ids", [])) - zen_present
        idx_df = df_all.set_index("record_id")
        drift: list[str] = []
        for spec in base_data.get("anchored_records", []):
            rid = spec["record_id"]
            if rid not in idx_df.index:
                drift.append(f"MISSING:{rid}")
                continue
            row = idx_df.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for fld in ("geographic_coverage_class", "processing_level", "temporal_cadence_class", "operational_maturity_class", "sentinel_lineage_status"):
                got = None if pd.isna(row[fld]) else str(row[fld])
                exp = spec.get(fld)
                if got != exp:
                    drift.append(f"{rid}:{fld}={got}!={exp}")
        baseline_ok = len(df_all) >= int(base_data.get("min_harmonized_records", 0)) and len(cf_missing) == 0 and len(drift) == 0
        pag_ok = int(pag_data.get("truncated_query_count", -1)) == 0 and len(pag_data.get("carry_forward_missing_ids", ["x"])) == 0
        base_detail = f"cf_missing={len(cf_missing)}, drift={len(drift)}, trunc={pag_data.get('truncated_query_count')}"
    qa.append((
        "21. Verified-release baseline delta & Source 26 Zenodo pagination completeness",
        bool(baseline_ok and pag_ok),
        base_detail,
    ))

    # 22. Full snapshot ID delta, prose provenance separation, GEDI L4A conflict, TCSIF/SST variants & EUMETSAT crosswalk
    all_ids_path = BASE_DIR / "verified_record_ids.json"
    id_delta_ok = False
    id_missing_cnt = -1
    if all_ids_path.exists():
        exp_ids = set(json.loads(all_ids_path.read_text(encoding="utf-8")).get("record_ids", []))
        got_ids = set(df_all["record_id"].astype(str))
        id_missing_cnt = len(exp_ids - got_ids)
        id_delta_ok = len(exp_ids) >= 38900 and id_missing_cnt == 0

    amsr2 = df_all[df_all["raw_id"] == "C1886605828-LANCEAMSR2"]
    remo = df_all[df_all["raw_id"] == "cd0b21c3-dc1c-4e44-860d-eeb714da19c6"]
    gl_lai = df_all[df_all["raw_id"] == "ZENODO:12698637"]
    gedi4a = df_all[df_all["record_id"] == "16_nasa_cmr::C4212593885-ORNL_CLOUD"]
    wk0088 = df_all[df_all["record_id"] == "09_wekeo_dias::EO:EUM:DAT:0088"]
    df_var_22 = pd.read_parquet(variants_path) if variants_path.exists() else pd.DataFrame()
    v821 = df_var_22[df_var_22["raw_id"] == "ZENODO:8210518"] if len(df_var_22) else pd.DataFrame()
    vsst_22 = df_var_22[df_var_22["raw_id"] == "satellite-sea-surface-temperature"] if len(df_var_22) else pd.DataFrame()
    snpp_end_ongoing = int((df_all["title"].fillna("").str.contains(r"end of mission|2026-11-02", case=False, regex=True) & (df_all["operational_maturity_class"] == "operational_ongoing_series")).sum())

    prose_sep_ok = (
        len(amsr2) == 1 and pd.isna(amsr2.iloc[0]["spatial_resolution_raw"]) and float(amsr2.iloc[0]["adjudicated_output_grid_meters"]) == 6250.0
        and len(remo) == 1 and pd.isna(remo.iloc[0]["spatial_resolution_raw"]) and float(remo.iloc[0]["adjudicated_output_grid_meters"]) == 5566.0
        and pd.isna(remo.iloc[0]["observation_domain_mask"])
        and len(gl_lai) == 1 and gl_lai.iloc[0]["temporal_output_cadences"] == "semimonthly; 8-day"
        and "1981-2000: semimonthly" in str(gl_lai.iloc[0]["temporal_cadence_by_epoch"])
    )
    var_evidence_ok = (
        len(gedi4a) == 1
        and bool(gedi4a.iloc[0]["processing_level_conflict"])
        and float(gedi4a.iloc[0]["spatial_footprint_meters"]) == 25.0
        and pd.isna(gedi4a.iloc[0]["adjudicated_output_grid_meters"])
        and len(df_var_22[df_var_22["raw_id"] == "C4212593885-ORNL_CLOUD"]) == 0
        and len(v821) == 2 and set(v821["variant_level"]) == {"L2", "L3"}
        and bool(v821[v821["variant_level"] == "L2"]["variant_spatial_resolution_meters"].isna().all())
        and bool((v821[v821["variant_level"] == "L3"]["variant_spatial_resolution_meters"] == 55660.0).all())
        and len(vsst_22) == 2 and bool((vsst_22["variant_spatial_resolution_meters"] == 5566.0).all())
    )
    wk_cw_ok = (
        len(wk0088) == 1
        and wk0088.iloc[0]["processing_level"] == "L3"
        and wk0088.iloc[0]["processing_level_provenance"] == "crosswalked_primary_eumetsat_navigator"
        and wk0088.iloc[0]["temporal_cadence_class"] == "sub_daily_or_daily"
    )
    qa.append((
        "22. Full snapshot ID delta, prose provenance, GEDI/TCSIF/SST variants & EUMETSAT crosswalk",
        bool(id_delta_ok and prose_sep_ok and var_evidence_ok and wk_cw_ok and snpp_end_ongoing == 0),
        f"missing_prior_ids={id_missing_cnt}, prose_sep={prose_sep_ok}, variants={var_evidence_ok}, wk0088={wk_cw_ok}, snpp_end_on={snpp_end_ongoing}",
    ))

    # 23. Strict RFC 8259 JSONL, CEDA DOI provenance, antimeridian, variable-cadence, sublevel, NH SWE, new releases & harvest-anchored lifecycle
    strict_jsonl_ok = True
    jsonl_checked = 0
    for jpath in sorted(PREP_DIR.glob("*.jsonl")) + sorted(HARM_DIR.glob("*.jsonl")):
        jsonl_checked += 1
        with jpath.open("r", encoding="utf-8") as jf:
            for jline in jf:
                try:
                    json.loads(jline, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
                except Exception:
                    strict_jsonl_ok = False
                    break
        if not strict_jsonl_ok:
            break

    ceda_sub = df_all[
        (df_all["source_catalogue"] == "14_ceos_fedeo_agencies")
        & (df_all["raw_id"].astype(str).str.match(r"^[0-9a-f]{32}$"))
    ]
    bad_ceda_doi = int((ceda_sub["doi"].fillna("").str.contains(r"MOD021KM|10\.1016/|10\.5194/", regex=True, case=False)).sum())
    bad_doi_src = int((ceda_sub["doi_source"].fillna("") == "stac_link_describedby").sum())
    snow7_uuids = [
        "edf8abd23f4a40aabd4d52e48dec06ea",
        "bc13bb02a958449aac139853c4638f32",
        "38a71d034b5c4097821de29ee3bc2498",
        "80d96e3a14854420b6f742d70877c431",
        "375ffdb8f0a445e380b4b9548655f5f9",
        "f5dce1f7bec2447093cf460a4d3ba2c2",
        "3c71c04cf08a410fac2c680cbf88cfd7",
    ]
    snow7_rows = df_all[df_all["raw_id"].isin(snow7_uuids)]
    snow7_ok = len(snow7_rows) == 7 and bool((snow7_rows["doi"].str.lower() == ("10.5285/" + snow7_rows["raw_id"]).str.lower()).all())
    ceda_ok = bad_ceda_doi == 0 and bad_doi_src == 0 and snow7_ok

    tpca = df_all[df_all["raw_id"] == "ZENODO:4922168"]
    gl_lai_23 = df_all[df_all["raw_id"] == "ZENODO:12698637"]
    cam_a = df_all[df_all["raw_id"] == "ZENODO:3403005"]
    v1081 = df_all[df_all["raw_id"] == "ZENODO:10815170"]
    v1081_vars = df_var_22[df_var_22["raw_id"] == "ZENODO:10815170"] if len(df_var_22) else pd.DataFrame()
    antim_var_ok = (
        len(tpca) == 1
        and float(tpca.iloc[0]["bbox_west"]) == 150.0
        and float(tpca.iloc[0]["bbox_east"]) == -90.0
        and bool(tpca.iloc[0]["bbox_crosses_antimeridian"])
        and tpca.iloc[0]["geographic_coverage_class"] == "regional_or_local"
        and int(tpca.iloc[0]["temporal_start_year"]) == 1997
        and int(tpca.iloc[0]["temporal_end_year"]) == 2019
        and len(gl_lai_23) == 1 and int(gl_lai_23.iloc[0]["temporal_end_year"]) == 2023
        and len(cam_a) == 1 and int(cam_a.iloc[0]["temporal_end_year"]) == 2017
        and len(v1081) == 1
        and v1081.iloc[0]["prose_cadence_role"] == "variable_dependent_multi_cadence"
        and v1081.iloc[0]["temporal_output_cadences"] == "monthly; annual; static"
        and pd.isna(v1081.iloc[0]["temporal_cadence_by_epoch"])
        and len(v1081_vars) == 3
        and set(v1081_vars["variant_cadence"]) == {"monthly", "annual", "static"}
    )

    sublevel_nulls = int((
        df_canon["title"].fillna("").str.contains(r"\bLevel[- ]?[34][A-Z]\b", regex=True, case=False)
        & df_canon["processing_level"].isna()
    ).sum())
    nh5_ids = [
        "16_nasa_cmr::C4017534264-FEDEO",
        "14_ceos_fedeo_agencies::edf8abd23f4a40aabd4d52e48dec06ea",
        "14_ceos_fedeo_agencies::9d9bfc488ec54b1297eca2c9662f9c81",
        "14_ceos_fedeo_agencies::4647cc9ad3c044439d6c643208d3c494",
        "16_nasa_cmr::C2254686682-ORNL_CLOUD",
    ]
    nh5 = df_all[df_all["record_id"].isin(nh5_ids)]
    nh5_ok = (
        len(nh5) == 5
        and bool((nh5["geographic_coverage_class"] == "hemispheric_or_pantropical").all())
        and bool((nh5["observation_domain_class"] == "northern_hemisphere_only").all())
    )

    jof = df_all[df_all["raw_id"] == "DATACITE:10.20783/dias.667"]
    coseb = df_all[df_all["raw_id"] == "DATACITE:10.57760/sciencedb.27228"]
    hugo = df_all[df_all["raw_id"] == "DATACITE:10.6096/13"]
    millan = df_all[df_all["raw_id"] == "DATACITE:10.6096/1007"]
    pml22 = df_all[df_all["raw_id"] == "DATACITE:10.11888/Terre.tpdc.303314"]
    pml22_vars = df_var_22[df_var_22["raw_id"] == "DATACITE:10.11888/Terre.tpdc.303314"] if len(df_var_22) else pd.DataFrame()
    new_rel_ok = (
        len(jof) == 1 and float(jof.iloc[0]["adjudicated_output_grid_meters"]) == 27830.0
        and jof.iloc[0]["observation_domain_mask"] == "ice_free_ocean_mask"
        and int(jof.iloc[0]["temporal_start_year"]) == 1988 and int(jof.iloc[0]["temporal_end_year"]) == 2022
        and len(coseb) == 1 and float(coseb.iloc[0]["adjudicated_output_grid_meters"]) == 5566.0
        and coseb.iloc[0]["temporal_cadence_class"] == "sub_daily_or_daily"
        and len(hugo) == 1 and hugo.iloc[0]["provider_pool"] == "eu_non_copernicus"
        and float(hugo.iloc[0]["adjudicated_output_grid_meters"]) == 100.0
        and float(hugo.iloc[0]["bbox_west"]) == -180.0 and float(hugo.iloc[0]["bbox_east"]) == 180.0
        and not bool(hugo.iloc[0]["bbox_crosses_antimeridian"])
        and hugo.iloc[0]["temporal_cadence_class"] == "multi_year_periodic"
        and len(millan) == 1 and millan.iloc[0]["provider_pool"] == "eu_non_copernicus"
        and len(pml22) == 1 and float(pml22.iloc[0]["adjudicated_output_grid_meters"]) == 11132.0
        and int(pml22.iloc[0]["temporal_start_year"]) == 1982 and int(pml22.iloc[0]["temporal_end_year"]) == 2024
        and len(pml22_vars) == 3
    )

    s5p_aut = df_all[df_all["raw_id"] == "SENTINEL5P_DAILY_AUT"]
    gee_mcd = df_all[df_all["record_id"] == "27_gee_official_stac::MODIS/061/MCD43A4"]
    pool_sum = pd.read_csv(REPO_ROOT / "03_harmonization" / "provider_pool_and_sentinel_summary.csv")
    life_sum_ok = (
        len(s5p_aut) >= 1 and bool((s5p_aut["operational_maturity_class"] != "operational_ongoing_series").all())
        and len(gee_mcd) == 1 and gee_mcd.iloc[0]["operational_maturity_class"] == "operational_ongoing_series"
        and "sentinel_lineage_status" in pool_sum.columns
        and "unassessed_no_platform_metadata" in set(pool_sum["sentinel_lineage_status"].astype(str))
    )

    qa.append((
        "23. Strict JSONL, CEDA DOIs, antimeridian, variable variants, sublevels, NH SWE, thematic releases & lifecycle",
        bool(strict_jsonl_ok and ceda_ok and antim_var_ok and sublevel_nulls == 0 and nh5_ok and new_rel_ok and life_sum_ok),
        f"jsonl={strict_jsonl_ok}({jsonl_checked}), ceda={ceda_ok}(snow7={snow7_ok}), antim_var={antim_var_ok}, sublevel_null={sublevel_nulls}, nh5={nh5_ok}, new_rel={new_rel_ok}, life_sum={life_sum_ok}",
    ))

    # 24. HTML comparison preservation (JRC 162768e1, ForestCO2 15201316), paired SMAP vs Snow CCI SWE + HTAPv3 geo, GPCP v3.3 & CERES 4.2.1
    import importlib.util as _imp_u
    _pspec = _imp_u.spec_from_file_location("prep_all_mod", str(REPO_ROOT / "02_preparation" / "prepare_all.py"))
    _pmod = _imp_u.module_from_spec(_pspec)
    _pspec.loader.exec_module(_pmod)
    html_fx_ok = (
        _pmod.clean_str("<p>Compare years &lt;2010 and 2019. Sentinel-2 was used in 2019.</p>")
        == "Compare years <2010 and 2019. Sentinel-2 was used in 2019."
        and _pmod.clean_str("<p>Exclude canopy height &lt;5 m. Sentinel-2 supplies the vegetation mask.</p>")
        == "Exclude canopy height <5 m. Sentinel-2 supplies the vegetation mask."
        and _pmod.clean_str("Watch: 0<HotSpot<1 warning threshold & <product_type>_<mission>")
        == "Watch: 0<HotSpot<1 warning threshold & <product_type>_<mission>"
        and _pmod.clean_str("Link (<https://glad.geog.umd.edu/>) end")
        == "Link (<https://glad.geog.umd.edu/>) end"
        and _pmod.clean_url("https://www.wekeo.eu/data?view=dataset&dataset=EO:ECMWF:DAT:DERIVED_UTCI_HISTORICAL")
        == "https://www.wekeo.eu/data?view=dataset&dataset=EO:ECMWF:DAT:DERIVED_UTCI_HISTORICAL"
        and _pmod.clean_url("https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/Geology/iso/xml/G02889.xml&view=getDataView&header=none")
        == "https://data.noaa.gov/metaview/page?xml=NOAA/NESDIS/NGDC/MGG/Geology/iso/xml/G02889.xml&view=getDataView&header=none"
        and _pmod.clean_url("https://example.org/search?a=1&full_datasource=Millennium%20Assessment&ds_id=")
        == "https://example.org/search?a=1&full_datasource=Millennium%20Assessment&ds_id="
    )
    wekeo_utci = df_all[df_all["record_id"] == "09_wekeo_dias::EO:ECMWF:DAT:DERIVED_UTCI_HISTORICAL"]
    noaa_mesa = df_all[df_all["record_id"] == "20_noaa_onestop::6dc622ed-8446-4bc0-af4c-0b6b15bcacdd"]
    wgc_10500 = df_all[df_all["record_id"] == "12_ceos_cgms_wgclimate_ecv::WGCLIMATE_CDR_10500"]
    rw_68455 = df_all[df_all["record_id"] == "23_wri_gfw_and_resourcewatch::RW:68455cb5-bfe3-4528-83a2-00fab1c52fb9"]
    usgs_glccd = df_all[df_all["record_id"] == "16_nasa_cmr::C1220566586-USGS_LTA"]
    url_bare_amp = int(df_all["access_url"].fillna("").str.contains(r"&[A-Za-z0-9_]+$", regex=True).sum())
    url_query_ok = (
        url_bare_amp == 0
        and len(wekeo_utci) == 1 and str(wekeo_utci.iloc[0]["access_url"]).endswith("?view=dataset&dataset=EO:ECMWF:DAT:DERIVED_UTCI_HISTORICAL")
        and len(noaa_mesa) == 1 and str(noaa_mesa.iloc[0]["access_url"]).endswith("&view=getDataView&header=none")
        and len(wgc_10500) == 1 and str(wgc_10500.iloc[0]["access_url"]).endswith("?action=dataset&typeid=21")
        and len(rw_68455) == 1 and str(rw_68455.iloc[0]["access_url"]).endswith("?keywords=co2&start=1920-01-01&end=2017-01-04")
        and len(usgs_glccd) == 1 and "&full_datasource=" in str(usgs_glccd.iloc[0]["access_url"])
    )
    jrc_s2 = df_all[df_all["record_id"] == "07_ec_jrc_data_catalogue::162768e1-570f-462c-b757-37e57085d773"]
    fco2_s2 = df_all[df_all["raw_id"] == "ZENODO:15201316"]
    jrc_s2_ok = (
        len(jrc_s2) == 1
        and "<2010 and 2019" in str(jrc_s2.iloc[0]["description"])
        and bool(jrc_s2.iloc[0]["uses_sentinel_data"])
        and "S2" in str(jrc_s2.iloc[0]["sentinel_missions"])
    )
    fco2_s2_ok = (
        len(fco2_s2) == 1
        and "CH<5m" in str(fco2_s2.iloc[0]["description"])
        and "Sentinel-2" in str(fco2_s2.iloc[0]["description"])
        and bool(fco2_s2.iloc[0]["uses_sentinel_data"])
        and "S2" in str(fco2_s2.iloc[0]["sentinel_missions"])
    )

    smap_gl = df_all[df_all["record_id"] == "16_nasa_cmr::C2938664439-NSIDC_CPRD"]
    htap_gl = df_all[df_all["record_id"] == "07_ec_jrc_data_catalogue::8626f4f4-67e3-4d0a-9457-d432f35ae3da"]
    swe_c_nh = df_canon[df_canon["canonical_product_key"] == "doi:10.5285/edf8abd23f4a40aabd4d52e48dec06ea"]
    paired_geo_ok = (
        len(smap_gl) == 1 and smap_gl.iloc[0]["geographic_coverage_class"] == "global"
        and len(htap_gl) == 1 and htap_gl.iloc[0]["geographic_coverage_class"] == "global"
        and len(swe_c_nh) == 1 and swe_c_nh.iloc[0]["geographic_coverage_class"] == "hemispheric_or_pantropical"
        and swe_c_nh.iloc[0]["observation_domain_class"] == "northern_hemisphere_only"
        and float(swe_c_nh.iloc[0]["bbox_west"]) == -180.0 and float(swe_c_nh.iloc[0]["bbox_east"]) == 180.0
    )

    gpcp_n = df_all[df_all["record_id"] == "20_noaa_onestop::7180fc6a-a5a4-4857-925a-a0fa4f7f7727"]
    ce1 = df_all[df_all["record_id"] == "16_nasa_cmr::C3880496704-LARC_CLOUD"]
    ce2 = df_all[df_all["record_id"] == "16_nasa_cmr::C3880497643-LARC_CLOUD"]
    spot_panel_ok = (
        len(gpcp_n) == 1
        and gpcp_n.iloc[0]["temporal_output_cadences"] == "monthly_or_seasonal"
        and float(gpcp_n.iloc[0]["adjudicated_output_grid_meters"]) == 55660.0
        and len(ce1) == 1 and ce1.iloc[0]["geographic_coverage_class"] == "global"
        and ce1.iloc[0]["geographic_coverage_provenance"] == "provider_documented_specification"
        and len(ce2) == 1 and ce2.iloc[0]["geographic_coverage_class"] == "global"
        and ce2.iloc[0]["geographic_coverage_provenance"] == "provider_documented_specification"
    )

    qa.append((
        "24. HTML comparison & URL query-param preservation, S2 lineage, paired SMAP/SWE geo, NOAA GPCP & CERES EBAF specs",
        bool(html_fx_ok and url_query_ok and jrc_s2_ok and fco2_s2_ok and paired_geo_ok and spot_panel_ok),
        f"html_fx={html_fx_ok}, url_query={url_query_ok}(bare_amp={url_bare_amp}), jrc_s2={jrc_s2_ok}, fco2_s2={fco2_s2_ok}, smap_swe_htap={paired_geo_ok}, gpcp_ceres={spot_panel_ok}",
    ))

    # 25. Non-European primary portals (JAXA, DE Africa/Pacific, NRCan) & cross-pool mirror / L1 leakage hygiene
    s30_q = df_all[df_all["source_catalogue"] == "30_jaxa_earth_stac"]
    s31_q = df_all[df_all["source_catalogue"] == "31_digital_earth_africa_and_pacific"]
    s32_q = df_all[df_all["source_catalogue"] == "32_nrcan_ccmeo_stac"]
    jaxa_c3s_q = s30_q[s30_q["raw_id"] == "Copernicus.C3S_PROBA-V_LCCS_global_yearly"]
    jaxa_aw3d_q = s30_q[s30_q["raw_id"] == "JAXA.EORC_ALOS.PRISM_AW3D30.v4.1_global"]
    jaxa_polar_q = s30_q[s30_q["raw_id"].isin(["JAXA.JASMES_GCOM-W.AMSR2_ic0.v201_north_daily", "JAXA.JASMES_GCOM-W.AMSR2_ic0.v201_south_daily"])]
    dep_q = s31_q[s31_q["raw_id"].astype(str).str.startswith("DEPACIFIC:")]
    nrcan_reg_q = s32_q[s32_q["raw_id"].isin(["modis-land-cover", "canadian-wetland-inventory"])]
    ghsl_all_q = df_canon[(df_canon["title"].fillna("") + " " + df_canon["raw_id"].fillna("")).str.contains(r"\bghs[-_ ]|\bghsl\b|global human settlement", case=False, regex=True)]
    s2gm_q = df_all[df_all["record_id"] == "14_ceos_fedeo_agencies::sentinel-2-global-mosaics"]
    asap_q = df_canon[(df_canon["source_catalogue"] == "15_geobon_and_geokhub") & df_canon["title"].fillna("").str.contains(r"\bASAP\s+(?:Warnings|Hotspots|warnings and hotspots)\b", case=False, regex=True)]
    wsf_q = df_canon[(df_canon["title"].fillna("") + " " + df_canon["raw_id"].fillna("")).str.contains(r"world settlement footprint|\bwsf[-_](?:2015|2019|evolution|3d)\b", case=False, regex=True)]
    wpop_q = df_canon[(df_canon["title"].fillna("") + " " + df_canon["raw_id"].fillna("")).str.contains(r"\bworldpop\b", case=False, regex=True)]
    fedeo_bio_q = df_all[(df_all["source_catalogue"] == "14_ceos_fedeo_agencies") & df_all["title"].fillna("").str.contains("Biomass_cci", case=False)]
    spot_q = df_all[df_all["raw_id"].astype(str).str.startswith("EO:CNES:DAT:SWH")]
    s2cp_q = df_all[df_all["record_id"] == "27_gee_official_stac::COPERNICUS/S2_CLOUD_PROBABILITY"]

    portals_ok = (
        len(s30_q) == 112
        and len(jaxa_c3s_q) == 1 and jaxa_c3s_q.iloc[0]["provider_pool"] == "copernicus_services" and jaxa_c3s_q.iloc[0]["copernicus_service"] == "C3S"
        and len(jaxa_aw3d_q) == 1 and str(jaxa_aw3d_q.iloc[0]["doi"]) == "10.57746/EO.01kkfx0ejw8r5ndc9bg5nt13ws"
        and len(jaxa_polar_q) == 2 and bool((jaxa_polar_q["geographic_coverage_class"] == "hemispheric_or_pantropical").all())
        and len(s31_q) >= 85
        and len(dep_q) == 13 and bool((dep_q["geographic_coverage_class"] == "regional_or_local").all())
        and (dep_q["temporal_start_year"].dropna().empty or float(dep_q["temporal_start_year"].dropna().min()) >= 2017.0)
        and len(s32_q) == 39 and bool((s32_q["provider_pool"] == "non_eu").all())
        and len(nrcan_reg_q) == 2 and bool((nrcan_reg_q["geographic_coverage_class"] == "regional_or_local").all())
    )
    hygiene_ok = (
        len(ghsl_all_q) >= 300 and bool((ghsl_all_q["provider_pool"] == "copernicus_services").all())
        and len(s2gm_q) == 1 and s2gm_q.iloc[0]["provider_pool"] == "copernicus_services" and s2gm_q.iloc[0]["product_level_subtype"] == "L3_ARD_REFLECTANCE"
        and len(asap_q) == 3 and bool((asap_q["provider_pool"] == "eu_non_copernicus").all())
        and len(wsf_q) >= 10 and bool((wsf_q["provider_pool"] == "eu_non_copernicus").all())
        and len(wpop_q) >= 15 and bool((wpop_q["provider_pool"] == "eu_non_copernicus").all())
        and len(fedeo_bio_q) >= 2 and bool((fedeo_bio_q["provider_pool"] == "eu_non_copernicus").all())
        and len(spot_q) == 0 and len(s2cp_q) == 0
    )
    qa.append((
        "25. Non-EU primary portals (JAXA, DE Africa/Pacific, NRCan) & cross-pool mirror / L1 leakage hygiene",
        bool(portals_ok and hygiene_ok),
        f"portals={portals_ok}(jaxa={len(s30_q)}, deap={len(s31_q)}, nrcan={len(s32_q)}), hygiene={hygiene_ok}(ghsl_cop={len(ghsl_all_q)}, spot_l1={len(spot_q)}, s2_cprob={len(s2cp_q)})",
    ))

    return qa


def audit_eu_policy_benchmark(df_all: pd.DataFrame, df_excl: pd.DataFrame) -> pd.DataFrame:
    rid_s = df_all["raw_id"].fillna("")
    ttl_s = df_all["title"].fillna("")
    dsc_s = df_all["description"].fillna("")
    blob_excl = (df_excl["raw_id"].fillna("") + " " + df_excl["title"].fillna("")).str.lower()
    rows = []
    for svc, pname, scope, src_hint, pat, note in EU_POLICY_PRODUCTS_BENCHMARK:
        hit_rid_ttl = rid_s.str.contains(pat, case=False, regex=True) | ttl_s.str.contains(pat, case=False, regex=True)
        hit_desc = dsc_s.str.contains(pat, case=False, regex=True)
        hit_any = hit_rid_ttl | hit_desc

        if src_hint:
            m_src_title = df_all[(df_all["source_catalogue"] == src_hint) & hit_rid_ttl]
            m_any_title = df_all[hit_rid_ttl]
            m_src_all = df_all[(df_all["source_catalogue"] == src_hint) & hit_any]
            if len(m_src_title) > 0:
                m = m_src_title.copy()
            elif len(m_any_title) > 0:
                m = m_any_title.copy()
            elif len(m_src_all) > 0:
                m = m_src_all.copy()
            else:
                m = df_all[hit_any].copy()
        else:
            m = df_all[hit_rid_ttl].copy()
            if len(m) == 0:
                m = df_all[hit_any].copy()

        mex = df_excl[blob_excl.str.contains(pat, regex=True, na=False)]

        if len(m) > 0:
            m["_is_depr"] = (m["operational_maturity_class"] == "deprecated_or_superseded").astype(int)
            r0 = m.sort_values(["_is_depr", "source_authority_rank", "raw_id"]).iloc[0]
            status = "IN_CATALOGUE" if scope == "IN_SCOPE_L3_L4_EO" else "IN_CATALOGUE_TAGGED_SENSITIVITY_OR_RELATED"
            rows.append({
                "service_or_domain": svc,
                "benchmark_product_name": pname,
                "scope_classification": scope,
                "catalogue_status": status,
                "matched_record_count": int(len(m)),
                "primary_source_catalogue": r0["source_catalogue"],
                "sample_raw_id": r0["raw_id"],
                "sample_title": str(r0["title"])[:100],
                "notes": note,
            })
        elif len(mex) > 0:
            r0 = mex.iloc[0]
            rows.append({
                "service_or_domain": svc,
                "benchmark_product_name": pname,
                "scope_classification": scope,
                "catalogue_status": "CORRECTLY_EXCLUDED_IN_AUDIT_LOG",
                "matched_record_count": int(len(mex)),
                "primary_source_catalogue": r0["source_id"],
                "sample_raw_id": r0["raw_id"],
                "sample_title": str(r0["title"])[:100],
                "notes": f"{note} [excluded: {r0['exclusion_reason']}]",
            })
        else:
            status = "EXPECTED_ABSENT_OUT_OF_SCOPE" if scope != "IN_SCOPE_L3_L4_EO" else "MISSING_IN_SCOPE"
            rows.append({
                "service_or_domain": svc,
                "benchmark_product_name": pname,
                "scope_classification": scope,
                "catalogue_status": status,
                "matched_record_count": 0,
                "primary_source_catalogue": None,
                "sample_raw_id": None,
                "sample_title": None,
                "notes": note,
            })
    return pd.DataFrame(rows)


GCOS_ECV_EXPLICIT_PATTERNS: dict[str, str] = {
    "Above-ground biomass": r"\b(?:above[- ]?ground\s+biomass|\bagb\b|forest\s+biomass|woody\s+biomass|biomass\s+density|gedi04_b|cci\s+biomass|globbiomass)\b",
    "Aerosols properties": r"\b(?:aerosols?|aerosol\s+optical\s+depth|(?<!vegetation\s)optical\s+depth\s+of\s+aerosol|\baod\b|\baai\b|absorbing\s+aerosol|particulate\s+matter|pm2\.5|pm10)\b",
    "Albedo": r"\b(?:albedo|surface\s+albedo|black[- ]sky\s+albedo|white[- ]sky\s+albedo|blue[- ]sky\s+albedo|mcd43)\b",
    "Carbon Dioxide, Methane and other greenhouse gases": r"\b(?:carbon\s+dioxide|\bco2\b|\bxco2\b|\bmethane\b|\bch4\b|\bxch4\b|\bn2o\b|nitrous\s+oxide|greenhouse\s+gas(?:es)?)\b",
    "Cloud properties": r"\b(?:cloud\s+(?:properties|fraction|cover|optical|top|mask|phase|water|amount|droplet|effective\s+radius)|isccp|clara[- ]a)\b",
    "Earth radiation budget": r"\b(?:earth\s+radiation\s+budget|top[- ]of[- ]atmosphere.*(?:radiation|flux|irradiance)|outgoing\s+longwave\s+radiation|\bolr\b|\bceres\b|erbe)\b",
    "Evaporation from land": r"\b(?:evapotranspiration|land\s+evaporation|actual\s+evaporation|potential\s+evaporation|canopy\s+transpiration|\bgleam\d?\b|\bmod16\b|\bwapor\b.*evaporation)\b",
    "Fire": r"\b(?:burned\s+area|burnt\s+area|active\s+fire|fire\s+radiative\s+power|\bfrp\b|fire\s+danger|fire\s+weather\s+index|wildfire)\b",
    "Fraction of Absorbed Photosythetically Active Radiation": r"\b(?:fapar|fraction\s+of\s+absorbed\s+photosynthetically\s+active\s+radiation)\b",
    "Glaciers": r"\b(?:glaciers?|glacier\s+(?:mass|area|elevation|extent|velocity|outlines)|randolph\s+glacier)\b",
    "Groundwater": r"\b(?:groundwater|\bg3p\b|aquifer\s+storage)\b",
    "Ice sheets and ice shelves": r"\b(?:ice\s+sheets?|ice\s+shel(?:f|ves)|greenland\s+ice|antarctic\s+ice|grounded\s+ice)\b",
    "Lakes": r"\b(?:lake\s+(?:water|surface|level|extent|quality|temperature|ice)|inland\s+water|global\s+lakes|hydrolakes)\b",
    "Land cover": r"\b(?:land\s+cover|land\s+use|corine|worldcover|dynamic\s+world|glc_fcs|globcover|modis\s+land\s+cover|mcd12)\b",
    "Land-surface temperature": r"\b(?:land[- ]surface\s+temperature|\blst\b|mod11|myd11|vnp21)\b",
    "Leaf area index": r"\b(?:leaf\s+area\s+index|\blai\b|mod15|vnp15)\b",
    "Lightning": r"\b(?:lightning\s+(?:flashes|imaging|detection|flash|density)|lis/otd|\bglm\b.*lightning)\b",
    "Ocean colour": r"\b(?:ocean\s+colou?r|water[- ]leaving\s+radiance|remote[- ]sensing\s+reflectance|\brrs\b|globcolour|chlorophyll[- ]a)\b",
    "Ocean-surface heat flux": r"\b(?:ocean[- ]surface\s+heat\s+flux(?:es)?|air[- ]sea\s+(?:heat\s+)?flux(?:es)?|sea[- ]surface\s+heat\s+flux(?:es)?|(?:latent(?:\s+and\s+sensible)?|sensible)\s+heat\s+flux(?:es)?|\boaflux\b|\btropflux\b|\bj-ofuro3?\b|\bseaflux\b)\b",
    "Ozone": r"\b(?:ozone|\bo3\b|total\s+column\s+ozone|stratospheric\s+ozone|tropospheric\s+ozone)\b",
    "Permafrost": r"\b(?:permafrost|active\s+layer\s+thickness|rock\s+glacier|frozen\s+ground)\b",
    "Plankton": r"\b(?:plankton|phytoplankton|zooplankton|chlorophyll|primary\s+production.*ocean|marine\s+biogeochemistry)\b",
    "Precipitation": r"\b(?:precipitation|rainfall|rain\s+rate|\bgpm\b|\bimerg\b|\bchirps\b|\bgpcp\b|\btrmm\b|\bcmorph\b|\bpersiann\b)\b",
    "Precursors (supporting the aerosol and ozone ECVs)": r"\b(?:nitrogen\s+dioxide|\bno2\b|sulfur\s+dioxide|sulphur\s+dioxide|\bso2\b|formaldehyde|\bhcho\b|carbon\s+monoxide|\bco\b\s+column|aerosol\s+precursors?|ozone\s+precursors?)\b",
    "River Discharge": r"\b(?:river\s+discharge|streamflow|runoff|glofas|efas)\b",
    "Sea ice": r"\b(?:sea[- ]ice|ice\s+concentration|ice\s+extent|ice\s+thickness|ice\s+drift|ice\s+edge)\b",
    "Sea level": r"\b(?:sea[- ]level|sea\s+surface\s+height|\bssh\b|sea\s+level\s+anomaly|mean\s+dynamic\s+topography|\bmdt\b)\b",
    "Sea state": r"\b(?:sea\s+state|significant\s+wave\s+height|ocean\s+waves?|wave\s+period|swell\s+height)\b",
    "Sea-surface salinity": r"\b(?:sea[- ]surface\s+salinity|\bsss\b|ocean\s+salinity|(?:smos|smap|aquarius).{0,35}salinity)\b",
    "Sea-surface temperature": r"\b(?:sea[- ]surface\s+temperature|\bsst\b|ghrsst|oisst|ostia|pathfinder\s+sst)\b",
    "Snow": r"\b(?:snow\s+(?:cover|water\s+equivalent|depth|extent|melt|albedo|fraction)|\bswe\b|\bfsca\b|mod10|myd10)\b",
    "Soil moisture": r"\b(?:soil\s+moisture|soil\s+water\s+index|\bswi\b|\bssm\b|root[- ]zone\s+soil\s+moisture)\b",
    "Surface currents": r"\b(?:surface\s+currents?|ocean\s+currents?|geostrophic\s+currents?|oscar.*currents?|globcurrent)\b",
    "Surface radiation budget": r"\b(?:surface\s+radiation\s+budget|surface\s+solar\s+irradiance|downward\s+shortwave|downward\s+longwave|(?<!absorbed\s)(?<!absorbed\s\bof\s)(?<!fraction\s)(?<!fraction\sabsorbed\s)photosynthetically\s+active\s+radiation(?!\s*\(?fapar\b)|surface\s+net\s+radiation)\b",
    "Surface stress": r"\b(?:surface\s+wind\s+stress|ocean\s+wind\s+stress|wind\s+stress\s+curl|momentum\s+flux)\b",
    "Surface wind speed and direction": r"\b(?:surface\s+winds?|10[- ]?m(?:eter)?\s+winds?|ocean\s+surface\s+winds?|scatterometer\s+winds?|ccmp\s+winds?|ascat.*winds?|quikscat)\b",
    "Terrestrial Water Storage": r"\b(?:terrestrial\s+water\s+storage|\btws\b|liquid\s+water\s+equivalent\s+thickness|grace.*mascon|grace.*tellus)\b",
    "Upper-air Water Vapour": r"\b(?:water\s+vapou?r|total\s+column\s+water\s+vapou?r|\btcwv\b|precipitable\s+water|upper[- ]tropospher(?:e|ic)\s+humidity|specific\s+humidity\s+profile)\b",
    "Upper-air Wind Speed and Direction": r"\b(?:upper[- ]air\s+winds?|atmospheric\s+motion\s+vectors?|\bamvs?\b|tropospheric\s+wind|stratospheric\s+wind|wind\s+profile|aeolus)\b",
    "Upper-air temperature": r"\b(?:upper[- ]air\s+temperature|atmospheric\s+temperature\s+profile|stratospheric\s+temperature|tropospheric\s+temperature|msu/amsu|radio\s+occultation.*temperature)\b",
}


def audit_gcos_ecv(df_canon: pd.DataFrame) -> pd.DataFrame:
    ecv_path = RAW_DIR / "12_ceos_cgms_wgclimate_ecv" / "data" / "wgclimate_ecv_data_records.json"
    ecv_data = json.loads(ecv_path.read_text(encoding="utf-8"))
    ecv_names: dict[str, int] = {}
    for rec in ecv_data:
        found_ecvs = set()
        for vn in (rec.get("variables_nodes") or []):
            vnode = (vn or {}).get("variables_node") or {}
            vtype = (vnode.get("type") or {}).get("name")
            if vtype == "ECV" and vnode.get("name"):
                found_ecvs.add(str(vnode["name"]).strip())
        for ecv in (found_ecvs or {"Other"}):
            ecv_names[ecv] = ecv_names.get(ecv, 0) + 1

    blob = (df_canon["title"].fillna("") + " " + df_canon["keywords"].fillna("")).str.lower()
    # Extract leading deliverable sentences from descriptions (stripping citation/reference sentences) for domain-qualified heat flux matching
    desc_lead = (
        df_canon["description"]
        .fillna("")
        .apply(lambda s: " ".join(re.split(r"(?<=[.!?])\s+|\n+", str(s))[:3]))
        .str.lower()
    )
    blob_heat = blob + " " + desc_lead
    is_non_wgc = df_canon["source_catalogue"] != "12_ceos_cgms_wgclimate_ecv"
    rows = []
    for ecv, n_wg in sorted(ecv_names.items(), key=lambda x: (-x[1], x[0])):
        pat = GCOS_ECV_EXPLICIT_PATTERNS.get(ecv)
        if not pat:
            pat = r"\b" + re.escape(ecv.lower()) + r"\b"
        hit_mask = blob.str.contains(pat, case=False, regex=True, na=False)
        if ecv == "Surface radiation budget":
            ttl_low = df_canon["title"].fillna("").str.lower()
            has_rad_budget = blob.str.contains(
                r"\b(?:surface\s+radiation\s+budget|surface\s+solar\s+irradiance|downward\s+shortwave|downward\s+longwave|surface\s+net\s+radiation)\b",
                regex=True,
                na=False,
            )
            veg_fapar_mask = blob.str.contains(
                r"\b(?:fapar|fpar|fraction\s+(?:of\s+)?absorbed(?:\s+of)?\s+photosynthetically\s+active\s+radiation|leaf\s+area\s+index)\b",
                regex=True,
                na=False,
            )
            rad_ttl_mask = ttl_low.str.contains(
                r"\b(?:surface\s+(?:solar\s+)?(?:radiation|irradiance)|downward\s+(?:shortwave|longwave)|net\s+radiation|radiation\s+budget|solar\s+irradiance)\b",
                regex=True,
                na=False,
            )
            hit_mask = hit_mask & (~veg_fapar_mask | has_rad_budget | rad_ttl_mask)
            hit_mask = hit_mask & ~(
                blob.str.contains(
                    r"\b(?:fapar|fraction\s+(?:of\s+)?absorbed(?:\s+of)?\s+photosynthetically\s+active\s+radiation)\b",
                    regex=True,
                    na=False,
                )
                & ~rad_ttl_mask
            )
        elif ecv == "Ocean-surface heat flux":
            hit_heat = blob_heat.str.contains(pat, regex=True, na=False)
            is_marine = blob_heat.str.contains(r"\b(?:ocean|marine|sea[- ]surface|air[- ]sea|oaflux|tropflux|j-ofuro3?)\b", regex=True, na=False)
            is_land_et = blob_heat.str.contains(
                r"\b(?:evapotranspiration|mod16|myd16|vnp17|vj117|vj217|terrestrial|land\s+surface|land\s+energy\s+balance|"
                r"soil\s+heat\s+flux|coseb|canopy|ameriflux|fluxnet|nacp|permafrost|aircraft\s+flux|fife|ground\s+validation|icepop)\b",
                regex=True,
                na=False,
            )
            is_carbon_flux = blob_heat.str.contains(
                r"\b(?:co2\s+flux(?:es)?|sea[- ]air\s+co2|carbon\s+flux(?:es)?|pco2|fco2)\b",
                regex=True,
                na=False,
            ) & ~blob_heat.str.contains(r"\b(?:latent(?:\s+and\s+sensible)?|sensible)\s+heat\s+flux(?:es)?\b|\bocean[- ]surface\s+heat\s+flux", regex=True, na=False)
            hit_mask = hit_heat & is_marine & ~is_land_et & ~is_carbon_flux
        elif ecv == "Plankton":
            ttl_low = df_canon["title"].fillna("").str.lower()
            is_land_veg_pigment = blob.str.contains(
                r"\b(?:landvi|land\s+vegetation\s+indic(?:es|ex)|terrestrial\s+vegetation|vegetation\s+index|"
                r"chlorophyll\s+index\s+red\s+edge|chlorophyll[- ]carotenoid|leaf\s+chlorophyll|canopy\s+chlorophyll|"
                r"ndvi|evi|fapar|leaf\s+area\s+index|land\s+surface)\b",
                regex=True,
                na=False,
            ) & ~blob.str.contains(
                r"\b(?:phytoplankton|zooplankton|marine\s+plankton|algal\s+bloom|ocean\s+chlorophyll|chlorophyll[- ]a\s+concentration)\b",
                regex=True,
                na=False,
            )
            is_marine_bio = blob.str.contains(
                r"\b(?:phytoplankton|zooplankton|plankton|ocean|marine|sea\b|coastal|water|estuar|aquatic|pelagic|chlorophyll[- ]a|oc[- ]cci|globcolour|seawifs|olci|modis[- ]?aqua)\b",
                regex=True,
                na=False,
            ) & ~ttl_low.str.contains(r"\bland\s+vegetation\b|\blandvi\b", regex=True, na=False)
            hit_mask = hit_mask & is_marine_bio & ~is_land_veg_pigment
        elif ecv == "Sea-surface salinity":
            ttl_low = df_canon["title"].fillna("").str.lower()
            hit_mask = hit_mask & ~(
                ttl_low.str.contains(r"\b(?:soil\s+moisture|wetland|forest)\b", regex=True, na=False)
                & ~ttl_low.str.contains(r"\b(?:salinity|sss)\b", regex=True, na=False)
            )
        elif ecv == "Sea-surface temperature":
            ttl_low = df_canon["title"].fillna("").str.lower()
            hit_mask = hit_mask & ~(
                ttl_low.str.contains(r"\b(?:wetland|soil\s+organic|fire\s+weather)\b", regex=True, na=False)
                & ~ttl_low.str.contains(r"\b(?:sea[- ]surface\s+temperature|sst)\b", regex=True, na=False)
            )
        matched = int(hit_mask.sum())
        matched_non_wgc = int((hit_mask & is_non_wgc).sum())
        rows.append({
            "gcos_ecv_name": ecv,
            "wgclimate_inventory_records": n_wg,
            "canonical_catalogue_matches": matched,
            "independent_non_wgclimate_matches": matched_non_wgc,
            "matching_pattern": pat,
        })
    return pd.DataFrame(rows)


def audit_independent_version_benchmark(df_all: pd.DataFrame) -> pd.DataFrame:
    bench_path = BASE_DIR / "independent_product_version_benchmark.csv"
    df_bench = pd.read_csv(bench_path, dtype=str).fillna("")
    rows = []
    for _, spec in df_bench.iterrows():
        id_type = spec["identifier_type"]
        pat = spec["identifier_pattern"]
        if id_type == "doi":
            sub = df_all[df_all["doi"].fillna("").str.contains(pat, case=False, regex=True)]
        elif id_type == "raw_id":
            sub = df_all[df_all["raw_id"].fillna("").str.contains(pat, case=False, regex=True)]
        elif id_type == "record_id":
            sub = df_all[df_all["record_id"].fillna("").str.contains(pat, case=False, regex=True)]
        else:
            sub = df_all[df_all["title"].fillna("").str.contains(pat, case=False, regex=True)]

        min_m = int(spec["expected_min_matches"] or "1")
        recalled = len(sub) >= min_m
        field_mismatches: list[str] = []
        if recalled:
            if spec["expected_processing_level"]:
                if not bool((sub["processing_level"].fillna("") == spec["expected_processing_level"]).all()):
                    field_mismatches.append(f"level={list(sub['processing_level'].unique())}")
            if spec["expected_is_ard"]:
                exp_ard = spec["expected_is_ard"].lower() == "true"
                if not bool((sub["is_ard_reflectance_only"] == exp_ard).all()):
                    field_mismatches.append(f"ard={list(sub['is_ard_reflectance_only'].unique())}")
            if spec["expected_cadence"]:
                if not bool((sub["temporal_cadence_class"].fillna("") == spec["expected_cadence"]).all()):
                    field_mismatches.append(f"cadence={list(sub['temporal_cadence_class'].unique())}")
            if spec["expected_geo_not_global"].lower() == "true":
                if bool((sub["geographic_coverage_class"] == "global").any()):
                    field_mismatches.append(f"geo={list(sub['geographic_coverage_class'].unique())}")
            if spec["expected_uses_sentinel"]:
                exp_s = spec["expected_uses_sentinel"].lower() == "true"
                if not bool((sub["uses_sentinel_data"] == exp_s).all()):
                    field_mismatches.append(f"sentinel={list(sub['uses_sentinel_data'].unique())}")
            if spec["expected_maturity"]:
                if not bool((sub["operational_maturity_class"] == spec["expected_maturity"]).all()):
                    field_mismatches.append(f"maturity={list(sub['operational_maturity_class'].unique())}")
            if spec.get("expected_geo_class"):
                if not bool((sub["geographic_coverage_class"].fillna("") == spec["expected_geo_class"]).all()):
                    field_mismatches.append(f"geo_class={list(sub['geographic_coverage_class'].unique())}")
            if spec.get("expected_output_cadence"):
                if not bool((sub["temporal_output_cadences"].fillna("") == spec["expected_output_cadence"]).all()):
                    field_mismatches.append(f"out_cad={list(sub['temporal_output_cadences'].unique())}")
            if spec.get("expected_adjudicated_grid_meters"):
                exp_g = float(spec["expected_adjudicated_grid_meters"])
                if not bool((sub["adjudicated_output_grid_meters"].fillna(-1.0).sub(exp_g).abs() < 0.05).all()):
                    field_mismatches.append(f"adj_grid={list(sub['adjudicated_output_grid_meters'].unique())}")
            if spec.get("expected_sentinel_status"):
                if not bool((sub["sentinel_lineage_status"].fillna("") == spec["expected_sentinel_status"]).all()):
                    field_mismatches.append(f"sent_status={list(sub['sentinel_lineage_status'].unique())}")
            if spec.get("expected_pool"):
                if not bool((sub["provider_pool"].fillna("") == spec["expected_pool"]).all()):
                    field_mismatches.append(f"pool={list(sub['provider_pool'].unique())}")
            if spec.get("expected_domain_class"):
                if not bool((sub["observation_domain_class"].fillna("") == spec["expected_domain_class"]).all()):
                    field_mismatches.append(f"domain={list(sub['observation_domain_class'].unique())}")
            if spec.get("expected_domain_mask"):
                if not bool((sub["observation_domain_mask"].fillna("") == spec["expected_domain_mask"]).all()):
                    field_mismatches.append(f"mask={list(sub['observation_domain_mask'].unique())}")

        status = "PASS" if (recalled and not field_mismatches) else ("MISSING" if not recalled else "FIELD_MISMATCH")
        rows.append({
            "case_id": spec["case_id"],
            "benchmark_group": spec.get("benchmark_group") or "core_version_panel",
            "product_name": spec["product_name"],
            "matched_records": len(sub),
            "recalled": recalled,
            "scientific_attributes_ok": len(field_mismatches) == 0,
            "status": status,
            "mismatches": "; ".join(field_mismatches) if field_mismatches else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df_all = pd.read_parquet(HARM_DIR / "eo_l3_l4_all_catalogue_records.parquet")
    df_canon = pd.read_parquet(HARM_DIR / "eo_l3_l4_canonical_products.parquet")
    df_excl = pd.read_parquet(PREP_DIR / "excluded_records_log.parquet")

    print("=" * 80)
    print("CATALOGUE QA & EXTERNAL BENCHMARK VALIDATION SUITE")
    print("=" * 80)
    print(f"Total harmonized L3/L4 records: {len(df_all):,}")
    print(f"Unique canonical L3/L4 products: {len(df_canon):,}")
    print(f"Logged excluded records:         {len(df_excl):,}")

    # 1. Internal QA assertions
    print("\n--- 1. INTERNAL QUALITY ASSURANCE ASSERTIONS ---")
    qa = run_internal_qa(df_all, df_canon)
    failures = 0
    for name, ok, detail in qa:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{tag}] {name:56s} | {detail}")

    # 2. EU Policy & Operational Service Product Benchmark
    print(f"\n--- 2. EXTERNAL BENCHMARK 1: EU POLICY & OPERATIONAL SERVICE PRODUCT FAMILIES (N={len(EU_POLICY_PRODUCTS_BENCHMARK)}) ---")
    df_a2 = audit_eu_policy_benchmark(df_all, df_excl)
    df_a2.to_csv(BASE_DIR / "eu_policy_products_benchmark_audit.csv", index=False)
    print(df_a2["catalogue_status"].value_counts().to_string())
    in_scope = df_a2[df_a2["scope_classification"] == "IN_SCOPE_L3_L4_EO"]
    in_scope_found = in_scope[in_scope["catalogue_status"] == "IN_CATALOGUE"]
    print(f"In-scope L3/L4 EO coverage: {len(in_scope_found)} / {len(in_scope)} ({len(in_scope_found)/len(in_scope)*100:.1f}%)")

    # 3. GCOS ECV Coverage
    print("\n--- 3. EXTERNAL BENCHMARK 2: GCOS ESSENTIAL CLIMATE VARIABLES ---")
    df_ecv = audit_gcos_ecv(df_canon)
    df_ecv.to_csv(BASE_DIR / "gcos_ecv_domain_coverage.csv", index=False)
    print(f"Audited {len(df_ecv)} GCOS/WGClimate ECV groups; {(df_ecv['canonical_catalogue_matches'] > 0).sum()} / {len(df_ecv)} have canonical matches.")

    # Verify Ocean-surface heat flux positive & negative controls strictly
    hf_pat = str(df_ecv[df_ecv["gcos_ecv_name"] == "Ocean-surface heat flux"].iloc[0]["matching_pattern"])
    blob_c = (df_canon["title"].fillna("") + " " + df_canon["keywords"].fillna("")).str.lower()
    desc_c = df_canon["description"].fillna("").apply(lambda s: " ".join(re.split(r"(?<=[.!?])\s+|\n+", str(s))[:3])).str.lower()
    bh_c = blob_c + " " + desc_c
    hf_mask = (
        bh_c.str.contains(hf_pat, regex=True, na=False)
        & bh_c.str.contains(r"\b(?:ocean|marine|sea[- ]surface|air[- ]sea|oaflux|tropflux|j-ofuro3?)\b", regex=True, na=False)
        & ~bh_c.str.contains(r"\b(?:evapotranspiration|mod16|myd16|vnp17|vj117|vj217|terrestrial|land\s+surface|land\s+energy\s+balance|soil\s+heat\s+flux|coseb|canopy|ameriflux|fluxnet|nacp|permafrost|aircraft\s+flux|fife|ground\s+validation|icepop)\b", regex=True, na=False)
        & ~(bh_c.str.contains(r"\b(?:co2\s+flux(?:es)?|sea[- ]air\s+co2|carbon\s+flux(?:es)?|pco2|fco2)\b", regex=True, na=False) & ~bh_c.str.contains(r"\b(?:latent(?:\s+and\s+sensible)?|sensible)\s+heat\s+flux(?:es)?\b|\bocean[- ]surface\s+heat\s+flux", regex=True, na=False))
    )
    hf_ids = set(df_canon[hf_mask]["raw_id"].astype(str))
    hf_ctrl_ok = (
        "C1995869798-GHRC_DAAC" in hf_ids
        and "DATACITE:10.20783/dias.667" in hf_ids
        and "ZENODO:8280457" not in hf_ids
        and "DATACITE:10.57760/sciencedb.27228" not in hf_ids
    )
    print(f"Ocean-surface heat flux controls (SeaFlux+J-OFURO3 in; CO2+CoSEB out): {hf_ctrl_ok} (matched={len(hf_ids)})")
    if not hf_ctrl_ok:
        failures += 1

    # 4. Specified Regression & Holdout Case Suite (does not estimate catalogue-wide statistical completeness or field accuracy)
    print("\n--- 4. EXTERNAL BENCHMARK 3: SPECIFIED REGRESSION & HOLDOUT CASES ---")
    df_ind = audit_independent_version_benchmark(df_all)
    df_ind.to_csv(BASE_DIR / "independent_product_version_audit.csv", index=False)
    n_rec = int(df_ind["recalled"].sum())
    n_attr = int((df_ind["recalled"] & df_ind["scientific_attributes_ok"]).sum())
    n_pass = int((df_ind["status"] == "PASS").sum())
    print(f"Specified cases matched: {n_rec} / {len(df_ind)} | Checked fields matched: {n_attr} / {len(df_ind)} | Specified regression cases passed: {n_pass} / {len(df_ind)}")
    for grp_name, gdf in df_ind.groupby("benchmark_group"):
        g_rec = int(gdf["recalled"].sum())
        g_attr = int((gdf["recalled"] & gdf["scientific_attributes_ok"]).sum())
        print(f"  - {grp_name:25s}: matched={g_rec}/{len(gdf)}, fields_ok={g_attr}/{len(gdf)}")
    ind_fails = df_ind[df_ind["status"] != "PASS"]
    if len(ind_fails) > 0:
        failures += len(ind_fails)
        for _, fr in ind_fails.iterrows():
            print(f"  [FAIL] {fr['case_id']}: status={fr['status']}, mismatches={fr['mismatches']}")

    if failures > 0:
        sys.exit(1)
    print("\nALL INTERNAL QA ASSERTIONS & BENCHMARKS PASSED.")


if __name__ == "__main__":
    main()
