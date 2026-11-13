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
1. Internal QA Suite (6 structural & semantic consistency assertions against `03_harmonization/data/`).
2. External Benchmark 1: Curated EU Policy & Operational Service Product Families (80 benchmark families).
3. External Benchmark 2: GCOS / CEOS-CGMS WGClimate Essential Climate Variable (ECV) thematic coverage.

Outputs:
- `04_validation_and_review/eu_policy_products_benchmark_audit.csv`
- `04_validation_and_review/gcos_ecv_domain_coverage.csv`
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
    ("CLMS", "HRL Croplands - Crop Types (CT) & Cropping Patterns (CP)", "IN_SCOPE_L3_L4_EO", "02_copernicus_clms", r"crop types|cropping patterns", "Crop Types & Cropping Patterns (10 m, Europe) indexed in CLMS"),
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
    ("Eurostat/MS", "Eurostat (env_nwat, NUTS2 Pop, LUCAS) & Member State LPIS", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", None, r"\blucas\b|\blpis\b|\beurostat\b", "Administrative/statistical tabular surveys and Member-State cadastral parcels (non-EO reference)"),
    ("External EO", "VIIRS Day/Night Band Nighttime Lights (VNP46)", "IN_SCOPE_L3_L4_EO", "16_nasa_cmr", r"VNP46|nighttime lights", "NASA Black Marble VNP46 L3 & NOAA VIIRS DNB indexed in NASA CMR & GEE"),
    ("External EO", "USGS Landsat Collection 2 Level-3 Science Products", "IN_SCOPE_L3_L4_EO", "21_usgs_landsat_and_eros", r"Landsat Collection 2|surface temperature", "USGS Landsat C2 L3 BA/DSWE/fSCA collections indexed (L1/L2 scenes excluded by L3/L4 rule)"),
    ("External EO", "Global Gravity-based Groundwater Product (G3P / GRACE TWS)", "IN_SCOPE_L3_L4_EO", None, r"DATACITE:10\.5880/g3p|gravity-based groundwater|groundwater storage change data from 2002", "Horizon 2020 G3P indexed via GFZ DataCite API + GRACE/C3S TWS in CDS & NASA CMR"),
    ("External/Model", "EDGAR & CEDS Emission Inventories", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", "07_ec_jrc_data_catalogue", r"\bedgar\b|\bceds\b", "Inventory-based bottom-up emissions indexed under `non_eo_inventory_or_census` (`strict_eo_c1_eligible=False`)"),
    ("External/Non-EO", "EUBUCCO, OpenStreetMap, Building Stock Observatory, FloraVeg.EU, MeteoAlarm", "OUT_OF_SCOPE_NON_EO_STATISTICAL_OR_CADASTRAL", None, r"\beubucco\b|\bopenstreetmap\b|\bfloraveg\b|\bmeteoalarm\b", "Non-EO building registries, crowd-sourced OSM vectors, botanical checklists, meteorological warnings"),
    ("CDSE L1/L2", "CDSE Sentinel Level-1 and Level-2 Operational Imagery", "OUT_OF_SCOPE_L1_L2_RAW", "10_cdse_stac", r"^sentinel-[1235]-.*l[12]", "Excluded in `02_preparation` (`excluded_records_log.parquet`) by L3/L4 eligibility filter"),
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
    bad_title_res = both_res[(res_ratio > 10.0) | (res_ratio < 0.1)]
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
        "13. High-recall rescued EO series present in canonical",
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
    rows = []
    for ecv, n_wg in sorted(ecv_names.items(), key=lambda x: -x[1]):
        kws = [w for w in re.findall(r"[a-z]+", ecv.lower()) if len(w) >= 4 and w not in ("surface", "earth", "ocean", "above", "water", "land", "other")]
        pat = "|".join(kws[:2]) if kws else re.escape(ecv.lower())
        matched = int(blob.str.contains(pat, regex=True, na=False).sum())
        rows.append({
            "gcos_ecv_name": ecv,
            "wgclimate_inventory_records": n_wg,
            "canonical_catalogue_matches": matched,
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
    print("\n--- 2. EXTERNAL BENCHMARK 1: EU POLICY & OPERATIONAL SERVICE PRODUCT FAMILIES (N=80) ---")
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

    if failures > 0:
        sys.exit(1)
    print("\nALL INTERNAL QA ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
