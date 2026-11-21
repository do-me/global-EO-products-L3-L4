#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "beautifulsoup4>=4.12.0",
#   "openpyxl>=3.1.2",
#   "pandas>=2.2.0",
#   "pdfplumber>=0.11.0",
# ]
# ///
"""
05_external_policy_validation/02_extract_products.py
====================================================
Deterministically parses all 9 downloaded external EO-to-policy / treaty / law
benchmarks under `raw/` into a harmonized, auditable product inventory:
  - `results/extracted_policy_products.csv`
  - `results/extracted_policy_products.jsonl`

Every row traces back to its exact primary file, section/page/row, policy or
treaty framework, raw product name, extracted DOI/URL, and transparent
`scope_class`:
  - IN_SCOPE_OPEN_L3_L4_EO             : Open, released gridded L3/L4 satellite/model EO product
  - OUT_OF_SCOPE_RAW_L1_L2_OR_SENSOR   : Raw L0/L1/L2 imagery, ungridded swaths, or generic sensor archives
  - OUT_OF_SCOPE_IN_SITU_OR_TABULAR    : Ground/buoy networks, censuses, OSM vectors, bottom-up surveys/tables
  - OUT_OF_SCOPE_COMMERCIAL_OR_RESTRICTED : Commercial/proprietary VHR or licensed-only products
  - OUT_OF_SCOPE_PLANNED_OR_TOOL       : Pre-launch/future missions, forecast feeds, or software/platforms

Run via:
    uv run 05_external_policy_validation/02_extract_products.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
RES_DIR = BASE_DIR / "results"

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.I)


def _norm_doi(text: str | None) -> str | None:
    if not text:
        return None
    m = DOI_RE.search(str(text))
    if not m:
        return None
    d = m.group(1).rstrip(".,;:)\\]\"'").split("?")[0].split("#")[0]
    return d.lower()


def _extract_urls(text: str | None) -> str:
    if not text:
        return ""
    urls = re.findall(r"https?://[^\s,;\"'<>)\]]+", str(text))
    return "; ".join(dict.fromkeys(urls))


def _row(
    benchmark_id: str,
    benchmark_name: str,
    policy_framework: str,
    policy_target_or_indicator: str,
    product_name: str,
    provider_or_source: str,
    scope_class: str,
    scope_rationale: str,
    source_file: str,
    source_locator: str,
    doi: str | None = None,
    url: str = "",
    raw_identifier: str = "",
    spatial_res_raw: str = "",
    kpso_impact_count: int | None = None,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_name,
        "policy_framework": policy_framework,
        "policy_target_or_indicator": policy_target_or_indicator,
        "product_name": re.sub(r"\s+", " ", product_name).strip(),
        "raw_identifier": raw_identifier.strip(),
        "provider_or_source": re.sub(r"\s+", " ", provider_or_source).strip(),
        "spatial_res_raw": re.sub(r"\s+", " ", spatial_res_raw).strip(),
        "doi": doi or "",
        "url": url.strip(),
        "scope_class": scope_class,
        "scope_rationale": scope_rationale,
        "kpso_impact_count": kpso_impact_count if kpso_impact_count is not None else "",
        "source_file": source_file,
        "source_locator": source_locator,
    }


# ---------------------------------------------------------------------------
# 1. CEOS-UNFCCC Global Stocktake (GST) Portal (Paris Agreement)
# ---------------------------------------------------------------------------
def extract_01_ceos_gst() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d = RAW_DIR / "01_ceos_gst_portal"
    page_specs = [
        ("carbon-dioxide.html", "GHG / Top-Down CO2 Budgets", "OCO-2 v10 MIP Pilot Top-Down CO2 Budgets", "NASA / NOAA / CEOS WGClimate"),
        ("methane.html", "GHG / Top-Down CH4 Budgets", "GOSAT / TROPOMI Pilot Top-Down Methane Budgets", "ESA / JAXA / NASA / CEOS WGClimate"),
        ("GEDI-biomass.html", "AFOLU / Above-Ground Biomass", "GEDI L4B Gridded Aboveground Biomass Density", "NASA / UMD / ORNL DAAC"),
        ("cci-biomass.html", "AFOLU / Above-Ground Biomass", "ESA CCI Biomass Global Aboveground Biomass", "ESA CCI"),
        ("icesat2-boreal-biomass.html", "AFOLU / Above-Ground Biomass", "ICESat-2 Boreal Aboveground Biomass 2020 Map", "NASA / ORNL DAAC"),
        ("jpl-biomass.html", "AFOLU / Above-Ground Biomass", "JPL 2020 Global Forest Biomass Dataset", "NASA JPL"),
        ("africa-biomass.html", "AFOLU / Above-Ground Biomass", "NCEO Africa Aboveground Woody Biomass 2020", "NCEO / Univ. of Leicester"),
        ("forest.html", "AFOLU / Forests", "Global Forest Watch / Hansen UMD Global Tree Cover Loss", "UMD GLAD / WRI / USGS / NASA"),
        ("HILDAplus.html", "AFOLU / Land Cover & Land Use", "HILDA+ Global Land Use Change (1960-2019)", "Wageningen Univ. / PANGAEA"),
        ("cci-lc.html", "AFOLU / Land Cover & Land Use", "ESA CCI Land Cover (300 m Annual)", "ESA CCI / C3S"),
        ("copernicus.html", "AFOLU / Land Cover & Land Use", "Copernicus Global Land Service 100 m Land Cover (CGLS-LC100)", "Copernicus CLMS"),
        ("worldcover.html", "AFOLU / Land Cover & Land Use", "ESA WorldCover 10 m Global Land Cover (2020/2021)", "ESA"),
        ("agriculture.html", "AFOLU / Agriculture", "ESA WorldCereal 10 m Global Cropland & Crop Type", "ESA"),
        ("mangrove.html", "AFOLU / Wetlands & Mangroves", "Global Mangrove Watch (GMW)", "JAXA / Aberystwyth / Wetlands Intl"),
        ("fire.html", "AFOLU / Fire Disturbances", "Sentinel-3 World Fire Atlas (S3 WFA)", "ESA"),
    ]
    for fname, sector, fallback_name, provider in page_specs:
        fp = d / fname
        html = fp.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        h1 = soup.find("h1")
        h2 = soup.find("h2")
        heading = (h1.get_text(" ", strip=True) if h1 else "") or (h2.get_text(" ", strip=True) if h2 else "")
        # Keep descriptive name while recording exact page heading
        name = fallback_name if heading.lower() in ("carbon dioxide", "methane", "forests", "agriculture", "mangroves", "fire", "") else heading
        # Extract primary dataset DOI (avoid journal article DOIs unless dataset DOI)
        ext_links = sorted(
            set(
                a.get("href", "").strip()
                for a in soup.find_all("a", href=True)
                if a["href"].startswith("http") and "ceos.org" not in a["href"] and "twitter.com" not in a["href"]
            )
        )
        # Pick dataset DOI (PANGAEA, Zenodo, Figshare, Leicester, ORNL, etc.)
        doi = None
        for lnk in ext_links:
            cand = _norm_doi(lnk)
            if cand and not any(j in cand for j in ("10.5194/", "10.3390/", "10.1038/")):
                doi = cand
                break
        if fname == "carbon-dioxide.html":
            # Page explicitly cites dataset DOI 10.48588/npf6-sw92
            m = re.search(r"10\.48588/npf6-sw92", html, re.I)
            if m:
                doi = m.group(0).lower()
        rows.append(
            _row(
                benchmark_id="01_ceos_gst",
                benchmark_name="CEOS-UNFCCC Global Stocktake (GST) Portal",
                policy_framework="UNFCCC Paris Agreement (Global Stocktake)",
                policy_target_or_indicator=sector,
                product_name=name,
                provider_or_source=provider,
                scope_class="IN_SCOPE_OPEN_L3_L4_EO",
                scope_rationale="Open gridded/global L3/L4 satellite EO product curated by CEOS for Paris Agreement GST",
                source_file=f"raw/01_ceos_gst_portal/{fname}",
                source_locator=f"HTML page heading: {heading or fname}",
                doi=doi,
                url="; ".join(ext_links[:4]),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 2. CEOS SDG Support Sheets (v2023-2025): SDG 6.6.1, 11.3.1, 14.1.1a, 15.3.1
# ---------------------------------------------------------------------------
def extract_02_ceos_sdg_sheets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d = RAW_DIR / "02_ceos_sdg_support_sheets"

    def _pdf_text(fname: str) -> str:
        parts: list[str] = []
        with pdfplumber.open(d / fname) as doc:
            for p in doc.pages:
                parts.append(p.extract_text() or "")
                for tb in p.extract_tables() or []:
                    for row in tb:
                        for cell in row:
                            if cell:
                                parts.append(str(cell))
        return re.sub(r"\s+", " ", " ".join(parts))

    # Verify every extracted product string directly in the downloaded PDF text (reproducible provenance!)
    specs = [
        # SDG 6.6.1 (Water-related ecosystems, UNEP & Ramsar, Nov 2023)
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "JRC Global Surface Water Explorer (GSWE)", "EC JRC", r"Global Surface Water", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open 30 m Landsat/Sentinel global surface water L3 product"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "DHI Global Wetland Map (2016-2018)", "DHI / UNEP", r"DHI global wetland map", "10-100 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global EO wetland extent product cited as default in SDG 6.6.1 sheet"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "Global Mangrove Watch (GMW)", "JAXA / Wetlands Intl", r"Global Mangrove Watch", "25 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open L-band SAR + optical global mangrove extent L3 time series"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "Copernicus Global Land Service Lake Water Quality (LWQ)", "Copernicus CLMS", r"Water quality of lakes|land\.copernicus\.eu/global/products/lwq", "100-1000 m", "IN_SCOPE_OPEN_L3_L4_EO", "Operational CLMS global lake turbidity/TSI/chlorophyll L3 product"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "ESA CCI Lakes", "ESA CCI", r"CCI Lakes", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "ESA CCI global lake ECV L3/L4 time series"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "Hydroweb THEIA Inland Water Levels", "CNES / Data Terra THEIA", r"Hydroweb THEIA|hydroweb\.theia-land\.fr", "Lake/River altimetry", "IN_SCOPE_OPEN_L3_L4_EO", "Multi-mission satellite radar altimetry inland water level L3/L4 series"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "DAHITI (Database for Hydrological Time Series of Inland Waters)", "TUM DGFI", r"DAHITI", "Altimetry time series", "IN_SCOPE_OPEN_L3_L4_EO", "Satellite altimetry hydrological inland water time-series database"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "USDA Global Reservoir and Lake Monitor (G-REALM)", "USDA / NASA", r"G-\s*REALM|global_reservoir", "Altimetry time series", "IN_SCOPE_OPEN_L3_L4_EO", "Satellite radar altimetry global reservoir height L3 series"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "ESA WorldWater Surface Water Dynamics", "ESA", r"WorldWater", "10-20 m", "IN_SCOPE_OPEN_L3_L4_EO", "ESA EO4Society global surface water extent/volume project products"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "Global Dam Watch (GOODD2 / GRanD)", "Global Dam Watch", r"Global Dam Watch", "Point/Vector", "OUT_OF_SCOPE_IN_SITU_OR_TABULAR", "Global dam/reservoir point inventory (GRanD attribute DB + digitized points)"),
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "SDG 6.6.1 (Water-related ecosystems)", "JAXA ALOS / ALOS-2 PALSAR Global Mosaic", "JAXA EORC", r"ALOS-2 global mosaic|eorc\.jaxa\.jp/ALOS", "25 m", "IN_SCOPE_OPEN_L3_L4_EO", "JAXA global 25 m L-band SAR mosaic L3 dataset"),
        # SDG 11.3.1 (Urbanization, UN-Habitat, 2024)
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "Global Human Settlement Built-up (GHS-BUILT)", "EC JRC / CEMS", r"Global Human Settlement Built-up|GHS BUILT", "10-100 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open global multi-epoch built-up surface L3 grid"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "World Settlement Footprint (WSF 2015 / 2019 / Evolution)", "DLR / ESA", r"World Settlement Footprint", "10-30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open global S1/S2/Landsat human settlement extent L3 grid"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "High Resolution Settlement Layer (HRSL - Built-up & Population)", "Meta / CIESIN", r"High Resolution Settlement Layer", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open satellite-derived 30 m settlement/population grid"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "Human Built-up and Settlement Extent (HBASE)", "NASA SEDAC", r"HBASE", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open Landsat-derived 30 m global built-up L3 product"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "Google Open Buildings / Open Buildings 2.5D Temporal", "Google Research", r"Google Open Buildings", "High-res / 4 m", "IN_SCOPE_OPEN_L3_L4_EO", "Open satellite-derived building footprint & 2.5D temporal dataset"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "WorldPop Gridded Population & Settlement Datasets", "WorldPop (Univ. Southampton)", r"WorldPop", "100 m - 1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Open EO-constrained gridded demographic/settlement dataset"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "SDG 11.3.1 (Urban land consumption)", "Sub-Saharan Africa Settlement Extents (GRID3)", "GRID3 / CIESIN", r"Sub-Saharan Africa Settlement Extents", "Settlement polygons", "IN_SCOPE_OPEN_L3_L4_EO", "Open regional EO-derived settlement extent product"),
        # SDG 14.1.1a (Coastal Eutrophication, UNEP, 2025)
        ("SDG_14.1.1a_EO_Support_Sheet_v2025_1.pdf", "SDG 14.1.1a (Coastal eutrophication)", "SDG 14.1.1a Indicator Maps EEZ (UNEP / Esri)", "UNEP / Esri", r"SDG 14\.1\.1a Indicator Maps EEZ", "4 km / EEZ", "IN_SCOPE_OPEN_L3_L4_EO", "Derived coastal chlorophyll anomaly indicator maps built from OC-CCI/VIIRS"),
        ("SDG_14.1.1a_EO_Support_Sheet_v2025_1.pdf", "SDG 14.1.1a (Coastal eutrophication)", "ESA Ocean Colour CCI (OC-CCI) Chlorophyll-a", "ESA CCI", r"OC-CCI", "4 km", "IN_SCOPE_OPEN_L3_L4_EO", "Merged multi-sensor global ocean colour L3/L4 climate data record"),
        ("SDG_14.1.1a_EO_Support_Sheet_v2025_1.pdf", "SDG 14.1.1a (Coastal eutrophication)", "NOAA VIIRS Daily Global Chlorophyll-a", "NOAA NESDIS / CoastWatch", r"NOAA VIIRS", "2-4 km", "IN_SCOPE_OPEN_L3_L4_EO", "Operational NOAA VIIRS global L3 ocean colour chlorophyll product"),
        # SDG 15.3.1 (Land Degradation, UNCCD, 2025)
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "ESA CCI Land Cover (ESA-CCI-LC 300 m)", "ESA CCI / C3S", r"ESA-CCI-LC|ESA Climate Change Initiative Land Cover", "300 m", "IN_SCOPE_OPEN_L3_L4_EO", "Default global 300 m annual land cover L3/L4 time series"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "Copernicus Global Land Service LC100 (CGLS-LC100)", "Copernicus CLMS", r"CGLS-LC100", "100 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 100 m annual land cover L3 product"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "Pan-European Copernicus High Resolution Layers (HRL)", "Copernicus CLMS", r"Pan-European Copernicus High resolution Layers", "10-20 m", "IN_SCOPE_OPEN_L3_L4_EO", "Operational pan-European 10-20 m thematic land cover/imperviousness/forest L3 layers"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "JRC Land Productivity Dynamics (LPD)", "EC JRC", r"Land Productivity Dynamics \(LPD\)", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "SPOT-VGT / PROBA-V global land productivity dynamics L3/L4 dataset"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "Trends.Earth / MODIS MOD13Q1 Land Productivity", "NASA / Conservation Intl", r"MODIS|Trends\.Earth", "250 m", "IN_SCOPE_OPEN_L3_L4_EO", "250 m MODIS NDVI / Trends.Earth land productivity L3/L4 product"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "Copernicus Global Land Service NDVI", "Copernicus CLMS", r"Copernicus NDVI", "300 m - 1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Operational CLMS global NDVI L3 time series"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "ISRIC SoilGrids250m (v1 / v2)", "ISRIC", r"SoilGrids250m", "250 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 250 m EO-constrained soil organic carbon hybrid grid"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "Harmonized World Soil Database (HWSD v1.2)", "FAO / IIASA / ISRIC", r"Harmonized World Soil Database", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Global gridded harmonized soil property/SOC baseline database"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "SDG 15.3.1 (Land degradation)", "FAO Global Soil Organic Carbon Map (GSOC v1.0)", "FAO GSP", r"Global Soil Organic Carbon Map", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Global 1 km gridded soil organic carbon stock map"),
    ]
    text_cache: dict[str, str] = {}
    for fname, ind, pname, prov, rx, res, scope, rat in specs:
        if fname not in text_cache:
            text_cache[fname] = _pdf_text(fname)
        assert re.search(rx, text_cache[fname], re.I), f"Regex {rx!r} not found in {fname}!"
        rows.append(
            _row(
                benchmark_id="02_ceos_sdg_sheets",
                benchmark_name="CEOS SDG Official EO Support Sheets (v2023-2025)",
                policy_framework="UN 2030 Agenda (SDGs 6.6.1, 11.3.1, 14.1.1a, 15.3.1)",
                policy_target_or_indicator=ind,
                product_name=pname,
                provider_or_source=prov,
                scope_class=scope,
                scope_rationale=rat,
                source_file=f"raw/02_ceos_sdg_support_sheets/{fname}",
                source_locator="Verified in PDF 'EO-based global products' section",
                spatial_res_raw=res,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 3. CEOS Ecosystem Extent Task Team (EETT) White Paper v1.1 (CBD GBF A.2 / SEEA)
# ---------------------------------------------------------------------------
def extract_03_ceos_eett() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fp = RAW_DIR / "03_ceos_eett_gbf" / "CEOS_EETT_White_Paper_v1.1_2023.pdf"
    parts: list[str] = []
    with pdfplumber.open(fp) as doc:
        for p in (doc.pages[36], doc.pages[37]):
            parts.append(p.extract_text() or "")
            for tb in p.extract_tables() or []:
                for row in tb:
                    for cell in row:
                        if cell:
                            parts.append(str(cell))
    app2_text = re.sub(r"\s+", " ", " ".join(parts))

    specs = [
        ("Land Cover", "Copernicus Global Land Cover (CGLS-LC100, Buchhorn et al. 2020)", "Copernicus CLMS", r"Buchhorn|Copernicus\s+Global\s+Land", "100 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 100 m land cover L3 dataset cited in Appendix 2"),
        ("Land Cover", "ESA CCI Land Cover (300 m, 1992-2015+)", "ESA CCI", r"CCI\s+Land\s+Cover", "300 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 300 m annual land cover CDR cited in Appendix 2"),
        ("Vegetation Height", "GEDI Global Canopy / Vegetation Height (Dubayah et al. 2020)", "NASA / UMD", r"GEDI.*Dubayah|Dubayah", "25 m - 1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Spaceborne lidar canopy height L3/L4 product"),
        ("Vegetation Cover", "MODIS Vegetation Continuous Fields / Canopy Cover (250 m)", "NASA LP DAAC", r"Canopy and\s+vegetation cover.*MODIS|MODIS.*250m", "250 m", "IN_SCOPE_OPEN_L3_L4_EO", "Annual 250 m MODIS MOD44B vegetation continuous fields L3 product"),
        ("Terrain Elevation", "Copernicus DEM GLO-30", "Copernicus / ESA", r"Copernicus\s+GLO-30", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 30 m digital elevation model L3 product"),
        ("Terrain Elevation", "ASTER GDEM", "NASA / METI", r"ASTER", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 30 m optical stereo DEM L3 product"),
        ("Terrain Elevation", "SRTM DEM", "NASA / NGA", r"SRTM", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Quasi-global 30 m radar DEM L3 product"),
        ("Terrain Elevation", "TanDEM-X DEM", "DLR", r"TanDEM-X", "12-90 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global radar DEM L3 product (90 m open)"),
        ("Terrain Elevation", "ALOS World 3D (AW3D30)", "JAXA", r"ALOS World 3D", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 30 m optical DEM L3 product"),
        ("Leaf Area Index (LAI)", "MODIS LAI (MOD15A2H / Myneni et al. 2015)", "NASA LP DAAC", r"Myneni", "500 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global MODIS LAI L4/L3 product"),
        ("Leaf Area Index (LAI)", "Copernicus Global Land Service LAI", "Copernicus CLMS", r"LAI.*Copernicus|Copernicus.*LAI", "300 m - 1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Operational CLMS global LAI L3 product"),
        ("Phenology Dynamics", "MODIS Vegetation Indices / Land Surface Phenology (Huete & Justice)", "NASA LP DAAC", r"Huete", "250-500 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global MODIS VI / phenology L3 product"),
        ("Phenology Dynamics", "Copernicus Land Surface Phenology (HR-VPP / CGLS)", "Copernicus CLMS", r"Phenology.*Copernicus|Vegetation indices.*Copernicus", "10-300 m", "IN_SCOPE_OPEN_L3_L4_EO", "Operational CLMS vegetation phenology L3 product"),
        ("Snow Cover", "ESA CCI Snow Cover", "ESA CCI", r"CCI Snow", "500 m - 1 km", "IN_SCOPE_OPEN_L3_L4_EO", "ESA CCI global snow cover extent L3/L4 CDR"),
        ("Biomass", "ESA CCI Biomass (100 m Annual Series)", "ESA CCI", r"CCI Biomass", "100 m", "IN_SCOPE_OPEN_L3_L4_EO", "ESA CCI global 100 m aboveground biomass L4 series"),
        ("Biomass", "Spawn et al. 2020 Global Harmonized Above & Belowground Biomass (300 m)", "ORNL DAAC / Spawn et al.", r"Spawn et al", "300 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global harmonized carbon density L4 dataset"),
        ("Chlorophyll", "Global Leaf Chlorophyll Content (MODIS / GCOM-C SGLI, Xu et al. 2022)", "Academic / JAXA GCOM-C", r"Xu et al|GCOM-C/SGLI", "250 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global satellite leaf chlorophyll content L3/L4 product"),
        ("Leaf Traits", "Hyperspectral Leaf Traits (Dechant et al.)", "Airborne / Prototype Hyperspectral", r"Dechant", "3-30 m", "OUT_OF_SCOPE_PLANNED_OR_TOOL", "Airborne/prototype limited monitoring capacity hyperspectral trait research"),
        ("fAPAR", "Copernicus Global Land Service fAPAR", "Copernicus CLMS", r"fAPAR.*Copernicus|Copernicus", "300 m", "IN_SCOPE_OPEN_L3_L4_EO", "Operational CLMS global fAPAR L3 product"),
        ("fAPAR", "MODIS / GIMMS fAPAR (Zhu et al. 2013)", "NASA / Academic", r"Zhu et al", "500 m - 8 km", "IN_SCOPE_OPEN_L3_L4_EO", "Global satellite fAPAR L3/L4 product"),
        ("Productivity", "MODIS GPP / NPP (MOD17, Running & Zhao 2019)", "NASA LP DAAC", r"Running", "500 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 500 m 8-day/annual MODIS GPP/NPP L4 product"),
        ("Fluorescence (SIF)", "Solar-Induced Fluorescence (SIF, Joiner et al. 2023 / OCO-3 / GOSAT)", "NASA / Academic", r"Joiner et al", "> 1 km", "IN_SCOPE_OPEN_L3_L4_EO", "Global satellite SIF L3/L4 product"),
        ("Deforestation & Degradation", "Global Forest Watch (UMD Hansen Tree Cover Loss)", "UMD GLAD / WRI", r"Global.{0,40}Forest.{0,40}Watch", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "Global 30 m annual tree cover loss L3 product"),
        ("Wetland Extent & Change", "ALOS-2 / Global Wetland Inundation & Mangrove Maps (Rosenqvist 2020 / Chapman 2015)", "JAXA / NASA / GMW", r"Rosenqvist|Chapman", "10-50 m", "IN_SCOPE_OPEN_L3_L4_EO", "L-band SAR global mangrove/wetland inundation L3/L4 products"),
    ]
    for char, pname, prov, rx, res, scope, rat in specs:
        assert re.search(rx, app2_text, re.I | re.S), f"EETT regex {rx!r} missing in Appendix 2!"
        rows.append(
            _row(
                benchmark_id="03_ceos_eett_gbf",
                benchmark_name="CEOS Ecosystem Extent Task Team White Paper v1.1 (2023)",
                policy_framework="CBD Kunming-Montreal Global Biodiversity Framework (A.2) & UN SEEA EA",
                policy_target_or_indicator=f"GBF Indicator A.2 / Ecosystem Characteristic: {char}",
                product_name=pname,
                provider_or_source=prov,
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/03_ceos_eett_gbf/CEOS_EETT_White_Paper_v1.1_2023.pdf",
                source_locator="Appendix 2 table (pp. 37-38)",
                spatial_res_raw=res,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 4. UN-Habitat / GEO / NASA EO4SDG Sustainable Cities Toolkit (SDG 11)
# ---------------------------------------------------------------------------
def extract_04_eotoolkit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fp = RAW_DIR / "04_unhabitat_geo_nasa_eotoolkit" / "eotoolkit_datasets.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    for feat in data.get("features", []):
        a = feat.get("attributes", {})
        oid = a.get("objectid")
        name = str(a.get("dataset_name") or "").strip()
        if not name:
            continue
        desc = str(a.get("description") or "").strip()
        ind = str(a.get("indicators") or "").replace("_", " ").strip()
        src = str(a.get("data_source") or "").strip()
        res = str(a.get("resolution") or "").strip()
        url = str(a.get("download_from") or a.get("documentation") or "").strip()
        doi = _norm_doi(url) or _norm_doi(str(a.get("scientific_publication_with_qua") or "")) or _norm_doi(desc)
        # Classify scope cleanly
        low = name.lower()
        if "historical urban population" in low:
            scope = "OUT_OF_SCOPE_IN_SITU_OR_TABULAR"
            rat = "Historical census/city population point time-series (non-EO tabular/point DB, 3700 BC-2000 AD)"
        elif "popgrid-compare" in low:
            scope = "OUT_OF_SCOPE_PLANNED_OR_TOOL"
            rat = "Intercomparison tool/data-folder helper rather than a primary EO dataset product"
        elif "landcover 2050" in low:
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = "EO-based global land cover change scenario model grid (Esri/Clark Labs)"
        else:
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = "Open EO/gridded urban, settlement, land cover, heat, air quality, or EO-constrained demographic product curated for SDG 11"
        # Disambiguate the two GHSL Built-Up rows (Landsat vs Sentinel)
        display_name = name
        if name == "Global Human Settlement-Built-Up Presence" and src:
            display_name = f"{name} ({src})"
        rows.append(
            _row(
                benchmark_id="04_unhabitat_eotoolkit",
                benchmark_name="UN-Habitat / GEO / NASA EO4SDG Sustainable Cities Toolkit",
                policy_framework="UN 2030 Agenda (SDG 11) & New Urban Agenda",
                policy_target_or_indicator=ind or "SDG 11",
                product_name=display_name,
                provider_or_source=src or "UN-Habitat EO Toolkit",
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/04_unhabitat_geo_nasa_eotoolkit/eotoolkit_datasets.json",
                source_locator=f"ArcGIS FeatureServer objectid={oid}",
                doi=doi,
                url=url,
                raw_identifier=f"EOTOOLKIT:{oid}",
                spatial_res_raw=res,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 5. EC JRC Knowledge Centre on Earth Observation (KCEO) Deep Dives
# ---------------------------------------------------------------------------
def extract_05_kceo() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d = RAW_DIR / "05_ec_jrc_kceo_deep_dives"

    # 5A. EU Biodiversity Strategy 2030 Deep Dive (Annex 3 Excel - both sheets)
    f_bio = d / "KCEO-Datasets_KCEO_Biodiversity_DD_Indicators_Annex_3.xlsx"
    # Collect normalized distinct EO product references across both sheets with exact sheet/row provenance
    bio_entries: list[tuple[str, str, str, str, str]] = []
    for sh, ind_col, prod_col, url_col, svc_col in [
        ("Main KPIs", 3, 10, 16, 17),
        ("BDS indicators examined", 3, 8, 13, 14),
    ]:
        df = pd.read_excel(f_bio, sheet_name=sh)
        for idx, r in df.iterrows():
            p_raw = str(r.iloc[prod_col] if pd.notna(r.iloc[prod_col]) else "").strip()
            if not p_raw or p_raw in ("EXISTING EO PRODUCT CHARACTERISTICS", "PRODUCT NAME", "PRODUCT / EO-derived indicator or variable"):
                continue
            ind = str(r.iloc[ind_col] if pd.notna(r.iloc[ind_col]) else "").strip()
            u_raw = str(r.iloc[url_col] if pd.notna(r.iloc[url_col]) else "").strip()
            s_raw = str(r.iloc[svc_col] if pd.notna(r.iloc[svc_col]) else "").strip()
            bio_entries.append((f"{sh}:row{idx+2}", ind, p_raw, s_raw, u_raw))

    # Canonical extraction of every distinct named product cited in the KCEO Biodiversity Excel cells
    bio_text_all = "\n".join(f"{p} | {s} | {u}" for _, _, p, s, u in bio_entries)
    bio_catalog = [
        ("Natura 2000 (N2K) Land Cover / Land Use", "Copernicus CLMS", r"Natura 2000|\bN2K\b", "https://land.copernicus.eu/local/natura", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS local 10 m LC/LU product over Natura 2000 sites"),
        ("CLMS Global Hot Spot Monitoring (C-HSM)", "Copernicus CLMS", r"Hot\s*Spot\s+monitoring", "https://land.copernicus.eu/global/about-hot-spot", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS protected-area land cover change mapping"),
        ("CMEMS Global Ocean Physics Analysis and Forecast (GLOBAL_ANALYSIS_FORECAST_PHY_001_024)", "Copernicus CMEMS", r"GLOBAL_ANALYSIS_FORECAST_PHY_001_024", "https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/INFORMATION", "IN_SCOPE_OPEN_L3_L4_EO", "CMEMS L4 global 3D ocean physics analysis & forecast"),
        ("Corine Land Cover (CLC)", "Copernicus CLMS", r"Corine Land Cover", "https://land.copernicus.eu/pan-european/corine-land-cover", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS pan-European 100 m land cover L3 series"),
        ("CLC+ Backbone (CLC+ BB)", "Copernicus CLMS", r"CLC\+\s*BB", "https://land.copernicus.eu/pan-european/clc-plus", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS pan-European 10 m CLC+ Backbone L3 product"),
        ("ESA CCI High Resolution Land Cover (HRLC)", "ESA CCI", r"HR Land Cover|high-resolution-land-cover", "https://climate.esa.int/en/projects/high-resolution-land-cover/about/", "IN_SCOPE_OPEN_L3_L4_EO", "ESA CCI 10-30 m regional high-resolution land cover L3 product"),
        ("CLMS VHR European Image Mosaics", "Copernicus CLMS", r"VHR MOSAICS|very-high-resolution", "https://land.copernicus.eu/imagery-in-situ/european-image-mosaics/very-high-resolution", "OUT_OF_SCOPE_COMMERCIAL_OR_RESTRICTED", "Commercial VHR (Pléiades/PlanetScope/KOMPSAT) ortho-imagery mosaic (non-L3 thematic)"),
        ("EMODnet EUSeaMap Seabed Habitats", "EMODnet", r"EUSEAMAP|seabed-habitats", "https://emodnet.ec.europa.eu/en/seabed-habitats", "IN_SCOPE_OPEN_L3_L4_EO", "European broad-scale seabed habitat L3/L4 map"),
        ("CLMS High Resolution Layer (HRL) Forest (TCD / DLT / FTY)", "Copernicus CLMS", r"HRL Forest|Forest High Resolution Layer", "https://land.copernicus.eu/pan-european/high-resolution-layers/forests", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS pan-European 10 m forest status/change L3 layers"),
        ("CMEMS Ocean Monitoring Indicators (OMIs)", "Copernicus CMEMS", r"Ocean Monitoring Indicators|ocean-monitoring-indicators", "https://marine.copernicus.eu/access-data/ocean-monitoring-indicators", "IN_SCOPE_OPEN_L3_L4_EO", "CMEMS L4 ocean climate & biogeochemical monitoring indicators"),
        ("CMEMS Global Ocean Biogeochemistry Analysis and Forecast", "Copernicus CMEMS", r"biogeochemical global ocean analysis", "https://marine.copernicus.eu/access-data/myocean-viewer", "IN_SCOPE_OPEN_L3_L4_EO", "CMEMS L4 global ocean BGC analysis & forecast"),
        ("CLMS High-Resolution Vegetation Phenology and Productivity (HR-VPP)", "Copernicus CLMS", r"HR-VPP|HR VPP", "https://land.copernicus.eu/pan-european/biophysical-parameters/high-resolution-vegetation-phenology-and-productivity", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS 10 m Sentinel-2 vegetation phenology & productivity L3 suite"),
        ("CLMS High Resolution Snow & Ice (RLIE S1+S2)", "Copernicus CLMS", r"RLIE S1\+S2", "https://land.copernicus.eu/pan-european/biophysical-parameters/high-resolution-snow-and-ice-monitoring", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS 20 m river and lake ice extent L3 product"),
        ("CLMS High Resolution Layer (HRL) Imperviousness", "Copernicus CLMS", r"Imperviousness", "https://land.copernicus.eu/pan-european/high-resolution-layers/imperviousness", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS 10 m imperviousness density/change L3 product"),
        ("CLMS Global Lake Water Quality (LWQ)", "Copernicus CLMS", r"Lake Water Quality", "https://land.copernicus.eu/global/products/lwq", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS global lake water quality L3 product"),
        ("CLMS High Resolution Layer (HRL) Water and Wetness", "Copernicus CLMS", r"Water and Wetness|water-wetness", "https://land.copernicus.eu/pan-european/high-resolution-layers/water-wetness", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS 10 m water & wetness L3 layer"),
        ("CLMS Riparian Zones", "Copernicus CLMS", r"Riparian Zones", "https://land.copernicus.eu/local/riparian-zones", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS local 10 m riparian zones LC/LU L3 layer"),
        ("CLMS Small Woody Features (HRL SWF)", "Copernicus CLMS", r"Small Woody Features", "https://land.copernicus.eu/pan-european/high-resolution-layers/small-woody-features", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS pan-European 5 m small woody features L3 layer"),
        ("CLMS Urban Atlas", "Copernicus CLMS", r"Urban Atlas", "https://land.copernicus.eu/local/urban-atlas", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS European Functional Urban Areas LC/LU L3 product"),
        ("CLMS Global Water Bodies (WB)", "Copernicus CLMS", r"Water Bodies|global/products/wb", "https://land.copernicus.eu/global/products/wb", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS global 100-300 m water bodies L3 product"),
        ("CLMS Global Water Level (WL)", "Copernicus CLMS", r"Water Level", "https://land.copernicus.eu/global/products/wl", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS global lake/river water level altimetry L3 product"),
        ("CLMS FAPAR / FCOVER", "Copernicus CLMS", r"\bFAPAR\b|Fraction of vegetation cover", "https://land.copernicus.eu/global/products/fapar", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS biophysical vegetation L3 product"),
        ("CLMS Soil Water Index (SWI)", "Copernicus CLMS", r"Soil Water Index", "https://land.copernicus.eu/global/products/swi", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS root-zone soil moisture L3/L4 product"),
        ("CLMS Pan-European / S2GM Leaf Area Index (LAI)", "Copernicus CLMS", r"Leaf Area Index \(LAI\)", "https://land.copernicus.eu/pan-european/biophysical-parameters/high-resolution-vegetation-phenology-and-productivity", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS Sentinel-2 LAI biophysical L3 product"),
        ("CEMS EFFIS Burnt Area Product", "Copernicus CEMS", r"Burnt Area product \(EFFIS\)", "https://effis.jrc.ec.europa.eu/about-effis/technical-background/rapid-damage-assessment", "IN_SCOPE_OPEN_L3_L4_EO", "CEMS EFFIS operational burned area L3/L4 product"),
        ("CEMS Global Human Settlement Layer (GHSL)", "Copernicus CEMS / JRC", r"Global Human Settlement Layer|\bGHSL\b", "https://ghsl.jrc.ec.europa.eu/datasets.php", "IN_SCOPE_OPEN_L3_L4_EO", "CEMS global built-up & population L3/L4 suite"),
        ("JRC Global Surface Water Explorer (GSWE)", "EC JRC", r"\bGSWE\b|global-surface-water", "https://global-surface-water.appspot.com/", "IN_SCOPE_OPEN_L3_L4_EO", "EC JRC 30 m global surface water L3 time series"),
        ("JRC EU Crop Map (EUCROPMAP)", "EC JRC", r"EU Crop Map", "https://joint-research-centre.ec.europa.eu/jrc-news/eu-crop-map-2021-10-18_en", "IN_SCOPE_OPEN_L3_L4_EO", "EC JRC 10 m Sentinel-1/2 EU crop type L3 map"),
        ("Global Forest Watch Forest Carbon Flux Maps (Harris et al.)", "WRI / GFW", r"Forest Carbon Flux Maps", "https://www.globalforestwatch.org/blog/data-and-research/forest-carbon-flux-data/", "IN_SCOPE_OPEN_L3_L4_EO", "Global 30 m forest greenhouse gas emissions/removals L4 map"),
        ("C3S Hydrology-Related Climate Impact Indicators", "Copernicus C3S", r"Hydrology related climate impact", "https://cds.climate.copernicus.eu/", "IN_SCOPE_OPEN_L3_L4_EO", "C3S European hydrological climate impact L4 indicators"),
        ("Sentinel-2 Soil Organic Carbon (SOC) Research Maps", "ESA / Academic", r"SOC Maps", "https://sentinels.copernicus.eu/", "OUT_OF_SCOPE_PLANNED_OR_TOOL", "News/exploratory Sentinel-2 bare-soil SOC research demonstration rather than an archived catalogue product"),
        ("LUCAS In-Situ Land Cover / Land Use Survey", "Eurostat / JRC", r"LUCAS in situ", "https://land.copernicus.eu/imagery-in-situ/lucas", "OUT_OF_SCOPE_IN_SITU_OR_TABULAR", "Ground field survey points (in-situ reference dataset)"),
        ("WISE-SoE / WISE-6 Freshwater Quality & River BOD/Phosphate", "EEA / Eionet", r"WISE SOE|Biochemical oxygen demand", "https://cdr.eionet.europa.eu/help/WISE_SoE/wise6", "OUT_OF_SCOPE_IN_SITU_OR_TABULAR", "In-situ river/groundwater monitoring station reporting tables"),
        ("iMAP WebGIS Platform", "ESA / Academic", r"iMAP webGIS", "https://sentinels.copernicus.eu/", "OUT_OF_SCOPE_PLANNED_OR_TOOL", "Interactive webGIS platform rather than a standalone gridded EO dataset"),
    ]
    for pname, prov, rx, url, scope, rat in bio_catalog:
        matching_locs = [loc for loc, _, p, s, u in bio_entries if re.search(rx, f"{p} {s} {u}", re.I)]
        assert matching_locs, f"KCEO Biodiversity regex {rx!r} not found in Excel!"
        rows.append(
            _row(
                benchmark_id="05_ec_jrc_kceo",
                benchmark_name="EC JRC KCEO Deep Dives (EU Biodiversity & Urban Adaptation)",
                policy_framework="EU Biodiversity Strategy 2030 & Nature Restoration Law",
                policy_target_or_indicator=f"EU BDS 2030 ({len(matching_locs)} indicator rows)",
                product_name=pname,
                provider_or_source=prov,
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/05_ec_jrc_kceo_deep_dives/KCEO-Datasets_KCEO_Biodiversity_DD_Indicators_Annex_3.xlsx",
                source_locator="; ".join(matching_locs[:4]),
                url=url,
            )
        )

    # 5B. EU Urban Climate Adaptation Deep Dive (Indicators sheet)
    f_urb = d / "KCEO_Datasets_KCEO_Urban_Adaptation_Indicators.xlsx"
    df_u = pd.read_excel(f_urb, sheet_name="Indicators")
    urb_notes = "\n".join(df_u["Source Type"].dropna().astype(str).tolist())
    urb_catalog = [
        ("Copernicus C3S Urban Climate / ERA5 Products", "Copernicus C3S", r"\bC3S\b|ERA5", "IN_SCOPE_OPEN_L3_L4_EO", "C3S urban climate reanalysis/indicators referenced across urban adaptation indicators"),
        ("Copernicus CAMS Air Quality Products", "Copernicus CAMS", r"\bCAMS\b", "IN_SCOPE_OPEN_L3_L4_EO", "CAMS atmospheric composition/air quality L4 products referenced for urban health/air quality"),
        ("Copernicus CLMS Urban Atlas (Adaptation)", "Copernicus CLMS", r"Urban Atlas", "IN_SCOPE_OPEN_L3_L4_EO", "CLMS Urban Atlas referenced for urban green/blue & residential exposure"),
        ("CEMS GHSL Gridded Population & Built-Up (Adaptation)", "Copernicus CEMS / JRC", r"\bGHSL\b", "IN_SCOPE_OPEN_L3_L4_EO", "GHSL referenced for hazard exposure overlay in urban adaptation"),
        ("EEA Floods Viewer / European Flood Hazard Maps", "EEA / CEMS", r"discomap\.eea\.europa\.eu/floodsviewer|EEA.*flood", "IN_SCOPE_OPEN_L3_L4_EO", "European flood hazard spatial datasets"),
        ("JRC Risk Data Hub (Loss & Damage)", "EC JRC", r"Risk Data Hub", "OUT_OF_SCOPE_IN_SITU_OR_TABULAR", "Historical disaster economic/mortality loss registry (non-EO tabular/insurance records)"),
        ("OpenStreetMap Building Footprints & Transport", "OpenStreetMap", r"OpenStreetMap", "OUT_OF_SCOPE_IN_SITU_OR_TABULAR", "Crowdsourced vector basemap (non-EO)"),
    ]
    for pname, prov, rx, scope, rat in urb_catalog:
        m_rows = df_u[df_u["Source Type"].fillna("").astype(str).str.contains(rx, case=False, regex=True)]
        if len(m_rows) == 0:
            continue
        rows.append(
            _row(
                benchmark_id="05_ec_jrc_kceo",
                benchmark_name="EC JRC KCEO Deep Dives (EU Biodiversity & Urban Adaptation)",
                policy_framework="EU Climate Adaptation Strategy (Urban Adaptation)",
                policy_target_or_indicator=f"Urban Adaptation ({len(m_rows)} indicator rows)",
                product_name=pname,
                provider_or_source=prov,
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/05_ec_jrc_kceo_deep_dives/KCEO_Datasets_KCEO_Urban_Adaptation_Indicators.xlsx",
                source_locator=f"Indicators sheet ({len(m_rows)} rows)",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 6. White House USGEO / NASA-USGS Earth Observation Assessment (EOA) 2023
# ---------------------------------------------------------------------------
def extract_06_usgeo_eoa() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d = RAW_DIR / "06_usgeo_eoa_2023"
    af = pd.read_csv(d / "EOA2023_Tableau_Dashboard_Data_DataSources_AF.csv")
    cl = pd.read_csv(d / "EOA2023_Tableau_Dashboard_Data_DataSources_CL.csv")
    k_af = pd.read_csv(d / "EOA2023_Tableau_Dashboard_Data_KPSO_Counts_AF.csv").set_index("Data Source PALMA ID")["Count of KPSOs Impacted"].to_dict()
    k_cl = pd.read_csv(d / "EOA2023_Tableau_Dashboard_Data_KPSO_Counts_CL.csv").set_index("Data Source PALMA ID")["Count of KPSOs Impacted"].to_dict()

    both = pd.concat([af.assign(sba="Agriculture & Forestry"), cl.assign(sba="Climate")], ignore_index=True)
    uniq = both.groupby("Data Source PALMA ID", as_index=False).agg({
        "Data Source Name": "first",
        "Data Source Type": "first",
        "Data Source Description": "first",
        "sba": lambda s: "; ".join(sorted(set(s))),
    })
    uniq["kpso_total"] = uniq["Data Source PALMA ID"].map(lambda x: int(k_af.get(x, 0)) + int(k_cl.get(x, 0)))

    # Select all EO/satellite/gridded Database (DB) and Intermediate Product (IntPrd) sources
    eo_re = re.compile(
        r"MODIS|Landsat|Sentinel|VIIRS|ASTER|SRTM|GPM|TRMM|SMAP|SMOS|GRACE|GEDI|ICESat|OCO-|GOES|AVHRR|AMSR|MISR|AIRS|MOPITT|\bOMI\b|TROPOMI|MERRA|ERA5|ERA-Interim|CLMS|Copernicus|NLCD|NLDAS|GLDAS|PRISM|Daymet|MTBS|LCMS|C-CAP|Cropland Data Layer|\bNED\b|3DEP|LANDFIRE|NCEP|NARR|OISST|GPCP|CMAP|CHIRPS|ALEXI|CALIPSO|CloudSat|CERES|Jason|TOPEX|ALOS|PALSAR|CryoSat|SWOT|TEMPO|GCOM|Himawari|EUMETSAT|Meteosat|WorldPop|GHSL|SoilGrids|HWSD|Global Surface Water|Hansen|GLAD|GFED|FINN|ODIAC|CarbonTracker|FIRMS|LandScan|NALCMS|GLSDEM|CDEM|GIMP|RAMP|GMTED2010|GTOPO30|GLOBE|GLIMS|GLCC|OSISAF|GHRSST|SMOPS|GridSat|nClimGrid|SSMI|SSM/I|ScatSat|QuikSCAT|ASCAT|CYGNSS|SeaWiFS|SNPP|Suomi|NOAA.*CDR|NSIDC|MIrAD|LANID|LandID|GLEAM|GIMMS|MIRCA2000|Gridded.*CH4|Biomass|Land Cover|Sea Ice|Soil Moisture|Vegetation Health|Hazard Mapping System|CPC Global Gridded",
        re.I,
    )
    sub = uniq[uniq["Data Source Type"].isin(["DB", "IntPrd"])]
    eo_sub = sub[sub["Data Source Name"].fillna("").str.contains(eo_re)].sort_values("kpso_total", ascending=False)

    raw_sensor_re = re.compile(
        r"archive|MOD03|VNP03|Geolocation|TOPEX/Poseidon|^SeaWiFS$|Jason-1|TRMM.*satellite|AQUA.*AMSR-E|ScatSat-1|Quikscat|ICESat Altimetry|ICESAT.*GLAS.*Database|ALOS1\(PALSAR\)|ERS-1|GOES System Historical Database|Microwave Sensor Data|SAR Satellite Data",
        re.I,
    )
    insitu_re = re.compile(
        r"OpenStreets?Map|SeaBASS|GHCN|NCEP ADP|HURDAT|ULCD|Climate Data Online|U\.S\. Climate Normals|Drought Monitor.*table|USDM-DSCI|State/Local|State and Local|Field Campaigns|Archive Management System|Ocean Acidification: NCEI|NAO_NCEI|Global Time Series, NCEI|Wind Observations|Storm Frequency|High Tide Flooding|GoMMAPPS|Environmental Sensitivity Index|gNATSGO|Soil Survey Geographic|Fuel Rules|polygons \(Federal|Internal Disturbance Detection|Air Temperature Maps|^NCDC Precipitation$|^NCDC Temperature$|Probable Maximum Precipitation|Temp and Precip Maps|Hurricane Preliminary|Sea ice macro-|Kelly|Russian Arctic|Danish Meteorological Institute.*1893|National Biomass & Carbon Dataset|Land Cover Trends|USFS Historical|Ontario Land Cover|Mangrove forests - current and future|Billion-Dollar Disasters|NIDIS_SERCC|Pan-Arctic.*1850|Pan-Arctic.*1954|1850-1978|Sea ice maps for Alaska.*1953|Dehn Collection",
        re.I,
    )
    tool_or_nwp_re = re.compile(
        r"ArcMap|Toolset|Total Fuels Change Tool|National Digital Forecast Database|Flood Forecast|GEFS-Wave|Nearshore Wave|ADCIRC|SPARC Reanalysis Intercomparison|NREL Reanalysis|Gridded Wind Fields|Leaky Bucket",
        re.I,
    )

    for _, r in eo_sub.iterrows():
        pid = str(r["Data Source PALMA ID"])
        name = str(r["Data Source Name"]).strip()
        dst = str(r["Data Source Type"])
        kpso = int(r["kpso_total"])
        sba = str(r["sba"])

        if tool_or_nwp_re.search(name):
            scope = "OUT_OF_SCOPE_PLANNED_OR_TOOL"
            rat = f"EOA {dst}: software tool, operational NWP/wave forecast feed, or intercomparison project"
        elif raw_sensor_re.search(name) or raw_sensor_re.search(pid):
            scope = "OUT_OF_SCOPE_RAW_L1_L2_OR_SENSOR"
            rat = f"EOA {dst}: raw sensor/satellite archive or L1 geolocation database (not gridded L3/L4 product)"
        elif insitu_re.search(name) or insitu_re.search(pid):
            scope = "OUT_OF_SCOPE_IN_SITU_OR_TABULAR"
            rat = f"EOA {dst}: in-situ station/ship/field archive, vector/census DB, or historical ship-chart registry"
        else:
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = f"EOA {dst}: gridded/satellite L3/L4 EO dataset impacting {kpso} federal KPSOs"

        rows.append(
            _row(
                benchmark_id="06_usgeo_eoa_2023",
                benchmark_name="White House USGEO / NASA-USGS Earth Observation Assessment 2023",
                policy_framework="U.S. Federal Agency Statutory & Operational Mandates (OSTP/USGEO)",
                policy_target_or_indicator=f"{sba} (KPSOs impacted: {kpso})",
                product_name=name,
                provider_or_source=f"USGEO EOA ({dst})",
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/06_usgeo_eoa_2023/EOA2023_Tableau_Dashboard_Data_DataSources_{AF,CL}.csv",
                source_locator=f"PALMA_ID={pid}",
                raw_identifier=pid,
                kpso_impact_count=kpso,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 7. NASA Satellite Needs Working Group (SNWG / NSITE)
# ---------------------------------------------------------------------------
def extract_07_nasa_snwg() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d = RAW_DIR / "07_nasa_snwg_nsite"

    # 7A. Live NASA CMR SNWG collections (excluding JAXA ADEOS-II/GLI 'SNWG' snow-grain homograph hits)
    cmr = json.loads((d / "cmr_snwg_collections.json").read_text(encoding="utf-8"))
    for c in cmr:
        cid = str(c.get("id") or "")
        sn = str(c.get("short_name") or "")
        title = str(c.get("title") or "").strip()
        if "ADEOS-II/GLI" in title or "Snow grain size" in title:
            continue
        lv = str(c.get("processing_level_id") or "").strip()
        daac = str(c.get("data_center") or "")
        if lv in ("3", "4", "3A", "3B", "4A", "4B"):
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = f"NASA SNWG delivered CMR Level-{lv} EO collection (note: distributed via NASA CMR)"
        else:
            scope = "OUT_OF_SCOPE_RAW_L1_L2_OR_SENSOR"
            rat = f"NASA SNWG CMR Level-{lv or 'NA'} swath/scene/quick-look product (outside L3/L4 scope)"
        rows.append(
            _row(
                benchmark_id="07_nasa_snwg_cmr",
                benchmark_name="NASA SNWG Delivered Collections (CMR)",
                policy_framework="U.S. Federal Agency Biennial Satellite Needs Survey (SNWG)",
                policy_target_or_indicator=f"NASA SNWG CMR (Level {lv or 'NA'})",
                product_name=title,
                provider_or_source=f"NASA {daac}",
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/07_nasa_snwg_nsite/cmr_snwg_collections.json",
                source_locator=f"CMR concept_id={cid}",
                raw_identifier=f"{cid}|{sn}",
            )
        )

    # 7B. NASA NSITE Solutions Portfolio page headings
    html = (d / "nsite_solutions.html").read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    skip_h = {
        "SNWG Solutions Provided by NSITE",
        "Commercial Satellite Data Acquisition (CSDA) Activities Informed by SNWG",
        "Find Data",
        "By Platform",
        "By Topic",
        "Data Catalog",
        "Data Tools",
        "Footer",
        "Footer Submenu",
    }
    in_csda = False
    for h in soup.find_all(["h2", "h3"]):
        txt = h.get_text(" ", strip=True)
        if not txt:
            continue
        if "Commercial Satellite Data Acquisition" in txt:
            in_csda = True
            continue
        if txt in skip_h:
            continue
        low = txt.lower()
        if in_csda or any(k in low for k in ("planet", "spire", "desis", "ghgsat", "commercial")):
            scope = "OUT_OF_SCOPE_COMMERCIAL_OR_RESTRICTED"
            rat = "NASA CSDA commercial SmallSat/VHR acquisition activity (licensed commercial data)"
        elif any(k in low for k in ("software", "training", "curation", "admg-casei", "animal tracking", "gaban")):
            scope = "OUT_OF_SCOPE_PLANNED_OR_TOOL"
            rat = "Software tool, metadata curation service, airborne campaign inventory, or training activity (not gridded EO dataset)"
        elif any(k in low for k in ("nisar", "geos-cr", "merged pbl", "harmonized land surface temperature", "vlm product")):
            scope = "OUT_OF_SCOPE_PLANNED_OR_TOOL"
            rat = "SNWG in-development / pre-operational or newly commissioned solution suite"
        else:
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = "Operational NASA SNWG/NSITE L3/L4 EO product suite fulfilling U.S. federal agency needs"
        rows.append(
            _row(
                benchmark_id="07b_nasa_nsite_portfolio",
                benchmark_name="NASA NSITE / SNWG Solutions Portfolio",
                policy_framework="U.S. Federal Agency Biennial Satellite Needs Survey (SNWG)",
                policy_target_or_indicator="NSITE Federal Solution Portfolio",
                product_name=txt,
                provider_or_source="NASA NSITE / SNWG",
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/07_nasa_snwg_nsite/nsite_solutions.html",
                source_locator="NSITE Solutions heading",
                url="https://www.earthdata.nasa.gov/data/projects/nsite/solutions",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 8. US Greenhouse Gas Center (NASA / EPA / NIST / NOAA) STAC Catalog
# ---------------------------------------------------------------------------
def extract_08_us_ghg_center() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fp = RAW_DIR / "08_us_ghg_center_stac" / "us_ghg_center_collections.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    for col in data.get("collections", []):
        cid = str(col.get("id") or "")
        title = str(col.get("title") or cid).strip()
        desc = str(col.get("description") or "")
        provs = "; ".join(p.get("name", "") for p in (col.get("providers") or []) if isinstance(p, dict) and p.get("name"))
        doi = _norm_doi(desc)
        # Classify scope transparently (note: VIIRS Black Marble VNP46A1_NRT is gridded L3)
        if cid in ("emit-ch4plume-v1", "emit-ch4plume-v2", "goes-ch4plume-v1"):
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = "Derived satellite methane point-source plume complex product (EO event/plume L3/L4)"
        else:
            scope = "IN_SCOPE_OPEN_L3_L4_EO"
            rat = "Open gridded L3/L4 GHG flux, concentration, biomass, nighttime-lights, or emissions dataset in US GHG Center"
        rows.append(
            _row(
                benchmark_id="08_us_ghg_center",
                benchmark_name="U.S. Greenhouse Gas Center (NASA/EPA/NIST/NOAA)",
                policy_framework="UNFCCC Paris Agreement & U.S. National GHG Monitoring Strategy",
                policy_target_or_indicator="US GHG Center STAC Collection",
                product_name=title,
                provider_or_source=provs or "US GHG Center",
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/08_us_ghg_center_stac/us_ghg_center_collections.json",
                source_locator=f"STAC collection id={cid}",
                doi=doi,
                url=f"https://earth.gov/ghgcenter/api/stac/collections/{cid}",
                raw_identifier=cid,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 9. UNCCD Good Practice Guidance Version 2 for SDG 15.3.1 (2021)
# ---------------------------------------------------------------------------
def extract_09_unccd_gpg() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fp = RAW_DIR / "09_unccd_gpg_sdg1531" / "UNCCD_GPG_SDG_15_3_1_v2_2021.pdf"
    parts_u: list[str] = []
    with pdfplumber.open(fp) as doc:
        for p in (doc.pages[33], doc.pages[46]):
            parts_u.append(p.extract_text() or "")
            for tb in p.extract_tables() or []:
                for row in tb:
                    for cell in row:
                        if cell:
                            parts_u.append(str(cell))
    txt = re.sub(r"\s+", " ", " ".join(parts_u))
    specs = [
        ("Land Cover (Default)", "ESA CCI-LC (300 m Annual Global, 1992-2019)", "ESA CCI", r"ESA.CCI.LC", "300 m", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 default global land cover dataset"),
        ("Land Cover (Alternative)", "Copernicus Global Land Service LC100 (Collection 3, 100 m)", "Copernicus CLMS", r"CGLS.LC100", "100 m", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 alternative global land cover dataset"),
        ("Land Cover (Crosswalk)", "FAO Global Land Cover SHARE (GLC-Share)", "FAO", r"GLC-Share", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 3-3 reference global land cover product"),
        ("Land Productivity (Default)", "JRC Land Productivity Dynamics (LPD, 1 km Annual)", "EC JRC", r"Land Productivity Dynamics", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 default global LPD dataset"),
        ("Land Productivity (Alternative)", "MODIS Vegetation Indices (MOD13Q1 / MYD13Q1, 250 m 16-day)", "NASA LP DAAC", r"MOD13Q1|MYD13Q1", "250 m", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 alternative MODIS NDVI L3 dataset"),
        ("Land Productivity (Alternative)", "Copernicus Global Land Service NDVI (1 km)", "Copernicus CLMS", r"Copernicus Global Land Service NDVI", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 alternative CLMS NDVI L3 dataset"),
        ("Land Productivity (ARD Alternative)", "UMD GLAD Landsat Analysis Ready Data (ARD)", "UMD GLAD", r"glad\.geog\.umd\.edu/ard", "30 m", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 p.34 alternative 16-day global Landsat ARD composite"),
        ("Soil Organic Carbon (Default)", "ISRIC SoilGrids250m (v1 & v2, 250 m)", "ISRIC", r"SoilGrids250m", "250 m", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 default global SOC hybrid grid"),
        ("Soil Organic Carbon (Alternative)", "FAO Global Soil Organic Carbon Map (GSOC, 1 km)", "FAO GSP", r"Global Soil Organic Carbon Map", "1 km", "IN_SCOPE_OPEN_L3_L4_EO", "UNCCD GPG v2 Table 2-1 alternative global SOC grid"),
        ("Soil Organic Carbon (Emerging)", "Soils Revealed Historic & Projected SOC Stocks (250 m)", "TNC / ISRIC / Woodwell", r"Soils Revealed", "250 m", "OUT_OF_SCOPE_PLANNED_OR_TOOL", "UNCCD GPG v2 p.34 explicitly notes 'not clear whether there are plans to make these datasets downloadable'"),
    ]
    for subind, pname, prov, rx, res, scope, rat in specs:
        assert re.search(rx, txt, re.I), f"UNCCD regex {rx!r} not found!"
        rows.append(
            _row(
                benchmark_id="09_unccd_gpg_v2",
                benchmark_name="UNCCD Good Practice Guidance v2 (SDG 15.3.1, 2021)",
                policy_framework="UN Convention to Combat Desertification (UNCCD / LDN)",
                policy_target_or_indicator=f"SDG 15.3.1 — {subind}",
                product_name=pname,
                provider_or_source=prov,
                scope_class=scope,
                scope_rationale=rat,
                source_file="raw/09_unccd_gpg_sdg1531/UNCCD_GPG_SDG_15_3_1_v2_2021.pdf",
                source_locator="Table 2-1 (p. 34) & Table 3-3 (p. 47)",
                spatial_res_raw=res,
            )
        )
    return rows


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for fn in [
        extract_01_ceos_gst,
        extract_02_ceos_sdg_sheets,
        extract_03_ceos_eett,
        extract_04_eotoolkit,
        extract_05_kceo,
        extract_06_usgeo_eoa,
        extract_07_nasa_snwg,
        extract_08_us_ghg_center,
        extract_09_unccd_gpg,
    ]:
        part = fn()
        print(f"  {fn.__name__:30s} -> {len(part):4d} rows")
        all_rows.extend(part)

    df = pd.DataFrame(all_rows)
    df.insert(0, "item_id", [f"EXT_{i:04d}" for i in range(1, len(df) + 1)])

    csv_path = RES_DIR / "extracted_policy_products.csv"
    jsonl_path = RES_DIR / "extracted_policy_products.jsonl"
    df.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in df.to_dict(orient="records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Extracted {len(df)} total policy-product entries -> {csv_path}")
    print("\nBreakdown by benchmark & scope_class:")
    ct = pd.crosstab(df["benchmark_id"], df["scope_class"], margins=True)
    print(ct.to_string())


if __name__ == "__main__":
    main()
