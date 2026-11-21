#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""
05_external_policy_validation/01_download_benchmarks.py
=======================================================
Reproducibly downloads primary public tables, spreadsheets, JSON/STAC responses,
HTML dataset pages, and official PDF guidance documents from 9 independent
international & national EO-to-policy/treaty product-mapping efforts:

  01_ceos_gst_portal            - CEOS-UNFCCC Global Stocktake (GST) Portal (Paris Agreement)
  02_ceos_sdg_support_sheets    - CEOS SDG Coordination Group Official EO Support Sheets (v2023-2025)
  03_ceos_eett_gbf              - CEOS Ecosystem Extent Task Team White Paper v1.1 (CBD GBF A.2 / SEEA)
  04_unhabitat_geo_nasa_eotoolkit - UN-Habitat / GEO / NASA EO4SDG Sustainable Cities Toolkit (SDG 11)
  05_ec_jrc_kceo_deep_dives     - EC JRC Knowledge Centre on EO (EU Biodiversity & Urban Adaptation XLSX)
  06_usgeo_eoa_2023             - White House USGEO / NASA-USGS Earth Observation Assessment 2023 CSVs
  07_nasa_snwg_nsite            - NASA Satellite Needs Working Group (SNWG / NSITE) Federal Solutions
  08_us_ghg_center_stac         - US Greenhouse Gas Center (NASA/EPA/NIST/NOAA) STAC Catalog
  09_unccd_gpg_sdg1531          - UNCCD Good Practice Guidance v2 for SDG Indicator 15.3.1 (PDF)

Run via:
    uv run 05_external_policy_validation/01_download_benchmarks.py
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
UA = "Global-EO-L3-L4-Catalogue/1.0 (https://github.com/do-me/global-EO-products-L3-L4; Independent Policy Validation)"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def _download_file(sess: requests.Session, url: str, dest: Path, timeout: int = 60) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = sess.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return {
        "url": url,
        "final_url": r.url,
        "status": r.status_code,
        "bytes": len(r.content),
        "path": str(dest.relative_to(BASE_DIR)),
    }


def _download_json(sess: requests.Session, url: str, dest: Path, timeout: int = 60) -> tuple[Any, dict[str, Any]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = sess.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    data = r.json()
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data, {
        "url": url,
        "final_url": r.url,
        "status": r.status_code,
        "bytes": dest.stat().st_size,
        "path": str(dest.relative_to(BASE_DIR)),
    }


def main() -> None:
    sess = _session()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    # 1. CEOS Global Stocktake (GST) Portal - all English product pages
    print("[1/9] Downloading CEOS-UNFCCC Global Stocktake (GST) Portal pages...")
    gst_pages = [
        "index.html",
        "ghg.html",
        "carbon-dioxide.html",
        "methane.html",
        "biomass.html",
        "GEDI-biomass.html",
        "cci-biomass.html",
        "icesat2-boreal-biomass.html",
        "jpl-biomass.html",
        "africa-biomass.html",
        "forest.html",
        "land.html",
        "HILDAplus.html",
        "cci-lc.html",
        "copernicus.html",
        "worldcover.html",
        "agriculture.html",
        "mangrove.html",
        "fire.html",
    ]
    for page in gst_pages:
        url = f"https://ceos.org/gst/{page}"
        manifest.append(
            _download_file(sess, url, RAW_DIR / "01_ceos_gst_portal" / page)
        )

    # 2. CEOS SDG Coordination Group Official EO Support Sheets (v2023-2025)
    print("[2/9] Downloading CEOS SDG EO Support Sheets (6.6.1, 11.3.1, 14.1.1a, 15.3.1)...")
    sdg_files = [
        ("SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf", "https://ceos.org/document_management/Ad_Hoc_Teams/SDG-AHT/Support%20Sheets/SDG_6.6.1_EO_Support_Sheet_v2023_1.pdf"),
        ("SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf", "https://ceos.org/document_management/Ad_Hoc_Teams/SDG-AHT/Support%20Sheets/SDG_11.3.1_EO_Support_Sheet_v2024_1.pdf"),
        ("SDG_14.1.1a_EO_Support_Sheet_v2025_1.pdf", "https://ceos.org/document_management/Ad_Hoc_Teams/SDG-AHT/Support%20Sheets/SDG_14.1.1a_EO_Support_Sheet_v2025_1.pdf"),
        ("SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf", "https://ceos.org/document_management/Ad_Hoc_Teams/SDG-AHT/Support%20Sheets/SDG_15.3.1_EO_Support_Sheet_v2025_1.pdf"),
        ("CEOS_SDG_Support_v2.pdf", "https://ceos.org/document_management/Ad_Hoc_Teams/SDG-AHT/CEOS_SDG_Support_v2.pdf"),
    ]
    for fname, url in sdg_files:
        manifest.append(
            _download_file(sess, url, RAW_DIR / "02_ceos_sdg_support_sheets" / fname)
        )

    # 3. CEOS Ecosystem Extent Task Team (EETT) White Paper v1.1 (CBD GBF Indicator A.2 & UN SEEA)
    print("[3/9] Downloading CEOS EETT White Paper v1.1 (CBD GBF A.2 / SEEA)...")
    manifest.append(
        _download_file(
            sess,
            "https://ceos.org/document_management/Publications/Publications-and-Key-Documents/Cross-Cutting/EETT%20White%20Paper%20V1.1%202023-12-04.pdf",
            RAW_DIR / "03_ceos_eett_gbf" / "CEOS_EETT_White_Paper_v1.1_2023.pdf",
        )
    )

    # 4. UN-Habitat / GEO / NASA EO4SDG Earth Observations Toolkit for Sustainable Cities (SDG 11)
    print("[4/9] Downloading UN-Habitat / GEO / NASA EO Toolkit for Sustainable Cities (ArcGIS REST)...")
    _, m_ds = _download_json(
        sess,
        "https://services6.arcgis.com/uWtJiVzcBsV6C7NV/arcgis/rest/services/survey123_fa868d0c12dc49ec9b25d42ea0d0eea4/FeatureServer/0/query?where=1%3D1&outFields=*&f=json&resultRecordCount=1000",
        RAW_DIR / "04_unhabitat_geo_nasa_eotoolkit" / "eotoolkit_datasets.json",
    )
    manifest.append(m_ds)
    _, m_tools = _download_json(
        sess,
        "https://services6.arcgis.com/uWtJiVzcBsV6C7NV/arcgis/rest/services/survey123_b92f87d8532c40b1a82c140b64650c80_results/FeatureServer/0/query?where=1%3D1&outFields=*&f=json&resultRecordCount=1000",
        RAW_DIR / "04_unhabitat_geo_nasa_eotoolkit" / "eotoolkit_tools.json",
    )
    manifest.append(m_tools)

    # 5. EC JRC Knowledge Centre on Earth Observation (KCEO) Deep Dive Official Excel Workbooks
    print("[5/9] Downloading EC JRC KCEO Deep Dive Excel workbooks (Biodiversity & Urban Adaptation)...")
    for rec_id, label in [
        ("b9hh0-r2t58", "kceo_biodiversity"),
        ("axggr-ft816", "kceo_urban_adaptation"),
    ]:
        rec_meta, m_meta = _download_json(
            sess,
            f"https://research-data.ec.europa.eu/api/records/{rec_id}",
            RAW_DIR / "05_ec_jrc_kceo_deep_dives" / f"{label}_record.json",
        )
        manifest.append(m_meta)
        for fname, finfo in rec_meta.get("files", {}).get("entries", {}).items():
            dl_url = finfo.get("links", {}).get("content") or f"https://research-data.ec.europa.eu/api/records/{rec_id}/files/{fname}/content"
            manifest.append(
                _download_file(sess, dl_url, RAW_DIR / "05_ec_jrc_kceo_deep_dives" / fname)
            )

    # 6. White House USGEO / NASA-USGS Earth Observation Assessment (EOA) 2023 Tableau Dashboard CSVs
    print("[6/9] Downloading USGEO / NASA-USGS EOA 2023 ScienceBase CSV tables...")
    sb_meta, m_sb = _download_json(
        sess,
        "https://www.sciencebase.gov/catalog/item/661020a2d34e633466504b6a?format=json",
        RAW_DIR / "06_usgeo_eoa_2023" / "sciencebase_item.json",
    )
    manifest.append(m_sb)
    target_eoa_files = {
        "EOA_2023_Agriculture_&_Forestry_Data_Dictionary.csv",
        "EOA_2023_Climate_Data_Dictionary.csv",
        "EOA2023_Tableau_Dashboard_Data_DataSources_AF.csv",
        "EOA2023_Tableau_Dashboard_Data_DataSources_CL.csv",
        "EOA2023_Tableau_Dashboard_Data_KPSO_Counts_AF.csv",
        "EOA2023_Tableau_Dashboard_Data_KPSO_Counts_CL.csv",
        "EOA2023_Tableau_Dashboard_Data_VT_AF.csv",
        "EOA2023_Tableau_Dashboard_Data_VT_CL.csv",
    }
    for finfo in sb_meta.get("files", []):
        fname = finfo.get("name", "")
        if fname in target_eoa_files and finfo.get("url"):
            manifest.append(
                _download_file(sess, finfo["url"], RAW_DIR / "06_usgeo_eoa_2023" / fname)
            )

    # 7. NASA SNWG / NSITE Solutions Portfolio + NASA CMR SNWG tagged/keyword collections
    print("[7/9] Downloading NASA SNWG / NSITE Solutions page & CMR SNWG metadata...")
    manifest.append(
        _download_file(
            sess,
            "https://www.earthdata.nasa.gov/data/projects/nsite/solutions",
            RAW_DIR / "07_nasa_snwg_nsite" / "nsite_solutions.html",
        )
    )
    # Paginate CMR collections mentioning SNWG
    cmr_entries: list[dict[str, Any]] = []
    for page_num in range(1, 10):
        r = sess.get(
            "https://cmr.earthdata.nasa.gov/search/collections.json",
            params={"keyword": "SNWG", "page_size": 200, "page_num": page_num},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json().get("feed", {}).get("entry", [])
        if not batch:
            break
        cmr_entries.extend(batch)
        if len(batch) < 200:
            break
    cmr_path = RAW_DIR / "07_nasa_snwg_nsite" / "cmr_snwg_collections.json"
    cmr_path.parent.mkdir(parents=True, exist_ok=True)
    cmr_path.write_text(json.dumps(cmr_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.append({
        "url": "https://cmr.earthdata.nasa.gov/search/collections.json?keyword=SNWG",
        "final_url": "https://cmr.earthdata.nasa.gov/search/collections.json?keyword=SNWG",
        "status": 200,
        "bytes": cmr_path.stat().st_size,
        "records": len(cmr_entries),
        "path": str(cmr_path.relative_to(BASE_DIR)),
    })

    # 8. US Greenhouse Gas Center (NASA / EPA / NIST / NOAA) STAC Catalog
    print("[8/9] Downloading US Greenhouse Gas Center STAC collections...")
    _, m_ghg = _download_json(
        sess,
        "https://earth.gov/ghgcenter/api/stac/collections?limit=200",
        RAW_DIR / "08_us_ghg_center_stac" / "us_ghg_center_collections.json",
    )
    manifest.append(m_ghg)

    # 9. UNCCD Good Practice Guidance Version 2 for SDG Indicator 15.3.1
    print("[9/9] Downloading UNCCD Good Practice Guidance v2 for SDG 15.3.1 (PDF)...")
    manifest.append(
        _download_file(
            sess,
            "https://www.unccd.int/sites/default/files/documents/2021-09/UNCCD_GPG_SDG-Indicator-15.3.1_version2_2021.pdf",
            RAW_DIR / "09_unccd_gpg_sdg1531" / "UNCCD_GPG_SDG_15_3_1_v2_2021.pdf",
            timeout=120,
        )
    )

    summary = {
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_files": len(manifest),
        "total_bytes": sum(int(m["bytes"]) for m in manifest),
        "files": manifest,
    }
    (RAW_DIR / "download_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[DONE] Downloaded {len(manifest)} files ({summary['total_bytes']/1e6:.2f} MB) into {RAW_DIR}")


if __name__ == "__main__":
    main()
