#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: European Commission Joint Research Centre (JRC) Data Catalogue (DCAT-AP API & CKAN)
Source ID: 07_ec_jrc_data_catalogue
Authoritative Upstream API: GET https://data.europa.eu/api/hub/search/search?filter=dataset&facets={"catalog":["jrc"]} + data.jrc.ec.europa.eu/api/3/action/package_list

Run standalone via:
    uv run mine.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAST_ACCESS_FILE = BASE_DIR / "last_access.txt"

SOURCE_ID = "07_ec_jrc_data_catalogue"
SOURCE_TITLE = "European Commission Joint Research Centre (JRC) Data Catalogue (DCAT-AP API & CKAN)"


def build_session(max_retries: int = 4, pool_maxsize: int = 20) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Global-EO-L3-L4-Catalogue/1.0 (https://github.com/do-me/global-EO-products-L3-L4; Reproducible Metadata Miner)",
        "Accept": "application/json, */*",
    })
    return session


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_last_access(
    endpoints: list[str],
    total_records: int,
    status_summary: str = "200 OK",
    notes: str = "",
) -> None:
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files_info = []
    for p in sorted(DATA_DIR.glob("*")):
        if p.is_file():
            files_info.append({
                "file": f"data/{p.name}",
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    lines = [
        f"source_id: {SOURCE_ID}",
        f"source_title: {SOURCE_TITLE}",
        f"retrieved_at_utc: {now_utc}",
        f"http_status_summary: {status_summary}",
        f"total_records_retrieved: {total_records}",
        "endpoints_queried:",
    ]
    for ep in endpoints:
        lines.append(f"  - {ep}")
    if notes:
        lines.append(f"notes: {notes}")
    lines.append("files_written:")
    for fi in files_info:
        lines.append(f"  - path: {fi['file']}")
        lines.append(f"    bytes: {fi['bytes']}")
        lines.append(f"    sha256: {fi['sha256']}")
    LAST_ACCESS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{SOURCE_ID}] Wrote {LAST_ACCESS_FILE.name} ({total_records} records across {len(files_info)} files)")

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    queried: list[str] = []

    # 1. Fetch JRC CKAN package_list
    ckan_list_url = "https://data.jrc.ec.europa.eu/api/3/action/package_list"
    r_ckan = session.get(ckan_list_url, timeout=60)
    if r_ckan.status_code == 200:
        (DATA_DIR / "jrc_ckan_package_list.json").write_text(
            json.dumps(r_ckan.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        queried.append(ckan_list_url)

    # 2. Paginate deterministically through all JRC datasets on data.europa.eu DCAT-AP Search API
    base_url = "https://data.europa.eu/api/hub/search/search"
    page = 0
    limit = 200
    all_results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    while True:
        params = {
            "filter": "dataset",
            "facets": json.dumps({"catalog": ["jrc"]}),
            "sort": "id+asc",
            "limit": str(limit),
            "page": str(page),
        }
        r = session.get(base_url, params=params, timeout=60)
        r.raise_for_status()
        res = r.json().get("result", {})
        items = res.get("results", [])
        if not items:
            break
        for item in items:
            iid = str(item.get("id") or "")
            if iid and iid in seen_ids:
                continue
            if iid:
                seen_ids.add(iid)
            all_results.append(item)
        if page == 0:
            queried.append(f"{base_url}?filter=dataset&facets={{'catalog':['jrc']}}&sort=id+asc&limit={limit}&page=0..N")
        page += 1
        if len(items) < limit or page > 500:
            break

    # Save in chunked JSONL files (~500 records per chunk) to stay well under 25 MB per file
    for old in DATA_DIR.glob("jrc_dcat_ap_datasets_part*.jsonl"):
        old.unlink()
    chunk_size = 500
    for idx in range(0, len(all_results), chunk_size):
        chunk = all_results[idx : idx + chunk_size]
        part = (idx // chunk_size) + 1
        out_path = DATA_DIR / f"jrc_dcat_ap_datasets_part{part:02d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 3. Mine EC JRC FOROBS Tropical Moist Forest (TMF) official portal & GEE asset registry
    tmf_url = "https://forobs.jrc.ec.europa.eu/TMF/data"
    tmf_records: list[dict[str, Any]] = []
    r_tmf = session.get(tmf_url, timeout=60)
    if r_tmf.status_code == 200:
        queried.append(tmf_url)
        html_tmf = r_tmf.text
        assets = sorted(set(re.findall(r"projects/JRC/TMF/[A-Za-z0-9_./-]+", html_tmf)))
        tmf_records.append({
            "id": "JRC-FOROBS-TMF-v2025",
            "title": "JRC Tropical Moist Forest (TMF) v2025 - Transition Map and Annual Change Collection (1990-2025)",
            "description": (
                "Pan-tropical annual and cumulative monitoring of tropical moist forest cover extent, "
                "deforestation, degradation, and regrowth at 30 m spatial resolution (Landsat 1990-2025) "
                f"produced by EC JRC FOROBS. Official GEE ImageCollections/Assets listed: {'; '.join(assets)}"
            ),
            "spatial_resolution_raw": "30 m",
            "temporal_start_raw": "1990",
            "temporal_end_raw": "2025",
            "update_frequency_raw": "annual",
            "landing_url_raw": tmf_url,
            "platform_sensor_raw": "Landsat TM/ETM+/OLI",
            "keywords_raw": "tropical moist forest; TMF; deforestation; forest degradation; FOROBS; JRC",
            "gee_assets": assets,
        })
        # Also record each distinct latest v1_2025 / v1_2022 layer product explicitly published on the FOROBS page
        for asset in assets:
            if "/v1_2025/" in asset or "/v1_2022/TransitionMapHybrid" in asset:
                layer_name = asset.split("/")[-1]
                tmf_records.append({
                    "id": asset,
                    "title": f"JRC Tropical Moist Forest (TMF) - {layer_name} ({asset.split('/')[-2]})",
                    "description": f"EC JRC FOROBS Tropical Moist Forest layer published at {tmf_url} (GEE Asset ID: {asset}).",
                    "spatial_resolution_raw": "10 m" if "Hybrid" in asset else "30 m",
                    "temporal_start_raw": "1990",
                    "temporal_end_raw": "2025",
                    "update_frequency_raw": "annual",
                    "landing_url_raw": tmf_url,
                    "platform_sensor_raw": "Landsat / Sentinel-2" if "Hybrid" in asset else "Landsat TM/ETM+/OLI",
                    "keywords_raw": "tropical moist forest; TMF; FOROBS; JRC",
                    "gee_assets": [asset],
                })
    (DATA_DIR / "jrc_forobs_tmf_products.json").write_text(
        json.dumps(tmf_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_last_access(
        queried,
        len(all_results) + len(tmf_records),
        "200 OK",
        notes=(
            f"Retrieved {len(all_results)} unique DCAT-AP dataset records from JRC catalog (sort=id+asc, {page} pages) "
            f"+ {len(tmf_records)} JRC FOROBS TMF records."
        ),
    )


if __name__ == "__main__":
    main()
