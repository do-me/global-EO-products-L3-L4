#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: European Environment Agency (EEA SDI) & European Marine Observation and Data Network (EMODnet) GeoNetwork ES APIs
Source ID: 11_eea_and_emodnet
Authoritative Upstream API: POST https://sdi.eea.europa.eu/catalogue/srv/api/search/records/_search + POST https://emodnet.ec.europa.eu/geonetwork/srv/api/search/records/_search

Run standalone via:
    uv run mine.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
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

SOURCE_ID = "11_eea_and_emodnet"
SOURCE_TITLE = "European Environment Agency (EEA SDI) & European Marine Observation and Data Network (EMODnet) GeoNetwork ES APIs"


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
        "User-Agent": "EC4GEO-EO-L3-L4-Catalogue/2.0 (European Commission JRC Scientific Audit; Reproducible Metadata Miner)",
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

def fetch_geonetwork(session: requests.Session, url: str, query_filter: dict[str, Any], max_records: int = 2500) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    page_size = 200
    from_idx = 0
    while from_idx < max_records:
        body = {
            "from": from_idx,
            "size": page_size,
            "_source": {
                "excludes": [
                    "overview",
                    "overview.*",
                    "document",
                    "th_*_tree",
                    "th_*_tree.*",
                    "allKeywords",
                    "recordLink",
                ]
            },
            "query": query_filter,
        }
        r = session.post(url, json=body, headers={"Accept": "application/json"}, timeout=60)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h.get("_source", {})
            src["_id"] = h.get("_id")
            records.append(src)
        if len(hits) < page_size:
            break
        from_idx += page_size
    return records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()

    eea_url = "https://sdi.eea.europa.eu/catalogue/srv/api/search/records/_search"
    eea_query = {
        "bool": {
            "must": [{"terms": {"resourceType": ["dataset", "series"]}}],
            "should": [
                {"terms": {"cl_spatialRepresentationType.key": ["grid", "raster"]}},
                {"match": {" default": "Copernicus"}},
                {"match": {"default": "Natura 2000"}},
                {"match": {"default": "Corine"}},
            ],
            "minimum_should_match": 1,
        }
    }
    eea_records = fetch_geonetwork(session, eea_url, eea_query, max_records=2000)
    with (DATA_DIR / "eea_sdi_datasets.jsonl").open("w", encoding="utf-8") as f:
        for rec in eea_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    emodnet_url = "https://emodnet.ec.europa.eu/geonetwork/srv/api/search/records/_search"
    emodnet_query = {
        "bool": {
            "must": [
                {"terms": {"resourceType": ["dataset", "series"]}},
                {"terms": {"cl_spatialRepresentationType.key": ["grid", "raster"]}},
            ]
        }
    }
    emodnet_records = fetch_geonetwork(session, emodnet_url, emodnet_query, max_records=1500)
    with (DATA_DIR / "emodnet_datasets.jsonl").open("w", encoding="utf-8") as f:
        for rec in emodnet_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_last_access(
        [eea_url, emodnet_url],
        len(eea_records) + len(emodnet_records),
        "200 OK",
        notes=f"EEA gridded/thematic datasets: {len(eea_records)}, EMODnet gridded datasets: {len(emodnet_records)}",
    )


if __name__ == "__main__":
    main()
