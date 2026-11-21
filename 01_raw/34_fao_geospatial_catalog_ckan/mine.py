#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
Raw Metadata Miner: FAO Hand-in-Hand Geospatial & Map Catalog (CKAN / ISO-19115)
Source ID: 34_fao_geospatial_catalog_ckan
Authoritative Upstream API: GET https://data.apps.fao.org/catalog/api/3/action/package_search

Harvests FAO's dedicated Earth Observation and gridded geospatial organizations (excluding
scanned paper maps, tabular FAOSTAT/CountrySTAT indicators, and WaPOR already covered via
FAO GISMGR v2 in Source 26), preserving full ISO-19115 spatial, temporal, and resolution fields.

Run standalone via:
    uv run mine.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAST_ACCESS_FILE = BASE_DIR / "last_access.txt"

SOURCE_ID = "34_fao_geospatial_catalog_ckan"
SOURCE_TITLE = "FAO Geospatial & Hand-in-Hand Map Catalog (CKAN / ISO-19115)"
CKAN_SEARCH_URL = "https://data.apps.fao.org/catalog/api/3/action/package_search"

# Curated FAO EO / gridded geospatial organizations (excludes fao-paper-maps, faostat,
# fao-countrystat, hand-in-hand-platform-master-config, and wapor already in Source 26 GISMGR)
FAO_EO_ORGANIZATIONS = [
    "fao-land-cover-land-use",
    "glosis",
    "fao-gaez",
    "fao-giews",
    "fao-eostat",
    "climate-risk-toolbox-crtb",
    "fao-aquastat",
    "fao-food-insecurity-poverty-and-environment-global-gis-database-fggd",
    "drive-global-drought-vulnerability-inventory",
    "national-forest-monitoring-systems-nfms",
]


def build_session(max_retries: int = 4) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
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

    all_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    org_counts: dict[str, int] = {}
    queried: list[str] = []

    for org in FAO_EO_ORGANIZATIONS:
        start = 0
        rows = 100
        fetched_org = 0
        total_org: int | None = None
        while True:
            resp = session.get(
                CKAN_SEARCH_URL,
                params={"fq": f"organization:{org}", "rows": rows, "start": start, "sort": "id asc"},
                timeout=60,
            )
            resp.raise_for_status()
            res = resp.json().get("result") or {}
            if total_org is None:
                total_org = int(res.get("count", 0))
            batch = res.get("results") or []
            if not batch:
                break
            for rec in batch:
                rid = str(rec.get("id") or rec.get("name") or "")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    all_records.append(rec)
                    fetched_org += 1
            start += len(batch)
            if start >= total_org:
                break
        org_counts[org] = fetched_org
        queried.append(f"{CKAN_SEARCH_URL}?fq=organization:{org} ({fetched_org}/{total_org} records)")
        print(f"[{SOURCE_ID}] {org}: {fetched_org}/{total_org} records")

    all_records.sort(key=lambda r: str(r.get("id") or r.get("name") or ""))
    out_file = DATA_DIR / "fao_geospatial_catalog_ckan.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_last_access(
        endpoints=queried,
        total_records=len(all_records),
        status_summary="200 OK",
        notes=(
            f"Harvested {len(all_records)} records across {len(FAO_EO_ORGANIZATIONS)} FAO EO/geospatial "
            f"CKAN organizations (GLC-SHARE, HWSD v1.2/v2.01/v2.2, GAEZ v4/v5, GSOCmap/GloSIS, GLADA, "
            f"FAO ASIS, GMIA v5 irrigation, CRTB, FGGD, DRIVE, EOSTAT, NFMS). Non-grid vectors, "
            f"software packages, and socioeconomic/health indicators are filtered deterministically in 02_preparation."
        ),
    )


if __name__ == "__main__":
    main()
