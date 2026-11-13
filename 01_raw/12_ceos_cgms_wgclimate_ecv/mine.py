#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: Joint CEOS/CGMS Working Group on Climate (WGClimate) ECV Climate Data Record (CDR) Inventory API
Source ID: 12_ceos_cgms_wgclimate_ecv
Authoritative Upstream API: GET https://www.cdrinventory.climatemonitoring.info/backend/api/v1/data_records/ + /variables/nodes/graph/

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

SOURCE_ID = "12_ceos_cgms_wgclimate_ecv"
SOURCE_TITLE = "Joint CEOS/CGMS Working Group on Climate (WGClimate) ECV Climate Data Record (CDR) Inventory API"


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

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    url_records = "https://www.cdrinventory.climatemonitoring.info/backend/api/v1/data_records/"
    r1 = session.get(url_records, timeout=60)
    r1.raise_for_status()
    d1 = r1.json()
    records = d1 if isinstance(d1, list) else d1.get("results", d1.get("data", []))
    (DATA_DIR / "wgclimate_ecv_data_records.json").write_text(
        json.dumps(d1, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    url_graph = "https://www.cdrinventory.climatemonitoring.info/backend/api/v1/variables/nodes/graph/"
    r2 = session.get(url_graph, timeout=60)
    if r2.status_code == 200:
        (DATA_DIR / "wgclimate_ecv_variables_graph.json").write_text(
            json.dumps(r2.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_last_access([url_records, url_graph], len(records), f"{r1.status_code} OK")


if __name__ == "__main__":
    main()
