#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: World Resources Institute (WRI) — Global Forest Watch (GFW) & Resource Watch Data APIs
Source ID: 23_wri_gfw_and_resourcewatch
Authoritative Upstream API: GET https://data-api.globalforestwatch.org/datasets + GET https://api.resourcewatch.org/v1/dataset

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

SOURCE_ID = "23_wri_gfw_and_resourcewatch"
SOURCE_TITLE = "World Resources Institute (WRI) — Global Forest Watch (GFW) & Resource Watch Data APIs"


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
    gfw_url = "https://data-api.globalforestwatch.org/datasets"
    r1 = session.get(gfw_url, timeout=60)
    r1.raise_for_status()
    gfw_data = r1.json()
    gfw_items = gfw_data.get("data", [])
    (DATA_DIR / "wri_gfw_datasets.json").write_text(json.dumps(gfw_data, ensure_ascii=False, indent=2), encoding="utf-8")

    rw_items: list[dict[str, Any]] = []
    page = 1
    while page <= 200:
        rw_url = f"https://api.resourcewatch.org/v1/dataset?page[size]=250&page[number]={page}&includes=metadata"
        r2 = session.get(rw_url, timeout=60)
        if r2.status_code != 200:
            break
        batch = r2.json().get("data", [])
        if not batch:
            break
        rw_items.extend(batch)
        meta = r2.json().get("meta", {})
        if page >= meta.get("total-pages", 1):
            break
        page += 1

    with (DATA_DIR / "wri_resourcewatch_datasets.jsonl").open("w", encoding="utf-8") as f:
        for it in rw_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_last_access(
        [gfw_url, "https://api.resourcewatch.org/v1/dataset?page[size]=250&includes=metadata"],
        len(gfw_items) + len(rw_items),
        "200 OK",
        notes=f"GFW datasets: {len(gfw_items)}, Resource Watch datasets: {len(rw_items)}",
    )


if __name__ == "__main__":
    main()
