#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: Copernicus Data Space Ecosystem (CDSE) STAC API & CEOS FedEO CDSE Collection Index
Source ID: 10_cdse_stac
Authoritative Upstream API: GET https://stac.dataspace.copernicus.eu/v1/collections + GET https://fedeo.ceos.org/collections?parentIdentifier=EOP:EU:CDSE

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

SOURCE_ID = "10_cdse_stac"
SOURCE_TITLE = "Copernicus Data Space Ecosystem (CDSE) STAC API & CEOS FedEO CDSE Collection Index"


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
    url1 = "https://stac.dataspace.copernicus.eu/v1/collections?limit=1000"
    cols1: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    next_url: str | None = url1
    first_payload: dict[str, Any] = {}
    while next_url:
        r1 = session.get(next_url, timeout=60)
        r1.raise_for_status()
        d1 = r1.json()
        if not first_payload:
            first_payload = d1
        batch = d1.get("collections", [])
        new_in_page = 0
        for c in batch:
            cid = str(c.get("id", ""))
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                cols1.append(c)
                new_in_page += 1
        next_link = None
        for lk in d1.get("links", []):
            if isinstance(lk, dict) and lk.get("rel") == "next" and lk.get("href"):
                next_link = str(lk["href"])
                break
        if not next_link or next_link == next_url or new_in_page == 0:
            break
        next_url = next_link

    first_payload["collections"] = cols1
    (DATA_DIR / "cdse_native_stac_collections.json").write_text(
        json.dumps(first_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fedeo_cols: list[dict[str, Any]] = []
    start = 1
    while True:
        f_url = f"https://fedeo.ceos.org/collections?parentIdentifier=EOP:EU:CDSE&httpAccept=application/json&maximumRecords=100&startRecord={start}"
        rf = session.get(f_url, timeout=60)
        if rf.status_code != 200:
            break
        batch = rf.json().get("collections", [])
        if not batch:
            break
        fedeo_cols.extend(batch)
        if len(batch) < 100:
            break
        start += 100

    (DATA_DIR / "cdse_fedeo_stac_collections.json").write_text(
        json.dumps({"collections": fedeo_cols}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_last_access(
        [url1, "https://fedeo.ceos.org/collections?parentIdentifier=EOP:EU:CDSE"],
        len(cols1) + len(fedeo_cols),
        "200 OK",
        notes=f"Native CDSE STAC: {len(cols1)}, FedEO CDSE: {len(fedeo_cols)}",
    )


if __name__ == "__main__":
    main()
