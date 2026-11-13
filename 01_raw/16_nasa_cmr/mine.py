#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: NASA Common Metadata Repository (CMR) Collections API — All Explicit L3/L4 + SEDAC/ORNL Gridded Collections
Source ID: 16_nasa_cmr
Authoritative Upstream API: GET https://cmr.earthdata.nasa.gov/search/collections.json?processing_level_id[]=3&processing_level_id[]=4

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

SOURCE_ID = "16_nasa_cmr"
SOURCE_TITLE = "NASA Common Metadata Repository (CMR) Collections API — All Explicit L3/L4 + SEDAC/ORNL Gridded Collections"


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

def fetch_cmr_entries(session: requests.Session, params_list: list[tuple[str, str]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    page_num = 1
    page_size = 2000
    while True:
        q = list(params_list) + [("page_size", str(page_size)), ("page_num", str(page_num))]
        r = session.get("https://cmr.earthdata.nasa.gov/search/collections.json", params=q, timeout=90)
        r.raise_for_status()
        batch = r.json().get("feed", {}).get("entry", [])
        if not batch:
            break
        entries.extend(batch)
        if len(batch) < page_size:
            break
        page_num += 1
    return entries


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()

    # 1. All explicit Level 3 and Level 4 collections across all CMR providers
    l3_l4_params = [("processing_level_id[]", "3"), ("processing_level_id[]", "4")]
    l3_l4_entries = fetch_cmr_entries(session, l3_l4_params)

    # Save in chunks of 2,000 entries (~5-7 MB per file)
    chunk_size = 2000
    for idx in range(0, len(l3_l4_entries), chunk_size):
        chunk = l3_l4_entries[idx : idx + chunk_size]
        part = (idx // chunk_size) + 1
        out_path = DATA_DIR / f"nasa_cmr_l3_l4_part{part:02d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for e in chunk:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 2. SEDAC collections (socioeconomic/environmental grids often lacking processing_level_id)
    sedac_entries = fetch_cmr_entries(session, [("provider", "SEDAC")])
    with (DATA_DIR / "nasa_cmr_sedac.jsonl").open("w", encoding="utf-8") as f:
        for e in sedac_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total = len(l3_l4_entries) + len(sedac_entries)
    write_last_access(
        [
            "https://cmr.earthdata.nasa.gov/search/collections.json?processing_level_id[]=3&processing_level_id[]=4&page_size=2000",
            "https://cmr.earthdata.nasa.gov/search/collections.json?provider=SEDAC&page_size=2000",
        ],
        total,
        "200 OK",
        notes=f"Explicit L3/L4 collections: {len(l3_l4_entries)}, SEDAC collections: {len(sedac_entries)}",
    )


if __name__ == "__main__":
    main()
