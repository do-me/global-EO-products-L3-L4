#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: WMO OSCAR (Observing Systems Capability Analysis and Review Tool) Space & GCOS Variables
Source ID: 13_wmo_oscar_and_gcos
Authoritative Upstream API: GET https://space.oscar.wmo.int/variables + GET https://space.oscar.wmo.int/satellites

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

SOURCE_ID = "13_wmo_oscar_and_gcos"
SOURCE_TITLE = "WMO OSCAR (Observing Systems Capability Analysis and Review Tool) Space & GCOS Variables"


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

def fetch_page(session: requests.Session, resource: str, page: int) -> list[dict[str, Any]]:
    url = f"https://space.oscar.wmo.int/api/v1/{resource}?page={page}"
    try:
        r = session.get(url, timeout=25)
        if r.status_code == 200:
            return r.json().get("_embedded", {}).get(resource, [])
    except Exception:
        pass
    return []


def fetch_oscar_resource(session: requests.Session, resource: str) -> list[dict[str, Any]]:
    url1 = f"https://space.oscar.wmo.int/api/v1/{resource}?page=1"
    r1 = session.get(url1, timeout=25)
    r1.raise_for_status()
    d1 = r1.json()
    items: list[dict[str, Any]] = list(d1.get("_embedded", {}).get(resource, []))
    total_pages = int(d1.get("page", {}).get("totalPages", 1))
    if total_pages > 1:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_page, session, resource, p): p for p in range(2, total_pages + 1)}
            for fut in as_completed(futs):
                items.extend(fut.result())
    items.sort(key=lambda x: str(x.get("id", x.get("slug", ""))))
    return items


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session(max_retries=2, pool_maxsize=16)
    variables = fetch_oscar_resource(session, "variables")
    satellites = fetch_oscar_resource(session, "satellites")

    (DATA_DIR / "wmo_oscar_variables.json").write_text(
        json.dumps({"variables": variables}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA_DIR / "wmo_oscar_satellites.json").write_text(
        json.dumps({"satellites": satellites}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = len(variables) + len(satellites)
    write_last_access(
        [
            "https://space.oscar.wmo.int/api/v1/variables",
            "https://space.oscar.wmo.int/api/v1/satellites",
        ],
        total,
        "200 OK",
        notes=f"Variables: {len(variables)}, Satellites: {len(satellites)}",
    )


if __name__ == "__main__":
    main()
