#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: WorldPop (University of Southampton) STAC API & CIESIN (Columbia University) STAC Catalogue
Source ID: 25_worldpop_and_ciesin_stac
Authoritative Upstream API: GET https://api.stac.worldpop.org/collections + GET https://ciesin.github.io/sci-apps-stac/stac/catalog.json

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

SOURCE_ID = "25_worldpop_and_ciesin_stac"
SOURCE_TITLE = "WorldPop (University of Southampton) STAC API & CIESIN (Columbia University) STAC Catalogue"


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

from urllib.parse import urljoin


def crawl_stac(session: requests.Session, url: str, visited: set[str], results: list[dict[str, Any]]) -> None:
    if url in visited or len(visited) > 300:
        return
    visited.add(url)
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            return
        data = r.json()
    except Exception:
        return
    if data.get("type") == "Collection" or "extent" in data:
        results.append(data)
    for lk in data.get("links", []):
        if lk.get("rel") == "child" and lk.get("href"):
            crawl_stac(session, urljoin(url, lk["href"]), visited, results)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    wp_url = "https://api.stac.worldpop.org/collections"
    r1 = session.get(wp_url, timeout=60)
    r1.raise_for_status()
    wp_data = r1.json()
    wp_cols = wp_data.get("collections", [])
    (DATA_DIR / "worldpop_stac_collections.json").write_text(
        json.dumps(wp_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ciesin_root = "https://ciesin.github.io/sci-apps-stac/stac/catalog.json"
    visited: set[str] = set()
    ciesin_cols: list[dict[str, Any]] = []
    crawl_stac(session, ciesin_root, visited, ciesin_cols)
    with (DATA_DIR / "ciesin_stac_collections.jsonl").open("w", encoding="utf-8") as f:
        for c in ciesin_cols:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    write_last_access(
        [wp_url, ciesin_root],
        len(wp_cols) + len(ciesin_cols),
        "200 OK",
        notes=f"WorldPop collections: {len(wp_cols)}, CIESIN STAC nodes: {len(ciesin_cols)}",
    )


if __name__ == "__main__":
    main()
