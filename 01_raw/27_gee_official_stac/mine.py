#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: Google Earth Engine (GEE) Official STAC Catalogue (Full Tree Traversal)
Source ID: 27_gee_official_stac
Authoritative Upstream API: GET https://earthengine-stac.storage.googleapis.com/catalog/catalog.json + all provider & dataset STAC JSONs

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

SOURCE_ID = "27_gee_official_stac"
SOURCE_TITLE = "Google Earth Engine (GEE) Official STAC Catalogue (Full Tree Traversal)"


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


def fetch_json(session: requests.Session, url: str) -> dict[str, Any] | None:
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session(pool_maxsize=32)
    root_url = "https://earthengine-stac.storage.googleapis.com/catalog/catalog.json"
    r = session.get(root_url, timeout=60)
    r.raise_for_status()
    root = r.json()

    provider_urls = [
        urljoin(root_url, lk["href"])
        for lk in root.get("links", [])
        if lk.get("rel") == "child" and lk.get("href")
    ]

    collection_urls: list[str] = []
    collections: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(fetch_json, session, u): u for u in provider_urls}
        for fut in as_completed(futs):
            parent_url = futs[fut]
            res = fut.result()
            if not res:
                continue
            if res.get("type") == "Collection":
                collections.append(res)
            else:
                for lk in res.get("links", []):
                    if lk.get("rel") == "child" and lk.get("href"):
                        collection_urls.append(urljoin(parent_url, lk["href"]))

    with ThreadPoolExecutor(max_workers=24) as ex:
        futs2 = {ex.submit(fetch_json, session, u): u for u in collection_urls}
        for fut in as_completed(futs2):
            res = fut.result()
            if res and res.get("type") == "Collection":
                collections.append(res)

    collections.sort(key=lambda x: str(x.get("id", "")))
    out_file = DATA_DIR / "gee_official_stac_collections.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for c in collections:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    write_last_access(
        [root_url, f"provider catalogs (x{len(provider_urls)})", f"collection JSONs (x{len(collections)})"],
        len(collections),
        "200 OK",
    )


if __name__ == "__main__":
    main()
