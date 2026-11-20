#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
Raw Metadata Miner: Natural Resources Canada (NRCan / CCMEO) Datacube STAC API
Source ID: 32_nrcan_ccmeo_stac
Authoritative Upstream API: GET https://datacube.services.geo.ca/stac/api/collections

Run standalone via:
    uv run mine.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAST_ACCESS_FILE = BASE_DIR / "last_access.txt"

SOURCE_ID = "32_nrcan_ccmeo_stac"
SOURCE_TITLE = "Natural Resources Canada (NRCan / CCMEO) Geospatial Datacube STAC API"
COLLECTIONS_URL = "https://datacube.services.geo.ca/stac/api/collections"


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
        "User-Agent": "EC4GEO-EO-L3-L4-Catalogue/2.0 (Scientific EO Metadata Miner)",
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

    collections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    url: str | None = COLLECTIONS_URL
    while url:
        resp = session.get(url, timeout=45)
        resp.raise_for_status()
        payload = json.loads(resp.content.decode("utf-8"))
        for col in payload.get("collections") or []:
            cid = str(col.get("id") or "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                collections.append(col)
        next_links = [
            urljoin(url, lk["href"])
            for lk in (payload.get("links") or [])
            if isinstance(lk, dict) and lk.get("rel") == "next" and lk.get("href")
        ]
        url = next_links[0] if next_links else None

    collections.sort(key=lambda c: str(c.get("id") or ""))
    out_file = DATA_DIR / "nrcan_ccmeo_stac.json"
    out_file.write_text(
        json.dumps({"collections": collections}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_last_access(
        endpoints=[f"{COLLECTIONS_URL} ({len(collections)} collections)"],
        total_records=len(collections),
        status_summary="200 OK",
        notes=(
            "Full harvest of Natural Resources Canada / Canada Centre for Mapping and Earth Observation "
            "(CCMEO) Geo.ca STAC collections (satellite ARD, land cover, water, ET, snow, DEMs, plus "
            "aerial/socioeconomic collections filtered deterministically in 02_preparation)."
        ),
    )


if __name__ == "__main__":
    main()
