#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
Raw Metadata Miner: JAXA Earth API STAC Catalog (COG v1) — Japan Aerospace Exploration Agency
Source ID: 30_jaxa_earth_stac
Authoritative Upstream API: GET https://data.earth.jaxa.jp/stac/cog/v1/catalog.json (+ child collection.json endpoints)

Run standalone via:
    uv run mine.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAST_ACCESS_FILE = BASE_DIR / "last_access.txt"

SOURCE_ID = "30_jaxa_earth_stac"
SOURCE_TITLE = "JAXA Earth API STAC Catalog (COG v1) — Japan Aerospace Exploration Agency (EORC / G-Portal / JASMES)"
CATALOG_URL = "https://data.earth.jaxa.jp/stac/cog/v1/catalog.json"


def build_session(max_retries: int = 4, pool_maxsize: int = 16) -> requests.Session:
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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
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


def fetch_collection(session: requests.Session, href: str) -> tuple[str, dict[str, Any]]:
    slug = href.rstrip("/").split("/")[-2]
    resp = session.get(href, timeout=45)
    resp.raise_for_status()
    doc = json.loads(resp.content.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"Expected JSON object for {href}")
    doc["_catalog_slug"] = slug
    doc["_collection_url"] = href
    return slug, doc


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    root_resp = session.get(CATALOG_URL, timeout=45)
    root_resp.raise_for_status()
    root_doc = json.loads(root_resp.content.decode("utf-8"))
    (DATA_DIR / "jaxa_earth_stac_root.json").write_text(
        json.dumps(root_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    child_hrefs = [
        urljoin(CATALOG_URL, lk["href"])
        for lk in (root_doc.get("links") or [])
        if isinstance(lk, dict) and lk.get("rel") == "child" and lk.get("href")
    ]
    print(f"[{SOURCE_ID}] Discovered {len(child_hrefs)} child collection links from {CATALOG_URL}")

    results: list[tuple[str, dict[str, Any]]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut_map = {ex.submit(fetch_collection, session, href): href for href in child_hrefs}
        for fut in as_completed(fut_map):
            href = fut_map[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                errors.append(f"{href}: {exc}")

    if errors:
        raise RuntimeError(f"Failed fetching {len(errors)} JAXA STAC collections: {errors[:5]}")

    results.sort(key=lambda item: item[0])
    out_jsonl = DATA_DIR / "jaxa_earth_stac_collections.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, doc in results:
            f.write(json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n")

    write_last_access(
        endpoints=[
            f"{CATALOG_URL} ({len(results)} child STAC collections harvested)",
        ],
        total_records=len(results),
        status_summary="200 OK",
        notes=(
            "Full harvest of JAXA Earth API COG v1 STAC catalog (JAXA EORC, G-Portal, JASMES, "
            "JMA, plus hosted NASA EOSDIS & Copernicus C3S COG collections with explicit producer roles)."
        ),
    )


if __name__ == "__main__":
    main()
