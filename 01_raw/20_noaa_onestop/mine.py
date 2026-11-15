#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: NOAA OneStop Data Search REST API (NOAA CDRs, NESDIS/STAR/NCEI Gridded Satellite Products)
Source ID: 20_noaa_onestop
Authoritative Upstream API: POST https://data.noaa.gov/onestop/api/search/search/collection

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

SOURCE_ID = "20_noaa_onestop"
SOURCE_TITLE = "NOAA OneStop Data Search REST API (NOAA CDRs, NESDIS/STAR/NCEI Gridded Satellite Products)"


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

QUERIES = [
    "Climate Data Record",
    "Level 3",
    "Level 4",
    "Coral Reef Watch",
    "VIIRS gridded",
    "AVHRR Pathfinder",
    "Optimum Interpolation Sea Surface Temperature",
    "World Ocean Atlas",
    "ETOPO",
    "NOAA STAR",
    "NESDIS global",
]


def fetch_query_records(
    session: requests.Session,
    url: str,
    query: str,
    page_size: int = 200,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch all records for a NOAA OneStop query with summary=False.

    OneStop enforces an Elasticsearch max_result_window of 10,000 (offset + max <= 10,000).
    When meta.total > 10,000, we retrieve the full set by paging ascending (stagedDate asc,
    offset 0..9800) and descending (stagedDate desc, offset 0..remaining+overlap) and deduplicating
    by record id. Every HTTP response is verified to be 200 OK.
    """
    first_payload = {
        "queries": [{"type": "queryText", "value": query}],
        "page": {"max": page_size, "offset": 0},
        "summary": False,
    }
    r0 = session.post(url, json=first_payload, headers={"Content-Type": "application/json"}, timeout=60)
    r0.raise_for_status()
    res0 = r0.json()
    total_hits = int((res0.get("meta") or {}).get("total") or 0)
    if total_hits == 0:
        return [], 0

    by_id: dict[str, dict[str, Any]] = {}
    if total_hits <= 10000:
        for item in res0.get("data", []):
            iid = item.get("id")
            if iid and iid not in by_id:
                by_id[iid] = item
        offset = len(res0.get("data", []))
        while offset < total_hits:
            payload = {
                "queries": [{"type": "queryText", "value": query}],
                "page": {"max": page_size, "offset": offset},
                "summary": False,
            }
            r = session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            r.raise_for_status()
            batch = r.json().get("data", [])
            if not batch:
                break
            for item in batch:
                iid = item.get("id")
                if iid and iid not in by_id:
                    by_id[iid] = item
            offset += len(batch)
            if len(batch) < page_size:
                break
    else:
        # Ascending pass: offset 0 .. 9800 (10,000 records)
        for offset in range(0, 10000, page_size):
            payload = {
                "queries": [{"type": "queryText", "value": query}],
                "page": {"max": page_size, "offset": offset},
                "sort": [{"stagedDate": "asc"}],
                "summary": False,
            }
            r = session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            r.raise_for_status()
            batch = r.json().get("data", [])
            if not batch:
                break
            for item in batch:
                iid = item.get("id")
                if iid and iid not in by_id:
                    by_id[iid] = item
        # Descending pass to cover the remaining tail beyond 10,000 with a 600-record safety overlap
        desc_limit = min(10000, max(page_size, total_hits - 10000 + 600))
        for offset in range(0, desc_limit, page_size):
            payload = {
                "queries": [{"type": "queryText", "value": query}],
                "page": {"max": page_size, "offset": offset},
                "sort": [{"stagedDate": "desc"}],
                "summary": False,
            }
            r = session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            r.raise_for_status()
            batch = r.json().get("data", [])
            if not batch:
                break
            for item in batch:
                iid = item.get("id")
                if iid and iid not in by_id:
                    by_id[iid] = item

    if total_hits > 20000:
        raise RuntimeError(
            f"NOAA OneStop query '{query}' returned {total_hits} hits, exceeding the 20,000 dual-sort pagination window."
        )
    if len(by_id) != total_hits:
        raise RuntimeError(
            f"Incomplete NOAA OneStop harvest for query '{query}': retrieved {len(by_id)} unique IDs vs {total_hits} expected hits."
        )

    return list(by_id.values()), total_hits


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    url = "https://data.noaa.gov/onestop/api/search/search/collection"
    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    query_stats: list[str] = []

    for q in QUERIES:
        q_items, total_hits = fetch_query_records(session, url, q, page_size=200)
        added = 0
        for item in q_items:
            iid = item.get("id")
            if iid and iid not in seen_ids:
                seen_ids.add(iid)
                records.append(item)
                added += 1
        query_stats.append(f"{q}: retrieved={len(q_items)}/{total_hits} (new={added})")
        print(f"[{SOURCE_ID}] {query_stats[-1]}")

    for old in DATA_DIR.glob("noaa_onestop_collections*.jsonl"):
        old.unlink()
    # Save in chunks of 1,500 full-metadata records (~7-9 MB per file, well below 48 MB limit)
    chunk_size = 1500
    for idx in range(0, len(records), chunk_size):
        chunk = records[idx : idx + chunk_size]
        part = (idx // chunk_size) + 1
        out_file = DATA_DIR / f"noaa_onestop_collections_part{part:02d}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for rec in chunk:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_last_access(
        [f"POST {url} (summary=False, queries={QUERIES}, dual-sort pagination for >10k window)"],
        len(records),
        "200 OK",
        notes="; ".join(query_stats),
    )


if __name__ == "__main__":
    main()
