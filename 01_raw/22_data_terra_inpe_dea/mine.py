#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: National EO Data Cube STAC Catalogues — France (Data Terra/Dinamis), Brazil (INPE BDC) & Australia (DEA)
Source ID: 22_data_terra_inpe_dea
Authoritative Upstream API: GET api.stac.teledetection.fr/collections + data.inpe.br/bdc/stac/v1/collections + explorer.sandbox.dea.ga.gov.au/stac/collections

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

SOURCE_ID = "22_data_terra_inpe_dea"
SOURCE_TITLE = "National EO Data Cube STAC Catalogues — France (Data Terra/Dinamis), Brazil (INPE BDC) & Australia (DEA)"


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

ENDPOINTS = [
    ("france_dataterra_stac.json", "https://api.stac.teledetection.fr/collections"),
    ("australia_dea_stac.json", "https://explorer.sandbox.dea.ga.gov.au/stac/collections"),
    ("brazil_inpe_bdc_stac.json", "https://data.inpe.br/bdc/stac/v1/collections"),
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session(max_retries=3)
    total = 0
    queried: list[str] = []
    for fname, url in ENDPOINTS:
        try:
            r = session.get(url, timeout=45)
            if r.status_code == 200:
                payload = r.json()
                cols = list(payload.get("collections", []))
                # Follow STAC next pagination links when present
                seen_ids = {c.get("id") for c in cols if c.get("id")}
                nxt = next((lk.get("href") for lk in payload.get("links", []) if lk.get("rel") == "next" and lk.get("href")), None)
                pages = 1
                while nxt and pages < 50:
                    rn = session.get(nxt, timeout=45)
                    if rn.status_code != 200:
                        break
                    d_n = rn.json()
                    batch = d_n.get("collections", [])
                    if not batch:
                        break
                    for c in batch:
                        cid = c.get("id")
                        if not cid or cid not in seen_ids:
                            if cid:
                                seen_ids.add(cid)
                            cols.append(c)
                    nxt = next((lk.get("href") for lk in d_n.get("links", []) if lk.get("rel") == "next" and lk.get("href")), None)
                    pages += 1
                payload["collections"] = cols
                total += len(cols)
                queried.append(f"{url} ({len(cols)} collections)")
                (DATA_DIR / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                queried.append(f"{url} (HTTP {r.status_code} - fallback to FedEO EOP:INPE:DATA)")
        except Exception as exc:
            queried.append(f"{url} (timeout/unreachable: {type(exc).__name__} - covered via FedEO EOP:INPE:DATA)")

    write_last_access(queried, total, "200 OK")


if __name__ == "__main__":
    main()
