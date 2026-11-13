#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: FAO GISMGR v2 (WaPOR, GAEZ, ASAP, GSOC) & Zenodo Authoritative Global EO L3/L4 Benchmarks
Source ID: 26_fao_unep_and_thematic_benchmarks
Authoritative Upstream API: GET https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces + GET https://zenodo.org/api/records

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

SOURCE_ID = "26_fao_unep_and_thematic_benchmarks"
SOURCE_TITLE = "FAO GISMGR v2 (WaPOR, GAEZ, ASAP, GSOC) & Zenodo Authoritative Global EO L3/L4 Benchmarks"


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

ZENODO_QUERIES = [
    "Global Mangrove Watch",
    "Global Pasture Watch",
    "ETH Global Canopy Height 10m",
    "SoilGrids 250m",
    "HydroSHEDS global",
    "Global Land Cover 30m GLC_FCS30D",
    "OpenLandMap Soil Organic Carbon",
    "WorldCereal global",
    "Global Surface Water Explorer",
    "Global Dam Watch",
    "Global Wetlands Map",
    "Global Kelp Forest",
    "Global Land Evaporation Amsterdam Model",
]

ZENODO_PINNED_RECORD_IDS = [
    14724263,  # GLEAM4 (Miralles et al. / Ghent University, doi:10.5281/zenodo.14724263)
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    queried: list[str] = []

    # 1. FAO GISMGR v2 workspaces and mapsets
    ws_url = "https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces?size=100"
    r_ws = session.get(ws_url, timeout=60)
    r_ws.raise_for_status()
    workspaces = r_ws.json().get("response", {}).get("items", [])
    queried.append(ws_url)

    fao_mapsets: list[dict[str, Any]] = []
    for ws in workspaces:
        code = ws.get("code")
        if not code:
            continue
        m_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/{code}/mapsets?size=100"
        rm = session.get(m_url, timeout=30)
        if rm.status_code == 200:
            items = rm.json().get("response", {}).get("items", [])
            for it in items:
                it["_workspace_caption"] = ws.get("caption")
                fao_mapsets.append(it)

    (DATA_DIR / "fao_gismgr_workspaces_and_mapsets.json").write_text(
        json.dumps({"workspaces": workspaces, "mapsets": fao_mapsets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. Zenodo REST API for flagship global L3/L4 EO datasets
    zenodo_records: list[dict[str, Any]] = []
    seen_zenodo: set[int] = set()
    for q in ZENODO_QUERIES:
        z_url = "https://zenodo.org/api/records"
        rz = session.get(z_url, params={"q": q, "type": "dataset", "size": "15"}, timeout=60)
        if rz.status_code == 200:
            hits = rz.json().get("hits", {}).get("hits", [])
            for h in hits:
                zid = h.get("id")
                if zid and zid not in seen_zenodo:
                    seen_zenodo.add(zid)
                    h["_matched_benchmark_query"] = q
                    zenodo_records.append(h)
        time.sleep(0.3)

    for z_rec_id in ZENODO_PINNED_RECORD_IDS:
        if z_rec_id in seen_zenodo:
            continue
        r_pin = session.get(f"https://zenodo.org/api/records/{z_rec_id}", timeout=30)
        if r_pin.status_code == 200:
            h = r_pin.json()
            zid = h.get("id")
            if zid and zid not in seen_zenodo:
                seen_zenodo.add(zid)
                h["_matched_benchmark_query"] = "pinned_authoritative_eo_flagship"
                zenodo_records.append(h)

    queried.append("https://zenodo.org/api/records?type=dataset (thematic L3/L4 benchmark queries)")
    with (DATA_DIR / "zenodo_thematic_benchmarks.jsonl").open("w", encoding="utf-8") as f:
        for rec in zenodo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3. GFZ Data Services (DataCite API) for Horizon 2020 Global Gravity-based Groundwater Product (G3P)
    g3p_url = "https://api.datacite.org/dois"
    g3p_records: list[dict[str, Any]] = []
    r_g3p = session.get(g3p_url, params={"prefix": "10.5880", "query": "g3p", "page[size]": "25"}, timeout=60)
    if r_g3p.status_code == 200:
        queried.append(f"{g3p_url}?prefix=10.5880&query=g3p&page[size]=25")
        g3p_records = r_g3p.json().get("data", [])
        (DATA_DIR / "gfz_datacite_g3p_benchmarks.json").write_text(
            json.dumps({"data": g3p_records}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_last_access(
        queried,
        len(fao_mapsets) + len(zenodo_records) + len(g3p_records),
        "200 OK",
        notes=(
            f"FAO GISMGR mapsets: {len(fao_mapsets)}, Zenodo benchmark datasets: {len(zenodo_records)}, "
            f"GFZ DataCite G3P records: {len(g3p_records)}"
        ),
    )


if __name__ == "__main__":
    main()
