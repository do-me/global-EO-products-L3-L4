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
    "LCSPP LCSIF LCREF",
    "GOSIF Global OCO-2 SIF",
    "CSIF Contiguous Solar-Induced Fluorescence",
    "GLOBMAP Leaf Area Index",
    "TROPOSIF fluorescence",
]

ZENODO_PINNED_RECORD_IDS = [
    14724263,  # GLEAM4 (Miralles et al. / Ghent University, doi:10.5281/zenodo.14724263)
    20683570,  # Global LCREF, LCSIF, and LCSPP (latest version, doi:10.5281/zenodo.20683570)
    13922379,  # Global LCREF, LCSIF, and LCSPP (v2, doi:10.5281/zenodo.13922379)
    7916879,   # Global LCREF, LCSIF, and LCSPP (v1, doi:10.5281/zenodo.7916879)
    10212472,  # Global fluorescence yield based on TROPOSIF data, normalized with NIRvP (doi:10.5281/zenodo.10212472)
    7331411,   # eSIF product: enhanced SIF by combining TROPOSIF, MODIS reflectance and ERA-5 (doi:10.5281/zenodo.7331411)
    2575599,   # VODCA Global Long-term Microwave Vegetation Optical Depth Climate Archive (doi:10.5281/zenodo.2575599)
    11206366,  # OceanSODA-ETHZ-v2 Surface ocean sea-air CO2 fluxes (doi:10.5281/zenodo.11206366)
]

DATACITE_PINNED_DOIS = [
    "10.18160/5NZG-JMJE",        # FLUXCOM-X-BASE global carbon & energy fluxes (ICOS Carbon Portal)
    "10.5281/zenodo.20683570",   # LCSPP / LCSIF / LCREF global SIF & GPP
    "10.5281/zenodo.13922379",   # LCSPP / LCSIF / LCREF v2
    "10.5281/zenodo.7916879",    # LCSPP / LCSIF / LCREF v1
    "10.5281/zenodo.10212472",   # Global fluorescence yield based on TROPOSIF data, normalized with NIRvP
    "10.15454/tihhyp",           # SMOS-IC v2 global soil moisture and L-band vegetation optical depth (INRAE / CESBIO)
    "10.1594/PANGAEA.911944",    # Global terrestrial carbon/water flux EO benchmark (PANGAEA)
    "10.1594/PANGAEA.897399",    # Global EO biophysical benchmark (PANGAEA)
    "10.7910/DVN/EJSXHS",        # CSIF Contiguous Solar-Induced Fluorescence (Harvard Dataverse)
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    queried: list[str] = []

    # 1. FAO GISMGR v2 workspaces and mapsets (with full pagination across all pages)
    workspaces: list[dict[str, Any]] = []
    ws_page = 0
    while True:
        ws_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces?size=100&page={ws_page}"
        r_ws = session.get(ws_url, timeout=60)
        r_ws.raise_for_status()
        ws_resp = r_ws.json().get("response", {})
        ws_items = ws_resp.get("items", [])
        workspaces.extend(ws_items)
        if not ws_resp.get("hasNext") or not ws_items:
            break
        ws_page += 1
    queried.append("https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces?size=100&page=0..N")

    fao_mapsets: list[dict[str, Any]] = []
    seen_mapsets: set[str] = set()
    for ws in workspaces:
        code = ws.get("code")
        if not code:
            continue
        m_page = 0
        while True:
            m_url = f"https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/{code}/mapsets?size=100&page={m_page}"
            rm = session.get(m_url, timeout=30)
            if rm.status_code != 200:
                break
            m_resp = rm.json().get("response", {})
            items = m_resp.get("items", [])
            for it in items:
                m_code = f"{code}:{it.get('code')}"
                if m_code not in seen_mapsets:
                    seen_mapsets.add(m_code)
                    it["_workspace_caption"] = ws.get("caption")
                    fao_mapsets.append(it)
            if not m_resp.get("hasNext") or not items:
                break
            m_page += 1

    (DATA_DIR / "fao_gismgr_workspaces_and_mapsets.json").write_text(
        json.dumps({"workspaces": workspaces, "mapsets": fao_mapsets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. Zenodo REST API for flagship global L3/L4 EO datasets (paginated across pages 1..2)
    zenodo_records: list[dict[str, Any]] = []
    seen_zenodo: set[int] = set()
    for q in ZENODO_QUERIES:
        z_url = "https://zenodo.org/api/records"
        for z_page in (1, 2):
            rz = session.get(
                z_url,
                params={"q": q, "type": "dataset", "size": "15", "page": str(z_page)},
                timeout=60,
            )
            if rz.status_code != 200:
                break
            hits = rz.json().get("hits", {}).get("hits", [])
            for h in hits:
                zid = h.get("id")
                if zid and zid not in seen_zenodo:
                    seen_zenodo.add(zid)
                    h["_matched_benchmark_query"] = q
                    zenodo_records.append(h)
            time.sleep(0.3)
            if len(hits) < 15:
                break

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

    # 3. DataCite API for G3P (GFZ), FLUXCOM-X-BASE (ICOS), LCSPP/LCSIF, CSIF, and PANGAEA benchmarks
    g3p_url = "https://api.datacite.org/dois"
    g3p_records: list[dict[str, Any]] = []
    seen_dc_ids: set[str] = set()

    for dc_params in [
        {"prefix": "10.5880", "query": "g3p", "page[size]": "25"},
        {"prefix": "10.18160", "query": "FLUXCOM", "page[size]": "50"},
    ]:
        r_dc = session.get(g3p_url, params=dc_params, timeout=60)
        if r_dc.status_code == 200:
            queried.append(f"{g3p_url}?prefix={dc_params['prefix']}&query={dc_params['query']}")
            for item in r_dc.json().get("data", []):
                dc_id = str(item.get("id", "")).lower()
                if dc_id and dc_id not in seen_dc_ids:
                    seen_dc_ids.add(dc_id)
                    g3p_records.append(item)

    for doi in DATACITE_PINNED_DOIS:
        if doi.lower() in seen_dc_ids:
            continue
        r_doi = session.get(f"{g3p_url}/{doi}", timeout=30)
        if r_doi.status_code == 200:
            item = r_doi.json().get("data")
            if isinstance(item, dict) and item.get("id"):
                seen_dc_ids.add(str(item["id"]).lower())
                g3p_records.append(item)

    (DATA_DIR / "gfz_datacite_g3p_benchmarks.json").write_text(
        json.dumps({"data": g3p_records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_last_access(
        queried,
        len(fao_mapsets) + len(zenodo_records) + len(g3p_records),
        "200 OK",
        notes=(
            f"FAO GISMGR mapsets (all pages): {len(fao_mapsets)}, Zenodo benchmark datasets: {len(zenodo_records)}, "
            f"DataCite benchmark records (G3P, FLUXCOM-X, LCSPP, PANGAEA, CSIF): {len(g3p_records)}"
        ),
    )


if __name__ == "__main__":
    main()
