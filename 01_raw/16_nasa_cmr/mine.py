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

UMM_KEEP_KEYS = (
    "ShortName",
    "Version",
    "EntryTitle",
    "DOI",
    "AssociatedDOIs",
    "CollectionProgress",
    "DataMaturity",
    "CollectionDataType",
    "StandardProduct",
    "ProcessingLevel",
    "SpatialExtent",
    "TilingIdentificationSystems",
    "TemporalExtents",
    "TemporalKeywords",
    "ScienceKeywords",
    "Platforms",
    "Projects",
    "DataCenters",
    "ArchiveAndDistributionInformation",
    "DirectDistributionInformation",
    "RelatedUrls",
    "UseConstraints",
    "AccessConstraints",
    "ISOTopicCategories",
    "LocationKeywords",
)


def compact_umm_item(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("meta") or {}
    umm = item.get("umm") or {}
    compact: dict[str, Any] = {
        "_umm_meta": {
            "concept-id": meta.get("concept-id"),
            "revision-id": meta.get("revision-id"),
            "provider-id": meta.get("provider-id"),
            "native-id": meta.get("native-id"),
        }
    }
    for k in UMM_KEEP_KEYS:
        if k in umm and umm[k] is not None:
            compact[k] = umm[k]
    add_attrs = umm.get("AdditionalAttributes")
    if isinstance(add_attrs, list):
        kept_attrs = [
            {
                "Name": a.get("Name"),
                "DataType": a.get("DataType"),
                "Value": a.get("Value"),
                "ParameterUnitsOfMeasure": a.get("ParameterUnitsOfMeasure"),
            }
            for a in add_attrs
            if isinstance(a, dict)
            and (
                a.get("Value") is not None
                or any(
                    tok in str(a.get("Name", "")).lower()
                    for tok in ("doi", "resolution", "level", "frequency", "grid", "projection")
                )
            )
        ]
        if kept_attrs:
            compact["AdditionalAttributes"] = kept_attrs
    return compact


def fetch_cmr_entries(session: requests.Session, params_list: list[tuple[str, str]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    umm_by_id: dict[str, dict[str, Any]] = {}
    page_num = 1
    page_size = 2000
    while True:
        q = list(params_list) + [("page_size", str(page_size)), ("page_num", str(page_num))]
        r = session.get("https://cmr.earthdata.nasa.gov/search/collections.json", params=q, timeout=120)
        r.raise_for_status()
        batch = r.json().get("feed", {}).get("entry", [])
        if not batch:
            break
        entries.extend(batch)

        r_umm = session.get("https://cmr.earthdata.nasa.gov/search/collections.umm_json", params=q, timeout=120)
        r_umm.raise_for_status()
        umm_items = r_umm.json().get("items", [])
        for u_item in umm_items:
            cid = (u_item.get("meta") or {}).get("concept-id")
            if cid:
                umm_by_id[str(cid)] = compact_umm_item(u_item)

        if len(batch) < page_size:
            break
        page_num += 1

    for e in entries:
        cid = str(e.get("id", ""))
        if cid in umm_by_id:
            e["_umm"] = umm_by_id[cid]
    return entries


SUPPLEMENTARY_EO_KEYWORDS = [
    "OAFlux",
    "Global Web-Enabled Landsat Data",
    "Fengyun",
    "EarthCARE",
    "GPCP",
    "CMAP precipitation",
    "Global Forest Cover Change",
    "Daymet",
    "MERRA",
    "LPJ-EOSIM",
    "CYGNSS Heat Flux",
    "GlobeLand30",
    "AVISO altimetry",
    "CCI Land Cover",
    "GRACE Tellus",
]

L3_L4_PROCESSING_LEVEL_IDS = [
    "3",
    "4",
    "3A",
    "3B",
    "3C",
    "3G",
    "4A",
    "4B",
    "L3",
    "L4",
    "Level 3",
    "Level 4",
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()

    # 1. All explicit Level 3 and Level 4 collections (including 3A/3B/3C/3G/4A/4B/L3/L4) across all CMR providers
    l3_l4_params = [("processing_level_id[]", lvl) for lvl in L3_L4_PROCESSING_LEVEL_IDS]
    raw_l3_l4 = fetch_cmr_entries(session, l3_l4_params)
    l3_l4_entries: list[dict[str, Any]] = []
    seen_l3_l4: set[str] = set()
    for e in raw_l3_l4:
        cid = str(e.get("id", ""))
        if cid and cid not in seen_l3_l4:
            seen_l3_l4.add(cid)
            l3_l4_entries.append(e)

    for old in DATA_DIR.glob("nasa_cmr_l3_l4_part*.jsonl"):
        old.unlink()

    # Save in chunks of 1,000 entries (~9-12 MB per file, well below 48 MB limit)
    chunk_size = 1000
    for idx in range(0, len(l3_l4_entries), chunk_size):
        chunk = l3_l4_entries[idx : idx + chunk_size]
        part = (idx // chunk_size) + 1
        out_path = DATA_DIR / f"nasa_cmr_l3_l4_part{part:02d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for e in chunk:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 2. SEDAC collections (NASA consolidated SEDAC into ESDIS; query keyword=SEDAC & filter legacy provider metadata cleanly)
    sedac_entries = fetch_cmr_entries(session, [("keyword", "SEDAC")])
    with (DATA_DIR / "nasa_cmr_sedac.jsonl").open("w", encoding="utf-8") as f:
        for e in sedac_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 3. High-recall supplementary EO queries for gridded/CDR/mosaic series lacking L3/L4 tags
    seen_ids = {e.get("id") for e in l3_l4_entries + sedac_entries if e.get("id")}
    supp_entries: list[dict[str, Any]] = []
    for kw in SUPPLEMENTARY_EO_KEYWORDS:
        for e in fetch_cmr_entries(session, [("keyword", kw)]):
            cid = e.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                supp_entries.append(e)

    supp_path = DATA_DIR / "nasa_cmr_supplementary_eo.jsonl"
    with supp_path.open("w", encoding="utf-8") as f:
        for e in supp_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total = len(l3_l4_entries) + len(sedac_entries) + len(supp_entries)
    write_last_access(
        [
            "https://cmr.earthdata.nasa.gov/search/collections.json + collections.umm_json (processing_level_id[]=3,4,3A,3B,3C,3G,4A,4B,L3,L4,Level 3,Level 4)",
            "https://cmr.earthdata.nasa.gov/search/collections.json + collections.umm_json?keyword=SEDAC&page_size=2000",
            f"https://cmr.earthdata.nasa.gov/search/collections.json + collections.umm_json?keyword=<high-recall-eo>&page_size=2000 (keywords={SUPPLEMENTARY_EO_KEYWORDS})",
        ],
        total,
        "200 OK",
        notes=(
            f"Explicit L3/L4 collections (with UMM-JSON): {len(l3_l4_entries)}, "
            f"SEDAC collections (with UMM-JSON): {len(sedac_entries)}, "
            f"Supplementary unlabelled/L2-tagged EO collections (with UMM-JSON): {len(supp_entries)}"
        ),
    )


if __name__ == "__main__":
    main()
