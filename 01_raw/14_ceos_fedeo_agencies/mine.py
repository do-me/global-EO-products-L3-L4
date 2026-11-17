#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
# ]
# ///
"""
 Raw Metadata Miner: CEOS FedEO Gateway — ESA CCI (CEDA), ESA Earth Online, ESA MAAP, VITO Terrascope, JAXA, ISRO, DLR, INPE, NRCan & USGS
Source ID: 14_ceos_fedeo_agencies
Authoritative Upstream API: GET https://fedeo.ceos.org/collections?parentIdentifier=<agency> + GET https://api.stac.ceda.ac.uk/collections

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

SOURCE_ID = "14_ceos_fedeo_agencies"
SOURCE_TITLE = "CEOS FedEO Gateway — ESA CCI (CEDA), ESA Earth Online, ESA MAAP, VITO Terrascope, JAXA, ISRO, DLR, INPE, NRCan & USGS"


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

AGENCIES = [
    ("ceda_cci", "EOP:STFC:CEDA-CCI"),
    ("esa_earth_online", "EOP:ESA:EARTH-ONLINE"),
    ("esa_maap", "EOP:ESA:MAAP"),
    ("vito_terrascope", "EOP:VITO:TERRASCOPE"),
    ("jaxa_gportal", "EOP:JAXA:G-PORTAL"),
    ("isro_mosdac", "EOP:ISRO:MOSDAC"),
    ("isro_bhoonidhi", "EOP:NRSC:BHOONIDHI"),
    ("dlr_eoc", "EOP:DLR:EOC"),
    ("dlr_geoservice", "EOP:DLR:GEOSERVICE"),
    ("inpe_data", "EOP:INPE:DATA"),
    ("nrcan_eodms", "EOP:NRCAN:EODMS"),
    ("usgs_landsatlook", "EOP:USGS:LANDSATLOOK"),
]


def fetch_agency(session: requests.Session, slug: str, parent_id: str) -> tuple[str, list[dict[str, Any]]]:
    cols: list[dict[str, Any]] = []
    start = 1
    page_size = 100
    while True:
        url = (
            f"https://fedeo.ceos.org/collections?parentIdentifier={parent_id}"
            f"&httpAccept=application/json&maximumRecords={page_size}&startRecord={start}"
        )
        r = session.get(url, timeout=60)
        if r.status_code != 200:
            break
        batch = r.json().get("collections", [])
        if not batch:
            break
        cols.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return slug, cols


def resolve_ceda_datacite_crosswalk(
    session: requests.Session,
    ceda_collections: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Resolve primary CEDA dataset DOIs (`10.5285/<uuid>`) directly via the DataCite REST API
    instead of falling back to STAC `describedby` links (which frequently point to input L1/L2
    products or methodology journal articles rather than the dataset itself).
    """
    uuids = [
        str(c.get("id")).strip()
        for c in ceda_collections
        if isinstance(c, dict) and c.get("id") and len(str(c.get("id")).strip()) == 32
    ]
    uuids = list(dict.fromkeys(uuids))
    by_uuid: dict[str, dict[str, Any]] = {}
    dc_api = "https://api.datacite.org/dois"
    for i in range(0, len(uuids), 40):
        batch = uuids[i : i + 40]
        q = "doi:(" + " OR ".join(f"10.5285/{u}" for u in batch) + ")"
        r = session.get(dc_api, params={"query": q, "page[size]": 100}, timeout=30)
        if r.status_code != 200:
            continue
        for item in r.json().get("data", []):
            did = str(item.get("id") or "").strip().lower()
            if not did.startswith("10.5285/"):
                continue
            uid = did.split("10.5285/", 1)[1]
            attr = item.get("attributes", {}) or {}
            titles = [t.get("title") for t in (attr.get("titles") or []) if isinstance(t, dict) and t.get("title")]
            by_uuid[uid] = {
                "doi": did,
                "url": attr.get("url") or f"https://catalogue.ceda.ac.uk/uuid/{uid}/",
                "publisher": attr.get("publisher"),
                "publicationYear": attr.get("publicationYear"),
                "title": titles[0] if titles else None,
            }
    return {
        "source": "https://api.datacite.org/dois?query=doi:(10.5285/<uuid>)",
        "queried_uuid_count": len(uuids),
        "matched_count": len(by_uuid),
        "by_uuid": by_uuid,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session(pool_maxsize=16)
    crosswalk_only = "--crosswalk-only" in sys.argv[1:]

    if crosswalk_only:
        ceda_path = DATA_DIR / "fedeo_ceda_cci.jsonl"
        ceda_cols = [json.loads(line) for line in ceda_path.open(encoding="utf-8")] if ceda_path.exists() else []
        cw = resolve_ceda_datacite_crosswalk(session, ceda_cols)
        (DATA_DIR / "ceda_datacite_doi_crosswalk.json").write_text(
            json.dumps(cw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Preserve prior queried endpoint counts & append DataCite crosswalk cleanly
        prior_lines = LAST_ACCESS_FILE.read_text(encoding="utf-8").splitlines() if LAST_ACCESS_FILE.exists() else []
        prior_eps = [ln.strip()[2:].strip() for ln in prior_lines if ln.strip().startswith("- https://")]
        prior_eps = [ep for ep in prior_eps if "api.datacite.org/dois" not in ep]
        prior_eps.append(f"https://api.datacite.org/dois?prefix=10.5285 ({cw['matched_count']}/{cw['queried_uuid_count']} CEDA UUIDs)")
        total = 0
        for p in DATA_DIR.glob("fedeo_*.jsonl"):
            total += sum(1 for _ in p.open(encoding="utf-8"))
        if (DATA_DIR / "ceda_native_stac.json").exists():
            total += len(json.loads((DATA_DIR / "ceda_native_stac.json").read_text(encoding="utf-8")).get("collections", []))
        if (DATA_DIR / "esa_maap_stac.json").exists():
            total += len(json.loads((DATA_DIR / "esa_maap_stac.json").read_text(encoding="utf-8")).get("collections", []))
        write_last_access(sorted(dict.fromkeys(prior_eps)), total, "200 OK", notes="Added CEDA DataCite 10.5285/<uuid> primary DOI crosswalk.")
        return

    total = 0
    queried: list[str] = []

    ceda_cols_for_cw: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_agency, session, slug, pid): (slug, pid) for slug, pid in AGENCIES}
        for fut in as_completed(futs):
            slug, pid = futs[fut]
            _, cols = fut.result()
            if slug == "ceda_cci":
                ceda_cols_for_cw = cols
            total += len(cols)
            queried.append(f"https://fedeo.ceos.org/collections?parentIdentifier={pid} ({len(cols)} collections)")
            out_path = DATA_DIR / f"fedeo_{slug}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for c in cols:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if ceda_cols_for_cw:
        cw = resolve_ceda_datacite_crosswalk(session, ceda_cols_for_cw)
        (DATA_DIR / "ceda_datacite_doi_crosswalk.json").write_text(
            json.dumps(cw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        queried.append(f"https://api.datacite.org/dois?prefix=10.5285 ({cw['matched_count']}/{cw['queried_uuid_count']} CEDA UUIDs)")

    # Also fetch native UK CEDA STAC API
    ceda_url = "https://api.stac.ceda.ac.uk/collections"
    r_ceda = session.get(ceda_url, timeout=60)
    if r_ceda.status_code == 200:
        ceda_cols = r_ceda.json().get("collections", [])
        total += len(ceda_cols)
        queried.append(f"{ceda_url} ({len(ceda_cols)} collections)")
        (DATA_DIR / "ceda_native_stac.json").write_text(
            json.dumps(r_ceda.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Also fetch native ESA MAAP STAC API (full collection catalogue beyond FedEO subsets)
    maap_base = "https://catalog.maap.eo.esa.int/catalogue/collections?limit=100"
    maap_cols: list[dict[str, Any]] = []
    maap_seen: set[str] = set()
    nxt_url: str | None = maap_base
    maap_pages = 0
    while nxt_url and maap_pages < 25:
        r_maap = session.get(nxt_url, timeout=45)
        if r_maap.status_code != 200:
            break
        d_maap = r_maap.json()
        batch = d_maap.get("collections", [])
        if not batch:
            break
        for c in batch:
            cid = c.get("id")
            if cid and cid not in maap_seen:
                maap_seen.add(cid)
                maap_cols.append(c)
        nxt_url = next(
            (lk.get("href") for lk in d_maap.get("links", []) if lk.get("rel") == "next" and lk.get("href")),
            None,
        )
        maap_pages += 1
    if maap_cols:
        total += len(maap_cols)
        queried.append(f"{maap_base} ({len(maap_cols)} collections)")
        (DATA_DIR / "esa_maap_stac.json").write_text(
            json.dumps({"collections": maap_cols}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_last_access(sorted(queried), total, "200 OK")


if __name__ == "__main__":
    main()
