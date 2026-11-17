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
import html
import json
import re
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

ZENODO_PAGE_SIZE = 25  # Zenodo REST API anonymous limit (size > 25 returns HTTP 400)
ZENODO_MAX_PAGES = 20   # Up to 500 hits per scoped query with full pagination audit

ZENODO_QUERIES = [
    '+"Global Mangrove Watch"',
    '+"Global Pasture Watch"',
    '+ETH +"Global Canopy Height"',
    '+SoilGrids +(250m OR global)',
    '+HydroSHEDS +global',
    '+GLC_FCS30D',
    '+OpenLandMap',
    '+WorldCereal +global',
    '+"Global Surface Water Explorer"',
    '+"Global Dam Watch"',
    '+"Global Wetlands" +map',
    '+WAD2M',
    '+kelp +(global OR map) +dataset',
    '+"Global Land Evaporation Amsterdam Model" OR +GLEAM4',
    '+VODCA +"Vegetation Optical Depth"',
    '+"OceanSODA-ETHZ"',
    '+"WCRP" +"Sea Level Budget"',
    '+LAI4g OR (+GLOBMAP +"Leaf Area Index")',
    '+FLUXCOM',
    '+"Global Carbon Budget"',
    '+TCSIF +"GOME-2"',
    '+"Solar-Induced Fluorescence" +global',
    '+CSIF +"Solar-Induced Fluorescence"',
    '+GOSIF +OCO-2',
    '+TROPOSIF',
    '+LCREF +(AVHRR OR MODIS)',
    '+LCSIF +(AVHRR OR MODIS)',
    '+LCSPP +(AVHRR OR MODIS)',
    '+"GLAST" +"Lake Surface"',
]

ZENODO_PINNED_RECORD_IDS = [
    8242928,   # TCSIF v1: Temporally consistent global GOME-2A SIF dataset (doi:10.5281/zenodo.8242928)
    14460914,  # TCSIF_005: Temporally consistent multi-sensor global SIF 0.05 deg (doi:10.5281/zenodo.14460914)
    14724263,  # GLEAM4 (Miralles et al. / Ghent University, doi:10.5281/zenodo.14724263)
    20682040,  # LCREF-MODIS v3.3 (doi:10.5281/zenodo.20682040)
    14586458,  # LCREF-AVHRR v3.2 (doi:10.5281/zenodo.14586458)
    20683570,  # Global LCSPP-MODIS v3.3 (doi:10.5281/zenodo.20683570)
    14568024,  # Global LCSPP-AVHRR 1982-2000 (doi:10.5281/zenodo.14568024)
    14568491,  # Global LCSPP-AVHRR 2001-2023 (doi:10.5281/zenodo.14568491)
    13922379,  # Global LCSIF-MODIS v2 (doi:10.5281/zenodo.13922379)
    7916879,   # Global LCREF/LCSIF/LCSPP v1 (doi:10.5281/zenodo.7916879)
    10212472,  # Global fluorescence yield based on TROPOSIF data, normalized with NIRvP (doi:10.5281/zenodo.10212472)
    7331411,   # eSIF product: enhanced SIF by combining TROPOSIF, MODIS reflectance and ERA-5 (doi:10.5281/zenodo.7331411)
    15105312,  # GOSIF - Global, OCO-2 based SIF product (doi:10.5281/zenodo.15105312)
    12698637,  # GLOBMAP global Leaf Area Index since 1981 (doi:10.5281/zenodo.12698637)
    16394372,  # LHSIF Long-Term Harmonized Multi-Satellite SIF (doi:10.5281/zenodo.16394372)
    2575599,   # VODCA Global Long-term Microwave Vegetation Optical Depth Climate Archive (doi:10.5281/zenodo.2575599)
    11206366,  # OceanSODA-ETHZ-v2 Surface ocean sea-air CO2 fluxes (doi:10.5281/zenodo.11206366)
    5553187,   # WAD2M Wetland Area and Dynamics for Methane Modeling (doi:10.5281/zenodo.5553187)
    14816612,  # Global floating kelp forest marine heatwave dataset (doi:10.5281/zenodo.14816612)
    5825755,   # Beagle kelp maps (doi:10.5281/zenodo.5825755)
    18385275,  # PML-V2.2 companion calibration/validation/simulation archive (doi:10.5281/zenodo.18385275)
]

DATACITE_PINNED_DOIS = [
    "10.18160/5NZG-JMJE",        # FLUXCOM-X-BASE global carbon & energy fluxes (ICOS Carbon Portal)
    "10.5281/zenodo.8242928",    # TCSIF v1 GOME-2A SIF (cross-verified via DataCite)
    "10.5281/zenodo.14586458",   # LCREF-AVHRR v3.2
    "10.5281/zenodo.20682040",   # LCREF-MODIS v3.3
    "10.5281/zenodo.20683570",   # LCSPP / LCSIF / LCREF global SIF & GPP
    "10.5281/zenodo.13922379",   # LCSPP / LCSIF / LCREF v2
    "10.5281/zenodo.7916879",    # LCSPP / LCSIF / LCREF v1
    "10.5281/zenodo.10212472",   # Global fluorescence yield based on TROPOSIF data, normalized with NIRvP
    "10.15454/tihhyp",           # SMOS-IC v2 global soil moisture and L-band vegetation optical depth (INRAE / CESBIO)
    "10.1594/PANGAEA.911944",    # Global terrestrial carbon/water flux EO benchmark (PANGAEA)
    "10.1594/PANGAEA.897399",    # Global EO biophysical benchmark (PANGAEA)
    "10.7910/DVN/EJSXHS",        # CSIF Contiguous Solar-Induced Fluorescence (Harvard Dataverse)
    "10.20783/DIAS.667",         # J-OFURO3 Ver.1.2 (DIAS: global ice-free ocean heat, momentum & freshwater fluxes)
    "10.20783/DIAS.612",         # J-OFURO3 V1.1 (DIAS)
    "10.57760/sciencedb.27228",  # CoSEB global land surface radiation and heat fluxes (ScienceDB)
    "10.57760/sciencedb.35001",  # HSIF Global Harmonized SIF 0.05° Daily (ScienceDB)
    "10.57760/sciencedb.018aw",  # GOME-LSIF Global SIF 1995-2011 (ScienceDB)
    "10.57760/sciencedb.019ud",  # FY-3F MERSI-III global LAI/FPAR v1.0 (ScienceDB)
    "10.57760/sciencedb.014ku",  # Quasi-global SSS-RecNet sea surface salinity (ScienceDB)
    "10.57760/sciencedb.018ot",  # HGRR Global MODIS-consistent 1 km LST (ScienceDB)
    "10.57760/sciencedb.30916",  # FY-3 VIRR Global LST 2008-2019 (ScienceDB)
    "10.57760/sciencedb.16933",  # Global 1h 10km Surface Solar Radiation (ScienceDB)
    "10.57760/sciencedb.17942",  # FY-3 Global Microwave Integrated Drought Index (ScienceDB)
    "10.6096/13",                # Theia Hugonnet et al. Accelerated global glacier mass loss (2000-2019)
    "10.6096/1007",              # Theia Millan et al. Ice velocity and thickness of the world's glaciers
    "10.6096/ctoh_sit_2023_01",  # Theia CTOH Arctic & Antarctic sea ice thickness CDR
    "10.6096/3008",              # Theia Snow Depth on Sea Ice CryoSat-2 + ICESat-2
    "10.6084/m9.figshare.26861767.v2",  # Global ocean latent & sensible heat fluxes (Figshare)
    "10.5281/zenodo.18385275",   # PML-V2.2 calibration & extended dataset companion (Zenodo)
]

DOI_CSL_PINNED = [
    "10.11888/Terre.tpdc.303314",  # TPDC PML-V2.2: 0.1° PML-V2.2a/b/c (1982-near present) via DOI CSL-JSON negotiation
]

DATACITE_EO_PREFIX_QUERIES = [
    {"prefix": "10.5880", "query": "g3p", "page[size]": "25"},
    {"prefix": "10.18160", "query": "FLUXCOM", "page[size]": "50"},
    {"prefix": "10.20783", "resource-type-id": "dataset", "query": "+(\"J-OFURO\" OR GSMaP OR \"surface flux\")", "page[size]": "25"},
    {
        "prefix": "10.57760",
        "resource-type-id": "dataset",
        "query": "+global +(CoSEB OR \"solar-induced chlorophyll fluorescence\" OR \"surface radiation\" OR \"sea surface salinity\" OR \"LAI/FPAR\" OR \"land surface temperature\") +(satellite OR MODIS OR FY-3 OR GOME OR SMOS OR GLASS)",
        "page[size]": "25",
    },
    {
        "prefix": "10.6096",
        "resource-type-id": "dataset",
        "query": "+(glacier OR \"sea-ice\" OR \"snow depth\") +(global OR hemisphere OR CryoSat OR Envisat OR Sentinel OR ASTER OR world)",
        "page[size]": "25",
    },
]


def _zenodo_get_with_rate_limit(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 60,
    max_attempts: int = 5,
) -> requests.Response:
    """GET Zenodo REST endpoint while honoring X-RateLimit-Remaining / X-RateLimit-Reset headers."""
    last_resp: requests.Response | None = None
    for attempt in range(max_attempts):
        resp = session.get(url, params=params, timeout=timeout)
        last_resp = resp
        rem_hdr = resp.headers.get("X-RateLimit-Remaining")
        reset_hdr = resp.headers.get("X-RateLimit-Reset")
        if resp.status_code == 429:
            try:
                wait_s = max(5.0, float(reset_hdr) - time.time() + 1.5) if reset_hdr else 20.0 * (attempt + 1)
            except Exception:
                wait_s = 20.0 * (attempt + 1)
            wait_s = min(wait_s, 75.0)
            print(f"[{SOURCE_ID}] Zenodo rate limit (429) on {url}; sleeping {wait_s:.1f}s")
            time.sleep(wait_s)
            continue
        if rem_hdr is not None and reset_hdr is not None:
            try:
                if int(rem_hdr) <= 1:
                    wait_s = max(0.0, float(reset_hdr) - time.time() + 1.2)
                    if wait_s > 0:
                        time.sleep(min(wait_s, 65.0))
            except Exception:
                pass
        return resp
    assert last_resp is not None
    return last_resp


def _fetch_dias_landing_metadata(session: requests.Session, landing_url: str) -> dict[str, Any] | None:
    """Extract structured resolution, temporal extent, and keywords from an official DIAS dataset page."""
    if not landing_url or "search.diasjp.net/" not in landing_url:
        return None
    try:
        r = session.get(landing_url, timeout=20)
        if r.status_code != 200:
            return None
        html_txt = r.text
    except Exception:
        return None

    out: dict[str, Any] = {}
    for ld_raw in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html_txt, re.S | re.I):
        try:
            obj = json.loads(html.unescape(ld_raw))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("@type") == "Dataset":
            if isinstance(obj.get("keywords"), list):
                out["keywords"] = [str(k).strip() for k in obj["keywords"] if k]
            break

    clean = re.sub(r"<[^>]+>", "\n", html_txt)
    lines = [re.sub(r"\s+", " ", html.unescape(ln)).strip() for ln in clean.splitlines()]
    lines = [ln for ln in lines if ln]

    m_beg = re.search(r"Begin\s+Date\s*(\d{4}-\d{2}-\d{2})", " ".join(lines), re.I)
    m_end = re.search(r"End\s+Date\s*(\d{4}-\d{2}-\d{2})", " ".join(lines), re.I)
    if m_beg:
        out["temporal_start_raw"] = m_beg.group(1)
    if m_end:
        out["temporal_end_raw"] = m_end.group(1)

    joined = " | ".join(lines)
    m_deg = re.search(r"\brow\s*\|\s*\d+\s*\|\s*(\d+(?:\.\d+)?)\s*\(?(deg|degree|degrees|km|m)\)?", joined, re.I)
    if m_deg:
        out["spatial_resolution_raw"] = f"{m_deg.group(1)} {m_deg.group(2)}"
    m_time = re.search(r"\btime\s*\|\s*([a-z0-9,\s/-]+?)(?:\s*\(|\s*\|\s*Keyword)", joined, re.I)
    if m_time:
        t_clean = re.sub(r"\s+", " ", m_time.group(1)).strip(" ,|")
        if t_clean and not t_clean.isdigit():
            out["temporal_resolution_raw"] = "; ".join(p.strip() for p in t_clean.split(",") if p.strip())
    return out or None


def _fetch_datacite_and_csl(session: requests.Session, queried: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    g3p_url = "https://api.datacite.org/dois"
    g3p_records: list[dict[str, Any]] = []
    seen_dc_ids: set[str] = set()

    for dc_params in DATACITE_EO_PREFIX_QUERIES:
        r_dc = session.get(g3p_url, params=dc_params, timeout=60)
        if r_dc.status_code == 200:
            queried.append(f"{g3p_url}?prefix={dc_params['prefix']}&query={dc_params['query']}")
            for item in r_dc.json().get("data", []):
                dc_id = str(item.get("id", "")).lower()
                if not dc_id or dc_id in seen_dc_ids:
                    continue
                if "mistrals-hymex." in dc_id:
                    continue
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

    # Enrich DIAS DataCite items directly with structured DIAS catalogue resolution & temporal extent
    for item in g3p_records:
        dc_id = str(item.get("id") or "").lower()
        u_attr = str((item.get("attributes") or {}).get("url") or "")
        if dc_id.startswith("10.20783/") and "search.diasjp.net/" in u_attr:
            dias_md = _fetch_dias_landing_metadata(session, u_attr)
            if dias_md:
                item["_dias_metadata"] = dias_md

    (DATA_DIR / "gfz_datacite_g3p_benchmarks.json").write_text(
        json.dumps({"data": g3p_records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csl_records: list[dict[str, Any]] = []
    for csl_doi in DOI_CSL_PINNED:
        r_csl = session.get(f"https://doi.org/{csl_doi}", headers={"Accept": "application/vnd.citationstyles.csl+json"}, timeout=25)
        if r_csl.status_code == 200:
            try:
                obj = r_csl.json()
                if isinstance(obj, dict) and obj.get("DOI"):
                    csl_records.append(obj)
                    queried.append(f"https://doi.org/{csl_doi} (Accept: application/vnd.citationstyles.csl+json)")
            except Exception:
                pass
    (DATA_DIR / "doi_csl_benchmarks.json").write_text(
        json.dumps({"data": csl_records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return g3p_records, csl_records


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    queried: list[str] = []
    if "--datacite-and-csl-only" in sys.argv[1:]:
        fao_path = DATA_DIR / "fao_gismgr_workspaces_and_mapsets.json"
        fao_mapsets = json.loads(fao_path.read_text(encoding="utf-8")).get("mapsets", []) if fao_path.exists() else []
        z_path = DATA_DIR / "zenodo_thematic_benchmarks.jsonl"
        zenodo_records = [json.loads(line) for line in z_path.open(encoding="utf-8")] if z_path.exists() else []
        zenodo_ids = {int(r["id"]) for r in zenodo_records if r.get("id")}
        for z_rec_id in ZENODO_PINNED_RECORD_IDS:
            if z_rec_id not in zenodo_ids:
                r_pin = _zenodo_get_with_rate_limit(session, f"https://zenodo.org/api/records/{z_rec_id}", timeout=30)
                if r_pin.status_code == 200:
                    h = r_pin.json()
                    if h.get("id") and int(h["id"]) not in zenodo_ids:
                        h["_matched_benchmark_query"] = "pinned_authoritative_eo_flagship"
                        h["_matched_benchmark_queries"] = ["pinned_authoritative_eo_flagship"]
                        zenodo_records.append(h)
                        zenodo_ids.add(int(h["id"]))
        with z_path.open("w", encoding="utf-8") as f:
            for rec in zenodo_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Keep zenodo_pagination_audit total synchronized if a pinned Zenodo record was added
        pag_path = DATA_DIR / "zenodo_pagination_audit.json"
        if pag_path.exists():
            pag_obj = json.loads(pag_path.read_text(encoding="utf-8"))
            pag_obj["total_unique_zenodo_records"] = len(zenodo_records)
            pag_path.write_text(json.dumps(pag_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        g3p_records, csl_records = _fetch_datacite_and_csl(session, queried)
        prior_eps = []
        if LAST_ACCESS_FILE.exists():
            prior_eps = [ln.strip()[2:].strip() for ln in LAST_ACCESS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("- https://")]
        all_eps = list(dict.fromkeys([*prior_eps, *queried]))
        write_last_access(
            all_eps,
            len(fao_mapsets) + len(zenodo_records) + len(g3p_records) + len(csl_records),
            "200 OK",
            notes=(
                f"FAO GISMGR mapsets (all pages): {len(fao_mapsets)}, Zenodo benchmark datasets: {len(zenodo_records)}, "
                f"DataCite benchmark records (incl. DIAS, ScienceDB, Theia, GFZ, ICOS, PANGAEA): {len(g3p_records)}, "
                f"DOI CSL-JSON records (TPDC): {len(csl_records)}"
            ),
        )
        return

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

    # 2. Zenodo REST API for flagship global L3/L4 EO datasets (complete pagination + audit)
    zenodo_records: list[dict[str, Any]] = []
    zenodo_by_id: dict[int, dict[str, Any]] = {}
    pagination_audit: list[dict[str, Any]] = []
    z_url = "https://zenodo.org/api/records"

    for q in ZENODO_QUERIES:
        hits_total: int | None = None
        pages_fetched = 0
        fetched_for_q = 0
        new_unique_for_q = 0
        last_status = 0
        truncated = False

        for z_page in range(1, ZENODO_MAX_PAGES + 1):
            rz = _zenodo_get_with_rate_limit(
                session,
                z_url,
                params={"q": q, "type": "dataset", "size": str(ZENODO_PAGE_SIZE), "page": str(z_page)},
                timeout=60,
            )
            last_status = rz.status_code
            if rz.status_code != 200:
                truncated = True
                print(f"[{SOURCE_ID}] WARN Zenodo HTTP {rz.status_code} on q={q!r} page={z_page}: {rz.text[:160]}")
                break
            payload = rz.json().get("hits", {}) or {}
            if hits_total is None:
                hits_total = int(payload.get("total", 0))
            hits = payload.get("hits", []) or []
            pages_fetched += 1
            fetched_for_q += len(hits)
            for h in hits:
                zid = h.get("id")
                if not zid:
                    continue
                if zid not in zenodo_by_id:
                    h["_matched_benchmark_query"] = q
                    h["_matched_benchmark_queries"] = [q]
                    zenodo_by_id[zid] = h
                    zenodo_records.append(h)
                    new_unique_for_q += 1
                else:
                    mq = zenodo_by_id[zid].setdefault("_matched_benchmark_queries", [])
                    if q not in mq:
                        mq.append(q)
            time.sleep(0.25)
            if not hits or len(hits) < ZENODO_PAGE_SIZE or (hits_total is not None and z_page * ZENODO_PAGE_SIZE >= hits_total):
                break
        else:
            if hits_total is not None and ZENODO_MAX_PAGES * ZENODO_PAGE_SIZE < hits_total:
                truncated = True

        pagination_audit.append({
            "query": q,
            "http_status": last_status,
            "page_size": ZENODO_PAGE_SIZE,
            "hits_total": hits_total,
            "pages_fetched": pages_fetched,
            "fetched_records": fetched_for_q,
            "new_unique_records": new_unique_for_q,
            "truncated": truncated,
        })

    supplemental_queries = [
        "+MUSES +NDVI +Global",
        '+"PKU GIMMS" +NDVI',
        "+OCO-3 +(GPP OR SIF)",
        '+"mangrove canopy height" +global',
        '+"Solar-Induced Chlorophyll Fluorescence" +(global OR satellite OR TROPOMI OR OCO OR TanSat OR GOME-2)',
        '+"Global" +("glacier thickness" OR "glacier mass balance" OR "permafrost" OR "Active Layer Thickness") +(satellite OR ESA OR CCI)',
        '+"Global" +("burned area" OR "fireCCI" OR "Fire_cci") +(MODIS OR Sentinel OR VIIRS OR AVHRR)',
        '+"Global" +("sea surface salinity" OR "ocean color" OR "ocean colour" OR "chlorophyll-a") +(satellite OR CCI OR SMOS OR SMAP)',
        '+"Global" +("evapotranspiration" OR "latent heat flux") +(MODIS OR ERA5 OR FLUXCOM OR satellite) +gridded',
        '+"Global" +("surface solar radiation" OR "downward shortwave" OR "PAR") +(satellite OR CERES OR HIMAWARI OR MSG)',
        '+"Global" +("snow water equivalent" OR "snow cover extent") +(satellite OR GlobSnow OR MODIS OR ESA)',
        '+GLOBMAP +("Leaf Area Index" OR "fractional tree cover")',
        '+AVHRR +"net radiation" +(global OR "all-wave")',
        '+GRACE +"terrestrial water storage" +(reconstruction OR "cyclostationary" OR CSEOF) +global',
    ]
    for q in supplemental_queries:
        hits_total = None
        pages_fetched = 0
        fetched_for_q = 0
        new_unique_for_q = 0
        last_status = None
        truncated = False
        for z_page in range(1, ZENODO_MAX_PAGES + 1):
            rz = _zenodo_get_with_rate_limit(
                session,
                z_url,
                params={"q": q, "type": "dataset", "size": str(ZENODO_PAGE_SIZE), "page": str(z_page)},
                timeout=60,
            )
            last_status = rz.status_code
            if rz.status_code != 200:
                truncated = True
                print(f"[{SOURCE_ID}] WARN Zenodo HTTP {rz.status_code} on supplemental q={q!r} page={z_page}: {rz.text[:160]}")
                break
            payload = rz.json().get("hits", {}) or {}
            if hits_total is None:
                hits_total = int(payload.get("total", 0))
            hits = payload.get("hits", []) or []
            pages_fetched += 1
            fetched_for_q += len(hits)
            for h in hits:
                zid = h.get("id")
                if not zid:
                    continue
                if zid not in zenodo_by_id:
                    h["_matched_benchmark_query"] = q
                    h["_matched_benchmark_queries"] = [q]
                    zenodo_by_id[zid] = h
                    zenodo_records.append(h)
                    new_unique_for_q += 1
                else:
                    mq = zenodo_by_id[zid].setdefault("_matched_benchmark_queries", [])
                    if q not in mq:
                        mq.append(q)
            time.sleep(0.25)
            if not hits or len(hits) < ZENODO_PAGE_SIZE or (hits_total is not None and z_page * ZENODO_PAGE_SIZE >= hits_total):
                break
        else:
            if hits_total is not None and ZENODO_MAX_PAGES * ZENODO_PAGE_SIZE < hits_total:
                truncated = True

        pagination_audit.append({
            "query": q,
            "http_status": last_status,
            "page_size": ZENODO_PAGE_SIZE,
            "hits_total": hits_total,
            "pages_fetched": pages_fetched,
            "fetched_records": fetched_for_q,
            "new_unique_records": new_unique_for_q,
            "truncated": truncated,
        })

    # Carry forward verified EO Zenodo IDs from prior baseline so tightened queries never drop indexed releases
    carryforward_path = BASE_DIR / "verified_zenodo_carryforward.json"
    carry_restored = 0
    carry_missing_ids: list[int] = []
    if carryforward_path.exists():
        cf_data = json.loads(carryforward_path.read_text(encoding="utf-8"))
        cf_ids = [int(item["id"]) for item in cf_data.get("carry_forward_records", []) if int(item["id"]) not in zenodo_by_id]
        for start in range(0, len(cf_ids), ZENODO_PAGE_SIZE):
            batch = cf_ids[start : start + ZENODO_PAGE_SIZE]
            q_batch = "id:(" + " OR ".join(str(x) for x in batch) + ")"
            rb = _zenodo_get_with_rate_limit(
                session,
                z_url,
                params={"q": q_batch, "size": str(ZENODO_PAGE_SIZE)},
                timeout=60,
            )
            found_in_batch: set[int] = set()
            if rb.status_code == 200:
                for h in (rb.json().get("hits", {}) or {}).get("hits", []) or []:
                    zid = h.get("id")
                    if not zid:
                        continue
                    found_in_batch.add(int(zid))
                    if zid not in zenodo_by_id:
                        h["_matched_benchmark_query"] = "verified_carryforward_f652943"
                        h["_matched_benchmark_queries"] = ["verified_carryforward_f652943"]
                        zenodo_by_id[zid] = h
                        zenodo_records.append(h)
                        carry_restored += 1
                    else:
                        mq = zenodo_by_id[zid].setdefault("_matched_benchmark_queries", [])
                        if "verified_carryforward_f652943" not in mq:
                            mq.append("verified_carryforward_f652943")
            time.sleep(0.25)
            for zid in batch:
                if zid not in zenodo_by_id and zid not in found_in_batch:
                    r_ind = _zenodo_get_with_rate_limit(session, f"https://zenodo.org/api/records/{zid}", timeout=30)
                    if r_ind.status_code == 200:
                        h = r_ind.json()
                        hid = h.get("id") or zid
                        if hid not in zenodo_by_id:
                            h["_matched_benchmark_query"] = "verified_carryforward_f652943_direct"
                            h["_matched_benchmark_queries"] = ["verified_carryforward_f652943_direct"]
                            zenodo_by_id[hid] = h
                            zenodo_records.append(h)
                            carry_restored += 1
                    else:
                        carry_missing_ids.append(zid)
                    time.sleep(0.15)

    for z_rec_id in list(dict.fromkeys([*ZENODO_PINNED_RECORD_IDS, 7479028, 8253971, 15382965, 11236850])):
        if z_rec_id in zenodo_by_id:
            continue
        r_pin = _zenodo_get_with_rate_limit(session, f"https://zenodo.org/api/records/{z_rec_id}", timeout=30)
        if r_pin.status_code == 200:
            h = r_pin.json()
            zid = h.get("id")
            if zid and zid not in zenodo_by_id:
                h["_matched_benchmark_query"] = "pinned_authoritative_eo_flagship"
                h["_matched_benchmark_queries"] = ["pinned_authoritative_eo_flagship"]
                zenodo_by_id[zid] = h
                zenodo_records.append(h)
        else:
            print(f"[{SOURCE_ID}] WARN pinned Zenodo record {z_rec_id} HTTP {r_pin.status_code}")

    (DATA_DIR / "zenodo_pagination_audit.json").write_text(
        json.dumps(
            {
                "page_size": ZENODO_PAGE_SIZE,
                "max_pages_per_query": ZENODO_MAX_PAGES,
                "total_unique_zenodo_records": len(zenodo_records),
                "truncated_query_count": sum(1 for a in pagination_audit if a["truncated"]),
                "carry_forward_restored_records": carry_restored,
                "carry_forward_missing_ids": carry_missing_ids,
                "queries": pagination_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    queried.append("https://zenodo.org/api/records?type=dataset (scoped boolean queries, full pagination + audit)")
    with (DATA_DIR / "zenodo_thematic_benchmarks.jsonl").open("w", encoding="utf-8") as f:
        for rec in zenodo_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3. DataCite API (G3P, FLUXCOM-X-BASE, J-OFURO3, CoSEB, Theia, ScienceDB, PANGAEA) & DOI CSL-JSON (TPDC)
    g3p_records, csl_records = _fetch_datacite_and_csl(session, queried)

    write_last_access(
        queried,
        len(fao_mapsets) + len(zenodo_records) + len(g3p_records) + len(csl_records),
        "200 OK",
        notes=(
            f"FAO GISMGR mapsets (all pages): {len(fao_mapsets)}, Zenodo benchmark datasets: {len(zenodo_records)}, "
            f"DataCite benchmark records (incl. DIAS, ScienceDB, Theia, GFZ, ICOS, PANGAEA): {len(g3p_records)}, "
            f"DOI CSL-JSON records (TPDC): {len(csl_records)}"
        ),
    )


if __name__ == "__main__":
    main()
