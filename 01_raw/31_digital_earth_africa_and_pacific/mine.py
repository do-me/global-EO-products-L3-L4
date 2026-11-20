#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "urllib3>=2.0.0",
#   "pyyaml>=6.0.1",
# ]
# ///
"""
Raw Metadata Miner: Operational Continental & Regional EO Hubs — Digital Earth Africa & Digital Earth Pacific
Source ID: 31_digital_earth_africa_and_pacific
Authoritative Upstream APIs:
  - Digital Earth Africa ODC Product Registry: GET https://explorer.digitalearth.africa/products.txt
    + GET https://explorer.digitalearth.africa/products/<product_name>.odc-product.yaml
  - Digital Earth Pacific STAC API: GET https://stac.digitalearthpacific.org/collections (paginated)

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
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LAST_ACCESS_FILE = BASE_DIR / "last_access.txt"

SOURCE_ID = "31_digital_earth_africa_and_pacific"
SOURCE_TITLE = "Operational Continental & Regional EO Hubs — Digital Earth Africa (ODC Registry) & Digital Earth Pacific (STAC)"

DE_AFRICA_LIST_URL = "https://explorer.digitalearth.africa/products.txt"
DE_AFRICA_YAML_TMPL = "https://explorer.digitalearth.africa/products/{pid}.odc-product.yaml"
DE_AFRICA_WEB_TMPL = "https://explorer.digitalearth.africa/products/{pid}"
DE_PACIFIC_STAC_URL = "https://stac.digitalearthpacific.org/collections"


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
        "User-Agent": "Global-EO-L3-L4-Catalogue/1.0 (https://github.com/do-me/global-EO-products-L3-L4; Reproducible Metadata Miner)",
        "Accept": "application/json, text/yaml, text/plain, */*",
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


def fetch_de_africa_product(session: requests.Session, pid: str) -> tuple[str, dict[str, Any]]:
    yaml_url = DE_AFRICA_YAML_TMPL.format(pid=pid)
    resp = session.get(yaml_url, timeout=45)
    resp.raise_for_status()
    doc = yaml.safe_load(resp.content.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"Expected YAML mapping for {pid}")
    doc["_odc_product_id"] = pid
    doc["_yaml_url"] = yaml_url
    doc["_product_url"] = DE_AFRICA_WEB_TMPL.format(pid=pid)
    return pid, doc


def fetch_de_pacific_collections(session: requests.Session) -> list[dict[str, Any]]:
    collections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    url: str | None = DE_PACIFIC_STAC_URL
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
    return collections


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()

    # 1. Digital Earth Africa ODC Product Registry
    list_resp = session.get(DE_AFRICA_LIST_URL, timeout=45)
    list_resp.raise_for_status()
    product_ids = sorted({line.strip() for line in list_resp.content.decode("utf-8").splitlines() if line.strip()})
    print(f"[{SOURCE_ID}] Digital Earth Africa products listed: {len(product_ids)}")

    africa_rows: list[tuple[str, dict[str, Any]]] = []
    africa_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut_map = {ex.submit(fetch_de_africa_product, session, pid): pid for pid in product_ids}
        for fut in as_completed(fut_map):
            pid = fut_map[fut]
            try:
                africa_rows.append(fut.result())
            except Exception as exc:
                africa_errors.append(f"{pid}: {exc}")

    if africa_errors:
        raise RuntimeError(f"Failed fetching {len(africa_errors)} DE Africa YAMLs: {africa_errors[:5]}")

    africa_rows.sort(key=lambda x: x[0])
    africa_out = DATA_DIR / "digital_earth_africa_products.jsonl"
    with africa_out.open("w", encoding="utf-8") as f:
        for _, doc in africa_rows:
            f.write(json.dumps(doc, ensure_ascii=False, allow_nan=False, default=str) + "\n")

    # 2. Digital Earth Pacific STAC Collections
    pacific_cols = fetch_de_pacific_collections(session)
    print(f"[{SOURCE_ID}] Digital Earth Pacific STAC collections harvested: {len(pacific_cols)}")
    pacific_out = DATA_DIR / "digital_earth_pacific_stac.json"
    pacific_out.write_text(
        json.dumps({"collections": pacific_cols}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total = len(africa_rows) + len(pacific_cols)
    write_last_access(
        endpoints=[
            f"{DE_AFRICA_LIST_URL} + /products/<id>.odc-product.yaml ({len(africa_rows)} ODC product definitions)",
            f"{DE_PACIFIC_STAC_URL} ({len(pacific_cols)} STAC collections across paginated responses)",
        ],
        total_records=total,
        status_summary="200 OK",
        notes=(
            "DE Africa harvested via authoritative ODC product registry YAML endpoints because "
            "explorer.digitalearth.africa/stac/collections returned HTTP 200 HTML internal server errors; "
            "DE Pacific harvested via paginated STAC /collections API."
        ),
    )


if __name__ == "__main__":
    main()
