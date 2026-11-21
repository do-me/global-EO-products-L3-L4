#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
build_manifest.py
=================
Generates a cryptographic provenance manifest (`MANIFEST.json` and `SHA256SUMS.txt`) across all
raw (`01_raw/`), prepared (`02_preparation/`), harmonized (`03_harmonization/`),
internal QA (`04_validation_and_review/`), and external policy validation (`05_external_policy_validation/`)
repository artifacts.

Run via:
    uv run build_manifest.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_last_access(path: Path) -> dict[str, object]:
    info: dict[str, object] = {}
    endpoints: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("source_id: "):
            info["source_id"] = line.split(": ", 1)[1]
        elif line.startswith("source_title: "):
            info["source_title"] = line.split(": ", 1)[1]
        elif line.startswith("retrieved_at_utc: "):
            info["retrieved_at_utc"] = line.split(": ", 1)[1]
        elif line.startswith("http_status_summary: "):
            info["http_status_summary"] = line.split(": ", 1)[1]
        elif line.startswith("total_records_retrieved: "):
            info["total_records_retrieved"] = int(line.split(": ", 1)[1])
        elif line.startswith("  - ") and "path:" not in line:
            endpoints.append(line[4:])
    info["endpoints_queried"] = endpoints
    return info


def main() -> None:
    sources_meta = []
    for la in sorted((REPO_ROOT / "01_raw").glob("*/last_access.txt")):
        sources_meta.append(parse_last_access(la))

    tracked_files = []
    sha_lines = []
    for stage in ("01_raw", "02_preparation", "03_harmonization", "04_validation_and_review", "05_external_policy_validation"):
        stage_dir = REPO_ROOT / stage
        if not stage_dir.exists():
            continue
        for p in sorted(stage_dir.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts:
                rel = p.relative_to(REPO_ROOT).as_posix()
                digest = sha256_file(p)
                size_b = p.stat().st_size
                tracked_files.append({
                    "path": rel,
                    "stage": stage,
                    "bytes": size_b,
                    "sha256": digest,
                })
                sha_lines.append(f"{digest}  {rel}")

    manifest = {
        "catalogue_title": "Global Harmonized L3/L4 Earth Observation Product Catalogue",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_raw_sources": len(sources_meta),
        "total_raw_records_retrieved": sum(int(s.get("total_records_retrieved", 0)) for s in sources_meta),
        "total_files_hashed": len(tracked_files),
        "total_bytes": sum(int(f["bytes"]) for f in tracked_files),
        "sources": sources_meta,
        "files": tracked_files,
    }

    (REPO_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPO_ROOT / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"[build_manifest] Wrote MANIFEST.json and SHA256SUMS.txt ({len(tracked_files)} files, {manifest['total_bytes']/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
