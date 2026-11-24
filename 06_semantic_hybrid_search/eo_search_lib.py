"""Reusable SOTA hybrid semantic + lexical EO product search library.

Built specifically for the harmonized Global L3/L4 EO Product Catalogue
(39,474 canonical products across 34 catalogues) using:
- Apple Silicon MLX inference with Google EmbeddingGemma 2 (BF16 / 8-bit)
- EO-aware structured document engineering & GCMD/boilerplate cleaning
- DuckDB native multi-field BM25 full-text search with EO normalization
- Reciprocal Rank Fusion (RRF)
- Null-safe hard filtering (respecting ~70% upstream null rates)
- Product-family mirror & multi-epoch edition collapsing
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import queue
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

# Avoid HuggingFace fork/thread warning chatter while keeping Rust fast tokenizer threaded
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

DEFAULT_MODEL_BF16 = "mlx-community/embeddinggemma-2-bf16"
DEFAULT_MODEL_8BIT = "mlx-community/embeddinggemma-2-8bit"
DEFAULT_PT_EMBED_MODEL = "google/embeddinggemma-2"
QUERY_PROMPT_PREFIX = "task: search result | query: "
GITHUB_REPO = "do-me/global-EO-products-L3-L4"
DEFAULT_INDEX_RELEASE_TAG = "v4.1.0-indexes"
INDEX_ASSET_NAMES = ("full_bm25.duckdb", "full_embeddings_bf16.npz", "full_index_manifest.json")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using bounded 1 MiB reads."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_record_ids_sha256(record_ids: Iterable[str]) -> str:
    """Compute deterministic SHA-256 over an ordered sequence of catalogue record_ids."""
    h = hashlib.sha256()
    for rid in record_ids:
        h.update(str(rid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _download_release_asset(
    base_url: str,
    filename: str,
    dest_path: Path,
    expected_sha256: str = "",
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{filename}"
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".download.tmp")
    print(f"[eo-search] Downloading {filename} from {url} ...", file=sys.stderr, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "global-eo-search/4.1"})
    with urllib.request.urlopen(req, timeout=180) as resp, tmp_path.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    actual_sha = compute_file_sha256(tmp_path)
    if expected_sha256 and actual_sha != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch after downloading {filename}: expected {expected_sha256}, got {actual_sha}"
        )
    tmp_path.replace(dest_path)
    return dest_path


def ensure_search_indexes(
    artifacts_dir: Path,
    require_dense: bool = True,
    allow_download: bool = True,
    release_tag: str = DEFAULT_INDEX_RELEASE_TAG,
) -> dict[str, Path]:
    """Ensure full_bm25.duckdb (and optionally full_embeddings_bf16.npz) exist and match manifest SHA-256.

    When missing or stale vs `full_index_manifest.json`, downloads the published release assets from
    GitHub Releases (or `EO_SEARCH_INDEX_RELEASE_URL` when set) unless `allow_download=False`.
    """
    base_url = os.environ.get(
        "EO_SEARCH_INDEX_RELEASE_URL",
        f"https://github.com/{GITHUB_REPO}/releases/download/{release_tag}",
    )
    manifest_path = artifacts_dir / "full_index_manifest.json"
    if not manifest_path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing index manifest: {manifest_path}")
        _download_release_asset(base_url, "full_index_manifest.json", manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha_map: dict[str, str] = {}
    for k, v in (manifest.get("sha256") or {}).items():
        if isinstance(v, str) and v:
            sha_map[k] = v
    for k, meta in (manifest.get("release_assets") or {}).items():
        if isinstance(meta, dict) and isinstance(meta.get("sha256"), str):
            sha_map.setdefault(k, meta["sha256"])

    needed = ["full_bm25.duckdb"]
    if require_dense:
        needed.append("full_embeddings_bf16.npz")

    for fname in needed:
        target = artifacts_dir / fname
        expected = sha_map.get(fname, "")
        valid = target.exists() and (not expected or compute_file_sha256(target) == expected)
        if not valid:
            if not allow_download:
                reason = "missing" if not target.exists() else f"SHA-256 mismatch (expected {expected})"
                raise FileNotFoundError(f"Search index asset {target} is {reason} and auto-download is disabled.")
            _download_release_asset(base_url, fname, target, expected_sha256=expected)

    return {
        "manifest": manifest_path,
        "bm25": artifacts_dir / "full_bm25.duckdb",
        "embeddings": artifacts_dir / "full_embeddings_bf16.npz",
    }


def resolve_device(device: str = "auto") -> str:
    """Resolve the best available compute backend: MLX (Apple Silicon) -> CUDA -> MPS -> CPU."""
    req = (device or "auto").strip().lower()
    if req not in ("auto", "mlx", "cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported device '{device}'. Choose from: auto, mlx, cuda, mps, cpu.")

    has_mlx = (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_vlm") is not None
    )
    has_torch = importlib.util.find_spec("torch") is not None

    if req == "mlx":
        if not has_mlx:
            raise RuntimeError("Requested device='mlx', but mlx/mlx_vlm on Apple Silicon is unavailable.")
        return "mlx"

    if req == "auto":
        if has_mlx:
            return "mlx"
        if has_torch:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        raise RuntimeError(
            "No neural backend available: install `mlx` + `mlx-vlm` on Apple Silicon, or `torch` + `transformers` on CUDA/CPU."
        )

    if not has_torch:
        raise RuntimeError(f"Requested device='{req}', but PyTorch (`torch`) is not installed.")
    import torch

    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device='cuda', but torch.cuda.is_available() is False.")
    if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested device='mps', but torch.backends.mps.is_available() is False.")
    return req

# ---------------------------------------------------------------------------
# 1. EO-Aware Text Cleaning & Normalization
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_HEX_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_HEX_HASH_RE = re.compile(r"^[0-9a-f]{20,}$", re.IGNORECASE)
_CMR_CONCEPT_RE = re.compile(r"^C\d{7,}-[A-Z0-9_]+$")
_OPAQUE_ID_RE = re.compile(
    r"^(ZENODO:\d+|DATACITE:10\.\d+/|DOI:10\.\d+/|WGCLIMATE_CDR_\d+|RW:[0-9a-f-]+|FAO_ISO:[0-9a-f-]+|GEOKHUB:[a-z0-9-]+)$",
    re.IGNORECASE,
)

# Boilerplate sections at the tail of Zenodo/CMR/STAC descriptions that cause
# secondary-citation false positives and waste embedding tokens.
_BOILERPLATE_CUT_RE = re.compile(
    r"(?:\n|^)\s*(?:#{1,4}\s*)?"
    r"(?:references|bibliography|citation|how to cite|acknowledgements?|acknowledgments?|"
    r"funding|data availability|license and terms|terms of use|disclaimer|changelog|version history)"
    r"\s*(?::|\n)",
    re.IGNORECASE,
)

_GCMD_NOISE_PREFIXES = (
    "doc/noaa/",
    "nasa/",
    "esa/",
    "eumetsat/",
    "usgs/",
    "vertical location >",
    "location >",
    "geographic region >",
    "data resolution >",
    "horizontal data resolution >",
    "temporal data resolution >",
    "vertical data resolution >",
    "platforms >",
    "instruments >",
    "projects >",
    "providers >",
)

_ONTOLOGY_LABELS = {
    "eo_thematic_or_biophysical": "satellite biophysical/thematic EO product",
    "eo_reanalysis_or_model": "EO climate reanalysis/model product",
    "eo_constrained_hybrid": "EO-constrained hybrid geospatial product",
    "in_situ_gridded_interpolation": "in-situ gridded interpolation dataset",
    "non_eo_inventory_or_census": "non-EO inventory/census dataset",
    "eo_ard_surface_reflectance_mosaic": "EO analysis-ready surface reflectance mosaic",
    "authoritative_reference_geometry": "authoritative reference boundary/geometry",
}

_GEO_LABELS = {
    "global": "global",
    "quasi_global": "quasi-global",
    "europe_regional": "Europe regional",
    "hemispheric_or_pantropical": "hemispheric/pantropical",
    "regional_or_local": "regional/local",
}

_CADENCE_LABELS = {
    "sub_daily_or_daily": "daily/sub-daily",
    "multi_day_or_dekadal": "multi-day/10-daily",
    "monthly_or_seasonal": "monthly/seasonal",
    "annual": "annual",
    "multi_year_periodic": "multi-year periodic",
    "static_or_single_epoch": "static/single-epoch",
}

_SUBSCRIPT_MAP = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "⁻": "-",
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
    }
)

# Generic EO-domain orthographic, chemistry & standard variable-acronym normalization for BM25.
# Strictly zero product brand names and zero satellite/mission names to prevent any benchmark leakage.
_EO_LEXICAL_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bnormalised\b", re.I), "normalized"),
    (re.compile(r"\bcolour\b", re.I), "color"),
    (re.compile(r"\bcolours\b", re.I), "colors"),
    (re.compile(r"\bmodelling\b", re.I), "modeling"),
    (re.compile(r"\bmodelled\b", re.I), "modeled"),
    (re.compile(r"\burbanisation\b", re.I), "urbanization"),
    (re.compile(r"\bcharacterisation\b", re.I), "characterization"),
    (re.compile(r"\bclimatological\b", re.I), "climatology"),
    (re.compile(r"\bburnt\b", re.I), "burned"),
    (re.compile(r"\babove[-\s]?ground\b", re.I), "aboveground"),
    (re.compile(r"\bbelow[-\s]?ground\b", re.I), "belowground"),
    (re.compile(r"\bbuilt[-\s]?up\b", re.I), "builtup built up"),
    (re.compile(r"\bland[-\s]?cover\b", re.I), "land cover landcover"),
    (re.compile(r"\bland[-\s]?use\b", re.I), "land use landuse"),
    (re.compile(r"\bsea[-\s]?ice\b", re.I), "sea ice"),
    (re.compile(r"\bsea[-\s]?level\b", re.I), "sea level"),
    (re.compile(r"\bchlorophyll[-\s]?a\b", re.I), "chlorophyll a chla"),
    (re.compile(r"\bpm\s*2\.?5\b", re.I), "pm25 pm2.5 fine particulate"),
    (re.compile(r"\bco\s*2\b", re.I), "co2 carbon dioxide"),
    (re.compile(r"\bch\s*4\b", re.I), "ch4 methane"),
    (re.compile(r"\bno\s*2\b", re.I), "no2 nitrogen dioxide"),
    (re.compile(r"\bhigh[-\s]?resolution\b", re.I), "high resolution"),
    (re.compile(r"\bnear[-\s]?real[-\s]?time\b", re.I), "near real time nrt"),
    (re.compile(r"\bsub[-\s]?surface\b", re.I), "subsurface"),
    (re.compile(r"\bfresh[-\s]?water\b", re.I), "freshwater"),
    (re.compile(r"\btree[-\s]?cover\b", re.I), "tree cover"),
    (re.compile(r"\bsoil organic carbon\b", re.I), "soil organic carbon soc"),
    (re.compile(r"\bleaf area index\b", re.I), "leaf area index lai"),
    (re.compile(r"\bgross primary product\w*\b", re.I), "gross primary productivity gpp"),
    (re.compile(r"\bnet primary product\w*\b", re.I), "net primary productivity npp"),
    (re.compile(r"\bsea surface temperature\b", re.I), "sea surface temperature sst"),
    (re.compile(r"\bland surface temperature\b", re.I), "land surface temperature lst"),
    (re.compile(r"\bsea surface salinity\b", re.I), "sea surface salinity sss"),
    (re.compile(r"\bdigital elevation model\b", re.I), "digital elevation model dem"),
    (re.compile(r"\bsolar[-\s]?induced\s+(?:chlorophyll\s+)?fluorescence\b", re.I), "solar induced fluorescence sif"),
    (re.compile(r"\bwater[-\s]?quality\b", re.I), "water quality"),
    (re.compile(r"\bheat[-\s]?island\b", re.I), "heat island uhi"),
    (re.compile(r"\bburned\s+area\b", re.I), "burned area wildfire"),
]


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("none", "nan", "null", "n/a") else s


def normalize_eo_text(text: str) -> str:
    """Normalize EO acronyms, unicode subscripts, underscores/separators, hyphens, and UK/US variants for BM25."""
    if not text:
        return ""
    s = text.translate(_SUBSCRIPT_MAP)
    # Replace underscore and non-hyphen/non-dot punctuation BEFORE lexical rules so CO2_flux / land_cover match!
    s = re.sub(r"[/|;:,()\[\]{}_]+", " ", s)
    for pat, repl in _EO_LEXICAL_SUBS:
        s = pat.sub(repl, s)
    return _WS_RE.sub(" ", s).strip()


def clean_gcmd_keywords(raw_kw: Any, max_terms: int = 24) -> str:
    """Clean and deduplicate raw keyword strings (including bloated GCMD paths up to 472 KB)."""
    s = _safe_str(raw_kw)
    if not s:
        return ""
    # Fast cap on pathological blobs before splitting
    if len(s) > 16000:
        s = s[:16000]
    parts = re.split(r"[;|\n]+", s)
    if len(parts) == 1 and "," in s:
        parts = s.split(",")

    seen: set[str] = set()
    cleaned: list[str] = []
    for raw_part in parts:
        p = raw_part.strip().strip('"').strip("'")
        if not p or len(p) < 2:
            continue
        pl = p.lower()
        if any(pl.startswith(pref) for pref in _GCMD_NOISE_PREFIXES):
            continue
        if _HEX_UUID_RE.match(p) or _URL_RE.match(p):
            continue
        # Handle GCMD hierarchy: EARTH SCIENCE > BIOSPHERE > VEGETATION > LEAF CHARACTERISTICS
        if ">" in p:
            segs = [seg.strip() for seg in p.split(">") if seg.strip()]
            # Drop top-level generic prefix ("EARTH SCIENCE", "EARTH SCIENCE SERVICES")
            if segs and segs[0].upper().startswith("EARTH SCIENCE"):
                segs = segs[1:]
            # Keep the most specific 2 levels (e.g. "VEGETATION: LEAF CHARACTERISTICS")
            if not segs:
                continue
            cand = " / ".join(segs[-2:]) if len(segs) >= 2 else segs[-1]
            # Convert ALL CAPS GCMD labels to readable lowercase
            if cand.isupper():
                cand = cand.lower()
        else:
            cand = p.lower() if (p.isupper() and len(p) > 6) else p

        cand = _WS_RE.sub(" ", cand).strip(" .-")
        if not cand or len(cand) < 2 or len(cand) > 90:
            continue
        key = cand.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(cand)
            if len(cleaned) >= max_terms:
                break
    return ", ".join(cleaned)


def clean_sensors(raw_sensors: Any, max_items: int = 10) -> str:
    """Deduplicate and compact platform/sensor metadata (capping pathological 95 KB lists)."""
    s = _safe_str(raw_sensors)
    if not s:
        return ""
    if len(s) > 8000:
        s = s[:8000]
    parts = re.split(r"[;|,\n]+", s)
    seen: set[str] = set()
    out: list[str] = []
    for raw_part in parts:
        p = raw_part.strip()
        if not p or len(p) < 2 or len(p) > 60:
            continue
        if ">" in p:
            p = p.split(">")[-1].strip()
        pl = p.lower()
        if pl in ("not applicable", "n/a", "unknown", "in situ", "none", "null"):
            continue
        if pl not in seen:
            seen.add(pl)
            out.append(p)
            if len(out) >= max_items:
                break
    return ", ".join(out)


def extract_meaningful_shortname(raw_id: Any, title: str) -> str:
    """Extract informative EO product codes (e.g. MOD13Q1, GEDI_L4B) while ignoring UUIDs/hashes."""
    rid = _safe_str(raw_id)
    if not rid:
        return ""
    if _HEX_UUID_RE.match(rid) or _HEX_HASH_RE.match(rid) or _CMR_CONCEPT_RE.match(rid) or _OPAQUE_ID_RE.match(rid):
        return ""
    # Strip STAC/GEE path noise while retaining readable tail
    if "/" in rid:
        segs = [x for x in rid.split("/") if x and x.lower() not in ("projects", "sat-io", "open-datasets", "v1", "v2", "latest")]
        rid = "/".join(segs[-2:]) if len(segs) >= 2 else (segs[-1] if segs else "")
    if ":" in rid:
        tail = rid.split(":")[-1]
        if not _HEX_UUID_RE.match(tail) and len(tail) >= 3:
            rid = tail
        else:
            return ""
    if len(rid) < 3 or len(rid) > 64 or rid.isdigit():
        return ""
    # Skip if already verbatim in title
    if rid.lower() in title.lower():
        return ""
    return rid


def clean_description(desc: Any, max_chars: int = 2400) -> str:
    """Strip HTML, URLs, and citation/reference boilerplate tails from EO descriptions."""
    s = _safe_str(desc)
    if not s:
        return ""
    # Trim ultra-long READMEs early before regex scanning
    if len(s) > 12000:
        s = s[:12000]
    s = _HTML_TAG_RE.sub(" ", s)
    s = s.translate(_SUBSCRIPT_MAP)
    # Cut off bibliography/citation/funding tail if there is meaningful text before it
    m = _BOILERPLATE_CUT_RE.search(s)
    if m and m.start() >= 180:
        s = s[: m.start()]
    s = _URL_RE.sub("", s)
    # Collapse markdown formatting clutter
    s = re.sub(r"[#*_`~]{2,}", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    if len(s) > max_chars:
        # Try to trim cleanly at a sentence boundary near max_chars
        cut = s[:max_chars]
        last_dot = cut.rfind(". ")
        if last_dot > max_chars * 0.65:
            s = cut[: last_dot + 1]
        else:
            s = cut.rsplit(" ", 1)[0]
    return s


def format_resolution_human(meters: Any, raw_val: Any = None, raw_units: Any = None) -> str:
    """Format adjudicated spatial grid meters into natural EO phrasing."""
    try:
        if meters is not None and not (isinstance(meters, float) and np.isnan(meters)):
            m = float(meters)
            if m > 0:
                if m < 1000:
                    return f"{m:g} m" if m < 10 else f"{int(round(m))} m"
                km = m / 1000.0
                return f"{km:.1f} km" if km < 10 else f"{int(round(km))} km"
    except (TypeError, ValueError):
        pass
    rv = _safe_str(raw_val)
    ru = _safe_str(raw_units)
    if rv:
        return f"{rv} {ru}".strip()
    return ""


def build_eo_document_parts(row: dict[str, Any]) -> tuple[str, str]:
    """Build (clean_title, structured_body) for EO-aware EmbeddingGemma 2 encoding."""
    title = _WS_RE.sub(" ", _safe_str(row.get("title")).translate(_SUBSCRIPT_MAP)).strip()
    if not title:
        title = "none"

    meta_bits: list[str] = []
    shortname = extract_meaningful_shortname(row.get("raw_id"), title)
    if shortname:
        meta_bits.append(f"ID: {shortname}")

    inst = _safe_str(row.get("producing_institution"))
    cop_svc = _safe_str(row.get("copernicus_service"))
    if cop_svc:
        inst_part = f"Copernicus {cop_svc}" + (f" ({inst[:60]})" if inst and cop_svc.lower() not in inst.lower() else "")
        meta_bits.append(f"Provider: {inst_part}")
    elif inst:
        meta_bits.append(f"Provider: {inst[:80]}")

    ont = _ONTOLOGY_LABELS.get(_safe_str(row.get("product_ontology_class")), "")
    lvl = _safe_str(row.get("processing_level"))
    if ont and lvl:
        meta_bits.append(f"Type: {lvl} {ont}")
    elif ont:
        meta_bits.append(f"Type: {ont}")
    elif lvl:
        meta_bits.append(f"Level: {lvl}")

    geo = _GEO_LABELS.get(_safe_str(row.get("geographic_coverage_class")), "")
    if geo:
        meta_bits.append(f"Coverage: {geo}")

    res_str = format_resolution_human(
        row.get("adjudicated_output_grid_meters"),
        row.get("adjudicated_spatial_grid_value"),
        row.get("adjudicated_spatial_grid_units"),
    )
    if res_str:
        meta_bits.append(f"Grid: {res_str}")

    cad = _CADENCE_LABELS.get(_safe_str(row.get("temporal_cadence_class")), "")
    if cad:
        meta_bits.append(f"Cadence: {cad}")

    sy = row.get("temporal_start_year")
    ey = row.get("temporal_end_year")
    ongoing = row.get("is_ongoing_archive")
    try:
        sy_i = int(sy) if sy is not None and not (isinstance(sy, float) and np.isnan(sy)) else None
        ey_i = int(ey) if ey is not None and not (isinstance(ey, float) and np.isnan(ey)) else None
    except (TypeError, ValueError):
        sy_i, ey_i = None, None
    if sy_i and ongoing is True:
        meta_bits.append(f"Period: {sy_i}-present")
    elif sy_i and ey_i:
        meta_bits.append(f"Period: {sy_i}-{ey_i}")
    elif sy_i:
        meta_bits.append(f"Start: {sy_i}")

    sensors = clean_sensors(row.get("platform_sensor_raw"), max_items=8)
    if sensors:
        meta_bits.append(f"Sensors: {sensors}")

    kw = clean_gcmd_keywords(row.get("keywords"), max_terms=20)
    if kw:
        meta_bits.append(f"Topics: {kw}")

    header = " | ".join(meta_bits)
    desc = clean_description(row.get("description"), max_chars=2400)

    if header and desc:
        body = f"{header}. {desc}"
    elif header:
        body = header
    else:
        body = desc or title

    return title, body


def build_eo_document(row: dict[str, Any]) -> str:
    """Official EmbeddingGemma 2 document format with EO-structured metadata."""
    title, body = build_eo_document_parts(row)
    return f"title: {title} | text: {body}"


def build_raw_document(row: dict[str, Any]) -> str:
    """Naive uncleaned raw text concatenation (ablation baseline, no official prompt)."""
    title = _safe_str(row.get("title"))
    desc = _safe_str(row.get("description"))[:4000]
    kw = _safe_str(row.get("keywords"))[:1500]
    parts = [p for p in (title, desc, kw) if p]
    return _WS_RE.sub(" ", " ".join(parts)).strip()


def build_query_prompt(query: str) -> str:
    """Official EmbeddingGemma 2 asymmetric retrieval query format."""
    q = _WS_RE.sub(" ", query.translate(_SUBSCRIPT_MAP)).strip()
    return f"{QUERY_PROMPT_PREFIX}{q}"


def build_bm25_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return (title_norm, keywords_norm, body_norm) for DuckDB multi-field BM25."""
    title, body = build_eo_document_parts(row)
    shortname = extract_meaningful_shortname(row.get("raw_id"), title)
    title_raw = f"{title} {shortname}".strip()
    kw_raw = " ".join(
        filter(
            None,
            [
                clean_gcmd_keywords(row.get("keywords"), max_terms=28),
                clean_sensors(row.get("platform_sensor_raw"), max_items=10),
                _safe_str(row.get("copernicus_service")),
                _safe_str(row.get("producing_institution"))[:80],
            ],
        )
    )
    return (
        normalize_eo_text(title_raw),
        normalize_eo_text(kw_raw),
        normalize_eo_text(body),
    )


def chunk_document_texts(
    tokenizer: Any,
    title: str,
    body: str,
    max_length: int = 512,
    stride: int = 384,
    max_chunks: int = 4,
) -> list[str]:
    """Split a long EO document into overlapping chunks preserving the title prefix.

    Short documents fitting inside `max_length` return a single formatted string
    identical to `build_eo_document`.
    """
    prefix = f"title: {title} | text: "
    full_text = prefix + body
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    # Reserve 4 special tokens (BOS/EOS safety margin)
    avail = max(64, max_length - len(prefix_ids) - 4)
    if len(body_ids) <= avail:
        return [full_text]

    step = max(64, min(avail, stride))
    chunks: list[str] = []
    start = 0
    while start < len(body_ids) and len(chunks) < max_chunks:
        piece_ids = body_ids[start : start + avail]
        piece_txt = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        if piece_txt:
            chunks.append(prefix + piece_txt)
        if start + avail >= len(body_ids):
            break
        start += step
    return chunks or [full_text]


# ---------------------------------------------------------------------------
# 2. Apple Silicon MLX EmbeddingGemma 2 Engine
# ---------------------------------------------------------------------------


def matryoshka_truncate(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Slice EmbeddingGemma 2 vectors to a Matryoshka prefix (768/512/256/128) and L2-renormalize."""
    if dim >= embeddings.shape[1]:
        return embeddings
    sub = embeddings[:, :dim].astype(np.float32, copy=True)
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return sub / norms


class MLXEmbeddingGemma2:
    """High-throughput MLX EmbeddingGemma 2 encoder with length-sorted bucketing & async pipelining.

    Uses:
    - Text-only fast forward path (bit-exact with stock mlx_vlm, avoids (B,1,L,L) sliding mask
      materialization when L <= sliding_window + 1 = 513).
    - Length-sorted descending bucketing (cuts padding waste from ~39% to ~2%).
    - Double-buffered `mx.async_eval` GPU pipelining + threaded CPU padding prefetch.
    - Optional fixed-shape coarse bucketing (`fixed_buckets`) so `mx.compile` compiles only
      a handful of shapes instead of recompiling on every dynamic length.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_BF16):
        import mlx.core as mx
        from mlx_vlm.embedding_loader import load_embedding_model
        from mlx_vlm.models.embedding_gemma2.embedding_gemma2 import (
            mean_pooling,
            normalize_embeddings,
        )
        from mlx_vlm.utils import get_model_path
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.mx = mx
        self._mean_pooling = mean_pooling
        self._normalize_embeddings = normalize_embeddings
        t0 = time.perf_counter()
        self.model_path = get_model_path(model_id)
        self.model = load_embedding_model(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pad_token_id = int(self.tokenizer.pad_token_id or 0)
        self.load_seconds = time.perf_counter() - t0
        self._fast_fwd = self._build_fast_text_forward()
        self._compiled_fn = None
        self._warmed_shapes: set[tuple[int, int]] = set()

    def _build_fast_text_forward(self):
        mx = self.mx
        lm = self.model.language_model
        tc = self.model.config.text_config
        sw = int(tc.sliding_window)
        scale = float(tc.hidden_size) ** 0.5
        layer_types = list(lm.config.layer_types)
        layers = lm.layers
        mean_pooling = self._mean_pooling
        normalize_embeddings = self._normalize_embeddings

        def fast_fwd(input_ids: mx.array, attention_mask: mx.array) -> mx.array:
            h = lm.embed_tokens(input_ids)
            h = h * mx.array(scale, dtype=h.dtype)
            length = h.shape[1]
            full_mask = attention_mask[:, None, None, :].astype(mx.bool_)
            positions = mx.arange(length)[None]
            if length <= sw + 1:
                sliding_mask = full_mask
            else:
                pos1d = mx.arange(length)
                sliding_mask = full_mask & (mx.abs(pos1d[:, None] - pos1d[None, :]) <= sw)
            per_layer_inputs = lm.ple(h) if lm.ple is not None else None
            for i, layer in enumerate(layers):
                mask = sliding_mask if layer_types[i] == "sliding_attention" else full_mask
                h = layer(
                    h,
                    mask,
                    positions,
                    None if per_layer_inputs is None else per_layer_inputs[:, :, i],
                )
            hidden = lm.embedding_projection(lm.norm(h))
            return normalize_embeddings(mean_pooling(hidden, attention_mask))

        return fast_fwd

    def _get_forward_fn(self, use_compile: bool):
        if not use_compile:
            return self._fast_fwd
        if self._compiled_fn is None:
            self._compiled_fn = self.mx.compile(self._fast_fwd)
        return self._compiled_fn

    def _pad_token_lists(
        self,
        token_lists: list[list[int]],
        pad_to_multiple_of: int = 8,
        fixed_len: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(token_lists)
        if fixed_len is not None:
            max_len = fixed_len
        else:
            max_len = max((len(t) for t in token_lists), default=1)
            if pad_to_multiple_of > 1:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        ids = np.full((bsz, max_len), self.pad_token_id, dtype=np.int32)
        mask = np.zeros((bsz, max_len), dtype=np.int32)
        for i, seq in enumerate(token_lists):
            n = min(len(seq), max_len)
            if n > 0:
                ids[i, :n] = seq[:n]
                mask[i, :n] = 1
        return ids, mask

    def _build_batches(
        self,
        lengths: np.ndarray,
        batch_size: int,
        sort_by_length: bool,
        token_budget: Optional[int],
        fixed_buckets: Optional[list[int]] = None,
    ) -> list[tuple[np.ndarray, Optional[int]]]:
        n = len(lengths)
        if sort_by_length:
            # Descending sort allocates largest Metal buffers first -> cleaner memory reuse
            order = np.argsort(-lengths, kind="stable")
        else:
            order = np.arange(n, dtype=np.int64)

        batches: list[tuple[np.ndarray, Optional[int]]] = []
        if fixed_buckets:
            sb = sorted(fixed_buckets)
            bmap: dict[int, list[int]] = {b: [] for b in sb}
            for idx in order:
                L = int(lengths[idx])
                b = next((x for x in sb if L <= x), sb[-1])
                bmap[b].append(int(idx))
            for b in reversed(sb):
                arr = bmap[b]
                for i in range(0, len(arr), batch_size):
                    batches.append((np.array(arr[i : i + batch_size], dtype=np.int64), b))
            return batches

        if token_budget is not None and token_budget > 0:
            i = 0
            while i < n:
                cur_max = int(lengths[order[i]])
                j = i + 1
                while j < n and (j - i) < batch_size:
                    nxt_max = max(cur_max, int(lengths[order[j]]))
                    if nxt_max * (j - i + 1) > token_budget:
                        break
                    cur_max = nxt_max
                    j += 1
                batches.append((order[i:j], None))
                i = j
        else:
            for i in range(0, n, batch_size):
                batches.append((order[i : i + batch_size], None))
        return batches

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 512,
        sort_by_length: bool = True,
        prefetch: bool = True,
        async_pipe: bool = True,
        pad_to_multiple_of: int = 8,
        use_compile: bool = False,
        fixed_buckets: Optional[list[int]] = None,
        token_budget: Optional[int] = None,
        clear_cache_every: int = 64,
        progress_every: int = 0,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Encode texts into L2-normalized float32 (N, 768) embeddings on MLX."""
        mx = self.mx
        n = len(texts)
        if n == 0:
            empty = np.zeros((0, 768), dtype=np.float32)
            return (empty, {}) if return_stats else empty

        t_total_0 = time.perf_counter()

        # Step 1: Fast Rust multi-threaded tokenization (unpadded so exact lengths are known)
        t_tok_0 = time.perf_counter()
        enc = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        all_ids: list[list[int]] = enc["input_ids"]
        lengths = np.fromiter((len(x) for x in all_ids), dtype=np.int32, count=n)
        t_tok = time.perf_counter() - t_tok_0

        # Step 2: Build sorted (or fixed-bucket) batches
        batches = self._build_batches(lengths, batch_size, sort_by_length, token_budget, fixed_buckets)
        fwd = self._get_forward_fn(use_compile)

        # Warm up fixed bucket shapes once when compiling
        if use_compile and fixed_buckets:
            for b in fixed_buckets:
                sh = (batch_size, b)
                if sh not in self._warmed_shapes:
                    w = fwd(mx.zeros(sh, dtype=mx.int32), mx.ones(sh, dtype=mx.int32))
                    mx.eval(w)
                    self._warmed_shapes.add(sh)

        out = np.empty((n, 768), dtype=np.float32)
        total_padded_tokens = 0
        total_real_tokens = int(lengths.sum())

        def _iter_prepared():
            if prefetch and len(batches) > 1:
                q: queue.Queue[Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]] = queue.Queue(maxsize=3)
                err_box: list[BaseException] = []

                def _producer():
                    try:
                        for idx_arr, flen in batches:
                            sub_lists = [all_ids[int(k)] for k in idx_arr]
                            ids_np, mask_np = self._pad_token_lists(sub_lists, pad_to_multiple_of, fixed_len=flen)
                            q.put((idx_arr, ids_np, mask_np))
                    except BaseException as exc:
                        err_box.append(exc)
                    finally:
                        q.put(None)

                prod = threading.Thread(target=_producer, daemon=True)
                prod.start()
                while True:
                    item = q.get()
                    if item is None:
                        break
                    yield item
                prod.join()
                if err_box:
                    raise err_box[0]
            else:
                for idx_arr, flen in batches:
                    sub_lists = [all_ids[int(k)] for k in idx_arr]
                    ids_np, mask_np = self._pad_token_lists(sub_lists, pad_to_multiple_of, fixed_len=flen)
                    yield idx_arr, ids_np, mask_np

        t_gpu_0 = time.perf_counter()
        pending: Optional[tuple[np.ndarray, Any]] = None
        docs_done = 0
        for bi, (idx_arr, ids_np, mask_np) in enumerate(_iter_prepared(), start=1):
            total_padded_tokens += int(ids_np.size)
            cur_embeds = fwd(mx.array(ids_np), mx.array(mask_np))
            if async_pipe:
                mx.async_eval(cur_embeds)
                if pending is not None:
                    p_idx, p_emb = pending
                    out[p_idx] = np.asarray(p_emb, dtype=np.float32)
                    docs_done += len(p_idx)
                pending = (idx_arr, cur_embeds)
            else:
                mx.eval(cur_embeds)
                out[idx_arr] = np.asarray(cur_embeds, dtype=np.float32)
                docs_done += len(idx_arr)
            if clear_cache_every and (bi % clear_cache_every == 0):
                if pending is not None:
                    p_idx, p_emb = pending
                    out[p_idx] = np.asarray(p_emb, dtype=np.float32)
                    docs_done += len(p_idx)
                    pending = None
                mx.clear_cache()
            if progress_every and (bi % progress_every == 0 or bi == len(batches)):
                elapsed = time.perf_counter() - t_gpu_0
                rate = docs_done / max(1e-6, elapsed)
                rem_s = (n - docs_done) / max(1e-6, rate)
                mem_gb = mx.get_peak_memory() / (1024**3)
                print(
                    f"  [MLX] {docs_done:6d}/{n:,} ({100.0*docs_done/n:5.1f}%) | "
                    f"{rate:5.1f} doc/s | elapsed={elapsed:5.1f}s | ETA={rem_s:5.1f}s | peak={mem_gb:.2f} GB",
                    flush=True,
                )

        if pending is not None:
            p_idx, p_emb = pending
            out[p_idx] = np.asarray(p_emb, dtype=np.float32)

        t_gpu = time.perf_counter() - t_gpu_0
        t_total = time.perf_counter() - t_total_0

        if return_stats:
            stats = {
                "n_texts": n,
                "n_batches": len(batches),
                "batch_size": batch_size,
                "token_budget": token_budget,
                "fixed_buckets": fixed_buckets,
                "max_length": max_length,
                "sort_by_length": sort_by_length,
                "prefetch": prefetch,
                "async_pipe": async_pipe,
                "use_compile": use_compile,
                "real_tokens": total_real_tokens,
                "padded_tokens": total_padded_tokens,
                "padding_overhead_pct": round(
                    100.0 * (total_padded_tokens - total_real_tokens) / max(1, total_padded_tokens), 2
                ),
                "tok_seconds": round(t_tok, 4),
                "gpu_seconds": round(t_gpu, 4),
                "total_seconds": round(t_total, 4),
                "docs_per_sec": round(n / max(1e-9, t_total), 2),
                "real_tok_per_sec": round(total_real_tokens / max(1e-9, t_total), 1),
                "peak_mem_gb": round(mx.get_peak_memory() / (1024**3), 3),
            }
            return out, stats
        return out

    def encode_queries(self, queries: list[str], max_length: int = 256) -> np.ndarray:
        prompts = [build_query_prompt(q) for q in queries]
        return self.encode_texts(
            prompts,
            batch_size=min(16, max(1, len(prompts))),
            max_length=max_length,
            sort_by_length=False,
            prefetch=False,
            async_pipe=False,
        )

    def encode_documents_chunked(
        self,
        rows: list[dict[str, Any]],
        batch_size: int = 16,
        max_length: int = 512,
        stride: int = 384,
        max_chunks: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode documents with overlapping chunks; returns (chunk_vecs, chunk_doc_idx)."""
        chunk_texts: list[str] = []
        chunk_doc_idx: list[int] = []
        for i, row in enumerate(rows):
            title, body = build_eo_document_parts(row)
            chunks = chunk_document_texts(
                self.tokenizer, title, body, max_length=max_length, stride=stride, max_chunks=max_chunks
            )
            for c in chunks:
                chunk_texts.append(c)
                chunk_doc_idx.append(i)
        chunk_vecs = self.encode_texts(
            chunk_texts,
            batch_size=batch_size,
            max_length=max_length,
            sort_by_length=True,
            prefetch=True,
            async_pipe=True,
        )
        return chunk_vecs, np.asarray(chunk_doc_idx, dtype=np.int32)


class PyTorchEmbeddingGemma2:
    """Device-agnostic PyTorch/Transformers EmbeddingGemma 2 encoder (CUDA / MPS / CPU).

    Uses official ungated `google/embeddinggemma-2` (Apache-2.0) in BF16 (CUDA Ampere+ / CPU)
    or FP32 (older CUDA / MPS; never float16 per model card), with mask-aware mean pooling
    over `last_hidden_state` and float32 L2 normalization (cosine > 0.9999 vs MLX BF16).
    """

    def __init__(self, model_id: str = DEFAULT_PT_EMBED_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.torch = torch
        t0 = time.perf_counter()

        if device == "cuda":
            use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        elif device == "cpu":
            self.dtype = torch.bfloat16
        else:
            # MPS: use float32 to avoid bfloat16/float16 kernel edge cases
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 512,
        sort_by_length: bool = True,
        return_stats: bool = False,
        progress_every: int = 0,
        **_unused_mlx_kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        torch = self.torch
        n = len(texts)
        if n == 0:
            empty = np.zeros((0, 768), dtype=np.float32)
            return (empty, {}) if return_stats else empty

        t0 = time.perf_counter()
        if sort_by_length and n > batch_size:
            order = sorted(range(n), key=lambda i: len(texts[i]))
        else:
            order = list(range(n))

        out = np.empty((n, 768), dtype=np.float32)
        total_real_tokens = 0
        total_padded_tokens = 0
        n_batches = (n + batch_size - 1) // batch_size

        with torch.inference_mode():
            for bi in range(n_batches):
                idxs = order[bi * batch_size : (bi + 1) * batch_size]
                batch_texts = [texts[i] for i in idxs]
                inp = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                attn = inp["attention_mask"]
                total_real_tokens += int(attn.sum().item())
                total_padded_tokens += int(attn.numel())
                inp = {k: v.to(self.device) for k, v in inp.items()}
                hidden = self.model(**inp).last_hidden_state.float()
                mask = inp["attention_mask"].unsqueeze(-1).to(torch.float32)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(pooled, p=2, dim=-1).cpu().numpy()
                out[idxs] = emb
                if progress_every and ((bi + 1) % progress_every == 0 or bi == n_batches - 1):
                    dt = time.perf_counter() - t0
                    done = min(n, (bi + 1) * batch_size)
                    print(f"    [{done:5d}/{n}] {done / max(dt, 1e-6):6.1f} doc/s ({dt:5.1f}s)", flush=True)

        t_total = time.perf_counter() - t0
        if return_stats:
            peak_gb = (
                round(torch.cuda.max_memory_allocated() / (1024**3), 3)
                if self.device == "cuda"
                else 0.0
            )
            stats = {
                "n_texts": n,
                "device": self.device,
                "dtype": str(self.dtype),
                "batch_size": batch_size,
                "max_length": max_length,
                "real_tokens": total_real_tokens,
                "padded_tokens": total_padded_tokens,
                "padding_overhead_pct": round(
                    100.0 * (total_padded_tokens - total_real_tokens) / max(1, total_padded_tokens), 2
                ),
                "total_seconds": round(t_total, 4),
                "docs_per_sec": round(n / max(1e-9, t_total), 2),
                "real_tok_per_sec": round(total_real_tokens / max(1e-9, t_total), 1),
                "peak_mem_gb": peak_gb,
            }
            return out, stats
        return out

    def encode_queries(self, queries: list[str], max_length: int = 256) -> np.ndarray:
        prompts = [build_query_prompt(q) for q in queries]
        return self.encode_texts(
            prompts,
            batch_size=min(16, max(1, len(prompts))),
            max_length=max_length,
            sort_by_length=False,
        )


def load_dense_embedder(
    device: str = "auto",
    model_id: Optional[str] = None,
) -> tuple[MLXEmbeddingGemma2 | PyTorchEmbeddingGemma2, str, str]:
    """Auto-select and load the best EmbeddingGemma 2 backend (MLX on Apple Silicon, else CUDA/MPS/CPU PyTorch)."""
    dev = resolve_device(device)
    if dev == "mlx":
        mid = model_id or DEFAULT_MODEL_BF16
        return MLXEmbeddingGemma2(mid), "mlx", mid
    mid = model_id or DEFAULT_PT_EMBED_MODEL
    return PyTorchEmbeddingGemma2(mid, device=dev), dev, mid


def maxsim_chunk_scores(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_doc_idx: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    """Aggregate multi-chunk cosine similarities into per-document MaxSim scores."""
    c_scores = chunk_vecs @ query_vec
    doc_scores = np.full(n_docs, -1e9, dtype=np.float32)
    np.maximum.at(doc_scores, chunk_doc_idx, c_scores)
    return doc_scores


# ---------------------------------------------------------------------------
# 3. DuckDB Native Multi-Field BM25 & Naive ILIKE Baselines
# ---------------------------------------------------------------------------


class DuckDBLexicalIndex:
    """Embedded DuckDB multi-field BM25 FTS index + naive ILIKE baseline.

    Keyed by immutable `record_id` (not mutable row position) so prefiltering,
    FTS scoring, and dense vector fusion remain aligned even if row order shifts.
    """

    def __init__(self, db_path: str = ":memory:", read_only: bool = False):
        self.db_path = db_path
        self.read_only = bool(read_only and db_path != ":memory:")
        self.con = duckdb.connect(db_path, read_only=self.read_only)
        if self.read_only:
            # Pin single-threaded FTS aggregation on read-only query connections to eliminate
            # parallel float64 reduction jitter (~1e-15 ULP) in match_bm25().
            self.con.execute("SET threads=1;")
        try:
            self.con.execute("LOAD fts;")
        except Exception:
            self.con.execute("INSTALL fts; LOAD fts;")
        self._bound_rids: Optional[np.ndarray] = None
        self._bound_rid_to_idx: Optional[dict[str, int]] = None
        self._db_rids: Optional[np.ndarray] = None
        self._db_rid_to_idx: Optional[dict[str, int]] = None
        if self.read_only:
            try:
                rows = self.con.execute("SELECT record_id FROM eo_docs ORDER BY doc_idx ASC").fetchall()
                rids = [str(r[0]) for r in rows]
                self._db_rids = np.asarray(rids, dtype=object)
                self._db_rid_to_idx = {r: i for i, r in enumerate(rids)}
            except Exception:
                pass

    def get_meta(self) -> dict[str, str]:
        """Return index alignment metadata stored inside `eo_meta`."""
        try:
            rows = self.con.execute("SELECT key, val FROM eo_meta").fetchall()
            return {str(k): str(v) for k, v in rows}
        except Exception:
            return {}

    def compute_content_sha256(self) -> str:
        """Compute deterministic SHA-256 over indexed lexical fields independent of DuckDB storage pages."""
        h = hashlib.sha256()
        cur = self.con.execute(
            "SELECT record_id, title_bm25, keywords_bm25, body_bm25 FROM eo_docs ORDER BY doc_idx ASC, record_id ASC"
        )
        while True:
            rows = cur.fetchmany(2048)
            if not rows:
                break
            for rid, t_bm, k_bm, b_bm in rows:
                h.update(str(rid or "").encode("utf-8"))
                h.update(b"\x00")
                h.update(str(t_bm or "").encode("utf-8"))
                h.update(b"\x00")
                h.update(str(k_bm or "").encode("utf-8"))
                h.update(b"\x00")
                h.update(str(b_bm or "").encode("utf-8"))
                h.update(b"\n")
        return h.hexdigest()

    def verify_against_dataframe(self, df: pd.DataFrame, catalogue_sha256: str = "") -> dict[str, str]:
        """Verify ordered `record_id` and `catalogue_sha256` alignment and bind DataFrame row positions."""
        db_rows = self.con.execute("SELECT record_id FROM eo_docs ORDER BY doc_idx ASC").fetchall()
        db_rids = [str(r[0]) for r in db_rows]
        df_rids = [str(r) for r in df["record_id"]]
        if len(db_rids) != len(df_rids):
            raise RuntimeError(
                f"BM25 index row count mismatch: index has {len(db_rids)} rows vs DataFrame {len(df_rids)} rows."
            )
        if db_rids != df_rids:
            mismatch_idx = next((i for i, (a, b) in enumerate(zip(db_rids, df_rids)) if a != b), 0)
            raise RuntimeError(
                f"BM25 index record_id ordering mismatch at row {mismatch_idx}: "
                f"index={db_rids[mismatch_idx]!r} vs catalogue={df_rids[mismatch_idx]!r}. "
                "Rebuild search indexes via 05_index_full_catalogue.py."
            )
        meta = self.get_meta()
        meta_cat_sha = meta.get("catalogue_sha256", "")
        if catalogue_sha256 and meta_cat_sha and catalogue_sha256 != meta_cat_sha:
            raise RuntimeError(
                f"BM25 catalogue_sha256 mismatch: index={meta_cat_sha} vs parquet={catalogue_sha256}. "
                "Rebuild search indexes via 05_index_full_catalogue.py."
            )
        meta_rids_sha = meta.get("record_ids_sha256", "")
        actual_rids_sha = compute_record_ids_sha256(df_rids)
        if meta_rids_sha and meta_rids_sha != actual_rids_sha:
            raise RuntimeError(
                f"BM25 record_ids_sha256 mismatch: index={meta_rids_sha} vs catalogue={actual_rids_sha}."
            )
        self._bound_rids = np.asarray(df_rids, dtype=object)
        self._bound_rid_to_idx = {r: i for i, r in enumerate(df_rids)}
        self._db_rids = self._bound_rids
        self._db_rid_to_idx = self._bound_rid_to_idx
        return meta

    def build_from_dataframe(self, df: pd.DataFrame, catalogue_sha256: str = "") -> float:
        t0 = time.perf_counter()
        records = []
        rids_ordered: list[str] = []
        for idx, row in enumerate(df.to_dict(orient="records")):
            t_bm, k_bm, b_bm = build_bm25_fields(row)
            raw_title = _safe_str(row.get("title"))
            raw_desc = _safe_str(row.get("description"))[:3000]
            raw_kw = _safe_str(row.get("keywords"))[:1000]
            rid = str(row["record_id"])
            rids_ordered.append(rid)
            records.append(
                (
                    idx,
                    rid,
                    t_bm,
                    k_bm,
                    b_bm,
                    f"{raw_title} {raw_kw} {raw_desc}".lower(),
                )
            )
        self.con.execute("DROP TABLE IF EXISTS eo_docs")
        self.con.execute("DROP TABLE IF EXISTS eo_meta")
        self.con.execute(
            """
            CREATE TABLE eo_docs (
                doc_idx INTEGER,
                record_id VARCHAR PRIMARY KEY,
                title_bm25 VARCHAR,
                keywords_bm25 VARCHAR,
                body_bm25 VARCHAR,
                raw_lower VARCHAR
            )
            """
        )
        self.con.executemany("INSERT INTO eo_docs VALUES (?, ?, ?, ?, ?, ?)", records)
        self.con.execute(
            """
            PRAGMA create_fts_index(
                'eo_docs', 'record_id', 'title_bm25', 'keywords_bm25', 'body_bm25',
                stemmer='porter', stopwords='english', lower=1, strip_accents=1, overwrite=1
            )
            """
        )
        content_sha = self.compute_content_sha256()
        rids_sha = compute_record_ids_sha256(rids_ordered)
        self.con.execute("CREATE TABLE eo_meta (key VARCHAR PRIMARY KEY, val VARCHAR)")
        self.con.executemany(
            "INSERT INTO eo_meta VALUES (?, ?)",
            [
                ("schema_version", "4.1"),
                ("n_rows", str(len(df))),
                ("catalogue_sha256", catalogue_sha256),
                ("record_ids_sha256", rids_sha),
                ("bm25_content_sha256", content_sha),
            ],
        )
        self._bound_rids = np.asarray(rids_ordered, dtype=object)
        self._bound_rid_to_idx = {r: i for i, r in enumerate(rids_ordered)}
        self._db_rids = self._bound_rids
        self._db_rid_to_idx = self._bound_rid_to_idx
        return time.perf_counter() - t0

    def search_bm25(
        self,
        query: str,
        top_k: int = 200,
        w_title: float = 2.5,
        w_keywords: float = 1.5,
        w_body: float = 1.0,
        k1: float = 1.2,
        b: float = 0.75,
        eligible_mask: Optional[np.ndarray] = None,
        eligible_record_ids: Optional[Iterable[str]] = None,
    ) -> list[tuple[int, float]]:
        q_norm = normalize_eo_text(query)
        if not q_norm:
            return []

        rid_to_idx = self._bound_rid_to_idx if self._bound_rid_to_idx is not None else self._db_rid_to_idx
        rids_arr = self._bound_rids if self._bound_rids is not None else self._db_rids
        if rid_to_idx is None:
            rows = self.con.execute("SELECT record_id FROM eo_docs ORDER BY doc_idx ASC").fetchall()
            rids_list = [str(r[0]) for r in rows]
            rids_arr = np.asarray(rids_list, dtype=object)
            rid_to_idx = {r: i for i, r in enumerate(rids_list)}
            self._db_rids = rids_arr
            self._db_rid_to_idx = rid_to_idx

        elig_rids: Optional[list[str]] = None
        if eligible_record_ids is not None:
            elig_rids = [str(r) for r in eligible_record_ids]
        elif eligible_mask is not None:
            if rids_arr is None or len(eligible_mask) != len(rids_arr):
                raise RuntimeError(
                    f"eligible_mask length {len(eligible_mask)} != BM25 index length {0 if rids_arr is None else len(rids_arr)}"
                )
            elig_rids = [str(r) for r in rids_arr[np.asarray(eligible_mask, dtype=bool)]]

        if elig_rids is not None:
            if not elig_rids:
                return []
            elig_df = pd.DataFrame({"record_id": elig_rids})
            self.con.register("_elig_rids", elig_df)
            try:
                rows = self.con.execute(
                    """
                    SELECT record_id,
                           COALESCE(s_t, 0.0) * ? + COALESCE(s_k, 0.0) * ? + COALESCE(s_b, 0.0) * ? AS score
                    FROM (
                        SELECT e.record_id,
                               fts_main_eo_docs.match_bm25(e.record_id, ?, fields := 'title_bm25', k := ?, b := ?) AS s_t,
                               fts_main_eo_docs.match_bm25(e.record_id, ?, fields := 'keywords_bm25', k := ?, b := ?) AS s_k,
                               fts_main_eo_docs.match_bm25(e.record_id, ?, fields := 'body_bm25', k := ?, b := ?) AS s_b
                        FROM eo_docs e
                        INNER JOIN _elig_rids f ON e.record_id = f.record_id
                    ) t
                    WHERE s_t IS NOT NULL OR s_k IS NOT NULL OR s_b IS NOT NULL
                    ORDER BY ROUND(score, 10) DESC, record_id ASC
                    LIMIT ?
                    """,
                    [w_title, w_keywords, w_body, q_norm, k1, b, q_norm, k1, b, q_norm, k1, b, top_k],
                ).fetchall()
            finally:
                self.con.unregister("_elig_rids")
        else:
            rows = self.con.execute(
                """
                SELECT record_id,
                       COALESCE(s_t, 0.0) * ? + COALESCE(s_k, 0.0) * ? + COALESCE(s_b, 0.0) * ? AS score
                FROM (
                    SELECT record_id,
                           fts_main_eo_docs.match_bm25(record_id, ?, fields := 'title_bm25', k := ?, b := ?) AS s_t,
                           fts_main_eo_docs.match_bm25(record_id, ?, fields := 'keywords_bm25', k := ?, b := ?) AS s_k,
                           fts_main_eo_docs.match_bm25(record_id, ?, fields := 'body_bm25', k := ?, b := ?) AS s_b
                    FROM eo_docs
                ) t
                WHERE s_t IS NOT NULL OR s_k IS NOT NULL OR s_b IS NOT NULL
                ORDER BY ROUND(score, 10) DESC, record_id ASC
                LIMIT ?
                """,
                [w_title, w_keywords, w_body, q_norm, k1, b, q_norm, k1, b, q_norm, k1, b, top_k],
            ).fetchall()

        out: list[tuple[int, float]] = []
        for rid_raw, sc in rows:
            rid = str(rid_raw)
            if rid not in rid_to_idx:
                raise RuntimeError(f"BM25 returned unmapped record_id={rid!r}; index is stale.")
            out.append((int(rid_to_idx[rid]), float(sc)))
        return out

    def search_naive_like(self, query: str, top_k: int = 200) -> list[tuple[int, float]]:
        """Naive plain-string token ILIKE baseline (counts raw substring token matches)."""
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "into", "over", "under",
            "per", "all", "are", "using", "based", "data", "dataset", "product", "products",
            "maps", "map", "monitoring", "assessment", "indicator", "indicators", "change",
            "changes", "global", "european", "europe", "area", "areas", "annual", "monthly",
            "daily", "high", "resolution", "spatial", "temporal", "satellite", "earth",
            "observation", "gridded", "estimates", "surface", "cover",
        }
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3 and t not in stop]
        if not tokens:
            tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
        if not tokens:
            return []
        uniq: list[str] = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
            if len(uniq) >= 8:
                break
        rid_to_idx = self._bound_rid_to_idx if self._bound_rid_to_idx is not None else self._db_rid_to_idx
        if rid_to_idx is None:
            rows_r = self.con.execute("SELECT record_id FROM eo_docs ORDER BY doc_idx ASC").fetchall()
            rid_to_idx = {str(r[0]): i for i, r in enumerate(rows_r)}
        expr = " + ".join([f"(CASE WHEN instr(raw_lower, ?) > 0 THEN 1 ELSE 0 END)" for _ in uniq])
        rows = self.con.execute(
            f"""
            SELECT record_id, ({expr}) AS hits
            FROM eo_docs
            WHERE ({expr}) > 0
            ORDER BY hits DESC, record_id ASC
            LIMIT ?
            """,
            uniq + uniq + [top_k],
        ).fetchall()
        return [(int(rid_to_idx[str(r[0])]), float(r[1])) for r in rows if str(r[0]) in rid_to_idx]


def verify_index_alignment(
    df: pd.DataFrame,
    lex_index: Optional[DuckDBLexicalIndex] = None,
    emb_npz: Optional[Any] = None,
    catalogue_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Fail fast if catalogue row count, ordered `record_id` vector, or `catalogue_sha256` diverges."""
    df_rids = [str(r) for r in df["record_id"]]
    cat_sha = compute_file_sha256(catalogue_path) if catalogue_path is not None and catalogue_path.exists() else ""

    if lex_index is not None:
        lex_index.verify_against_dataframe(df, catalogue_sha256=cat_sha)

    if emb_npz is not None:
        if "record_ids" not in emb_npz:
            raise RuntimeError("Embeddings NPZ missing `record_ids` array; rebuild search indexes.")
        emb_rids = [str(r) for r in emb_npz["record_ids"].tolist()]
        if len(emb_rids) != len(df_rids):
            raise RuntimeError(
                f"Embeddings row count mismatch: npz has {len(emb_rids)} vs catalogue {len(df_rids)}."
            )
        if emb_rids != df_rids:
            mismatch_idx = next((i for i, (a, b) in enumerate(zip(emb_rids, df_rids)) if a != b), 0)
            raise RuntimeError(
                f"Embeddings record_id ordering mismatch at row {mismatch_idx}: "
                f"npz={emb_rids[mismatch_idx]!r} vs catalogue={df_rids[mismatch_idx]!r}."
            )
        if cat_sha and "catalogue_sha256" in emb_npz:
            raw_cat = emb_npz["catalogue_sha256"]
            npz_cat_sha = str(raw_cat.item() if np.ndim(raw_cat) == 0 else raw_cat[0])
            if npz_cat_sha and npz_cat_sha != cat_sha:
                raise RuntimeError(
                    f"Embeddings catalogue_sha256 mismatch: npz={npz_cat_sha} vs parquet={cat_sha}."
                )

    return {
        "n_rows": len(df_rids),
        "record_ids_sha256": compute_record_ids_sha256(df_rids),
        "catalogue_sha256": cat_sha,
    }


# ---------------------------------------------------------------------------
# 4. Null-Safe Hard Filtering (vs Naive Strict Filtering)
# ---------------------------------------------------------------------------


@dataclass
class SearchFilters:
    geo_scope: Optional[str] = None  # "global", "europe", "north_america", "africa", None
    exclude_local: bool = False  # exclude confirmed regional_or_local while keeping global/europe/hemispheric
    max_grid_m: Optional[float] = None
    min_grid_m: Optional[float] = None
    grid_tolerance: float = 1.25  # 30 m requirement matches 30.9 m 1-arc-sec grids
    processing_levels: Optional[list[str]] = None  # e.g. ["L3", "L4"]
    cadence_classes: Optional[list[str]] = None
    strict_eo_only: bool = False
    ontology_mode: str = "all"  # "all" | "eo_and_hybrid" | "strict_eo"
    exclude_superseded: bool = True
    provider_pool: Optional[str] = None  # "copernicus_services", "eu_non_copernicus", "non_eu"
    min_end_year: Optional[int] = None
    max_start_year: Optional[int] = None
    null_safe: bool = True
    hard_limits: bool = False  # Verified mode: null_safe=False, 1.0x grid tolerance, strict global footprint


_REGION_BBOX = {
    "europe": (-15.0, 34.0, 45.0, 72.0),
    "north_america": (-170.0, 15.0, -50.0, 75.0),
    "africa": (-20.0, -36.0, 52.0, 38.0),
}

_OBSOLETE_TITLE_RE = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)


def compute_unknown_metadata_counts(df: pd.DataFrame, active_only: bool = True) -> dict[str, int]:
    """Count records missing structured metadata fields (excluded in Hard Limits / Verified mode)."""
    n = len(df)
    if n == 0:
        return {"total_records": 0, "processing_level": 0, "grid_meters": 0, "geographic_coverage": 0, "cadence": 0}
    mask = np.ones(n, dtype=bool)
    if active_only:
        if "is_superseded_edition" in df.columns:
            mask &= ~df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)
        if "title" in df.columns:
            mask &= ~df["title"].fillna("").str.contains(_OBSOLETE_TITLE_RE, regex=True).to_numpy(dtype=bool, copy=True)

    pl_null = (
        int((mask & (df["processing_level"].isna() | (df["processing_level"].astype(str) == ""))).sum())
        if "processing_level" in df.columns else 0
    )
    g_null = (
        int((mask & pd.to_numeric(df["adjudicated_output_grid_meters"], errors="coerce").isna()).sum())
        if "adjudicated_output_grid_meters" in df.columns else 0
    )
    geo_null = (
        int((mask & (df["geographic_coverage_class"].isna() | (df["geographic_coverage_class"].astype(str) == ""))).sum())
        if "geographic_coverage_class" in df.columns else 0
    )
    cad_null = (
        int((mask & (df["temporal_cadence_class"].isna() | (df["temporal_cadence_class"].astype(str) == ""))).sum())
        if "temporal_cadence_class" in df.columns else 0
    )
    return {
        "total_records": int(mask.sum()),
        "processing_level": pl_null,
        "grid_meters": g_null,
        "geographic_coverage": geo_null,
        "cadence": cad_null,
    }


def filter_benchmark_family_mask(
    df: pd.DataFrame,
    hit_mask: np.ndarray,
    query_filters: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Restrict ground-truth family matches so global queries never credit regional/narrow-strip rows."""
    out = np.asarray(hit_mask, dtype=bool).copy()
    scope = str((query_filters or {}).get("geo_scope") or "").lower()
    if scope == "global":
        if "geographic_coverage_class" in df.columns:
            gc = df["geographic_coverage_class"].fillna("").to_numpy()
            out &= gc != "regional_or_local"
        if "observation_domain_mask" in df.columns:
            odm = df["observation_domain_mask"].fillna("").to_numpy()
            out &= ~np.isin(odm, ["northern_hemisphere_only", "southern_hemisphere_only"])
        if "bbox_south" in df.columns and "bbox_north" in df.columns:
            bs = pd.to_numeric(df["bbox_south"], errors="coerce").to_numpy(dtype=np.float64)
            bn = pd.to_numeric(df["bbox_north"], errors="coerce").to_numpy(dtype=np.float64)
            has_lat = ~(np.isnan(bs) | np.isnan(bn))
            out &= ~has_lat | ((bn - bs) >= 90.0)
    return out


def apply_filters_mask(df: pd.DataFrame, flt: SearchFilters) -> np.ndarray:
    """Return boolean mask of eligible rows respecting null-safety, hard-limits mode & spherical bbox geometry."""
    n = len(df)
    mask = np.ones(n, dtype=bool)
    eff_null_safe = False if flt.hard_limits else flt.null_safe
    eff_grid_tol = 1.0 if flt.hard_limits else flt.grid_tolerance

    if flt.exclude_superseded:
        if "is_superseded_edition" in df.columns:
            sup = df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)
            mask &= ~sup
        if "title" in df.columns:
            obs = df["title"].fillna("").str.contains(_OBSOLETE_TITLE_RE, regex=True).to_numpy(dtype=bool, copy=True)
            mask &= ~obs

    if (flt.strict_eo_only or flt.ontology_mode == "strict_eo") and "strict_eo_c1_eligible" in df.columns:
        elig = df["strict_eo_c1_eligible"].fillna(False).to_numpy(dtype=bool, copy=True)
        mask &= elig
    elif flt.ontology_mode == "eo_and_hybrid" and "product_ontology_class" in df.columns:
        allowed_ont = {
            "eo_thematic_or_biophysical",
            "eo_reanalysis_or_model",
            "eo_constrained_hybrid",
            "eo_ard_surface_reflectance_mosaic",
        }
        ont = df["product_ontology_class"].fillna("").to_numpy()
        mask &= np.isin(ont, list(allowed_ont))

    if flt.provider_pool and "provider_pool" in df.columns:
        pp = df["provider_pool"].fillna("").to_numpy()
        mask &= pp == flt.provider_pool

    if flt.processing_levels and "processing_level" in df.columns:
        allowed = set(flt.processing_levels)
        pl = df["processing_level"].to_numpy()
        is_null = pd.isna(pl) | (pl == "")
        in_allowed = np.array(
            [any(tok.strip() in allowed for tok in str(v).split(";")) if not n_ else False for v, n_ in zip(pl, is_null)],
            dtype=bool,
        )
        mask &= (in_allowed | is_null) if eff_null_safe else in_allowed

    if flt.max_grid_m is not None and "adjudicated_output_grid_meters" in df.columns:
        g = pd.to_numeric(df["adjudicated_output_grid_meters"], errors="coerce").to_numpy()
        # Grid tolerance is orthogonal to null_safe when hard_limits=False so NaiveStrict isolates NULL-dropping!
        thresh = flt.max_grid_m * eff_grid_tol
        is_null = np.isnan(g)
        ok = np.where(is_null, eff_null_safe, g <= thresh)
        mask &= ok

    if flt.min_grid_m is not None and "adjudicated_output_grid_meters" in df.columns:
        g = pd.to_numeric(df["adjudicated_output_grid_meters"], errors="coerce").to_numpy()
        is_null = np.isnan(g)
        ok = np.where(is_null, eff_null_safe, g >= flt.min_grid_m)
        mask &= ok

    if flt.cadence_classes and "temporal_cadence_class" in df.columns:
        allowed = set(flt.cadence_classes)
        cad = df["temporal_cadence_class"].to_numpy()
        is_null = pd.isna(cad) | (cad == "")
        in_allowed = np.isin(cad, list(allowed))
        mask &= (in_allowed | is_null) if eff_null_safe else in_allowed

    if "geographic_coverage_class" in df.columns:
        gc = df["geographic_coverage_class"].to_numpy()
        is_null = pd.isna(gc) | (gc == "")
        if flt.exclude_local:
            not_local = np.isin(gc, ["global", "quasi_global", "hemispheric_or_pantropical", "europe_regional"])
            mask &= (not_local | is_null) if eff_null_safe else not_local
        if flt.geo_scope:
            scope = flt.geo_scope.lower()
            if scope == "global":
                if flt.hard_limits:
                    ok = gc == "global"
                    if "observation_domain_class" in df.columns:
                        od = df["observation_domain_class"].fillna("").to_numpy()
                        ok &= ~np.isin(od, ["hemispheric_or_pantropical", "regional_or_local"])
                    if "observation_domain_mask" in df.columns:
                        odm = df["observation_domain_mask"].fillna("").to_numpy()
                        ok &= ~np.isin(odm, ["northern_hemisphere_only", "southern_hemisphere_only"])
                    if "bbox_south" in df.columns and "bbox_north" in df.columns:
                        bs = pd.to_numeric(df["bbox_south"], errors="coerce").to_numpy(dtype=np.float64)
                        bn = pd.to_numeric(df["bbox_north"], errors="coerce").to_numpy(dtype=np.float64)
                        has_lat = ~(np.isnan(bs) | np.isnan(bn))
                        ok &= ~has_lat | ((bs <= -55.0) & (bn >= 55.0))
                    mask &= ok
                else:
                    # Broad discovery mode includes quasi_global & hemispheric_or_pantropical
                    ok = np.isin(gc, ["global", "quasi_global", "hemispheric_or_pantropical"])
                    mask &= (ok | is_null) if eff_null_safe else ok
            elif scope in _REGION_BBOX:
                bw_min, bs_min, be_max, bn_max = _REGION_BBOX[scope]
                # Global / quasi-global products cover every region; hemispheric/regional require bbox check.
                ok = np.isin(gc, ["global", "quasi_global"])
                if scope == "europe":
                    ok |= gc == "europe_regional"
                if all(c in df.columns for c in ("bbox_west", "bbox_south", "bbox_east", "bbox_north")):
                    bw = pd.to_numeric(df["bbox_west"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    bs = pd.to_numeric(df["bbox_south"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    be = pd.to_numeric(df["bbox_east"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    bn = pd.to_numeric(df["bbox_north"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
                    has_bbox = ~(np.isnan(bw) | np.isnan(bs) | np.isnan(be) | np.isnan(bn))
                    # Clamp minor floating-point latitude fuzz outside [-90, 90] (e.g., GAEZ v5 metadata)
                    bs = np.where(has_bbox, np.clip(bs, -90.0, 90.0), bs)
                    bn = np.where(has_bbox, np.clip(bn, -90.0, 90.0), bn)
                    # Normalize out-of-range longitudes (e.g., 0..360° -> -180..180°) while preserving full-span 360° boxes
                    span = be - bw
                    full_globe = has_bbox & (span >= 359.5)
                    bw = np.where(bw > 180.0, ((bw + 180.0) % 360.0) - 180.0, np.where(bw < -180.0, ((bw + 180.0) % 360.0) - 180.0, bw))
                    be = np.where(be > 180.0, ((be + 180.0) % 360.0) - 180.0, np.where(be < -180.0, ((be + 180.0) % 360.0) - 180.0, be))
                    # Antimeridian-safe longitude overlap (handles bw > be wrap-around)
                    lon_ok = full_globe | np.where(
                        bw <= be,
                        (be >= bw_min) & (bw <= be_max),
                        (be_max >= bw) | (bw_min <= be),
                    )
                    intersects = has_bbox & lon_ok & (bn >= bs_min) & (bs <= bn_max)
                    bbox_classes = np.isin(gc, ["regional_or_local", "hemispheric_or_pantropical"])
                    reg_ok = bbox_classes & (intersects | (~has_bbox & eff_null_safe))
                    ok |= reg_ok
                mask &= (ok | is_null) if eff_null_safe else ok

    if flt.min_end_year is not None and "temporal_end_year" in df.columns:
        ey = pd.to_numeric(df["temporal_end_year"], errors="coerce").to_numpy()
        ongoing = (
            df["is_ongoing_archive"].fillna(False).to_numpy(dtype=bool, copy=True)
            if "is_ongoing_archive" in df.columns
            else np.zeros(n, dtype=bool)
        )
        is_null = np.isnan(ey)
        ok = ongoing | np.where(is_null, eff_null_safe, ey >= flt.min_end_year)
        mask &= ok

    if flt.max_start_year is not None and "temporal_start_year" in df.columns:
        sy = pd.to_numeric(df["temporal_start_year"], errors="coerce").to_numpy()
        is_null = np.isnan(sy)
        ok = np.where(is_null, eff_null_safe, sy <= flt.max_start_year)
        mask &= ok

    return mask


def deterministic_topk(
    scores: np.ndarray,
    k: int = 300,
    mask: Optional[np.ndarray] = None,
) -> list[tuple[int, float]]:
    """Bit-deterministic top-K selection breaking score ties strictly by ascending doc_idx."""
    s = np.where(mask, scores, -1e9) if mask is not None else np.asarray(scores, dtype=np.float32)
    valid_idx = np.flatnonzero(s > -1e8)
    if len(valid_idx) == 0:
        return []
    take = min(k, len(valid_idx))
    vs = s[valid_idx]
    order = np.lexsort((valid_idx, -vs))[:take]
    chosen = valid_idx[order]
    return [(int(i), float(s[i])) for i in chosen]


# ---------------------------------------------------------------------------
# 5. Reciprocal Rank Fusion (RRF) & Product-Family Mirror Collapse
# ---------------------------------------------------------------------------


def rrf_fuse(
    dense_ranked: list[tuple[int, float]],
    bm25_ranked: list[tuple[int, float]],
    rrf_k: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 0.6,
) -> list[tuple[int, float, dict[str, Any]]]:
    """Fuse dense and lexical rankings via Weighted Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    meta: dict[int, dict[str, Any]] = {}

    for rank, (doc_idx, d_score) in enumerate(dense_ranked, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + w_dense / (rrf_k + rank)
        meta.setdefault(doc_idx, {})["dense_rank"] = rank
        meta[doc_idx]["dense_score"] = round(float(d_score), 5)

    for rank, (doc_idx, b_score) in enumerate(bm25_ranked, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + w_bm25 / (rrf_k + rank)
        meta.setdefault(doc_idx, {})["bm25_rank"] = rank
        meta[doc_idx]["bm25_score"] = round(float(b_score), 5)

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc_idx, float(sc), meta[doc_idx]) for doc_idx, sc in fused]


_VERSION_YEAR_STRIP_RE = re.compile(
    r"\b(?:19\d{2}|20\d{2})(?:\s*[-–/]\s*(?:19\d{2}|20\d{2}|present))?\b|"
    r"\b(?:v|ver|version|r|rel|release|edition|ed|collection|coll)[\s._-]*\d+(?:\.\d+)*[a-z]?\b|"
    r"\((?:raster|vector|cog|geotiff|netcdf|hdf\d?|csv|shapefile)[^)]*\)|"
    r"\b\d+(?:\.\d+)?\s*(?:m|km|deg|degree|degrees|arc-?sec(?:ond)?s?|min)\b|"
    r"\b(?:nrt|near\s+real\s+time|provisional|validated|reprocessed|interim|"
    r"status|change|product|products|dataset|data\s+set|map|maps|layer|layers|grid|gridded|series)\b",
    re.IGNORECASE,
)


_LEADING_CODE_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,45}:\s+")

# Physical orbital instrument/platform tokens used ONLY as a conflict guard during family
# collapsing (two records explicitly naming distinct physical sensors are never mirrors of each other).
# Uses underscore-safe alphanumeric boundaries `(?<![a-z0-9])...(?![a-z0-9])` so underscore-delimited IDs/titles match!
_SENSOR_CONFLICT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("modis", re.compile(r"(?<![a-z0-9])(?:modis|mod\d{2}[a-z]|myd\d{2}[a-z]|mcd\d{2}[a-z])(?![a-z0-9])", re.I)),
    ("viirs", re.compile(r"(?<![a-z0-9])(?:viirs|vnp\d{2}[a-z0-9]*|vj1\d{2}[a-z0-9]*|vj2\d{2}[a-z0-9]*)(?![a-z0-9])", re.I)),
    ("seawifs", re.compile(r"(?<![a-z0-9])(?:seawifs|orbview-?2)(?![a-z0-9])", re.I)),
    ("olci", re.compile(r"(?<![a-z0-9])(?:olci|sentinel-?3)(?![a-z0-9])", re.I)),
    ("meris", re.compile(r"(?<![a-z0-9])(?:meris|envisat)(?![a-z0-9])", re.I)),
    ("pace", re.compile(r"(?<![a-z0-9])pace(?![a-z0-9]).*(?<![a-z0-9])oci(?![a-z0-9])|(?<![a-z0-9])pace_oci(?![a-z0-9])", re.I)),
    ("landsat", re.compile(r"(?<![a-z0-9])landsat(?![a-z0-9])", re.I)),
    ("sentinel1", re.compile(r"(?<![a-z0-9])(?:sentinel-?1[a-d]?|s1[a-d])(?![a-z0-9])", re.I)),
    ("sentinel2", re.compile(r"(?<![a-z0-9])(?:sentinel-?2[a-d]?|s2[a-d])(?![a-z0-9])", re.I)),
    ("sentinel5p", re.compile(r"(?<![a-z0-9])(?:sentinel-?5p?|tropomi)(?![a-z0-9])", re.I)),
    ("omi", re.compile(r"(?<![a-z0-9])omi(?![a-z0-9])", re.I)),
    ("tempo", re.compile(r"(?<![a-z0-9])tempo(?![a-z0-9])", re.I)),
    ("avhrr", re.compile(r"(?<![a-z0-9])avhrr(?![a-z0-9])", re.I)),
    ("icesat", re.compile(r"(?<![a-z0-9])icesat-?2?(?![a-z0-9])", re.I)),
    ("cryosat", re.compile(r"(?<![a-z0-9])cryosat(?![a-z0-9])", re.I)),
    ("grace", re.compile(r"(?<![a-z0-9])grace(?:-?fo)?(?![a-z0-9])", re.I)),
    ("smap", re.compile(r"(?<![a-z0-9])smap(?![a-z0-9])", re.I)),
    ("smos", re.compile(r"(?<![a-z0-9])smos(?![a-z0-9])", re.I)),
    ("amsr", re.compile(r"(?<![a-z0-9])amsr-?[e2](?![a-z0-9])", re.I)),
    ("gpm", re.compile(r"(?<![a-z0-9])(?:gpm|imerg)(?![a-z0-9])", re.I)),
    ("trmm", re.compile(r"(?<![a-z0-9])trmm(?![a-z0-9])", re.I)),
    ("airs", re.compile(r"(?<![a-z0-9])airs(?![a-z0-9])", re.I)),
    ("iasi", re.compile(r"(?<![a-z0-9])iasi(?![a-z0-9])", re.I)),
    ("palsar", re.compile(r"(?<![a-z0-9])(?:palsar-?2?|alos-?2?)(?![a-z0-9])", re.I)),
    ("srtm", re.compile(r"(?<![a-z0-9])srtm(?![a-z0-9])", re.I)),
    ("aster", re.compile(r"(?<![a-z0-9])aster(?![a-z0-9])", re.I)),
    ("scatsat", re.compile(r"(?<![a-z0-9])scatsat-?1?(?![a-z0-9])", re.I)),
    ("ascat", re.compile(r"(?<![a-z0-9])ascat(?![a-z0-9])", re.I)),
    ("quikscat", re.compile(r"(?<![a-z0-9])quikscat(?![a-z0-9])", re.I)),
]


def _extract_sensor_set(title: str, raw_id: str = "", platform_sensor: str = "") -> frozenset[str]:
    parts = [p for p in (_safe_str(title), _safe_str(raw_id), _safe_str(platform_sensor)[:400]) if p]
    if not parts:
        return frozenset()
    text = " | ".join(parts)
    return frozenset(name for name, pat in _SENSOR_CONFLICT_PATTERNS if pat.search(text))


def _grid_bucket_tag(grid_meters: Optional[float]) -> str:
    if grid_meters is None:
        return ""
    try:
        g = float(grid_meters)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(g) or g <= 0.0:
        return ""
    return f"grid:b{int(round(math.log10(g) * 4.0))}"


def compute_generic_family_signature(
    title: str,
    canonical_key: str = "",
    cadence_class: str = "",
    grid_meters: Optional[float] = None,
    geo_class: str = "",
) -> str:
    """100% generic, zero-hardcoding product-family signature separated by cadence/grid/geo."""
    t = _safe_str(title).translate(_SUBSCRIPT_MAP)
    t_clean = _LEADING_CODE_PREFIX_RE.sub("", t).strip()
    t_clean = re.sub(r"\([^)]{1,45}\)\s*$", "", t_clean).strip()
    stem = _VERSION_YEAR_STRIP_RE.sub(" ", t_clean.lower())
    stem = re.sub(r"[^a-z0-9]+", " ", stem).strip()
    stop = {"the", "of", "for", "from", "in", "on", "to", "and", "by", "at", "with", "derived", "based"}
    tokens = [w for w in stem.split() if len(w) > 1 and w not in stop][:6]
    base = "gstem:" + "_".join(tokens) if len(tokens) >= 2 else "key:" + (_safe_str(canonical_key) or stem or t.lower())
    qualifiers: list[str] = []
    cad = _safe_str(cadence_class)
    if cad:
        qualifiers.append(f"cad:{cad}")
    gb = _grid_bucket_tag(grid_meters)
    if gb:
        qualifiers.append(gb)
    geo = _safe_str(geo_class)
    if geo:
        qualifiers.append(f"geo:{geo}")
    return f"{base}|{'|'.join(qualifiers)}" if qualifiers else base


def compute_family_signature(
    title: str,
    raw_id: str = "",
    canonical_key: str = "",
    cadence_class: str = "",
    grid_meters: Optional[float] = None,
    geo_class: str = "",
) -> str:
    """100% generic, domain-agnostic product-family signature (zero benchmark-specific regexes)."""
    _ = raw_id
    return compute_generic_family_signature(
        title,
        canonical_key=canonical_key,
        cadence_class=cadence_class,
        grid_meters=grid_meters,
        geo_class=geo_class,
    )


_Q_RES_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:-\s*)?(meter|meters|m|kilometer|kilometers|km|deg|degree|degrees)\b", re.I)
_Q_CAD_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:hourly|sub-?hourly|sub-?daily|\d+-?hourly)\b", re.I), "hourly"),
    (re.compile(r"\bdaily\b", re.I), "daily"),
    (re.compile(r"\b(?:weekly|8-?day|10-?daily|16-?day|pentad|dekadal|semi-?monthly|half-?monthly)\b", re.I), "multi_day"),
    (re.compile(r"\b(?:monthly|seasonal)\b", re.I), "monthly"),
    (re.compile(r"\b(?:annual|yearly|multi-?year|climatolog)\b", re.I), "annual"),
]
_Q_GEO_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bglobal\b", re.I), "global"),
    (re.compile(r"\b(?:europe|european|pan-?european)\b", re.I), "europe"),
    (re.compile(r"\bafrica\b", re.I), "africa"),
    (re.compile(r"\bnorth\s+america\b", re.I), "north_america"),
]


def _extract_query_variant_prefs(query_text: str) -> tuple[list[float], list[str], list[str]]:
    if not query_text:
        return [], [], []
    res_targets: list[float] = []
    for num_s, unit_s in _Q_RES_RE.findall(query_text):
        v = float(num_s)
        u = unit_s.lower()
        if u in ("km", "kilometer", "kilometers"):
            v *= 1000.0
        elif u in ("deg", "degree", "degrees"):
            v *= 111320.0
        if v > 0:
            res_targets.append(v)
    cad_prefs = [lbl for pat, lbl in _Q_CAD_MAP if pat.search(query_text)]
    geo_prefs = [lbl for pat, lbl in _Q_GEO_MAP if pat.search(query_text)]
    return res_targets, cad_prefs, geo_prefs


def _variant_query_match_score(
    res_targets: list[float],
    cad_prefs: list[str],
    geo_prefs: list[str],
    title: str,
    cad: str,
    grid_m: float,
    geo: str,
) -> int:
    score = 0
    t_low = title.lower()
    for target_m in res_targets:
        if math.isfinite(grid_m) and grid_m > 0 and max(grid_m, target_m) / min(grid_m, target_m) <= 1.25:
            score += 2
    for cp in cad_prefs:
        if (
            (cp == "hourly" and "hourly" in cad)
            or (cp == "daily" and "daily" in cad)
            or (cp == "multi_day" and ("multi_day" in cad or "weekly" in cad))
            or (cp == "monthly" and "monthly" in cad)
            or (cp == "annual" and ("annual" in cad or "climatolog" in cad or "multi_year" in cad))
            or re.search(rf"\b{cp}\b", t_low)
        ):
            score += 2
    for gp in geo_prefs:
        if (
            (gp == "global" and geo == "global")
            or (gp == "europe" and (geo == "europe_regional" or "europe" in t_low))
            or (gp in t_low)
        ):
            score += 2
    return score


def collapse_product_families(
    ranked_items: list[tuple[int, float, dict[str, Any]]],
    df: pd.DataFrame,
    top_k: int = 20,
    family_sigs: Optional[list[str]] = None,
    vectors: Optional[np.ndarray] = None,
    near_dup_cos: float = 0.94,
    sensor_guard: bool = True,
    query_text: str = "",
) -> list[tuple[int, float, dict[str, Any]]]:
    """Collapse sibling editions & unlinked catalogue mirrors into single family representatives.

    Enforces distinct-DOI, distinct-cadence, distinct-geographic-coverage, >2x resolution, and
    physical-sensor guards so distinct products are never merged. When `query_text` is provided,
    promotes any collapsed cluster sibling matching explicit resolution/cadence/scope tokens.
    """
    cache = df.attrs.get("_eo_collapse_cache_v41")
    if cache is None or cache[0] != len(df):
        titles_c = df["title"].fillna("").astype(str).tolist()
        rids_c = df["record_id"].fillna("").astype(str).tolist()
        srcs_c = df["source_catalogue"].fillna("").astype(str).tolist()
        raw_ids_c = df["raw_id"].fillna("").astype(str).tolist() if "raw_id" in df.columns else [""] * len(df)
        psens_c = df["platform_sensor_raw"].fillna("").astype(str).tolist() if "platform_sensor_raw" in df.columns else [""] * len(df)
        dois_c = df["doi"].fillna("").astype(str).str.lower().str.strip().tolist() if "doi" in df.columns else [""] * len(df)
        cads_c = df["temporal_cadence_class"].fillna("").astype(str).tolist() if "temporal_cadence_class" in df.columns else [""] * len(df)
        geos_c = df["geographic_coverage_class"].fillna("").astype(str).tolist() if "geographic_coverage_class" in df.columns else [""] * len(df)
        grids_c = (
            pd.to_numeric(df["adjudicated_output_grid_meters"], errors="coerce").to_numpy(dtype=np.float64)
            if "adjudicated_output_grid_meters" in df.columns
            else np.full(len(df), np.nan, dtype=np.float64)
        )
        df.attrs["_eo_collapse_cache_v41"] = (len(df), titles_c, rids_c, srcs_c, raw_ids_c, psens_c, dois_c, cads_c, geos_c, grids_c)
    else:
        _, titles_c, rids_c, srcs_c, raw_ids_c, psens_c, dois_c, cads_c, geos_c, grids_c = cache

    if family_sigs is None:
        ckeys = df["canonical_product_key"].fillna("").astype(str).tolist() if "canonical_product_key" in df.columns else [""] * len(df)
        family_sigs = [
            compute_generic_family_signature(t, c, cad, float(g) if np.isfinite(g) else None, geo)
            for t, c, cad, g, geo in zip(titles_c, ckeys, cads_c, grids_c, geos_c)
        ]

    kept: list[tuple[int, float, dict[str, Any]]] = []
    seen_sig_to_positions: dict[str, list[int]] = {}
    kept_vecs: list[np.ndarray] = []
    kept_sensors: list[frozenset[str]] = []
    scan_limit = min(len(ranked_items), max(top_k * 3, 200))

    def _pair_conflicts(cpos: int, cur_sensors: Optional[frozenset[str]], cur_doi: str, cur_cad: str, cur_geo: str, cur_grid: float) -> bool:
        k_idx = kept[cpos][0]
        if sensor_guard and cur_sensors and kept_sensors[cpos] and cur_sensors.isdisjoint(kept_sensors[cpos]):
            return True
        k_doi = dois_c[k_idx]
        if cur_doi and k_doi and cur_doi != k_doi:
            return True
        k_cad = cads_c[k_idx]
        if cur_cad and k_cad and cur_cad != k_cad:
            return True
        k_geo = geos_c[k_idx]
        if cur_geo and k_geo and cur_geo != k_geo:
            return True
        k_grid = float(grids_c[k_idx])
        if np.isfinite(cur_grid) and np.isfinite(k_grid) and cur_grid > 0.0 and k_grid > 0.0:
            if max(cur_grid, k_grid) / min(cur_grid, k_grid) > 2.0:
                return True
        return False

    for pos_i, (doc_idx, score, meta) in enumerate(ranked_items):
        sig = family_sigs[doc_idx]
        cur_sensors: Optional[frozenset[str]] = (
            _extract_sensor_set(titles_c[doc_idx], raw_ids_c[doc_idx], psens_c[doc_idx]) if sensor_guard else None
        )
        cur_doi = dois_c[doc_idx]
        cur_cad = cads_c[doc_idx]
        cur_geo = geos_c[doc_idx]
        cur_grid = float(grids_c[doc_idx])
        matched_pos: Optional[int] = None

        # 1. Exact generic stem match (with DOI/cadence/geo/resolution/sensor guards)
        for cpos in seen_sig_to_positions.get(sig, ()):
            if _pair_conflicts(cpos, cur_sensors, cur_doi, cur_cad, cur_geo, cur_grid):
                continue
            matched_pos = cpos
            break

        # 2. Generic high-cosine mirror catch (with DOI/cadence/geo/resolution/sensor guards)
        if matched_pos is None and vectors is not None and kept_vecs and near_dup_cos < 1.0:
            v = vectors[doc_idx]
            sims = np.dot(np.asarray(kept_vecs), v)
            above = np.flatnonzero(sims >= near_dup_cos)
            if len(above) > 0:
                order = above[np.lexsort((above, -sims[above]))]
                for cand_pos in order:
                    cpos = int(cand_pos)
                    if _pair_conflicts(cpos, cur_sensors, cur_doi, cur_cad, cur_geo, cur_grid):
                        continue
                    matched_pos = cpos
                    break

        if matched_pos is not None:
            rep_meta = kept[matched_pos][2]
            sibs = rep_meta.setdefault("collapsed_siblings", [])
            sibs.append({"record_id": rids_c[doc_idx], "source": srcs_c[doc_idx], "doc_idx": doc_idx})
            rep_meta["sibling_count"] = len(sibs)
            if len(kept) >= top_k and pos_i + 1 >= scan_limit:
                break
            continue

        if len(kept) >= top_k:
            if pos_i + 1 >= scan_limit:
                break
            continue

        new_meta = dict(meta)
        new_meta["family_signature"] = sig
        new_meta["sibling_count"] = 0
        new_meta["collapsed_siblings"] = []
        seen_sig_to_positions.setdefault(sig, []).append(len(kept))
        kept.append((doc_idx, score, new_meta))
        if vectors is not None:
            kept_vecs.append(vectors[doc_idx])
        if sensor_guard:
            kept_sensors.append(cur_sensors if cur_sensors is not None else frozenset())

    res_targets, cad_prefs, geo_prefs = _extract_query_variant_prefs(query_text)
    if res_targets or cad_prefs or geo_prefs:
        promoted: list[tuple[int, float, dict[str, Any]]] = []
        for rep_idx, rep_sc, rep_meta in kept:
            sibs = rep_meta.get("collapsed_siblings", [])
            if not sibs:
                promoted.append((rep_idx, rep_sc, rep_meta))
                continue
            rep_q = _variant_query_match_score(
                res_targets, cad_prefs, geo_prefs,
                titles_c[rep_idx], cads_c[rep_idx], float(grids_c[rep_idx]), geos_c[rep_idx],
            )
            best_sib_pos = -1
            best_q = rep_q
            for s_pos, sib in enumerate(sibs):
                s_idx = int(sib["doc_idx"])
                sq = _variant_query_match_score(
                    res_targets, cad_prefs, geo_prefs,
                    titles_c[s_idx], cads_c[s_idx], float(grids_c[s_idx]), geos_c[s_idx],
                )
                if sq > best_q:
                    best_q = sq
                    best_sib_pos = s_pos
            if best_sib_pos >= 0:
                chosen_idx = int(sibs[best_sib_pos]["doc_idx"])
                new_sibs = [
                    {"record_id": rids_c[rep_idx], "source": srcs_c[rep_idx], "doc_idx": rep_idx}
                ] + [s for j, s in enumerate(sibs) if j != best_sib_pos]
                upd_meta = dict(rep_meta)
                upd_meta["family_signature"] = family_sigs[chosen_idx]
                upd_meta["collapsed_siblings"] = new_sibs
                upd_meta["sibling_count"] = len(new_sibs)
                promoted.append((chosen_idx, rep_sc, upd_meta))
            else:
                promoted.append((rep_idx, rep_sc, rep_meta))
        kept = promoted

    return kept[:top_k]


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni family-wise error rate adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = round(running_max, 5)
    return adj
