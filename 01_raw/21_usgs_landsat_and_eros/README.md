# `21_usgs_landsat_and_eros` — USGS LandsatLook STAC Server & USGS/EROS Collection Metadata

## Overview
- **Source ID**: `21_usgs_landsat_and_eros`
- **Authoritative Endpoint**: `GET https://landsatlook.usgs.gov/stac-server/collections + CMR USGS providers`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Because `sciencebase.gov` applies interactive Cloudflare Turnstile challenges to automated scripts, USGS Earth Observation collections are harvested via the official unauthenticated USGS LandsatLook STAC Server (`landsatlook.usgs.gov/stac-server/collections`) and the USGS EROS/LTA provider feeds in NASA CMR (`USGS_EROS`, `USGS_LTA`, `LPDAAC_ECS`).
