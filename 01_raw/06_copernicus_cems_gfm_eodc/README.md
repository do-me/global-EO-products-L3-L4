# `06_copernicus_cems_gfm_eodc` — EODC STAC API — CEMS Global Flood Monitoring (GFM), Copernicus DEM & Sentinel Thematic Products

## Overview
- **Source ID**: `06_copernicus_cems_gfm_eodc`
- **Authoritative Endpoint**: `GET https://stac.eodc.eu/api/v1/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the Earth Observation Data Centre (EODC) STAC v1 API (`https://stac.eodc.eu/api/v1/collections`), which is the official consortium dissemination backend for **CEMS Global Flood Monitoring (`GFM`)** (20 m Sentinel-1 SAR near-real-time global flood extent, reference water mask, and exclusion mask, operational since 2021), as well as `COP_DEM` and Sentinel-1/2/3 harmonic backscatter and soil moisture collections.
