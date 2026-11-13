# `04_copernicus_cams_ads` — Copernicus Atmosphere Monitoring Service (CAMS) Atmosphere Data Store (ADS) CADS STAC API

## Overview
- **Source ID**: `04_copernicus_cams_ads`
- **Authoritative Endpoint**: `GET https://ads.atmosphere.copernicus.eu/api/catalogue/v1/datasets?limit=500`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the official ECMWF CADS v1 STAC REST API for CAMS ADS directly, avoiding Next.js build-hash breakage.
