# `03_copernicus_c3s_cds` — Copernicus Climate Change Service (C3S) Climate Data Store (CDS) CADS STAC API

## Overview
- **Source ID**: `03_copernicus_c3s_cds`
- **Authoritative Endpoint**: `GET https://cds.climate.copernicus.eu/api/catalogue/v1/datasets?limit=500`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the official ECMWF CADS v1 STAC REST API directly instead of brittle Next.js `_next/data/<build_id>` frontend URLs. Returns full STAC Collections including spatial/temporal extents, licences, DOIs, and keywords.
