# `05_copernicus_cems_ewds` — Copernicus Emergency Management Service (CEMS) Early Warning Data Store (EWDS) CADS STAC API

## Overview
- **Source ID**: `05_copernicus_cems_ewds`
- **Authoritative Endpoint**: `GET https://ewds.climate.copernicus.eu/api/catalogue/v1/datasets?limit=500`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the ECMWF-hosted Early Warning Data Store (EWDS) CADS v1 STAC REST API (12 collections covering GloFAS hydrological river discharge reanalysis/forecasts/seasonal, EFAS European floods, and CEMS Fire Danger Indices). Note: JRC/EODC-hosted CEMS pillars (GFM, GHSL, EDO/GDO, EFFIS/GWIS) are mined in sources 06, 07, and 08.
