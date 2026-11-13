# `01_copernicus_cmems` — Copernicus Marine Service (CMEMS) Marine Data Store API (Detailed Variant)

## Overview
- **Source ID**: `01_copernicus_cmems`
- **Authoritative Endpoint**: `POST https://data-be-prd.marine.copernicus.eu/api/datasets (variant=detailed)`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries `POST https://data-be-prd.marine.copernicus.eu/api/datasets` with `variant='detailed'` rather than `variant='summary'`. This unlocks all 44 native metadata fields per product (including `abstract`, `doi`, `updateFrequencies`, `geoResolution`, `tempResolutions`, `geoExtent`, `assimilatedData`, `originatingCenter`, and `layers`). Raw dataset includes Level 3, Level 4, 13 Level 2 swath products, and unlabelled numerical models/in-situ series; Level 2 and pure in-situ station logs are deterministically filtered out in `02_preparation`.
