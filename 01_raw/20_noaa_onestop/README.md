# `20_noaa_onestop` — NOAA OneStop Data Search REST API (NOAA CDRs, NESDIS/STAR/NCEI Gridded Satellite Products)

## Overview
- **Source ID**: `20_noaa_onestop`
- **Authoritative Endpoint**: `POST https://data.noaa.gov/onestop/api/search/search/collection`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries NOAA's official OneStop Elasticsearch REST API (`POST https://data.noaa.gov/onestop/api/search/search/collection`) with `"summary": false` (retrieving full collection descriptions, GCMD science keywords, platforms/sensors, spatial MultiPolygons, DOIs, and typed links) and dual-sort pagination (`stagedDate asc` + `stagedDate desc`) to traverse queries exceeding the 10,000-record Elasticsearch result window without truncation across 15,165 unique NOAA Climate Data Records (CDRs), Coral Reef Watch, OISST, VIIRS/AVHRR/GOES/JPSS Level 3 & Level 4 satellite products, and NCEI global bathymetries/climatologies (`noaa_onestop_collections_part01..11.jsonl`, <22 MB each).
