# `02_copernicus_clms` — Copernicus Land Monitoring Service (CLMS) Plone REST API (Catalog + Full Product Detail JSONs)

## Overview
- **Source ID**: `02_copernicus_clms`
- **Authoritative Endpoint**: `GET https://land.copernicus.eu/++api++/en/dataset-catalog/@querystring-search + item ++api++ detail endpoints`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
A two-stage authoritative harvest: first retrieves the full CLMS `@querystring-search` index (~338 items), then fetches each product's individual Plone REST `++api++` JSON endpoint (`https://land.copernicus.eu/++api++/en/products/...`). The second stage recovers ~95 technical ISO-19115 metadata fields omitted from the search summary, including `characteristics_spatial_resolution`, `characteristics_temporal_extent`, `characteristics_update_frequency`, `temporalExtentStart`, `temporalExtentEnd`, `geographicBoundingBox`, and `citation`.
