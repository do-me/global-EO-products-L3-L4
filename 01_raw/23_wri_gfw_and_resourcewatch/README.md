# `23_wri_gfw_and_resourcewatch` — World Resources Institute (WRI) — Global Forest Watch (GFW) & Resource Watch Data APIs

## Overview
- **Source ID**: `23_wri_gfw_and_resourcewatch`
- **Authoritative Endpoint**: `GET https://data-api.globalforestwatch.org/datasets + GET https://api.resourcewatch.org/v1/dataset`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests WRI's Global Forest Watch Data API (`data-api.globalforestwatch.org/datasets`, 382 datasets including UMD/GLAD forest loss alerts, primary forests, forest carbon fluxes, and Land & Carbon Lab layers) and WRI's Resource Watch API (`api.resourcewatch.org/v1/dataset`, environmental/climate/water/energy datasets).
