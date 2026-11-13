# `12_ceos_cgms_wgclimate_ecv` — Joint CEOS/CGMS Working Group on Climate (WGClimate) ECV Climate Data Record (CDR) Inventory API

## Overview
- **Source ID**: `12_ceos_cgms_wgclimate_ecv`
- **Authoritative Endpoint**: `GET https://www.cdrinventory.climatemonitoring.info/backend/api/v1/data_records/ + /variables/nodes/graph/`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Operated by EUMETSAT on behalf of CEOS and CGMS WGClimate, providing 821 official space-agency Essential Climate Variable (ECV) Climate Data Record (CDR) entries across EUMETSAT, ESA, Copernicus, NASA, NOAA, JAXA, CMA, and ISRO, together with GCOS ECV target variable taxonomy.
