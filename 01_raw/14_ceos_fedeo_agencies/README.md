# `14_ceos_fedeo_agencies` — CEOS FedEO Gateway — ESA CCI (CEDA), ESA Earth Online, ESA MAAP, VITO Terrascope, JAXA, ISRO, DLR, INPE, NRCan & USGS

## Overview
- **Source ID**: `14_ceos_fedeo_agencies`
- **Authoritative Endpoint**: `GET https://fedeo.ceos.org/collections?parentIdentifier=<agency> + GET https://api.stac.ceda.ac.uk/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the ESA-operated CEOS Federated Earth Observation (FedEO) STAC API (`https://fedeo.ceos.org/collections`) across 12 primary space agency collection catalogues, plus the native UK CEDA STAC API (`api.stac.ceda.ac.uk`). Critical discovery: `EOP:STFC:CEDA-CCI` on FedEO exposes all **399 ESA Climate Change Initiative (CCI)** collections (compared to only 29 migrated collections on `api.stac.ceda.ac.uk`), and `EOP:ESA:EARTH-ONLINE` exposes 162 ESA collections without timeout. Note: `EOP:ESA:ECV` (532 items) is intentionally excluded because it indexes publication abstracts rather than datasets.
