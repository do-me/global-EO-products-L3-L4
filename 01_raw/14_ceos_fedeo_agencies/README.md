# `14_ceos_fedeo_agencies` — CEOS FedEO Gateway — ESA CCI (CEDA), ESA Earth Online, ESA MAAP, VITO Terrascope, JAXA, ISRO, DLR, INPE, NRCan & USGS

## Overview
- **Source ID**: `14_ceos_fedeo_agencies`
- **Authoritative Endpoint**: `GET https://fedeo.ceos.org/collections?parentIdentifier=<agency> + GET https://api.stac.ceda.ac.uk/collections + GET https://catalog.maap.eo.esa.int/catalogue/collections + GET https://api.datacite.org/dois?prefix=10.5285`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the ESA-operated CEOS Federated Earth Observation (FedEO) STAC API (`https://fedeo.ceos.org/collections`) across 12 primary space agency collection catalogues, plus the native UK CEDA STAC API (`api.stac.ceda.ac.uk`), the paginated native ESA MAAP STAC API (`catalog.maap.eo.esa.int/catalogue/collections`), and the DataCite CEDA prefix crosswalk (`data/ceda_datacite_doi_crosswalk.json`, resolving `200 / 399` CEDA UUIDs—including all 7 ESA Snow CCI releases—to exact `10.5285/<uuid>` primary DOIs so downstream `rel="describedby"` paper/input-radiance DOIs are never promoted into `doi`; `2,817` total retrieved entries in `last_access.txt`). `EOP:STFC:CEDA-CCI` on FedEO exposes **399 ESA Climate Change Initiative (CCI)** collections alongside `EOP:ESA:EARTH-ONLINE` (**162** collections) and native ESA MAAP (**301** collections). Note: `EOP:ESA:ECV` (532 items) is intentionally excluded because it indexes publication abstracts rather than datasets.
