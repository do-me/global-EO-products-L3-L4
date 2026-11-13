# `24_openlandmap_stac` — OpenGeoHub OpenLandMap Global Soil, Land Cover & Climate STAC Catalogue

## Overview
- **Source ID**: `24_openlandmap_stac`
- **Authoritative Endpoint**: `GET https://s3.eu-central-1.wasabisys.com/stac/openlandmap/catalog.json + 104 child collection JSONs`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests all 104 global high-resolution soil, land cover, vegetation, and climate collections from OpenGeoHub Foundation (Wageningen, NL). Native OpenGeoHub / OpenLandMap products are attributed by **producing institution** (`eu_non_copernicus`) regardless of secondary cloud hosting, while mirrored third-party datasets (e.g., UMD GLAD, MODIS, GLC_FCS30D) retain their non-EU producer attribution.
