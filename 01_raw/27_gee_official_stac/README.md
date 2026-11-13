# `27_gee_official_stac` — Google Earth Engine (GEE) Official STAC Catalogue (Full Tree Traversal)

## Overview
- **Source ID**: `27_gee_official_stac`
- **Authoritative Endpoint**: `GET https://earthengine-stac.storage.googleapis.com/catalog/catalog.json + all provider & dataset STAC JSONs`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Traverses the official Google Cloud Storage STAC tree (`earthengine-stac.storage.googleapis.com/catalog/catalog.json`) across all 133 provider sub-catalogs and concurrently downloads every dataset STAC Collection JSON (`1,182` collections). Includes full `gee:terms_of_use`, `sci:citation`, `summaries` (bands & `gsd` in meters), and temporal/spatial extents. Used in `03_harmonization` both as a cross-catalogue discovery layer and to preserve hosting-vs-producer institutional attribution.
