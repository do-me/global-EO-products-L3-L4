# `19_dlr_eoc_stac` — DLR Earth Observation Center (EOC) Geoservice STAC API

## Overview
- **Source ID**: `19_dlr_eoc_stac`
- **Authoritative Endpoint**: `GET https://geoservice.dlr.de/eoc/ogc/stac/v1/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Retrieves all 70 collections from the German Aerospace Center (DLR) EOC Geoservice STAC v1 API, including the World Settlement Footprint suite (`WSF_2015`, `WSF_2019`, `WSF_Evolution`, `WSF_3D`), Global WaterPack (`GWP`), `TIMELINE` AVHRR series, TanDEM-X 30m/90m DEMs, and Sentinel-5P/GOME-2 trace gas products.
