# `30_jaxa_earth_stac` — JAXA Earth API STAC Catalog (COG v1)

## Overview
- **Source ID**: `30_jaxa_earth_stac`
- **Authoritative Endpoint**: `GET https://data.earth.jaxa.jp/stac/cog/v1/catalog.json` (+ child `collection.json` metadata documents)
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests all public COG v1 STAC collections published by the Japan Aerospace Exploration Agency (JAXA) Earth Observation Research Center (EORC), spanning JAXA EORC (`AW3D30`, `ALOS-2 PALSAR-2 FNF`, `GSMaP`, `ALOS HRLULC`), JAXA G-Portal (`GCOM-W AMSR2`, `GCOM-C SGLI`, `AMSR-E`), JAXA JASMES (`MODIS` / `GCOM-C` / `AMSR2` gridded records), and JMA (`COBE-SST`), together with hosted NASA EOSDIS and Copernicus C3S COG collections whose upstream `producer` roles are preserved separately from the JAXA host role.
