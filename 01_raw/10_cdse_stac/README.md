# `10_cdse_stac` — Copernicus Data Space Ecosystem (CDSE) STAC API & CEOS FedEO CDSE Collection Index

## Overview
- **Source ID**: `10_cdse_stac`
- **Authoritative Endpoint**: `GET https://stac.dataspace.copernicus.eu/v1/collections + GET https://fedeo.ceos.org/collections?parentIdentifier=EOP:EU:CDSE`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Combines the native CDSE STAC v1 API (`stac.dataspace.copernicus.eu/v1/collections?limit=1000` with `rel="next"` pagination, 428 collections) with the expanded CEOS FedEO CDSE index (`parentIdentifier=EOP:EU:CDSE`, 426 collections) which exposes granular CLMS, CMEMS, Copernicus DEM (`COP-DEM_GLO-30/90`), CCM Contributing Missions, and Sentinel mosaic collections.
