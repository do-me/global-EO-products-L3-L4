# `08_destine_dedl_stac` — Destination Earth Data Lake (DEDL) Harmonized Data Access (HDA) STAC v2 API

## Overview
- **Source ID**: `08_destine_dedl_stac`
- **Authoritative Endpoint**: `GET https://hda.data.destination-earth.eu/stac/v2/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Returns 348 STAC v2 collections federated across EUMETSAT, ECMWF, ESA, Mercator Océan, and JRC, including all 14 JRC Global Human Settlement Layer collections (`EO.GHSL.DAT.POP`, `BUILT-S`, `BUILT-H`, `BUILT-V`, `BUILT-C`, `SMOD`, etc.), all 6 JRC Global Surface Water collections (`EO.GSW.DAT.*`), and 4 Copernicus DEM collections (`EO.DEM.DAT.COP-DEM_GLO-30/90`).
