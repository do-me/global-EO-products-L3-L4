# `35_cnes_geodes_stac` — CNES GEODES STAC API (THEIA / Form@Ter / SAGSA)

## Overview
- **Source ID**: `35_cnes_geodes_stac`
- **Authoritative Endpoint**: `GET https://geodes-portal.cnes.fr/api/stac/collections/paginate`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests all collections exposed by the French Space Agency (CNES) GEODES unified EO portal (`geodes-portal.cnes.fr`), spanning THEIA land surfaces (`OSO` France land cover L3B, multi-satellite snow L3B, Sentinel-2 and Venµs L3A monthly reflectance syntheses, and POSTEL multi-year vegetation/water/radiation/land-cover archives), Form@Ter (`FLATSIM` InSAR cumulative deformation cubes), and CNES SAGSA GRACE L3 gravity field ensembles. Raw L1/L2 scene collections (`SWH_SPOT*`, `PEPS_S1/S3`, `THEIA_*_L1C/L2A/L2B`, `SAGSA_L1B/L2A/L2B`) and intermediate auxiliary/interferogram bundles (`FLATSIM_AUXILIARYDATA`, `FLATSIM_INTERFEROGRAM`) are excluded deterministically in `02_preparation/`, where placeholder `[-180, -90, 180, 90]` bounding boxes on regional THEIA/FLATSIM collections are also stripped cleanly.
