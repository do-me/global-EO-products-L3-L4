# `33_us_ghg_center_stac` — U.S. Greenhouse Gas Center (NASA / EPA / NIST / NOAA) STAC API

## Overview
- **Source ID**: `33_us_ghg_center_stac`
- **Authoritative Endpoint**: `GET https://earth.gov/ghgcenter/api/stac/collections?limit=200`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests all analysis-ready STAC collections published by the U.S. Greenhouse Gas Center (`earth.gov/ghgcenter`), jointly operated by NASA, U.S. EPA, NIST, and NOAA alongside international partners such as NIES Japan (ODIAC). Covers top-down satellite-constrained atmospheric GHG inversions (`OCO-2 MIP`, `GOSAT CH4 Budget`, `CarbonTracker-CH4`, `TM5-4DVar`), land/ocean flux models (`ECCO-Darwin`, `LPJ-wsl`, `LPJ-EOSIM`, `MiCASA`, `CASA-GFED3`, `BlueFlux`), satellite GHG column/plume/biomass grids (`GEOS`, `OCO-3 SAMs`, `GOES CH4 Plumes`, `EMIT CH4 Plumes`, `GEDI L4B`, `OMI NO2`, `VIIRS Black Marble`), and gridded anthropogenic inventories (`ODIAC`, `EPA CH4`, `GRA2PES`, `Vulcan`). Upstream bounding-box anomalies (inverted latitudes on `GRA2PES`/`Vulcan`, half-cell overhangs on `GEOS`/`GOSAT`, swapped lat/lon on `GEDI`, and placeholder global extent on U.S.-only `EPA CH4`) are normalized deterministically in `02_preparation/`.
