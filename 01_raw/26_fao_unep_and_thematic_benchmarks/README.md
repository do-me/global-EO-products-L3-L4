# `26_fao_unep_and_thematic_benchmarks` — FAO GISMGR v2 (WaPOR, GAEZ, ASAP, GSOC) & Zenodo Authoritative Global EO L3/L4 Benchmarks

## Overview
- **Source ID**: `26_fao_unep_and_thematic_benchmarks`
- **Authoritative Endpoint**: `GET https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces + GET https://zenodo.org/api/records`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests (1) UN FAO's Hand-in-Hand / GISMGR v2 REST catalogue (`data.apps.fao.org/gismgr/api/v2/catalog/workspaces` and all paginated `hasNext` mapset pages across `WAPOR-2`, `WAPOR-3`, `GAEZ-V4`, `GAEZ-V5`, `ASAP`, `GLOSIS`, `LCLU`, `WATER`, `ECOSYS`, `EOSTAT`, `FGGD`, `CRU`, etc., 1,028 mapsets), (2) Zenodo's REST API (`zenodo.org/api/records`) and DataCite REST API (`api.datacite.org/dois`) for peer-reviewed global EO L3/L4 benchmark datasets (`G3P` groundwater `10.5880`, ICOS `FLUXCOM-X-BASE` `10.18160/5NZG-JMJE`, `LCSPP`/`LCSIF`/`LCREF`, `GOSIF`, `CSIF`, `GLEAM4`, Global Mangrove Watch, Global Pasture Watch, ETH Global Canopy Height, GlobPermafrost, HydroSHEDS, Global River Widths, SoilGrids/ISRIC, and WorldCereal), and (3) Carbon Mapper's plume STAC API.
