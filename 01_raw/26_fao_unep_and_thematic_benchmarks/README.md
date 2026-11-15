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
Harvests (1) UN FAO's Hand-in-Hand / GISMGR v2 REST catalogue (`data.apps.fao.org/gismgr/api/v2/catalog/workspaces` and all paginated `hasNext` mapset pages across `WAPOR-2`, `WAPOR-3`, `GAEZ-V4`, `GAEZ-V5`, `ASAP`, `GLOSIS`, `LCLU`, `WATER`, `ECOSYS`, `EOSTAT`, `FGGD`, `CRU`, etc., 1,028 mapsets), (2) Zenodo's REST API (`zenodo.org/api/records`) and DataCite REST API (`api.datacite.org/dois`) across paginated query pages (`page=1..2`, `size=15`) and pinned DOIs for peer-reviewed global EO L3/L4 benchmark datasets (`G3P` groundwater `10.5880`, ICOS `FLUXCOM-X-BASE` `10.18160/5NZG-JMJE`, INRAE/CESBIO `SMOS-IC v2` `10.15454/tihhyp`, `TROPOSIF` `10.5281/zenodo.10212472`, `LCSPP`/`LCSIF`/`LCREF`, `GOSIF`, `CSIF`, `eSIF`, `VODCA`, `OceanSODA-ETHZ-v2`, `GlobSnow v3.0` `10.1594/pangaea.911944`, `GLEAM4`, Global Mangrove Watch, Global Pasture Watch, ETH Global Canopy Height, GlobPermafrost, HydroSHEDS, Global River Widths, SoilGrids/ISRIC, and WorldCereal), and (3) Carbon Mapper's plume STAC API (`1,524` total retrieved records; `1,520` prepared records after deduplicating `4` `DATACITE:10.5281/zenodo.*` records already retrieved from Zenodo).
