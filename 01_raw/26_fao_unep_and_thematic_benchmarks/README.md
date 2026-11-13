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
Harvests (1) UN FAO's Hand-in-Hand / GISMGR v2 REST catalogue (`data.apps.fao.org/gismgr/api/v2/catalog/workspaces` and workspace mapsets for `WAPOR-3`, `GAEZ-V4`, `ASAP`, `LCLU`, `WATER`, `ECOSYS`, `EOSTAT`, `FGGD`, `CRU`) and (2) Zenodo's REST API (`zenodo.org/api/records`) for peer-reviewed global EO L3/L4 benchmark datasets hosted on Zenodo (including Global Mangrove Watch, Global Pasture Watch, ETH Global Canopy Height, GlobPermafrost, HydroSHEDS, Global River Widths, SoilGrids/ISRIC, and WorldCereal).
