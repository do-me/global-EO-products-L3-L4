# `26_fao_unep_and_thematic_benchmarks` — FAO GISMGR v2 (WaPOR, GAEZ, ASAP, GSOC) & Zenodo Authoritative Global EO L3/L4 Benchmarks

## Overview
- **Source ID**: `26_fao_unep_and_thematic_benchmarks`
- **Authoritative Endpoint**: `GET https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces + GET https://zenodo.org/api/records + GET https://api.datacite.org/dois + CSL-JSON DOI resolution`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests (1) UN FAO's Hand-in-Hand / GISMGR v2 REST catalogue (`data.apps.fao.org/gismgr/api/v2/catalog/workspaces` and all paginated `hasNext` mapset pages across `WAPOR-2`, `WAPOR-3`, `GAEZ-V4`, `GAEZ-V5`, `ASAP`, `GLOSIS`, `LCLU`, `WATER`, `ECOSYS`, `EOSTAT`, `FGGD`, `CRU`, etc., 1,028 mapsets), (2) Zenodo's REST API (`zenodo.org/api/records`) and DataCite REST API (`api.datacite.org/dois`, plus DIAS HTML metadata & DOI Foundation CSL-JSON resolution in `data/doi_csl_benchmarks.json`) across all paginated `links.next` pages (`data/zenodo_pagination_audit.json`, `1,363` unique Zenodo records including `ZENODO:18385275`, `55` DataCite benchmark records including `J-OFURO3 v1.1/v1.2` `10.20783/dias.612`/`667`, `CoSEB` `10.57760/sciencedb.27228`, `Hugonnet` `10.6096/13`, `Millan` `10.6096/1007`, CTOH/Odatis sea-ice thickness, ScienceDB/DIAS/Theia/GFZ/ICOS/PANGAEA EO releases, plus `1` TPDC CSL-JSON release `10.11888/Terre.tpdc.303314` `PML-V2.2`, `truncated_query_count = 0`) and pinned DOIs/concept DOIs (`G3P` groundwater `10.5880`, ICOS `FLUXCOM-X-BASE` `10.18160/5NZG-JMJE`, INRAE/CESBIO `SMOS-IC v2` `10.15454/tihhyp`, `TROPOSIF` `10.5281/zenodo.10212472`, `TCSIF v1` `10.5281/zenodo.8242928`, `GLOBMAP FTC`, `AVHRR Net Radiation`, `GRACE CSEOF TWS`, `LCSPP`/`LCSIF`/`LCREF`, `GOSIF`, `CSIF`, `eSIF`, `QeRSIF`, `TanSat SIF`, `MUSES`, `PKU GIMMS NDVI v1.2`, `OCO-3` GPP, `GlobeMCH`, `FireCCI`, `OC-CCI`, `VODCA`, `OceanSODA-ETHZ-v2`, `GlobSnow v3.0`, `GLEAM4`, Global Mangrove Watch, Global Pasture Watch, ETH Global Canopy Height, GlobPermafrost, HydroSHEDS, Global River Widths, SoilGrids/ISRIC, and WorldCereal) (`2,447` retrieved entries in `last_access.txt`; `2,439` prepared records after deduplicating cross-harvester DOIs).
