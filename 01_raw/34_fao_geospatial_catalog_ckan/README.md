# `34_fao_geospatial_catalog_ckan` — FAO Geospatial & Hand-in-Hand Map Catalog (CKAN / ISO-19115)

## Overview
- **Source ID**: `34_fao_geospatial_catalog_ckan`
- **Authoritative Endpoint**: `GET https://data.apps.fao.org/catalog/api/3/action/package_search`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests FAO's official GeoNetwork/ISO-19115-backed CKAN Geospatial Catalog across 10 dedicated Earth Observation and gridded agro-environmental organizations (`fao-land-cover-land-use`, `glosis`, `fao-gaez`, `fao-giews`, `fao-eostat`, `climate-risk-toolbox-crtb`, `fao-aquastat`, `fao-food-insecurity-poverty-and-environment-global-gis-database-fggd`, `drive-global-drought-vulnerability-inventory`, and `national-forest-monitoring-systems-nfms`). Deliberately excludes scanned legacy paper maps (`fao-paper-maps`), tabular country statistics (`faostat`, `fao-countrystat`), platform configuration records, and `wapor` (already harvested via FAO GISMGR v2 in `26_fao_unep_and_thematic_benchmarks`). In `02_preparation/`, only gridded ISO-19115 geospatial datasets (`spatialRepresentationType: grid`) and flagship gridded land-cover products (e.g., `GLC-SHARE`, `HWSD v2.01/v2.2`, `GAEZ v5`, `GSOCmap`, `GLADA`, `FAO ASIS`, `GMIA v5`) are retained, while vector boundaries, software packages, and purely socioeconomic/health indicators are filtered out deterministically.
