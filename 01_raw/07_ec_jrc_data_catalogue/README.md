# `07_ec_jrc_data_catalogue` — European Commission Joint Research Centre (JRC) Data Catalogue (DCAT-AP API & CKAN)

## Overview
- **Source ID**: `07_ec_jrc_data_catalogue`
- **Authoritative Endpoint**: `GET https://data.europa.eu/api/hub/search/search?filter=dataset&facets={"catalog":["jrc"]} + data.jrc.ec.europa.eu/api/3/action/package_list`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Direct `package_search` calls to `data.jrc.ec.europa.eu` are blocked by the EC WAF (`Request Rejected`), whereas the official EU Open Data Portal DCAT-AP Search API (`https://data.europa.eu/api/hub/search/search` with `catalog=['jrc']`) serves all ~4,354 JRC datasets with zero WAF blocks, and `data.jrc.ec.europa.eu/api/3/action/package_list` returns all 4,417 dataset IDs. Covers CEMS GHSL (`GHS-POP`, `GHS-BUILT-S/H/V`, `GHS-SMOD`), CEMS EDO/GDO Drought Observatories, CEMS EFFIS/GWIS Fire, GloFAS flood hazard maps, JRC Tropical Moist Forests (TMF), Global Surface Water Explorer (GSWE), EDGAR emissions, DOPA biodiversity, ESDAC soil (GloSEM), ASAP, and PVGIS.
