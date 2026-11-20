# `32_nrcan_ccmeo_stac` — Natural Resources Canada (NRCan / CCMEO) Geospatial Datacube STAC API

## Overview
- **Source ID**: `32_nrcan_ccmeo_stac`
- **Authoritative Endpoint**: `GET https://datacube.services.geo.ca/stac/api/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests all geospatial EO collections published by the Canada Centre for Mapping and Earth Observation (CCMEO / Natural Resources Canada) under the Open Government Licence – Canada (`OGL-Canada-2.0`), covering Radarsat Constellation Mission (RCM) ARD mosaics, Pan-Arctic and Canadian wetland inventories, dynamic surface water, evapotranspiration, 30 m snow dynamics, national land cover, flood susceptibility, river ice, and DEMs. Historical aerial photography (`napl-*`, `banff-national-park`), TOA mosaics, and non-EO socioeconomic collections (`lfd-*`, `cancvi`) are filtered deterministically in `02_preparation/`.
