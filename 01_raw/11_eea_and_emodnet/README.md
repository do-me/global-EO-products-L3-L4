# `11_eea_and_emodnet` — European Environment Agency (EEA SDI) & European Marine Observation and Data Network (EMODnet) GeoNetwork ES APIs

## Overview
- **Source ID**: `11_eea_and_emodnet`
- **Authoritative Endpoint**: `POST https://sdi.eea.europa.eu/catalogue/srv/api/search/records/_search + POST https://emodnet.ec.europa.eu/geonetwork/srv/api/search/records/_search`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries the Elasticsearch REST backends of EEA SDI and EMODnet GeoNetwork catalogues for all gridded/raster datasets (`cl_spatialRepresentationType.key: ['grid', 'raster']`) plus key European thematic series. Base64 thumbnail strings (`overview`) and verbose GEMET thesaurus trees (`th_*_tree`, `document`) are excluded via ES `_source` projection so pure metadata records remain compact (<5 KB/record).
