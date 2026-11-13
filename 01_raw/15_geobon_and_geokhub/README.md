# `15_geobon_and_geokhub` — GEO BON Essential Biodiversity Variables (EBV) STAC API & GEO Knowledge Hub InvenioRDM API

## Overview
- **Source ID**: `15_geobon_and_geokhub`
- **Authoritative Endpoint**: `GET https://stac.geobon.org/collections + GET https://gkhub.earthobservations.org/api/records`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests the Group on Earth Observations Biodiversity Observation Network (GEO BON) EBV Data Portal STAC API (`stac.geobon.org/collections`, 34 global/regional L3/L4 biodiversity cubes) and the GEO Knowledge Hub (`gkhub.earthobservations.org/api/records`).
