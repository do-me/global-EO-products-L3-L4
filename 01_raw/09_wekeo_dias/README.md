# `09_wekeo_dias` — WEkEO Copernicus DIAS Catalogue APIs (MOI Datasets & HDA Broker v1)

## Overview
- **Source ID**: `09_wekeo_dias`
- **Authoritative Endpoint**: `GET https://moi-be.wekeo.eu/api/datasets?size=2000 + GET https://gateway.prod.wekeo2.eu/hda-broker/api/v1/datasets?size=2000`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests both the WEkEO Mercator Océan portal catalogue API (`moi-be.wekeo.eu/api/datasets`, 462 rich product records) and the underlying WEkEO2 Harmonized Data Access Broker API (`gateway.prod.wekeo2.eu/hda-broker/api/v1/datasets`, 1,421 records) covering CLMS, CMEMS, C3S, CAMS, CEMS, EEA, EUMETSAT, and ESA.
