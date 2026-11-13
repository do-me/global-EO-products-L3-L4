# `22_data_terra_inpe_dea` — National EO Data Cube STAC Catalogues — France (Data Terra/Dinamis), Brazil (INPE BDC) & Australia (DEA)

## Overview
- **Source ID**: `22_data_terra_inpe_dea`
- **Authoritative Endpoint**: `GET api.stac.teledetection.fr/collections + data.inpe.br/bdc/stac/v1/collections + explorer.sandbox.dea.ga.gov.au/stac/collections`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests three major non-Copernicus national Earth Observation STAC catalogues: France's Data Terra / Dinamis / Theia (`api.stac.teledetection.fr`), Brazil's National Institute for Space Research (INPE) Brazil Data Cube (`data.inpe.br/bdc/stac/v1`), and Geoscience Australia's Digital Earth Australia (`explorer.sandbox.dea.ga.gov.au/stac`).
