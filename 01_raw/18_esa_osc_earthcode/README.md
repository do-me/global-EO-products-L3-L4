# `18_esa_osc_earthcode` — ESA Open Science Catalog (EarthCODE) STAC Product Catalogue

## Overview
- **Source ID**: `18_esa_osc_earthcode`
- **Authoritative Endpoint**: `GET https://esa-earthcode.github.io/open-science-catalog-metadata/products/catalog.json + 366 child STAC JSONs`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests the European Space Agency's Open Science Catalog (EarthCODE) static STAC tree (`esa-earthcode.github.io`), fetching the root `products/catalog.json` and concurrently retrieving all ~366 curated ESA scientific L3/L4 geoscience product STAC Collection JSONs.
