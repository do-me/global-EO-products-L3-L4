# `25_worldpop_and_ciesin_stac` — WorldPop (University of Southampton) STAC API & CIESIN (Columbia University) STAC Catalogue

## Overview
- **Source ID**: `25_worldpop_and_ciesin_stac`
- **Authoritative Endpoint**: `GET https://api.stac.worldpop.org/collections + GET https://ciesin.github.io/sci-apps-stac/stac/catalog.json`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests the WorldPop STAC API (`api.stac.worldpop.org/collections`, 248 gridded population/demography/settlement layers) and Columbia University CIESIN's STAC tree (`ciesin.github.io/sci-apps-stac/stac/catalog.json`), providing authoritative primary metadata for demographic and human-exposure comparators.
