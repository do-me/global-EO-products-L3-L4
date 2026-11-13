# `29_awesome_gee_community` — Awesome GEE Community Catalog (Samapriya Roy Peer-Reviewed EO Dataset Index)

## Overview
- **Source ID**: `29_awesome_gee_community`
- **Authoritative Endpoint**: `GET https://raw.githubusercontent.com/samapriya/awesome-gee-community-datasets/master/community_datasets.json`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Retrieves all 4,401 community-ingested peer-reviewed Earth Observation and geospatial datasets from `samapriya/awesome-gee-community-datasets`. Serves as a critical safety net for academic L3/L4 benchmarks (GLC_FCS30D, Global Canopy Height, Global River Widths, GlobPermafrost, etc.) published outside traditional space-agency DAACs.
