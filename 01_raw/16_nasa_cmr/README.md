# `16_nasa_cmr` — NASA Common Metadata Repository (CMR) Collections API — All Explicit L3/L4 + SEDAC/ORNL Gridded Collections

## Overview
- **Source ID**: `16_nasa_cmr`
- **Authoritative Endpoint**: `GET https://cmr.earthdata.nasa.gov/search/collections.json?processing_level_id[]=3&processing_level_id[]=4`
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Queries NASA CMR directly for `processing_level_id[] in ['3', '4']` without restrictive `boxes == '[-90 -180 90 180]'` or open-ended date filters, preserving quasi-global missions (GEDI +/-51.6 deg, SRTM, GPM, MODIS Land) and completed Climate Data Records, and additionally harvests NASA SEDAC (`SEDAC`) and ORNL DAAC (`ORNL_DAAC`) collections where gridded L3/L4 socio-environmental products omit `processing_level_id`. Saved in chunked `.jsonl` files (<15 MB each).
