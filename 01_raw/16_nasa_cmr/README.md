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
Queries NASA CMR directly across `collections.json` and `collections.umm_json` for `processing_level_id[] in ['3', '4', '3A', '3B', '3C', '3G', '4A', '4B', 'L3', 'L4', 'Level 3', 'Level 4']` without restrictive `boxes == '[-90 -180 90 180]'` or open-ended date filters, preserving quasi-global missions (GEDI +/-51.6 deg, SRTM, GPM, MODIS Land), completed Climate Data Records, and full UMM-JSON collection metadata (DOIs, horizontal grid resolutions, temporal resolutions, `CollectionProgress`, GCMD science keywords, and typed `RelatedUrls`). Additionally harvests ESDIS SEDAC collections (`keyword=SEDAC`, covering 327 socioeconomic/environmental gridded products) and supplementary high-recall EO series (`OAFlux`, `GWELD`, `GPCP`, `CMAP`, `Daymet`, `LPJ-EOSIM`, `CYGNSS Heat Flux`, `MERRA`, `GRACE Tellus`, etc.) where gridded L3/L4 products omit `processing_level_id`. Saved in 8 chunked `.jsonl` files (<15 MB each).
