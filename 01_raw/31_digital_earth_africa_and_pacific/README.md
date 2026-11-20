# `31_digital_earth_africa_and_pacific` — Operational Continental & Regional EO Hubs (Digital Earth Africa & Digital Earth Pacific)

## Overview
- **Source ID**: `31_digital_earth_africa_and_pacific`
- **Authoritative Endpoints**:
  - `GET https://explorer.digitalearth.africa/products.txt` + `GET https://explorer.digitalearth.africa/products/<id>.odc-product.yaml`
  - `GET https://stac.digitalearthpacific.org/collections` (paginated)
- **Output Directory**: `./data/`
- **Provenance & Checksum Log**: `./last_access.txt`

## How to Run
From inside this folder (or from the repository root):
```bash
uv run mine.py
```

## Particularities & Technical Notes
Harvests operational continental/regional Open Data Cube product registries for Africa (**Digital Earth Africa**, via authoritative `.odc-product.yaml` definitions) and the Pacific Islands (**Digital Earth Pacific**, SPC, via paginated STAC `/collections`), including GeoMAD composites, Water Observations from Space (WOfS) summaries, crop masks, mangroves, seagrass, satellite-derived bathymetry, intertidal elevation, and iSDA/IWMI thematic layers. Raw L1/L2 scene collections are filtered deterministically in `02_preparation/`, placeholder template coordinates/dates on Pacific regional layers are guarded against false global/pre-launch promotion, and mirrored third-party layers (Copernicus, ESA, DLR, JAXA, NASA, FAO) retain their originating producer attribution.
