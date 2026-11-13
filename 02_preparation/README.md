# `02_preparation` — Deterministic Preprocessing & L3/L4 Filtering

## Purpose
This stage processes the 29 raw authoritative metadata dumps in `../01_raw/` into cleaned, structurally flattened per-source tables (`./data/<source_id>.parquet` and `./data/<source_id>.jsonl`) and reference taxonomies (`./reference_taxonomies/`).

## Strict Scientific & Reproducibility Rules
1. **Zero Manual Imputation (`let null be null`)**:
   - No missing metadata values are ever filled in "by hand".
   - If an upstream API omits `spatial_resolution`, `temporal_start`, `temporal_end`, `bbox`, or `doi`, the prepared field remains `null` (`None`), and its exact missingness rate is recorded in `./exclusion_audit_summary.csv`.
   - Deterministic structural hygiene normalizes coordinate axis conventions (`[0, 360]` longitudes, inverted latitude/longitude bounds while preserving valid Pacific antimeridian crossings), inverted ISO date pairs, explicit HTTP/FTP URL schemes, strict `10.\d{4,9}/` DOIs, and WGClimate unit codes without fabricating absent metadata.
2. **Strict Level-3 / Level-4 Product Filtering**:
   - **Excluded (`excluded_l0_l1_l2_or_visualisation`)**: Explicit Level 0, Level 1 (`1A`, `1B`, `1C`), and Level 2 (`2A`, `2B`) raw/swath/tile sensor collections (e.g., Sentinel-1 GRD/SLC, Sentinel-2 L1C/L2A scene archives, Sentinel-3 OLCI/SLSTR L1/L2 swaths, Sentinel-5P L2 orbital swaths, Landsat Collection 2 L1/L2 scenes, EUMETSAT L1/L2 and WMS `Visualisation` layers).
   - **Excluded (`excluded_pure_insitu_station_logs`)**: Non-gridded discrete buoy/gauge station logs (e.g., `INSITU_*_DISCRETE_*`).
   - **Excluded (`excluded_non_eo_or_non_dataset`)**: Non-EO administrative/economic survey tables in general institutional repositories (e.g., Eurobarometer spreadsheets in JRC or non-geospatial tabular indicators in Resource Watch).
   - **Retained (`prepared_l3_l4_records`)**:
     - Explicit Level 3 and Level 4 satellite and model products;
     - Implicit Level 3/Level 4 gridded biophysical/thematic products, Climate Data Records (CDRs), reanalyses, forecasts, and Digital Elevation Models (DEMs);
     - Gridded L3 Analysis-Ready Surface Reflectance Mosaics (`L3_ARD_REFLECTANCE`, e.g., NASA `HLSL30`/`HLSS30` and Copernicus `CLMS Sentinel-2 Global Mosaic`) retained symmetrically across both pools;
     - EO-constrained hybrid spatial grids (e.g., `GHSL`, `OpenLandMap`, `WorldPop`) and authoritative policy reference boundaries (`WDPA`, `Natura 2000`) retained with explicit ontology flags for downstream eligibility filtering in `03_harmonization`.
3. **Reference Taxonomies Separated from Product Rows**:
   - Source `13_wmo_oscar_and_gcos` (237 WMO/GCOS geophysical variables and 1,063 satellite mission definitions) is exported to `./reference_taxonomies/wmo_oscar_variables.parquet` and `./reference_taxonomies/wmo_oscar_satellites.parquet` so hardware registry entries are never conflated with L3/L4 data products.

## How to Run
```bash
uv run 02_preparation/prepare_all.py
```

## Outputs
- `./data/<source_id>.parquet` & `./data/<source_id>.jsonl` — Cleaned L3/L4 records for each of the 28 product sources.
- `./data/excluded_records_log.parquet` — Complete item-level log of every excluded record and its deterministic `exclusion_reason`.
- `./exclusion_audit_summary.csv` — Source-by-source accounting of raw records, excluded counts by reason, retained L3/L4 records, and raw field null percentages.
- `./reference_taxonomies/` — WMO OSCAR Space variables and satellites reference tables.
