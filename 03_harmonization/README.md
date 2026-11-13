# `03_harmonization` — Unified L3/L4 Schema & Harmonized Catalogue

## Purpose
Harmonizes all 28 prepared Earth Observation (EO) L3/L4 product sources (`26,609` records) from `../02_preparation/data/` into a unified, reproducible metadata schema with producer-origin institutional attribution, processing-level ontology classes, Sentinel-dependency indicators, and conservative canonical deduplication.

## How to Run
```bash
uv run 03_harmonization/harmonize_catalogue.py
```

## Harmonized Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `record_id` | `string` | Unique source-qualified record identifier (`<source_catalogue>::<raw_id>`). |
| `canonical_product_key` | `string` | Cross-catalogue deduplication key (normalized DOI if available, otherwise pool + full normalized title slug). |
| `source_catalogue` | `string` | Upstream source folder ID (`01_copernicus_cmems` .. `29_awesome_gee_community`). |
| `source_authority_rank` | `int` | Authority tier (`1` = primary space agency / Copernicus service API; `5–6` = secondary cloud mirror). |
| `raw_id`, `title`, `description` | `string` | Native product identifier, title, and cleaned abstract/description. |
| `producing_institution` | `string` | Originating institution that produced the dataset (strictly separated from hosting platform). |
| `hosting_platform` | `string` | Dissemination portal or cloud data lake hosting the record (`GEE`, `Planetary Computer`, `WEkEO`, `DEDL`, `CMR`, etc.). |
| `provider_pool` | `string` | `copernicus_services` \| `eu_non_copernicus` \| `non_eu` — assigned strictly by `producing_institution`, independent of `hosting_platform`. |
| `copernicus_service` | `string \| null` | `CMEMS` \| `CLMS` \| `C3S` \| `CAMS` \| `CEMS` \| `COP_DEM` \| `null`. |
| `is_un_or_international_body` | `bool` | Flags UN and international bodies (`FAO`, `UNEP-WCMC`, `WMO`, `GEO BON`) for transparent attribution filtering. |
| `processing_level` | `string` | `L3` \| `L4`. |
| `product_level_subtype` | `string` | `L3_THEMATIC_BIOPHYSICAL` \| `L4_MODEL_REANALYSIS_SYNTHESIS` \| `L3_ARD_REFLECTANCE` (retains surface-reflectance mosaics symmetrically across pools). |
| `product_ontology_class` | `string` | `eo_thematic_or_biophysical` \| `eo_reanalysis_or_model` \| `eo_ard_surface_reflectance_mosaic` \| `eo_constrained_hybrid` \| `in_situ_gridded_interpolation` \| `non_eo_inventory_or_census` \| `authoritative_reference_geometry`. |
| `strict_eo_c1_eligible` | `bool` | `True` for strict satellite/model EO products; `False` for hybrid demographic/soil models, gridded in-situ interpolations, non-EO inventories, and reference boundaries. |
| `uses_sentinel_data` | `bool` | `True` if the product uses Copernicus Sentinel (`S1`, `S2`, `S3`, `S5P`, `S6`) observations as input. |
| `sentinel_missions` | `string \| null` | Semicolon-separated list of Sentinel missions used (`S1`, `S2`, `S3`, `S5P`, `S6`). |
| `spatial_resolution_raw` | `string \| null` | Exact upstream spatial resolution string (`null` if omitted by upstream API). |
| `spatial_resolution_meters` | `float \| null` | Deterministically parsed from `spatial_resolution_raw` (`null` if omitted upstream). |
| `title_token_resolution_meters` | `float \| null` | Explicit resolution tokens (e.g., `10m`, `30m`, `250m`, `1km`) parsed from `title`/`raw_id` without modifying `spatial_resolution_meters`. |
| `temporal_cadence_class` | `string \| null` | `sub_daily_or_daily` \| `multi_day_or_dekadal` \| `monthly_or_seasonal` \| `annual` \| `multi_year_periodic` \| `static_or_single_epoch` \| `null`. |
| `temporal_start_year`, `temporal_end_year` | `int \| null` | Extracted strictly from `temporal_start_raw` / `temporal_end_raw` (`null` if omitted). |
| `is_ongoing_archive` | `bool \| null` | `True` if open-ended active series (`temporal_end_raw >= 2025` or open-ended with ongoing/regular cadence), `False` if completed/planned/deprecated or release-dated without ongoing updates, `null` if timestamps are omitted. |
| `operational_maturity_class` | `string` | `operational_ongoing_series` \| `completed_or_mission_bounded_release` \| `research_or_benchmark_release` \| `planned_or_preoperational` \| `deprecated_or_superseded`. |
| `bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` | `float \| null` | Bounding box coordinates (`null` if omitted). |
| `geographic_coverage_class` | `string \| null` | `global` \| `quasi_global` \| `hemispheric_or_pantropical` \| `europe_regional` \| `regional_or_local` \| `null`. |
| `doi`, `license`, `access_url` | `string \| null` | Persistent identifier, upstream product license, and landing page URL. |

## Outputs
- `./data/eo_l3_l4_all_catalogue_records.parquet` (and chunked `.jsonl` files) — All `26,609` harmonized L3/L4 records across all 28 product sources.
- `./data/eo_l3_l4_canonical_products.parquet` — `26,330` conservatively deduplicated canonical L3/L4 products (`273` mirror groups collapsing `279` duplicate records) prioritizing primary-agency metadata over secondary cloud mirrors.
- `./provider_pool_and_sentinel_summary.csv` — Breakdown of canonical products by `provider_pool`, `processing_level`, and `uses_sentinel_data`.
- `./schema_completeness_report.csv` — Exact non-null and null counts across all harmonized columns.
