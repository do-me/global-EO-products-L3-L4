# `03_harmonization` — Unified L3/L4 Schema & Harmonized Catalogue

## Purpose
Harmonizes all 28 prepared Earth Observation (EO) L3/L4 product sources (`38,919` records) from `../02_preparation/data/` into a unified, reproducible metadata schema (`61` record columns; `63` canonical columns) with two-layer producer-origin institutional attribution (`provider_pool` + `provider_jurisdiction_tier`), processing-level ontology and provenance classes (`null` preserved for unlabelled records; primary-source citations required for any provider specification or grid adjudication), Sentinel-dependency lineage indicators (`confirmed_input_mission`, `retrospective_or_mixed_epoch_series`, `target_grid_or_harmonization_reference_only`, `explicit_non_sentinel`, `unassessed_no_platform_metadata`), spatial conflict flagging and cited grid adjudications, cadence vs. rolling-window separation, supersession tracking, an explicit sub-product variant table (`eo_l3_l4_product_variants.parquet`, `58` rows, `16` columns), and conservative canonical deduplication.

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
| `provider_pool` | `string` | `copernicus_services` \| `eu_non_copernicus` \| `non_eu` — assigned strictly by `producing_institution` in alignment with Regulation (EU) 2021/696 (EU-27, EEA Participating States Norway & Iceland, UK Copernicus Association effective 2024-01-01, and ESA/EUMETSAT/ECMWF; third countries including non-Copernicus-associated Switzerland and ESA Associate Member Canada, plus UN/global bodies, sit in `non_eu`). |
| `provider_jurisdiction_tier` | `string` | Granular institutional tier enabling policy-boundary sensitivity analysis: `copernicus_service` \| `eu27_national_or_eu_agency` \| `european_space_met_igo` \| `copernicus_associated_state` \| `european_esa_non_copernicus_state` \| `un_or_international_body` \| `third_country_non_european`. |
| `copernicus_service` | `string \| null` | `CMEMS` \| `CLMS` \| `C3S` \| `CAMS` \| `CEMS` \| `COP_DEM` \| `null`. |
| `is_un_or_international_body` | `bool` | Flags UN and international multilateral bodies (`FAO`, `UNEP-WCMC`, `WMO`, `GEO BON`, `CGIAR`, `World Bank`) for transparent attribution filtering. |
| `processing_level_raw` | `string \| null` | Exact upstream processing level tag prior to harmonization (`null` if omitted upstream). |
| `processing_level` | `string \| null` | `L1` \| `L2` \| `L3` \| `L3; L4` \| `L4` \| `null` (retains `null` where upstream metadata lacks explicit evidence rather than defaulting unlabelled rows to `L3`). |
| `processing_levels_available` | `string \| null` | Multi-level deliverables disclosed in upstream metadata/descriptions (e.g., `L2; L3`, `L3; L4`). |
| `processing_level_provenance` | `string` | Explicit provenance of `processing_level` (`explicit_raw_metadata`, `explicit_title_or_id`, `explicit_raw_multi_level`, `explicit_description_multi_level`, `explicit_title_l4_over_raw_l3`, `provider_documented_specification`, `inferred_from_model_or_service`, `inferred_from_ard_reflectance`, `not_applicable_non_satellite_eo`, `unlabelled_no_explicit_level`). |
| `product_level_subtype` | `string` | `L3_THEMATIC_BIOPHYSICAL` \| `L4_MODEL_REANALYSIS_SYNTHESIS` \| `L3_ARD_REFLECTANCE` \| `L3_L4_MULTI_LEVEL` \| `UPSTREAM_L1_OR_L2_UNFILTERED` \| `UNLABELLED_LEVEL`. |
| `product_ontology_class` | `string` | `eo_thematic_or_biophysical` \| `eo_reanalysis_or_model` \| `eo_ard_surface_reflectance_mosaic` \| `eo_constrained_hybrid` \| `in_situ_gridded_interpolation` \| `non_eo_inventory_or_census` \| `authoritative_reference_geometry`. |
| `product_output_category` | `string` | Output thematic domain/variable category separated from input sensor descriptions. |
| `is_ard_reflectance_only` | `bool` | `True` strictly for L3 ARD surface reflectance mosaics (`HLSL30`, `MCD43A4` NBAR, `S2GM`, `LCREF`) without leaking derived biophysical indices. |
| `strict_eo_c1_eligible` | `bool` | `True` for strict satellite/model EO products; `False` for hybrid demographic/soil models, gridded in-situ interpolations, non-EO inventories, and reference boundaries. |
| `uses_sentinel_data` | `bool` | `True` if the product uses Copernicus Sentinel (`S1`, `S2`, `S3`, `S5P`, `S6`) observations as input (excludes `HLSL30` Landsat-only target-grid references and inspects structured platform metadata prior to pre-2014 suppressions). |
| `sentinel_missions` | `string \| null` | Semicolon-separated list of Sentinel missions used (`S1`, `S2`, `S3`, `S5P`, `S6`). |
| `sentinel_lineage_status` | `string` | Disambiguates `confirmed_input_mission`, `retrospective_or_mixed_epoch_series`, `target_grid_or_harmonization_reference_only` (`HLSL30`), `explicit_non_sentinel` (explicitly non-Sentinel platforms/pre-2014 sensors), and `unassessed_no_platform_metadata`. |
| `spatial_resolution_raw` | `string \| null` | Exact upstream spatial resolution string (`null` if omitted by upstream API). |
| `spatial_resolution_meters` | `float \| null` | Deterministically parsed from `spatial_resolution_raw` (`null` on >=10x/<=0.1x metadata conflicts or upstream omission). |
| `title_token_resolution_meters` | `float \| null` | Explicit resolution tokens (e.g., `10m`, `30m`, `250m`, `1km`) parsed from `title`/`raw_id` without modifying `spatial_resolution_meters`. |
| `spatial_resolution_conflict` | `bool` | Flags `True` when `spatial_resolution_meters` and `title_token_resolution_meters` diverge by >=10x (`null`s `spatial_resolution_meters` instead of guessing). |
| `spatial_resolution_conflict_reason` | `string \| null` | Diagnostic description of detected spatial resolution unit/metadata conflict. |
| `adjudicated_output_grid_meters` | `float \| null` | Reconciled output grid spacing when unconflicted or resolved from a cited primary DOI specification; `null` under unresolvable metadata conflict or omission. |
| `adjudicated_spatial_grid_value`, `adjudicated_spatial_grid_units`, `adjudicated_spatial_grid_axes` | `string \| null` | Structured native grid spacing, unit (`degrees` / `meters`), and axis ordering (`lon x lat` / `lat x lon`) recorded alongside cited spatial conflict adjudications. |
| `adjudicated_spatial_resolution_source`, `adjudication_source_citation` | `string \| null` | Provenance (`raw_metadata`, `title_token`, `cited_provider_specification`, `null`) and authoritative documentation/DOI URL backing cited level or spatial adjudications. |
| `temporal_resolution_raw`, `update_frequency_raw` | `string \| null` | Upstream observation/product step vs. archive update/publication cadence. |
| `temporal_cadence_class` | `string \| null` | `sub_daily_or_daily` \| `multi_day_or_dekadal` \| `monthly_or_seasonal` \| `annual` \| `multi_year_periodic` \| `static_or_single_epoch` \| `null`. |
| `temporal_output_cadences` | `string \| null` | Normalized output sampling step(s) (separated from rolling compositing windows). |
| `temporal_aggregation_window_raw` | `string \| null` | Extracted rolling retrieval/compositing window (e.g., `16-day retrieval/compositing window` for `MCD43A4` daily output). |
| `temporal_start_raw`, `temporal_end_raw` | `string \| null` | Cleaned raw temporal start/end ISO strings (`null` if omitted). |
| `temporal_start_year`, `temporal_end_year` | `int \| null` | Extracted strictly from `temporal_start_raw` / `temporal_end_raw` (`null` if omitted). |
| `is_ongoing_archive` | `bool \| null` | `True` if open-ended active series (`temporal_end_raw >= 2025` or open-ended with `end_year >= 2025` / null end date and ongoing/regular cadence), `False` if completed/planned/deprecated/not-planned or release-dated without ongoing updates, `null` if timestamps and upstream status are omitted. |
| `operational_maturity_class` | `string` | `operational_ongoing_series` \| `completed_or_mission_bounded_release` \| `research_or_benchmark_release` \| `planned_or_preoperational` \| `deprecated_or_superseded` \| `unknown` (never assigns `operational_ongoing_series` when `is_ongoing_archive` is `null`). |
| `is_superseded_edition`, `superseded_by_record` | `bool`, `string \| null` | Identifies superseded editions (`SOCATv1.5..v2025`, closed CLMS/EEA v1 series) and links to active successor `record_id`. |
| `bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` | `float \| null` | Bounding box coordinates (`null` if omitted). |
| `geographic_coverage_class` | `string \| null` | `global` \| `quasi_global` \| `hemispheric_or_pantropical` \| `europe_regional` \| `regional_or_local` \| `null` (downgrades single-sensor GEO disks to `regional_or_local`). |
| `observation_domain_class` | `string \| null` | Physical observation domain (e.g., `geostationary_disk_or_sector`, `full_global_domain`, `quasi_global_band`, `european_regional`, `sub_global_or_regional`). |
| `platform_sensor_raw`, `keywords` | `string \| null` | Raw platform/sensor strings and normalized upstream keyword lists. |
| `doi`, `license`, `access_url` | `string \| null` | Persistent identifier, upstream product license, and role-prioritized non-image landing page URL. |

## Outputs
- `./data/eo_l3_l4_all_catalogue_records.parquet` (and chunked `.jsonl` files) — All `38,919` harmonized L3/L4 records across all 28 product sources (`61` columns).
- `./data/eo_l3_l4_canonical_products.parquet` — `38,352` conservatively deduplicated canonical L3/L4 products (`63` columns; `543` mirror groups collapsing `567` duplicate records) prioritizing primary-agency metadata over secondary cloud mirrors.
- `./data/eo_l3_l4_product_variants.parquet` (and `.jsonl`) — `58` explicit sub-product variant rows (`16` columns) exposing component levels, cadences, formats, and grid spacings inside multi-deliverable collections (e.g., `TCSIF v1` `L3` monthly `0.5°` GeoTIFF vs. `L2` daily HDF5, `C3S SST` `L3C` vs. `L4`).
- `./provider_pool_and_sentinel_summary.csv` — Breakdown of canonical products by `provider_pool`, `processing_level`, and `uses_sentinel_data`.
- `./schema_completeness_report.csv` — Exact non-null and null counts across all `61` harmonized record columns.
