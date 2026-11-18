# Global Harmonized L3/L4 Earth Observation Product Catalogue

An authoritative, reproducible, primary-source **Level-3 and Level-4 Earth Observation (EO) product catalogue** harvested directly from official space agency, European Union, United Nations, and cloud data provider APIs.

---

## 1. Key Highlights & Summary Statistics

- **29 Primary & Federated Metadata Sources (`01_raw/`)**:
  - **48,527 parsed product records** (`50,876` total retrieved records across all 29 `last_access.txt` endpoints, before deduplicating cross-harvester DOIs in Source 26) across 28 EO product catalogues (+ **1,300 WMO OSCAR** satellite and variable taxonomy records) retrieved directly from official REST, STAC, CSW, Elasticsearch, DataCite, CSL-JSON, and DCAT-AP APIs with full pagination and deep metadata retrieval (`summary: false` in NOAA OneStop; UMM-JSON in NASA CMR & USGS; CEDA `10.5285` DataCite DOI crosswalk in FedEO; DIAS/ScienceDB/Theia/TPDC releases in Source 26) ([`docs/SOURCES.md`](docs/SOURCES.md)).
- **Deterministic L3/L4 Filtering & Provenance Audit (`02_preparation/`)**:
  - **38,974 prepared L3/L4 product records** retained across 28 product catalogues (plus **1,300 WMO OSCAR Space** reference records in `02_preparation/reference_taxonomies/`).
  - Deterministically excludes **4,552** Level-0, Level-1 (`1A/1B/1C/1G/1R/1T/RDR/TDR/SDR`), swath/tile Level-2 (`2A/2B/2P` swaths and uncomposited scene reflectance), uncalibrated raw/TOA scene collections, aerial orthophotos, WMS visualisation layers, and discrete in-situ/survey logs in `02_preparation/data/excluded_records_log.parquet` (plus **4,200** pre-excluded non-EO records across JRC, GEO Knowledge Hub, and WRI), summarized in [`02_preparation/exclusion_audit_summary.csv`](02_preparation/exclusion_audit_summary.csv).
- **Harmonized Schema & Two-Layer Institutional Attribution (`03_harmonization/`)**:
  - **38,342 canonical catalogue records** ([`03_harmonization/data/eo_l3_l4_canonical_products.parquet`](03_harmonization/data/eo_l3_l4_canonical_products.parquet); `32,675` tagged `strict_eo_c1_eligible = True` + `5,667` tagged sensitivity/hybrid/reference records) derived from `38,974` harvested records across `28` product catalogues after conservative cross-catalogue deduplication (`600` mirror groups collapsing `632` duplicate records; counts reflect the harvested catalogue inventory with documented field-completeness limits rather than an exhaustive universe census):
    - **`copernicus_services`**: **3,520** canonical catalogue records (`CMEMS`, `CLMS`, `C3S`, `CAMS`, `CEMS` — including WEkEO, DestinE, EODC `CEMS-GFM`, JRC `CEMS-GHSL`/`EDO`/`GDO`/`EFFIS`/`GWIS`/`GloFAS`/Rapid Mapping, EEA-catalogued `CLMS` products, cross-platform `ERA5`/`AgERA5`, and Copernicus WGClimate CDRs — plus `Copernicus DEM`).
    - **`eu_non_copernicus`**: **5,260** canonical catalogue records across EU-27 national agencies/institutes & EC flagships (`3,107` — EC JRC `TMF`/`GSWE`/`EDGAR`/`DOPA`/`ESDAC`, EEA, EMODnet, DLR, CNES/Data Terra/Theia `Hugonnet`/`Millan`/`CTOH`, INRAE/CESBIO `SMOS-IC`, VITO, OpenGeoHub, ISRIC, Deltares, DTU, MPI-BGC/ICOS `FLUXCOM-X`, TU Wien `VODCA`), European space/meteorological IGOs (`1,751` — ESA CCI/EarthCODE/WorldCover/WorldCereal/`TROPOSIF`/`GlobSnow`/`FireCCI`/`OC-CCI`/`Snow CCI`, EUMETSAT & 8 SAFs, ECMWF DestinE), and Copernicus Participating/Associated States under Regulation (EU) 2021/696 (`402` — Norway, Iceland, and UK post-Jan 2024 association, e.g., WorldPop, FABDEM, UKCEH, Met Office, MET Norway).
    - **`non_eu`**: **29,562** canonical catalogue records spanning third-country agencies (`28,060` — NASA/SEDAC, NOAA, USGS, JAXA, DIAS `J-OFURO3`, CAS `ScienceDB CoSEB`/TPDC `PML-V2.2`, ISRO, INPE BDC, Geoscience Australia DEA, NRCan, WRI/GFW, Esri, UMD GLAD, CIESIN), Swiss institutions (`25` — ESA/EUMETSAT/ECMWF member outside Copernicus association, tagged `european_esa_non_copernicus_state`), and UN/international bodies (`1,477` — FAO `WaPOR`/`GAEZ`/`GSOC`/`WATER`, UNEP-WCMC, WMO, GEO BON, CGIAR). Every record carries both `provider_pool` and `provider_jurisdiction_tier` so analysts can evaluate sensitivity across EU-27 vs. Copernicus-associated vs. ESA boundaries.
  - **Copernicus Sentinel Dependency Across Non-Copernicus Pools**:
    - **1,553 canonical catalogue records** (**570** in `eu_non_copernicus` and **983** in `non_eu`; **2,225** canonical records total across all pools, alongside **672** in `copernicus_services`) document Copernicus Sentinel (`S1`, `S2`, `S3`, `S5P`, `S6`) observations (e.g., Google/WRI *Dynamic World*, NASA/JPL *OPERA DSWx-HLS*, ESA *WorldCover*, INPE *BDC*, *TROPOSIF*, *QeRSIF*, *GlobeMCH*, *BICEP*, *Millan* glaciers), separating direct input lineage (`sentinel_lineage_status`) from negative non-Sentinel provenance (`explicit_non_sentinel` vs. `unassessed_no_platform_metadata`, stratified explicitly in `provider_pool_and_sentinel_summary.csv`). Note that `platform_sensor_raw` is absent on `49.25%` (`19,193 / 38,974`) of records (populated on `50.75%`, `19,781 / 38,974`), leaving unattributed records classified as `unassessed_no_platform_metadata` rather than `explicit_non_sentinel`.
- **Automated Quality Assurance & Specified Regression Panels (`04_validation_and_review/`)**:
  - Passes all **23 structural/ontology suites** (`verify_catalogue.py`) and **24 internal QA assertions** (`validate_catalogue_benchmarks.py`, including strict RFC 8259 JSONL validation), **`77 / 77`** checked in-scope open L3/L4 EU policy & operational service families (`N = 86` total panel including `5` tagged sensitivity/related, `3` out-of-scope, `1` excluded imagery), **`40 / 40`** checked CEOS/CGMS WGClimate GCOS ECV groups (`Ocean-surface heat flux` matching `34` canonical records including `SeaFlux` + `J-OFURO3 v1.2` while excluding `CO2`/`CoSEB`), and **`79 / 79` specified regression & holdout cases pass** (`55 / 55` core cases + `24 / 24` independent holdout/regression cases, verified by match/mismatch checks rather than statistical completeness or catalogue-wide accuracy estimates), plus full snapshot ID & field diff auditing (`verified_record_ids.json`, `release_field_diff_audit.json`) ([`04_validation_and_review/README.md`](04_validation_and_review/README.md)).
- **Explicit Separation of Provider Attributes, Variant Deliverables & Provenance (`let null be null`)**:
  - Raw native structured attributes (`spatial_resolution_raw`, `spatial_resolution_meters`, `temporal_resolution_raw`, `temporal_start_raw`, `temporal_end_raw`, `bbox_*`, `platform_sensor_raw`, verified primary dataset `doi`) are never overwritten by prose heuristics, container types (`asset_type_raw`), or `rel="describedby"` input/paper citations (`related_dois`, `doi_source`, `doi_relation_type`); omitted, unlabelled, or conflicted upstream attributes (`processing_level`, `processing_level_conflict`, `spatial_resolution_conflict`) remain `null` and are audited in [`03_harmonization/schema_completeness_report.csv`](03_harmonization/schema_completeness_report.csv) alongside separated prose-extraction provenance, `bbox_crosses_antimeridian`, `observation_domain_mask` (`ice_free_ocean_mask` on `J-OFURO3`, `cryosphere_or_glacier_mask`, `mangrove_biome_mask`), and `60` sub-product variant rows in [`03_harmonization/data/eo_l3_l4_product_variants.parquet`](03_harmonization/data/eo_l3_l4_product_variants.parquet).

---

## 2. Scope, Design Choices & Limitations

The pipeline deliberately prioritizes **high recall over precision** so eligible L3/L4 products are not silently dropped by overly aggressive heuristics. Conservative deduplication (exact normalized DOI or pool + normalized full title) leaves unlinked cross-platform cloud mirrors without shared DOIs, versioned releases, multi-level agency collections bundling L3/L4 with lower processing levels, asset-level shards in community catalogs (`29_awesome_gee_community`, retained for unique global EO datasets not yet indexed in upstream agency STACs such as *FABDEM*, *Meta/WRI Global Canopy Height 1 m*, and *Hydrography90m*), and tagged sensitivity datasets (gridded in-situ interpolations, constrained hybrid demographic grids, bottom-up inventories, and authoritative reference boundaries) as distinct rows. Operational status distinguishes `operational_ongoing_series`, `completed_or_mission_bounded_release`, `research_or_benchmark_release`, `planned_or_preoperational`, `deprecated_or_superseded`, and `unknown` records. Downstream users should therefore interpret pool totals as harmonized catalogue-inventory counts and perform expert screening and task-specific deduplication against primary documentation. Datasets distributed outside machine-readable STAC/REST/CSW endpoints and real-time operational NWP streams lie outside this harvest.

---

## 3. Repository Structure

```text
.
├── README.md                                         # Repository overview, limitations & reproduction guide
├── LICENSE                                           # MIT License (pipeline code)
├── LICENSE-DATA                                      # CC-BY-4.0 License (catalogue datasets & documentation)
├── CITATION.cff                                      # Citation metadata (CFF 1.2.0)
├── MANIFEST.json                                     # Cryptographic provenance & API timestamp manifest
├── SHA256SUMS.txt                                    # POSIX sha256sum checksums of all pipeline & data files
├── build_manifest.py                                 # `uv run` script to regenerate MANIFEST.json & SHA256SUMS.txt
├── docs/
│   └── SOURCES.md                                    # Processing-level ontology & 29-source API registry
│
├── 01_raw/                                           # Raw metadata in native API formats (.json / .jsonl)
│   ├── 01_copernicus_cmems/                          # Each source contains: mine.py, README.md, last_access.txt, data/
│   ├── ...
│   └── 29_awesome_gee_community/
│
├── 02_preparation/                                   # Deterministic cleaning, normalization & L3/L4 filtering
│   ├── README.md
│   ├── prepare_all.py                                # `uv run` script generating 02_preparation/data/*
│   ├── exclusion_audit_summary.csv                   # Source-by-source L0/L1/L2 & non-EO exclusion accounting
│   ├── data/                                         # Per-source .parquet & .jsonl + excluded_records_log.parquet
│   └── reference_taxonomies/                         # WMO OSCAR variables & satellites reference tables
│
├── 03_harmonization/                                 # Unified L3/L4 schema, ontology & pool attribution
│   ├── README.md
│   ├── harmonize_catalogue.py                        # `uv run` script generating the harmonized master catalogue
│   ├── verify_catalogue.py                           # `uv run` structural & ontology verification suite
│   ├── provider_pool_and_sentinel_summary.csv        # Summary by pool, level, Sentinel & lineage status
│   ├── schema_completeness_report.csv                # Exact non-null & null counts across all 78 harmonized columns
│   └── data/
│       ├── eo_l3_l4_all_catalogue_records.parquet    # All 38,974 harmonized L3/L4 records (78 columns)
│       ├── eo_l3_l4_all_catalogue_records_part*.jsonl
│       ├── eo_l3_l4_canonical_products.parquet       # 38,342 conservatively deduplicated canonical L3/L4 products (80 columns)
│       └── eo_l3_l4_product_variants.parquet         # 60 explicit evidence-backed sub-product variant specs (19 columns)
│
└── 04_validation_and_review/                         # Automated QA & specified regression/benchmark checks
    ├── README.md                                     # Validation summary & catalogue limitations
    ├── validate_catalogue_benchmarks.py              # `uv run` QA & specified case validation suite
    ├── verified_releases_baseline.json               # 66 regression anchors & 468 Zenodo carry-forward IDs
    ├── verified_record_ids.json                      # 38,974-ID snapshot diff baseline (zero silent omissions)
    ├── release_field_diff_audit.json                 # Automated snapshot field-by-field delta vs. 92a74e3
    ├── eu_policy_products_benchmark_audit.csv        # 86-family EU policy & operational service coverage matrix
    ├── gcos_ecv_domain_coverage.csv                  # 40/40 CEOS/CGMS WGClimate GCOS ECV coverage matrix
    ├── independent_product_version_benchmark.csv     # 79-case panel (55 core + 24 independent holdout/regression cases)
    └── independent_product_version_audit.csv         # 79/79 specified regression & holdout case check results
```

---

## 4. Reproducing the Catalogue End-to-End (`uv run`)

All scripts use PEP 723 inline dependency metadata and require only [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Re-mine any upstream source (writes data/ and updates last_access.txt):
uv run 01_raw/01_copernicus_cmems/mine.py

# 2. Run deterministic L3/L4 preparation across all 29 raw sources:
uv run 02_preparation/prepare_all.py

# 3. Harmonize into the unified L3/L4 schema:
uv run 03_harmonization/harmonize_catalogue.py

# 4. Run verification and benchmark coverage suites:
uv run 03_harmonization/verify_catalogue.py
uv run 04_validation_and_review/validate_catalogue_benchmarks.py

# 5. Refresh cryptographic MANIFEST.json and SHA256SUMS.txt:
uv run build_manifest.py
```

---

## 5. Licensing & Citation

- **Pipeline Code (`*.py`)**: Licensed under the [MIT License](LICENSE).
- **Catalogue Data & Documentation**: Licensed under [CC-BY-4.0](LICENSE-DATA). Individual EO products retain their original upstream provider licenses (`license` column).
- **Citation**: See [`CITATION.cff`](CITATION.cff).
