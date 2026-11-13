# Global Harmonized L3/L4 Earth Observation Product Catalogue

An authoritative, reproducible, primary-source **Level-3 and Level-4 Earth Observation (EO) product catalogue** harvested directly from official space agency, European Union, United Nations, and cloud data provider APIs.

---

## 1. Key Highlights & Summary Statistics

- **29 Primary & Federated Metadata Sources (`01_raw/`)**:
  - **33,663 raw product records** across 28 EO product catalogues (+ **1,300 WMO OSCAR** satellite and variable taxonomy records) retrieved directly from official REST, STAC, CSW, Elasticsearch, and DCAT-AP APIs ([`docs/SOURCES.md`](docs/SOURCES.md)).
- **Deterministic L3/L4 Filtering & Provenance Audit (`02_preparation/`)**:
  - **26,609 prepared L3/L4 product records** retained across 28 product catalogues (plus **1,300 WMO OSCAR Space** reference records in `02_preparation/reference_taxonomies/`).
  - Deterministically excludes **2,787** Level-0, Level-1 (`1A/1B/1C/1G/1R/1T/RDR/TDR/SDR`), Level-2 (`2A/2B/2P/L2G`), uncalibrated raw/TOA scene collections, aerial orthophotos, swath archives, WMS visualisation layers, and discrete in-situ/survey logs in `02_preparation/data/excluded_records_log.parquet` (plus **4,200** pre-excluded non-EO records across JRC, GEO Knowledge Hub, and WRI), summarized in [`02_preparation/exclusion_audit_summary.csv`](02_preparation/exclusion_audit_summary.csv).
- **Harmonized Schema & Producer-Origin Attribution (`03_harmonization/`)**:
  - **26,330 unique canonical L3/L4 products** ([`03_harmonization/data/eo_l3_l4_canonical_products.parquet`](03_harmonization/data/eo_l3_l4_canonical_products.parquet)) after conservative cross-catalogue deduplication (`273` mirror groups collapsing `279` duplicate records) prioritizing primary agency metadata:
    - **`copernicus_services`**: **2,776** canonical products (`CMEMS`, `CLMS`, `C3S`, `CAMS`, `CEMS` — including WEkEO, DestinE, EODC `CEMS-GFM`, JRC `CEMS-GHSL`/`EDO`/`GDO`/`EFFIS`/`GWIS`/`GloFAS`, and Copernicus WGClimate CDRs — plus `Copernicus DEM`).
    - **`eu_non_copernicus`**: **5,690** canonical products (ESA CCI/EarthCODE/WorldCover/WorldCereal, EUMETSAT & 8 SAFs, EC JRC flagships `TMF`/`GSWE`/`EDGAR`/`DOPA`/`ESDAC`, Horizon 2020/GFZ `G3P`, UGent `GLEAM4`, EEA, EMODnet, DLR `WSF`, CNES/Data Terra, VITO, OpenGeoHub `OpenLandMap`, ISRIC `SoilGrids`, WorldPop, Deltares, DTU).
    - **`non_eu`**: **17,864** canonical products (NASA, NOAA, USGS, JAXA, ISRO, INPE, Geoscience Australia DEA, NRCan, WRI/GFW, Esri `10m LULC`, UMD GLAD, CIESIN, FAO `WaPOR`/`GAEZ`/`GSOC`, UNEP-WCMC, GEO BON).
  - **Copernicus Sentinel Dependency Across Non-Copernicus Pools**:
    - **1,598 products** (**806** in `eu_non_copernicus` and **792** in `non_eu`) explicitly depend on Copernicus Sentinel (`S1`, `S2`, `S3`, `S5P`, `S6`) observations (e.g., Google/WRI *Dynamic World*, NASA/JPL *OPERA DSWx-HLS*, ESA *WorldCover*).
- **Automated Quality Assurance & Benchmark Coverage (`04_validation_and_review/`)**:
  - Passes all **12 structural/ontology suites** (`verify_catalogue.py`) and **9 internal QA assertions** (`validate_catalogue_benchmarks.py`), with **`71 / 71` (`100.0%`)** coverage across audited open L3/L4 EU policy & operational service families and **`40 / 40`** CEOS/CGMS WGClimate GCOS ECV groups ([`04_validation_and_review/README.md`](04_validation_and_review/README.md)).
- **Zero Synthetic Imputation (`let null be null`)**:
  - Every field is parsed deterministically from upstream API responses. Omitted upstream attributes remain `null` and are audited in [`03_harmonization/schema_completeness_report.csv`](03_harmonization/schema_completeness_report.csv).

---

## 2. Scope, Design Choices & Limitations

The pipeline deliberately prioritizes **high recall over precision** so eligible L3/L4 products are not silently dropped by overly aggressive heuristics. Conservative deduplication (exact normalized DOI or pool + normalized full title) leaves unlinked cross-platform cloud mirrors without shared DOIs, versioned releases, multi-level agency collections bundling L3/L4 with lower processing levels, asset-level shards in community catalogs (`29_awesome_gee_community`, retained for unique global EO datasets not yet indexed in upstream agency STACs such as *FABDEM*, *Meta/WRI Global Canopy Height 1 m*, and *Hydrography90m*), and tagged sensitivity datasets (gridded in-situ interpolations, constrained hybrid demographic grids, bottom-up inventories, and authoritative reference boundaries) as distinct rows. Operational status distinguishes `operational_ongoing_series`, `completed_or_mission_bounded_release`, `research_or_benchmark_release`, `planned_or_preoperational`, and `deprecated_or_superseded` records. Downstream users should therefore interpret pool totals as harmonized catalogue-inventory counts and perform expert screening and task-specific deduplication against primary documentation. Datasets distributed outside machine-readable STAC/REST/CSW endpoints and real-time operational NWP streams lie outside this harvest.

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
│   ├── provider_pool_and_sentinel_summary.csv        # Summary by provider_pool, level & Sentinel usage
│   ├── schema_completeness_report.csv                # Exact non-null & null counts per column
│   └── data/
│       ├── eo_l3_l4_all_catalogue_records.parquet    # All 26,609 harmonized L3/L4 records
│       ├── eo_l3_l4_all_catalogue_records_part*.jsonl
│       └── eo_l3_l4_canonical_products.parquet       # 26,330 conservatively deduplicated canonical L3/L4 products
│
└── 04_validation_and_review/                         # Automated QA & external benchmark validation
    ├── README.md                                     # Validation summary & catalogue limitations
    ├── validate_catalogue_benchmarks.py              # `uv run` QA & benchmark coverage validation suite
    ├── eu_policy_products_benchmark_audit.csv        # 80-family EU policy & operational service coverage matrix
    └── gcos_ecv_domain_coverage.csv                  # 40/40 CEOS/CGMS WGClimate GCOS ECV coverage matrix
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
