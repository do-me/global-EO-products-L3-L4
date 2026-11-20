# Global Harmonized L3/L4 Earth Observation Product Catalogue

A reproducible **Level-3 and Level-4 Earth Observation (EO) product discovery catalogue** harvested directly from **31 public space-agency, European Union, United Nations, and cloud metadata APIs** (plus the WMO OSCAR/Space variable and satellite taxonomy).

**Interactive viewer:** [do-me.github.io/global-EO-products-L3-L4](https://do-me.github.io/global-EO-products-L3-L4/) · **Source registry & ontology:** [`docs/SOURCES.md`](docs/SOURCES.md)

---

## 1. What This Catalogue Is — and Is Not

This repository helps EO researchers and analysts **discover candidate gridded L3/L4 products across institutional boundaries** (Copernicus Services, European non-Copernicus institutions, and non-EU/international providers) and trace **Copernicus Sentinel input dependencies**. It is a structured starting point for discovery and screening, not a substitute for primary product documentation.

Keep three properties in mind when working with these tables:

1. **Catalogue-inventory counts, not distinct global products.** The pipeline favors high recall so eligible L3/L4 products are not silently dropped. Deduplication is deliberately conservative (matching exact normalized DOI or pool + normalized full title). Unlinked cloud mirrors without shared DOIs, versioned releases, multi-level agency collections, and asset-level sub-collections in `29_awesome_gee_community` remain as separate rows. Treat pool totals as harmonized catalogue-inventory counts and perform task-specific screening and deduplication against primary documentation.
2. **Structured metadata is only as complete as upstream APIs.** Agency and cloud portals frequently omit structured fields: in the canonical table, `processing_level` is null on `73.5%` of rows, `adjudicated_output_grid_meters` on `72.0%`, `temporal_cadence_class` on `69.6%`, `doi` on `67.6%`, and `platform_sensor_raw` on `49.21%` (leaving unattributed records in `unassessed_no_platform_metadata` rather than `explicit_non_sentinel`). Missing or conflicting upstream attributes remain `null`, while text-extracted hints are stored in separate provenance columns. Pair free-text search with faceted filters rather than relying on structured fields alone.
3. **Open-API harvest boundary, not an exhaustive world census.** Only public, unauthenticated machine-readable endpoints (STAC, REST, CSW, Elasticsearch, DCAT-AP, DataCite, Zenodo) are harvested directly. Catalogues and operational streams excluded from direct harvesting are listed explicitly in Section 3 below.

---

## 2. Inventory at a Glance

| Stage / Dimension | Count | Notes |
| :--- | ---: | :--- |
| **Upstream sources (`01_raw/`)** | **32** | `31` EO product catalogues + `1` WMO OSCAR/Space reference taxonomy (`1,300` variable/satellite rows) |
| **Raw records parsed** | **48,811** | Parsed across `31` product catalogues (`51,160` total retrieved across all `32` `last_access.txt` endpoints) |
| **Prepared L3/L4 records (`02_preparation/`)** | **39,202** | Excludes `4,605` logged L0–L2/aerial/in-situ records (`2,577` L0–L2/aerial + `2,028` in-situ), `4,203` pre-excluded non-EO records, and `801` within-source duplicates |
| **Canonical catalogue records (`03_harmonization/`)** | **38,567** | `32,900` strict EO + `5,667` flagged sensitivity/hybrid/reference (`603` mirror groups collapsing `635` duplicates); plus `60` explicit sub-product variant rows |
| — `copernicus_services` | **3,792** | `CMEMS`, `CLMS`, `C3S`, `CAMS`, `CEMS` (incl. `GHSL`, `GFM`, `GloFAS`, `EFAS`, `EDO/GDO`, `EFFIS/GWIS`), and `COP_DEM` |
| — `eu_non_copernicus` | **5,040** | `2,874` EU-27 national/EC agencies (JRC, EEA, EMODnet, DLR, CNES/Theia, VITO, OpenGeoHub), `1,760` European space/met IGOs (ESA, EUMETSAT/SAFs, ECMWF DestinE), `406` Copernicus Associated States (UK, NO, IS under Reg. (EU) 2021/696) |
| — `non_eu` | **29,735** | `28,235` third-country agencies & consortia (NASA, NOAA, USGS, JAXA, NRCan, GA DEA, DE Africa/Pacific, INPE, WRI, Esri), `1,475` UN/international bodies (FAO, UNEP-WCMC, WMO, GEO BON), `25` Swiss institutions (`european_esa_non_copernicus_state`) |
| **Documented Sentinel input (canonical)** | **2,253** | `673` Copernicus Services, `586` EU non-Copernicus, `994` non-EU (`1,580` outside Copernicus); separates `confirmed_input_mission` / `retrospective_or_mixed_epoch_series` from `explicit_non_sentinel`, target-grid references (`HLSL30`), and `unassessed_no_platform_metadata` |
| **Automated verification (`04_validation_and_review/`)** | **PASS** | `24` structural/ontology suites, `25` QA assertions, `77/77` checked EU policy families, `40/40` GCOS ECV groups, `79/79` curated product & holdout checks |

Every canonical record carries both `provider_pool` and `provider_jurisdiction_tier` so analysts can compare counts across EU-27, Copernicus-associated, and ESA boundaries. Full column-level completeness is reported in [`03_harmonization/schema_completeness_report.csv`](03_harmonization/schema_completeness_report.csv), pool and Sentinel breakdowns in [`03_harmonization/provider_pool_and_sentinel_summary.csv`](03_harmonization/provider_pool_and_sentinel_summary.csv), and per-source filtering counts in [`02_preparation/exclusion_audit_summary.csv`](02_preparation/exclusion_audit_summary.csv).

---

## 3. Catalogues Evaluated & Not Included

The following portals, operational services, and archives were evaluated and intentionally not ingested as standalone primary sources—either because they sit outside the open gridded L3/L4 archive scope or because no public, unauthenticated machine-readable collection API is exposed:

| Catalogue / Portal | Why Not Directly Included | Partial or Parallel Coverage in Catalogue |
| :--- | :--- | :--- |
| **JAXA G-Portal CSW** (`gportal.jaxa.jp/csw/csw`) | Returned HTTP `503` (maintenance) during harvest. | `568` JAXA G-Portal collections via `14_ceos_fedeo_agencies`, primary `30_jaxa_earth_stac` (`112`), and DIAS releases in Source `26`. |
| **ISRO Portals** (`Bhoonidhi`, `Bhuvan`, `MOSDAC`) | `Bhoonidhi` STAC returns HTTP `403` (authentication required), `Bhuvan` CSW returns `404`, and `MOSDAC` exposes no open collection API. | Selected ISRO L3/L4 holdings via `14_ceos_fedeo_agencies` and `12_ceos_cgms_wgclimate_ecv`. |
| **Chinese EO Portals** (`CMA/NSMC`, `CASEarth`, `TPDC`, `ScienceDB`) | National web portals lack open unauthenticated STAC/CSW collection APIs; `TPDC` prefix `10.11888` has `0` entries in DataCite; `ScienceDB` (`10.57760`) mixes general multidisciplinary supplements with EO. | Targeted peer-reviewed releases (`CoSEB`, `PML-V2.2`, `GLC_FCS30D`, `CLCD`) via Sources `24`, `26`, and `29`. |
| **KARI / KMA & SANSA** | KOMPSAT imagery is distributed commercially via partners without an open L3/L4 STAC; no open unauthenticated SANSA L3/L4 collection API. | Open continental African EO datacubes are covered via `31_digital_earth_africa_and_pacific`. |
| **MSC GeoMet** (`api.weather.gc.ca/collections`) | Predominantly hydrometric/climate in-situ station feeds and real-time operational NWP forecasts (outside satellite L3/L4 archive scope). | Canadian satellite EO datacube products are covered via primary `32_nrcan_ccmeo_stac`. |
| **CEMS On-Demand Mapping Portal** (`mapping.emergency.copernicus.eu`) | `/stacapi/` returns HTTP `403`, and `/activations/api/activations/` exposes `1,070` individual event-specific disaster activations (`EMSR*`) rather than continuous gridded archives. | `395+` historical CEMS Rapid Mapping & Risk and Recovery activations are indexed via `07_ec_jrc_data_catalogue`; continuous CEMS products (`GFM`, `GloFAS`, `EFAS`, `EDO/GDO`, `EFFIS/GWIS`, `GHSL`) are indexed via Sources `05`–`07`. |
| **Copernicus Security Service** (`EMSA CleanSeaNet`, `Frontex`, `SatCen`) | Restricted operational security services with no open public dataset API. | Documented as out-of-scope in [`04_validation_and_review/eu_policy_products_benchmark_audit.csv`](04_validation_and_review/eu_policy_products_benchmark_audit.csv). |
| **Commercial VHR Tasking Portals** (`Maxar`, `Planet`, `Airbus`, `Capella`, `ICEYE`) | Proprietary commercial tasking and scene archives without open public L3/L4 metadata APIs. | Outside open public catalogue scope. |
| **Instantaneous L2 Event Streams & Raw Archives** (e.g., EUMETSAT MTG FCI Active Fire `EO:EUM:DAT:0801`/`0682`, raw Sentinel/Landsat L1–L2, aerial orthophotos, discrete stations) | Raw L0–L2 sensor imagery, aerial photography, discrete station logs, real-time NWP feeds, and instantaneous L2 pixel detections (MTG FCI Active Fire is classified as `Level 2` upstream) fall outside the L3/L4 inclusion criteria. | Excluded deterministically and logged in `02_preparation/data/excluded_records_log.parquet`; gridded daily/monthly fire CDRs (`EO:EUM:DAT:0148`, `MSG:FRPGRID-SEVIRI`) are retained. |
| **Standalone Web-Only Climate Archives** (`Berkeley Earth`, `GloH2O MSWEP/MSWX`, etc.) | Distributed as direct web downloads or cloud folders without machine-readable STAC/REST/CSW collection APIs (`Berkeley Earth` is also primarily gridded in-situ station interpolation). | Outside the automated API harvest boundary. |

Full per-source API endpoints, processing-level rules, and attribution logic are documented in [`docs/SOURCES.md`](docs/SOURCES.md).

---

## 4. Repository Layout

```text
.
├── README.md                                         # Overview, excluded catalogues & local rebuild guide
├── LICENSE / LICENSE-DATA / CITATION.cff             # MIT (code), CC-BY-4.0 (data & docs), CFF 1.2.0
├── MANIFEST.json & SHA256SUMS.txt                    # Cryptographic provenance & API timestamp manifest
├── build_manifest.py                                 # `uv run` script regenerating MANIFEST.json & SHA256SUMS.txt
├── docs/
│   ├── SOURCES.md                                    # Processing-level ontology, 32-source registry & boundaries
│   ├── index.html & app.js                           # Static Tailwind CSS catalogue viewer (GitHub Pages `/docs`)
│   ├── sources.html                                  # HTML / Raw Markdown viewer for docs/SOURCES.md
│   ├── build_viewer_data.py                          # `uv run` script exporting sharded index & detail chunks
│   └── data/                                         # Sharded JSON index & detail chunks for the static viewer
│
├── 01_raw/                                           # Native API harvests (mine.py, README.md, last_access.txt, data/)
│   ├── 01_copernicus_cmems/ .. 29_awesome_gee_community/
│   ├── 30_jaxa_earth_stac/                           # JAXA Earth API COG v1 STAC
│   ├── 31_digital_earth_africa_and_pacific/          # Digital Earth Africa ODC & Digital Earth Pacific STAC
│   └── 32_nrcan_ccmeo_stac/                          # Natural Resources Canada (CCMEO) Datacube STAC
│
├── 02_preparation/                                   # Deterministic cleaning, normalization & L3/L4 filtering
│   ├── prepare_all.py & README.md
│   ├── exclusion_audit_summary.csv                   # Per-source L0–L2, in-situ & non-EO accounting
│   ├── data/                                         # Per-source .parquet/.jsonl + excluded_records_log.parquet
│   └── reference_taxonomies/                         # WMO OSCAR/Space variables & satellites tables
│
├── 03_harmonization/                                 # Unified L3/L4 schema, ontology & institutional attribution
│   ├── harmonize_catalogue.py, verify_catalogue.py & README.md
│   ├── provider_pool_and_sentinel_summary.csv
│   ├── schema_completeness_report.csv                # Non-null & null counts across all 78 harmonized columns
│   └── data/
│       ├── eo_l3_l4_all_catalogue_records.parquet    # 39,202 harmonized L3/L4 records (78 columns) + .jsonl parts
│       ├── eo_l3_l4_canonical_products.parquet       # 38,567 canonical L3/L4 records (80 columns)
│       └── eo_l3_l4_product_variants.parquet         # 60 explicit sub-product variant rows (19 columns)
│
└── 04_validation_and_review/                         # Automated QA & targeted domain verification suites
    ├── validate_catalogue_benchmarks.py & README.md
    ├── verified_releases_baseline.json, verified_record_ids.json, release_field_diff_audit.json
    ├── eu_policy_products_benchmark_audit.csv        # 86-family EU policy & operational service matrix
    ├── gcos_ecv_domain_coverage.csv                  # 40/40 CEOS/CGMS WGClimate GCOS ECV matrix
    └── independent_product_version_benchmark.csv & _audit.csv  # 79-case curated product & holdout checks
```

---

## 5. Rebuilding Locally (`uv run`)

All scripts use PEP 723 inline dependency metadata and require only [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Re-mine any upstream endpoint (updates data/ and last_access.txt):
uv run 01_raw/01_copernicus_cmems/mine.py

# 2. Clean, normalize, and filter L3/L4 records across all 32 raw sources:
uv run 02_preparation/prepare_all.py

# 3. Harmonize into the unified schema and canonical table:
uv run 03_harmonization/harmonize_catalogue.py

# 4. Run structural verification and domain benchmark suites:
uv run 03_harmonization/verify_catalogue.py
uv run 04_validation_and_review/validate_catalogue_benchmarks.py

# 5. Rebuild the static GitHub Pages viewer assets (`docs/data/`) and preview locally:
uv run docs/build_viewer_data.py
python3 -m http.server 8080 -d docs

# 6. Regenerate cryptographic MANIFEST.json and SHA256SUMS.txt:
uv run build_manifest.py
```

> [!IMPORTANT]
> **Manual Rebuilds & GitHub Pages Sync:** Upstream API harvesting is intentionally manual—external agency endpoints evolve, paginate differently, or experience transient outages, so no GitHub Actions workflows run in this repository. Whenever the catalogue is rebuilt (`prepare_all.py` -> `harmonize_catalogue.py` -> validation -> `build_manifest.py`), also run `uv run docs/build_viewer_data.py` so the GitHub Pages viewer (`docs/data/`) matches the Parquet tables.

---

## 6. Interactive Static Viewer (`docs/`)

The `docs/` directory hosts a zero-backend Tailwind CSS viewer published directly from `main` (`/docs` with `.nojekyll`) covering all **38,567 canonical records** across **31 sources** (`docs/data/index_meta.json`, 4 index shards, and 65 on-demand provenance detail chunks). It supports multi-term search, faceted filtering (provider pool, jurisdiction tier, processing level, Sentinel mission/lineage, output domain, grid spacing, temporal coverage year, cadence, maturity, strict EO, DOI presence), column sorting, shareable URL state, CSV and Parquet export, an HTML/Markdown toggle for the source registry (`docs/sources.html`), and a slide-over provenance panel with a **MapLibre GL + OpenFreeMap** bounding-box viewer (handling antimeridian crossings).

---

## 7. Licensing & Citation

- **Pipeline Code (`*.py`)**: [MIT License](LICENSE).
- **Catalogue Data & Documentation**: [CC-BY-4.0](LICENSE-DATA). Individual EO datasets retain their upstream provider licenses (`license` column).
- **Citation**: See [`CITATION.cff`](CITATION.cff).
