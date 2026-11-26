# Global Harmonized L3/L4 Earth Observation Product Catalogue

A fully reproducible, deterministic **global Level-3 and Level-4 Earth Observation (EO) product discovery catalogue** harvested directly from **34 public space-agency, European Union, United Nations, and cloud metadata APIs** (plus the WMO OSCAR/Space variable and satellite taxonomy). Includes a hybrid search engine **evaluated against curated family rules** achieving in best hybrid config (EmbeddingGemma 2 + DuckDB BM25 + Jina v3.5): 85.45% Fam-nDCG@10 (76.67% P@1, 96.39% family Recall@10, 100.00% Hit@10) on a 60-query retrospectively audited cross-domain holdout.

- **Interactive catalogue viewer:** [do-me.github.io/global-EO-products-L3-L4](https://do-me.github.io/global-EO-products-L3-L4/)
- **In-browser Advanced Search (EO BM25 + 512-d EmbeddingGemma 2 + optional Qwen3-0.6B):** [do-me.github.io/global-EO-products-L3-L4/search/](https://do-me.github.io/global-EO-products-L3-L4/search/)
- **Source registry & ontology:** [`docs/SOURCES.md`](docs/SOURCES.md)
- **Local Python hybrid search quickstart:** [`06_semantic_hybrid_search/`](06_semantic_hybrid_search#1-quick-start-uv-run)

<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/dfc5aad0-a119-403d-8ce8-e812c3de87e8" />

---

## 1. What This Catalogue Is — and Is Not

This repository helps EO researchers and analysts **discover candidate gridded L3/L4 products across institutional boundaries** (Copernicus Services, European non-Copernicus institutions, and non-EU/international providers) and trace **Copernicus Sentinel input dependencies**. It is a structured starting point for discovery and screening, not a substitute for primary product documentation.

Keep three properties in mind when working with these tables:

1. **Catalogue-inventory counts, not distinct global products.** The pipeline favors high recall so eligible L3/L4 products are not silently dropped. Deduplication is deliberately conservative (matching exact normalized DOI or pool + normalized full title). Unlinked cloud mirrors without shared DOIs, versioned releases, multi-level agency collections, and asset-level sub-collections in `29_awesome_gee_community` remain as separate rows. Treat pool totals as harmonized catalogue-inventory counts and perform task-specific screening and deduplication against primary documentation.
2. **Structured metadata is only as complete as upstream APIs.** Agency and cloud portals frequently omit structured fields: in the canonical table, `processing_level` is null on `73.7%` of rows, `adjudicated_output_grid_meters` on `70.8%`, `temporal_cadence_class` on `69.6%`, `doi` on `68.3%`, and `platform_sensor_raw` on `50.29%` (leaving unattributed records in `unassessed_no_platform_metadata` rather than `explicit_non_sentinel`). Missing or conflicting upstream attributes remain `null`, while text-extracted hints are stored in separate provenance columns. Pair free-text search with faceted filters rather than relying on structured fields alone.
3. **Open-API harvest boundary, not an exhaustive world census.** Only public, unauthenticated machine-readable endpoints (STAC, REST, CSW, CKAN, Elasticsearch, DCAT-AP, DataCite, Zenodo) are harvested directly. Catalogues and operational streams excluded from direct harvesting are listed explicitly in Section 3 below.

---

## 2. Inventory at a Glance

| Stage / Dimension | Count | Notes |
| :--- | ---: | :--- |
| **Upstream sources (`01_raw/`)** | **35** | `34` EO product catalogues + `1` WMO OSCAR/Space reference taxonomy (`1,300` variable/satellite rows) |
| **Raw records parsed** | **50,263** | Parsed across `34` product catalogues (`52,612` total retrieved across all `35` `last_access.txt` endpoints) |
| **Prepared L3/L4 records (`02_preparation/`)** | **40,116** | Excludes `4,652` logged L0–L2/aerial/in-situ records (`2,624` L0–L2/aerial + `2,028` in-situ), `4,694` pre-excluded non-EO records, and `801` within-source duplicates |
| **Canonical catalogue records (`03_harmonization/`)** | **39,472** | `33,248` strict EO + `6,224` flagged sensitivity/hybrid/reference (`611` mirror groups collapsing `644` duplicates); plus `60` explicit sub-product variant rows |
| — `copernicus_services` | **3,792** | `CMEMS`, `CLMS`, `C3S`, `CAMS`, `CEMS` (incl. `GHSL`, `GFM`, `GloFAS`, `EFAS`, `EDO/GDO`, `EFFIS/GWIS`), and `COP_DEM` |
| — `eu_non_copernicus` | **5,066** | `2,898` EU-27 national/EC agencies (JRC, EEA, EMODnet, DLR, CNES GEODES/Theia, VITO, OpenGeoHub), `1,760` European space/met IGOs (ESA, EUMETSAT/SAFs, ECMWF DestinE), `408` Copernicus Associated States (UK, NO, IS under Reg. (EU) 2021/696) |
| — `non_eu` | **30,614** | `28,267` third-country agencies & consortia (NASA, NOAA, USGS, U.S. GHG Center, JAXA, NRCan, GA DEA, DE Africa/Pacific, INPE, WRI, Esri), `2,322` UN/international bodies (FAO GISMGR & Geospatial CKAN, UNEP-WCMC, WMO, GEO BON), `25` Swiss institutions (`european_esa_non_copernicus_state`) |
| **Documented Sentinel input (canonical)** | **2,280** | `673` Copernicus Services, `592` EU non-Copernicus, `1,015` non-EU (`1,607` outside Copernicus); separates `confirmed_input_mission` / `retrospective_or_mixed_epoch_series` from `explicit_non_sentinel`, target-grid references (`HLSL30`), and `unassessed_no_platform_metadata` |
| **Internal QA (`04_validation_and_review/`)** | **PASS** | `25` structural/ontology suites, `26` QA assertions, `77/77` checked EU policy families, `40/40` GCOS ECV groups, `79/79` curated product & holdout checks |
| **External policy validation (`05_external_policy_validation/`)** | **90.4% strict / 96.1% broad** | `208/230` (`90.4%` exact/compatible) and `221/230` (`96.1%` incl. related) across `8` strictly independent international & U.S. policy benchmarks (`313/335 = 93.4%` strict / `326/335 = 97.3%` broad across all `10` tracks; `1,442/1,739 = 82.9%` strict / `1,526/1,739 = 87.8%` broad USGEO EOA 2023 KPSO-weighted) |

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
| **Instantaneous L2 Event Streams & Raw Archives** (e.g., EUMETSAT MTG FCI Active Fire `EO:EUM:DAT:0801`/`0682`, NASA PO.DAAC `G-REALM` `10.5067/UCLRS-GREV2` classified as `Level 2`, raw Sentinel/Landsat L1–L2, aerial orthophotos, discrete stations) | Raw L0–L2 sensor imagery, aerial photography, discrete station logs, real-time NWP feeds, and upstream `Level 2` products/event streams fall outside the L3/L4 inclusion criteria. | Excluded deterministically and logged in `02_preparation/data/excluded_records_log.parquet`; gridded daily/monthly fire CDRs (`EO:EUM:DAT:0148`, `MSG:FRPGRID-SEVIRI`) and Pre-SWOT L3/L4 lake area/storage series (`10.5067/UCLRS-AREV2`, `STOV2`) are retained. |
| **Auth-Gated Inland Water Altimetry Portals** (`Hydroweb.next`, `DAHITI`) | Require personal API keys/registration (`HTTP 403` / connection reset without auth) and serve virtual-station inland water height time series rather than global gridded raster archives. | Surface water occurrence and altimetry-derived L3/L4 lake products are covered via `02_copernicus_clms` (Lake/River Water Level), `14_ceos_fedeo_agencies` (`Lakes_cci`), `16_nasa_cmr` (`Pre-SWOT`), and `22_data_terra_inpe_dea`. |
| **NASA VEDA Dashboard STAC** (`openveda.cloud/api/stac`) | Mixes single-event disaster visualizations, Doppler radar/tornado storymap layers, demo collections, and commercial Planet/Maxar subsets alongside mirrored CMR holdings. | Federal multi-agency L3/L4 GHG inventories, inversions, and flux grids are harvested cleanly via primary `33_us_ghg_center_stac` (`earth.gov/ghgcenter`), and NASA DAAC archives via `16_nasa_cmr`. |
| **Standalone Web-Only Climate & Thematic Archives** (`Berkeley Earth`, `GloH2O MSWEP/MSWX`, `S3 World Fire Atlas`, `DHI Global Wetland Map`, `ESA WorldWater`) | Distributed via web dashboards, direct downloads, or validation-only deposits without open machine-readable STAC/REST/CSW collection APIs or DataCite dataset DOIs (`Berkeley Earth` is also primarily gridded in-situ station interpolation). | Outside the automated API harvest boundary; parallel gridded datasets (`GLEAM4`, `CHIRPS`, `GPM IMERG`, `Fire_cci`, `GLC-SHARE`, `GMIA v5`, `MIRCA2000`, `MIRCA-OS`, `LANID-US`, `MIrAD-US`, `WFDEI`) are indexed via Sources `01`–`35`. |

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
│   ├── SOURCES.md                                    # Processing-level ontology, 35-source registry & boundaries
│   ├── index.html & app.js                           # Static Tailwind CSS catalogue viewer
│   ├── sources.html                                  # HTML / Raw Markdown viewer for docs/SOURCES.md
│   ├── build_viewer_data.py                          # `uv run` script exporting sharded index & detail chunks
│   └── data/                                         # Sharded JSON index & detail chunks for the static viewer
├── web_search/                                       # `pnpm` + Vite static browser-only Advanced Search (`/search/`)
│   ├── search/index.html, vite.config.js & package.json  # Client-side EO BM25 + Transformers.js v4 (WebGPU/WASM) UI
│   ├── src/                                          # search-core.js, search.worker.js, exporters.js, main.js
│   └── tests/engine.test.mjs                         # Automated BM25, 512-d int8, Hard-Limits & collapse tests
├── .github/workflows/pages.yml                       # CI: `pnpm` build + release-index export + tests + Pages deploy
│
├── 01_raw/                                           # Native API harvests (mine.py, README.md, last_access.txt, data/)
│   ├── 01_copernicus_cmems/ .. 29_awesome_gee_community/
│   ├── 30_jaxa_earth_stac/                           # JAXA Earth API COG v1 STAC
│   ├── 31_digital_earth_africa_and_pacific/          # Digital Earth Africa ODC & Digital Earth Pacific STAC
│   ├── 32_nrcan_ccmeo_stac/                          # Natural Resources Canada (CCMEO) Datacube STAC
│   ├── 33_us_ghg_center_stac/                        # U.S. Greenhouse Gas Center (NASA / EPA / NIST / NOAA) STAC
│   ├── 34_fao_geospatial_catalog_ckan/               # FAO Hand-in-Hand Geospatial & Map Catalog (CKAN / ISO-19115)
│   └── 35_cnes_geodes_stac/                          # CNES GEODES STAC (THEIA / Form@Ter / SAGSA)
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
│       ├── eo_l3_l4_all_catalogue_records.parquet    # 40,116 harmonized L3/L4 records (78 columns) + .jsonl parts
│       ├── eo_l3_l4_canonical_products.parquet       # 39,472 canonical L3/L4 records (80 columns)
│       └── eo_l3_l4_product_variants.parquet         # 60 explicit sub-product variant rows (19 columns)
│
├── 04_validation_and_review/                         # Internal QA & domain verification suites
│   ├── validate_catalogue_benchmarks.py & README.md
│   ├── verified_releases_baseline.json, verified_record_ids.json, release_field_diff_audit.json
│   ├── eu_policy_products_benchmark_audit.csv        # 86-family EU policy & operational service matrix
│   ├── gcos_ecv_domain_coverage.csv                  # 40/40 CEOS/CGMS WGClimate GCOS ECV matrix
│   └── independent_product_version_benchmark.csv & _audit.csv  # 79-case curated product & holdout checks
│
├── 05_external_policy_validation/                    # External cross-validation against 9 policy-to-EO benchmarks
│   ├── 01_download_benchmarks.py, 02_extract_products.py, 03_validate_against_catalogue.py & README.md
│   ├── raw/                                          # 44 downloaded primary benchmark files (CEOS, UN, EC, USGEO, NASA)
│   └── results/                                      # extracted_policy_products.csv, validation_matches.csv, validation_summary.csv, gap_analysis.csv
│
└── 06_semantic_hybrid_search/                        # Local hybrid dense + BM25 + reranker search engine, web UI & evaluation
    ├── README.md                                     # Quick-start, methodology, validation design & reproducible audits
    ├── eo_search_lib.py, 04_search_cli.py, 04_search_server.py & web/  # CLI + local FastAPI/Tailwind/MapLibre search UI
    ├── 01_..11_*.py                                  # Indexing, reranking, blind holdout evaluation & 8 verification audits
    └── artifacts/                                    # Frozen SHA-256 manifests, benchmark JSONs, results & audit outputs
```

---

## 5. Rebuilding Locally (`uv run`)

All scripts use PEP 723 inline dependency metadata and require only [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Re-mine any upstream endpoint (updates data/ and last_access.txt):
uv run 01_raw/01_copernicus_cmems/mine.py

# 2. Clean, normalize, and filter L3/L4 records across all 35 raw sources:
uv run 02_preparation/prepare_all.py

# 3. Harmonize into the unified schema and canonical table:
uv run 03_harmonization/harmonize_catalogue.py

# 4. Run structural verification, internal QA, and external policy validation:
uv run 03_harmonization/verify_catalogue.py
uv run 04_validation_and_review/validate_catalogue_benchmarks.py
uv run 05_external_policy_validation/03_validate_against_catalogue.py

# 5. Rebuild the static GitHub Pages viewer + in-browser Advanced Search locally:
uv run docs/build_viewer_data.py
pnpm --dir web_search install --frozen-lockfile && pnpm --dir web_search build
uv run 06_semantic_hybrid_search/export_browser_index.py --out-dir dist/data/search --dim 512
pnpm --dir web_search test
python3 -m http.server 8080 -d dist

# 6. Launch the local Python search server (auto-downloads & SHA-256-verifies v4.3.0-indexes; or `--rebuild-index`):
uv run 06_semantic_hybrid_search/04_search_server.py --port 8080 --open
# Optional local index rebuild (auto-selects MLX / CUDA / CPU):
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py   # or `--bm25-only` (~36 s, zero models)

# 7. Regenerate cryptographic MANIFEST.json and SHA256SUMS.txt:
uv run build_manifest.py
```

> [!IMPORTANT]
> **Manual Harvesting vs. Automated Pages CI:** Upstream API harvesting is intentionally manual—external agency endpoints evolve, paginate differently, or experience transient outages, so no scheduled harvesting workflows run in this repository. Static site assembly and deployment run automatically in GitHub Actions ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)) using `pnpm` and `uv`: CI builds `web_search/` + `docs/`, fetches the pinned `v4.3.0-indexes` release artifacts to export the compact 512-d browser search index (`dist/data/search/`, keeping `main` free of binary index blobs), runs `pnpm --dir web_search test`, and publishes the artifact via `actions/deploy-pages`. Whenever the catalogue Parquet tables are rebuilt locally, also run `uv run docs/build_viewer_data.py` so `docs/data/` stays synchronized.

---

## 6. Interactive Catalogue Viewer & Browser Advanced Search (`docs/`, `web_search/`)

The GitHub Pages site serves two zero-backend apps sharing the exact same **39,472 canonical products** (`34` sources), structured filter criteria (including the reconciled `Hide Deprecated / Superseded` toggle removing `622` superseded/obsolete editions), MapLibre GL bounding-box provenance drawer, and CSV/JSON/Excel/Parquet exports:

- **Catalogue Viewer ([`/`](https://do-me.github.io/global-EO-products-L3-L4/)):** Fast multi-term string filter, faceted browsing, sortable columns, and HTML/Markdown source registry ([`sources.html`](https://do-me.github.io/global-EO-products-L3-L4/sources.html)).
- **In-Browser Advanced Search ([`/search/`](https://do-me.github.io/global-EO-products-L3-L4/search/)):** Replaces plain substring matching with client-side hybrid retrieval inside a Web Worker (`@huggingface/transformers` v4, WebGPU with WASM fallback) backed by CI-exported indexes (`3.1 MB` lossless 3-field EO BM25 with `0.00` DuckDB divergence, `15.0 MB` 512-d Matryoshka `int8` vectors, `0.4 MB` filter/family metadata):
  - **EO BM25 Lexical** runs immediately (`~14 ms`);
  - **Semantic Hybrid** fetches text-only `onnx-community/embeddinggemma-2-ONNX` (`q4`, ~175 MB, `Apache-2.0`) directly from Hugging Face and fuses 512-d vectors with BM25 via RRF;
  - **Optional Stage-2 Reranker** (off by default) pulls `onnx-community/Qwen3-Reranker-0.6B-ONNX` (`q4`, ~995 MB, `Apache-2.0`) for higher Top-10 precision at slower per-query latency, with Verified Hard-Limits and guarded family collapsing.

---

## 7. Licensing & Citation

- **Pipeline Code (`*.py`)**: [MIT License](LICENSE).
- **Catalogue Data & Documentation**: [CC-BY-4.0](LICENSE-DATA). Individual EO datasets retain their upstream provider licenses (`license` column).
- **Citation**: See [`CITATION.cff`](CITATION.cff).
