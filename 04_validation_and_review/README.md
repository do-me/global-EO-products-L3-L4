# Quality Assurance & External Coverage Validation

Automated structural verification and external benchmark validation for the Global Harmonized L3/L4 Earth Observation Product Catalogue, executed via:

```bash
uv run 03_harmonization/verify_catalogue.py
uv run 04_validation_and_review/validate_catalogue_benchmarks.py
```

---

## 1. Summary Inventory Statistics

| Metric | Value | Source Artifact |
|---|---:|---|
| Primary & federated sources harvested (`01_raw/`) | `29` | 28 EO product sources + 1 WMO OSCAR/GCOS taxonomy source |
| Raw records parsed (`01_raw/`) | `47,722` | Parsed across 28 EO sources (`50,070` total retrieved across all 29 `last_access.txt` endpoints before Source 26 deduplication) + `1,300` WMO OSCAR reference records (`237` variables, `1,063` satellites) |
| Prepared L3/L4 records (`02_preparation/`) | `38,169` | Stored per source in `02_preparation/data/` |
| Logged excluded records (`02_preparation/`) | `4,552` | `02_preparation/data/excluded_records_log.parquet` (+ `4,200` pre-excluded non-EO records across JRC, GEO KH, WRI) |
| Harmonized L3/L4 records (`03_harmonization/`) | `38,169` | [`03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet`](../03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet) (`56` columns) |
| Canonical L3/L4 products (`03_harmonization/`) | `37,605` | `540` mirror groups collapsing `564` duplicate records ([`eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet), `58` columns) |
| Canonical pool: `copernicus_services` | `3,520` | `L3`: `240` (`137` Sentinel), `L3; L4`: `6` (`2` Sentinel), `L4`: `808` (`86` Sentinel), `L1`: `1` (`1` Sentinel), unlabelled `null`: `2,465` (`446` Sentinel) — **`672` Sentinel-dependent** |
| Canonical pool: `eu_non_copernicus` | `5,260` | `L3`: `411` (`110` Sentinel), `L4`: `195` (`29` Sentinel), `L2`: `37` (`11` Sentinel), `L1`: `1` (`1` Sentinel), unlabelled `null`: `4,616` (`427` Sentinel) — **`578` Sentinel-dependent** (`eu27_national_or_eu_agency`: `3,068`; `european_space_met_igo`: `1,811`; `copernicus_associated_state`: `381`) |
| Canonical pool: `non_eu` | `28,825` | `L3`: `5,940` (`618` Sentinel), `L4`: `2,282` (`45` Sentinel), `L2`: `114` (`1` Sentinel), `L1`: `13` (`0` Sentinel), unlabelled `null`: `20,476` (`263` Sentinel) — **`927` Sentinel-dependent** (`third_country_non_european`: `27,326`; `un_or_international_body`: `1,479`; `european_esa_non_copernicus_state`: `20`) |

---

## 2. Automated Quality Assurance Suites (`19 + 19 PASS`)

1. **`03_harmonization/verify_catalogue.py`** (`19` structural and ontology assertions):
   - Verifies completeness of all `29` source directories (`mine.py`, `README.md`, `last_access.txt`, non-empty `data/`) and enforces GitHub's single-file size bound (`<= 48 MB`).
   - Verifies Sentinel-dependency detection (with mission launch-year, `HLSL30` target-grid exclusion, and contextual-clause guards), producer-origin pool and `provider_jurisdiction_tier` attribution, operational maturity classification (`operational_ongoing_series` vs. `completed_or_mission_bounded_release`, `research_or_benchmark_release`, `planned_or_preoperational`, `deprecated_or_superseded`, and `unknown`, with zero `operational_ongoing_series` rows lacking `is_ongoing_archive` or having closed `<= 2024` end dates), symmetric `L3_ARD_REFLECTANCE` tagging across pools (without leaking derived biophysical indices), leading-dot, anisotropic (`0.625 x 0.5`), and fractional-degree resolution parsing (`27,830.0 m`, CLMS Burnt Area `1/336°` = `331.31 m`), Suomi-NPP L3 and C3S `satellite-*` honest level provenance, full NOAA `summary: false` (`99.3%` descriptions, `99.1%` MultiPolygon/Polygon bounds) and NASA CMR UMM-JSON (`93.4%` DOIs, `2,335` native grid resolutions) metadata recovery, GEE structured `gee:interval` cadence extraction, separation of sampling cadence from rolling aggregation windows and publication update frequency, all-56-column `schema_completeness_report.csv` coverage, supersession linking (`SOCATv2025` -> `SOCATv2026`), and high-recall benchmark collections (`FAO/WAPOR`, `Sentinel-3 SYNERGY V10/VG1`, `TransCom 3`, `SEDAC`, `INPE BDC`, `FLUXCOM-X-BASE`, `LCSPP`, `TCSIF v1`, `GWELD`, `EO:EUM:DAT:0088`, `TROPOSIF`, `SMOS-IC v2`).
2. **`04_validation_and_review/validate_catalogue_benchmarks.py`** (`19` integrity assertions):
   - Zero duplicate `raw_id` within any source catalogue.
   - Zero Level-0, SLC-off/Burst, swath grid, or test placeholder contamination in canonical titles.
   - Zero non-EO laboratory/nuclear/LCA entries from institutional repositories.
   - Zero implausible unit-conversion, fractional-degree (`<= 600 km` ceiling), cartographic scale-denominator (`EEA >= 500 km` non-degree), >10x title-token discrepancy (conflicting pairs null-flagged cleanly via `spatial_resolution_conflict`), or leading-dot degree artifacts (`AE_DyOcn`/`AU_DyOcn` = `27,830.0 m`) in spatial resolution fields.
   - Zero in-situ/campaign/socioeconomic/physical-sample/aerial records (`INSITU_*`, `radiosonde`, `Argo float`, `Polarstern`, `XBT`/`CTD`/buoy logs, `Geosample Data`, `DWD-GPCC`, `ISTI`, `USGSPHOTOS`/`NASAPHOTOS`, `ARDECO`, `LUCAS Copernicus`, `gridded GDP`) marked `strict_eo_c1_eligible = True`.
   - Zero within-catalogue `canonical_product_key` collisions.
   - Zero null/empty product titles (`38,169 / 38,169` non-null).
   - Zero invalid coordinates (`bbox`), unprefixed or preview-image (`.png`/`.jpg`/`.gif`/`/browse`) `access_url` links, non-canonical `doi` formats, or inverted ISO dates.
   - Zero producer-pool overrides on Copernicus/ESA/EU/INPE/FedEO records and valid `provider_jurisdiction_tier` invariants; zero future scenarios (`temporal_end_year >= 2028`) tagged as `is_ongoing_archive = True`; zero Suomi-NPP L3-to-L4, acronym homograph (`CAS/IGSNRR/PML`, NASA `HS3`), or `temporal_start_year > 2026` leaks.
   - Verified deep provider metadata recovery (`>= 98%` NOAA descriptions, `>= 90%` CMR DOIs, `>= 2,300` CMR native resolutions) and zero multi-day/monthly-title vs. `sub_daily_or_daily` cadence contradictions (excluding rolling-window `MCD43A4`/`VNP43`), zero `operational_ongoing_series` rows with null `is_ongoing_archive`, zero pre-2014 bounded records tagged `uses_sentinel_data = True`, zero `temporal_resolution_raw == 'cadence'` GEE interval losses, and zero explicit L4-to-L3 downgrades.
   - Verified closed-archive maturity (`op_closed = 0`), European/polar geographic coverage disambiguation (`eu_glob = 0`, `polar_bad = 0`), EU benchmark routing (`VODCA`, `TROPOSIF`, `GlobSnow v3.0`), `BICEP` Sentinel-3 lineage with literature citation-year stripping, and suppression of single-epoch multi-year acquisition windows.
   - Verified independent review hardening: `0` HTML tags/entities in titles/descriptions, `0` null Zenodo URLs (`ZENODO:8242928` `TCSIF v1` present with `L3` / `L2; L3`), `CBERS4-MUX-2M-1` title-token guard, `0` single-sensor geostationary disks tagged `global`, `0` ARD biophysical leaks, `0` `HLSL30` Landsat-only Sentinel false positives, and `C3S SST` preserving `L3; L4`.

---

## 3. External Domain Coverage Benchmarks

### 3.1 EU Policy & Operational Service Product Families (`N = 86`)
Evaluated across `86` consolidated EU environmental policy, Copernicus operational service, and global scientific benchmark product families in [`eu_policy_products_benchmark_audit.csv`](eu_policy_products_benchmark_audit.csv):

- **In-Scope Open L3/L4 EO Product Families**: **`77 / 77` (`100.0%`)** present (`IN_CATALOGUE`) across CLMS (including all 9 pan-European 10 m HRL Croplands Crop Types & Cropping Patterns layers), CEMS/JRC (GFM, GloFAS, EFAS/HERA, EDO/GDO, EFFIS/GWIS, GHSL, GSWE 30 m, TMF v2025, Rapid Mapping & RRM, MARS/ASAP, ESDAC), C3S (ERA5, ERA5-Land, CERRA, seasonal forecasts, PECD, CMIP6, ECVs, UTCI, agroclimatic indicators), CAMS (emission inventories, GHG/AQ forecasts & reanalyses, inversions, CRS), CMEMS (Physics, BGC, HR-OC 100 m, GlobColour, OMIs, SDB 100 m), Copernicus DEM / EU-DEM, EMODnet, and global external benchmarks (VIIRS `VNP46`, USGS Landsat C2 L3, `G3P` / `GRACE` TWS, `FLUXCOM-X-BASE`, `LCSPP`/`LCSIF`/`LCREF`, `GWELD`, EUMETSAT `EO:EUM:DAT:0088` MSG LST Level 3, `TROPOSIF` SIF, and `SMOS-IC v2` soil moisture / L-VOD).
- **Sensitivity / Non-EO Tagged & Related Entries**: `5` families (`INSITU_*_OA_*` gridded interpolations, `EDGAR`/`CEDS` emission inventories, CAMS Carbon Monitor statistical grid, Eurostat/JRC socio-economic support, and OSM-derived coastal road reference).
- **Expected Out-of-Scope**: `3` families (restricted Copernicus Security Services / EMSA CleanSeaNet; interactive web-only CAMS Methane Hotspot Explorer; EEA tabular WEI+/WISE Flood Risk/EUCRA reports).
- **Excluded Imagery**: `1` family (`CDSE Sentinel Level-1 and Level-2 Operational Imagery`, verified as `CORRECTLY_EXCLUDED_IN_AUDIT_LOG` in `excluded_records_log.parquet`).

### 3.2 GCOS Essential Climate Variables (`N = 40`)
Evaluated against the CEOS/CGMS WGClimate GCOS Essential Climate Variable (ECV) taxonomy using explicit domain-qualified variable patterns in [`gcos_ecv_domain_coverage.csv`](gcos_ecv_domain_coverage.csv), reporting both total canonical matches and independent non-WGClimate matches: **`40 / 40`** GCOS ECV groups are covered across both the canonical inventory and independent non-WGClimate sources.

### 3.3 Independent Product & Version-Level Recall / Attribute Audit (`N = 26`)
Evaluated across `26` independently specified product releases and versioned series (spanning Zenodo/DataCite SIF releases including `TCSIF v1` `10.5281/zenodo.8242928` and `TCSIF_005`, `GOSIF`, `TROPOSIF`, `LCSPP`/`LCREF`/`LCSIF`, multi-source EO conflicts/titles `N18_AVH13C1`, `CBERS4-MUX-2M-1`, `AERDB_M3_ABI_G16`, `MCD43A4`, `HLSL30`, `Boreal_AGB_Density_ICESat2_2186`, `SOCATv2025`, `C3S SST`, `C3S ocean-colour`, `CoCO2-MOSAIC`, and `Seasonal Trajectories`) in [`independent_product_version_benchmark.csv`](independent_product_version_benchmark.csv) and [`independent_product_version_audit.csv`](independent_product_version_audit.csv): **`26 / 26`** version-level recall and **`26 / 26`** exact harmonized attribute accuracy.

---

## 4. Scope, Design Choices & Known Limitations

1. **High Recall Over Precision (Intentional Design)**: To avoid silently omitting candidate products, filtering and canonical deduplication (`canonical_product_key` matching exact normalized DOI or pool + normalized full title) prioritize high recall. Consequently, unlinked cloud mirrors lacking shared DOIs and with differing upstream titles, versioned product releases (`SOCAT`, closed/active `CLMS` series tracked via `is_superseded_edition` and `superseded_by_record`), multi-level collections, asset-level shards in community catalogs (`29_awesome_gee_community`), and tagged sensitivity records remain discoverable for downstream expert screening.
2. **Metadata Completeness & Provenance (`let null be null`)**: Upstream APIs frequently omit horizontal grid spacing (`spatial_resolution_meters`), archive dates (`temporal_start_year`), DOIs, or explicit processing levels (`processing_level`). Omitted or unlabelled attributes remain `null` (with explicit provenance in `processing_level_provenance`, `processing_levels_available`, `spatial_resolution_conflict`, `temporal_output_cadences`, `temporal_aggregation_window_raw`, `observation_domain_class`, and `sentinel_lineage_status`) and are audited in [`03_harmonization/schema_completeness_report.csv`](../03_harmonization/schema_completeness_report.csv); automated catalogue metadata cannot replace primary product documentation when evaluating thematic fitness-for-purpose.
3. **Multi-Level Agency Collections**: Where upstream agency portals (`14_ceos_fedeo_agencies`, `17_eumetsat_product_navigator`, `18_esa_osc_earthcode`, `26_fao_unep_and_thematic_benchmarks` Zenodo, `03_copernicus_c3s_cds`) bundle L2/L3/L4 deliverables inside a single parent collection or deposit, those mixed collections are retained with explicit `processing_levels_available` and `processing_level` (`L3; L4` or highest gridded component `L3`) so their L3/L4 component remains discoverable.
4. **Harvest Boundary**: Research datasets lacking machine-readable STAC/REST/CSW/Zenodo/DataCite metadata endpoints (e.g., standalone cloud-drive distributions such as GloH2O `MSWEP`/`MSWX`), real-time operational NWP forecast streams, and forthcoming releases not yet published at harvest time sit outside an automated API catalogue.
5. **Direct Title/ID Attribution Heuristics**: Institutional attribution for decentralized programme outputs matches authoritative identifiers, keywords, and producing-institution fields without relying on free-text prose descriptions, supplemented by the two-layer `provider_pool` + `provider_jurisdiction_tier` classification aligned with Regulation (EU) 2021/696.
6. **Community Asset Collections (`29_awesome_gee_community`)**: Included intentionally for high recall (`4,378` prepared records, `source_authority_rank = 6`) because it captures unique community global EO products absent from upstream agency STACs (e.g., *FABDEM*, *DeltaDTM*, *Meta/WRI Global Canopy Height 1 m*, *Hydrography90m*, *CloudSEN12*, *Climate TRACE*, *ERA5 Heat*). Users analyzing raw source counts should note that parts of this community catalog list asset-level national or sub-regional Earth Engine sub-collections separately.
