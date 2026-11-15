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
| Raw records retrieved (`01_raw/`) | `47,360` | Plus `1,300` WMO OSCAR reference records (`237` variables, `1,063` satellites) |
| Prepared L3/L4 records (`02_preparation/`) | `37,822` | Stored per source in `02_preparation/data/` |
| Logged excluded records (`02_preparation/`) | `4,537` | `02_preparation/data/excluded_records_log.parquet` (+ `4,200` pre-excluded non-EO records across JRC, GEO KH, WRI) |
| Harmonized L3/L4 records (`03_harmonization/`) | `37,822` | [`03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet`](../03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet) |
| Canonical L3/L4 products (`03_harmonization/`) | `37,264` | `534` mirror groups collapsing `558` duplicate records ([`eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet)) |
| Canonical pool: `copernicus_services` | `3,520` | `L3`: `2,709` (`547` Sentinel), `L4`: `811` (`46` Sentinel) — **`593` Sentinel-dependent** |
| Canonical pool: `eu_non_copernicus` | `5,047` | `L3`: `4,869` (`539` Sentinel), `L4`: `178` (`27` Sentinel) — **`566` Sentinel-dependent** (`eu27_national_or_eu_agency`: `2,869`; `european_space_met_igo`: `1,807`; `copernicus_associated_state`: `371`) |
| Canonical pool: `non_eu` | `28,697` | `L3`: `26,421` (`854` Sentinel), `L4`: `2,276` (`36` Sentinel) — **`890` Sentinel-dependent** (`third_country_non_european`: `27,206`; `un_or_international_body`: `1,479`; `european_esa_non_copernicus_state`: `12`) |

---

## 2. Automated Quality Assurance Suites (`18 + 16 PASS`)

1. **`03_harmonization/verify_catalogue.py`** (`18` structural and ontology assertions):
   - Verifies completeness of all `29` source directories (`mine.py`, `README.md`, `last_access.txt`, non-empty `data/`) and enforces GitHub's single-file size bound (`<= 48 MB`).
   - Verifies Sentinel-dependency detection (with mission launch-year and heritage-platform guards), producer-origin pool and `provider_jurisdiction_tier` attribution, operational maturity classification (`operational_ongoing_series` vs. `completed_or_mission_bounded_release`, `research_or_benchmark_release`, `planned_or_preoperational`, `deprecated_or_superseded`, and `unknown`, with zero `operational_ongoing_series` rows lacking `is_ongoing_archive`), symmetric `L3_ARD_REFLECTANCE` tagging across pools, leading-dot and fractional-degree resolution parsing (`27,830.0 m`, CLMS Burnt Area `1/336°` = `331.31 m`), Suomi-NPP L3 and C3S `satellite-*` L3 preservation, full NOAA `summary: false` (`99.3%` descriptions, `99.1%` MultiPolygon/Polygon bounds) and NASA CMR UMM-JSON (`93.4%` DOIs, `2,367` native grid resolutions) metadata recovery, separation of sampling cadence from publication update frequency, all-43-column `schema_completeness_report.csv` coverage, and high-recall benchmark collections (`FAO/WAPOR`, `Sentinel-3 SYNERGY V10/VG1`, `TransCom 3`, `SEDAC`, `INPE BDC`, `FLUXCOM-X-BASE`, `LCSPP`, `GWELD`, `EO:EUM:DAT:0088`).
2. **`04_validation_and_review/validate_catalogue_benchmarks.py`** (`16` integrity assertions):
   - Zero duplicate `raw_id` within any source catalogue.
   - Zero Level-0, SLC-off/Burst, swath grid, or test placeholder contamination in canonical titles.
   - Zero non-EO laboratory/nuclear/LCA entries from institutional repositories.
   - Zero implausible unit-conversion, fractional-degree (`<= 600 km` ceiling), cartographic scale-denominator (`EEA >= 500 km` non-degree), >10x title-token discrepancy, or leading-dot degree artifacts (`AE_DyOcn`/`AU_DyOcn` = `27,830.0 m`) in spatial resolution fields.
   - Zero in-situ/campaign/socioeconomic records (`INSITU_*`, `radiosonde`, `Argo float`, `Polarstern`, `XBT`/`CTD`/buoy logs, `ARDECO`, `LUCAS Copernicus`, `gridded GDP`) marked `strict_eo_c1_eligible = True`.
   - Zero within-catalogue `canonical_product_key` collisions.
   - Zero null/empty product titles (`37,822 / 37,822` non-null).
   - Zero invalid coordinates (`bbox`), unprefixed or preview-image (`.png`/`.jpg`/`.gif`/`/browse`) `access_url` links, non-canonical `doi` formats, or inverted ISO dates.
   - Zero producer-pool overrides on Copernicus/ESA/EU/INPE/FedEO records and valid `provider_jurisdiction_tier` invariants; zero future scenarios (`temporal_end_year >= 2028`) tagged as `is_ongoing_archive = True`; zero Suomi-NPP L3-to-L4, acronym homograph (`CAS/IGSNRR/PML`, NASA `HS3`), or `temporal_start_year > 2026` leaks.
   - Verified deep provider metadata recovery (`>= 98%` NOAA descriptions, `>= 90%` CMR DOIs, `>= 2,300` CMR native resolutions) and zero multi-day-title vs. `sub_daily_or_daily` cadence contradictions, zero `operational_ongoing_series` rows with null `is_ongoing_archive`, and zero pre-2014 bounded records tagged `uses_sentinel_data = True`.

---

## 3. External Domain Coverage Benchmarks

### 3.1 EU Policy & Operational Service Product Families (`N = 84`)
Evaluated across `84` consolidated EU environmental policy, Copernicus operational service, and global scientific benchmark product families in [`eu_policy_products_benchmark_audit.csv`](eu_policy_products_benchmark_audit.csv):

- **In-Scope Open L3/L4 EO Product Families**: **`75 / 75` (`100.0%`)** present (`IN_CATALOGUE`) across CLMS (including all 9 pan-European 10 m HRL Croplands Crop Types & Cropping Patterns layers), CEMS/JRC (GFM, GloFAS, EFAS/HERA, EDO/GDO, EFFIS/GWIS, GHSL, GSWE 30 m, TMF v2025, Rapid Mapping & RRM, MARS/ASAP, ESDAC), C3S (ERA5, ERA5-Land, CERRA, seasonal forecasts, PECD, CMIP6, ECVs, UTCI, agroclimatic indicators), CAMS (emission inventories, GHG/AQ forecasts & reanalyses, inversions, CRS), CMEMS (Physics, BGC, HR-OC 100 m, GlobColour, OMIs, SDB 100 m), Copernicus DEM / EU-DEM, EMODnet, and global external benchmarks (VIIRS `VNP46`, USGS Landsat C2 L3, `G3P` / `GRACE` TWS, `FLUXCOM-X-BASE`, `LCSPP`/`LCSIF`/`LCREF`, `GWELD`, and EUMETSAT `EO:EUM:DAT:0088` MSG LST Level 3).
- **Sensitivity / Non-EO Tagged & Related Entries**: `4` families (`INSITU_*_OA_*` gridded interpolations, `EDGAR`/`CEDS` emission inventories, Eurostat/JRC socio-economic support, and non-EO reference registries).
- **Expected Out-of-Scope**: `4` families (restricted Copernicus Security Services / EMSA CleanSeaNet; interactive web-only CAMS Methane Hotspot Explorer; activity-based CAMS Carbon Monitor; EEA tabular WEI+/WISE Flood Risk/EUCRA reports).
- **Excluded Imagery**: `1` family (`CDSE Sentinel Level-1 and Level-2 Operational Imagery`, verified as `CORRECTLY_EXCLUDED_IN_AUDIT_LOG` in `excluded_records_log.parquet`).

### 3.2 GCOS Essential Climate Variables (`N = 40`)
Evaluated against the CEOS/CGMS WGClimate GCOS Essential Climate Variable (ECV) taxonomy using explicit domain-qualified variable patterns in [`gcos_ecv_domain_coverage.csv`](gcos_ecv_domain_coverage.csv), reporting both total canonical matches and independent non-WGClimate matches: **`40 / 40`** GCOS ECV groups are covered across both the canonical inventory and independent non-WGClimate sources.

---

## 4. Scope, Design Choices & Known Limitations

1. **High Recall Over Precision (Intentional Design)**: To avoid silently omitting candidate products, filtering and canonical deduplication (`canonical_product_key` matching exact normalized DOI or pool + normalized full title) prioritize high recall. Consequently, unlinked cloud mirrors lacking shared DOIs and with differing upstream titles, versioned product releases, and boundary sensitivity records remain as separate rows and require expert screening and de-duplication for specific downstream assessments.
2. **Metadata Completeness (`let null be null`)**: Upstream APIs frequently omit horizontal grid spacing (`spatial_resolution_meters`), archive dates (`temporal_start_year`), or DOIs at the collection level. Missing attributes remain `null` and are audited in [`03_harmonization/schema_completeness_report.csv`](../03_harmonization/schema_completeness_report.csv); automated catalogue metadata cannot replace primary product documentation when evaluating thematic fitness-for-purpose.
3. **Multi-Level Agency Collections**: Where upstream agency portals (`14_ceos_fedeo_agencies`, `17_eumetsat_product_navigator`, `18_esa_osc_earthcode`) bundle L1/L2 and L3/L4 deliverables inside a single parent collection record, those mixed collections are retained so their L3/L4 component remains discoverable.
4. **Harvest Boundary**: Research datasets lacking machine-readable STAC/REST/CSW metadata endpoints (e.g., standalone cloud-drive distributions such as GloH2O `MSWEP`/`MSWX`), real-time operational NWP forecast streams, and forthcoming releases not yet published at harvest time sit outside an automated API catalogue.
5. **Direct Title/ID Attribution Heuristics**: Institutional attribution for decentralized programme outputs matches authoritative identifiers, keywords, and producing-institution fields without relying on free-text prose descriptions, supplemented by the two-layer `provider_pool` + `provider_jurisdiction_tier` classification aligned with Regulation (EU) 2021/696.
6. **Community Asset Collections (`29_awesome_gee_community`)**: Included intentionally for high recall (`4,378` prepared records, `source_authority_rank = 6`) because it captures unique community global EO products absent from upstream agency STACs (e.g., *FABDEM*, *DeltaDTM*, *Meta/WRI Global Canopy Height 1 m*, *Hydrography90m*, *CloudSEN12*, *Climate TRACE*, *ERA5 Heat*). Users analyzing raw source counts should note that parts of this community catalog list asset-level national or sub-regional Earth Engine sub-collections separately.
