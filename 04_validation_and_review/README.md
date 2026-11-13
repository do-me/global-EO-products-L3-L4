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
| Raw records retrieved (`01_raw/`) | `33,663` | Plus `1,300` WMO OSCAR reference records (`237` variables, `1,063` satellites) |
| Prepared L3/L4 records (`02_preparation/`) | `26,609` | Stored per source in `02_preparation/data/` |
| Logged excluded records (`02_preparation/`) | `2,787` | `02_preparation/data/excluded_records_log.parquet` (+ `4,200` pre-excluded non-EO records across JRC, GEO KH, WRI) |
| Harmonized L3/L4 records (`03_harmonization/`) | `26,609` | [`03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet`](../03_harmonization/data/eo_l3_l4_all_catalogue_records.parquet) |
| Canonical L3/L4 products (`03_harmonization/`) | `26,330` | `273` mirror groups collapsing `279` duplicate records ([`eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet)) |
| Canonical pool: `copernicus_services` | `2,776` | `L3`: `1,933` (`399` Sentinel), `L4`: `843` (`110` Sentinel) — **`509` Sentinel-dependent** |
| Canonical pool: `eu_non_copernicus` | `5,690` | `L3`: `5,507` (`777` Sentinel), `L4`: `183` (`29` Sentinel) — **`806` Sentinel-dependent** |
| Canonical pool: `non_eu` | `17,864` | `L3`: `15,163` (`740` Sentinel), `L4`: `2,701` (`52` Sentinel) — **`792` Sentinel-dependent** |

---

## 2. Automated Quality Assurance Suites (`12 + 9 PASS`)

1. **`03_harmonization/verify_catalogue.py`** (`12` structural and ontology assertions):
   - Verifies completeness of all `29` source directories (`mine.py`, `README.md`, `last_access.txt`, non-empty `data/`) and enforces GitHub's single-file size bound (`<= 48 MB`).
   - Verifies Sentinel-dependency detection, producer-origin pool attribution, operational maturity classification (ongoing vs. mission-bounded/single-epoch/planned), symmetric `L3_ARD_REFLECTANCE` tagging across pools, and WGClimate / OpenLandMap / source-22 institutional attribution.
2. **`04_validation_and_review/validate_catalogue_benchmarks.py`** (`9` integrity assertions):
   - Zero duplicate `raw_id` within any source catalogue.
   - Zero Level-0, SLC-off/Burst, swath grid, or test placeholder contamination in canonical titles.
   - Zero non-EO laboratory/nuclear/LCA entries from institutional repositories.
   - Zero implausible unit-conversion or cartographic scale-denominator artifacts in `spatial_resolution_meters`.
   - Zero in-situ observation products marked `strict_eo_c1_eligible = True`.
   - Zero within-catalogue `canonical_product_key` collisions.
   - Zero null/empty product titles (`26,609 / 26,609` non-null).
   - Zero invalid coordinates (`bbox`), unprefixed `access_url` schemes, non-canonical `doi` formats, or inverted ISO dates.
   - Zero producer-pool overrides on Copernicus/ESA/EU records (`cop_marine_non_eu = 0`, `esa_esrin_non_eu = 0`, `fifra_eu = 0`) and zero WGClimate L2 leaks.

---

## 3. External Domain Coverage Benchmarks

### 3.1 EU Policy & Operational Service Product Families (`N = 80`)
Evaluated across `80` consolidated EU environmental policy and Copernicus operational service product families in [`eu_policy_products_benchmark_audit.csv`](eu_policy_products_benchmark_audit.csv):

- **In-Scope Open L3/L4 EO Product Families**: **`71 / 71` (`100.0%`)** present (`IN_CATALOGUE`) across CLMS, CEMS/JRC (GFM, GloFAS, EFAS/HERA, EDO/GDO, EFFIS/GWIS, GHSL, GSWE 30 m, TMF v2025, Rapid Mapping & RRM, MARS/ASAP, ESDAC), C3S (ERA5, ERA5-Land, CERRA, seasonal forecasts, PECD, CMIP6, ECVs, UTCI, agroclimatic indicators), CAMS (emission inventories, GHG/AQ forecasts & reanalyses, inversions, CRS), CMEMS (Physics, BGC, HR-OC 100 m, GlobColour, OMIs, SDB 100 m), Copernicus DEM / EU-DEM, EMODnet, and global external benchmarks (VIIRS `VNP46`, USGS Landsat C2 L3, `G3P` / `GRACE` TWS).
- **Sensitivity / Non-EO Tagged Entries**: `4` families indexed with explicit non-strict-EO sensitivity flags (`INSITU_*_OA_*` gridded interpolations, `EDGAR`/`CEDS` emission inventories, socio-economic/cadastral references).
- **Expected Out-of-Scope**: `4` families (restricted Copernicus Security Services / EMSA CleanSeaNet; interactive web-only CAMS Methane Hotspot Explorer; activity-based CAMS Carbon Monitor; EEA tabular WEI+/WISE Flood Risk/EUCRA reports).
- **Excluded Imagery**: `1` family (CDSE Sentinel L1/L2 operational imagery, logged in `excluded_records_log.parquet`).

### 3.2 GCOS Essential Climate Variables (`N = 40`)
Evaluated against the CEOS/CGMS WGClimate GCOS Essential Climate Variable (ECV) taxonomy in [`gcos_ecv_domain_coverage.csv`](gcos_ecv_domain_coverage.csv): **`40 / 40`** GCOS ECV groups have canonical L3/L4 matches in the harmonized catalogue.

---

## 4. Scope, Design Choices & Known Limitations

1. **High Recall Over Precision (Intentional Design)**: To avoid silently omitting candidate products, filtering and canonical deduplication (`canonical_product_key` matching exact normalized DOI or pool + normalized full title) prioritize high recall. Consequently, unlinked cloud mirrors lacking shared DOIs and with differing upstream titles, versioned product releases, and boundary sensitivity records remain as separate rows and require expert screening and de-duplication for specific downstream assessments.
2. **Metadata Completeness (`let null be null`)**: Upstream APIs frequently omit horizontal grid spacing (`spatial_resolution_meters`), archive dates (`temporal_start_year`), or DOIs at the collection level. Missing attributes remain `null` and are audited in [`03_harmonization/schema_completeness_report.csv`](../03_harmonization/schema_completeness_report.csv); automated catalogue metadata cannot replace primary product documentation when evaluating thematic fitness-for-purpose.
3. **Multi-Level Agency Collections**: Where upstream agency portals (`14_ceos_fedeo_agencies`, `17_eumetsat_product_navigator`, `18_esa_osc_earthcode`) bundle L1/L2 and L3/L4 deliverables inside a single parent collection record, those mixed collections are retained so their L3/L4 component remains discoverable.
4. **Harvest Boundary**: Research datasets lacking machine-readable STAC/REST/CSW metadata endpoints (e.g., standalone cloud-drive distributions such as GloH2O `MSWEP`/`MSWX`), real-time operational NWP forecast streams, and forthcoming releases not yet published at harvest time sit outside an automated API catalogue.
5. **Institutional Description Heuristics**: Description-based tagging for decentralized programme outputs (e.g., JRC-hosted CEMS datasets) should be spot-checked when compiling strict institutional subsets.
6. **Community Asset Collections (`29_awesome_gee_community`)**: Included intentionally for high recall (`4,378` prepared records, `source_authority_rank = 6`) because it captures unique community global EO products absent from upstream agency STACs (e.g., *FABDEM*, *DeltaDTM*, *Meta/WRI Global Canopy Height 1 m*, *Hydrography90m*, *CloudSEN12*, *Climate TRACE*, *ERA5 Heat*). Users analyzing raw source counts should note that parts of this community catalog list asset-level national or sub-regional Earth Engine sub-collections separately.
