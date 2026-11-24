# `05_external_policy_validation` — External Policy-to-EO Catalogue Validation

Independent, reproducible cross-validation of the Global Harmonized L3/L4 EO Product Catalogue (**`39,474`** canonical products across **`34`** product catalogues) against **9 external international and national initiatives** (`10` evaluation tracks) that map treaties, policy frameworks, UN Sustainable Development Goals, or federal agency mandates to concrete Earth Observation (EO) products.

---

## 1. Benchmark Coverage Summary

To prevent circularity, aggregate recall is reported both across all `10` tracks and across the **`8` strictly independent tracks**—excluding `07_nasa_snwg_cmr` (derived from NASA CMR, Source `16`) and `08_us_ghg_center` (derived from the U.S. GHG Center STAC, Source `33`).

| Track ID | Initiative & Authority | Policy / Legal Framework | Primary Downloaded Source(s) | Extracted | In-Scope L3/L4 EO | Exact | Compat. | Related | Miss | Strict Recall | Broad Recall |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01_ceos_gst` | **CEOS–UNFCCC Global Stocktake (GST) Portal** | UNFCCC Paris Agreement (Global Stocktake) | 19 HTML pages (`ceos.org/gst/`) | 15 | 15 | 11 | 3 | 0 | 1 | **93.3%** | **93.3%** |
| `02_ceos_sdg_sheets` | **CEOS SDG EO Support Sheets** (v2023–2025) | UN 2030 Agenda (SDGs 6.6.1, 11.3.1, 14.1.1a, 15.3.1) | 5 official CEOS PDFs (`ceos.org/sdg/`) | 30 | 29 | 12 | 11 | 0 | 6 | **79.3%** | **79.3%** |
| `03_ceos_eett_gbf` | **CEOS Ecosystem Extent Task Team (EETT) v1.1** | CBD Kunming-Montreal GBF (A.2) & UN SEEA EA | White Paper PDF Appendix 2 (`pp. 37–38`) | 24 | 23 | 15 | 7 | 1 | 0 | **95.7%** | **100.0%** |
| `04_unhabitat_eotoolkit` | **UN-Habitat / GEO / NASA EO Toolkit** | UN 2030 Agenda (SDG 11) & New Urban Agenda | ArcGIS FeatureServer JSON (`DatasetsHub2023`) | 37 | 35 | 20 | 12 | 2 | 1 | **91.4%** | **97.1%** |
| `05_ec_jrc_kceo` | **EC JRC KCEO Deep Dives** | EU Biodiversity Strategy 2030 & EU Urban Adaptation | 2 official EC XLSX workbooks | 41 | 34 | 11 | 21 | 2 | 0 | **94.1%** | **100.0%** |
| `06_usgeo_eoa_2023` | **White House USGEO / NASA–USGS EOA 2023** (`10.5066/P13BNHCN`) | U.S. Federal Statutory Mandates (AF & Climate SBAs) | 8 ScienceBase CSVs (`DataSources`, `KPSO_Counts`, `VT`) | 125 | 74 | 42 | 25 | 5 | 2 | **90.5%** | **97.3%** |
| `07b_nasa_nsite_portfolio` | **NASA NSITE / SNWG Solutions Portfolio** | U.S. Federal Agency Biennial Satellite Needs | `earthdata.nasa.gov/data/projects/nsite/solutions` | 32 | 12 | 1 | 8 | 3 | 0 | **75.0%** | **100.0%** |
| `09_unccd_gpg_v2` | **UNCCD Good Practice Guidance v2** (2021) | UN Convention to Combat Desertification (SDG 15.3.1) | Official UNCCD PDF (`Table 2-1`, `Table 3-3`) | 10 | 9 | 5 | 4 | 0 | 0 | **100.0%** | **100.0%** |
| **Strictly Independent (8 tracks)** | *Zero overlap with ingested APIs* | | | **314** | **231** | **117** | **91** | **13** | **10** | **90.0%** | **95.7%** |
| `07_nasa_snwg_cmr` *(ingested: Source 16)* | **NASA SNWG Delivered CMR Collections** | NASA CMR (`16_nasa_cmr`) | Live NASA CMR SNWG JSON | 104 | 70 | 70 | 0 | 0 | 0 | **100.0%** | **100.0%** |
| `08_us_ghg_center` *(ingested: Source 33)* | **U.S. Greenhouse Gas Center** (NASA/EPA/NIST/NOAA) | UNFCCC Paris Agreement & U.S. GHG Strategy | Live STAC API (`earth.gov/ghgcenter`) | 37 | 37 | 37 | 0 | 0 | 0 | **100.0%** | **100.0%** |
| **All 10 Tracks Combined** | **44 primary benchmark files (`27.16 MB`)** | | | **455** | **338** | **224** | **91** | **13** | **10** | **93.2%** | **97.0%** |

- **Strict vs. Broad Recall:** Across the 8 strictly independent tracks, **strict product/edition recall** (`EXACT_PRODUCT` + `COMPATIBLE_EDITION_OR_COLLECTION`) is **`208 / 231` (`90.0%`)**, and **broad family/proxy recall** (adding `RELATED_PROXY_OR_SUBSET`) is **`221 / 231` (`95.7%`)**. Across the 9 non-CMR tracks (including `08_us_ghg_center`), strict recall is **`245 / 268` (`91.4%`)** and broad recall is **`258 / 268` (`96.3%`)**. Across all 10 tracks combined, strict recall is **`315 / 338` (`93.2%`)** and broad recall is **`328 / 338` (`97.0%`)**.
- **USGEO EOA 2023 KPSO-Weighted Recall:** **`1,442 / 1,739` (`82.9%`)** strict (`EXACT_PRODUCT` + `COMPATIBLE_EDITION_OR_COLLECTION`) and **`1,526 / 1,739` (`87.8%`)** broad across impacted federal Key Products, Services, and Outcomes (KPSOs). Of the `213` broad-uncovered KPSO impacts, **`212` (`99.5%`)** stem from a single legacy internal USGS DEM (`GLSDEM`), while all modern global DEM replacements (`COP-DEM`, `NASADEM`, `SRTM`, `ASTER GDEM`, `ALOS AW3D30`, `USGS 3DEP/NED`, `GMTED2010`, `CDEM`) are indexed; excluding `GLSDEM`, broad weighted KPSO recall is **`1,526 / 1,527` (`99.9%`)**.
- **Excluded Circular / Non-Product Frameworks:** CEOS/CGMS WGClimate ECV Inventory, WMO OSCAR/Space, and GEO Knowledge Hub are already ingested as primary catalogue sources (`12`, `13`, `15`) and therefore excluded here; EEA CIS², C3S URDB, and EUSPA User Needs Reports document observation requirements rather than concrete multi-agency L3/L4 product lists.

---

## 2. Matching Protocol & Anti-Inflation Guardrails

1. **Explicit Scope Classification (`02_extract_products.py`):** Every extracted entry is written to [`results/extracted_policy_products.csv`](results/extracted_policy_products.csv) and classified as either `IN_SCOPE_OPEN_L3_L4_EO` (`338`) or out-of-scope (`52` raw L0–L2/sensor archives, `32` in-situ/census/historical-ship-log/OSM tables, `24` pre-launch missions or software platforms, `9` commercial VHR/CSDA feeds).
2. **Strict Tiered Matching (`03_validate_against_catalogue.py`):** Matches strictly against `raw_id`, `title`, and `doi` (never broad abstract text or noisy web URLs) in priority order:
   - `EXACT_DOI` / `RELATED_DOI`
   - `EXACT_RAW_ID` / `EXACT_CMR_SHORTNAME`
   - `PRODUCT_FAMILY_SIGNATURE` (explicit product regexes with optional source/pool constraints)
   - `EXCLUDED_IN_AUDIT_LOG` (verifies out-of-scope L1/L2 items against `02_preparation/data/excluded_records_log.parquet`)
3. **Specificity Stratification (`match_specificity_tier`):** Rather than conflating exact product hits with substitutions, every in-scope item is classified into one of four specificity tiers:
   - `EXACT_PRODUCT` (`224` across 10 tracks; `117` across 8 independent tracks): Exact DOI, exact `raw_id`/shortname, or exact canonical family match (`GLC-Share`, `ALOS PALSAR Global Mosaic`, `HRSL`, `GIMMS LAI`, `GIMMS 3g`).
   - `COMPATIBLE_EDITION_OR_COLLECTION` (`91`): Direct official successor edition, alternate-version release, or superset collection (e.g., `GSOCmap v1.5` for `GSOC v1.0`). Strict recall reports `EXACT_PRODUCT + COMPATIBLE_EDITION_OR_COLLECTION`.
   - `RELATED_PROXY_OR_SUBSET` (`13`): Thematic proxy, regional subset, or upstream/derived relative rather than the exact standalone policy release. Included only in **Broad Recall**.
   - `NOT_FOUND` (`10`): Absent from open unauthenticated catalogue feeds ([`results/gap_analysis.csv`](results/gap_analysis.csv)).

Every item-level match, specificity tier, and matched canonical identifier is recorded in [`results/validation_matches.csv`](results/validation_matches.csv).

---

## 3. Residual Gaps & Accepted Scope Trade-Offs (`10` Items)

Ten in-scope items remain absent in [`results/gap_analysis.csv`](results/gap_analysis.csv). Exhaustive checks across DataCite, Zenodo, HydroShare, ISIMIP, USGS ScienceBase, NASA CMR, CNES GEODES, and FAO CKAN confirm that none are recoverable from an open, unauthenticated primary EO catalogue API. We accept this trade-off explicitly rather than scraping brittle HTML dashboards, relying on personal API tokens, or ingesting noisy general-purpose web-map portals:

| Category | Count | Items | Rationale / Accepted Trade-Off |
| :--- | :---: | :--- | :--- |
| **Upstream `Level 2` Ontology Boundary** | `1` | `USDA/NASA G-REALM` (`10.5067/UCLRS-GREV2`) | Explicitly classified as `processing_level_id = 2` in NASA PO.DAAC CMR; its L3/L4 Pre-SWOT lake area and storage siblings (`UCLRS-AREV2`, `UCLRS-STOV2`) are indexed in Source `16`. |
| **Auth-Gated Inland Water Altimetry Portals** | `2` | `Hydroweb.next THEIA`, `DAHITI` | Require user registration/API keys (`HTTP 403` / connection reset without credentials) and serve virtual-station height time series rather than global gridded raster archives. |
| **Standalone Web Dashboards / ArcGIS Items** | `6` | `Sentinel-3 World Fire Atlas`, `DHI Global Wetland Map (2016–2018)`, `ESA WorldWater`, `UNEP/Esri SDG 14.1.1a EEZ Maps`, `Esri/Clark Landcover 2050`, `GLSDEM` | Distributed as standalone web viewers, ArcGIS Online items, or internal legacy processing auxiliaries without open STAC/CSW/REST collection APIs or DataCite dataset DOIs (`ESA WorldWater` publishes only validation point samples on Zenodo `10.5281/zenodo.6539508`). |
| **Sub-National Research Study** | `1` | `Forest Biomass Mapping in CA and OR` | Two-state regional research dataset absent from NASA CMR, USGS ScienceBase, and USFS DataCite (`10.2737`). |

---

## 4. Reproducing Locally (`uv run`)

```bash
uv run 05_external_policy_validation/01_download_benchmarks.py
uv run 05_external_policy_validation/02_extract_products.py
uv run 05_external_policy_validation/03_validate_against_catalogue.py
```
