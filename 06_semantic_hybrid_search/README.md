# Local Hybrid Semantic Search (`06_semantic_hybrid_search`)

Local hybrid search engine, CLI, web interface, and validation benchmark over all **39,472 canonical Level-3 and Level-4 Earth Observation (EO) catalogue records** from [`../03_harmonization/data/eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet). Evaluated against curated family rules, the default hybrid configuration (`EmbeddingGemma 2 + DuckDB BM25 + Jina v3.5`) achieves **85.45% Fam-nDCG@10** (**76.67% P@1**, **96.39% family Recall@10**, **100.00% Hit@10**) on the 60-query retrospectively audited cross-domain holdout (`v4.3-standalone`).

The search engine combines dense semantic embeddings (`EmbeddingGemma 2`), EO-normalized multi-field BM25 (`DuckDB >= 1.5.0` FTS keyed by `record_id` and bound to catalogue content hashes), variable/layer-role/cadence/resolution/footprint-guarded family collapsing, and optional Stage-2 neural reranking (`Jina Reranker v3.5` or `Qwen3-Reranker-0.6B`). Backends are **device-agnostic (`--device auto`)**: scripts automatically select native **MLX** on Apple Silicon (`darwin/arm64`), **PyTorch CUDA** when an NVIDIA GPU is present, or **PyTorch CPU** otherwise, with an instant `--bm25-only` lexical mode requiring zero neural weights.

<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/d6dbff1f-5b93-4a4a-9d90-d3240c526510" />
<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/64c066dc-cb58-42ec-aceb-774c352c16cc" />
<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/5a070fdf-1221-49f9-ac3f-e29d6f56baf7" />

---

## 1. Quick Start (`uv run`)

All scripts use PEP 723 inline dependencies and require only [`uv`](https://docs.astral.sh/uv/). Run commands from the repository root.
Recommended quickstart (auto-downloads & SHA-256-verifies the prebuilt `v4.3.0-indexes` release assets on first launch):

```bash
uv run 06_semantic_hybrid_search/04_search_server.py --port 8080 --open
```

### 1.1 Prebuilt Release Indexes & Local Index Rebuilds

Large binary index artifacts (`full_bm25.duckdb`, `124.5 MB`; `full_embeddings_bf16.npz`, `54.3 MB`; `full_index_manifest.json`) are excluded from git and hosted on GitHub Releases (`v4.3.0-indexes`). Both `04_search_server.py` and `04_search_cli.py` automatically fetch and verify them via `ensure_search_indexes()` when missing locally, while validating locally rebuilt `full_bm25.duckdb` files by semantic content (`bm25_content_sha256`, `record_ids_sha256`, and `catalogue_content_sha256`) rather than raw DuckDB binary page bytes:

- `--rebuild-index`: build `full_bm25.duckdb` and `full_embeddings_bf16.npz` locally from `eo_l3_l4_canonical_products.parquet` instead of downloading release binaries.
- `--no-download-index`: disallow automatic release downloads (fail fast unless local index files are already present).

```bash
# Build local full hybrid index explicitly (auto-selects MLX / CUDA / CPU)
# ~6.5 min on Apple Silicon M3 Max (2.55 GB peak memory; ~36 s for BM25 alone)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py

# Fast lexical-only local build on any OS (zero neural models, ~36 s)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py --bm25-only
```

### 1.2 Launch the Local Web UI & API Server

Starts a local FastAPI server (auto-selecting `MLX` / `CUDA` / `CPU`, or `--bm25-only`) and serves the Tailwind CSS + MapLibre GL search interface (`web/`). The interface supports all ranking modes, structured facets, a **Verified (Hard Limits)** toggle (`--hard-limits`) with unknown- and unverified-metadata exclusion reporting, family-collapsed sibling drawers, interactive bounding-box maps, and a single **Download** dropdown exporting results to **CSV**, **JSON**, **Excel (`.xlsx`)**, or **Parquet**:

```bash
# Auto-detect best backend (auto-downloads v4.3.0-indexes if absent) and open http://127.0.0.1:8080
uv run 06_semantic_hybrid_search/04_search_server.py --port 8080 --open

# Optional overrides: rebuild locally, disallow download, preload reranker, or run lexical-only
# (On Apple Silicon macOS, add `--with torch` if forcing PyTorch `--device cpu`)
uv run 06_semantic_hybrid_search/04_search_server.py --rebuild-index --port 8080
uv run 06_semantic_hybrid_search/04_search_server.py --device cpu --port 8080
uv run 06_semantic_hybrid_search/04_search_server.py --preload-reranker jina-v3.5 --port 8080
uv run 06_semantic_hybrid_search/04_search_server.py --bm25-only --port 8080
```

### 1.3 Command-Line Search (Standalone or Client Mode)

```bash
# Zero-cold-start client querying the running server
uv run 06_semantic_hybrid_search/04_search_cli.py \
  --server http://127.0.0.1:8080 \
  --query "Global 30 meter inland surface water occurrence and seasonality" \
  --geo global --max-grid-m 300 --top-k 10

# Verified hard-limits mode: enforces exact global footprint, verified <=250 m output grid, documented end years
uv run 06_semantic_hybrid_search/04_search_cli.py \
  --query "global land cover" --geo global --max-grid-m 250 --hard-limits --rerank none --top-k 10

# Standalone CLI (auto-selects MLX / CUDA / CPU; add --json for strict machine-readable stdout)
uv run 06_semantic_hybrid_search/04_search_cli.py \
  --query "Global flood hazard mapping" \
  --pool copernicus_services --rerank jina-v3.5 --top-k 5 --json

# Lexical-only CLI on any machine without dense vectors
uv run 06_semantic_hybrid_search/04_search_cli.py \
  --query "Global flood hazard mapping" --bm25-only --top-k 5
```

| Preset | CLI Flag | License | Mean Latency (M3 Max) | When to Use |
| :--- | :--- | :--- | ---: | :--- |
| **Best Quality (Default)** | `--rerank jina-v3.5 --rerank-pool 50` | CC BY-NC 4.0 | ~5.5 s (HB) / ~3.1 s (CLI warm) | Highest ranking accuracy for research & non-commercial analysis |
| **Fast Rerank** | `--rerank jina-v3.5 --rerank-pool 25 --no-blend` | CC BY-NC 4.0 | ~2.6 s (HB) / ~1.1 s (CLI warm) | Interactive screening with near-identical Top-10 quality (~2.1× faster) |
| **Permissive Rerank** | `--rerank qwen3-0.6b --rerank-pool 50` | Apache-2.0 | ~5.2 s (HB) / ~1.8 s (CLI warm) | Commercial or permissive-license Stage-2 reranking |
| **Stage-1 Hybrid Only** | `--rerank none` | Apache-2.0 | ~1.2 s (HB) / ~167–179 ms (CLI warm) | Instant interactive retrieval (`98.33%` Top-10 hit rate on Holdout B) |
| **BM25 Only** | `--bm25-only` | MIT | ~149 ms (CLI warm) | Zero-model lexical search on any CPU |

Server API endpoints: `GET /api/status`, `POST /api/search` (and `GET /api/search`), `POST /api/export` (`csv`, `json`, `xlsx`, `parquet`), and `GET /api/product/{record_id}`.

### 1.4 Static Browser-Only Advanced Search (`web_search/` & GitHub Pages)

A zero-backend client-side port is deployed automatically to [`https://do-me.github.io/global-EO-products-L3-L4/search.html`](https://do-me.github.io/global-EO-products-L3-L4/search.html) via `.github/workflows/pages.yml` (`pnpm` + Vite + `@huggingface/transformers` v4 WebGPU/WASM). `export_browser_index.py` converts the pinned `v4.3.0-indexes` release into compact gzipped browser shards (`3.1 MB` lossless 3-field EO BM25 with exact `0.00` DuckDB score parity, `15.0 MB` 512-d Matryoshka `int8` embeddings, and `0.4 MB` filter/family metadata) while pulling `onnx-community/embeddinggemma-2-ONNX` (text-only `q4`, `Apache-2.0`) and optional opt-in `onnx-community/Qwen3-Reranker-0.6B-ONNX` (`q4`, `Apache-2.0`) directly from Hugging Face in the user's browser.

---

## 2. Methodology

The pipeline is built around two practical realities of federated EO metadata catalogues:

1. **Upstream metadata is sparse:** `70.8%` of canonical records lack an adjudicated pixel grid value and `50.3%` lack structured platform metadata. Standard strict SQL filters silently drop relevant products whose metadata fields are `null`.
2. **Catalogue listings are heavily versioned and mirrored:** A single scientific product family (e.g., an ESA CCI climate record or NASA MODIS product) often appears as dozens of versioned releases, annual editions, or cross-portal mirrors that would otherwise crowd out distinct alternative datasets in the Top-10.

Every query runs through four deterministic stages implemented in [`eo_search_lib.py`](eo_search_lib.py) and [`06_eval_rerankers.py`](06_eval_rerankers.py):

### Stage 1 — Structured Pre-Filtering & Verified Hard-Limits Mode
Records flagged as superseded (`is_superseded_edition = True` or obsolete/deprecated title tags) are excluded by default, leaving `38,850` active records. Optional facets (`provider_pool`, `geo_scope`, `max_grid_m`, `processing_levels`, `min_end_year`, `strict_eo_only`) are applied as a boolean mask *before* scoring in three explicit precision/recall modes:

- **Null-Safe Discovery (Default):** Keeps records whose requested metadata attribute is `null` and applies a `1.25×` tolerance on `max_grid_m` (e.g., allowing `300 m` Sentinel-3/OLCI products on a `250 m` request). `global` also permits `hemispheric_or_pantropical` collections, and ongoing archives satisfy `min_end_year` only up to the current harvest year (`<= 2026`).
- **Strict Known-Values (`--strict-filter` / `null_safe=False`):** Excludes records with missing (`null`) metadata on active filter facets while retaining the `1.25×` grid tolerance and broad global discovery semantics unless `--hard-limits` is enabled.
- **Verified Hard-Limits (`--hard-limits` / `hard_limits=True`):** Enforces strict scientific constraints (`null_safe=False`, exact `1.0×` grid tolerance `adjudicated_output_grid_meters <= max_grid_m` restricted to documented output-grid provenance—excluding coarse `gcmd_coarse_resolution_band`, WGClimate `ecv_inventory_horizontal_resolution` instrument IFOV entries, scalar aggregate time series such as Tellus GRACE mass-anomaly `.txt` series whose parent grid is kept separately in `spatial_resolution_raw`, and ocean vertical depth intervals such as `0–700 m`—exact `processing_levels`, documented `temporal_end_year >= min_end_year` with no ongoing-archive bypass, and strict `geographic_coverage == 'global'` excluding `northern_hemisphere_only`, `southern_hemisphere_only`, `geostationary_disk_or_sector`, and regional-titled records in `observation_domain_class`) while reporting `unknown_metadata_counts` (including `unverified_grid_source` and `temporal_end_year`) for excluded records.

### Stage 2 — Hybrid Dense + `record_id`-Keyed BM25 Retrieval (RRF)
Two complementary rankers score all eligible records in parallel (Top-300 candidates each), guarded by fail-fast `verify_index_alignment()` checks confirming exact length, row order, `record_id` sequence, and `catalogue_sha256` / `catalogue_content_sha256` agreement across the canonical Parquet table, dense matrix, and DuckDB index:

- **Dense bi-encoder (`EmbeddingGemma 2`, 768-d):** Each canonical record is serialized into a structured EO card (title, IDs, institution, domain, spatial/temporal metadata, deduplicated GCMD keywords, cleaned abstract; truncated at `512` tokens, no chunking). Queries use asymmetric query encoding (`batch_size=1`, `max_length=256`). Apple Silicon loads `mlx-community/embeddinggemma-2-bf16` via MLX; CUDA and CPU load `google/embeddinggemma-2` via PyTorch (`bfloat16` on Ampere+ CUDA and CPU, `float32` on older GPUs/MPS). Cosine ties break deterministically by ascending `record_id`.
- **Multi-field BM25 (`DuckDB >= 1.5.0` FTS):** Keyed directly by immutable `record_id`, with `eo_meta` recording `record_ids_sha256`, `bm25_content_sha256`, `catalogue_sha256`, and `catalogue_content_sha256` across all builds (including server BM25 auto-build fallback) so mutated metadata with unchanged `record_id`s is always rejected. Indexes three normalized fields (`title` weight `2.5`, `keywords` `1.5`, `body` `1.0`, `k1=1.2`, `b=0.75`) using deterministic single-thread execution (`SET threads = 1`, `ROUND(score, 10)`) and generic EO-domain lexical normalization (UK/US spelling such as `colour`/`color`, `burnt`/`burned`, hyphen/underscore splits, and standard physical variable acronyms such as `CO2`, `CH4`, `LAI`, `SST`, `LST`, `DEM`—zero product or mission names).
- **Reciprocal Rank Fusion (RRF):** Combines dense and lexical rankings using $S_{\text{RRF}}(d) = \frac{1.0}{60 + r_{\text{dense}}(d)} + \frac{0.6}{60 + r_{\text{bm25}}(d)}$.

### Stage 3 — Generic Sensor-, Variable-, Layer-Role- & Cadence-Guarded Family Collapsing
Scans up to `max(top_k * 3, 200)` fused candidates and groups versioned editions and unlinked portal mirrors under the representative record without any benchmark-specific regexes (`25,190` unique generic stems across the `39,472` catalogue rows). Two candidates collapse when either:
- They share the same generic normalized title stem (`gstem:`, stripping leading catalogue codes, parenthetical tags, versions, years, and resolution tokens), **or**
- Their 768-d dense cosine similarity reaches `near_dup_cos >= 0.94` (selected strictly on `Dev-45`).

To prevent distinct scientific deliverables from merging, collapsing is blocked whenever two records differ in **layer role** (e.g., primary product vs. `confidence`, `quality_flag`, `uncertainty`, `descriptor`, or `spectral_doy`), **delivered variable/theme** (e.g., `Annual NLCD` Land Cover vs. Land Cover Change vs. Fractional Impervious Surface), **physical instrument/sensor**, **temporal cadence class** (including title-inferred `annual` vs. `five-year` cadence such as `GLC_FCS30D`), **geographic coverage class** (e.g., global vs. regional), **`> 2.0×` spatial grid resolution**, or **distinct non-empty DOIs**. When a collapsed cluster contains multiple valid editions, **query-matching variant promotion** (`collapse_product_families(..., query_text)`) promotes the sibling whose cadence, resolution, or coverage tokens match the user's query. All collapsed siblings remain inspectable in `collapsed_siblings`.

### Stage 4 — Optional Stage-2 Neural Reranking
Rescores the Top-50 (or Top-25) collapsed family representatives using concise `<= 1,200`-character metadata cards:
- **`Jina Reranker v3.5`** (`jinaai/jina-reranker-v3.5-mlx` on Apple Silicon; `jinaai/jina-reranker-v3.5` on CUDA/CPU; `CC BY-NC 4.0`): listwise cross-attention reranker (`max_doc_length=420`), blended with Stage-1 RRF ranks via $S(d) = \frac{0.4}{30 + r_{\text{S1}}(d)} + \frac{1.0}{30 + r_{\text{S2}}(d)}$ (or pure reranker order for the fast Top-25 preset).
- **`Qwen3-Reranker-0.6B`** (`Qwen/Qwen3-Reranker-0.6B`; `Apache-2.0`): instruction-aware pairwise reranker scored via raw `yes`/`no` log-odds margin (`logit("yes") - logit("no")`) with deterministic stable sorting and the same RRF blend.

---

## 3. Validation Protocol & Benchmark Construction (`v4.3-standalone`)

### 3.1 What the Benchmarks Represent & How They Were Built

Each benchmark query represents a realistic Earth Observation discovery request phrased in domain and policy language—describing the biophysical variable, spatial/temporal requirement, or policy indicator *without* naming target products, satellite missions, or producing agencies. Because one scientific dataset often appears across dozens of versioned releases and cross-portal mirrors in the `39,472`-record catalogue, ground truth is defined at the **product-family level** (`1`–`3` distinct institutional product families per query) rather than single rows.

Every benchmark suite is constructed deterministically in code directly against the `38,850` active, non-superseded records of [`eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet):

1. **Source selection:** Queries and expected product families are derived from authoritative public EO policy frameworks, Essential Climate Variable (ECV) registries, and domain catalogues—never generated from search-engine outputs.
2. **Constraint-aligned family resolution:** Each expected family is matched to qualifying active records via `filter_benchmark_family_mask()`, combining explicit inclusion/exclusion rules (`title`, `raw_id`, `source_catalogue`) with the query's own structured scope constraints (`geo_scope`, `observation_domain_class` hemisphere/sector guards, `max_grid_m`, `strict_eo_only`, L3/L4 processing level, multi-decadal single-year snapshot exclusion, excluding auxiliary confidence/input-quality/uncertainty/quality-flag layers unless explicitly requested, and excluding superseded/obsolete editions) so regional subsets, single-hemisphere products, single-year snapshots on multi-decadal requests, coarse `2.5°` grids, auxiliary confidence layers, or L1/L2 records cannot receive ground-truth credit.
3. **Automated integrity checks:** Each builder script enforces four programmatic assertions before writing its locked benchmark JSON:
   - Every expected family matches `>= 1` active record and remains reachable under the query's structured filters (`geo_scope`, `max_grid_m`, `strict_eo_only`).
   - `0` `record_id` overlaps exist between distinct expected families inside any query.
   - `0` `record_id` overlaps exist across the three suites (`Dev-45 ∩ Holdout A = ∅`, `Dev-45 ∩ Holdout B = ∅`, `Holdout A ∩ Holdout B = ∅`), and `0` overlaps exist across any pair of `Holdout B` queries.
   - `0` product brand names, mission/sensor codes, or provider shortnames appear in holdout query text (enforced by automated anti-leakage blocklists of `70+` terms in `Holdout A` and `110+` terms in `Holdout B`).

### 3.2 Benchmark Suites & Source Provenance

| Suite | Builder & Artifact | Size | Expected Families | Unique Active Records | Sources & Role |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Dev-45** | [`02_build_policy_benchmark.py`](02_build_policy_benchmark.py) → `policy_retrieval_benchmark.json` | 45 queries (`32` multi-family) | 100 | 1,114 (`918` subset GT) | **Development set** drawn from 6 policy-to-EO frameworks cross-referenced in `05_external_policy_validation/`: EU Biodiversity Strategy 2030 (`14`, EC KCEO Annex 3), CBD Kunming-Montreal GBF Indicator A.2 (`7`, CEOS EETT), UNFCCC Global Stocktake (`7`, CEOS GST / US GHG Center / UNCCD), UN SDGs `6.6.1`/`11.3.1`/`11.6.2`/`14.1.1a`/`15.3.1` (`7`, CEOS / UN-Habitat / UNCCD GPG v2), EC KCEO Urban Climate Adaptation (`5`), and USGEO EOA 2023 / NASA SNWG (`5`). Used solely to select `near_dup_cos = 0.94`, ablate filters, and run nested 6-fold Leave-One-Framework-Out CV. |
| **Holdout A (Audited)** | [`08_build_holdout_a_external.py`](08_build_holdout_a_external.py) → `holdout_a_external_benchmark.json` | 35 queries (`22` multi-family) | 62 | 1,553 | **Retrospectively audited external domain set** drawn from unused international science/policy registries: GCOS ECV (WGClimate), WMO OSCAR/Space, USGEO EOA 2023, NASA SNWG, EC KCEO Marine/Urban Adaptation, UN Ocean Decade (Seabed 2030), and FAO WaPOR (`0` record overlap with Dev-45). Labeled *retrospectively audited* transparently because system-blind regex, L2-exclusion, and hemisphere-scope fixes were applied during ground-truth auditing. |
| **Holdout B (Retrospectively Audited)** | [`09_build_blind_holdout_b.py`](09_build_blind_holdout_b.py) → `holdout_b_blind_benchmark.json` | **60 queries** (`16` multi-family) | **80** | **686** | **Retrospectively audited 60-query cross-domain holdout** balanced across **10 EO domains** (`6` queries each; `0` record overlap with Dev-45 or Holdout A): Forests & Biomass, Urban & Settlements, Land Cover & Ecosystems, Greenhouse Gases & Carbon, Atmosphere & Air Quality, Climate & Reanalysis, Cryosphere & Polar, Oceans & Maritime, Hydrology & Soils, and Productivity & Hazards (`G = G_eff = 60` independent units). Originally authored blind, then explicitly reclassified as *retrospectively audited* because the same 60 query IDs and 59 of 60 prompts were retained across post-ranking ground-truth audits (`HB06` confidence layer and single-year 2019 map removed, `HB13` regional subset removed, `HB19` EMIT L2B removed, `HB30` coarse `2.5°` records removed via `max_grid_m=60000`). |

### 3.3 Artifact Freeze, Cold-Cache Snapshot Pinning & Statistical Controls
- **12-entry cryptographic lock (`frozen_v4_config.json`):** `07_deleak_and_lofo_dev.py` runs strictly on `Dev-45` (loaded directly from locked `policy_retrieval_benchmark.json`) and records SHA-256 digests of `eo_search_lib.py`, `02_build_policy_benchmark.py`, `06_eval_rerankers.py`, `10_run_blind_holdouts.py`, `full_catalogue.parquet`, `catalogue_content_sha256`, `full_embeddings_bf16.npz`, `bm25_content_sha256`, `record_ids_sha256`, and all three locked benchmark JSON files (`frozen_at_utc = 2026-11-25T13:39:32Z`), plus exact HuggingFace commit revisions across both MLX and PyTorch backends (`mlx-community/embeddinggemma-2-bf16` `1a4ffddb...`, `google/embeddinggemma-2` `f9972ac2...`, `jinaai/jina-reranker-v3.5-mlx` `3dd4ac90...`, `jinaai/jina-reranker-v3.5` `e8a93f33...`, `Qwen/Qwen3-Reranker-0.6B` `e61197ed...`). `10_run_blind_holdouts.py` loads all three benchmarks exclusively from their locked JSON files (zero unhashed Python builder imports), resolves and downloads exact locked HuggingFace snapshots for the selected backend via `ensure_hf_model_snapshot()` (working on empty cold caches as well as warm caches), loads all models directly from those snapshot paths, and verifies every hash and index alignment before scoring (`completed_at_utc = 2026-11-25T14:22:56Z`; `config_frozen_before_eval_run = true`).
- **Statistical inference:** Six hypotheses (`H1`–`H6`) are evaluated on the primary metric (`Fam-nDCG@10`) and four secondary metrics (`MRR@10`, `FamR@10`, `Hit@5`, `Hit@10`) using 10,000-resample paired bootstrap 95% confidence intervals, Monte Carlo sign-flip permutation p-values (`10,000` draws, `seed=2026`), step-down Holm-Bonferroni adjustment (`p_holm`), and small-sample Student-$t$ MDE80.
- **Evaluation boundary:** Benchmarks are **evaluated against curated family rules** rather than manual TREC pooling of every unjudged retrieved record; relevant datasets outside the pre-defined family rules receive zero score, while strict constraint compliance is controlled explicitly via `--hard-limits`.

---

## 4. Evaluation Results (`v4.3-standalone`)

### Reading the Metrics (Plain-Language Guide)

| Metric | Scale | Plain-Language Meaning |
| :--- | :--- | :--- |
| **`Fam-nDCG@10`** (`nDCG@10`, primary) | `0–100%` | **Overall Top-10 ranking quality.** Rewards finding all expected product families within the Top 10 *and* placing them near rank `#1`. Each expected family only counts once (on its first hit), so duplicate editions cannot inflate the score. *(Preferred over fixed-$k$ `F1@10 = 21.5%` / `P@10 = 12.3%`, which is bounded by the `1`–`3` expected families per query; at adaptive cutoff $k = R$, `R-Precision = P@R = Rec@R = F1@R` is `68.1%` on Holdout B.)* |
| **`P@1`** (`Hit@1`) | `0–100%` | **Top-1 precision.** Share of queries where the very first result (`#1`) belongs to an expected product family (`76.67%` for the default config on Holdout B). |
| **`Hit@10`** (and `Hit@5`) | `0–100%` | **Top-10 success rate.** Share of queries that return at least one right product family anywhere in the Top 10 (`100.00%` means zero complete misses across the suite). |
| **`MRR@10`** | `0–100%` | **How high the first right answer sits.** Rank `#1` = `100%`, `#2` = `50%`, `#3` = `33%`, missing from Top 10 = `0%`, averaged over all queries. |
| **`FamR@10`** (`Recall@10`, `FamR@50`) | `0–100%` | **Product-family recall.** Average fraction of a query's expected competing product families found in the Top 10 (or Top 50), regardless of exact rank order. |
| **`Mean Latency (HB)`** | `ms` | Average end-to-end query latency on Holdout B (`batch_size=1`, warm model on Apple Silicon M3 Max). |
| **`pp`, `[95% CI]`, `p_holm`** *(Table 4.2)* | delta / CI / p | Percentage-point gain between two systems, its paired bootstrap 95% confidence interval, and the Holm-Bonferroni p-value adjusted across all 6 comparisons (`p_holm < 0.05` = statistically confirmed). |

All numbers below are taken verbatim from `artifacts/dev45_v4_summary.csv`, `artifacts/holdout_a_v4_summary.csv`, `artifacts/holdout_b_v4_summary.csv`, and their corresponding `*_v4_significance.csv` files (evaluated with solo `batch_size=1` query encoding on Apple Silicon MLX; cross-backend PyTorch parity is verified in Audit 7).

### 4.1 Main Retrieval & Reranking Comparison (`%`)

| System | Dev-45 `nDCG@10` | Dev-45 `Hit@10` | Holdout A `nDCG@10` | Holdout A `Hit@10` | **Holdout B `nDCG@10`** | **Holdout B `MRR@10`** | **Holdout B `FamR@10`** | **Holdout B `Hit@10`** | Mean Latency (HB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01_BM25_PreFilter_Raw` | 42.62 | 82.22 | 64.04 | 88.57 | 67.62 | 66.27 | 87.64 | 93.33 | 1,220.7 ms |
| `02_BM25_PreFilter_GenericCollapse` | 45.62 | 84.44 | 66.30 | 91.43 | 69.75 | 67.30 | 90.14 | 95.00 | 1,238.1 ms |
| `03_Dense_768d_Raw` | 44.03 | 82.22 | 56.93 | 88.57 | 66.55 | 65.97 | 81.81 | 88.33 | 1,159.4 ms |
| `04_Dense_768d_GenericCollapse` | 47.11 | 84.44 | 57.38 | 88.57 | 68.06 | 67.51 | 83.89 | 90.00 | 1,175.4 ms |
| `05_Stage1_HybridRRF_NoCollapse` | 47.23 | 82.22 | 66.23 | 94.29 | 71.31 | 69.82 | 90.97 | 96.67 | 1,232.4 ms |
| `06_Stage1_HybridRRF_GenericCollapse` | 50.61 | 84.44 | 68.46 | 94.29 | 72.63 | 70.58 | 93.06 | 98.33 | 1,247.2 ms |
| `07_Stage1_HybridRRF_NaiveStrict_Collapse` | 51.85 | 91.11 | 69.86 | 94.29 | 74.94 | 74.06 | 92.92 | 98.33 | 1,220.6 ms |
| **`08_Frozen_Best_JinaV35_Top50_RRFBlend`** | **59.56** | 91.11 | **75.76** | **100.00** | **85.45** | **86.11** | **96.39** | **100.00** | 5,507.8 ms |
| `08b_JinaV35_Top50_Pure` (fixed-k ablation) | 60.73 | 91.11 | 71.11 | 100.00 | 85.40 | 85.69 | 96.39 | 100.00 | 5,507.8 ms |
| `09_Frozen_Fast_JinaV35_Top25_Pure` | 59.67 | 91.11 | 71.19 | 100.00 | 84.74 | 85.00 | 95.97 | 100.00 | 2,633.6 ms |
| `10_Frozen_Apache2_Qwen3_Top50_RRFBlend` | 57.80 | 93.33 | 72.25 | 97.14 | 78.12 | 75.96 | 93.89 | 100.00 | 5,187.2 ms |

### 4.2 Hypothesis Tests (`Fam-nDCG@10`, Holm-Bonferroni over 6 Comparisons)

| Hypothesis | Dev-45 (`G=45`) | Audited Holdout A (`G=35`) | **Audited Holdout B (`G=60`)** |
| :--- | :--- | :--- | :--- |
| **H1:** Collapser (`06` vs `05`) | **`+3.39 pp`** `[+1.4, +5.9]` (`p_holm=0.0042`) | **`+2.23 pp`** `[+0.7, +4.2]` (`p_holm=0.0126`) | `+1.33 pp` `[+0.3, +2.7]` (`p_holm=0.0620`) |
| **H2:** Hybrid vs BM25+Col (`06` vs `02`) | `+4.99 pp` `[-1.3, +11.3]` (`p_holm=0.4110`) | `+2.16 pp` `[-6.9, +11.9]` (`p_holm=0.6750`) | `+2.88 pp` `[-3.3, +9.0]` (`p_holm=0.3713`) |
| **H3:** Hybrid vs Dense+Col (`06` vs `04`) | `+3.50 pp` `[-2.0, +9.0]` (`p_holm=0.4554`) | **`+11.09 pp`** `[+4.7, +18.0]` (`p_holm=0.0126`) | `+4.57 pp` `[-1.2, +10.3]` (`p_holm=0.2928`) |
| **H4:** Jina v3.5 Blend vs S1 (`08` vs `06`) | **`+8.94 pp`** `[+3.6, +14.5]` (`p_holm=0.0105`) | **`+7.30 pp`** `[+2.2, +12.4]` (`p_holm=0.0244`) | **`+12.82 pp`** `[+7.6, +18.3]` (`p_holm=0.0006`) |
| **H5:** Qwen3-0.6B Blend vs S1 (`10` vs `06`) | `+7.18 pp` `[+0.5, +14.3]` (`p_holm=0.1624`) | `+3.79 pp` `[-2.7, +10.4]` (`p_holm=0.6422`) | `+5.49 pp` `[-0.9, +11.7]` (`p_holm=0.2928`) |
| **H6:** Jina v3.5 vs Qwen3 (`08` vs `10`) | `+1.76 pp` `[-4.4, +7.9]` (`p_holm=0.5757`) | `+3.51 pp` `[-1.8, +9.0]` (`p_holm=0.6422`) | **`+7.33 pp`** `[+2.5, +12.5]` (`p_holm=0.0200`) |

### Key Takeaways
- **Generic family collapsing (`H1`) improves ranking cleanly while preserving distinct variables, layer roles, and cadences:** Holm-significant across `Dev-45` (`+3.39 pp`, `p_holm = 0.0042`) and `Holdout A` (`+2.23 pp`, `p_holm = 0.0126`), and positive on `Holdout B` (`+1.33 pp`, `p_holm = 0.0620`), while Audit 3 confirms **`0/61` (`0.0%`)** false cross-family collapses on Holdout A and **`0/77` (`0.0%`)** on Holdout B (`2/98` on Dev-45, both true sibling pairs: EMODnet seabed PAR light and MODIS Aqua `MYD10A2` vs. Terra `MOD10A2`), and Audit 0 confirms all 6 `Annual NLCD` outputs and both `GLC_FCS30D` cadences stay distinct.
- **Stage-1 hybrid fusion outperforms single retrievers across every suite:** Hybrid RRF + Collapse (`06`) reaches `FamR@50 = 89.44%` / `92.38%` / `98.47%` and `98.33%` `Hit@10` on Holdout B (`1,137.9 ms` mean full-evaluation Stage-1 latency across all 140 queries including 39k filtering; `8.25 ms` mean solo query encoding; `167–179 ms` interactive CLI Stage-1; `artifacts/v4_runtime_manifest.json`).
- **Listwise `Jina v3.5` delivers a large, statistically decisive Stage-2 lift (`H4`, `H6`):** On Holdout B (`G = 60`), `Jina v3.5` raises `Fam-nDCG@10` by `+12.82 pp` (`72.63%` → `85.45%`, `p_holm = 0.0006`, `28W/27T/5L`) and `MRR@10` by `+15.53 pp` (`70.58%` → `86.11%`, `p_holm = 0.0006`), reaching `100.00%` `Hit@5`, `100.00%` `Hit@10`, and `76.67%` `P@1` (`77.91%` to `95.15%` `Fam-nDCG@10` across all 10 EO domains; `artifacts/holdout_b_v4_by_domain.csv`). The fast Top-25 preset (`09`) reaches `84.74%` `Fam-nDCG@10` in `2,633.6 ms` (`~2.1×` faster) while trading `-4.57 pp` on Holdout A where multi-family candidates sit in ranks `26–50`.
- **Honest `Qwen3-0.6B` (`Apache-2.0`) trade-off:** `Qwen3-0.6B` reaches `100.00%` `Hit@10` on Holdout B (`78.12%` `Fam-nDCG@10`, `+5.49 pp` over Stage-1, `p_holm = 0.2928`) and leads `Jina v3.5` in `Hydrology & Soils` (`84.36%` vs `79.58%`) and `Atmosphere & Air Quality` (`83.02%` vs `81.02%`), while trailing `Jina v3.5` overall by `-7.33 pp` (`p_holm = 0.0200`).
- **Unconfounded Top-50 blend vs pure ablation (`artifacts/v4_unconfounded_top50_blend_vs_pure.csv`):** At fixed `k=50`, blending Jina with Stage-1 RRF adds `+4.65 pp` on `Holdout A` (`p = 0.0266`) and `+0.05 pp` on `Holdout B` (`p = 0.9582`), while differing by `-1.17 pp` on `Dev-45` (`p = 0.3880`).
- **Null-safe vs naive strict vs hard-limits (`artifacts/audit_dev45_filter_stage2.json`):** On `all_45`, `NullSafe_Jina50_Blend` leads `NaiveStrict_Jina50_Blend` (`59.56%` vs `58.82%` `Fam-nDCG@10`). On the 13 grid-filtered `Dev-45` queries, `null_safe=True` preserves higher family recall (`FamR@50 = 90.38%` vs `83.33%`), whereas `null_safe=False` yields higher Top-10 ranking quality when targets have populated metadata (`Fam-nDCG@10`: `63.23%` vs `61.44%` after Jina reranking). Use `--hard-limits` whenever exact spatial footprint, verified output-grid spacing, documented end years, and verified processing levels are mandatory over discovery recall.

---

## 5. Reproducible Verification Audits

Running [`11_reproducible_audits.py`](11_reproducible_audits.py) verifies all auxiliary engineering and statistical claims (`8/8` passed) and writes persistent JSON/CSV artifacts:

| Audit | Artifact | Verified Result (`v4.3-standalone`) |
| :--- | :--- | :--- |
| **0. BM25 Determinism, Alignment, Hard Limits, Scope & Grouping** | `artifacts/audit_bm25_determinism.json` | `0/45` NullSafe and `0/45` NaiveStrict Top-300 diffs (`0/90` checks); full two-build DuckDB rebuild produces byte-identical `bm25_content_sha256` (`a88c6a5c78c9...`); `verify_index_alignment()` rejects row-swapped and metadata-mutated inputs; `ensure_search_indexes()` accepts equivalent local DuckDB rebuilds by semantic content; hard-limits `min_end_year=2035`, verified output-grid (`OTD` & CMEMS `0-700m` depth-range excluded), and global L4 `<= 60 km` (`Tellus` Greenland/Ocean/Antarctica scalar time series excluded) checks PASS; `0` hemisphere leaks on global queries across all 3 locked benchmarks & `HB06` excludes 2019-only map; all 6 `Annual NLCD` outputs and both `GLC_FCS30D` cadences stay distinct. |
| **1. Dense Encoder Exactness** | `artifacts/audit_mlx_bit_exactness.json` | Fast MLX path vs stock `mlx_vlm`: `max_abs_diff = 0.0` for both `L <= 512` (`4` docs) and `L > 513` sliding-window mask (`640`, `617` tokens); `batch=1` vs `batch=16` `min_cos = 0.999839`. |
| **2. Jina Causal Position Bias** | `artifacts/audit_jina_position_bias.json` | Top-50 forward vs 180° reversed (`N=15`): mean Spearman `rho = 0.8369` (`min = 0.6579`), Top-5 overlap `68.0%`, Top-1 exact agreement `60.0%`, recency boost `+0.01327`. |
| **3. Cross-Family False Collapse** | `artifacts/audit_false_collapse.json` | Exact `scan_limit = 200` denominator: Dev-45 `2/98` (`2.04%`, EMODnet seabed PAR & MODIS Aqua/Terra snow siblings), Holdout A **`0/61` (`0.0%`)**, Holdout B **`0/77` (`0.0%`)**. |
| **4. Nested 6-Fold LOFO-CV** | `artifacts/dev45_v4_nested_lofo_cv.csv` | `0.000000 pp` drift vs script `10`; `near_dup_cos = 0.94` selected in `6/6` folds; primary `JinaV35_Top50_RRFBlend` selected in `2/6` folds (`Top-25` in `4/6`) with `+3.22` to `+16.42 pp` out-of-fold lift (`08b_JinaV35_Top50_Pure` selected in `6/6` folds when included in pool). |
| **5. Filter × Stage-2 Check** | `artifacts/audit_dev45_filter_stage2.json` | Quantifies `NullSafe` vs `NaiveStrict` after Jina & Qwen3 reranking across `all_45` and `grid_filtered_13`. |
| **6. CLI Purity & Latency** | `artifacts/audit_cli_latency.json` | `6/6` CLI probes produce valid JSON (`bm25-only`: `149.3 ms`; `stage-1`: `166.9–178.9 ms`; `jina-25`: `1,102.2 ms`; `jina-50`: `3,052.6 ms`; `qwen3-50`: `1,804.4 ms`). |
| **7. Cross-Backend Parity** | `artifacts/audit_backend_parity.json` | Apple Silicon MLX BF16 vs PyTorch CPU BF16 (incl. `>512`-token truncated doc): `EmbeddingGemma 2` **`min_cos = 0.999950`** (`mean = 0.999955`); `Jina v3.5` `max_diff = 0.00482`, **`exact_rank_agreement = True`**; `Qwen3-0.6B` `max_diff = 0.12500`, **`exact_rank_agreement = True`**. |

---

## 6. Hardware, Licensing & Limitations

- **Auto-selected backends (`--device auto`):** Uses `MLX` (`mlx-community/embeddinggemma-2-bf16`, `jinaai/jina-reranker-v3.5-mlx`, `Qwen/Qwen3-Reranker-0.6B`) on Apple Silicon macOS; automatically switches to standard PyTorch (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`) on CUDA GPUs and CPUs. CPU indexing of all `39,472` records is slower than GPU/MLX; use the prebuilt `v4.3.0-indexes` GitHub Release (downloaded automatically) or `--bm25-only` (`~36 s`) when rebuilding locally without an accelerator.
- **Model licenses:** `EmbeddingGemma 2` and `Qwen3-Reranker-0.6B` are `Apache-2.0`; `Jina Reranker v3.5` is `CC BY-NC 4.0` (non-commercial). All pipeline code in this directory is `MIT`; catalogue data and documentation are `CC-BY-4.0`.
- **Evaluation bounds:** Validation is **evaluated against curated family rules** without manual TREC pooling of every unjudged retrieved record; Holdout A and Holdout B are **retrospectively audited** holdouts (not fresh single-shot unseen blind tests); relevant datasets outside the pre-defined family rules receive zero credit; and strict compliance with hard spatial/grid/date/level constraints requires `--hard-limits`.

---

## 7. Reproducing the Full Validation Suite

```bash
# 1. Build local indexes (one-time; or allow auto-download of v4.3.0-indexes)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py

# 2. (Optional) Rebuild locked benchmark JSONs from canonical catalogue
uv run 06_semantic_hybrid_search/02_build_policy_benchmark.py
uv run 06_semantic_hybrid_search/08_build_holdout_a_external.py
uv run 06_semantic_hybrid_search/09_build_blind_holdout_b.py

# 3. Freeze Dev-45 config (12-entry SHA-256 + HF revision lock), evaluate all 3 suites, and run all 8 audits
uv run 06_semantic_hybrid_search/07_deleak_and_lofo_dev.py
uv run 06_semantic_hybrid_search/10_run_blind_holdouts.py
uv run 06_semantic_hybrid_search/11_reproducible_audits.py
```

### Directory Layout

```text
06_semantic_hybrid_search/
├── README.md                          # Documentation, methodology, results & audits
├── eo_search_lib.py                   # Core library (record_id/content-bound DuckDB BM25, snapshot download, RRF, collapser, hard-limits)
├── 04_search_cli.py                   # Standalone & --server client CLI (--hard-limits, --rebuild-index, --bm25-only, --json)
├── 04_search_server.py                # Local FastAPI server (/api/search, /api/export, /api/product) + web UI
├── web/
│   ├── index.html & app.js            # Tailwind + MapLibre GL UI with Verified hard-limits toggle & CSV/JSON/XLSX/Parquet export
├── 01_corpus_token_and_mlx_bench.py   # Corpus length profiling & MLX throughput benchmark
├── 02_build_policy_benchmark.py       # Dev-45 builder (45 queries, 100 expected families -> locked JSON)
├── 03_index_and_eval_subset.py        # Early subset ablation script
├── 05_index_full_catalogue.py         # Full 39,472-record index builder + full_index_manifest.json (--device auto / --bm25-only)
├── 06_eval_rerankers.py               # Stage-2 reranker loaders (MLX & PyTorch, snapshot-pinned) + reranker ablations
├── 07_deleak_and_lofo_dev.py          # Dev-45 collapser sweep, filter ablation & frozen_v4_config.json 12-hash lock
├── 08_build_holdout_a_external.py     # Retrospectively audited external domain benchmark builder (Holdout A, 35 queries)
├── 09_build_blind_holdout_b.py        # Retrospectively audited cross-domain benchmark builder (Holdout B, 60 queries, 80 families)
├── 10_run_blind_holdouts.py           # v4.3-standalone 3-suite evaluation (locked JSONs + cold-cache snapshots) & Holm-Bonferroni significance
├── 11_reproducible_audits.py          # 8 reproducible verification audits (incl. alignment, hard-limits, scope, grouping & MLX/PyTorch parity)
└── artifacts/                         # Frozen SHA-256 locks, benchmark JSONs, CSV results & audit outputs
```
