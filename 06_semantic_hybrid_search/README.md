# Local Hybrid Semantic Search (`06_semantic_hybrid_search`)

Local hybrid search engine, CLI, web interface, and validation benchmark over all **39,474 canonical Level-3 and Level-4 Earth Observation (EO) catalogue records** from [`../03_harmonization/data/eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet). Evaluated against curated family rules, the default hybrid configuration (`EmbeddingGemma 2 + DuckDB BM25 + Jina v3.5`) achieves **86.96% Fam-nDCG@10** (**80.00% P@1**, **96.39% family Recall@10**, **100.00% Hit@10**) on the 60-query blind holdout (`v4.1-standalone`).

The search engine combines dense semantic embeddings (`EmbeddingGemma 2`), EO-normalized multi-field BM25 (`DuckDB >= 1.5.0` FTS keyed by `record_id`), cadence/resolution/footprint-guarded family collapsing, and optional Stage-2 neural reranking (`Jina Reranker v3.5` or `Qwen3-Reranker-0.6B`). Backends are **device-agnostic (`--device auto`)**: scripts automatically select native **MLX** on Apple Silicon (`darwin/arm64`), **PyTorch CUDA** when an NVIDIA GPU is present, or **PyTorch CPU** otherwise, with an instant `--bm25-only` lexical mode requiring zero neural weights.

<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/d6dbff1f-5b93-4a4a-9d90-d3240c526510" />
<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/64c066dc-cb58-42ec-aceb-774c352c16cc" />
<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/5a070fdf-1221-49f9-ac3f-e29d6f56baf7" />

---

## 1. Quick Start (`uv run`)

All scripts use PEP 723 inline dependencies and require only [`uv`](https://docs.astral.sh/uv/). Run commands from the repository root.
Recommended quickstart (auto-downloads & SHA-256-verifies the prebuilt `v4.1.0-indexes` release assets on first launch):

```bash
uv run 06_semantic_hybrid_search/04_search_server.py --port 8080 --open
```

### 1.1 Prebuilt Release Indexes & Local Index Rebuilds

Large binary index artifacts (`full_bm25.duckdb`, `124.5 MB`; `full_embeddings_bf16.npz`, `54.2 MB`; `full_index_manifest.json`) are excluded from git and hosted on GitHub Releases (`v4.1.0-indexes`). Both `04_search_server.py` and `04_search_cli.py` automatically fetch and verify them via `ensure_search_indexes()` when missing locally:

- `--rebuild-index`: build `full_bm25.duckdb` and `full_embeddings_bf16.npz` locally from `eo_l3_l4_canonical_products.parquet` instead of downloading release binaries.
- `--no-download-index`: disallow automatic release downloads (fail fast unless local index files are already present).

```bash
# Build local full hybrid index explicitly (auto-selects MLX / CUDA / CPU)
# ~6.6 min on Apple Silicon M3 Max (2.55 GB peak memory; ~36 s for BM25 alone)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py

# Fast lexical-only local build on any OS (zero neural models, ~36 s)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py --bm25-only
```

### 1.2 Launch the Local Web UI & API Server

Starts a local FastAPI server (auto-selecting `MLX` / `CUDA` / `CPU`, or `--bm25-only`) and serves the Tailwind CSS + MapLibre GL search interface (`web/`). The interface supports all ranking modes, structured facets, a **Verified (Hard Limits)** toggle (`--hard-limits`) with unknown-metadata exclusion reporting, family-collapsed sibling drawers, interactive bounding-box maps, and a single **Download** dropdown exporting results to **CSV**, **JSON**, **Excel (`.xlsx`)**, or **Parquet**:

```bash
# Auto-detect best backend (auto-downloads v4.1.0-indexes if absent) and open http://127.0.0.1:8080
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

# Verified hard-limits mode: enforces exact global footprint, <=250 m grid, no nulls
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
| **Best Quality (Default)** | `--rerank jina-v3.5 --rerank-pool 50` | CC BY-NC 4.0 | ~5.0 s (HB) / ~3.4 s (CLI warm) | Highest ranking accuracy for research & non-commercial analysis |
| **Fast Rerank** | `--rerank jina-v3.5 --rerank-pool 25 --no-blend` | CC BY-NC 4.0 | ~2.1 s (HB) / ~1.2 s (CLI warm) | Interactive screening with near-identical Top-10 quality (~2.4× faster) |
| **Permissive Rerank** | `--rerank qwen3-0.6b --rerank-pool 50` | Apache-2.0 | ~5.2 s (HB) / ~1.8 s (CLI warm) | Commercial or permissive-license Stage-2 reranking |
| **Stage-1 Hybrid Only** | `--rerank none` | Apache-2.0 | ~617 ms (HB) / ~166–181 ms (CLI warm) | Instant interactive retrieval (`98.33%` Top-10 hit rate on blind holdout) |
| **BM25 Only** | `--bm25-only` | MIT | ~143 ms (CLI warm) | Zero-model lexical search on any CPU |

Server API endpoints: `GET /api/status`, `POST /api/search` (and `GET /api/search`), `POST /api/export` (`csv`, `json`, `xlsx`, `parquet`), and `GET /api/product/{record_id}`.

---

## 2. Methodology

The pipeline is built around two practical realities of federated EO metadata catalogues:

1. **Upstream metadata is sparse:** `70.9%` of canonical records lack an adjudicated pixel grid value and `50.3%` lack structured platform metadata. Standard strict SQL filters silently drop relevant products whose metadata fields are `null`.
2. **Catalogue listings are heavily versioned and mirrored:** A single scientific product family (e.g., an ESA CCI climate record or NASA MODIS product) often appears as dozens of versioned releases, annual editions, or cross-portal mirrors that would otherwise crowd out distinct alternative datasets in the Top-10.

Every query runs through four deterministic stages implemented in [`eo_search_lib.py`](eo_search_lib.py) and [`06_eval_rerankers.py`](06_eval_rerankers.py):

### Stage 1 — Structured Pre-Filtering & Verified Hard-Limits Mode
Records flagged as superseded (`is_superseded_edition = True` or obsolete/deprecated title tags) are excluded by default, leaving `38,851` active records. Optional facets (`provider_pool`, `geo_scope`, `max_grid_m`, `processing_levels`, `strict_eo_only`) are applied as a boolean mask *before* scoring in three explicit precision/recall modes:

- **Null-Safe Discovery (Default):** Keeps records whose requested metadata attribute is `null` and applies a `1.25×` tolerance on `max_grid_m` (e.g., allowing `300 m` Sentinel-3/OLCI products on a `250 m` request). `global` also permits `hemispheric_or_pantropical` collections (excluding explicitly single-hemisphere subsets when `geo_scope` is filtered).
- **Strict Known-Values (`--strict-filter` / `null_safe=False`):** Excludes records with missing (`null`) metadata on active filter facets while retaining the `1.25×` grid tolerance and broad global discovery semantics unless `--hard-limits` is enabled.
- **Verified Hard-Limits (`--hard-limits` / `hard_limits=True`):** Enforces strict scientific constraints (`null_safe=False`, exact `1.0×` grid tolerance so `adjudicated_output_grid_meters <= max_grid_m` with no 25% slack, exact `processing_levels`, and strict `geographic_coverage == 'global'` excluding hemispheric, pantropical, and single-hemisphere records) while reporting `unknown_metadata_counts` for records excluded due to `null` fields.

### Stage 2 — Hybrid Dense + `record_id`-Keyed BM25 Retrieval (RRF)
Two complementary rankers score all eligible records in parallel (Top-300 candidates each), guarded by fail-fast `verify_index_alignment()` checks confirming exact length, row order, and `record_id` agreement across the canonical Parquet table, dense matrix, and DuckDB index:

- **Dense bi-encoder (`EmbeddingGemma 2`, 768-d):** Each canonical record is serialized into a structured EO card (title, IDs, institution, domain, spatial/temporal metadata, deduplicated GCMD keywords, cleaned abstract; truncated at `512` tokens, no chunking). Queries use asymmetric query encoding (`batch_size=1`, `max_length=256`). Apple Silicon loads `mlx-community/embeddinggemma-2-bf16` via MLX; CUDA and CPU load `google/embeddinggemma-2` via PyTorch (`bfloat16` on Ampere+ CUDA and CPU, `float32` on older GPUs/MPS). Cosine ties break deterministically by ascending `record_id`.
- **Multi-field BM25 (`DuckDB >= 1.5.0` FTS):** Keyed directly by immutable `record_id` (with `eo_meta` storing `record_id` sequence and `bm25_content_sha256`). Indexes three normalized fields (`title` weight `2.5`, `keywords` `1.5`, `body` `1.0`, `k1=1.2`, `b=0.75`) using deterministic single-thread execution (`SET threads = 1`, `ROUND(score, 10)`) and generic EO-domain lexical normalization (UK/US spelling such as `colour`/`color`, `burnt`/`burned`, hyphen/underscore splits, and standard physical variable acronyms such as `CO2`, `CH4`, `LAI`, `SST`, `LST`, `DEM`—zero product or mission names).
- **Reciprocal Rank Fusion (RRF):** Combines dense and lexical rankings using $S_{\text{RRF}}(d) = \frac{1.0}{60 + r_{\text{dense}}(d)} + \frac{0.6}{60 + r_{\text{bm25}}(d)}$.

### Stage 3 — Generic Sensor- & Variant-Guarded Family Collapsing
Scans up to `max(top_k * 3, 200)` fused candidates and groups versioned editions and unlinked portal mirrors under the representative record without any benchmark-specific regexes (`25,042` unique generic stems across the `39,474` catalogue rows). Two candidates collapse when either:
- They share the same generic normalized title stem (`gstem:`, stripping leading catalogue codes, parenthetical tags, versions, years, and resolution tokens), **or**
- Their 768-d dense cosine similarity reaches `near_dup_cos >= 0.94` (selected strictly on `Dev-45`).

To prevent distinct scientific products from merging, collapsing is blocked whenever two records differ in **physical instrument/sensor**, **temporal cadence class** (e.g., daily vs. monthly vs. annual), **geographic coverage class** (e.g., global vs. regional), **`> 2.0×` spatial grid resolution**, or **distinct non-empty DOIs**. When a collapsed cluster contains multiple valid editions, **query-matching variant promotion** (`collapse_product_families(..., query_text)`) promotes the sibling whose cadence, resolution, or coverage tokens match the user's query (e.g., surfacing `MONTHLY_V3` or `300m` ahead of an arbitrary annual variant). All collapsed siblings remain inspectable in `collapsed_siblings`.

### Stage 4 — Optional Stage-2 Neural Reranking
Rescores the Top-50 (or Top-25) collapsed family representatives using concise `<= 1,200`-character metadata cards:
- **`Jina Reranker v3.5`** (`jinaai/jina-reranker-v3.5-mlx` on Apple Silicon; `jinaai/jina-reranker-v3.5` on CUDA/CPU; `CC BY-NC 4.0`): listwise cross-attention reranker (`max_doc_length=420`), blended with Stage-1 RRF ranks via $S(d) = \frac{0.4}{30 + r_{\text{S1}}(d)} + \frac{1.0}{30 + r_{\text{S2}}(d)}$ (or pure reranker order for the fast Top-25 preset).
- **`Qwen3-Reranker-0.6B`** (`Qwen/Qwen3-Reranker-0.6B`; `Apache-2.0`): instruction-aware pairwise reranker scored via raw `yes`/`no` log-odds margin (`logit("yes") - logit("no")`) with deterministic stable sorting and the same RRF blend.

---

## 3. Validation Protocol & Benchmark Construction (`v4.1-standalone`)

### 3.1 What the Benchmarks Represent & How They Were Built

Each benchmark query represents a realistic Earth Observation discovery request phrased in domain and policy language—describing the biophysical variable, spatial/temporal requirement, or policy indicator *without* naming target products, satellite missions, or producing agencies. Because one scientific dataset often appears across dozens of versioned releases and cross-portal mirrors in the `39,474`-record catalogue, ground truth is defined at the **product-family level** (`1`–`3` distinct institutional product families per query) rather than single rows.

Every benchmark suite is constructed deterministically in code directly against the `38,851` active, non-superseded records of [`eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet):

1. **Source selection:** Queries and expected product families are derived from authoritative public EO policy frameworks, Essential Climate Variable (ECV) registries, and domain catalogues—never generated from search-engine outputs.
2. **Constraint-aligned family resolution:** Each expected family is matched to qualifying active records via `filter_benchmark_family_mask()`, combining explicit inclusion/exclusion rules (`title`, `raw_id`, `source_catalogue`) with the query's own structured scope constraints (`geo_scope`, `max_grid_m`, `strict_eo_only`, processing level, excluding superseded/obsolete editions) so regional subsets or mismatched resolutions cannot receive ground-truth credit on global or fine-grid queries.
3. **Automated integrity checks:** Each builder script enforces four programmatic assertions before writing its benchmark JSON:
   - Every expected family matches `>= 1` active record and remains reachable under the query's structured filters (`geo_scope`, `max_grid_m`, `strict_eo_only`).
   - `0` `record_id` overlaps exist between distinct expected families inside any query.
   - `0` `record_id` overlaps exist across the three suites (`Dev-45 ∩ Holdout A = ∅`, `Dev-45 ∩ Holdout B = ∅`, `Holdout A ∩ Holdout B = ∅`), and `0` overlaps exist across any pair of `Holdout B` queries.
   - `0` product brand names, mission/sensor codes, or provider shortnames appear in holdout query text (enforced by automated anti-leakage blocklists of `70+` terms in `Holdout A` and `110+` terms in `Holdout B`).

### 3.2 Benchmark Suites & Source Provenance

| Suite | Builder & Artifact | Size | Expected Families | Unique Active Records | Sources & Role |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Dev-45** | [`02_build_policy_benchmark.py`](02_build_policy_benchmark.py) → `policy_retrieval_benchmark.json` | 45 queries (`32` multi-family) | 101 | 1,215 (`940` subset GT) | **Development set** drawn from 6 policy-to-EO frameworks cross-referenced in `05_external_policy_validation/`: EU Biodiversity Strategy 2030 (`14`, EC KCEO Annex 3), CBD Kunming-Montreal GBF Indicator A.2 (`7`, CEOS EETT), UNFCCC Global Stocktake (`7`, CEOS GST / US GHG Center / UNCCD), UN SDGs `6.6.1`/`11.3.1`/`11.6.2`/`14.1.1a`/`15.3.1` (`7`, CEOS / UN-Habitat / UNCCD GPG v2), EC KCEO Urban Climate Adaptation (`5`), and USGEO EOA 2023 / NASA SNWG (`5`). Used solely to select `near_dup_cos = 0.94`, ablate filters, and run nested 6-fold Leave-One-Framework-Out CV. |
| **Holdout A** | [`08_build_holdout_a_external.py`](08_build_holdout_a_external.py) → `holdout_a_external_benchmark.json` | 35 queries (`22` multi-family) | 62 | 1,645 | **Retrospective GT-audited external domain set** drawn from unused international science/policy registries: GCOS ECV (WGClimate), WMO OSCAR/Space, USGEO EOA 2023, NASA SNWG, EC KCEO Marine/Urban Adaptation, UN Ocean Decade (Seabed 2030), and FAO WaPOR (`0` record overlap with Dev-45). Labeled *retrospective* transparently because uniform, system-blind regex fixes (e.g., `sulfur`/`sulphur` spelling variants and omitted official editions) were applied during ground-truth auditing. |
| **Holdout B (Primary Blind)** | [`09_build_blind_holdout_b.py`](09_build_blind_holdout_b.py) → `holdout_b_blind_benchmark.json` | **60 queries** (`16` multi-family) | **80** | **697** | **Blind holdout suite** balanced across **10 EO domains** (`6` queries each; `0` record overlap with Dev-45 or Holdout A): Forests & Biomass, Urban & Settlements, Land Cover & Ecosystems, Greenhouse Gases & Carbon, Atmosphere & Air Quality, Climate & Reanalysis, Cryosphere & Polar, Oceans & Maritime, Hydrology & Soils, and Productivity & Hazards (`G = G_eff = 60` independent units). Updated in `v4.1-standalone` solely to align `HB13` and `HB30` ground-truth rules with their geographic scope prior to full re-locking. |

### 3.3 Prospective Freeze & Statistical Controls
- **10-entry cryptographic lock (`frozen_v4_config.json`):** `07_deleak_and_lofo_dev.py` runs strictly on `Dev-45` and records SHA-256 digests of `eo_search_lib.py`, `06_eval_rerankers.py`, `10_run_blind_holdouts.py`, `full_catalogue.parquet`, `full_embeddings_bf16.npz`, `bm25_content_sha256` (deterministic across DuckDB rebuilds), `record_ids_sha256`, and all three benchmark JSON files (`frozen_at_utc = 2026-11-24T12:24:29Z`). `10_run_blind_holdouts.py` verifies every hash and index alignment before scoring `Holdout B` once (`completed_at_utc = 2026-11-24T13:00:11Z`; `strictly_prospective_timestamp_order = true`).
- **Pre-registered inference:** Six hypotheses (`H1`–`H6`) are evaluated on the primary metric (`Fam-nDCG@10`) and four secondary metrics (`MRR@10`, `FamR@10`, `Hit@5`, `Hit@10`) using 10,000-resample paired bootstrap 95% confidence intervals, Monte Carlo sign-flip permutation p-values (`10,000` draws, `seed=2026`), step-down Holm-Bonferroni adjustment (`p_holm`), and small-sample Student-$t$ MDE80.
- **Evaluation boundary:** Benchmarks are **evaluated against curated family rules** rather than manual TREC pooling of every unjudged retrieved record; relevant datasets outside the pre-defined family rules receive zero score, while hard constraint compliance is controlled explicitly via `--hard-limits`.

---

## 4. Evaluation Results (`v4.1-standalone`)

### Reading the Metrics (Plain-Language Guide)

| Metric | Scale | Plain-Language Meaning |
| :--- | :--- | :--- |
| **`Fam-nDCG@10`** (`nDCG@10`, primary) | `0–100%` | **Overall Top-10 ranking quality.** Rewards finding all expected product families within the Top 10 *and* placing them near rank `#1`. Each expected family only counts once (on its first hit), so duplicate editions cannot inflate the score. *(Preferred over fixed-$k$ `F1@10 = 21.5%` / `P@10 = 12.3%`, which is bounded by the `1`–`3` expected families per query; at adaptive cutoff $k = R$, `R-Precision = P@R = Rec@R = F1@R` is `70.8%` on Blind Holdout B.)* |
| **`P@1`** (`Hit@1`) | `0–100%` | **Top-1 precision.** Share of queries where the very first result (`#1`) belongs to an expected product family (`80.00%` for the best config on Blind Holdout B). |
| **`Hit@10`** (and `Hit@5`) | `0–100%` | **Top-10 success rate.** Share of queries that return at least one right product family anywhere in the Top 10 (`100.00%` means zero complete misses across the suite). |
| **`MRR@10`** | `0–100%` | **How high the first right answer sits.** Rank `#1` = `100%`, `#2` = `50%`, `#3` = `33%`, missing from Top 10 = `0%`, averaged over all queries. |
| **`FamR@10`** (`Recall@10`, `FamR@50`) | `0–100%` | **Product-family recall.** Average fraction of a query's expected competing product families found in the Top 10 (or Top 50), regardless of exact rank order. |
| **`Mean Latency (HB)`** | `ms` | Average end-to-end query latency on Blind Holdout B (`batch_size=1`, warm model on Apple Silicon M3 Max). |
| **`pp`, `[95% CI]`, `p_holm`** *(Table 4.2)* | delta / CI / p | Percentage-point gain between two systems, its paired bootstrap 95% confidence interval, and the Holm-Bonferroni p-value adjusted across all 6 comparisons (`p_holm < 0.05` = statistically confirmed). |

All numbers below are taken verbatim from `artifacts/dev45_v4_summary.csv`, `artifacts/holdout_a_v4_summary.csv`, `artifacts/holdout_b_v4_summary.csv`, and their corresponding `*_v4_significance.csv` files (evaluated with solo `batch_size=1` query encoding on Apple Silicon MLX; cross-backend PyTorch parity is verified in Audit 7).

### 4.1 Main Retrieval & Reranking Comparison (`%`)

| System | Dev-45 `nDCG@10` | Dev-45 `Hit@10` | Holdout A `nDCG@10` | Holdout A `Hit@10` | **Blind Holdout B `nDCG@10`** | **Blind Holdout B `MRR@10`** | **Blind Holdout B `FamR@10`** | **Blind Holdout B `Hit@10`** | Mean Latency (HB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01_BM25_PreFilter_Raw` | 45.12 | 86.67 | 64.04 | 88.57 | 69.71 | 68.97 | 87.64 | 93.33 | 590.3 ms |
| `02_BM25_PreFilter_GenericCollapse` | 48.52 | 88.89 | 66.33 | 91.43 | 71.70 | 69.84 | 90.14 | 95.00 | 605.9 ms |
| `03_Dense_768d_Raw` | 45.14 | 86.67 | 56.59 | 88.57 | 67.05 | 66.16 | 83.47 | 90.00 | 526.1 ms |
| `04_Dense_768d_GenericCollapse` | 48.41 | 88.89 | 57.10 | 88.57 | 68.59 | 67.73 | 85.56 | 91.67 | 541.4 ms |
| `05_Stage1_HybridRRF_NoCollapse` | 49.02 | 86.67 | 66.23 | 94.29 | 72.87 | 71.90 | 90.97 | 96.67 | 602.3 ms |
| `06_Stage1_HybridRRF_GenericCollapse` | 52.65 | 88.89 | 68.46 | 94.29 | 74.19 | 72.67 | 93.06 | 98.33 | 617.2 ms |
| `07_Stage1_HybridRRF_NaiveStrict_Collapse` | 53.67 | 95.56 | 69.23 | 94.29 | 76.52 | 76.14 | 92.92 | 98.33 | 616.5 ms |
| **`08_Frozen_Best_JinaV35_Top50_RRFBlend`** | **63.03** | 95.56 | **75.05** | **100.00** | **86.96** | **88.19** | **96.39** | **100.00** | 5,009.9 ms |
| `08b_JinaV35_Top50_Pure` (fixed-k ablation) | 63.90 | 95.56 | 70.69 | 100.00 | 87.38 | 88.19 | 96.39 | 100.00 | 5,009.9 ms |
| `09_Frozen_Fast_JinaV35_Top25_Pure` | 62.41 | 95.56 | 71.32 | 100.00 | 86.51 | 87.22 | 95.97 | 100.00 | 2,061.2 ms |
| `10_Frozen_Apache2_Qwen3_Top50_RRFBlend` | 59.48 | 97.78 | 72.29 | 97.14 | 78.33 | 76.24 | 93.89 | 100.00 | 5,184.3 ms |

### 4.2 Pre-Registered Hypothesis Tests (`Fam-nDCG@10`, Holm-Bonferroni over 6 Comparisons)

| Hypothesis | Dev-45 (`G=45`) | Audited Holdout A (`G=35`) | **Blind Holdout B (`G=60`)** |
| :--- | :--- | :--- | :--- |
| **H1:** Collapser (`06` vs `05`) | **`+3.63 pp`** `[+1.6, +6.2]` (`p_holm=0.0036`) | **`+2.23 pp`** `[+0.7, +4.2]` (`p_holm=0.0105`) | `+1.33 pp` `[+0.3, +2.7]` (`p_holm=0.0620`) |
| **H2:** Hybrid vs BM25+Col (`06` vs `02`) | `+4.13 pp` `[-2.3, +10.6]` (`p_holm=0.4550`) | `+2.13 pp` `[-7.0, +11.9]` (`p_holm=0.8174`) | `+2.50 pp` `[-3.7, +8.6]` (`p_holm=0.4572`) |
| **H3:** Hybrid vs Dense+Col (`06` vs `04`) | `+4.24 pp` `[-1.4, +9.8]` (`p_holm=0.4332`) | **`+11.36 pp`** `[+5.0, +18.1]` (`p_holm=0.0096`) | `+5.60 pp` `[-0.4, +11.8]` (`p_holm=0.2451`) |
| **H4:** Jina v3.5 Blend vs S1 (`08` vs `06`) | **`+10.38 pp`** `[+4.8, +16.1]` (`p_holm=0.0036`) | **`+6.59 pp`** `[+1.9, +11.4]` (`p_holm=0.0332`) | **`+12.76 pp`** `[+7.5, +18.3]` (`p_holm=0.0006`) |
| **H5:** Qwen3-0.6B Blend vs S1 (`10` vs `06`) | `+6.82 pp` `[+0.2, +13.9]` (`p_holm=0.2100`) | `+3.83 pp` `[-2.7, +10.4]` (`p_holm=0.8174`) | `+4.14 pp` `[-2.4, +10.6]` (`p_holm=0.4572`) |
| **H6:** Jina v3.5 vs Qwen3 (`08` vs `10`) | `+3.56 pp` `[-2.9, +10.0]` (`p_holm=0.4550`) | `+2.76 pp` `[-2.2, +7.7]` (`p_holm=0.8174`) | **`+8.62 pp`** `[+3.6, +13.9]` (`p_holm=0.0050`) |

### Key Takeaways
- **Generic family collapsing (`H1`) improves ranking cleanly while preserving cadence/grid/footprint variants:** Holm-significant across `Dev-45` (`+3.63 pp`, `p_holm = 0.0036`) and `Holdout A` (`+2.23 pp`, `p_holm = 0.0105`), and positive on `Blind Holdout B` (`+1.33 pp`, `p_holm = 0.0620`), while Audit 3 confirms **`0/236` (`0.0%`)** false cross-family collapses across all three benchmark suites.
- **Stage-1 hybrid fusion outperforms single retrievers across every suite:** Hybrid RRF + Collapse (`06`) reaches `FamR@50 = 92.22%` / `92.38%` / `98.47%` and `98.33%` `Hit@10` on Blind Holdout B (`663.44 ms` mean Stage-1 latency across all 140 queries; `8.27 ms` mean solo query encoding; `artifacts/v4_runtime_manifest.json`).
- **Listwise `Jina v3.5` delivers a large, statistically decisive Stage-2 lift (`H4`, `H6`):** On Blind Holdout B (`G = 60`), `Jina v3.5` raises `Fam-nDCG@10` by `+12.76 pp` (`74.19%` → `86.96%`, `p_holm = 0.0006`, `27W/27T/6L`) and `MRR@10` by `+15.53 pp` (`72.67%` → `88.19%`, `p_holm = 0.0006`), reaching `100.00%` `Hit@10` and `80.00%` `P@1` (`79.07%` to `97.19%` `Fam-nDCG@10` across all 10 EO domains; `artifacts/holdout_b_v4_by_domain.csv`). The fast Top-25 preset (`09`) reaches `86.51%` `Fam-nDCG@10` in `2,061.2 ms` (`~2.4×` faster) while trading `-3.73 pp` on Holdout A where multi-family candidates sit in ranks `26–50`.
- **Honest `Qwen3-0.6B` (`Apache-2.0`) trade-off:** `Qwen3-0.6B` reaches `100.00%` `Hit@10` on Blind Holdout B (`78.33%` `Fam-nDCG@10`, `+4.14 pp` over Stage-1, `p_holm = 0.4572`) and leads `Jina v3.5` in `Hydrology & Soils` (`84.36%` vs `79.58%`) and `Atmosphere & Air Quality` (`83.02%` vs `81.02%`), while trailing `Jina v3.5` overall by `-8.62 pp` (`p_holm = 0.0050`).
- **Unconfounded Top-50 blend vs pure ablation (`artifacts/v4_unconfounded_top50_blend_vs_pure.csv`):** At fixed `k=50`, blending Jina with Stage-1 RRF adds `+4.36 pp` on `Holdout A` (`p = 0.0409`), while differing by `-0.87 pp` on `Dev-45` (`p = 0.4951`) and `-0.42 pp` on `Blind Holdout B` (`p = 0.6699`).
- **Null-safe vs naive strict vs hard-limits (`artifacts/audit_dev45_filter_stage2.json`):** On the 13 grid-filtered `Dev-45` queries, `null_safe=True` preserves higher family recall (`FamR@50 = 90.38%` vs `83.33%`), whereas `null_safe=False` yields higher Top-10 ranking quality when targets have populated metadata (`Fam-nDCG@10`: `63.12%` vs `61.44%` after Jina reranking). Use `--hard-limits` whenever exact spatial footprint, strict pixel grid limits, and verified processing levels are mandatory over discovery recall.

---

## 5. Reproducible Verification Audits

Running [`11_reproducible_audits.py`](11_reproducible_audits.py) verifies all auxiliary engineering and statistical claims (`8/8` passed) and writes persistent JSON/CSV artifacts:

| Audit | Artifact | Verified Result (`v4.1-standalone`) |
| :--- | :--- | :--- |
| **0. BM25 Determinism & Alignment** | `artifacts/audit_bm25_determinism.json` | `0/45` NullSafe and `0/45` NaiveStrict Top-300 diffs (`0/90` checks); full two-build DuckDB rebuild produces byte-identical `bm25_content_sha256`; `verify_index_alignment()` rejects row-swapped inputs (`same_length_different_order_rejected = True`). |
| **1. Dense Encoder Exactness** | `artifacts/audit_mlx_bit_exactness.json` | Fast MLX path vs stock `mlx_vlm`: `max_abs_diff = 0.0` for both `L <= 512` (`4` docs) and `L > 513` sliding-window mask (`640`, `617` tokens); `batch=1` vs `batch=16` `min_cos = 0.999839`. |
| **2. Jina Causal Position Bias** | `artifacts/audit_jina_position_bias.json` | Top-50 forward vs 180° reversed (`N=15`): mean Spearman `rho = 0.8373` (`min = 0.6629`), Top-5 overlap `69.3%`, Top-1 exact agreement `60.0%`, recency boost `+0.01377`. |
| **3. Cross-Family False Collapse** | `artifacts/audit_false_collapse.json` | Exact `scan_limit = 200` denominator: Dev-45 **`0/98` (`0.0%`)**, Holdout A **`0/61` (`0.0%`)**, Blind Holdout B **`0/77` (`0.0%`)** (`0/236` across all 3 suites). |
| **4. Nested 6-Fold LOFO-CV** | `artifacts/dev45_v4_nested_lofo_cv.csv` | `0.000000 pp` drift vs script `10`; `near_dup_cos = 0.94` selected in `6/6` folds; primary `JinaV35_Top50_RRFBlend` selected in `4/6` folds (`Top-25` in `2/6`) with `+3.15` to `+17.10 pp` out-of-fold lift (`08b_JinaV35_Top50_Pure` selected in `6/6` folds when included in pool). |
| **5. Filter × Stage-2 Check** | `artifacts/audit_dev45_filter_stage2.json` | Quantifies `NullSafe` vs `NaiveStrict` after Jina & Qwen3 reranking across `all_45` and `grid_filtered_13`. |
| **6. CLI Purity & Latency** | `artifacts/audit_cli_latency.json` | `6/6` CLI probes produce valid JSON (`bm25-only`: `143.4 ms`; `stage-1`: `166.2–180.6 ms`; `jina-25`: `1,164.9 ms`; `jina-50`: `3,373.6 ms`; `qwen3-50`: `1,822.9 ms`). |
| **7. Cross-Backend Parity** | `artifacts/audit_backend_parity.json` | Apple Silicon MLX BF16 vs PyTorch CPU BF16 (incl. `>512`-token truncated doc): `EmbeddingGemma 2` **`min_cos = 0.999950`** (`mean = 0.999955`); `Jina v3.5` `max_diff = 0.00482`, **`exact_rank_agreement = True`**; `Qwen3-0.6B` `max_diff = 0.12500`, **`exact_rank_agreement = True`**. |

---

## 6. Hardware, Licensing & Limitations

- **Auto-selected backends (`--device auto`):** Uses `MLX` (`mlx-community/embeddinggemma-2-bf16`, `jinaai/jina-reranker-v3.5-mlx`, `Qwen/Qwen3-Reranker-0.6B`) on Apple Silicon macOS; automatically switches to standard PyTorch (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`) on CUDA GPUs and CPUs. CPU indexing of all `39,474` records is slower than GPU/MLX; use the prebuilt `v4.1.0-indexes` GitHub Release (downloaded automatically) or `--bm25-only` (`~36 s`) when rebuilding locally without an accelerator.
- **Model licenses:** `EmbeddingGemma 2` and `Qwen3-Reranker-0.6B` are `Apache-2.0`; `Jina Reranker v3.5` is `CC BY-NC 4.0` (non-commercial). All pipeline code in this directory is `MIT`; catalogue data and documentation are `CC-BY-4.0`.
- **Evaluation bounds:** Validation is **evaluated against curated family rules** without manual TREC pooling of every unjudged retrieved record; relevant datasets outside the pre-defined family rules receive zero credit, and strict compliance with hard spatial/grid/level constraints should use `--hard-limits`.

---

## 7. Reproducing the Full Validation Suite

```bash
# 1. Build local indexes (one-time; or allow auto-download of v4.1.0-indexes)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py

# 2. (Optional) Rebuild benchmark JSONs from canonical catalogue
uv run 06_semantic_hybrid_search/02_build_policy_benchmark.py
uv run 06_semantic_hybrid_search/08_build_holdout_a_external.py
uv run 06_semantic_hybrid_search/09_build_blind_holdout_b.py

# 3. Freeze Dev-45 config (10-entry SHA-256 lock), evaluate all 3 suites, and run all 8 audits
uv run 06_semantic_hybrid_search/07_deleak_and_lofo_dev.py
uv run 06_semantic_hybrid_search/10_run_blind_holdouts.py
uv run 06_semantic_hybrid_search/11_reproducible_audits.py
```

### Directory Layout

```text
06_semantic_hybrid_search/
├── README.md                          # Documentation, methodology, results & audits
├── eo_search_lib.py                   # Core library (record_id-keyed DuckDB BM25, release auto-download, RRF, collapser, hard-limits)
├── 04_search_cli.py                   # Standalone & --server client CLI (--hard-limits, --rebuild-index, --bm25-only, --json)
├── 04_search_server.py                # Local FastAPI server (/api/search, /api/export, /api/product) + web UI
├── web/
│   ├── index.html & app.js            # Tailwind + MapLibre GL UI with Verified hard-limits toggle & CSV/JSON/XLSX/Parquet export
├── 01_corpus_token_and_mlx_bench.py   # Corpus length profiling & MLX throughput benchmark
├── 02_build_policy_benchmark.py       # Dev-45 builder (45 queries, 101 expected families)
├── 03_index_and_eval_subset.py        # Early subset ablation script
├── 05_index_full_catalogue.py         # Full 39,474-record index builder + full_index_manifest.json (--device auto / --bm25-only)
├── 06_eval_rerankers.py               # Stage-2 reranker loaders (MLX & PyTorch) + reranker ablations
├── 07_deleak_and_lofo_dev.py          # Dev-45 collapser sweep, filter ablation & frozen_v4_config.json lock
├── 08_build_holdout_a_external.py     # Audited external domain benchmark builder (Holdout A, 35 queries)
├── 09_build_blind_holdout_b.py        # Blind benchmark builder (Holdout B, 60 queries, 80 families)
├── 10_run_blind_holdouts.py           # v4.1-standalone 3-suite evaluation & Holm-Bonferroni significance
├── 11_reproducible_audits.py          # 8 reproducible verification audits (incl. alignment, determinism & MLX/PyTorch parity)
└── artifacts/                         # Frozen SHA-256 locks, benchmark JSONs, CSV results & audit outputs
```
