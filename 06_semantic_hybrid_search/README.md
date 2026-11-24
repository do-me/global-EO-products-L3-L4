# Local Hybrid Semantic Search (`06_semantic_hybrid_search`)

Local hybrid search engine, CLI, web interface, and validation benchmark over all **39,474 canonical Level-3 and Level-4 Earth Observation (EO) catalogue records** from [`../03_harmonization/data/eo_l3_l4_canonical_products.parquet`](../03_harmonization/data/eo_l3_l4_canonical_products.parquet).

The search engine combines dense semantic embeddings (`EmbeddingGemma 2`), EO-normalized multi-field BM25 (`DuckDB FTS`), generic sensor-guarded family deduplication, and optional Stage-2 neural reranking (`Jina Reranker v3.5` or `Qwen3-Reranker-0.6B`). Backends are **device-agnostic (`--device auto`)**: scripts automatically select native **MLX** on Apple Silicon (`darwin/arm64`), **PyTorch CUDA** when an NVIDIA GPU is present, or **PyTorch CPU** otherwise, with an instant `--bm25-only` lexical mode requiring zero neural weights.

<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/d6dbff1f-5b93-4a4a-9d90-d3240c526510" />
<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/64c066dc-cb58-42ec-aceb-774c352c16cc" />
<img width="1963" height="1522" alt="image" src="https://github.com/user-attachments/assets/5a070fdf-1221-49f9-ac3f-e29d6f56baf7" />

---

## 1. Quick Start (`uv run`)

All scripts use PEP 723 inline dependencies and require only [`uv`](https://docs.astral.sh/uv/). Run commands from the repository root.

### 1.1 Build Local Indexes (One-Time)

Large rebuildable binary indexes (`full_bm25.duckdb`, `120 MB`; `full_embeddings_bf16.npz`, `54 MB`) are excluded from git and generated locally from the tracked canonical Parquet table:

```bash
# Full hybrid index (BM25 + 768-d dense vectors; auto-selects MLX / CUDA / CPU)
# ~8.5 min on Apple Silicon M3 Max (2.55 GB peak memory); ~35 s with --bm25-only
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py

# Fast lexical-only index on any OS (zero neural models, ~35 s)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py --bm25-only
```

### 1.2 Launch the Local Web UI & API Server

Starts a local FastAPI server (auto-selecting `MLX` / `CUDA` / `CPU`, or `--bm25-only`) and serves the Tailwind CSS + MapLibre GL search interface (`web/`). The interface supports all ranking modes, structured filters, family-collapsed sibling drawers, interactive bounding-box maps, and a single **Download** dropdown exporting results to **CSV**, **JSON**, **Excel (`.xlsx`)**, or **Parquet**:

```bash
# Auto-detect best backend and open http://127.0.0.1:8080
uv run 06_semantic_hybrid_search/04_search_server.py --port 8080 --open

# Optional overrides: force CPU/CUDA, preload a Stage-2 reranker, or run lexical-only
# (On Apple Silicon macOS, add `--with torch` if forcing PyTorch `--device cpu`)
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
| **Best Quality (Default)** | `--rerank jina-v3.5 --rerank-pool 50` | CC BY-NC 4.0 | ~5.3 s (`150 ms` S1 + `3.1–5.1 s` S2) | Highest ranking accuracy for research & non-commercial analysis |
| **Fast Rerank** | `--rerank jina-v3.5 --rerank-pool 25 --no-blend` | CC BY-NC 4.0 | ~1.9 s (`155 ms` S1 + `1.1–1.5 s` S2) | Interactive screening with near-identical Top-10 quality (~2.8× faster) |
| **Permissive Rerank** | `--rerank qwen3-0.6b --rerank-pool 50` | Apache-2.0 | ~4.7 s (`155 ms` S1 + `1.8–4.5 s` S2) | Commercial or permissive-license Stage-2 reranking |
| **Stage-1 Hybrid Only** | `--rerank none` | Apache-2.0 | ~150–485 ms | Instant interactive retrieval (`100%` Top-10 hit rate on blind holdout) |
| **BM25 Only** | `--bm25-only` | MIT | ~131 ms | Zero-model lexical search on any CPU |

Server API endpoints: `GET /api/status`, `POST /api/search` (and `GET /api/search`), `POST /api/export` (`csv`, `json`, `xlsx`, `parquet`), and `GET /api/product/{record_id}`.

---

## 2. Methodology

The pipeline is built around two practical realities of federated EO metadata catalogues:

1. **Upstream metadata is sparse:** `70.9%` of canonical records lack an adjudicated pixel grid value and `50.3%` lack structured platform metadata. Standard strict SQL filters silently drop relevant products whose metadata fields are `null`.
2. **Catalogue listings are heavily versioned and mirrored:** A single scientific product family (e.g., an ESA CCI climate record or NASA MODIS product) often appears as dozens of versioned releases, annual editions, or cross-portal mirrors that would otherwise crowd out distinct alternative datasets in the Top-10.

Every query runs through four deterministic stages implemented in [`eo_search_lib.py`](eo_search_lib.py) and [`06_eval_rerankers.py`](06_eval_rerankers.py):

### Stage 1 — Null-Safe Structured Pre-Filtering
Records flagged as superseded (`is_superseded_edition = True` or obsolete/deprecated title tags) are excluded by default, leaving `38,852` active records. Optional facets (`provider_pool`, `geo_scope`, `max_grid_m`, `strict_eo_only`) are applied as a boolean mask *before* scoring. Because upstream grid and coverage fields are frequently `null`, `null_safe=True` retains records whose requested attribute is missing while applying a `1.25×` tolerance on `max_grid_m` (e.g., allowing `300 m` Sentinel-3/OLCI products on a `250 m` request). Strict filtering (`--strict-filter` / `null_safe=False`) drops records with missing attributes when users prefer precision over recall.

### Stage 2 — Hybrid Dense + EO-Normalized BM25 Retrieval (RRF)
Two complementary rankers score all eligible records in parallel (Top-300 candidates each):

- **Dense bi-encoder (`EmbeddingGemma 2`, 768-d):** Each canonical record is serialized into a structured EO card (title, IDs, institution, domain, spatial/temporal metadata, deduplicated GCMD keywords, cleaned abstract; truncated at `512` tokens, no chunking). Queries use asymmetric query encoding (`batch_size=1`, `max_length=256`). Apple Silicon loads `mlx-community/embeddinggemma-2-bf16` via MLX; CUDA and CPU load `google/embeddinggemma-2` via PyTorch (`bfloat16` on Ampere+ CUDA and CPU, `float32` on older GPUs/MPS). Cosine ties break deterministically by ascending document index.
- **Multi-field BM25 (`DuckDB FTS`):** Indexes three normalized fields (`title` weight `2.5`, `keywords` `1.5`, `body` `1.0`, `k1=1.2`, `b=0.75`) using deterministic single-thread execution (`SET threads = 1`, `ROUND(score, 10)`) and generic EO-domain lexical normalization (UK/US spelling such as `colour`/`color`, `burnt`/`burned`, hyphen/underscore splits, and standard physical variable acronyms such as `CO2`, `CH4`, `LAI`, `SST`, `LST`, `DEM`—zero product or mission names).
- **Reciprocal Rank Fusion (RRF):** Combines dense and lexical rankings using $S_{\text{RRF}}(d) = \frac{1.0}{60 + r_{\text{dense}}(d)} + \frac{0.6}{60 + r_{\text{bm25}}(d)}$.

### Stage 3 — Generic Sensor-Guarded Family Collapsing
Scans up to `max(top_k * 3, 200)` fused candidates and groups versioned editions and unlinked portal mirrors under the highest-ranked representative without any benchmark-specific regexes. Two candidates collapse when either:
- They share the same generic normalized title stem (`gstem:`, stripping leading catalogue codes, parenthetical tags, versions, years, and resolution tokens), **or**
- Their 768-d dense cosine similarity reaches `near_dup_cos >= 0.94` (selected strictly on `Dev-45`).

A **physical-instrument sensor guard** extracts mission/sensor tokens (`title`, `raw_id`, `platform_sensor_raw`) and blocks collapsing whenever two records explicitly name disjoint physical instruments (e.g., preventing a `Landsat` product from absorbing a `Sentinel-2` or `MODIS` product). All collapsed siblings remain inspectable in `collapsed_siblings`.

### Stage 4 — Optional Stage-2 Neural Reranking
Rescores the Top-50 (or Top-25) collapsed family representatives using concise `<= 1,200`-character metadata cards:
- **`Jina Reranker v3.5`** (`jinaai/jina-reranker-v3.5-mlx` on Apple Silicon; `jinaai/jina-reranker-v3.5` on CUDA/CPU; `CC BY-NC 4.0`): listwise cross-attention reranker (`max_doc_length=420`), blended with Stage-1 RRF ranks via $S(d) = \frac{0.4}{30 + r_{\text{S1}}(d)} + \frac{1.0}{30 + r_{\text{S2}}(d)}$ (or pure reranker order for the fast Top-25 preset).
- **`Qwen3-Reranker-0.6B`** (`Qwen/Qwen3-Reranker-0.6B`; `Apache-2.0`): instruction-aware pairwise reranker scored via raw `yes`/`no` log-odds margin (`logit("yes") - logit("no")`) with deterministic stable sorting and the same RRF blend.

---

## 3. Validation Protocol & Benchmark Suites (`v4.0-standalone`)

Validation uses three strictly disjoint benchmark suites built exclusively from active canonical catalogue records (`38,852` records) and public EO domain/policy registries:

| Suite | Builder & Artifact | Size | Expected Families | Unique Active Records | Provenance Classification |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Dev-45** | [`02_build_policy_benchmark.py`](02_build_policy_benchmark.py) → `policy_retrieval_benchmark.json` | 45 queries | 101 | 1,353 | Development set (used to select `near_dup_cos = 0.94` and verify nested 6-fold LOFO-CV) |
| **Holdout A** | [`08_build_holdout_a_external.py`](08_build_holdout_a_external.py) → `holdout_a_external_benchmark.json` | 35 queries | 62 | 1,645 | Retrospective GT-audited external domains (GCOS ECV, USGEO EOA, NASA SNWG, WMO OSCAR; `0` overlap with Dev-45) |
| **Holdout B (Primary Blind)** | [`09_build_blind_holdout_b.py`](09_build_blind_holdout_b.py) → `holdout_b_blind_benchmark.json` | **60 queries** | **79** | **704** | **Prospective single-shot blind holdout** across 10 EO domains (`6` queries/domain, `G = G_eff = 60`, `0` overlap with Dev-45 or Holdout A) |

### Anti-Leakage & Prospective Integrity Controls
- **Zero cross-suite overlap:** `10_run_blind_holdouts.py` asserts programmatically that `Dev-45 ∩ Holdout A = ∅`, `Dev-45 ∩ Holdout B = ∅`, and `Holdout A ∩ Holdout B = ∅`, and `09_build_blind_holdout_b.py` asserts zero overlap across any pair of Holdout B queries.
- **Zero product/mission leakage in queries:** Automated anti-leakage linters block product brand names, mission codes, and provider shortnames in `Holdout A` and `Holdout B` query text.
- **5-artifact cryptographic freeze (`frozen_v4_config.json`):** `07_deleak_and_lofo_dev.py` runs strictly on `Dev-45` and records SHA-256 hashes of `eo_search_lib.py`, `06_eval_rerankers.py`, `full_catalogue.parquet`, `full_embeddings_bf16.npz`, and `full_bm25.duckdb` (`frozen_at_utc = 2026-11-24T06:52:50Z`). `10_run_blind_holdouts.py` verifies all 5 hashes before scoring `Holdout B` once (`completed_at_utc = 2026-11-24T07:26:34Z`; `strictly_prospective_timestamp_order = true`).
- **Metrics & inference:** The pre-registered primary metric is **Family-level nDCG@10 (`Fam-nDCG@10`)**, where each distinct expected product family yields gain only on its first retrieved member (preventing duplicate editions from inflating scores). Secondary pre-registered metrics are `MRR@10`, `FamR@10`, `Hit@5`, and `Hit@10`. Paired 10,000-resample bootstrap 95% CIs, exact sign-flip permutation p-values, step-down Holm-Bonferroni family-wise adjustment (`p_holm` over 6 hypotheses), and small-sample Student-$t$ MDE80 are persisted per suite.

---

## 4. Evaluation Results (`v4.0-standalone`)

All numbers below are taken verbatim from `artifacts/dev45_v4_summary.csv`, `artifacts/holdout_a_v4_summary.csv`, `artifacts/holdout_b_v4_summary.csv`, and their corresponding `*_v4_significance.csv` files (evaluated with solo `batch_size=1` query encoding on Apple Silicon MLX; cross-backend PyTorch parity is verified in Audit 7).

### 4.1 Main Retrieval & Reranking Comparison (`%`)

| System | Dev-45 `nDCG@10` | Dev-45 `Hit@10` | Holdout A `nDCG@10` | Holdout A `Hit@10` | **Blind Holdout B `nDCG@10`** | **Blind Holdout B `MRR@10`** | **Blind Holdout B `FamR@10`** | **Blind Holdout B `Hit@10`** | Mean Latency (HB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01_BM25_PreFilter_Raw` | 45.08 | 86.67 | 64.04 | 88.57 | 69.30 | 68.14 | 87.22 | 91.67 | 419.1 ms |
| `02_BM25_PreFilter_GenericCollapse` | 51.50 | 97.78 | 71.53 | 97.14 | 72.01 | 69.28 | 90.56 | 93.33 | 434.4 ms |
| `03_Dense_768d_Raw` | 44.89 | 86.67 | 56.59 | 88.57 | 67.21 | 65.57 | 83.06 | 88.33 | 363.9 ms |
| `04_Dense_768d_GenericCollapse` | 49.29 | 91.11 | 60.28 | 91.43 | 69.62 | 67.79 | 86.94 | 93.33 | 379.0 ms |
| `05_Stage1_HybridRRF_NoCollapse` | 48.71 | 86.67 | 66.23 | 94.29 | 72.52 | 70.57 | 91.11 | 96.67 | 432.0 ms |
| `06_Stage1_HybridRRF_GenericCollapse` | 55.87 | 95.56 | 72.95 | 97.14 | 74.70 | 72.30 | 94.44 | **100.00** | 447.3 ms |
| `07_Stage1_HybridRRF_NaiveStrict_Collapse` | 56.67 | 97.78 | 73.72 | 97.14 | 76.72 | 75.21 | 94.17 | 98.33 | 447.9 ms |
| **`08_Frozen_Best_JinaV35_Top50_RRFBlend`** | **65.88** | 95.56 | **77.21** | **100.00** | **86.52** | **86.92** | **96.11** | **100.00** | 5,300.4 ms |
| `08b_JinaV35_Top50_Pure` (fixed-k ablation) | 64.37 | 95.56 | 73.38 | 100.00 | 86.95 | 87.22 | 95.56 | 98.33 | 5,300.4 ms |
| `09_Frozen_Fast_JinaV35_Top25_Pure` | 64.77 | 95.56 | 70.69 | 100.00 | 86.82 | 87.02 | 96.11 | 100.00 | 1,915.1 ms |
| `10_Frozen_Apache2_Qwen3_Top50_RRFBlend` | 61.55 | 97.78 | 74.56 | 97.14 | 75.25 | 71.44 | 94.72 | 98.33 | 4,653.8 ms |

### 4.2 Pre-Registered Hypothesis Tests (`Fam-nDCG@10`, Holm-Bonferroni over 6 Comparisons)

| Hypothesis | Dev-45 (`G=45`) | Audited Holdout A (`G=35`) | **Prospective Blind Holdout B (`G=60`)** |
| :--- | :--- | :--- | :--- |
| **H1:** Collapser (`06` vs `05`) | **`+7.16 pp`** `[+3.8, +10.7]` (`p_holm=0.0012`) | **`+6.72 pp`** `[+3.4, +10.7]` (`p_holm=0.0012`) | **`+2.18 pp`** `[+0.8, +4.0]` (`p_holm=0.0056`) |
| **H2:** Hybrid vs BM25+Col (`06` vs `02`) | `+4.37 pp` `[-1.7, +10.2]` (`p_holm=0.3308`) | `+1.42 pp` `[-6.3, +9.4]` (`p_holm=1.0000`) | `+2.70 pp` `[-3.7, +8.8]` (`p_holm=0.8181`) |
| **H3:** Hybrid vs Dense+Col (`06` vs `04`) | `+6.58 pp` `[+1.3, +11.8]` (`p_holm=0.0824`) | **`+12.67 pp`** `[+6.2, +19.6]` (`p_holm=0.0040`) | `+5.08 pp` `[-1.1, +11.3]` (`p_holm=0.3615`) |
| **H4:** Jina v3.5 Blend vs S1 (`08` vs `06`) | **`+10.01 pp`** `[+4.7, +15.6]` (`p_holm=0.0030`) | `+4.26 pp` `[-0.1, +8.5]` (`p_holm=0.2600`) | **`+11.82 pp`** `[+6.8, +17.0]` (`p_holm=0.0006`) |
| **H5:** Qwen3-0.6B Blend vs S1 (`10` vs `06`) | `+5.68 pp` `[-0.6, +12.1]` (`p_holm=0.2385`) | `+1.61 pp` `[-5.1, +8.2]` (`p_holm=1.0000`) | `+0.55 pp` `[-6.1, +7.3]` (`p_holm=0.8734`) |
| **H6:** Jina v3.5 vs Qwen3 (`08` vs `10`) | `+4.34 pp` `[-1.9, +10.5]` (`p_holm=0.3308`) | `+2.65 pp` `[-1.7, +6.8]` (`p_holm=0.7031`) | **`+11.27 pp`** `[+5.5, +17.1]` (`p_holm=0.0006`) |

### Key Takeaways
- **Generic family collapsing (`H1`) helps on every suite with zero degradation on holdouts:** Holm-significant across `Dev-45` (`+7.16 pp`), `Holdout A` (`+6.72 pp`), and `Blind Holdout B` (`+2.18 pp`; `0` regressions on both `Holdout A` and `Holdout B`).
- **Stage-1 hybrid fusion reaches `100.0%` `Hit@10` on Blind Holdout B:** Hybrid RRF outperforms both BM25+Collapse and Dense+Collapse on `Fam-nDCG@10` across all three suites (`FamR@50 = 93.89%` / `97.14%` / `97.78%`), supplying a clean Top-50 candidate pool in `485.18 ms` mean end-to-end Stage-1 latency (`9.75 ms` mean solo query encoding across all 140 queries; `artifacts/v4_runtime_manifest.json`).
- **Listwise `Jina v3.5` delivers a large, statistically decisive Stage-2 lift (`H4`, `H6`):** On Blind Holdout B (`G = 60`), `Jina v3.5` raises `Fam-nDCG@10` by `+11.82 pp` (`74.70%` → `86.52%`, `p_holm = 0.0006`, `26W/28T/6L`) and `MRR@10` by `+14.63 pp` (`72.30%` → `86.92%`, `p_holm = 0.0006`), reaching `100%` `Hit@10` across all 10 EO domains (`77.91%` to `94.91%` `Fam-nDCG@10` by domain; `artifacts/holdout_b_v4_by_domain.csv`). The fast Top-25 preset (`09`) matches Top-50 Blend on Blind Holdout B (`86.82%` `Fam-nDCG@10` in `1.92 s`) while trading `-6.52 pp` on Holdout A where multi-family candidates sit deeper in ranks `26–50`.
- **Honest `Qwen3-0.6B` (`Apache-2.0`) trade-off:** `Qwen3-0.6B` provides solid gains on `Dev-45` (`+5.68 pp`, `FamR@10 = 80.22%`) and leads `Jina v3.5` in `Hydrology & Soils` (`90.24%` vs `77.95%`), but is flat overall on Blind Holdout B (`+0.55 pp`, `p = 0.8734`) and trails `Jina v3.5` by `-11.27 pp` (`p_holm = 0.0006`). For permissive-license deployments where `CC BY-NC 4.0` is unsuitable, Stage-1 Hybrid (`74.70%`, `Apache-2.0` + `MIT`, `~0.45 s`) is fast and already achieves `100%` `Hit@10`, with `Qwen3-0.6B` optional for recall-oriented workflows.
- **Unconfounded Top-50 blend vs pure ablation (`artifacts/v4_unconfounded_top50_blend_vs_pure.csv`):** At fixed `k=50`, blending Jina with Stage-1 RRF adds `+1.51 pp` on `Dev-45` (`p = 0.44`) and `+3.83 pp` on `Holdout A` (`p = 0.11`) while differing by `-0.43 pp` on `Blind Holdout B` (`p = 0.73`), acting as a modest variance stabilizer.
- **Null-safe vs strict filtering (`artifacts/dev45_filter_ablation_v4.csv`, `artifacts/audit_dev45_filter_stage2.json`):** On the 13 grid-filtered `Dev-45` queries, `null_safe=True` preserves higher family recall at Top-50 (`92.95%` vs `90.38%`), whereas `null_safe=False` yields higher Top-10 ranking quality (`Fam-nDCG@10`: `64.28%` vs `59.35%` after Jina reranking) when targets have populated grid metadata. `null_safe=True` remains the default to avoid silent recall loss on the `70.9%` of records missing grid values; `--strict-filter` exposes the stricter mode explicitly.

---

## 5. Reproducible Verification Audits

Running [`11_reproducible_audits.py`](11_reproducible_audits.py) verifies all auxiliary engineering and statistical claims and writes persistent JSON/CSV artifacts:

| Audit | Artifact | Verified Result (`v4.0-standalone`) |
| :--- | :--- | :--- |
| **0. BM25 Bit-Determinism** | `artifacts/audit_bm25_determinism.json` | `0/45` NullSafe and `0/45` NaiveStrict Top-300 diffs (`0/90` back-to-back checks). |
| **1. Dense Encoder Exactness** | `artifacts/audit_mlx_bit_exactness.json` | Fast MLX path vs stock `mlx_vlm`: `max_abs_diff = 0.0` for both `L <= 512` (`4` docs) and `L > 513` sliding-window mask (`640`, `617` tokens); `batch=1` vs `batch=16` `min_cos = 0.999839`. |
| **2. Jina Causal Position Bias** | `artifacts/audit_jina_position_bias.json` | Top-50 forward vs 180° reversed (`N=15`): mean Spearman `rho = 0.8825` (`min = 0.8134`), Top-5 overlap `72.0%`, Top-1 exact agreement `66.7%`, recency boost `+0.02507`. |
| **3. Cross-Family False Collapse** | `artifacts/audit_false_collapse.json` | Exact `scan_limit = 200` denominator: Dev-45 `1/98` (`1.02%`, one `gstem` collision in `Q26`), Holdout A **`0/61` (`0.0%`)**, Blind Holdout B **`0/75` (`0.0%`)** (`0/136` across both holdouts). |
| **4. Nested 6-Fold LOFO-CV** | `artifacts/dev45_v4_nested_lofo_cv.csv` | `0.000000 pp` drift vs script `10`; `near_dup_cos = 0.94` selected in `6/6` folds; primary `JinaV35_Top50_RRFBlend` selected in `6/6` folds (3-preset pool) with `+5.56` to `+17.82 pp` out-of-fold gains (`5/6` folds when adding ablation `08b`). |
| **5. Filter × Stage-2 Check** | `artifacts/audit_dev45_filter_stage2.json` | Quantifies `NullSafe` vs `NaiveStrict` after Jina & Qwen3 reranking across `all_45` and `grid_filtered_13`. |
| **6. CLI Purity & Latency** | `artifacts/audit_cli_latency.json` | `6/6` CLI probes produce valid JSON (`bm25-only`: `131.1 ms`; `stage-1`: `149.8–182.5 ms`; `jina-25`: `1,085.5 ms`; `jina-50`: `3,066.5 ms`; `qwen3-50`: `1,791.2 ms`). |
| **7. Cross-Backend Parity** | `artifacts/audit_backend_parity.json` | Apple Silicon MLX BF16 vs PyTorch CPU BF16 (incl. `>512`-token truncated doc): `EmbeddingGemma 2` **`min_cos = 0.999950`** (`mean = 0.999955`); `Jina v3.5` `max_diff = 0.00482`, **`exact_rank_agreement = True`**; `Qwen3-0.6B` `max_diff = 0.12500`, **`exact_rank_agreement = True`**. |

---

## 6. Hardware, Licensing & Limitations

- **Auto-selected backends (`--device auto`):** Uses `MLX` (`mlx-community/embeddinggemma-2-bf16`, `jinaai/jina-reranker-v3.5-mlx`, `Qwen/Qwen3-Reranker-0.6B`) on Apple Silicon macOS; automatically switches to standard PyTorch (`google/embeddinggemma-2`, `jinaai/jina-reranker-v3.5`, `Qwen/Qwen3-Reranker-0.6B`) on CUDA GPUs and CPUs. CPU indexing of all `39,474` records is slower than GPU/MLX; use `--bm25-only` (`~35 s`) when no accelerator is available.
- **Model licenses:** `EmbeddingGemma 2` and `Qwen3-Reranker-0.6B` are `Apache-2.0`; `Jina Reranker v3.5` is `CC BY-NC 4.0` (non-commercial). All pipeline code in this directory is `MIT`; catalogue data and documentation are `CC-BY-4.0`.
- **Evaluation bounds:** Validation evaluates discovery against regex-defined institutional product families without manual TREC pooling of unjudged retrieved records. Retrieved relevant alternatives outside the pre-defined family regexes are treated as non-relevant, so reported recall and `Fam-nDCG@10` represent conservative lower bounds.

---

## 7. Reproducing the Full Validation Suite

```bash
# 1. Build local indexes (one-time)
uv run 06_semantic_hybrid_search/05_index_full_catalogue.py

# 2. (Optional) Rebuild benchmark JSONs from canonical catalogue
uv run 06_semantic_hybrid_search/02_build_policy_benchmark.py
uv run 06_semantic_hybrid_search/08_build_holdout_a_external.py
uv run 06_semantic_hybrid_search/09_build_blind_holdout_b.py

# 3. Freeze Dev-45 config (5-artifact SHA-256 lock), evaluate all 3 suites, and run all 8 audits
uv run 06_semantic_hybrid_search/07_deleak_and_lofo_dev.py
uv run 06_semantic_hybrid_search/10_run_blind_holdouts.py
uv run 06_semantic_hybrid_search/11_reproducible_audits.py
```

### Directory Layout

```text
06_semantic_hybrid_search/
├── README.md                          # Documentation, methodology, results & audits
├── eo_search_lib.py                   # Core library (device-agnostic embedder, DuckDB BM25, RRF, collapser)
├── 04_search_cli.py                   # Standalone & --server client CLI (--device auto, --bm25-only, --json)
├── 04_search_server.py                # Local FastAPI server (/api/search, /api/export, /api/product) + web UI
├── web/
│   ├── index.html & app.js            # Tailwind + MapLibre GL UI with CSV/JSON/XLSX/Parquet download dropdown
├── 01_corpus_token_and_mlx_bench.py   # Corpus length profiling & MLX throughput benchmark
├── 02_build_policy_benchmark.py       # Dev-45 builder (45 queries, 101 expected families)
├── 03_index_and_eval_subset.py        # Early subset ablation script
├── 05_index_full_catalogue.py         # Full 39,474-record index builder (--device auto / --bm25-only)
├── 06_eval_rerankers.py               # Stage-2 reranker loaders (MLX & PyTorch) + reranker ablations
├── 07_deleak_and_lofo_dev.py          # Dev-45 collapser sweep, filter ablation & frozen_v4_config.json lock
├── 08_build_holdout_a_external.py     # Audited external domain benchmark builder (Holdout A, 35 queries)
├── 09_build_blind_holdout_b.py        # Prospective single-shot blind benchmark builder (Holdout B, 60 queries)
├── 10_run_blind_holdouts.py           # v4.0-standalone 3-suite evaluation & Holm-Bonferroni significance
├── 11_reproducible_audits.py          # 8 reproducible verification audits (incl. MLX vs PyTorch parity)
└── artifacts/                         # Frozen SHA-256 locks, benchmark JSONs, CSV results & audit outputs
```
