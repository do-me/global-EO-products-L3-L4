# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.2.0",
#   "mlx>=0.32.3",
#   "mlx-vlm>=0.8.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "transformers>=5.18.0",
# ]
# ///
"""Part (a) & (b): Full-corpus token audit, chunking analysis, and M3 Max MLX throughput benchmark.

Runs strictly locally inside 06_semantic_hybrid_search/. Does NOT run full 39,474-record
embedding inference; benchmarks a stratified 1,024-record sample matching the exact token
distribution of the 39,474 canonical catalogue and extrapolates exact wall-clock ETA.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CANONICAL_PARQUET = REPO_ROOT / "03_harmonization" / "data" / "eo_l3_l4_canonical_products.parquet"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_MODEL_8BIT,
    DEFAULT_MODEL_BF16,
    MLXEmbeddingGemma2,
    build_eo_document,
    build_eo_document_parts,
    build_raw_document,
    chunk_document_texts,
)


def get_hw_summary() -> dict[str, Any]:
    def _sysctl(key: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except Exception:
            return "unknown"

    dev = mx.device_info()
    return {
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "cpu_cores": _sysctl("hw.ncpu"),
        "ram_gb": round(int(_sysctl("hw.memsize")) / (1024**3), 1) if _sysctl("hw.memsize").isdigit() else "unknown",
        "mlx_version": mx.__version__,
        "mlx_device": {k: str(v) for k, v in dev.items()},
    }


def summarize_lengths(lengths: np.ndarray) -> dict[str, Any]:
    pcts = np.percentile(lengths, [10, 25, 50, 75, 90, 95, 99]).tolist()
    return {
        "count": int(len(lengths)),
        "total_tokens": int(lengths.sum()),
        "mean": round(float(lengths.mean()), 1),
        "min": int(lengths.min()),
        "p10": round(pcts[0], 1),
        "p25": round(pcts[1], 1),
        "p50": round(pcts[2], 1),
        "p75": round(pcts[3], 1),
        "p90": round(pcts[4], 1),
        "p95": round(pcts[5], 1),
        "p99": round(pcts[6], 1),
        "max": int(lengths.max()),
        "gt_256_n": int((lengths > 256).sum()),
        "gt_256_pct": round(100.0 * float((lengths > 256).mean()), 2),
        "gt_384_n": int((lengths > 384).sum()),
        "gt_384_pct": round(100.0 * float((lengths > 384).mean()), 2),
        "gt_512_n": int((lengths > 512).sum()),
        "gt_512_pct": round(100.0 * float((lengths > 512).mean()), 2),
        "gt_768_n": int((lengths > 768).sum()),
        "gt_768_pct": round(100.0 * float((lengths > 768).mean()), 2),
        "gt_1024_n": int((lengths > 1024).sum()),
        "gt_1024_pct": round(100.0 * float((lengths > 1024).mean()), 2),
    }


def stratified_sample_indices(df: pd.DataFrame, n_target: int = 1024, seed: int = 42) -> np.ndarray:
    """Stratify by source_catalogue so length & metadata distributions mirror the full 39,474 corpus."""
    rng = np.random.default_rng(seed)
    counts = df["source_catalogue"].value_counts()
    total = len(df)
    chosen: list[int] = []
    for src, cnt in counts.items():
        idxs = np.flatnonzero((df["source_catalogue"] == src).to_numpy())
        k = max(1, int(round(n_target * cnt / total)))
        k = min(k, len(idxs))
        chosen.extend(rng.choice(idxs, size=k, replace=False).tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) > n_target:
        chosen = rng.choice(chosen, size=n_target, replace=False).tolist()
    elif len(chosen) < n_target:
        remaining = np.setdiff1d(np.arange(total), np.array(chosen))
        extra = rng.choice(remaining, size=n_target - len(chosen), replace=False).tolist()
        chosen.extend(extra)
    arr = np.array(chosen, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def main() -> None:
    print("=" * 84)
    print("01. CORPUS TOKENIZATION AUDIT, CHUNKING ANALYSIS & APPLE SILICON MLX BENCHMARK")
    print("=" * 84)

    hw = get_hw_summary()
    print(f"Hardware: {hw['chip']} | RAM={hw['ram_gb']} GB | MLX={hw['mlx_version']}")

    t0 = time.perf_counter()
    df = pd.read_parquet(CANONICAL_PARQUET)
    n_total = len(df)
    print(f"Loaded canonical parquet: {n_total:,} records ({len(df.columns)} cols) in {time.perf_counter()-t0:.2f}s")

    # -----------------------------------------------------------------------
    # PART A: Full 39,474-record document building & exact tokenization audit
    # -----------------------------------------------------------------------
    print("\n--- PART A: Full 39,474-Record EO-Structured vs Raw Token Audit ---")
    t_build_0 = time.perf_counter()
    rows = df.to_dict(orient="records")
    eo_docs = [build_eo_document(r) for r in rows]
    t_build_eo = time.perf_counter() - t_build_0

    t_build_raw_0 = time.perf_counter()
    raw_docs = [build_raw_document(r) for r in rows]
    t_build_raw = time.perf_counter() - t_build_raw_0
    print(f"Built {n_total:,} EO-structured docs in {t_build_eo:.2f}s (raw docs in {t_build_raw:.2f}s)")

    print(f"Loading {DEFAULT_MODEL_BF16} on MLX...")
    enc_bf16 = MLXEmbeddingGemma2(DEFAULT_MODEL_BF16)
    print(f"Loaded BF16 model & tokenizer in {enc_bf16.load_seconds:.2f}s")

    # Tokenize full 39,474 corpus (uncapped up to 4096 to see real tail accurately)
    t_tok0 = time.perf_counter()
    eo_tok_ids = enc_bf16.tokenizer(
        eo_docs, padding=False, truncation=True, max_length=4096,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]
    t_tok_eo = time.perf_counter() - t_tok0
    eo_lens = np.fromiter((len(x) for x in eo_tok_ids), dtype=np.int32, count=n_total)

    t_tok1 = time.perf_counter()
    raw_tok_ids = enc_bf16.tokenizer(
        raw_docs, padding=False, truncation=True, max_length=4096,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]
    t_tok_raw = time.perf_counter() - t_tok1
    raw_lens = np.fromiter((len(x) for x in raw_tok_ids), dtype=np.int32, count=n_total)

    eo_summary = summarize_lengths(eo_lens)
    raw_summary = summarize_lengths(raw_lens)

    print(f"\nFull-corpus Rust tokenization time (39,474 docs): EO={t_tok_eo:.2f}s | Raw={t_tok_raw:.2f}s")
    print(f"EO-Structured tokens : p50={eo_summary['p50']}, p75={eo_summary['p75']}, p90={eo_summary['p90']}, "
          f"p95={eo_summary['p95']}, p99={eo_summary['p99']}, max={eo_summary['max']}, mean={eo_summary['mean']}")
    print(f"  >256: {eo_summary['gt_256_n']:,} ({eo_summary['gt_256_pct']}%) | "
          f">384: {eo_summary['gt_384_n']:,} ({eo_summary['gt_384_pct']}%) | "
          f">512: {eo_summary['gt_512_n']:,} ({eo_summary['gt_512_pct']}%) | "
          f">768: {eo_summary['gt_768_n']:,} ({eo_summary['gt_768_pct']}%)")
    print(f"Raw-Concat tokens    : p50={raw_summary['p50']}, p75={raw_summary['p75']}, p90={raw_summary['p90']}, "
          f"p95={raw_summary['p95']}, p99={raw_summary['p99']}, max={raw_summary['max']}, mean={raw_summary['mean']}")
    print(f"  >256: {raw_summary['gt_256_n']:,} ({raw_summary['gt_256_pct']}%) | "
          f">384: {raw_summary['gt_384_n']:,} ({raw_summary['gt_384_pct']}%) | "
          f">512: {raw_summary['gt_512_n']:,} ({raw_summary['gt_512_pct']}%) | "
          f">768: {raw_summary['gt_768_n']:,} ({raw_summary['gt_768_pct']}%)")

    # Compute exact chunk expansion ratio if we chunk docs > 512 (stride=384, max_chunks=4)
    long_idxs = np.flatnonzero(eo_lens > 512)
    extra_chunks = 0
    for idx in long_idxs:
        # Each doc > 512 adds ceil((len - 512) / 384) chunks capped at max_chunks-1
        rem = int(eo_lens[idx]) - 512
        extra_chunks += min(3, max(1, (rem + 383) // 384))
    chunk_multiplier = (n_total + extra_chunks) / n_total

    # Per-source >512 breakdown (top 10 sources with longest EO docs)
    df_len_check = pd.DataFrame({
        "source_catalogue": df["source_catalogue"],
        "eo_tokens": eo_lens,
        "raw_tokens": raw_lens,
    })
    src_stats = (
        df_len_check.groupby("source_catalogue")
        .agg(
            n=("eo_tokens", "count"),
            eo_p50=("eo_tokens", "median"),
            eo_p90=("eo_tokens", lambda s: float(np.percentile(s, 90))),
            gt_512=("eo_tokens", lambda s: int((s > 512).sum())),
            gt_512_pct=("eo_tokens", lambda s: round(100.0 * float((s > 512).mean()), 1)),
        )
        .sort_values("gt_512", ascending=False)
    )
    print("\nTop sources exceeding 512 tokens (EO-structured):")
    print(src_stats.head(10).to_string())

    # Inspect 5 real examples of what gets truncated past token 512
    long_examples = []
    if len(long_idxs) > 0:
        sample_long = long_idxs[np.linspace(0, len(long_idxs) - 1, min(6, len(long_idxs)), dtype=int)]
        for idx in sample_long:
            toks = eo_tok_ids[int(idx)]
            tail_txt = enc_bf16.tokenizer.decode(toks[512:620], skip_special_tokens=True)
            long_examples.append({
                "record_id": str(df.iloc[int(idx)]["record_id"]),
                "source_catalogue": str(df.iloc[int(idx)]["source_catalogue"]),
                "title": str(df.iloc[int(idx)]["title"])[:100],
                "eo_tokens": int(eo_lens[int(idx)]),
                "text_after_tok_512_preview": tail_txt[:260],
            })

    # -----------------------------------------------------------------------
    # PART B: Apple M3 Max MLX Throughput Benchmark (stratified 1,024 docs)
    # -----------------------------------------------------------------------
    print("\n--- PART B: Apple M3 Max MLX Throughput & Strategy Benchmark (N=1,024 stratified) ---")
    sample_idx = stratified_sample_indices(df, n_target=1024, seed=42)
    sample_texts = [eo_docs[int(i)] for i in sample_idx]
    sample_lens = eo_lens[sample_idx]
    print(
        f"Sample (N=1,024) token distribution: mean={sample_lens.mean():.1f} "
        f"(full={eo_lens.mean():.1f}), p50={np.median(sample_lens):.0f}, p90={np.percentile(sample_lens,90):.0f}, "
        f"p99={np.percentile(sample_lens,99):.0f}"
    )

    # Warmup BF16 Metal kernels cleanly (excluded from timed trials)
    _ = enc_bf16.encode_texts(sample_texts[:64], batch_size=32, max_length=512, sort_by_length=True, prefetch=True)
    mx.clear_cache()

    bench_rows: list[dict[str, Any]] = []

    def _run_case(
        label: str,
        encoder: MLXEmbeddingGemma2,
        model_tag: str,
        batch_size: int,
        max_length: int = 512,
        sort_by_length: bool = True,
        prefetch: bool = True,
        pad_to_multiple_of: int = 8,
        use_compile: bool = False,
        token_budget: int | None = None,
    ) -> np.ndarray:
        mx.reset_peak_memory()
        vecs, st = encoder.encode_texts(
            sample_texts,
            batch_size=batch_size,
            max_length=max_length,
            sort_by_length=sort_by_length,
            prefetch=prefetch,
            pad_to_multiple_of=pad_to_multiple_of,
            use_compile=use_compile,
            token_budget=token_budget,
            return_stats=True,
        )
        # Scale exact ETA using total capped real tokens of the full 39,474 corpus
        full_real_tokens_capped = int(np.minimum(eo_lens, max_length).sum())
        sample_real_tokens_capped = max(1, st["real_tokens"])
        token_scale = full_real_tokens_capped / sample_real_tokens_capped
        eta_gpu_s = st["gpu_seconds"] * token_scale
        eta_tok_s = t_tok_eo  # measured exact full-corpus Rust tokenization time
        eta_total_s = eta_gpu_s + eta_tok_s + t_build_eo

        row = {
            "label": label,
            "model": model_tag,
            "batch_size": batch_size,
            "token_budget": token_budget or 0,
            "max_length": max_length,
            "sort_by_length": sort_by_length,
            "prefetch": prefetch,
            "pad_mult": pad_to_multiple_of,
            "use_compile": use_compile,
            "n_batches": st["n_batches"],
            "pad_overhead_pct": st["padding_overhead_pct"],
            "tok_s": st["tok_seconds"],
            "gpu_s": st["gpu_seconds"],
            "total_s": st["total_seconds"],
            "docs_per_sec": st["docs_per_sec"],
            "real_tok_per_sec": st["real_tok_per_sec"],
            "peak_mem_gb": st["peak_mem_gb"],
            "eta_39474_gpu_s": round(eta_gpu_s, 1),
            "eta_39474_total_s": round(eta_total_s, 1),
            "eta_39474_min": round(eta_total_s / 60.0, 2),
        }
        bench_rows.append(row)
        print(
            f"  {label:38s} | {st['docs_per_sec']:6.1f} doc/s | {st['real_tok_per_sec']:8.0f} tok/s | "
            f"pad={st['padding_overhead_pct']:5.1f}% | mem={st['peak_mem_gb']:5.2f} GB | "
            f"ETA(39.5k)={row['eta_39474_total_s']:5.1f}s ({row['eta_39474_min']:.2f}m)"
        )
        return vecs

    # 1) BF16 Batch-size sweep (sorted + prefetched, max_length=512)
    print("\n[1] BF16 Batch-Size Sweep (sorted=True, prefetch=True, max_length=512):")
    ref_vecs_bf16 = None
    for bsz in (8, 16, 32, 64, 128, 256):
        v = _run_case(f"bf16_bs{bsz}_sorted_prefetch", enc_bf16, "bf16", batch_size=bsz, max_length=512)
        if bsz == 64:
            ref_vecs_bf16 = v

    # Pick winning fixed batch size from sweep
    fixed_sweep = [r for r in bench_rows if r["label"].startswith("bf16_bs")]
    best_fixed_bs = max(fixed_sweep, key=lambda r: r["docs_per_sec"])["batch_size"]
    print(f"  -> Best fixed BF16 batch_size so far: {best_fixed_bs}")

    # 2) Process & Strategy Ablation (unsorted vs sorted, prefetch, pad_mult, dynamic token-budget, compile)
    print(f"\n[2] Process & Batching Strategy Ablation (BF16, max_length=512, ref bs={best_fixed_bs}):")
    _run_case(
        f"bf16_bs{best_fixed_bs}_UNSORTED_noprefetch",
        enc_bf16, "bf16", batch_size=best_fixed_bs, max_length=512, sort_by_length=False, prefetch=False,
    )
    _run_case(
        f"bf16_bs{best_fixed_bs}_sorted_noprefetch",
        enc_bf16, "bf16", batch_size=best_fixed_bs, max_length=512, sort_by_length=True, prefetch=False,
    )
    _run_case(
        f"bf16_bs{best_fixed_bs}_sorted_pad16",
        enc_bf16, "bf16", batch_size=best_fixed_bs, max_length=512, sort_by_length=True, prefetch=True, pad_to_multiple_of=16,
    )
    # Dynamic token-budget batching (packs many short docs into big batches, fewer long docs into smaller batches)
    for tbud in (8192, 16384, 32768, 49152):
        _run_case(
            f"bf16_tokbudget_{tbud}_maxbs256",
            enc_bf16, "bf16", batch_size=256, max_length=512, sort_by_length=True, prefetch=True, token_budget=tbud,
        )
    # Test mx.compile on winning fixed batch size (warm up 1 small pass first? Note dynamic shapes recompile!)
    _run_case(
        f"bf16_bs{best_fixed_bs}_mx_compile",
        enc_bf16, "bf16", batch_size=best_fixed_bs, max_length=512, sort_by_length=True, prefetch=True, use_compile=True,
    )

    # Identify overall winning BF16 batching config at max_length=512
    bf16_512 = [r for r in bench_rows if r["model"] == "bf16" and r["max_length"] == 512]
    best_cfg = max(bf16_512, key=lambda r: r["docs_per_sec"])
    best_bs = int(best_cfg["batch_size"])
    best_tbud = int(best_cfg["token_budget"]) or None

    # 3) max_length sweep (256, 384, 512, 768)
    print(f"\n[3] Context Length Sweep (BF16, winning strategy={best_cfg['label']}):")
    for mlen in (256, 384, 768):
        _run_case(
            f"bf16_maxlen{mlen}_beststrat",
            enc_bf16, "bf16", batch_size=best_bs, max_length=mlen,
            sort_by_length=True, prefetch=True, token_budget=best_tbud,
        )

    # 4) Compare mlx-community/embeddinggemma-2-8bit vs BF16 (speed + cosine fidelity)
    print(f"\n[4] Quantization Check: {DEFAULT_MODEL_8BIT} vs BF16 (speed & exact cosine fidelity):")
    q8_fidelity: dict[str, Any] = {}
    try:
        enc_8bit = MLXEmbeddingGemma2(DEFAULT_MODEL_8BIT)
        _ = enc_8bit.encode_texts(sample_texts[:32], batch_size=32, max_length=512, sort_by_length=True, prefetch=True)
        mx.clear_cache()
        vecs_8bit = _run_case(
            f"8bit_beststrat_maxlen512",
            enc_8bit, "8bit", batch_size=best_bs, max_length=512,
            sort_by_length=True, prefetch=True, token_budget=best_tbud,
        )
        _run_case(
            f"8bit_bs{best_fixed_bs}_sorted_prefetch",
            enc_8bit, "8bit", batch_size=best_fixed_bs, max_length=512,
            sort_by_length=True, prefetch=True,
        )
        if ref_vecs_bf16 is not None:
            cos_sims = np.sum(ref_vecs_bf16 * vecs_8bit, axis=1)
            q8_fidelity = {
                "model_8bit": DEFAULT_MODEL_8BIT,
                "load_seconds": round(enc_8bit.load_seconds, 2),
                "mean_cosine_vs_bf16": round(float(cos_sims.mean()), 6),
                "min_cosine_vs_bf16": round(float(cos_sims.min()), 6),
                "p05_cosine_vs_bf16": round(float(np.percentile(cos_sims, 5)), 6),
                "p50_cosine_vs_bf16": round(float(np.median(cos_sims)), 6),
            }
            print(
                f"  8-bit vs BF16 fidelity (N=1,024 EO docs): mean_cos={q8_fidelity['mean_cosine_vs_bf16']:.6f}, "
                f"p05={q8_fidelity['p05_cosine_vs_bf16']:.6f}, min={q8_fidelity['min_cosine_vs_bf16']:.6f}"
            )
    except Exception as exc:
        print(f"  [WARN] 8-bit model benchmark skipped due to error: {exc}")
        q8_fidelity = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # PART C: Full-Corpus Storage & ETA Summary
    # -----------------------------------------------------------------------
    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv(ARTIFACTS_DIR / "mlx_throughput_benchmark.csv", index=False)

    best_bf16_512 = max(
        [r for r in bench_rows if r["model"] == "bf16" and r["max_length"] == 512],
        key=lambda r: r["docs_per_sec"],
    )
    unsorted_bf16_512 = [r for r in bench_rows if "UNSORTED" in r["label"]][0]
    speedup_vs_unsorted = round(best_bf16_512["docs_per_sec"] / max(1e-9, unsorted_bf16_512["docs_per_sec"]), 2)

    footprint_mb = {
        f"dim_{d}_{dt}": round(n_total * d * (2 if dt == "fp16" else 4) / (1024**2), 2)
        for d in (768, 512, 256, 128)
        for dt in ("fp16", "fp32")
    }

    report = {
        "hardware": hw,
        "corpus_records": n_total,
        "doc_build_seconds_full": round(t_build_eo, 2),
        "rust_tokenization_seconds_full": round(t_tok_eo, 2),
        "bf16_model_load_seconds": round(enc_bf16.load_seconds, 2),
        "token_audit_eo_structured": eo_summary,
        "token_audit_raw_concat": raw_summary,
        "chunking_analysis": {
            "docs_gt_512_eo": eo_summary["gt_512_n"],
            "docs_gt_512_eo_pct": eo_summary["gt_512_pct"],
            "docs_gt_512_raw": raw_summary["gt_512_n"],
            "docs_gt_512_raw_pct": raw_summary["gt_512_pct"],
            "extra_chunks_if_chunked_512_stride384": extra_chunks,
            "chunk_expansion_multiplier": round(chunk_multiplier, 4),
            "top_long_sources": src_stats.head(12).reset_index().to_dict(orient="records"),
            "long_tail_examples": long_examples,
        },
        "best_bf16_512_config": best_bf16_512,
        "unsorted_bf16_512_baseline": unsorted_bf16_512,
        "speedup_sorted_vs_unsorted": speedup_vs_unsorted,
        "quantization_8bit_vs_bf16": q8_fidelity,
        "full_corpus_index_footprint_mb": footprint_mb,
        "all_benchmarks": bench_rows,
    }

    out_json = ARTIFACTS_DIR / "mlx_throughput_benchmark.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print("SUMMARY — FULL 39,474-RECORD PROJECTION (BF16, max_length=512)")
    print("=" * 84)
    print(f"Winning strategy    : {best_bf16_512['label']}")
    print(f"Throughput          : {best_bf16_512['docs_per_sec']:.1f} docs/s | {best_bf16_512['real_tok_per_sec']:.0f} real tok/s")
    print(f"Speedup vs unsorted : {speedup_vs_unsorted}x (padding overhead: {unsorted_bf16_512['pad_overhead_pct']}% -> {best_bf16_512['pad_overhead_pct']}%)")
    print(f"Peak unified memory : {best_bf16_512['peak_mem_gb']:.2f} GB / {hw['ram_gb']} GB")
    print(f"Projected full ETA  : {best_bf16_512['eta_39474_total_s']:.1f} s ({best_bf16_512['eta_39474_min']:.2f} min) "
          f"[doc build {t_build_eo:.1f}s + tok {t_tok_eo:.1f}s + GPU {best_bf16_512['eta_39474_gpu_s']:.1f}s]")
    print(f"If chunked (512/384): +{extra_chunks:,} chunks ({chunk_multiplier:.3f}x corpus) -> "
          f"~{best_bf16_512['eta_39474_total_s'] * chunk_multiplier:.1f} s ({best_bf16_512['eta_39474_min'] * chunk_multiplier:.2f} min)")
    print(f"Vector index size   : 768d fp16 = {footprint_mb['dim_768_fp16']} MB | 256d fp16 = {footprint_mb['dim_256_fp16']} MB")
    print(f"Saved artifacts     : {out_json}")


if __name__ == "__main__":
    main()
