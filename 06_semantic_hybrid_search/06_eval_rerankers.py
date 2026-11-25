# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.5.0",
#   "huggingface_hub>=0.30.0",
#   "numpy>=1.26.0",
#   "pandas>=2.2.0",
#   "pyarrow>=18.0.0",
#   "safetensors>=0.4.0",
#   "tokenizers>=0.21.0",
#   "transformers>=5.18.0",
#   "mlx>=0.32.3; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-lm>=0.25.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mlx-vlm>=0.8.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "torch>=2.4.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
# ]
# ///
"""Device-agnostic Stage-2 rerankers (Apple Silicon MLX, NVIDIA CUDA, CPU PyTorch) + Stage-0 evaluation.

Supported rerankers (auto-selected by hardware via `load_reranker()`):
  1. Jina v3.5 (`jinaai/jina-reranker-v3.5-mlx` on Apple Silicon; `jinaai/jina-reranker-v3.5` on CUDA/CPU; CC BY-NC 4.0)
  2. Qwen3-Reranker-0.6B (`Qwen/Qwen3-Reranker-0.6B` on MLX, CUDA, and CPU; Apache-2.0)
  3. OpenSearch Neural Embed v1 (`opensearch-project/opensearch-neural-embed-v1`, MLX Stage-0 rescorer)
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import mlx.core as mx
except ImportError:  # Non-Apple-Silicon hosts run PyTorch CUDA/CPU
    mx = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

sys.path.insert(0, str(SCRIPT_DIR))
from eo_search_lib import (  # noqa: E402
    DEFAULT_MODEL_BF16,
    DuckDBLexicalIndex,
    MLXEmbeddingGemma2,
    SearchFilters,
    apply_filters_mask,
    build_eo_document_parts,
    collapse_product_families,
    deterministic_topk,
    resolve_device,
    rrf_fuse,
)

DEFAULT_MLX_JINA_MODEL = "jinaai/jina-reranker-v3.5-mlx"
DEFAULT_PT_JINA_MODEL = "jinaai/jina-reranker-v3.5"
DEFAULT_QWEN_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Import benchmark specs from 02
_b02_spec = importlib.util.spec_from_file_location("b02", SCRIPT_DIR / "02_build_policy_benchmark.py")
_b02 = importlib.util.module_from_spec(_b02_spec)
assert _b02_spec and _b02_spec.loader
_b02_spec.loader.exec_module(_b02)
BENCHMARK_SPECS = _b02.BENCHMARK_SPECS

EO_RERANK_INSTRUCTION = (
    "Given an abstract environmental policy, international treaty, or scientific Earth Observation "
    "monitoring requirement, judge whether the candidate satellite or geospatial data product "
    "directly provides the required biophysical variable, observation domain, and spatial scope."
)

OPENSEARCH_EO_QUERY_PROMPT = (
    "Instruct: Given an environmental policy or treaty Earth Observation requirement, "
    "retrieve relevant satellite or geospatial data products that satisfy the requirement\nQuery:"
)


def topk_from_scores(scores: np.ndarray, k: int = 300, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    return deterministic_topk(scores, k=k, mask=mask)


def eval_ranking_full(
    ranked_doc_indices: list[int],
    rid_by_idx: list[str],
    family_rid_sets: dict[str, set[str]],
) -> dict[str, Any]:
    rid_to_fams: dict[str, set[str]] = {}
    for fid, rset in family_rid_sets.items():
        for rid in rset:
            rid_to_fams.setdefault(rid, set()).add(fid)

    n_fams = max(1, len(family_rid_sets))
    first_rel_rank = 0
    seen_fams: set[str] = set()
    dcg10 = 0.0
    dcg20 = 0.0
    fam_counts_at = {5: 0, 10: 0, 20: 0}
    hit_at = {5: 0.0, 10: 0.0, 20: 0.0}

    for rank, doc_idx in enumerate(ranked_doc_indices[:25], start=1):
        fams = rid_to_fams.get(rid_by_idx[doc_idx])
        if fams:
            if first_rel_rank == 0:
                first_rel_rank = rank
            new_fams = fams - seen_fams
            if new_fams:
                seen_fams.update(new_fams)
                gain = float(len(new_fams))
                if rank <= 10:
                    dcg10 += gain / math.log2(rank + 1)
                if rank <= 20:
                    dcg20 += gain / math.log2(rank + 1)
        if rank in fam_counts_at:
            fam_counts_at[rank] = len(seen_fams)
        if rank in hit_at:
            hit_at[rank] = 1.0 if len(seen_fams) > 0 else 0.0

    for k in (5, 10, 20):
        if len(ranked_doc_indices) < k:
            fam_counts_at[k] = len(seen_fams)
            hit_at[k] = 1.0 if len(seen_fams) > 0 else 0.0

    idcg10 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 10) + 1))
    idcg20 = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_fams, 20) + 1))
    return {
        "hit_5": hit_at[5],
        "hit_10": hit_at[10],
        "hit_20": hit_at[20],
        "fam_r_5": fam_counts_at[5] / n_fams,
        "fam_r_10": fam_counts_at[10] / n_fams,
        "fam_r_20": fam_counts_at[20] / n_fams,
        "mrr_10": (1.0 / first_rel_rank) if (0 < first_rel_rank <= 10) else 0.0,
        "fam_ndcg_10": dcg10 / idcg10 if idcg10 > 0 else 0.0,
        "fam_ndcg_20": dcg20 / idcg20 if idcg20 > 0 else 0.0,
    }


def format_reranker_doc(row: dict[str, Any], max_chars: int = 1200) -> str:
    """Compact, high-signal EO passage string for Stage-2 rerankers/rescorers."""
    title, body = build_eo_document_parts(row)
    return f"{title}\n{body}"[:max_chars]


def blend_rrf_and_rerank(
    candidates: list[int],
    rerank_scores: list[float],
    rrf_k: int = 30,
    w_rrf: float = 0.4,
    w_rerank: float = 1.0,
) -> list[int]:
    """Reciprocal-rank blend of stage-1 RRF order (`candidates`) and stage-2 reranker order.

    Uses deterministic stable tie-breaking (`kind='stable'`) so equal reranker scores always
    preserve the Stage-1 RRF order rather than arbitrary quicksort permutations.
    """
    rerank_order = np.argsort(-np.asarray(rerank_scores, dtype=np.float64), kind="stable")
    rerank_rank = {int(pos): rk + 1 for rk, pos in enumerate(rerank_order)}
    scored = []
    for stage1_pos, doc_idx in enumerate(candidates):
        s = w_rrf / (rrf_k + stage1_pos + 1) + w_rerank / (rrf_k + rerank_rank[stage1_pos])
        scored.append((doc_idx, s, stage1_pos))
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [d for d, _, _ in scored]


# ---------------------------------------------------------------------------
# Model 1: jinaai/jina-reranker-v3.5-mlx (Listwise LBNL, native MLX)
# ---------------------------------------------------------------------------


class JinaV35MLXReranker:
    def __init__(
        self,
        model_id: str = DEFAULT_MLX_JINA_MODEL,
        max_doc_length: int = 420,
        revision: str | None = None,
    ):
        t0 = time.perf_counter()
        if Path(model_id).exists():
            self.repo_dir = Path(model_id)
        else:
            rev = revision if (revision and revision != "unresolved") else None
            self.repo_dir = Path(snapshot_download(model_id, revision=rev))
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        import rerank as jina_rerank  # type: ignore

        self.reranker = jina_rerank.MLXReranker(
            model_path=str(self.repo_dir),
            projector_path=str(self.repo_dir / "projector.safetensors"),
            block_size=125,
            max_length=32768,
            max_query_length=512,
            max_doc_length=max_doc_length,
        )
        self.load_seconds = time.perf_counter() - t0

    def score_candidates(self, query: str, docs: list[str]) -> list[float]:
        res = self.reranker.rerank(query=query, documents=docs)
        scores = [0.0] * len(docs)
        for item in res:
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores


# ---------------------------------------------------------------------------
# Model 2: Qwen/Qwen3-Reranker-0.6B (Pairwise instruction-aware, native MLX)
# ---------------------------------------------------------------------------


class Qwen3MLXReranker:
    """Native Apple Silicon MLX runner for Qwen/Qwen3-Reranker-0.6B via mlx-lm."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = EO_RERANK_INSTRUCTION,
        verify_exactness: bool = True,
        revision: str | None = None,
    ):
        from mlx_lm import load as mlx_lm_load

        t0 = time.perf_counter()
        self.model_id = model_id
        self.instruction = instruction
        load_src = model_id
        if not Path(model_id).exists() and revision and revision != "unresolved":
            load_src = str(Path(snapshot_download(model_id, revision=revision)))
        self.model, self.tokenizer = mlx_lm_load(load_src)
        self.model.eval()
        self.pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        self.yes_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.no_id = int(self.tokenizer.convert_tokens_to_ids("no"))

        prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.load_seconds = time.perf_counter() - t0
        if verify_exactness:
            self._verify_batch_exactness()

    def _encode_pair(self, query: str, doc: str, max_length: int = 512) -> list[int]:
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        budget = max(64, max_length - len(self.prefix_ids) - len(self.suffix_ids))
        body_ids = self.tokenizer.encode(body, add_special_tokens=False)[:budget]
        return self.prefix_ids + body_ids + self.suffix_ids

    def _score_batch_ids(self, seqs: list[list[int]]) -> list[float]:
        bsz = len(seqs)
        lens = [len(s) for s in seqs]
        max_l = max(lens)
        arr = np.full((bsz, max_l), self.pad_id, dtype=np.int32)
        for i, s in enumerate(seqs):
            arr[i, : len(s)] = s
        logits = self.model(mx.array(arr))
        row_idx = mx.arange(bsz)
        last_idx = mx.array([l - 1 for l in lens], dtype=mx.int32)
        last_logits = logits[row_idx, last_idx, :]
        yes_l = last_logits[:, self.yes_id].astype(mx.float32)
        no_l = last_logits[:, self.no_id].astype(mx.float32)
        # Use un-squashed log-odds margin (yes_l - no_l) to avoid sigmoid saturation at 1.0/0.0
        margins = yes_l - no_l
        mx.eval(margins)
        return [float(x) for x in np.asarray(margins, dtype=np.float32)]

    def _verify_batch_exactness(self) -> None:
        s1 = self._encode_pair("global forest cover", "Annual 30 m tree cover loss map.")
        s2 = self._encode_pair(
            "global forest cover",
            "Global ocean chlorophyll-a concentration derived from satellite ocean color sensors. " * 12,
        )
        solo_1 = self._score_batch_ids([s1])[0]
        batched_1 = self._score_batch_ids([s1, s2])[0]
        diff = abs(solo_1 - batched_1)
        assert diff < 0.05, f"Right-padding invariance check failed: solo={solo_1:.5f} vs batch={batched_1:.5f}"
        print(f"  [OK] Qwen3 right-padding causal invariance verified (log-odds margin): solo={solo_1:.5f}, batched={batched_1:.5f}, diff={diff:.6f}")

    def score_candidates(self, query: str, docs: list[str], batch_size: int = 8, max_length: int = 512) -> list[float]:
        seqs = [self._encode_pair(query, d, max_length=max_length) for d in docs]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
        out = [0.0] * len(docs)
        for i in range(0, len(order), batch_size):
            idxs = order[i : i + batch_size]
            batch_scores = self._score_batch_ids([seqs[k] for k in idxs])
            for k, sc in zip(idxs, batch_scores):
                out[k] = sc
        return out


# ---------------------------------------------------------------------------
# Device-Agnostic PyTorch Rerankers (NVIDIA CUDA / MPS / CPU) & Unified Loader
# ---------------------------------------------------------------------------


class JinaV35PyTorchReranker:
    """PyTorch/Transformers loader for `jinaai/jina-reranker-v3.5` (CUDA / MPS / CPU)."""

    def __init__(
        self,
        model_id: str = DEFAULT_PT_JINA_MODEL,
        device: str = "cpu",
        max_doc_length: int = 420,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.max_doc_length = max_doc_length
        self.torch = torch
        t0 = time.perf_counter()

        if device == "cuda":
            use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        elif device == "cpu":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0

    def score_candidates(self, query: str, docs: list[str], **_unused: Any) -> list[float]:
        if not docs:
            return []
        trunc_docs = []
        for d in docs:
            toks = self.tokenizer.encode(d, add_special_tokens=False)
            if len(toks) > self.max_doc_length:
                trunc_docs.append(self.tokenizer.decode(toks[: self.max_doc_length], skip_special_tokens=True))
            else:
                trunc_docs.append(d)

        with self.torch.inference_mode():
            ranked = self.model.rerank(query, trunc_docs)
        scores = [0.0] * len(docs)
        for item in ranked:
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores


class Qwen3PyTorchReranker:
    """PyTorch/Transformers loader for `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0, log-odds margin)."""

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_RERANK_MODEL,
        device: str = "cpu",
        instruction: str = EO_RERANK_INSTRUCTION,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.instruction = instruction
        self.torch = torch
        t0 = time.perf_counter()

        if device == "cuda":
            use_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        elif device == "cpu":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(device)
        self.model.eval()

        self.pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.tokenizer.eos_token_id or 0))
        self.yes_id = int(self.tokenizer.convert_tokens_to_ids("yes"))
        self.no_id = int(self.tokenizer.convert_tokens_to_ids("no"))
        self.prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.load_seconds = time.perf_counter() - t0

    def _encode_pair(self, query: str, doc: str, max_length: int = 512) -> list[int]:
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        budget = max(64, max_length - len(self.prefix_ids) - len(self.suffix_ids))
        body_ids = self.tokenizer.encode(body, add_special_tokens=False)[:budget]
        return self.prefix_ids + body_ids + self.suffix_ids

    def score_candidates(self, query: str, docs: list[str], batch_size: int = 8, max_length: int = 512) -> list[float]:
        if not docs:
            return []
        torch = self.torch
        seqs = [self._encode_pair(query, d, max_length=max_length) for d in docs]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
        out = [0.0] * len(docs)
        with torch.inference_mode():
            for i in range(0, len(order), batch_size):
                idxs = order[i : i + batch_size]
                batch_seqs = [seqs[k] for k in idxs]
                max_len = max(len(s) for s in batch_seqs)
                input_ids = torch.full((len(batch_seqs), max_len), self.pad_id, dtype=torch.long, device=self.device)
                attn_mask = torch.zeros((len(batch_seqs), max_len), dtype=torch.long, device=self.device)
                for row_i, s in enumerate(batch_seqs):
                    pad_left = max_len - len(s)
                    input_ids[row_i, pad_left:] = torch.tensor(s, dtype=torch.long, device=self.device)
                    attn_mask[row_i, pad_left:] = 1
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits[:, -1, :]
                margins = (logits[:, self.yes_id].float() - logits[:, self.no_id].float()).cpu().tolist()
                for k, sc in zip(idxs, margins):
                    out[k] = float(sc)
        return out


def clear_device_cache(device: str) -> None:
    """Release device memory cleanly across MLX, CUDA, MPS, or CPU."""
    gc.collect()
    if device == "mlx" and mx is not None:
        mx.clear_cache()
    elif device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif device == "mps":
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def load_reranker(
    reranker_name: str,
    device: str = "auto",
    max_doc_length: int = 420,
    verify_exactness: bool = False,
    model_id: str | None = None,
    revision: str | None = None,
) -> tuple[Any, str, str]:
    """Auto-select and load a Stage-2 reranker across MLX (Apple Silicon), CUDA, MPS, or CPU."""
    dev = resolve_device(device)
    if reranker_name == "jina-v3.5":
        if dev == "mlx":
            mid = model_id or DEFAULT_MLX_JINA_MODEL
            return JinaV35MLXReranker(model_id=mid, max_doc_length=max_doc_length, revision=revision), "mlx", mid
        mid = model_id or DEFAULT_PT_JINA_MODEL
        if not Path(mid).exists() and revision and revision != "unresolved":
            mid = str(Path(snapshot_download(mid, revision=revision)))
        return JinaV35PyTorchReranker(mid, device=dev, max_doc_length=max_doc_length), dev, mid
    if reranker_name == "qwen3-0.6b":
        mid = model_id or DEFAULT_QWEN_RERANK_MODEL
        if dev == "mlx":
            return Qwen3MLXReranker(mid, verify_exactness=verify_exactness, revision=revision), "mlx", mid
        if not Path(mid).exists() and revision and revision != "unresolved":
            mid = str(Path(snapshot_download(mid, revision=revision)))
        return Qwen3PyTorchReranker(mid, device=dev), dev, mid
    raise ValueError(f"Unsupported reranker '{reranker_name}'. Choose 'jina-v3.5' or 'qwen3-0.6b'.")


# ---------------------------------------------------------------------------
# Model 3: opensearch-project/opensearch-neural-embed-v1 (Nov 2026, 598M 2048d Dense Bi-Encoder!)
# Evaluated natively on MLX as a zero-reindexing Stage-2 bi-encoder rescorer
# ---------------------------------------------------------------------------


class OpenSearchEmbedV1MLXRescorer:
    """Native MLX loader for opensearch-project/opensearch-neural-embed-v1.

    Architecture: Qwen3Model (596M causal) -> Mean Pooling over real tokens ->
    Linear(1024 -> 2048, bias=True) from 2_Dense/model.safetensors -> L2 normalize.
    Uses BF16 weights on MLX for fast M3 Max inference, with right-padding causal invariance.
    """

    def __init__(self, repo_id: str = "opensearch-project/opensearch-neural-embed-v1", query_prompt: str = OPENSEARCH_EO_QUERY_PROMPT):
        from mlx_lm.models import qwen3
        from transformers import AutoTokenizer

        t0 = time.perf_counter()
        self.repo_id = repo_id
        self.query_prompt = query_prompt
        self.repo_dir = Path(snapshot_download(repo_id))

        cfg = json.loads((self.repo_dir / "config.json").read_text(encoding="utf-8"))
        args = qwen3.ModelArgs.from_dict(cfg)
        self.backbone = qwen3.Qwen3Model(args)

        # Load backbone weights (handles both "model.*" and unprefixed keys)
        raw_weights = mx.load(str(self.repo_dir / "model.safetensors"))
        cleaned = {}
        for k, v in raw_weights.items():
            nk = k.removeprefix("model.")
            cleaned[nk] = v.astype(mx.bfloat16)
        self.backbone.load_weights(list(cleaned.items()), strict=True)
        self.backbone.eval()

        # Load 2_Dense projection (1024 -> 2048)
        with safe_open(str(self.repo_dir / "2_Dense" / "model.safetensors"), framework="numpy") as f:
            self.dense_w = mx.array(f.get_tensor("linear.weight")).astype(mx.float32)  # (2048, 1024)
            self.dense_b = mx.array(f.get_tensor("linear.bias")).astype(mx.float32)    # (2048,)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.repo_dir))
        self.pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        self.load_seconds = time.perf_counter() - t0
        self._verify_batch_exactness()

    def _encode_batch_ids(self, seqs: list[list[int]]) -> np.ndarray:
        bsz = len(seqs)
        lens = [len(s) for s in seqs]
        max_l = max(lens)
        arr = np.full((bsz, max_l), self.pad_id, dtype=np.int32)
        mask = np.zeros((bsz, max_l), dtype=np.float32)
        for i, s in enumerate(seqs):
            arr[i, : len(s)] = s
            mask[i, : len(s)] = 1.0

        hidden = self.backbone(mx.array(arr)).astype(mx.float32)  # (B, L, 1024)
        m_mx = mx.array(mask)[..., None]                          # (B, L, 1)
        pooled = (hidden * m_mx).sum(axis=1) / mx.maximum(m_mx.sum(axis=1), 1.0)  # (B, 1024)
        proj = pooled @ self.dense_w.T + self.dense_b             # (B, 2048)
        norms = mx.sqrt(mx.maximum((proj * proj).sum(axis=-1, keepdims=True), 1e-12))
        emb = proj / norms
        mx.eval(emb)
        return np.asarray(emb, dtype=np.float32)

    def _verify_batch_exactness(self) -> None:
        s1 = self.tokenizer.encode("Annual 30 m tree cover loss map.", add_special_tokens=True)
        s2 = self.tokenizer.encode("Global ocean chlorophyll concentration map. " * 15, add_special_tokens=True)
        e_solo = self._encode_batch_ids([s1])[0]
        e_batch = self._encode_batch_ids([s1, s2])[0]
        cos = float(np.dot(e_solo, e_batch))
        assert cos > 0.9995, f"OpenSearch right-padding invariance failed: cos={cos:.6f}"
        print(f"  [OK] OpenSearch-Embed-v1 right-padding causal invariance verified: cos={cos:.6f}")

    def encode_texts(self, texts: list[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        seqs = [self.tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=max_length) for t in texts]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
        out = np.empty((len(texts), 2048), dtype=np.float32)
        for i in range(0, len(order), batch_size):
            idxs = order[i : i + batch_size]
            vecs = self._encode_batch_ids([seqs[k] for k in idxs])
            for pos, k in enumerate(idxs):
                out[k] = vecs[pos]
        return out

    def score_candidates(self, query: str, docs: list[str], batch_size: int = 8, max_length: int = 512) -> list[float]:
        q_vec = self.encode_texts([f"{self.query_prompt}{query}"], batch_size=1, max_length=256)[0]
        d_vecs = self.encode_texts(docs, batch_size=batch_size, max_length=max_length)
        scores = d_vecs @ q_vec
        return [float(x) for x in scores]


def main() -> None:
    print("=" * 92)
    print("06. FULL 39,474-CATALOGUE STAGE-2 RERANKER & RESCORER BENCHMARK (APPLE SILICON MLX)")
    print("=" * 92)

    full_pq = ARTIFACTS_DIR / "full_catalogue.parquet"
    full_emb = ARTIFACTS_DIR / "full_embeddings_bf16.npz"
    full_db = ARTIFACTS_DIR / "full_bm25.duckdb"
    for p in (full_pq, full_emb, full_db):
        if not p.exists():
            raise FileNotFoundError(f"Missing full index artifact: {p}")

    df = pd.read_parquet(full_pq)
    rid_by_idx = df["record_id"].tolist()
    z = np.load(full_emb, allow_pickle=True)
    vecs = z["embeddings_f16"].astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    family_sigs = z["family_sigs"].tolist()
    lex = DuckDBLexicalIndex(str(full_db), read_only=True)
    print(f"Loaded FULL index: {len(df):,} records | embeddings={vecs.shape}")

    obs_re = re.compile(r"\b(?:obsolete\s+release|superseded|deprecated)\b", re.IGNORECASE)
    active_mask = (~df["is_superseded_edition"].fillna(False).to_numpy(dtype=bool, copy=True)) & (
        ~df["title"].fillna("").str.contains(obs_re, regex=True).to_numpy(dtype=bool, copy=True)
    )
    combined = (
        df["title"].fillna("")
        + " || "
        + df["raw_id"].fillna("")
        + " || "
        + df["source_catalogue"].fillna("")
    )
    gt_by_q: list[dict[str, set[str]]] = []
    for sp in BENCHMARK_SPECS:
        fam_rids: dict[str, set[str]] = {}
        for fam in sp["expected_families"]:
            hit = active_mask & combined.str.contains(re.compile(fam["regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool, copy=True)
            if fam.get("exclude_regex"):
                hit = hit & ~combined.str.contains(re.compile(fam["exclude_regex"], re.IGNORECASE), regex=True).to_numpy(dtype=bool)
            idxs = np.flatnonzero(hit)
            if len(idxs) > 0:
                fam_rids[fam["family_id"]] = set(df.iloc[idxs]["record_id"].tolist())
        gt_by_q.append(fam_rids)

    # Stage 1: Hybrid RRF + Null-Safe Filter + Family Collapse to Top-50 distinct product families (solo batch_size=1!)
    enc = MLXEmbeddingGemma2(DEFAULT_MODEL_BF16)
    _ = enc.encode_queries(["warmup satellite query"], max_length=256)
    q_texts = [sp["query"] for sp in BENCHMARK_SPECS]
    q_vecs_list, q_enc_ms = [], []
    for qt in q_texts:
        t_q0 = time.perf_counter()
        q_vecs_list.append(enc.encode_queries([qt], max_length=256)[0])
        q_enc_ms.append((time.perf_counter() - t_q0) * 1000.0)
    q_vecs = np.vstack(q_vecs_list)
    del enc
    mx.clear_cache()

    POOL_K = 50
    stage1_top50: list[list[int]] = []
    stage1_docs_50: list[list[str]] = []
    stage1_ms: list[float] = []

    print(f"[Stage 1] Retrieving Top-{POOL_K} collapsed product families across all {len(df):,} records...")
    for qi, sp in enumerate(BENCHMARK_SPECS):
        qtxt = sp["query"]
        flt_spec = sp.get("filters", {})
        flt = SearchFilters(
            geo_scope=flt_spec.get("geo_scope"),
            max_grid_m=flt_spec.get("max_grid_m"),
            strict_eo_only=bool(flt_spec.get("strict_eo_only", False)),
            exclude_superseded=True,
            null_safe=True,
        )
        t0 = time.perf_counter()
        mask = apply_filters_mask(df, flt)
        sc = vecs @ q_vecs[qi]
        d_res = topk_from_scores(sc, k=300, mask=mask)
        b_res = lex.search_bm25(qtxt, top_k=300, eligible_mask=mask)
        fused = rrf_fuse(d_res, b_res, rrf_k=60, w_dense=1.0, w_bm25=0.6)
        collapsed = collapse_product_families(
            fused, df, top_k=POOL_K, family_sigs=family_sigs, vectors=vecs, near_dup_cos=0.94, sensor_guard=True
        )
        stage1_ms.append(q_enc_ms[qi] + (time.perf_counter() - t0) * 1000.0)

        c_idxs = [idx for idx, _, _ in collapsed]
        stage1_top50.append(c_idxs)
        stage1_docs_50.append([format_reranker_doc(df.iloc[idx].to_dict(), max_chars=1200) for idx in c_idxs])

    print(f"  Stage-1 mean latency: {np.mean(stage1_ms):.1f} ms")

    results_map: dict[str, list[list[int]]] = {
        "00_Baseline_HybridRRF_Collapse_NoRerank": stage1_top50,
    }
    rerank_lat_ms: dict[str, list[float]] = {
        "00_Baseline_HybridRRF_Collapse_NoRerank": [0.0] * len(BENCHMARK_SPECS),
    }

    # -----------------------------------------------------------------------
    # Model 1: jinaai/jina-reranker-v3.5-mlx (July 2026, 596M Listwise LBNL)
    # -----------------------------------------------------------------------
    print("\n[Stage 2A] jinaai/jina-reranker-v3.5-mlx (0.6B Listwise LBNL, Jul 2026, CC BY-NC 4.0)...")
    mx.reset_peak_memory()
    jina = JinaV35MLXReranker(max_doc_length=420)
    print(f"  Loaded in {jina.load_seconds:.2f}s")
    _ = jina.score_candidates(q_texts[0], stage1_docs_50[0][:10])

    jina_p50, jina_b50, jina_p25, jina_b25 = [], [], [], []
    jina_ms50, jina_ms25 = [], []
    for qi, qtxt in enumerate(q_texts):
        c50, d50 = stage1_top50[qi], stage1_docs_50[qi]
        t0 = time.perf_counter()
        sc50 = jina.score_candidates(qtxt, d50)
        jina_ms50.append((time.perf_counter() - t0) * 1000.0)
        jina_p50.append([c50[int(k)] for k in np.argsort(-np.asarray(sc50), kind="stable")])
        jina_b50.append(blend_rrf_and_rerank(c50, sc50, rrf_k=30, w_rrf=0.4, w_rerank=1.0))

        c25, d25 = c50[:25], d50[:25]
        t0 = time.perf_counter()
        sc25 = jina.score_candidates(qtxt, d25)
        jina_ms25.append((time.perf_counter() - t0) * 1000.0)
        jina_p25.append([c25[int(k)] for k in np.argsort(-np.asarray(sc25), kind="stable")])
        jina_b25.append(blend_rrf_and_rerank(c25, sc25, rrf_k=30, w_rrf=0.4, w_rerank=1.0))

    jina_peak = round(mx.get_peak_memory() / (1024**3), 2)
    print(f"  Done: Top-50={np.mean(jina_ms50):.1f} ms | Top-25={np.mean(jina_ms25):.1f} ms | peak={jina_peak:.2f} GB")
    results_map["01_JinaV35_Listwise_Pure_Top50"] = jina_p50
    results_map["02_JinaV35_Listwise_RRFBlend_Top50"] = jina_b50
    results_map["03_JinaV35_Listwise_Pure_Top25"] = jina_p25
    results_map["04_JinaV35_Listwise_RRFBlend_Top25"] = jina_b25
    rerank_lat_ms["01_JinaV35_Listwise_Pure_Top50"] = jina_ms50
    rerank_lat_ms["02_JinaV35_Listwise_RRFBlend_Top50"] = jina_ms50
    rerank_lat_ms["03_JinaV35_Listwise_Pure_Top25"] = jina_ms25
    rerank_lat_ms["04_JinaV35_Listwise_RRFBlend_Top25"] = jina_ms25
    del jina
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # Model 2: Qwen/Qwen3-Reranker-0.6B (June 2025, 596M Pairwise, Apache-2.0)
    # -----------------------------------------------------------------------
    print("\n[Stage 2B] Qwen/Qwen3-Reranker-0.6B (0.6B Instruction-Aware Pairwise, Jun 2025, Apache-2.0)...")
    mx.reset_peak_memory()
    qwen = Qwen3MLXReranker("Qwen/Qwen3-Reranker-0.6B", instruction=EO_RERANK_INSTRUCTION)
    print(f"  Loaded in {qwen.load_seconds:.2f}s")
    _ = qwen.score_candidates(q_texts[0], stage1_docs_50[0][:8], batch_size=8, max_length=512)

    qwen_p50, qwen_b50, qwen_p25, qwen_b25 = [], [], [], []
    qwen_ms50, qwen_ms25 = [], []
    for qi, qtxt in enumerate(q_texts):
        c50, d50 = stage1_top50[qi], stage1_docs_50[qi]
        t0 = time.perf_counter()
        sc50 = qwen.score_candidates(qtxt, d50, batch_size=8, max_length=512)
        dt50 = (time.perf_counter() - t0) * 1000.0
        qwen_ms50.append(dt50)
        qwen_p50.append([c50[int(k)] for k in np.argsort(-np.asarray(sc50), kind="stable")])
        qwen_b50.append(blend_rrf_and_rerank(c50, sc50, rrf_k=30, w_rrf=0.4, w_rerank=1.0))

        c25, sc25 = c50[:25], sc50[:25]
        t0 = time.perf_counter()
        _ = qwen.score_candidates(qtxt, d50[:25], batch_size=8, max_length=512)
        qwen_ms25.append((time.perf_counter() - t0) * 1000.0)
        qwen_p25.append([c25[int(k)] for k in np.argsort(-np.asarray(sc25), kind="stable")])
        qwen_b25.append(blend_rrf_and_rerank(c25, sc25, rrf_k=30, w_rrf=0.4, w_rerank=1.0))

    qwen_peak = round(mx.get_peak_memory() / (1024**3), 2)
    print(f"  Done: Top-50={np.mean(qwen_ms50):.1f} ms | Top-25={np.mean(qwen_ms25):.1f} ms | peak={qwen_peak:.2f} GB")
    results_map["05_Qwen3_0.6B_Instruct_Pure_Top50"] = qwen_p50
    results_map["06_Qwen3_0.6B_Instruct_RRFBlend_Top50"] = qwen_b50
    results_map["07_Qwen3_0.6B_Instruct_Pure_Top25"] = qwen_p25
    results_map["08_Qwen3_0.6B_Instruct_RRFBlend_Top25"] = qwen_b25
    rerank_lat_ms["05_Qwen3_0.6B_Instruct_Pure_Top50"] = qwen_ms50
    rerank_lat_ms["06_Qwen3_0.6B_Instruct_RRFBlend_Top50"] = qwen_ms50
    rerank_lat_ms["07_Qwen3_0.6B_Instruct_Pure_Top25"] = qwen_ms25
    rerank_lat_ms["08_Qwen3_0.6B_Instruct_RRFBlend_Top25"] = qwen_ms25
    del qwen
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # Model 3: opensearch-project/opensearch-neural-embed-v1 (Nov 2026, 598M 2048d Dense Bi-Encoder!)
    # -----------------------------------------------------------------------
    print("\n[Stage 2C] opensearch-project/opensearch-neural-embed-v1 (Nov 2026, 598M 2048d Dense Bi-Encoder, Apache-2.0)...")
    mx.reset_peak_memory()
    os_enc = OpenSearchEmbedV1MLXRescorer()
    print(f"  Loaded in {os_enc.load_seconds:.2f}s")
    _ = os_enc.score_candidates(q_texts[0], stage1_docs_50[0][:8], batch_size=8, max_length=512)

    os_p50, os_b50 = [], []
    os_ms50 = []
    for qi, qtxt in enumerate(q_texts):
        c50, d50 = stage1_top50[qi], stage1_docs_50[qi]
        t0 = time.perf_counter()
        sc50 = os_enc.score_candidates(qtxt, d50, batch_size=8, max_length=512)
        os_ms50.append((time.perf_counter() - t0) * 1000.0)
        os_p50.append([c50[int(k)] for k in np.argsort(-np.asarray(sc50), kind="stable")])
        os_b50.append(blend_rrf_and_rerank(c50, sc50, rrf_k=30, w_rrf=0.4, w_rerank=1.0))

    os_peak = round(mx.get_peak_memory() / (1024**3), 2)
    print(f"  Done: Top-50 rescore={np.mean(os_ms50):.1f} ms | peak={os_peak:.2f} GB")
    results_map["09_OpenSearchEmbedV1_BiEnc_Pure_Top50"] = os_p50
    results_map["10_OpenSearchEmbedV1_BiEnc_RRFBlend_Top50"] = os_b50
    rerank_lat_ms["09_OpenSearchEmbedV1_BiEnc_Pure_Top50"] = os_ms50
    rerank_lat_ms["10_OpenSearchEmbedV1_BiEnc_RRFBlend_Top50"] = os_ms50
    del os_enc
    mx.clear_cache()

    # -----------------------------------------------------------------------
    # Compute full metrics & Win / Tie / Loss vs Baseline
    # -----------------------------------------------------------------------
    pq_rows: list[dict[str, Any]] = []
    for sname, q_lists in results_map.items():
        for qi, sp in enumerate(BENCHMARK_SPECS):
            m = eval_ranking_full(q_lists[qi], rid_by_idx, gt_by_q[qi])
            pq_rows.append({"system": sname, "query_id": sp["query_id"], **m})

    df_pq = pd.DataFrame(pq_rows)
    df_pq.to_csv(ARTIFACTS_DIR / "reranker_eval_per_query.csv", index=False)

    mcols = ["hit_5", "hit_10", "fam_r_5", "fam_r_10", "fam_r_20", "mrr_10", "fam_ndcg_10", "fam_ndcg_20"]
    summary = df_pq.groupby("system", sort=True)[mcols].mean().reset_index()
    for c in mcols:
        summary[c] = (summary[c] * 100.0).round(2)
    summary["stage2_ms"] = summary["system"].map(lambda s: round(float(np.mean(rerank_lat_ms[s])), 1))
    summary["total_ms"] = summary["system"].map(
        lambda s: round(float(np.mean(stage1_ms) + np.mean(rerank_lat_ms[s])), 1)
    )

    base_ndcg = df_pq[df_pq["system"] == "00_Baseline_HybridRRF_Collapse_NoRerank"].set_index("query_id")["fam_ndcg_10"]
    wtl_col = []
    for sname in summary["system"]:
        if sname == "00_Baseline_HybridRRF_Collapse_NoRerank":
            wtl_col.append("-")
            continue
        cur = df_pq[df_pq["system"] == sname].set_index("query_id")["fam_ndcg_10"]
        diff = cur - base_ndcg
        wtl_col.append(f"{int((diff > 1e-4).sum())}W/{int((diff.abs() <= 1e-4).sum())}T/{int((diff < -1e-4).sum())}L")
    summary["wtl_ndcg10"] = wtl_col
    summary.to_csv(ARTIFACTS_DIR / "reranker_eval_summary.csv", index=False)

    best_sys = summary.iloc[1:].sort_values(["fam_ndcg_10", "fam_r_10", "mrr_10"], ascending=False).iloc[0]["system"]
    cur_b = df_pq[df_pq["system"] == best_sys].set_index("query_id")
    base_b = df_pq[df_pq["system"] == "00_Baseline_HybridRRF_Collapse_NoRerank"].set_index("query_id")
    deltas = (cur_b["fam_ndcg_10"] - base_b["fam_ndcg_10"]).sort_values(ascending=False)

    report = {
        "corpus_size": len(df),
        "n_queries": len(BENCHMARK_SPECS),
        "stage1_pool_k": POOL_K,
        "peak_mem_gb": {
            "jina_v35_mlx": jina_peak,
            "qwen3_reranker_06b": qwen_peak,
            "opensearch_embed_v1": os_peak,
        },
        "best_stage2_variant": best_sys,
        "top5_gains_ndcg10": {k: round(float(v) * 100.0, 2) for k, v in deltas.head(5).items()},
        "top5_drops_ndcg10": {k: round(float(v) * 100.0, 2) for k, v in deltas.tail(5).items()},
        "summary": summary.to_dict(orient="records"),
    }
    (ARTIFACTS_DIR / "reranker_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("FULL 39,474-CATALOGUE STAGE-2 RERANKER / RESCORER BENCHMARK — 45 POLICY QUERIES (%)")
    print("=" * 92)
    print(summary.to_string(index=False))
    print(f"\nBest Stage-2 variant: {best_sys}")
    print("  Top-5 nDCG@10 gains (pp):", report["top5_gains_ndcg10"])
    print("  Top-5 nDCG@10 drops (pp):", report["top5_drops_ndcg10"])


if __name__ == "__main__":
    main()
