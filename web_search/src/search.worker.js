import {
  AutoConfig,
  AutoModel,
  AutoModelForCausalLM,
  AutoTokenizer,
  Tensor,
  env,
} from '@huggingface/transformers';
import {
  EO_RERANK_INSTRUCTION,
  QWEN3_PROMPT_PREFIX,
  QWEN3_PROMPT_SUFFIX,
  applyFiltersMask,
  assembleEmbStore,
  blendRrfAndRerank,
  buildQueryPrompt,
  buildQwen3Body,
  collapseProductFamilies,
  decompressGzip,
  formatRerankerDoc,
  formatResolutionHuman,
  parseBm25Binary,
  parseEmbChunkBinary,
  rrfFuse,
  scoreDenseInt8,
  searchBm25,
} from './search-core.js';

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

const state = {
  baseUrl: '../',
  manifest: null,
  catalogueMeta: null,
  records: [],
  recordIds: [],
  recordIdToIdx: new Map(),
  searchMeta: null,
  bm25: null,
  embStore: null,
  embLoadingPromise: null,
  embedder: null,
  embedderTokenizer: null,
  embedderDevice: null,
  embedderLoadingPromise: null,
  queryVecCache: new Map(),
  reranker: null,
  rerankerTokenizer: null,
  rerankerYesId: null,
  rerankerNoId: null,
  rerankerPrefixIds: null,
  rerankerSuffixIds: null,
  rerankerDevice: null,
  rerankerLoadingPromise: null,
  detailChunkCache: new Map(),
  activeSearchSeq: 0,
  cancelRerankFlag: false,
  webgpuAvailable: false,
};

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function detectWebGpu() {
  try {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      return Boolean(adapter);
    }
  } catch {
    // ignore WebGPU probe errors
  }
  return false;
}

async function fetchJson(relPath) {
  const url = new URL(relPath, state.baseUrl).href;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${relPath}`);
  return resp.json();
}

async function fetchGzipBuffer(relPath, onProgress = null) {
  const url = new URL(relPath, state.baseUrl).href;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${relPath}`);
  if (!onProgress || !resp.body) {
    const rawGz = await resp.arrayBuffer();
    return decompressGzip(rawGz);
  }
  const total = Number(resp.headers.get('content-length') || 0);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (total > 0) onProgress(loaded, total);
  }
  const joined = new Uint8Array(loaded);
  let off = 0;
  for (const ch of chunks) {
    joined.set(ch, off);
    off += ch.byteLength;
  }
  return decompressGzip(joined.buffer);
}

async function disposeNested(obj) {
  if (!obj) return;
  if (typeof obj.dispose === 'function') {
    await obj.dispose();
    return;
  }
  if (typeof obj === 'object') {
    for (const val of Object.values(obj)) {
      if (!val) continue;
      if (typeof val.dispose === 'function') {
        await val.dispose();
      } else if (typeof val === 'object') {
        for (const sub of Object.values(val)) {
          if (sub && typeof sub.dispose === 'function') await sub.dispose();
        }
      }
    }
  }
}

function hydrateCatalogueRows(partRows, enums) {
  const decode = (col, idx) => (idx === null || idx === undefined || idx < 0 ? null : enums[col]?.[idx] ?? null);
  const out = new Array(partRows.length);
  for (let i = 0; i < partRows.length; i++) {
    const row = partRows[i];
    const src = decode('source_catalogue', row[7]) || '';
    const rawId = row[2] || '';
    const recordId = row[1] ? row[1] : `${src}::${rawId}`;
    out[i] = {
      idx: row[0],
      record_id: recordId,
      raw_id: rawId,
      title: row[3] || '',
      desc_snippet: row[4] || '',
      desc_truncated: Boolean(row[5]),
      kw_snippet: row[6] || '',
      source_catalogue: src,
      producing_institution: decode('producing_institution', row[8]),
      hosting_platform: decode('hosting_platform', row[9]),
      provider_pool: decode('provider_pool', row[10]),
      provider_jurisdiction_tier: decode('provider_jurisdiction_tier', row[11]),
      copernicus_service: decode('copernicus_service', row[12]),
      processing_level: decode('processing_level', row[13]),
      product_ontology_class: decode('product_ontology_class', row[14]),
      product_output_category: decode('product_output_category', row[15]),
      strict_eo: Boolean(row[16]),
      uses_sentinel: Boolean(row[17]),
      sentinel_missions: decode('sentinel_missions', row[18]),
      sentinel_lineage_status: decode('sentinel_lineage_status', row[19]),
      grid_m: row[20],
      temporal_cadence_class: decode('temporal_cadence_class', row[21]),
      t0: row[22],
      t1: row[23],
      is_ongoing: row[24],
      operational_maturity_class: decode('operational_maturity_class', row[25]),
      geographic_coverage_class: decode('geographic_coverage_class', row[26]),
      observation_domain_class: decode('observation_domain_class', row[27]),
      observation_domain_mask: decode('observation_domain_mask', row[28]),
      doi: row[29],
      license: decode('license', row[30]),
      access_url: row[31],
      bbox: row[32],
      antimeridian: Boolean(row[33]),
      cluster_count: row[34] || 1,
    };
  }
  return out;
}

async function initEngine(baseUrl) {
  state.baseUrl = baseUrl || '../';
  state.webgpuAvailable = await detectWebGpu();
  post('initProgress', { pct: 8, message: 'Loading search manifest & catalogue metadata…' });

  const [manifest, catMeta] = await Promise.all([
    fetchJson('data/search/search_manifest.json'),
    fetchJson('data/index_meta.json'),
  ]);
  state.manifest = manifest;
  state.catalogueMeta = catMeta;

  post('initProgress', { pct: 24, message: 'Fetching catalogue shards, BM25 index & filter metadata…' });

  const partPromises = catMeta.meta.index_parts.map((part) => fetchJson(`data/${part}`));
  const bm25Promise = fetchGzipBuffer(`data/search/${manifest.bm25.file}`);
  const metaPromise = fetchGzipBuffer(`data/search/${manifest.search_meta.file}`);

  const [parts, bm25Raw, metaRaw] = await Promise.all([
    Promise.all(partPromises),
    bm25Promise,
    metaPromise,
  ]);

  post('initProgress', { pct: 76, message: 'Hydrating 39,472 products & verifying index alignment…' });

  const allRecords = [];
  for (const part of parts) {
    const hydrated = hydrateCatalogueRows(part, catMeta.enums);
    for (let i = 0; i < hydrated.length; i++) allRecords.push(hydrated[i]);
  }
  state.records = allRecords;
  state.recordIds = allRecords.map((r) => r.record_id);
  state.recordIdToIdx = new Map();
  for (let i = 0; i < allRecords.length; i++) {
    state.recordIdToIdx.set(allRecords[i].record_id, i);
  }

  state.bm25 = parseBm25Binary(bm25Raw);
  state.searchMeta = JSON.parse(new TextDecoder('utf-8').decode(metaRaw));

  if (
    state.records.length !== state.bm25.nDocs ||
    state.records.length !== state.searchMeta.n_docs ||
    state.records.length !== manifest.n_docs
  ) {
    throw new Error(
      `Index length mismatch: catalogue=${state.records.length}, bm25=${state.bm25.nDocs}, meta=${state.searchMeta.n_docs}, manifest=${manifest.n_docs}`,
    );
  }
  if (
    manifest.alignment?.record_ids_sha256 &&
    state.searchMeta.record_ids_sha256 &&
    manifest.alignment.record_ids_sha256 !== state.searchMeta.record_ids_sha256
  ) {
    throw new Error('Search index record_ids_sha256 mismatch between manifest and search_meta.');
  }

  post('initProgress', { pct: 100, message: 'Ready' });
  post('ready', {
    webgpuAvailable: state.webgpuAvailable,
    enums: {
      source_catalogue: catMeta.enums.source_catalogue || [],
      provider_jurisdiction_tier: catMeta.enums.provider_jurisdiction_tier || [],
      product_output_category: catMeta.enums.product_output_category || [],
      temporal_cadence_class: catMeta.enums.temporal_cadence_class || [],
      operational_maturity_class: catMeta.enums.operational_maturity_class || [],
      product_ontology_class: catMeta.enums.product_ontology_class || [],
    },
    manifest: {
      release_tag: manifest.release_tag,
      n_docs: manifest.n_docs,
      active_docs: manifest.active_docs,
      deprecated_docs: manifest.n_docs - manifest.active_docs,
      emb_dim: manifest.emb_dim,
      models: manifest.models,
      bm25_gz_bytes: manifest.bm25.gz_bytes,
      emb_gz_bytes: manifest.embeddings.reduce((a, b) => a + b.gz_bytes, 0),
      pool_counts: manifest.pool_counts,
      source_counts: manifest.source_counts,
      unknown_metadata_counts: manifest.unknown_metadata_counts,
    },
  });
}

async function ensureEmbeddingsLoaded() {
  if (state.embStore) return state.embStore;
  if (state.embLoadingPromise) return state.embLoadingPromise;

  state.embLoadingPromise = (async () => {
    const parts = state.manifest.embeddings;
    const totalBytes = parts.reduce((acc, p) => acc + p.gz_bytes, 0);
    const partProgress = new Array(parts.length).fill(0);

    post('modelProgress', {
      target: 'semantic',
      stage: 'vectors',
      file: `${state.manifest.emb_dim}-d document vectors`,
      pct: 2,
      message: `Downloading ${state.manifest.emb_dim}-d vector index (${(totalBytes / 1e6).toFixed(1)} MB)…`,
    });

    const parsedChunks = await Promise.all(
      parts.map(async (p, idx) => {
        const buf = await fetchGzipBuffer(`data/search/${p.file}`, (loaded) => {
          partProgress[idx] = loaded;
          const sum = partProgress.reduce((a, b) => a + b, 0);
          const pct = Math.min(99, Math.round((sum / Math.max(1, totalBytes)) * 100));
          post('modelProgress', {
            target: 'semantic',
            stage: 'vectors',
            file: p.file,
            pct,
            message: `Downloading ${state.manifest.emb_dim}-d vectors (${(sum / 1e6).toFixed(1)} / ${(totalBytes / 1e6).toFixed(1)} MB)…`,
          });
        });
        return parseEmbChunkBinary(buf);
      }),
    );

    state.embStore = assembleEmbStore(parsedChunks, state.manifest.n_docs, state.manifest.emb_dim);
    return state.embStore;
  })();

  try {
    return await state.embLoadingPromise;
  } catch (err) {
    state.embLoadingPromise = null;
    throw err;
  }
}

async function ensureEmbedderLoaded() {
  if (state.embedder && state.embedderTokenizer) {
    return { model: state.embedder, tokenizer: state.embedderTokenizer, device: state.embedderDevice };
  }
  if (state.embedderLoadingPromise) return state.embedderLoadingPromise;

  state.embedderLoadingPromise = (async () => {
    const modelId = state.manifest.models.embedder;
    const dtype = state.manifest.models.embedder_dtype || 'q4';
    const fileTracker = new Map();

    const progressCallback = (ev) => {
      if (!ev || !ev.file) return;
      if (ev.status === 'progress' || ev.status === 'done' || ev.status === 'initiate') {
        fileTracker.set(ev.file, {
          loaded: ev.loaded || 0,
          total: ev.total || 0,
          progress: ev.progress || 0,
          done: ev.status === 'done',
        });
      }
      let sumLoaded = 0;
      let sumTotal = 0;
      for (const info of fileTracker.values()) {
        if (info.total > 0) {
          sumLoaded += info.done ? info.total : info.loaded;
          sumTotal += info.total;
        }
      }
      const pct = sumTotal > 0 ? Math.min(99, Math.round((sumLoaded / sumTotal) * 100)) : Math.round(ev.progress || 0);
      const mbStr = sumTotal > 0 ? ` (${(sumLoaded / 1e6).toFixed(1)} / ${(sumTotal / 1e6).toFixed(1)} MB)` : '';
      post('modelProgress', {
        target: 'semantic',
        stage: 'model',
        file: ev.file,
        pct,
        message: `Loading EmbeddingGemma 2 (text-only ${dtype})${mbStr}…`,
      });
    };

    const [, config, tokenizer] = await Promise.all([
      ensureEmbeddingsLoaded(),
      AutoConfig.from_pretrained(modelId),
      AutoTokenizer.from_pretrained(modelId, { progress_callback: progressCallback }),
    ]);

    config.vision_config = null;
    config.audio_config = null;

    const preferredOrder = state.webgpuAvailable ? ['webgpu', 'wasm'] : ['wasm'];
    let model = null;
    let chosenDevice = null;
    let lastErr = null;

    for (const dev of preferredOrder) {
      try {
        post('modelProgress', {
          target: 'semantic',
          stage: 'compile',
          pct: 96,
          message: `Initializing EmbeddingGemma 2 on ${dev.toUpperCase()}…`,
        });
        model = await AutoModel.from_pretrained(modelId, {
          config,
          dtype,
          device: dev,
          progress_callback: progressCallback,
        });
        chosenDevice = dev;
        break;
      } catch (err) {
        lastErr = err;
        console.warn(`[search-worker] EmbeddingGemma 2 failed on ${dev}, trying fallback:`, err);
      }
    }
    if (!model) {
      throw lastErr || new Error('Failed to load EmbeddingGemma 2 ONNX model.');
    }

    state.embedder = model;
    state.embedderTokenizer = tokenizer;
    state.embedderDevice = chosenDevice;

    post('semanticReady', {
      device: chosenDevice,
      dim: state.manifest.emb_dim,
      modelId,
      dtype,
    });

    return { model, tokenizer, device: chosenDevice };
  })();

  try {
    return await state.embedderLoadingPromise;
  } catch (err) {
    state.embedderLoadingPromise = null;
    throw err;
  }
}

async function encodeQuery(query) {
  const prompt = buildQueryPrompt(query);
  if (state.queryVecCache.has(prompt)) {
    return state.queryVecCache.get(prompt);
  }
  const { model, tokenizer } = await ensureEmbedderLoaded();
  let inputs = null;
  let outputs = null;
  try {
    inputs = await tokenizer([prompt], { padding: true, truncation: true, max_length: 256 });
    outputs = await model(inputs);
    const embTensor = outputs.sentence_embedding || outputs.pooler_output || outputs.last_hidden_state;
    if (!embTensor || !embTensor.data || embTensor.data.length < state.manifest.emb_dim) {
      throw new Error('EmbeddingGemma 2 returned empty or unexpected embedding tensor.');
    }
    const vec = new Float32Array(embTensor.data);
    if (state.queryVecCache.size > 64) {
      const firstKey = state.queryVecCache.keys().next().value;
      state.queryVecCache.delete(firstKey);
    }
    state.queryVecCache.set(prompt, vec);
    return vec;
  } finally {
    await disposeNested(outputs);
    await disposeNested(inputs);
  }
}

async function ensureRerankerLoaded() {
  if (state.reranker && state.rerankerTokenizer) {
    return { model: state.reranker, tokenizer: state.rerankerTokenizer, device: state.rerankerDevice };
  }
  if (state.rerankerLoadingPromise) return state.rerankerLoadingPromise;

  state.rerankerLoadingPromise = (async () => {
    const modelId = state.manifest.models.reranker;
    const dtype = state.manifest.models.reranker_dtype || 'q4';
    const fileTracker = new Map();

    const progressCallback = (ev) => {
      if (!ev || !ev.file) return;
      if (ev.status === 'progress' || ev.status === 'done' || ev.status === 'initiate') {
        fileTracker.set(ev.file, {
          loaded: ev.loaded || 0,
          total: ev.total || 0,
          progress: ev.progress || 0,
          done: ev.status === 'done',
        });
      }
      let sumLoaded = 0;
      let sumTotal = 0;
      for (const info of fileTracker.values()) {
        if (info.total > 0) {
          sumLoaded += info.done ? info.total : info.loaded;
          sumTotal += info.total;
        }
      }
      const pct = sumTotal > 0 ? Math.min(99, Math.round((sumLoaded / sumTotal) * 100)) : Math.round(ev.progress || 0);
      const mbStr = sumTotal > 0 ? ` (${(sumLoaded / 1e6).toFixed(0)} / ${(sumTotal / 1e6).toFixed(0)} MB)` : '';
      post('modelProgress', {
        target: 'reranker',
        stage: 'model',
        file: ev.file,
        pct,
        message: `Loading Qwen3-Reranker-0.6B (${dtype})${mbStr}…`,
      });
    };

    const tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback: progressCallback });
    const preferredOrder = state.webgpuAvailable ? ['webgpu', 'wasm'] : ['wasm'];
    let model = null;
    let chosenDevice = null;
    let lastErr = null;

    for (const dev of preferredOrder) {
      try {
        post('modelProgress', {
          target: 'reranker',
          stage: 'compile',
          pct: 97,
          message: `Initializing Qwen3-Reranker-0.6B on ${dev.toUpperCase()}…`,
        });
        model = await AutoModelForCausalLM.from_pretrained(modelId, {
          dtype,
          device: dev,
          progress_callback: progressCallback,
        });
        chosenDevice = dev;
        break;
      } catch (err) {
        lastErr = err;
        console.warn(`[search-worker] Qwen3-Reranker failed on ${dev}, trying fallback:`, err);
      }
    }
    if (!model) {
      throw lastErr || new Error('Failed to load Qwen3-Reranker-0.6B ONNX model.');
    }

    const yesTok = typeof tokenizer.convert_tokens_to_ids === 'function'
      ? tokenizer.convert_tokens_to_ids('yes')
      : tokenizer.encode('yes', { add_special_tokens: false })[0];
    const noTok = typeof tokenizer.convert_tokens_to_ids === 'function'
      ? tokenizer.convert_tokens_to_ids('no')
      : tokenizer.encode('no', { add_special_tokens: false })[0];
    const yesId = Number(Array.isArray(yesTok) ? yesTok[0] : yesTok);
    const noId = Number(Array.isArray(noTok) ? noTok[0] : noTok);
    if (!Number.isInteger(yesId) || yesId < 0 || !Number.isInteger(noId) || noId < 0) {
      throw new Error(`Invalid Qwen3 yes/no token IDs: yes=${yesId}, no=${noId}`);
    }

    state.rerankerYesId = yesId;
    state.rerankerNoId = noId;
    state.rerankerPrefixIds = tokenizer.encode(QWEN3_PROMPT_PREFIX, { add_special_tokens: false }).map(Number);
    state.rerankerSuffixIds = tokenizer.encode(QWEN3_PROMPT_SUFFIX, { add_special_tokens: false }).map(Number);
    state.reranker = model;
    state.rerankerTokenizer = tokenizer;
    state.rerankerDevice = chosenDevice;

    post('rerankerReady', {
      device: chosenDevice,
      modelId,
      dtype,
    });

    return { model, tokenizer, device: chosenDevice };
  })();

  try {
    return await state.rerankerLoadingPromise;
  } catch (err) {
    state.rerankerLoadingPromise = null;
    throw err;
  }
}

async function fetchDetailRecord(docIdx) {
  const chunkSize = state.catalogueMeta.meta.detail_chunk_size || 600;
  const bucketIdx = Math.floor(docIdx / chunkSize);
  const localIdx = docIdx % chunkSize;
  if (!state.detailChunkCache.has(bucketIdx)) {
    const chunkName = `data/details/chunk_${String(bucketIdx).padStart(3, '0')}.json`;
    const arr = await fetchJson(chunkName);
    state.detailChunkCache.set(bucketIdx, arr);
  }
  return state.detailChunkCache.get(bucketIdx)?.[localIdx] || {};
}

async function fetchDetailsBatch(docIndices) {
  const chunkSize = state.catalogueMeta.meta.detail_chunk_size || 600;
  const neededBuckets = new Set();
  for (const idx of docIndices) {
    const b = Math.floor(idx / chunkSize);
    if (!state.detailChunkCache.has(b)) neededBuckets.add(b);
  }
  if (neededBuckets.size > 0) {
    await Promise.all(
      Array.from(neededBuckets).map(async (b) => {
        const chunkName = `data/details/chunk_${String(b).padStart(3, '0')}.json`;
        try {
          const arr = await fetchJson(chunkName);
          state.detailChunkCache.set(b, arr);
        } catch {
          state.detailChunkCache.set(b, []);
        }
      }),
    );
  }
  return docIndices.map((idx) => {
    const b = Math.floor(idx / chunkSize);
    return state.detailChunkCache.get(b)?.[idx % chunkSize] || {};
  });
}

async function scoreWithQwen3(query, docTexts, searchSeq, maxLength = 512) {
  const { model, tokenizer } = await ensureRerankerLoaded();
  const prefixIds = state.rerankerPrefixIds;
  const suffixIds = state.rerankerSuffixIds;
  const budget = Math.max(64, maxLength - prefixIds.length - suffixIds.length);
  const yesId = state.rerankerYesId;
  const noId = state.rerankerNoId;

  const scores = new Array(docTexts.length);
  const t0 = performance.now();

  for (let i = 0; i < docTexts.length; i++) {
    if (state.cancelRerankFlag || state.activeSearchSeq !== searchSeq) {
      return { aborted: true, scores: scores.slice(0, i), completed: i };
    }

    const body = buildQwen3Body(query, docTexts[i], EO_RERANK_INSTRUCTION);
    const bodyIds = tokenizer.encode(body, { add_special_tokens: false }).slice(0, budget).map(Number);
    const fullIds = new BigInt64Array(prefixIds.length + bodyIds.length + suffixIds.length);
    let k = 0;
    for (let j = 0; j < prefixIds.length; j++) fullIds[k++] = BigInt(prefixIds[j]);
    for (let j = 0; j < bodyIds.length; j++) fullIds[k++] = BigInt(bodyIds[j]);
    for (let j = 0; j < suffixIds.length; j++) fullIds[k++] = BigInt(suffixIds[j]);

    const seqLen = fullIds.length;
    const attnMask = new BigInt64Array(seqLen).fill(1n);

    let inputIdsTensor = null;
    let attnTensor = null;
    let output = null;
    try {
      inputIdsTensor = new Tensor('int64', fullIds, [1, seqLen]);
      attnTensor = new Tensor('int64', attnMask, [1, seqLen]);
      output = await model({
        input_ids: inputIdsTensor,
        attention_mask: attnTensor,
      });

      const logits = output.logits;
      const vocabSize = logits.dims[logits.dims.length - 1];
      const lastOffset = (logits.dims[1] - 1) * vocabSize;
      const yesLogit = Number(logits.data[lastOffset + yesId]);
      const noLogit = Number(logits.data[lastOffset + noId]);
      const margin = yesLogit - noLogit;
      scores[i] = Number.isFinite(margin) ? margin : 0.0;
    } finally {
      await disposeNested(output);
      inputIdsTensor?.dispose?.();
      attnTensor?.dispose?.();
    }

    const elapsedSec = (performance.now() - t0) / 1000;
    const rate = (i + 1) / Math.max(elapsedSec, 0.05);
    const etaSec = (docTexts.length - (i + 1)) / Math.max(rate, 0.01);
    post('rerankProgress', {
      searchSeq,
      done: i + 1,
      total: docTexts.length,
      etaSec: Math.round(etaSec),
    });
  }

  return { aborted: false, scores, completed: docTexts.length };
}

function buildResultObjects(rankedEntries, stage1RankMap, rerankInfoMap = null) {
  return rankedEntries.map(([docIdx, score, meta], i) => {
    const r = state.records[docIdx];
    const adjG = state.searchMeta.adj_grid_m[docIdx];
    const effGridM = adjG !== null && adjG !== undefined ? adjG : r.grid_m;
    const rr = rerankInfoMap ? rerankInfoMap.get(docIdx) : null;
    const sibs = meta?.collapsed_siblings || [];
    return {
      rank: i + 1,
      doc_idx: docIdx,
      record_id: r.record_id,
      title: r.title,
      snippet: r.desc_snippet || '',
      source_catalogue: r.source_catalogue,
      producing_institution: r.producing_institution,
      provider_pool: r.provider_pool,
      copernicus_service: r.copernicus_service,
      processing_level: r.processing_level,
      uses_sentinel: r.uses_sentinel,
      sentinel_missions: r.sentinel_missions,
      grid_m: effGridM,
      grid: formatResolutionHuman(effGridM),
      cadence: r.temporal_cadence_class,
      geographic_coverage: r.geographic_coverage_class,
      t0: r.t0,
      t1: r.t1,
      operational_maturity_class: r.operational_maturity_class,
      doi: r.doi,
      access_url: r.access_url,
      score: Math.round(score * 1e5) / 1e5,
      stage1_rank: stage1RankMap.get(docIdx) ?? null,
      dense_rank: meta?.dense_rank ?? null,
      dense_score: meta?.dense_score ?? null,
      bm25_rank: meta?.bm25_rank ?? null,
      bm25_score: meta?.bm25_score ?? null,
      rerank_rank: rr ? rr.rerankRank : null,
      rerank_score: rr ? Math.round(rr.rerankScore * 1e4) / 1e4 : null,
      rerank_prob: rr ? Math.round((1 / (1 + Math.exp(-rr.rerankScore))) * 1e4) / 1e4 : null,
      collapsed_siblings: sibs.length,
      collapsed_items: sibs.slice(0, 12).map((s) => ({
        record_id: s.record_id,
        source_catalogue: s.source,
        title: s.title,
      })),
    };
  });
}

async function runSearch(payload) {
  const searchSeq = ++state.activeSearchSeq;
  state.cancelRerankFlag = false;

  const query = String(payload.query || '').trim();
  if (!query) return;

  const requestedMode = payload.mode || 'bm25';
  const useReranker = Boolean(payload.use_reranker);
  const rerankTopN = Math.max(10, Math.min(50, Number(payload.rerank_top_n || 25)));
  const topK = Math.max(1, Math.min(100, Number(payload.top_k || 20)));
  const doCollapse = payload.collapse !== false;
  const rerankPure = Boolean(payload.rerank_pure);

  const flt = {
    pools: payload.pools || null,
    sources: payload.sources || null,
    geo_scope: payload.geo || null,
    exclude_local: Boolean(payload.exclude_local),
    grid_rule: payload.grid_rule || null,
    max_grid_m: payload.max_grid_m ?? null,
    processing_levels: payload.levels || null,
    sentinel: payload.sentinel || null,
    provider_jurisdiction_tier: payload.tier || null,
    product_output_category: payload.category || null,
    cadence_classes: payload.cadence ? [payload.cadence] : null,
    operational_maturity_class: payload.maturity || null,
    product_ontology_class: payload.ontology || null,
    covers_year: payload.covers_year ?? null,
    min_end_year: payload.min_end_year ?? null,
    has_doi: Boolean(payload.has_doi),
    has_url: Boolean(payload.has_url),
    strict_eo_only: Boolean(payload.strict_eo || payload.eo_scope === 'strict_eo'),
    ontology_mode: payload.eo_scope || 'all',
    null_safe: payload.null_safe !== false,
    hard_limits: Boolean(payload.hard_limits),
    exclude_superseded: Boolean(payload.exclude_superseded),
  };

  const tStage1Start = performance.now();
  const { mask, eligibleCount } = applyFiltersMask(state.records, state.searchMeta, flt);

  if (eligibleCount === 0) {
    post('searchResult', {
      searchSeq,
      query,
      mode_used: requestedMode,
      reranker_used: 'none',
      device: state.embedderDevice || 'bm25',
      eligible_after_filter: 0,
      total_canonical: state.records.length,
      exclude_superseded: flt.exclude_superseded,
      hard_limits: flt.hard_limits,
      null_safe_filter: !flt.hard_limits && flt.null_safe,
      unknown_metadata_counts: state.searchMeta.unknown_metadata_counts,
      total_collapsed_in_results: 0,
      stage1_ms: performance.now() - tStage1Start,
      rerank_ms: 0,
      results: [],
    });
    return;
  }

  let stage1Ranked = [];
  const needDense = requestedMode === 'hybrid' || requestedMode === 'dense';
  const needBm25 = requestedMode === 'hybrid' || requestedMode === 'bm25';

  let denseRanked = [];
  let bm25Ranked = [];

  if (needBm25) {
    bm25Ranked = searchBm25(state.bm25, query, mask, state.recordIds, { topK: 400 });
  }
  if (needDense) {
    const qVec = await encodeQuery(query);
    if (state.activeSearchSeq !== searchSeq) return;
    denseRanked = scoreDenseInt8(state.embStore, qVec, mask, 400);
  }

  if (requestedMode === 'hybrid') {
    stage1Ranked = rrfFuse(denseRanked, bm25Ranked, { rrfK: 60, wDense: 1.0, wBm25: 0.6 });
  } else if (requestedMode === 'dense') {
    stage1Ranked = denseRanked.map(([idx, sc], i) => [
      idx,
      sc,
      { dense_rank: i + 1, dense_score: Math.round(sc * 1e5) / 1e5 },
    ]);
  } else {
    stage1Ranked = bm25Ranked.map(([idx, sc], i) => [
      idx,
      sc,
      { bm25_rank: i + 1, bm25_score: Math.round(sc * 1e5) / 1e5 },
    ]);
  }

  const stage1PoolTarget = useReranker ? Math.max(rerankTopN, topK) : topK;
  let collapsedStage1 = stage1Ranked;
  if (doCollapse) {
    collapsedStage1 = collapseProductFamilies(stage1Ranked, state.records, state.searchMeta, {
      topK: stage1PoolTarget,
      embStore: state.embStore,
      nearDupCos: 0.94,
      sensorGuard: true,
      queryText: query,
    });
  } else {
    collapsedStage1 = stage1Ranked.slice(0, stage1PoolTarget);
  }

  const stage1RankMap = new Map();
  const metaByDoc = new Map();
  for (let i = 0; i < collapsedStage1.length; i++) {
    const [dIdx, , m] = collapsedStage1[i];
    stage1RankMap.set(dIdx, i + 1);
    metaByDoc.set(dIdx, m);
  }

  const stage1Ms = performance.now() - tStage1Start;
  const stage1Display = collapsedStage1.slice(0, topK);
  const stage1Results = buildResultObjects(stage1Display, stage1RankMap, null);
  const stage1CollapsedCount = stage1Results.reduce((a, r) => a + r.collapsed_siblings, 0);

  post('searchResult', {
    searchSeq,
    query,
    mode_used: requestedMode,
    reranker_used: 'none',
    reranking_pending: useReranker && collapsedStage1.length > 1,
    rerank_pool: Math.min(rerankTopN, collapsedStage1.length),
    device: state.embedderDevice || 'bm25',
    eligible_after_filter: eligibleCount,
    total_canonical: state.records.length,
    exclude_superseded: flt.exclude_superseded,
    hard_limits: flt.hard_limits,
    null_safe_filter: !flt.hard_limits && flt.null_safe,
    unknown_metadata_counts: state.searchMeta.unknown_metadata_counts,
    total_collapsed_in_results: stage1CollapsedCount,
    stage1_ms: stage1Ms,
    rerank_ms: 0,
    results: stage1Results,
  });

  if (!useReranker || collapsedStage1.length <= 1) return;

  const poolEntries = collapsedStage1.slice(0, rerankTopN);
  const candidateIdx = poolEntries.map(([dIdx]) => dIdx);
  const tRerankLoadStart = performance.now();
  const wasLoaded = Boolean(state.reranker);
  await ensureRerankerLoaded();
  if (state.activeSearchSeq !== searchSeq || state.cancelRerankFlag) return;
  const rerankLoadMs = wasLoaded ? 0 : performance.now() - tRerankLoadStart;

  const details = await fetchDetailsBatch(candidateIdx);
  if (state.activeSearchSeq !== searchSeq || state.cancelRerankFlag) return;

  const docTexts = candidateIdx.map((dIdx, i) => formatRerankerDoc(state.records[dIdx], details[i], 1200));

  const tRerankStart = performance.now();
  const rrOut = await scoreWithQwen3(query, docTexts, searchSeq, 512);
  const rerankMs = performance.now() - tRerankStart;

  if (state.activeSearchSeq !== searchSeq) return;
  if (rrOut.aborted) {
    post('rerankAborted', { searchSeq, completed: rrOut.completed, total: candidateIdx.length });
    return;
  }

  let finalRanked;
  const rerankInfoMap = new Map();
  const tailEntries = collapsedStage1.slice(poolEntries.length, topK);

  if (rerankPure) {
    const order = candidateIdx.map((_, idx) => idx);
    order.sort((a, bItem) => {
      const diff = rrOut.scores[bItem] - rrOut.scores[a];
      return diff !== 0 ? diff : a - bItem;
    });
    finalRanked = order.slice(0, topK).map((origPos, rk) => {
      const dIdx = candidateIdx[origPos];
      rerankInfoMap.set(dIdx, { rerankRank: rk + 1, rerankScore: rrOut.scores[origPos] });
      return [dIdx, rrOut.scores[origPos], metaByDoc.get(dIdx)];
    });
  } else {
    const blended = blendRrfAndRerank(candidateIdx, rrOut.scores, { rrfK: 30, wRrf: 0.4, wRerank: 1.0 });
    finalRanked = blended.slice(0, topK).map((item) => {
      rerankInfoMap.set(item.docIdx, { rerankRank: item.rerankRank, rerankScore: item.rerankScore });
      return [item.docIdx, item.blendedScore, metaByDoc.get(item.docIdx)];
    });
  }

  if (finalRanked.length < topK && tailEntries.length > 0) {
    for (const entry of tailEntries) {
      if (finalRanked.length >= topK) break;
      finalRanked.push(entry);
    }
  }

  const finalResults = buildResultObjects(finalRanked, stage1RankMap, rerankInfoMap);
  const finalCollapsed = finalResults.reduce((a, r) => a + r.collapsed_siblings, 0);

  post('searchResult', {
    searchSeq,
    query,
    mode_used: `${requestedMode}+qwen3`,
    reranker_used: state.manifest.models.reranker,
    reranking_pending: false,
    rerank_pool: candidateIdx.length,
    rerank_pure: rerankPure,
    device: state.rerankerDevice || state.embedderDevice || 'wasm',
    eligible_after_filter: eligibleCount,
    total_canonical: state.records.length,
    exclude_superseded: flt.exclude_superseded,
    hard_limits: flt.hard_limits,
    null_safe_filter: !flt.hard_limits && flt.null_safe,
    unknown_metadata_counts: state.searchMeta.unknown_metadata_counts,
    total_collapsed_in_results: finalCollapsed,
    stage1_ms: stage1Ms,
    rerank_ms: rerankMs,
    rerank_load_ms: rerankLoadMs,
    results: finalResults,
  });
}

async function getProductDetail(recordId, reqId) {
  const docIdx = state.recordIdToIdx.get(recordId);
  if (docIdx === undefined) {
    post('productDetail', { reqId, error: `Unknown record_id: ${recordId}` });
    return;
  }
  const base = state.records[docIdx];
  const det = await fetchDetailRecord(docIdx);
  const adjG = state.searchMeta.adj_grid_m[docIdx];
  const effGridM = adjG !== null && adjG !== undefined ? adjG : base.grid_m;
  post('productDetail', {
    reqId,
    detail: {
      ...base,
      ...det,
      record_id: base.record_id,
      strict_eo_c1_eligible: base.strict_eo,
      adjudicated_output_grid_meters: adjG,
      grid_human: formatResolutionHuman(effGridM, det.adjudicated_spatial_grid_value, det.adjudicated_spatial_grid_units),
    },
  });
}

self.addEventListener('message', async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === 'init') {
      await initEngine(msg.baseUrl);
    } else if (msg.type === 'enableSemantic') {
      await ensureEmbedderLoaded();
    } else if (msg.type === 'enableReranker') {
      await ensureRerankerLoaded();
    } else if (msg.type === 'cancelRerank') {
      state.cancelRerankFlag = true;
    } else if (msg.type === 'search') {
      await runSearch(msg.payload);
    } else if (msg.type === 'getProductDetail') {
      await getProductDetail(msg.recordId, msg.reqId);
    }
  } catch (err) {
    post('error', {
      source: msg.type,
      reqId: msg.reqId,
      message: err?.message || String(err),
    });
  }
});
