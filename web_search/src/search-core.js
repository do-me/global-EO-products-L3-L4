import snowball from 'snowball-stemmers';

const porterStemmer = snowball.newStemmer('porter');

export const QUERY_PROMPT_PREFIX = 'task: search result | query: ';
export const DISCOVERY_CURRENT_YEAR = 2026;

export const EO_RERANK_INSTRUCTION =
  'Given an abstract environmental policy, international treaty, or scientific Earth Observation ' +
  'monitoring requirement, judge whether the candidate satellite or geospatial data product ' +
  'directly provides the required biophysical variable, observation domain, and spatial scope.';

const SUBSCRIPT_MAP = {
  '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
  '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
  '⁻': '-', '–': '-', '—': '-',
  '’': "'", '‘': "'", '“': '"', '”': '"',
};
const SUBSCRIPT_RE = /[₀₁₂₃₄₅₆₇₈₉⁻–—’‘“”]/g;

const EO_LEXICAL_SUBS = [
  [/\bnormalised\b/gi, 'normalized'],
  [/\bcolour\b/gi, 'color'],
  [/\bcolours\b/gi, 'colors'],
  [/\bmodelling\b/gi, 'modeling'],
  [/\bmodelled\b/gi, 'modeled'],
  [/\burbanisation\b/gi, 'urbanization'],
  [/\bcharacterisation\b/gi, 'characterization'],
  [/\bclimatological\b/gi, 'climatology'],
  [/\bburnt\b/gi, 'burned'],
  [/\babove[-\s]?ground\b/gi, 'aboveground'],
  [/\bbelow[-\s]?ground\b/gi, 'belowground'],
  [/\bbuilt[-\s]?up\b/gi, 'builtup built up'],
  [/\bland[-\s]?cover\b/gi, 'land cover landcover'],
  [/\bland[-\s]?use\b/gi, 'land use landuse'],
  [/\bsea[-\s]?ice\b/gi, 'sea ice'],
  [/\bsea[-\s]?level\b/gi, 'sea level'],
  [/\bchlorophyll[-\s]?a\b/gi, 'chlorophyll a chla'],
  [/\bpm\s*2\.?5\b/gi, 'pm25 pm2.5 fine particulate'],
  [/\bco\s*2\b/gi, 'co2 carbon dioxide'],
  [/\bch\s*4\b/gi, 'ch4 methane'],
  [/\bno\s*2\b/gi, 'no2 nitrogen dioxide'],
  [/\bhigh[-\s]?resolution\b/gi, 'high resolution'],
  [/\bnear[-\s]?real[-\s]?time\b/gi, 'near real time nrt'],
  [/\bsub[-\s]?surface\b/gi, 'subsurface'],
  [/\bfresh[-\s]?water\b/gi, 'freshwater'],
  [/\btree[-\s]?cover\b/gi, 'tree cover'],
  [/\bsoil organic carbon\b/gi, 'soil organic carbon soc'],
  [/\bleaf area index\b/gi, 'leaf area index lai'],
  [/\bgross primary product\w*\b/gi, 'gross primary productivity gpp'],
  [/\bnet primary product\w*\b/gi, 'net primary productivity npp'],
  [/\bsea surface temperature\b/gi, 'sea surface temperature sst'],
  [/\bland surface temperature\b/gi, 'land surface temperature lst'],
  [/\bsea surface salinity\b/gi, 'sea surface salinity sss'],
  [/\bdigital elevation model\b/gi, 'digital elevation model dem'],
  [/\bsolar[-\s]?induced\s+(?:chlorophyll\s+)?fluorescence\b/gi, 'solar induced fluorescence sif'],
  [/\bwater[-\s]?quality\b/gi, 'water quality'],
  [/\bheat[-\s]?island\b/gi, 'heat island uhi'],
  [/\bburned\s+area\b/gi, 'burned area wildfire'],
];

const REGION_BBOX = {
  europe: [-15.0, 34.0, 45.0, 72.0],
  north_america: [-170.0, 15.0, -50.0, 75.0],
  africa: [-20.0, -36.0, 52.0, 38.0],
};

const HARD_GLOBAL_EXCLUDED_DOMAINS = new Set([
  'hemispheric_or_pantropical',
  'regional_or_local',
  'northern_hemisphere_only',
  'southern_hemisphere_only',
  'geostationary_disk_or_sector',
]);

const EXACT_GEO_CLASSES = new Set([
  'quasi_global',
  'hemispheric_or_pantropical',
  'europe_regional',
  'regional_or_local',
]);

const ONTOLOGY_LABELS = {
  eo_thematic_or_biophysical: 'satellite biophysical/thematic EO product',
  eo_reanalysis_or_model: 'EO climate reanalysis/model product',
  eo_constrained_hybrid: 'EO-constrained hybrid geospatial product',
  in_situ_gridded_interpolation: 'in-situ gridded interpolation dataset',
  non_eo_inventory_or_census: 'non-EO inventory/census dataset',
  eo_ard_surface_reflectance_mosaic: 'EO analysis-ready surface reflectance mosaic',
  authoritative_reference_geometry: 'authoritative reference boundary/geometry',
};

const GEO_LABELS_DOC = {
  global: 'global',
  quasi_global: 'quasi-global',
  europe_regional: 'Europe regional',
  hemispheric_or_pantropical: 'hemispheric/pantropical',
  regional_or_local: 'regional/local',
};

const CADENCE_LABELS_DOC = {
  sub_daily_or_daily: 'daily/sub-daily',
  multi_day_or_dekadal: 'multi-day/10-daily',
  monthly_or_seasonal: 'monthly/seasonal',
  annual: 'annual',
  multi_year_periodic: 'multi-year periodic',
  static_or_single_epoch: 'static/single-epoch',
};

const DUCKDB_SPLIT_RE = /[0-9!@#$%^&*()_+={}\[\]:;<>,.?~\\/|'"`\-\s]+/;

export function translateSubscripts(text) {
  if (!text) return '';
  return String(text).replace(SUBSCRIPT_RE, (ch) => SUBSCRIPT_MAP[ch] || ch);
}

export function normalizeEoText(text) {
  if (!text) return '';
  let s = translateSubscripts(text);
  s = s.replace(/[/|;:,()\[\]{}_]+/g, ' ');
  for (let i = 0; i < EO_LEXICAL_SUBS.length; i++) {
    s = s.replace(EO_LEXICAL_SUBS[i][0], EO_LEXICAL_SUBS[i][1]);
  }
  return s.replace(/\s+/g, ' ').trim();
}

export function buildQueryPrompt(query) {
  const q = translateSubscripts(query).replace(/\s+/g, ' ').trim();
  return `${QUERY_PROMPT_PREFIX}${q}`;
}

export function stripAccents(str) {
  return String(str).normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

export function stemPorter(word) {
  return porterStemmer.stem(word);
}

export function tokenizeForBm25(query, stopwordsSet) {
  const qNorm = normalizeEoText(query);
  if (!qNorm) return [];
  const rawTokens = stripAccents(qNorm).toLowerCase().split(DUCKDB_SPLIT_RE);
  const uniq = new Set();
  for (let i = 0; i < rawTokens.length; i++) {
    const raw = rawTokens[i];
    if (!raw) continue;
    if (stopwordsSet && stopwordsSet.has(raw)) continue;
    const stemmed = porterStemmer.stem(raw);
    if (stemmed) uniq.add(stemmed);
  }
  return Array.from(uniq);
}

export async function decompressGzip(arrayBuffer) {
  if (typeof DecompressionStream !== 'undefined') {
    const ds = new DecompressionStream('gzip');
    const stream = new Blob([arrayBuffer]).stream().pipeThrough(ds);
    return new Response(stream).arrayBuffer();
  }
  throw new Error('Browser does not support DecompressionStream("gzip").');
}

export function parseBm25Binary(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength < 48) {
    throw new Error('Truncated BM25 binary payload.');
  }
  const dv = new DataView(arrayBuffer);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x424d3235) {
    throw new Error(`Invalid BM25 binary magic: 0x${magic.toString(16)}`);
  }
  const version = dv.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported BM25 binary version: ${version}`);
  }
  const nDocs = dv.getUint32(8, true);
  const nTerms = dv.getUint32(12, true);
  const avgFieldLens = [
    dv.getFloat64(16, true),
    dv.getFloat64(24, true),
    dv.getFloat64(32, true),
  ];

  const totalBytes = arrayBuffer.byteLength;
  const checkRange = (off, len, label) => {
    if (off < 0 || len < 0 || off + len > totalBytes) {
      throw new Error(`BM25 binary out-of-bounds at ${label} (off=${off}, len=${len}, total=${totalBytes})`);
    }
  };

  let off = 40;
  const fieldLensBytes = nDocs * 3 * 2;
  checkRange(off, fieldLensBytes, 'fieldLens');
  const fieldLens = new Uint16Array(arrayBuffer.slice(off, off + fieldLensBytes));
  off += fieldLensBytes;

  checkRange(off, 4, 'swLen');
  const swLen = dv.getUint32(off, true);
  off += 4;
  checkRange(off, swLen, 'stopwords');
  const dec = new TextDecoder('utf-8');
  const swStr = dec.decode(new Uint8Array(arrayBuffer, off, swLen));
  const stopwords = new Set(swStr ? swStr.split('\n') : []);
  off += swLen;

  checkRange(off, 4, 'vocabLen');
  const vocabLen = dv.getUint32(off, true);
  off += 4;
  checkRange(off, vocabLen, 'vocab');
  const vocabStr = dec.decode(new Uint8Array(arrayBuffer, off, vocabLen));
  const vocabList = vocabStr ? vocabStr.split('\n') : [];
  if (vocabList.length !== nTerms) {
    throw new Error(`BM25 vocab length mismatch: ${vocabList.length} != ${nTerms}`);
  }
  const vocab = new Map();
  for (let i = 0; i < nTerms; i++) {
    vocab.set(vocabList[i], i);
  }
  off += vocabLen;

  const termDfBytes = nTerms * 4;
  checkRange(off, termDfBytes, 'termDf');
  const termDf = new Uint32Array(arrayBuffer.slice(off, off + termDfBytes));
  off += termDfBytes;

  const termOffsetsBytes = (nTerms + 1) * 4;
  checkRange(off, termOffsetsBytes, 'termOffsets');
  const termOffsets = new Uint32Array(arrayBuffer.slice(off, off + termOffsetsBytes));
  off += termOffsetsBytes;

  const postings = new Uint8Array(arrayBuffer, off);
  if (termOffsets[nTerms] > postings.byteLength) {
    throw new Error(`BM25 postings truncated: expected ${termOffsets[nTerms]} bytes, got ${postings.byteLength}`);
  }

  return {
    nDocs,
    nTerms,
    avgFieldLens,
    fieldLens,
    stopwords,
    vocab,
    termDf,
    termOffsets,
    postings,
  };
}

export function searchBm25(bm25, query, eligibleMask, recordIds, opts = {}) {
  const {
    topK = 200,
    wTitle = 2.5,
    wKeywords = 1.5,
    wBody = 1.0,
    k1 = 1.2,
    b = 0.75,
  } = opts;

  const tokens = tokenizeForBm25(query, bm25.stopwords);
  if (tokens.length === 0) return [];

  const { nDocs, avgFieldLens, fieldLens, vocab, termDf, termOffsets, postings } = bm25;
  const avg0 = avgFieldLens[0];
  const avg1 = avgFieldLens[1];
  const avg2 = avgFieldLens[2];
  const oneMinusB = 1.0 - b;
  const k1Plus1 = k1 + 1.0;

  const scores = new Float64Array(nDocs);
  const touched = [];

  for (let ti = 0; ti < tokens.length; ti++) {
    const termId = vocab.get(tokens[ti]);
    if (termId === undefined) continue;

    const df = termDf[termId];
    const idf = Math.log10(((nDocs - df + 0.5) / (df + 0.5)) + 1.0);
    let pos = termOffsets[termId];
    const end = termOffsets[termId + 1];
    let docId = 0;

    while (pos < end) {
      let delta = 0;
      let shift = 0;
      while (pos < end) {
        const byte = postings[pos++];
        delta |= (byte & 0x7f) << shift;
        if ((byte & 0x80) === 0) break;
        shift += 7;
      }
      docId += delta;
      if (docId >= nDocs || pos >= end) break;

      const flags = postings[pos++];
      let tf0 = (flags & 0x01) ? 1 : 0;
      let tf1 = (flags & 0x02) ? 1 : 0;
      let tf2 = (flags & 0x04) ? 1 : 0;

      if (flags & 0x08) {
        tf0 = 0;
        let sh = 0;
        while (pos < end) {
          const bt = postings[pos++];
          tf0 |= (bt & 0x7f) << sh;
          if ((bt & 0x80) === 0) break;
          sh += 7;
        }
      }
      if (flags & 0x10) {
        tf1 = 0;
        let sh = 0;
        while (pos < end) {
          const bt = postings[pos++];
          tf1 |= (bt & 0x7f) << sh;
          if ((bt & 0x80) === 0) break;
          sh += 7;
        }
      }
      if (flags & 0x20) {
        tf2 = 0;
        let sh = 0;
        while (pos < end) {
          const bt = postings[pos++];
          tf2 |= (bt & 0x7f) << sh;
          if ((bt & 0x80) === 0) break;
          sh += 7;
        }
      }

      if (eligibleMask && !eligibleMask[docId]) continue;

      let contrib = 0.0;
      const flBase = docId * 3;
      if (tf0 > 0) {
        const normTf = tf0 / (oneMinusB + b * (fieldLens[flBase] / avg0));
        contrib += wTitle * idf * ((normTf * k1Plus1) / (normTf + k1));
      }
      if (tf1 > 0) {
        const normTf = tf1 / (oneMinusB + b * (fieldLens[flBase + 1] / avg1));
        contrib += wKeywords * idf * ((normTf * k1Plus1) / (normTf + k1));
      }
      if (tf2 > 0) {
        const normTf = tf2 / (oneMinusB + b * (fieldLens[flBase + 2] / avg2));
        contrib += wBody * idf * ((normTf * k1Plus1) / (normTf + k1));
      }

      if (scores[docId] === 0.0 && contrib !== 0.0) {
        touched.push(docId);
      }
      scores[docId] += contrib;
    }
  }

  if (touched.length === 0) return [];

  const ranked = new Array(touched.length);
  for (let i = 0; i < touched.length; i++) {
    const idx = touched[i];
    const sc = scores[idx];
    ranked[i] = [idx, sc, Math.round(sc * 1e10)];
  }
  ranked.sort((a, bItem) => {
    if (bItem[2] !== a[2]) return bItem[2] - a[2];
    if (recordIds) {
      const ra = recordIds[a[0]];
      const rb = recordIds[bItem[0]];
      return ra < rb ? -1 : ra > rb ? 1 : 0;
    }
    return a[0] - bItem[0];
  });

  const limit = Math.min(topK, ranked.length);
  const out = new Array(limit);
  for (let i = 0; i < limit; i++) {
    out[i] = [ranked[i][0], ranked[i][1]];
  }
  return out;
}

export function parseEmbChunkBinary(arrayBuffer) {
  if (!arrayBuffer || arrayBuffer.byteLength < 20) {
    throw new Error('Truncated EMB8 binary payload.');
  }
  const dv = new DataView(arrayBuffer);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x454d4238) {
    throw new Error(`Invalid EMB8 binary magic: 0x${magic.toString(16)}`);
  }
  const version = dv.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported EMB8 binary version: ${version}`);
  }
  const startRow = dv.getUint32(8, true);
  const nRows = dv.getUint32(12, true);
  const dim = dv.getUint32(16, true);
  const scalesOffset = 20;
  const scalesBytes = nRows * 4;
  const quantBytes = nRows * dim;
  if (scalesOffset + scalesBytes + quantBytes > arrayBuffer.byteLength) {
    throw new Error(`EMB8 chunk truncated (need ${scalesOffset + scalesBytes + quantBytes}, got ${arrayBuffer.byteLength})`);
  }
  const scales = new Float32Array(arrayBuffer.slice(scalesOffset, scalesOffset + scalesBytes));
  const quant = new Int8Array(arrayBuffer, scalesOffset + scalesBytes, quantBytes);
  return { startRow, nRows, dim, scales, quant };
}

export function assembleEmbStore(chunks, totalDocs, dim) {
  const scales = new Float32Array(totalDocs);
  const quant = new Int8Array(totalDocs * dim);
  let rowsFilled = 0;
  for (const ch of chunks) {
    if (ch.dim !== dim) throw new Error(`Embedding dim mismatch: ${ch.dim} != ${dim}`);
    if (ch.startRow + ch.nRows > totalDocs) {
      throw new Error(`Embedding chunk out of bounds: ${ch.startRow}+${ch.nRows} > ${totalDocs}`);
    }
    scales.set(ch.scales, ch.startRow);
    quant.set(ch.quant, ch.startRow * dim);
    rowsFilled += ch.nRows;
  }
  if (rowsFilled !== totalDocs) {
    throw new Error(`Incomplete embedding shards: loaded ${rowsFilled} rows, expected ${totalDocs}`);
  }
  return { nDocs: totalDocs, dim, scales, quant };
}

export function truncateAndNormalizeQuery(rawVec, dim) {
  if (!rawVec || rawVec.length < dim) {
    throw new Error(`Query vector too short: got ${rawVec ? rawVec.length : 0}, need ${dim}`);
  }
  const q = new Float32Array(dim);
  let normSq = 0.0;
  for (let i = 0; i < dim; i++) {
    const v = rawVec[i];
    q[i] = Number.isFinite(v) ? v : 0.0;
    normSq += q[i] * q[i];
  }
  const invNorm = 1.0 / Math.max(Math.sqrt(normSq), 1e-12);
  for (let i = 0; i < dim; i++) {
    q[i] *= invNorm;
  }
  return q;
}

export function scoreDenseInt8(embStore, queryVecRaw, eligibleMask, topK = 300) {
  const { nDocs, dim, scales, quant } = embStore;
  const q = truncateAndNormalizeQuery(queryVecRaw, dim);
  const scores = new Float32Array(nDocs);
  const eligibleIdx = [];

  for (let i = 0; i < nDocs; i++) {
    if (eligibleMask && !eligibleMask[i]) {
      scores[i] = -1e9;
      continue;
    }
    const base = i * dim;
    let acc = 0.0;
    let d = 0;
    const lim = dim - 3;
    for (; d < lim; d += 4) {
      acc +=
        quant[base + d] * q[d] +
        quant[base + d + 1] * q[d + 1] +
        quant[base + d + 2] * q[d + 2] +
        quant[base + d + 3] * q[d + 3];
    }
    for (; d < dim; d++) {
      acc += quant[base + d] * q[d];
    }
    scores[i] = acc * scales[i];
    eligibleIdx.push(i);
  }

  eligibleIdx.sort((a, bItem) => {
    const diff = scores[bItem] - scores[a];
    if (diff !== 0) return diff;
    return a - bItem;
  });

  const limit = Math.min(topK, eligibleIdx.length);
  const out = new Array(limit);
  for (let i = 0; i < limit; i++) {
    const idx = eligibleIdx[i];
    out[i] = [idx, scores[idx]];
  }
  return out;
}

export function cosineEmbInt8(embStore, idxA, idxB) {
  const { dim, scales, quant } = embStore;
  const baseA = idxA * dim;
  const baseB = idxB * dim;
  let acc = 0;
  let d = 0;
  const lim = dim - 3;
  for (; d < lim; d += 4) {
    acc +=
      quant[baseA + d] * quant[baseB + d] +
      quant[baseA + d + 1] * quant[baseB + d + 1] +
      quant[baseA + d + 2] * quant[baseB + d + 2] +
      quant[baseA + d + 3] * quant[baseB + d + 3];
  }
  for (; d < dim; d++) {
    acc += quant[baseA + d] * quant[baseB + d];
  }
  return acc * scales[idxA] * scales[idxB];
}

function normLon(lon) {
  if (lon > 180.0 || lon < -180.0) {
    return ((((lon + 180.0) % 360.0) + 360.0) % 360.0) - 180.0;
  }
  return lon;
}

function parseGridRule(rule) {
  if (!rule) return { maxM: null, minM: null, wantNull: false };
  if (rule === '__NULL__') return { maxM: null, minM: null, wantNull: true };
  if (rule === 'gt50000') return { maxM: null, minM: 50000.0001, wantNull: false };
  const m = String(rule).match(/^le(\d+(?:\.\d+)?)$/);
  if (m) return { maxM: Number(m[1]), minM: null, wantNull: false };
  const n = Number(rule);
  if (Number.isFinite(n) && n > 0) return { maxM: n, minM: null, wantNull: false };
  return { maxM: null, minM: null, wantNull: false };
}

export function applyFiltersMask(records, searchMeta, flt = {}) {
  const n = records.length;
  const mask = new Uint8Array(n);
  const hardLimits = Boolean(flt.hard_limits);
  const effNullSafe = hardLimits ? false : flt.null_safe !== false;
  const effGridTol = hardLimits ? 1.0 : (flt.grid_tolerance ?? 1.25);
  const excludeSuperseded = Boolean(flt.exclude_superseded);

  const flags = searchMeta.flags;
  const adjGrid = searchMeta.adj_grid_m;

  const poolSet = flt.pools && flt.pools.length > 0
    ? new Set(flt.pools)
    : flt.provider_pool
      ? new Set([flt.provider_pool])
      : null;
  const sourceSet = flt.sources && flt.sources.length > 0 ? new Set(flt.sources) : null;

  const rawLevels = flt.processing_levels && flt.processing_levels.length > 0 ? flt.processing_levels : null;
  const wantNullLevel = Boolean(rawLevels && rawLevels.includes('__NULL__'));
  const levelSet = rawLevels && !wantNullLevel ? new Set(rawLevels) : null;
  const exactMultiLevel = Boolean(levelSet && levelSet.size === 1 && levelSet.has('L3; L4'));
  const allowsL1L2 = Boolean(levelSet && (levelSet.has('L1') || levelSet.has('L2')));

  const sentinelFilter = flt.sentinel ? String(flt.sentinel) : null;
  const sentinelMission = sentinelFilter && sentinelFilter.startsWith('mission:') ? sentinelFilter.slice(8) : null;

  const tierFilter = flt.provider_jurisdiction_tier || null;
  const categoryFilter = flt.product_output_category || null;
  const maturityFilter = flt.operational_maturity_class || null;
  const ontologyFilter = flt.product_ontology_class || null;
  const hasDoi = Boolean(flt.has_doi);
  const hasUrl = Boolean(flt.has_url);

  const rawCadences = flt.cadence_classes && flt.cadence_classes.length > 0 ? flt.cadence_classes : null;
  const wantNullCadence = Boolean(rawCadences && rawCadences.includes('__NULL__'));
  const cadSet = rawCadences && !wantNullCadence ? new Set(rawCadences) : null;

  const parsedRule = parseGridRule(flt.grid_rule);
  const wantNullGrid = parsedRule.wantNull;
  const maxGridM =
    flt.max_grid_m !== null && flt.max_grid_m !== undefined
      ? Number(flt.max_grid_m)
      : parsedRule.maxM;
  const minGridM =
    flt.min_grid_m !== null && flt.min_grid_m !== undefined
      ? Number(flt.min_grid_m)
      : parsedRule.minM;

  const coversYear = flt.covers_year !== null && flt.covers_year !== undefined && Number.isFinite(Number(flt.covers_year))
    ? Number(flt.covers_year)
    : null;
  const minEndYear = flt.min_end_year !== null && flt.min_end_year !== undefined && Number.isFinite(Number(flt.min_end_year))
    ? Number(flt.min_end_year)
    : null;
  const maxStartYear = flt.max_start_year !== null && flt.max_start_year !== undefined && Number.isFinite(Number(flt.max_start_year))
    ? Number(flt.max_start_year)
    : null;

  const geoScopeRaw = flt.geo_scope ? String(flt.geo_scope) : null;
  const geoScope = geoScopeRaw && geoScopeRaw !== '__NULL__' ? geoScopeRaw.toLowerCase() : geoScopeRaw;
  const excludeLocal = Boolean(flt.exclude_local || geoScope === 'exclude_local');
  const strictEo = Boolean(flt.strict_eo_only || flt.ontology_mode === 'strict_eo');
  const eoAndHybrid = !strictEo && flt.ontology_mode === 'eo_and_hybrid';

  let eligibleCount = 0;

  for (let i = 0; i < n; i++) {
    const f = flags[i];
    if (excludeSuperseded && (f & 1)) continue;
    if (strictEo && !(f & 16)) continue;
    if (eoAndHybrid && !(f & 32)) continue;

    const r = records[i];
    if (poolSet && !poolSet.has(r.provider_pool)) continue;
    if (sourceSet && !sourceSet.has(r.source_catalogue)) continue;
    if (hasDoi && !r.doi) continue;
    if (hasUrl && !r.access_url) continue;
    if (tierFilter && r.provider_jurisdiction_tier !== tierFilter) continue;
    if (categoryFilter && r.product_output_category !== categoryFilter) continue;
    if (maturityFilter && r.operational_maturity_class !== maturityFilter) continue;
    if (ontologyFilter && r.product_ontology_class !== ontologyFilter) continue;

    if (sentinelFilter) {
      if (sentinelFilter === 'yes') {
        if (!r.uses_sentinel) continue;
      } else if (sentinelFilter === 'no_explicit') {
        if (r.uses_sentinel || r.sentinel_lineage_status !== 'explicit_non_sentinel') continue;
      } else if (sentinelFilter === 'unassessed') {
        if (r.sentinel_lineage_status !== 'unassessed_no_platform_metadata') continue;
      } else if (sentinelMission) {
        if (!r.uses_sentinel || !r.sentinel_missions) continue;
        const toks = r.sentinel_missions.split(/;\s*/);
        if (!toks.includes(sentinelMission)) continue;
      }
    }

    if (wantNullLevel) {
      if (r.processing_level !== null && r.processing_level !== '') continue;
    } else if (levelSet) {
      const pl = r.processing_level;
      if (!pl) {
        if (!effNullSafe) continue;
      } else if (exactMultiLevel) {
        if (pl !== 'L3; L4') continue;
      } else {
        const toks = pl.split(';');
        let hit = false;
        for (let k = 0; k < toks.length; k++) {
          if (levelSet.has(toks[k].trim())) {
            hit = true;
            break;
          }
        }
        if (!hit) continue;
      }
    }

    if (wantNullGrid) {
      const g = hardLimits ? adjGrid[i] : (adjGrid[i] ?? r.grid_m);
      if (g !== null && g !== undefined && Number.isFinite(g)) continue;
    } else {
      if (maxGridM !== null) {
        const g = adjGrid[i];
        if (g === null || g === undefined || !Number.isFinite(g)) {
          if (!effNullSafe) continue;
        } else if (g > maxGridM * effGridTol) {
          continue;
        }
        if (hardLimits) {
          if (!(f & 2)) continue;
          if (f & 4) continue;
          if (!allowsL1L2 && (r.processing_level === 'L1' || r.processing_level === 'L2')) continue;
        }
      }

      if (minGridM !== null) {
        const g = adjGrid[i];
        if (g === null || g === undefined || !Number.isFinite(g)) {
          if (!effNullSafe) continue;
        } else if (g < minGridM) {
          continue;
        }
        if (hardLimits) {
          if (!(f & 2)) continue;
          if (f & 4) continue;
        }
      }
    }

    if (wantNullCadence) {
      if (r.temporal_cadence_class !== null && r.temporal_cadence_class !== '') continue;
    } else if (cadSet) {
      const cad = r.temporal_cadence_class;
      if (!cad) {
        if (!effNullSafe) continue;
      } else if (!cadSet.has(cad)) {
        continue;
      }
    }

    const gc = r.geographic_coverage_class || '';
    if (excludeLocal) {
      if (!gc) {
        if (!effNullSafe) continue;
      } else if (
        gc !== 'global' &&
        gc !== 'quasi_global' &&
        gc !== 'hemispheric_or_pantropical' &&
        gc !== 'europe_regional'
      ) {
        continue;
      }
    }

    if (geoScope && geoScope !== 'exclude_local') {
      if (geoScope === '__NULL__') {
        if (gc !== '') continue;
      } else if (EXACT_GEO_CLASSES.has(geoScope)) {
        if (!gc) {
          if (!effNullSafe) continue;
        } else if (gc !== geoScope) {
          continue;
        }
      } else if (geoScope === 'global') {
        if (hardLimits) {
          if (gc !== 'global') continue;
          const od = r.observation_domain_class || '';
          if (HARD_GLOBAL_EXCLUDED_DOMAINS.has(od)) continue;
          if (r.bbox && r.bbox.length === 4) {
            const bs = r.bbox[1];
            const bn = r.bbox[3];
            if (Number.isFinite(bs) && Number.isFinite(bn)) {
              if (!(bs <= -55.0 && bn >= 55.0)) continue;
            }
          }
        } else {
          if (!gc) {
            if (!effNullSafe) continue;
          } else if (gc !== 'global' && gc !== 'quasi_global' && gc !== 'hemispheric_or_pantropical') {
            continue;
          }
        }
      } else if (REGION_BBOX[geoScope]) {
        const [bwMin, bsMin, beMax, bnMax] = REGION_BBOX[geoScope];
        if (!gc) {
          if (!effNullSafe) continue;
        } else {
          let ok = gc === 'global' || gc === 'quasi_global' || (geoScope === 'europe' && gc === 'europe_regional');
          if (!ok && (gc === 'regional_or_local' || gc === 'hemispheric_or_pantropical')) {
            if (!r.bbox || r.bbox.length !== 4) {
              ok = effNullSafe;
            } else {
              let [bw, bs, be, bn] = r.bbox;
              if (Number.isFinite(bw) && Number.isFinite(bs) && Number.isFinite(be) && Number.isFinite(bn)) {
                bs = Math.max(-90.0, Math.min(90.0, bs));
                bn = Math.max(-90.0, Math.min(90.0, bn));
                const fullGlobe = be - bw >= 359.5;
                bw = normLon(bw);
                be = normLon(be);
                const lonOk = fullGlobe || (bw <= be ? (be >= bwMin && bw <= beMax) : (beMax >= bw || bwMin <= be));
                ok = lonOk && bn >= bsMin && bs <= bnMax;
              } else {
                ok = effNullSafe;
              }
            }
          }
          if (!ok) continue;
        }
      }
    }

    if (coversYear !== null) {
      const sy = r.t0;
      const ey = r.t1;
      const hasSy = sy !== null && sy !== undefined && Number.isFinite(sy);
      const hasEy = ey !== null && ey !== undefined && Number.isFinite(ey);
      if (!hasSy) continue;
      if (sy > coversYear) continue;
      const effEnd = hasEy ? ey : ((f & 8) !== 0 && !hardLimits ? DISCOVERY_CURRENT_YEAR : sy);
      if (effEnd < coversYear) continue;
    }

    if (minEndYear !== null) {
      const ey = r.t1;
      const hasEy = ey !== null && ey !== undefined && Number.isFinite(ey);
      if (hardLimits) {
        if (!hasEy || ey < minEndYear) continue;
      } else {
        const ongoingOk = (f & 8) !== 0 && minEndYear <= DISCOVERY_CURRENT_YEAR;
        if (!ongoingOk) {
          if (!hasEy) {
            if (!effNullSafe) continue;
          } else if (ey < minEndYear) {
            continue;
          }
        }
      }
    }

    if (maxStartYear !== null) {
      const sy = r.t0;
      const hasSy = sy !== null && sy !== undefined && Number.isFinite(sy);
      if (!hasSy) {
        if (!effNullSafe) continue;
      } else if (sy > maxStartYear) {
        continue;
      }
    }

    mask[i] = 1;
    eligibleCount++;
  }

  return { mask, eligibleCount };
}

export function rrfFuse(denseRanked, bm25Ranked, opts = {}) {
  const { rrfK = 60, wDense = 1.0, wBm25 = 0.6 } = opts;
  const scoreMap = new Map();
  const metaMap = new Map();

  for (let i = 0; i < denseRanked.length; i++) {
    const [docIdx, dScore] = denseRanked[i];
    const rank = i + 1;
    scoreMap.set(docIdx, (scoreMap.get(docIdx) || 0.0) + wDense / (rrfK + rank));
    let m = metaMap.get(docIdx);
    if (!m) {
      m = {};
      metaMap.set(docIdx, m);
    }
    m.dense_rank = rank;
    m.dense_score = Math.round(dScore * 1e5) / 1e5;
  }

  for (let i = 0; i < bm25Ranked.length; i++) {
    const [docIdx, bScore] = bm25Ranked[i];
    const rank = i + 1;
    scoreMap.set(docIdx, (scoreMap.get(docIdx) || 0.0) + wBm25 / (rrfK + rank));
    let m = metaMap.get(docIdx);
    if (!m) {
      m = {};
      metaMap.set(docIdx, m);
    }
    m.bm25_rank = rank;
    m.bm25_score = Math.round(bScore * 1e5) / 1e5;
  }

  const fused = Array.from(scoreMap.entries());
  fused.sort((a, bItem) => {
    if (bItem[1] !== a[1]) return bItem[1] - a[1];
    return a[0] - bItem[0];
  });

  return fused.map(([docIdx, score]) => [docIdx, score, metaMap.get(docIdx)]);
}

const Q_RES_RE = /\b(\d+(?:\.\d+)?)\s*(?:-\s*)?(meter|meters|m|kilometer|kilometers|km|deg|degree|degrees)\b/gi;
const Q_CAD_MAP = [
  [/\b(?:hourly|sub-?hourly|sub-?daily|\d+-?hourly)\b/i, 'hourly'],
  [/\bdaily\b/i, 'daily'],
  [/\b(?:weekly|8-?day|10-?daily|16-?day|pentad|dekadal|semi-?monthly|half-?monthly)\b/i, 'multi_day'],
  [/\b(?:monthly|seasonal)\b/i, 'monthly'],
  [/\b(?:annual|yearly|multi-?year|climatolog)\b/i, 'annual'],
];
const Q_GEO_MAP = [
  [/\bglobal\b/i, 'global'],
  [/\b(?:europe|european|pan-?european)\b/i, 'europe'],
  [/\bafrica\b/i, 'africa'],
  [/\bnorth\s+america\b/i, 'north_america'],
];

function extractQueryVariantPrefs(queryText) {
  if (!queryText) return { resTargets: [], cadPrefs: [], geoPrefs: [] };
  const resTargets = [];
  Q_RES_RE.lastIndex = 0;
  let m;
  while ((m = Q_RES_RE.exec(queryText)) !== null) {
    let v = Number(m[1]);
    const u = m[2].toLowerCase();
    if (u === 'km' || u === 'kilometer' || u === 'kilometers') v *= 1000.0;
    else if (u === 'deg' || u === 'degree' || u === 'degrees') v *= 111320.0;
    if (v > 0) resTargets.push(v);
  }
  const cadPrefs = [];
  for (const [pat, lbl] of Q_CAD_MAP) {
    if (pat.test(queryText)) cadPrefs.push(lbl);
  }
  const geoPrefs = [];
  for (const [pat, lbl] of Q_GEO_MAP) {
    if (pat.test(queryText)) geoPrefs.push(lbl);
  }
  return { resTargets, cadPrefs, geoPrefs };
}

function variantQueryMatchScore(prefs, title, cad, gridM, geo) {
  let score = 0;
  const tLow = (title || '').toLowerCase();
  for (const targetM of prefs.resTargets) {
    if (gridM !== null && gridM !== undefined && Number.isFinite(gridM) && gridM > 0) {
      if (Math.max(gridM, targetM) / Math.min(gridM, targetM) <= 1.25) score += 2;
    }
  }
  for (const cp of prefs.cadPrefs) {
    if (
      (cp === 'hourly' && cad.includes('hourly')) ||
      (cp === 'daily' && cad.includes('daily')) ||
      (cp === 'multi_day' && (cad.includes('multi_day') || cad.includes('weekly'))) ||
      (cp === 'monthly' && cad.includes('monthly')) ||
      (cp === 'annual' && (cad.includes('annual') || cad.includes('climatolog') || cad.includes('multi_year'))) ||
      new RegExp(`\\b${cp}\\b`).test(tLow)
    ) {
      score += 2;
    }
  }
  for (const gp of prefs.geoPrefs) {
    if (
      (gp === 'global' && geo === 'global') ||
      (gp === 'europe' && (geo === 'europe_regional' || tLow.includes('europe'))) ||
      tLow.includes(gp)
    ) {
      score += 2;
    }
  }
  return score;
}

export function collapseProductFamilies(rankedItems, records, searchMeta, opts = {}) {
  const {
    topK = 20,
    embStore = null,
    nearDupCos = 0.94,
    sensorGuard = true,
    queryText = '',
  } = opts;

  const { sig_table, sig_ids, role_ids, cad_table, cad_ids, vtag_masks, sensor_masks, adj_grid_m } = searchMeta;
  const kept = [];
  const seenSigToPositions = new Map();
  const scanLimit = Math.min(rankedItems.length, Math.max(topK * 3, 200));

  function pairConflicts(cpos, curSensors, curDoi, curCadId, curGeo, curGrid, curRoleId, curVtagMask) {
    const kIdx = kept[cpos][0];
    if (curRoleId !== role_ids[kIdx]) return true;
    const kVtag = vtag_masks[kIdx];
    if (curVtagMask !== 0 && kVtag !== 0 && (curVtagMask & kVtag) === 0) return true;
    if (sensorGuard) {
      const kSens = sensor_masks[kIdx];
      if (curSensors !== 0 && kSens !== 0 && (curSensors & kSens) === 0) return true;
    }
    const kDoi = (records[kIdx].doi || '').toLowerCase().trim();
    if (curDoi && kDoi && curDoi !== kDoi) return true;
    const kCadId = cad_ids[kIdx];
    const curCadStr = cad_table[curCadId];
    const kCadStr = cad_table[kCadId];
    if (curCadStr && kCadStr && curCadId !== kCadId) return true;
    const kGeo = records[kIdx].geographic_coverage_class || '';
    if (curGeo && kGeo && curGeo !== kGeo) return true;
    const kGrid = adj_grid_m[kIdx];
    if (
      curGrid !== null && curGrid !== undefined && Number.isFinite(curGrid) && curGrid > 0 &&
      kGrid !== null && kGrid !== undefined && Number.isFinite(kGrid) && kGrid > 0
    ) {
      if (Math.max(curGrid, kGrid) / Math.min(curGrid, kGrid) > 2.0) return true;
    }
    return false;
  }

  for (let posI = 0; posI < rankedItems.length; posI++) {
    const [docIdx, score, meta] = rankedItems[posI];
    const sigId = sig_ids[docIdx];
    const curSensors = sensor_masks[docIdx];
    const curDoi = (records[docIdx].doi || '').toLowerCase().trim();
    const curRoleId = role_ids[docIdx];
    const curVtagMask = vtag_masks[docIdx];
    const curCadId = cad_ids[docIdx];
    const curGeo = records[docIdx].geographic_coverage_class || '';
    const curGrid = adj_grid_m[docIdx];

    let matchedPos = -1;

    const sigPositions = seenSigToPositions.get(sigId);
    if (sigPositions) {
      for (let j = 0; j < sigPositions.length; j++) {
        const cpos = sigPositions[j];
        if (pairConflicts(cpos, curSensors, curDoi, curCadId, curGeo, curGrid, curRoleId, curVtagMask)) continue;
        matchedPos = cpos;
        break;
      }
    }

    if (matchedPos === -1 && embStore && kept.length > 0 && nearDupCos < 1.0) {
      const above = [];
      for (let cpos = 0; cpos < kept.length; cpos++) {
        const sim = cosineEmbInt8(embStore, kept[cpos][0], docIdx);
        if (sim >= nearDupCos) above.push([cpos, sim]);
      }
      if (above.length > 0) {
        above.sort((a, bItem) => (bItem[1] !== a[1] ? bItem[1] - a[1] : a[0] - bItem[0]));
        for (let j = 0; j < above.length; j++) {
          const cpos = above[j][0];
          if (pairConflicts(cpos, curSensors, curDoi, curCadId, curGeo, curGrid, curRoleId, curVtagMask)) continue;
          matchedPos = cpos;
          break;
        }
      }
    }

    if (matchedPos !== -1) {
      const repMeta = kept[matchedPos][2];
      repMeta.collapsed_siblings.push({
        record_id: records[docIdx].record_id,
        source: records[docIdx].source_catalogue,
        title: records[docIdx].title,
        doc_idx: docIdx,
      });
      repMeta.sibling_count = repMeta.collapsed_siblings.length;
      if (kept.length >= topK && posI + 1 >= scanLimit) break;
      continue;
    }

    if (kept.length >= topK) {
      if (posI + 1 >= scanLimit) break;
      continue;
    }

    const newMeta = {
      ...meta,
      family_signature: sig_table[sigId],
      sibling_count: 0,
      collapsed_siblings: [],
    };
    const newPos = kept.length;
    if (!sigPositions) seenSigToPositions.set(sigId, [newPos]);
    else sigPositions.push(newPos);
    kept.push([docIdx, score, newMeta]);
  }

  const prefs = extractQueryVariantPrefs(queryText);
  if (prefs.resTargets.length || prefs.cadPrefs.length || prefs.geoPrefs.length) {
    for (let i = 0; i < kept.length; i++) {
      const [repIdx, repSc, repMeta] = kept[i];
      const sibs = repMeta.collapsed_siblings;
      if (!sibs || sibs.length === 0) continue;
      const repQ = variantQueryMatchScore(
        prefs,
        records[repIdx].title,
        cad_table[cad_ids[repIdx]],
        adj_grid_m[repIdx],
        records[repIdx].geographic_coverage_class || '',
      );
      let bestSibPos = -1;
      let bestQ = repQ;
      for (let sPos = 0; sPos < sibs.length; sPos++) {
        const sIdx = sibs[sPos].doc_idx;
        const sq = variantQueryMatchScore(
          prefs,
          records[sIdx].title,
          cad_table[cad_ids[sIdx]],
          adj_grid_m[sIdx],
          records[sIdx].geographic_coverage_class || '',
        );
        if (sq > bestQ) {
          bestQ = sq;
          bestSibPos = sPos;
        }
      }
      if (bestSibPos >= 0) {
        const chosenIdx = sibs[bestSibPos].doc_idx;
        const newSibs = [
          {
            record_id: records[repIdx].record_id,
            source: records[repIdx].source_catalogue,
            title: records[repIdx].title,
            doc_idx: repIdx,
          },
          ...sibs.filter((_, j) => j !== bestSibPos),
        ];
        kept[i] = [
          chosenIdx,
          repSc,
          {
            ...repMeta,
            family_signature: sig_table[sig_ids[chosenIdx]],
            collapsed_siblings: newSibs,
            sibling_count: newSibs.length,
          },
        ];
      }
    }
  }

  return kept.slice(0, topK);
}

export function blendRrfAndRerank(candidates, rerankScores, opts = {}) {
  const { rrfK = 30, wRrf = 0.4, wRerank = 1.0 } = opts;
  const order = candidates.map((_, idx) => idx);
  order.sort((a, bItem) => {
    const diff = rerankScores[bItem] - rerankScores[a];
    if (diff !== 0) return diff;
    return a - bItem;
  });
  const rerankRank = new Int32Array(candidates.length);
  for (let rk = 0; rk < order.length; rk++) {
    rerankRank[order[rk]] = rk + 1;
  }
  const scored = candidates.map((docIdx, stage1Pos) => ({
    docIdx,
    stage1Pos,
    rerankRank: rerankRank[stage1Pos],
    rerankScore: rerankScores[stage1Pos],
    blendedScore: wRrf / (rrfK + stage1Pos + 1) + wRerank / (rrfK + rerankRank[stage1Pos]),
  }));
  scored.sort((a, bItem) => {
    if (bItem.blendedScore !== a.blendedScore) return bItem.blendedScore - a.blendedScore;
    return a.stage1Pos - bItem.stage1Pos;
  });
  return scored;
}

export function formatResolutionHuman(meters, rawVal = null, rawUnits = null) {
  if (meters !== null && meters !== undefined && Number.isFinite(Number(meters))) {
    const m = Number(meters);
    if (m > 0) {
      if (m < 1000) return m < 10 ? `${Number(m.toPrecision(3))} m` : `${Math.round(m)} m`;
      const km = m / 1000.0;
      return km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`;
    }
  }
  if (rawVal) return `${rawVal} ${rawUnits || ''}`.trim();
  return '';
}

export function formatRerankerDoc(record, detail = null, maxChars = 1200) {
  const title = translateSubscripts(record.title || '').replace(/\s+/g, ' ').trim() || 'none';
  const bits = [];
  if (record.raw_id && record.raw_id.length >= 3 && record.raw_id.length <= 64 && !title.toLowerCase().includes(record.raw_id.toLowerCase())) {
    bits.push(`ID: ${record.raw_id}`);
  }
  const inst = record.producing_institution || '';
  const cop = record.copernicus_service || '';
  if (cop) {
    const extra = inst && !inst.toLowerCase().includes(cop.toLowerCase()) ? ` (${inst.slice(0, 60)})` : '';
    bits.push(`Provider: Copernicus ${cop}${extra}`);
  } else if (inst) {
    bits.push(`Provider: ${inst.slice(0, 80)}`);
  }
  const ont = ONTOLOGY_LABELS[record.product_ontology_class] || '';
  const lvl = record.processing_level || '';
  if (ont && lvl) bits.push(`Type: ${lvl} ${ont}`);
  else if (ont) bits.push(`Type: ${ont}`);
  else if (lvl) bits.push(`Level: ${lvl}`);

  const geo = GEO_LABELS_DOC[record.geographic_coverage_class] || '';
  if (geo) bits.push(`Coverage: ${geo}`);

  const resStr = formatResolutionHuman(
    detail?.adjudicated_output_grid_meters ?? record.grid_m,
    detail?.adjudicated_spatial_grid_value,
    detail?.adjudicated_spatial_grid_units,
  );
  if (resStr) bits.push(`Grid: ${resStr}`);

  const cad = CADENCE_LABELS_DOC[record.temporal_cadence_class] || '';
  if (cad) bits.push(`Cadence: ${cad}`);

  if (record.t0 && record.is_ongoing === 1) bits.push(`Period: ${record.t0}-present`);
  else if (record.t0 && record.t1) bits.push(`Period: ${record.t0}-${record.t1}`);
  else if (record.t0) bits.push(`Start: ${record.t0}`);

  const sensors = (detail?.platform_sensor_raw || '').slice(0, 200).trim();
  if (sensors) bits.push(`Sensors: ${sensors}`);

  const kw = (detail?.keywords || record.kw_snippet || '').slice(0, 260).trim();
  if (kw) bits.push(`Topics: ${kw}`);

  const rawDesc = (detail?.description || record.desc_snippet || '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/https?:\/\/\S+/gi, '')
    .replace(/\s+/g, ' ')
    .trim();

  const header = bits.join(' | ');
  const body = header && rawDesc ? `${header}. ${rawDesc}` : header || rawDesc || title;
  return `${title}\n${body}`.slice(0, maxChars);
}

export const QWEN3_PROMPT_PREFIX =
  '<|im_start|>system\n' +
  'Judge whether the Document meets the requirements based on the Query and the Instruct provided. ' +
  'Note that the answer can only be "yes" or "no".<|im_end|>\n' +
  '<|im_start|>user\n';

export const QWEN3_PROMPT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';

export function buildQwen3Body(query, doc, instruction = EO_RERANK_INSTRUCTION) {
  return `<Instruct>: ${instruction}\n<Query>: ${query}\n<Document>: ${doc}`;
}
