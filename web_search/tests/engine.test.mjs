import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';
import {
  applyFiltersMask,
  assembleEmbStore,
  blendRrfAndRerank,
  collapseProductFamilies,
  cosineEmbInt8,
  normalizeEoText,
  parseBm25Binary,
  parseEmbChunkBinary,
  rrfFuse,
  scoreDenseInt8,
  searchBm25,
} from '../src/search-core.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const docsDataDir = path.join(repoRoot, 'docs', 'data');
const searchDataDir = path.join(repoRoot, 'dist', 'data', 'search');

function readGunzip(filePath) {
  const gz = fs.readFileSync(filePath);
  const raw = zlib.gunzipSync(gz);
  return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
}

function loadCatalogueRecords() {
  const catMeta = JSON.parse(fs.readFileSync(path.join(docsDataDir, 'index_meta.json'), 'utf8'));
  const enums = catMeta.enums;
  const decode = (col, idx) => (idx === null || idx === undefined || idx < 0 ? null : enums[col]?.[idx] ?? null);
  const records = [];
  for (const partName of catMeta.meta.index_parts) {
    const part = JSON.parse(fs.readFileSync(path.join(docsDataDir, partName), 'utf8'));
    for (const row of part) {
      const src = decode('source_catalogue', row[7]) || '';
      const rawId = row[2] || '';
      const recordId = row[1] ? row[1] : `${src}::${rawId}`;
      records.push({
        idx: row[0],
        record_id: recordId,
        raw_id: rawId,
        title: row[3] || '',
        desc_snippet: row[4] || '',
        kw_snippet: row[6] || '',
        source_catalogue: src,
        producing_institution: decode('producing_institution', row[8]),
        provider_pool: decode('provider_pool', row[10]),
        copernicus_service: decode('copernicus_service', row[12]),
        processing_level: decode('processing_level', row[13]),
        product_ontology_class: decode('product_ontology_class', row[14]),
        strict_eo: Boolean(row[16]),
        uses_sentinel: Boolean(row[17]),
        grid_m: row[20],
        temporal_cadence_class: decode('temporal_cadence_class', row[21]),
        t0: row[22],
        t1: row[23],
        is_ongoing: row[24],
        geographic_coverage_class: decode('geographic_coverage_class', row[26]),
        observation_domain_class: decode('observation_domain_class', row[27]),
        doi: row[29],
        bbox: row[32],
      });
    }
  }
  return records;
}

// 1. Text normalization tests
assert.equal(normalizeEoText('CO₂_flux & CH₄ burnt colour'), 'co2 carbon dioxide flux & ch4 methane burned color');
assert.equal(normalizeEoText('chlorophyll-a sea-surface temperature'), 'chlorophyll a chla sea-surface temperature');
assert.equal(normalizeEoText('sea surface temperature PM2.5'), 'sea surface temperature sst pm25 pm2.5 fine particulate');

if (!fs.existsSync(path.join(searchDataDir, 'search_manifest.json'))) {
  console.log('[engine.test] dist/data/search not built yet — unit checks passed, skipping full index assertions.');
  process.exit(0);
}

const manifest = JSON.parse(fs.readFileSync(path.join(searchDataDir, 'search_manifest.json'), 'utf8'));
const records = loadCatalogueRecords();
const recordIds = records.map((r) => r.record_id);
const bm25 = parseBm25Binary(readGunzip(path.join(searchDataDir, manifest.bm25.file)));
const searchMeta = JSON.parse(Buffer.from(readGunzip(path.join(searchDataDir, manifest.search_meta.file))).toString('utf8'));
const embChunks = manifest.embeddings.map((p) => parseEmbChunkBinary(readGunzip(path.join(searchDataDir, p.file))));
const embStore = assembleEmbStore(embChunks, manifest.n_docs, manifest.emb_dim);

assert.equal(records.length, 39472);
assert.equal(bm25.nDocs, 39472);
assert.equal(searchMeta.n_docs, 39472);
assert.equal(embStore.dim, 512);

// Self-cosine check on int8 vectors
for (const idx of [0, 100, 10000, 39471]) {
  const c = cosineEmbInt8(embStore, idx, idx);
  assert.ok(c > 0.99 && c < 1.01, `Unexpected self-cosine at ${idx}: ${c}`);
}

// 2. Active count check
const activeRes = applyFiltersMask(records, searchMeta, { exclude_superseded: true });
assert.equal(activeRes.eligibleCount, manifest.active_docs);

// 3. Hard limits verified checks
const futureRes = applyFiltersMask(records, searchMeta, { min_end_year: 2035, hard_limits: true });
for (let i = 0; i < records.length; i++) {
  if (futureRes.mask[i]) {
    assert.ok(records[i].t1 !== null && records[i].t1 >= 2035, `Hard min_end_year leaked ${records[i].record_id} (t1=${records[i].t1})`);
  }
}

const tellusRes = applyFiltersMask(records, searchMeta, {
  geo_scope: 'global',
  processing_levels: ['L4'],
  max_grid_m: 60000,
  hard_limits: true,
});
for (let i = 0; i < records.length; i++) {
  if (tellusRes.mask[i]) {
    assert.ok(!records[i].title.includes('Tellus Level-4 Greenland Mass Anomaly Time Series'));
    assert.equal(records[i].geographic_coverage_class, 'global');
    assert.ok(searchMeta.adj_grid_m[i] !== null && searchMeta.adj_grid_m[i] <= 60000);
    assert.ok((searchMeta.flags[i] & 2) !== 0, 'Must have verified output grid source');
    assert.ok((searchMeta.flags[i] & 4) === 0, 'Must not have spatial conflict');
  }
}

// 4. BM25 + dense + RRF + collapse smoke check
const bm25Hits = searchBm25(bm25, 'Global 30 meter inland surface water occurrence seasonality', activeRes.mask, recordIds, { topK: 50 });
assert.ok(bm25Hits.length >= 20);
// Query with synthetic vector taken from hit #0 to verify dense scoring & RRF & collapse
const qVec = new Float32Array(512);
const base = bm25Hits[0][0] * 512;
for (let d = 0; d < 512; d++) qVec[d] = embStore.quant[base + d] * embStore.scales[bm25Hits[0][0]];
const denseHits = scoreDenseInt8(embStore, qVec, activeRes.mask, 100);
assert.equal(denseHits[0][0], bm25Hits[0][0]);

const fused = rrfFuse(denseHits, bm25Hits);
assert.ok(fused.length >= 20);

const collapsed = collapseProductFamilies(fused, records, searchMeta, {
  topK: 20,
  embStore,
  queryText: 'Global 30 meter inland surface water occurrence seasonality',
});
assert.equal(collapsed.length, 20);

// 5. Blend rerank determinism
const blended = blendRrfAndRerank([10, 20, 30], [0.5, 5.0, 2.0]);
assert.equal(blended[0].docIdx, 20);
assert.equal(blended[1].docIdx, 30);
assert.equal(blended[2].docIdx, 10);

console.log(`[engine.test] ALL ASSERTIONS PASSED (active=${activeRes.eligibleCount}, bm25_top1=${records[bm25Hits[0][0]].record_id})`);
