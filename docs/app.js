'use strict';

/**
 * Client-side static catalogue explorer for the Global Harmonized L3/L4 EO Product Catalogue.
 * Loads sharded compact JSON (`data/index_meta.json` + `data/index_part_*.json`) and lazy-fetches
 * full record provenance (`data/details/chunk_*.json`) only when inspecting a product drawer.
 */

const state = {
  meta: null,
  enums: null,
  records: [],
  filtered: [],
  sortCol: 'default',
  sortDir: 'asc',
  page: 1,
  pageSize: 50,
  moreOpen: false,
  activeRecordId: null,
  activeRecordIdx: null,
  chunkCache: new Map(),
  map: null,
  mapResizeTimer: null,
  searchTimer: null,
  yearTimer: null,
  exporting: false,
  loading: true,
};

let parquetWriterPromise = null;

const EXPORT_COLUMNS = [
  ['record_id', 'STRING'],
  ['raw_id', 'STRING'],
  ['title', 'STRING'],
  ['source_catalogue', 'STRING'],
  ['producing_institution', 'STRING'],
  ['hosting_platform', 'STRING'],
  ['provider_pool', 'STRING'],
  ['provider_jurisdiction_tier', 'STRING'],
  ['copernicus_service', 'STRING'],
  ['processing_level', 'STRING'],
  ['product_ontology_class', 'STRING'],
  ['product_output_category', 'STRING'],
  ['strict_eo', 'BOOLEAN'],
  ['uses_sentinel', 'BOOLEAN'],
  ['sentinel_missions', 'STRING'],
  ['sentinel_lineage_status', 'STRING'],
  ['grid_m', 'DOUBLE'],
  ['temporal_cadence_class', 'STRING'],
  ['t0', 'INT32'],
  ['t1', 'INT32'],
  ['operational_maturity_class', 'STRING'],
  ['geographic_coverage_class', 'STRING'],
  ['observation_domain_class', 'STRING'],
  ['doi', 'STRING'],
  ['license', 'STRING'],
  ['access_url', 'STRING'],
];

const POOL_META = {
  copernicus_services: { label: 'Copernicus', cls: 'bg-indigo-50 text-indigo-700 ring-indigo-600/20' },
  eu_non_copernicus: { label: 'EU non-Copernicus', cls: 'bg-sky-50 text-sky-700 ring-sky-600/20' },
  non_eu: { label: 'Non-EU', cls: 'bg-slate-100 text-slate-700 ring-slate-500/20' },
};

const MATURITY_META = {
  operational_ongoing_series: { label: 'Ongoing', cls: 'bg-emerald-50 text-emerald-700 ring-emerald-600/20' },
  completed_or_mission_bounded_release: { label: 'Completed', cls: 'bg-slate-100 text-slate-700 ring-slate-500/20' },
  research_or_benchmark_release: { label: 'Research', cls: 'bg-violet-50 text-violet-700 ring-violet-600/20' },
  planned_or_preoperational: { label: 'Planned', cls: 'bg-amber-50 text-amber-700 ring-amber-600/20' },
  deprecated_or_superseded: { label: 'Superseded', cls: 'bg-rose-50 text-rose-700 ring-rose-600/20' },
  unknown: { label: 'Unknown', cls: 'bg-slate-50 text-slate-500 ring-slate-400/20' },
};

const GEO_LABELS = {
  global: 'Global',
  quasi_global: 'Quasi-Global',
  hemispheric_or_pantropical: 'Hemispheric',
  europe_regional: 'Europe',
  regional_or_local: 'Regional / Local',
};

const LEVEL_ORDER = { L1: 1, L2: 2, L3: 3, 'L3; L4': 4, L4: 5 };

const dom = {
  metaBadge: document.getElementById('meta-badge'),
  progressWrap: document.getElementById('load-progress-wrap'),
  progressBar: document.getElementById('load-progress-bar'),
  errorBanner: document.getElementById('error-banner'),
  statMatched: document.getElementById('stat-matched'),
  statMatchedPct: document.getElementById('stat-matched-pct'),
  statSentinel: document.getElementById('stat-sentinel'),
  statSentinelPct: document.getElementById('stat-sentinel-pct'),
  statPools: document.getElementById('stat-pools'),
  statDoi: document.getElementById('stat-doi'),
  statDoiPct: document.getElementById('stat-doi-pct'),

  search: document.getElementById('filter-search'),
  btnClearSearch: document.getElementById('btn-clear-search'),
  pool: document.getElementById('filter-pool'),
  level: document.getElementById('filter-level'),
  sentinel: document.getElementById('filter-sentinel'),
  geo: document.getElementById('filter-geo'),
  strictEo: document.getElementById('filter-strict-eo'),
  hasDoi: document.getElementById('filter-has-doi'),
  source: document.getElementById('filter-source'),
  tier: document.getElementById('filter-tier'),
  category: document.getElementById('filter-category'),
  cadence: document.getElementById('filter-cadence'),
  maturity: document.getElementById('filter-maturity'),
  ontology: document.getElementById('filter-ontology'),
  grid: document.getElementById('filter-grid'),
  year: document.getElementById('filter-year'),
  hasUrl: document.getElementById('filter-has-url'),
  exclDeprecated: document.getElementById('filter-exclude-deprecated'),

  btnToggleMore: document.getElementById('btn-toggle-more'),
  morePanel: document.getElementById('more-filters-panel'),
  moreBadge: document.getElementById('more-filters-badge'),
  activeChipsRow: document.getElementById('active-chips-row'),
  activeChips: document.getElementById('active-chips'),
  btnResetAll: document.getElementById('btn-reset-all'),
  presetBtns: Array.from(document.querySelectorAll('.preset-btn')),

  tableShowing: document.getElementById('table-showing'),
  tableShowingBottom: document.getElementById('table-showing-bottom'),
  tableBody: document.getElementById('table-body'),
  pageSize: document.getElementById('page-size'),
  paginationTop: document.getElementById('pagination-top'),
  paginationBottom: document.getElementById('pagination-bottom'),
  sortThs: Array.from(document.querySelectorAll('.sort-th')),
  downloadWrap: document.getElementById('download-wrap'),
  btnDownload: document.getElementById('btn-download'),
  btnDownloadLabel: document.getElementById('btn-download-label'),
  downloadMenu: document.getElementById('download-menu'),
  downloadOpts: Array.from(document.querySelectorAll('.download-opt')),

  drawerBackdrop: document.getElementById('drawer-backdrop'),
  drawer: document.getElementById('detail-drawer'),
  drawerBadges: document.getElementById('drawer-badges'),
  drawerTitle: document.getElementById('drawer-title'),
  drawerSubtitle: document.getElementById('drawer-subtitle'),
  drawerBody: document.getElementById('drawer-body'),
  drawerClose: document.getElementById('drawer-close'),
  drawerPrev: document.getElementById('drawer-prev'),
  drawerNext: document.getElementById('drawer-next'),

  btnInfo: document.getElementById('btn-info'),
  btnShareUrl: document.getElementById('btn-share-url'),
  btnShareLabel: document.getElementById('btn-share-label'),
  infoModal: document.getElementById('info-modal'),
  infoClose: document.getElementById('info-close'),
  infoOk: document.getElementById('info-ok'),
};

function esc(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function safeHref(url) {
  if (!url || typeof url !== 'string') return null;
  const trimmed = url.trim();
  if (/^(https?|ftp):\/\//i.test(trimmed)) return trimmed;
  return null;
}

function formatEnumLabel(val) {
  if (!val) return '';
  return String(val)
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bEo\b/g, 'EO')
    .replace(/\bEu\b/g, 'EU')
    .replace(/\bEu27\b/g, 'EU-27')
    .replace(/\bIgo\b/g, 'IGO')
    .replace(/\bUn\b/g, 'UN')
    .replace(/\bArd\b/g, 'ARD')
    .replace(/\bEsa\b/g, 'ESA');
}

function formatGridMeters(m) {
  if (m === null || m === undefined || Number.isNaN(m)) return null;
  if (m < 1000) {
    return `${Number.isInteger(m) ? m : m.toFixed(1)} m`;
  }
  const km = m / 1000;
  return `${km >= 100 ? km.toFixed(0) : km >= 10 ? km.toFixed(1).replace(/\.0$/, '') : km.toFixed(2).replace(/\.?0+$/, '')} km`;
}

function formatYears(t0, t1, isOngoing) {
  if (t0 === null && t1 === null) return isOngoing === 1 ? 'Ongoing' : '—';
  const s = t0 !== null ? String(t0) : '?';
  const e = t1 !== null ? String(t1) : (isOngoing === 1 ? 'Present' : '?');
  if (s === e && isOngoing !== 1) return s;
  return `${s}–${e}`;
}

function decodeRow(arr, enums) {
  const e = (col, i) => (i >= 0 && enums[col] && enums[col][i] !== undefined ? enums[col][i] : null);
  const src = e('source_catalogue', arr[7]) || '';
  const rawId = arr[2] || '';
  const recordId = arr[1] || (src ? `${src}::${rawId}` : rawId);
  const title = arr[3] || '';
  const inst = e('producing_institution', arr[8]);
  const desc = arr[4] || '';
  const kw = arr[6] || '';
  const doi = arr[29] || null;
  const searchStr = `${title} ${rawId} ${recordId} ${doi || ''} ${inst || ''} ${src} ${desc} ${kw}`.toLowerCase();

  return {
    idx: arr[0],
    record_id: recordId,
    raw_id: rawId,
    title,
    desc_snippet: desc,
    desc_truncated: arr[5] === 1,
    kw_snippet: kw,
    source_catalogue: src,
    producing_institution: inst,
    hosting_platform: e('hosting_platform', arr[9]),
    provider_pool: e('provider_pool', arr[10]),
    provider_jurisdiction_tier: e('provider_jurisdiction_tier', arr[11]),
    copernicus_service: e('copernicus_service', arr[12]),
    processing_level: e('processing_level', arr[13]),
    product_ontology_class: e('product_ontology_class', arr[14]),
    product_output_category: e('product_output_category', arr[15]),
    strict_eo: arr[16] === 1,
    uses_sentinel: arr[17] === 1,
    sentinel_missions: e('sentinel_missions', arr[18]),
    sentinel_lineage_status: e('sentinel_lineage_status', arr[19]),
    grid_m: arr[20],
    temporal_cadence_class: e('temporal_cadence_class', arr[21]),
    t0: arr[22],
    t1: arr[23],
    is_ongoing: arr[24],
    operational_maturity_class: e('operational_maturity_class', arr[25]),
    geographic_coverage_class: e('geographic_coverage_class', arr[26]),
    observation_domain_class: e('observation_domain_class', arr[27]),
    observation_domain_mask: e('observation_domain_mask', arr[28]),
    doi,
    license: e('license', arr[30]),
    access_url: arr[31] || null,
    bbox: arr[32],
    antimeridian: arr[33] === 1,
    cluster_count: arr[34] || 1,
    _search: searchStr,
  };
}

function populateSelectOptions(selectEl, values, formatter = formatEnumLabel, addNullOption = false) {
  const frag = document.createDocumentFragment();
  values.forEach((v) => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = formatter(v);
    frag.appendChild(opt);
  });
  if (addNullOption) {
    const opt = document.createElement('option');
    opt.value = '__NULL__';
    opt.textContent = 'Unlabelled (null)';
    frag.appendChild(opt);
  }
  selectEl.appendChild(frag);
}

function readFiltersFromUI() {
  return {
    q: dom.search.value.trim(),
    pool: dom.pool.value,
    level: dom.level.value,
    sentinel: dom.sentinel.value,
    geo: dom.geo.value,
    strictEo: dom.strictEo.checked,
    hasDoi: dom.hasDoi.checked,
    source: dom.source.value,
    tier: dom.tier.value,
    category: dom.category.value,
    cadence: dom.cadence.value,
    maturity: dom.maturity.value,
    ontology: dom.ontology.value,
    grid: dom.grid.value,
    year: dom.year.value.trim() ? Number.parseInt(dom.year.value.trim(), 10) : null,
    hasUrl: dom.hasUrl.checked,
    exclDeprecated: dom.exclDeprecated.checked,
  };
}

function writeUrlState(historyMode = 'replace') {
  if (historyMode === 'none') return;
  const f = readFiltersFromUI();
  const params = new URLSearchParams();
  if (f.q) params.set('q', f.q);
  if (f.pool) params.set('pool', f.pool);
  if (f.level) params.set('level', f.level);
  if (f.sentinel) params.set('sentinel', f.sentinel);
  if (f.geo) params.set('geo', f.geo);
  if (f.strictEo) params.set('strict', '1');
  if (f.hasDoi) params.set('doi', '1');
  if (f.source) params.set('source', f.source);
  if (f.tier) params.set('tier', f.tier);
  if (f.category) params.set('cat', f.category);
  if (f.cadence) params.set('cad', f.cadence);
  if (f.maturity) params.set('mat', f.maturity);
  if (f.ontology) params.set('ont', f.ontology);
  if (f.grid) params.set('grid', f.grid);
  if (f.year !== null && !Number.isNaN(f.year)) params.set('year', String(f.year));
  if (f.hasUrl) params.set('url', '1');
  if (f.exclDeprecated) params.set('no_dep', '1');
  if (state.moreOpen) params.set('adv', '1');
  if (state.sortCol !== 'default') {
    params.set('sort', state.sortCol);
    params.set('dir', state.sortDir);
  }
  if (state.page > 1) params.set('page', String(state.page));
  if (state.pageSize !== 50) params.set('ps', String(state.pageSize));
  if (state.activeRecordId) params.set('record', state.activeRecordId);
  const qs = params.toString();
  const target = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  if (target === `${window.location.pathname}${window.location.search}`) return;
  if (historyMode === 'push') window.history.pushState(null, '', target);
  else window.history.replaceState(null, '', target);
}

function hydrateFiltersFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const setVal = (el, k) => { el.value = params.get(k) || ''; };
  setVal(dom.search, 'q');
  setVal(dom.pool, 'pool');
  setVal(dom.level, 'level');
  setVal(dom.sentinel, 'sentinel');
  setVal(dom.geo, 'geo');
  dom.strictEo.checked = params.get('strict') === '1';
  dom.hasDoi.checked = params.get('doi') === '1';
  setVal(dom.source, 'source');
  setVal(dom.tier, 'tier');
  setVal(dom.category, 'cat');
  setVal(dom.cadence, 'cad');
  setVal(dom.maturity, 'mat');
  setVal(dom.ontology, 'ont');
  setVal(dom.grid, 'grid');
  setVal(dom.year, 'year');
  dom.hasUrl.checked = params.get('url') === '1';
  dom.exclDeprecated.checked = params.get('no_dep') === '1';
  state.sortCol = params.get('sort') || 'default';
  state.sortDir = params.get('dir') === 'desc' ? 'desc' : 'asc';
  const psParam = params.get('ps');
  state.pageSize = psParam && ['25', '50', '100'].includes(psParam) ? Number(psParam) : 50;
  dom.pageSize.value = String(state.pageSize);
  const pageParam = Number.parseInt(params.get('page') || '1', 10);
  state.page = pageParam > 0 ? pageParam : 1;
  state.activeRecordId = params.get('record') || null;
  const hasAdv = params.get('adv') === '1' || ['source', 'tier', 'cat', 'cad', 'mat', 'ont', 'grid', 'year', 'url', 'no_dep'].some((k) => params.has(k));
  setMorePanelOpen(hasAdv, 'none');
}

function setMorePanelOpen(open, historyMode = 'push') {
  state.moreOpen = Boolean(open);
  dom.morePanel.classList.toggle('hidden', !state.moreOpen);
  dom.btnToggleMore.setAttribute('aria-expanded', String(state.moreOpen));
  if (!state.loading && historyMode !== 'none') writeUrlState(historyMode);
}

function matchesGridFilter(gridM, rule) {
  if (!rule) return true;
  if (rule === '__NULL__') return gridM === null;
  if (gridM === null) return false;
  switch (rule) {
    case 'le10': return gridM <= 10;
    case 'le30': return gridM <= 30;
    case 'le250': return gridM <= 250;
    case 'le1000': return gridM <= 1000;
    case 'le10000': return gridM <= 10000;
    case 'le50000': return gridM <= 50000;
    case 'gt50000': return gridM > 50000;
    default: return true;
  }
}

function applyFiltersAndSort(resetPage = true, historyMode = 'push') {
  if (state.loading) return;
  clearTimeout(state.searchTimer);
  clearTimeout(state.yearTimer);
  const f = readFiltersFromUI();
  dom.btnClearSearch.classList.toggle('hidden', !f.q);

  const qTerms = f.q ? f.q.toLowerCase().split(/\s+/).filter(Boolean) : [];
  const sentinelMission = f.sentinel.startsWith('mission:') ? f.sentinel.slice(8) : null;

  const out = [];
  const recs = state.records;
  for (let i = 0; i < recs.length; i++) {
    const r = recs[i];
    if (f.pool && r.provider_pool !== f.pool) continue;
    if (f.level) {
      if (f.level === '__NULL__') {
        if (r.processing_level !== null) continue;
      } else if (r.processing_level !== f.level) continue;
    }
    if (f.sentinel) {
      if (f.sentinel === 'yes') {
        if (!r.uses_sentinel) continue;
      } else if (f.sentinel === 'no_explicit') {
        if (r.uses_sentinel || r.sentinel_lineage_status !== 'explicit_non_sentinel') continue;
      } else if (f.sentinel === 'unassessed') {
        if (r.sentinel_lineage_status !== 'unassessed_no_platform_metadata') continue;
      } else if (sentinelMission) {
        if (!r.uses_sentinel || !r.sentinel_missions) continue;
        const toks = r.sentinel_missions.split(/;\s*/);
        if (!toks.includes(sentinelMission)) continue;
      }
    }
    if (f.geo) {
      if (f.geo === '__NULL__') {
        if (r.geographic_coverage_class !== null) continue;
      } else if (r.geographic_coverage_class !== f.geo) continue;
    }
    if (f.strictEo && !r.strict_eo) continue;
    if (f.hasDoi && !r.doi) continue;
    if (f.source && r.source_catalogue !== f.source) continue;
    if (f.tier && r.provider_jurisdiction_tier !== f.tier) continue;
    if (f.category && r.product_output_category !== f.category) continue;
    if (f.cadence) {
      if (f.cadence === '__NULL__') {
        if (r.temporal_cadence_class !== null) continue;
      } else if (r.temporal_cadence_class !== f.cadence) continue;
    }
    if (f.maturity && r.operational_maturity_class !== f.maturity) continue;
    if (f.ontology && r.product_ontology_class !== f.ontology) continue;
    if (f.grid && !matchesGridFilter(r.grid_m, f.grid)) continue;
    if (f.year !== null && !Number.isNaN(f.year)) {
      if (r.t0 === null) continue;
      if (r.t0 > f.year) continue;
      const effEnd = r.t1 !== null ? r.t1 : (r.is_ongoing === 1 ? 2026 : r.t0);
      if (effEnd < f.year) continue;
    }
    if (f.hasUrl && !r.access_url) continue;
    if (f.exclDeprecated && r.operational_maturity_class === 'deprecated_or_superseded') continue;

    if (qTerms.length > 0) {
      let ok = true;
      for (let t = 0; t < qTerms.length; t++) {
        if (!r._search.includes(qTerms[t])) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
    }
    out.push(r);
  }

  sortRecordsInPlace(out);
  state.filtered = out;
  if (resetPage) state.page = 1;
  const maxPage = Math.max(1, Math.ceil(out.length / state.pageSize));
  if (state.page > maxPage) state.page = maxPage;

  updateStatsAndChips(f);
  updatePresetButtons(f);
  updateSortHeaders();
  renderTable();
  writeUrlState(historyMode);
}

function cmpNullableStrings(a, b, mult) {
  if (a === b) return 0;
  if (a === null || a === '') return 1;
  if (b === null || b === '') return -1;
  return a.localeCompare(b) * mult;
}

function cmpNullableNums(a, b, mult) {
  if (a === b) return 0;
  if (a === null || Number.isNaN(a)) return 1;
  if (b === null || Number.isNaN(b)) return -1;
  return (a - b) * mult;
}

function sortRecordsInPlace(arr) {
  const col = state.sortCol;
  if (col === 'default') return;
  const mult = state.sortDir === 'desc' ? -1 : 1;
  arr.sort((a, b) => {
    let c = 0;
    switch (col) {
      case 'title':
        c = cmpNullableStrings(a.title, b.title, mult);
        break;
      case 'level':
        c = cmpNullableNums(LEVEL_ORDER[a.processing_level] ?? null, LEVEL_ORDER[b.processing_level] ?? null, mult);
        break;
      case 'pool':
        c = cmpNullableStrings(a.provider_pool, b.provider_pool, mult);
        break;
      case 'sentinel':
        if (a.uses_sentinel !== b.uses_sentinel) {
          c = (a.uses_sentinel ? 0 : 1) - (b.uses_sentinel ? 0 : 1);
          if (mult === -1) c = -c;
        } else {
          c = cmpNullableStrings(a.sentinel_missions, b.sentinel_missions, mult);
        }
        break;
      case 'grid':
        c = cmpNullableNums(a.grid_m, b.grid_m, mult);
        break;
      case 'geo':
        c = cmpNullableStrings(a.geographic_coverage_class, b.geographic_coverage_class, mult);
        break;
      case 't0':
        c = cmpNullableNums(a.t0, b.t0, mult);
        break;
      default:
        c = a.idx - b.idx;
    }
    return c !== 0 ? c : a.idx - b.idx;
  });
}

function updateSortHeaders() {
  dom.sortThs.forEach((th) => {
    const key = th.dataset.sort;
    const ind = th.querySelector('.sort-ind');
    if (key === state.sortCol) {
      th.setAttribute('aria-sort', state.sortDir === 'asc' ? 'ascending' : 'descending');
      ind.textContent = state.sortDir === 'asc' ? '↑' : '↓';
      ind.classList.remove('text-slate-400');
      ind.classList.add('text-indigo-600');
    } else {
      th.removeAttribute('aria-sort');
      ind.textContent = '↕';
      ind.classList.add('text-slate-400');
      ind.classList.remove('text-indigo-600');
    }
  });
}

function updateStatsAndChips(f) {
  const total = state.records.length;
  const rows = state.filtered;
  const n = rows.length;
  let sentCnt = 0;
  let doiCnt = 0;
  let copCnt = 0;
  let euCnt = 0;
  let nonEuCnt = 0;
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    if (r.uses_sentinel) sentCnt++;
    if (r.doi) doiCnt++;
    if (r.provider_pool === 'copernicus_services') copCnt++;
    else if (r.provider_pool === 'eu_non_copernicus') euCnt++;
    else nonEuCnt++;
  }

  dom.statMatched.textContent = n.toLocaleString();
  dom.statMatchedPct.textContent = total ? `${((n / total) * 100).toFixed(1)}% of ${total.toLocaleString()}` : '';
  dom.statSentinel.textContent = sentCnt.toLocaleString();
  dom.statSentinelPct.textContent = n ? `${((sentCnt / n) * 100).toFixed(1)}%` : '0%';
  dom.statDoi.textContent = doiCnt.toLocaleString();
  dom.statDoiPct.textContent = n ? `${((doiCnt / n) * 100).toFixed(1)}%` : '0%';
  dom.statPools.innerHTML =
    `<span class="inline-flex items-center rounded bg-indigo-50 px-1.5 py-0.5 text-[11px] font-medium text-indigo-700">Cop ${copCnt.toLocaleString()}</span>` +
    `<span class="inline-flex items-center rounded bg-sky-50 px-1.5 py-0.5 text-[11px] font-medium text-sky-700">EU ${euCnt.toLocaleString()}</span>` +
    `<span class="inline-flex items-center rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-medium text-slate-700">Non-EU ${nonEuCnt.toLocaleString()}</span>`;

  // Count advanced filters for badge
  const advCount = [
    f.source, f.tier, f.category, f.cadence, f.maturity, f.ontology, f.grid,
    f.year !== null && !Number.isNaN(f.year), f.hasUrl, f.exclDeprecated,
  ].filter(Boolean).length;
  dom.moreBadge.textContent = String(advCount);
  dom.moreBadge.classList.toggle('hidden', advCount === 0);

  // Build active filter chips
  const chips = [];
  const addChip = (label, valText, clearFn) => chips.push({ label, valText, clearFn });
  if (f.q) addChip('Search', `"${f.q}"`, () => { dom.search.value = ''; });
  if (f.pool) addChip('Pool', POOL_META[f.pool]?.label || f.pool, () => { dom.pool.value = ''; });
  if (f.level) addChip('Level', f.level === '__NULL__' ? 'Null' : f.level, () => { dom.level.value = ''; });
  if (f.sentinel) addChip('Sentinel', f.sentinel, () => { dom.sentinel.value = ''; });
  if (f.geo) addChip('Geo', f.geo === '__NULL__' ? 'Null' : (GEO_LABELS[f.geo] || f.geo), () => { dom.geo.value = ''; });
  if (f.strictEo) addChip('Scope', 'Strict EO only', () => { dom.strictEo.checked = false; });
  if (f.hasDoi) addChip('DOI', 'Verified DOI', () => { dom.hasDoi.checked = false; });
  if (f.source) addChip('Source', f.source, () => { dom.source.value = ''; });
  if (f.tier) addChip('Tier', formatEnumLabel(f.tier), () => { dom.tier.value = ''; });
  if (f.category) addChip('Domain', formatEnumLabel(f.category), () => { dom.category.value = ''; });
  if (f.cadence) addChip('Cadence', f.cadence === '__NULL__' ? 'Null' : formatEnumLabel(f.cadence), () => { dom.cadence.value = ''; });
  if (f.maturity) addChip('Maturity', MATURITY_META[f.maturity]?.label || f.maturity, () => { dom.maturity.value = ''; });
  if (f.ontology) addChip('Ontology', formatEnumLabel(f.ontology), () => { dom.ontology.value = ''; });
  if (f.grid) addChip('Grid', dom.grid.selectedOptions[0]?.textContent || f.grid, () => { dom.grid.value = ''; });
  if (f.year !== null && !Number.isNaN(f.year)) addChip('Year', String(f.year), () => { dom.year.value = ''; });
  if (f.hasUrl) addChip('URL', 'Has Access URL', () => { dom.hasUrl.checked = false; });
  if (f.exclDeprecated) addChip('Filter', 'Hide Superseded', () => { dom.exclDeprecated.checked = false; });

  dom.activeChips.innerHTML = '';
  if (chips.length > 0) {
    dom.activeChipsRow.classList.remove('hidden');
    chips.forEach((ch) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'inline-flex items-center gap-1 rounded-md bg-indigo-50 px-2 py-0.5 text-[11px] font-medium text-indigo-700 ring-1 ring-inset ring-indigo-600/20 hover:bg-indigo-100 transition';
      btn.innerHTML = `<span class="text-indigo-500">${esc(ch.label)}:</span><span>${esc(ch.valText)}</span><span aria-hidden="true" class="ml-0.5 text-indigo-400 font-bold">×</span>`;
      btn.addEventListener('click', () => {
        ch.clearFn();
        applyFiltersAndSort(true);
      });
      dom.activeChips.appendChild(btn);
    });
  } else {
    dom.activeChipsRow.classList.add('hidden');
  }
}

function updatePresetButtons(f) {
  dom.presetBtns.forEach((btn) => {
    const p = btn.dataset.preset;
    let active = false;
    if (p === 'non_eu_sentinel') active = f.pool === 'non_eu' && f.sentinel === 'yes';
    else if (p === 'copernicus_l3_l4') active = f.pool === 'copernicus_services';
    else if (p === 'global_ongoing') active = f.geo === 'global' && f.maturity === 'operational_ongoing_series';
    else if (p === 'high_res_eo') active = f.strictEo && f.grid === 'le30';
    btn.dataset.active = active ? 'true' : 'false';
    btn.setAttribute('aria-pressed', String(active));
  });
}

function renderPagination(targetEl, totalRows) {
  targetEl.innerHTML = '';
  const totalPages = Math.max(1, Math.ceil(totalRows / state.pageSize));
  const cur = state.page;
  const mkBtn = (label, page, disabled = false, active = false) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.disabled = disabled;
    b.className = active
      ? 'px-2.5 py-1 rounded-md bg-slate-900 text-white font-semibold tabular-nums'
      : 'px-2.5 py-1 rounded-md bg-white border border-slate-300 text-slate-700 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed tabular-nums transition';
    if (!disabled && !active) {
      b.addEventListener('click', () => {
        state.page = page;
        renderTable();
        writeUrlState('push');
      });
    }
    return b;
  };

  targetEl.appendChild(mkBtn('«', 1, cur === 1));
  targetEl.appendChild(mkBtn('Prev', cur - 1, cur === 1));

  const pages = new Set([1, totalPages, cur - 1, cur, cur + 1].filter((p) => p >= 1 && p <= totalPages));
  const sorted = Array.from(pages).sort((a, b) => a - b);
  let prev = 0;
  sorted.forEach((p) => {
    if (p - prev > 1) {
      const ell = document.createElement('span');
      ell.textContent = '…';
      ell.className = 'px-1 text-slate-400';
      targetEl.appendChild(ell);
    }
    targetEl.appendChild(mkBtn(String(p), p, false, p === cur));
    prev = p;
  });

  targetEl.appendChild(mkBtn('Next', cur + 1, cur >= totalPages));
  targetEl.appendChild(mkBtn('»', totalPages, cur >= totalPages));
}

function renderTable() {
  const total = state.filtered.length;
  const start = (state.page - 1) * state.pageSize;
  const end = Math.min(total, start + state.pageSize);
  const showingText = total === 0
    ? 'No matching canonical products'
    : `Showing ${(start + 1).toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()} matching products`;
  dom.tableShowing.textContent = showingText;
  dom.tableShowingBottom.textContent = showingText;

  renderPagination(dom.paginationTop, total);
  renderPagination(dom.paginationBottom, total);
  if (dom.btnDownload) {
    dom.btnDownload.disabled = total === 0 || state.exporting;
  }

  if (total === 0) {
    dom.tableBody.innerHTML =
      '<tr><td colspan="8" class="px-4 py-12 text-center text-sm text-slate-500">No products match the current filters. <button id="empty-reset-btn" type="button" class="text-indigo-600 underline hover:text-indigo-800 ml-1">Reset all filters</button></td></tr>';
    const emptyReset = document.getElementById('empty-reset-btn');
    if (emptyReset) emptyReset.addEventListener('click', resetAllFilters);
    return;
  }

  const frag = document.createDocumentFragment();
  for (let i = start; i < end; i++) {
    const r = state.filtered[i];
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-indigo-50/40 cursor-pointer transition-colors group';
    tr.dataset.rowIndex = String(i);

    const poolInfo = POOL_META[r.provider_pool] || { label: r.provider_pool || '—', cls: 'bg-slate-100 text-slate-700 ring-slate-500/20' };
    const matInfo = MATURITY_META[r.operational_maturity_class] || { label: r.operational_maturity_class || '—', cls: 'bg-slate-100 text-slate-600 ring-slate-400/20' };
    const gridStr = formatGridMeters(r.grid_m);
    const cadenceStr = r.temporal_cadence_class ? formatEnumLabel(r.temporal_cadence_class) : null;
    const yearsStr = formatYears(r.t0, r.t1, r.is_ongoing);
    const safeUrl = safeHref(r.access_url);
    const safeDoiHref = r.doi ? `https://doi.org/${encodeURI(r.doi)}` : null;

    const sentinelCell = r.uses_sentinel
      ? `<span class="inline-flex items-center rounded-md bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">${esc(r.sentinel_missions || 'Sentinel')}</span>`
      : r.sentinel_lineage_status === 'explicit_non_sentinel'
        ? '<span class="text-slate-500">Non-Sentinel</span>'
        : r.sentinel_lineage_status === 'target_grid_or_harmonization_reference_only'
          ? '<span class="text-amber-700 text-[11px]">Target Grid Ref</span>'
          : '<span class="text-slate-400">Unassessed</span>';

    const levelBadge = r.processing_level
      ? `<span class="inline-flex items-center rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-800 ring-1 ring-inset ring-slate-300/70 tabular-nums">${esc(r.processing_level)}</span>`
      : '<span class="text-slate-400">—</span>';

    const nonStrictTag = !r.strict_eo
      ? '<span class="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-600/20 ml-1.5">Sensitivity / Ref</span>'
      : '';

    const dupTag = r.cluster_count > 1
      ? `<span class="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium bg-slate-100 text-slate-600 ml-1.5 tabular-nums" title="Merged across ${r.cluster_count} catalogue entries">${r.cluster_count}×</span>`
      : '';

    tr.innerHTML = `
      <td class="py-2.5 pl-4 pr-3 align-top">
        <div class="font-medium text-slate-900 group-hover:text-indigo-700 transition-colors leading-snug line-clamp-2">${esc(r.title)}</div>
        <div class="mt-0.5 text-[11px] text-slate-500 flex items-center flex-wrap gap-x-1.5">
          <span class="truncate max-w-[260px]" title="${esc(r.producing_institution || '')}">${esc(r.producing_institution || 'Unknown institution')}</span>
          <span class="text-slate-300">·</span>
          <span class="font-mono text-[10px] text-slate-400">${esc(r.source_catalogue)}</span>${nonStrictTag}${dupTag}
        </div>
      </td>
      <td class="px-3 py-2.5 align-top whitespace-nowrap">${levelBadge}</td>
      <td class="px-3 py-2.5 align-top whitespace-nowrap">
        <span class="inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${poolInfo.cls}">${esc(poolInfo.label)}</span>
        ${r.copernicus_service ? `<div class="mt-0.5 text-[10px] font-mono text-indigo-600">${esc(r.copernicus_service)}</div>` : ''}
      </td>
      <td class="px-3 py-2.5 align-top whitespace-nowrap">${sentinelCell}</td>
      <td class="px-3 py-2.5 align-top whitespace-nowrap tabular-nums">
        <div class="text-slate-800 font-medium">${gridStr ? esc(gridStr) : '<span class="text-slate-400 font-normal">—</span>'}</div>
        <div class="text-[11px] text-slate-500">${cadenceStr ? esc(cadenceStr) : '—'}</div>
      </td>
      <td class="px-3 py-2.5 align-top whitespace-nowrap">
        <div class="text-slate-700">${r.geographic_coverage_class ? esc(GEO_LABELS[r.geographic_coverage_class] || r.geographic_coverage_class) : '<span class="text-slate-400">—</span>'}</div>
        ${r.observation_domain_mask ? `<div class="text-[10px] text-slate-400">${esc(formatEnumLabel(r.observation_domain_mask))}</div>` : ''}
      </td>
      <td class="px-3 py-2.5 align-top whitespace-nowrap tabular-nums">
        <div class="text-slate-800">${esc(yearsStr)}</div>
        <span class="mt-0.5 inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ring-1 ring-inset ${matInfo.cls}">${esc(matInfo.label)}</span>
      </td>
      <td class="py-2.5 pl-3 pr-4 align-top text-right whitespace-nowrap">
        <div class="inline-flex items-center gap-1.5">
          ${safeDoiHref ? `<a href="${esc(safeDoiHref)}" target="_blank" rel="noopener noreferrer" data-ext="1" class="rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-medium text-indigo-700 hover:bg-indigo-100 ring-1 ring-inset ring-indigo-600/20">DOI</a>` : ''}
          ${safeUrl ? `<a href="${esc(safeUrl)}" target="_blank" rel="noopener noreferrer" data-ext="1" class="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-700 hover:bg-slate-200 ring-1 ring-inset ring-slate-300/80">Access ↗</a>` : '<span class="text-slate-300 text-[10px]">—</span>'}
        </div>
      </td>
    `;

    tr.addEventListener('click', (ev) => {
      if (ev.target.closest('a[data-ext]')) return;
      openDetailDrawer(i);
    });
    frag.appendChild(tr);
  }
  dom.tableBody.innerHTML = '';
  dom.tableBody.appendChild(frag);
}

async function fetchDetailRecord(row) {
  const chunkSize = state.meta.detail_chunk_size || 600;
  const chunkIdx = Math.floor(row.idx / chunkSize);
  const localIdx = row.idx % chunkSize;
  const chunkName = `data/details/chunk_${String(chunkIdx).padStart(3, '0')}.json`;

  if (!state.chunkCache.has(chunkIdx)) {
    const resp = await fetch(chunkName);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${chunkName}`);
    const arr = await resp.json();
    state.chunkCache.set(chunkIdx, arr);
  }
  return state.chunkCache.get(chunkIdx)[localIdx] || {};
}

function destroyBboxMap() {
  if (state.mapResizeTimer) {
    clearTimeout(state.mapResizeTimer);
    state.mapResizeTimer = null;
  }
  if (state.map) {
    try {
      state.map.remove();
    } catch {
      // Ignore teardown race during rapid switching
    }
    state.map = null;
  }
}

function renderBboxContainer(bbox, crossesAntimeridian) {
  if (!bbox || bbox.length < 4) {
    return '<div class="h-24 rounded-lg bg-slate-50 border border-dashed border-slate-200 flex items-center justify-center text-[11px] text-slate-400">No bounding box provided upstream</div>';
  }
  const [w, s, e, n] = bbox;
  return `
    <div class="rounded-lg border border-slate-200 bg-slate-50 p-2 space-y-1.5">
      <div id="bbox-map" class="w-full h-56 rounded-md border border-slate-200 bg-sky-50/40 overflow-hidden relative" aria-label="Interactive OpenFreeMap bounding box map [${w}, ${s}, ${e}, ${n}]"></div>
      <div class="flex items-center justify-between flex-wrap gap-2 text-[10px] font-mono text-slate-500 tabular-nums px-0.5">
        <span>W ${w}° · S ${s}° · E ${e}° · N ${n}°</span>
        <span class="font-sans text-slate-400">${crossesAntimeridian || w > e ? 'Antimeridian crossing · ' : ''}OpenFreeMap · Drag to pan, +/- to zoom</span>
      </div>
    </div>
  `;
}

function mountBboxMap(bbox, crossesAntimeridian) {
  destroyBboxMap();
  if (!bbox || bbox.length < 4) return;
  const container = document.getElementById('bbox-map');
  if (!container) return;

  if (typeof window.maplibregl === 'undefined') {
    container.innerHTML = '<div class="h-full flex items-center justify-center text-[11px] text-slate-500 px-3 text-center">MapLibre GL JS CDN unavailable offline; raw bounding coordinates shown below.</div>';
    return;
  }

  const [wRaw, sRaw, eRaw, nRaw] = bbox;
  const clampLat = (lat) => Math.max(-83.5, Math.min(83.5, lat));
  const clampLon = (lon) => Math.max(-180, Math.min(180, lon));
  const w = clampLon(wRaw);
  const e = clampLon(eRaw);
  const s = clampLat(Math.min(sRaw, nRaw));
  const n = clampLat(Math.max(sRaw, nRaw));
  const isAntim = Boolean(crossesAntimeridian || w > e);
  const isPointLike = !isAntim && Math.abs(e - w) < 0.05 && Math.abs(n - s) < 0.05;
  const isNearGlobal = !isAntim && (e - w) >= 350 && (n - s) >= 150;

  let geom;
  if (isAntim) {
    geom = {
      type: 'MultiPolygon',
      coordinates: [
        [[[w, s], [180, s], [180, n], [w, n], [w, s]]],
        [[[-180, s], [e, s], [e, n], [-180, n], [-180, s]]],
      ],
    };
  } else {
    const pw = isPointLike ? Math.max(-180, w - 0.15) : w;
    const pe = isPointLike ? Math.min(180, e + 0.15) : e;
    const ps = isPointLike ? clampLat(s - 0.15) : s;
    const pn = isPointLike ? clampLat(n + 0.15) : n;
    geom = {
      type: 'Polygon',
      coordinates: [[[pw, ps], [pe, ps], [pe, pn], [pw, pn], [pw, ps]]],
    };
  }

  const features = [{ type: 'Feature', geometry: geom, properties: {} }];
  if (isPointLike) {
    features.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [w, s] }, properties: {} });
  }

  try {
    const map = new window.maplibregl.Map({
      container,
      style: 'https://tiles.openfreemap.org/styles/positron',
      center: isAntim ? [(w + (e + 360)) / 2 - 360, (s + n) / 2] : [(w + e) / 2, isNearGlobal ? 14 : (s + n) / 2],
      zoom: isNearGlobal ? 0.55 : 1.5,
      scrollZoom: false,
      attributionControl: { compact: true },
    });
    state.map = map;
    map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    map.on('load', () => {
      if (state.map !== map) return;
      map.addSource('product-bbox', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features },
      });
      map.addLayer({
        id: 'product-bbox-fill',
        type: 'fill',
        source: 'product-bbox',
        filter: ['in', ['geometry-type'], ['literal', ['Polygon', 'MultiPolygon']]],
        paint: {
          'fill-color': '#4f46e5',
          'fill-opacity': 0.18,
        },
      });
      map.addLayer({
        id: 'product-bbox-line',
        type: 'line',
        source: 'product-bbox',
        filter: ['in', ['geometry-type'], ['literal', ['Polygon', 'MultiPolygon']]],
        paint: {
          'line-color': '#4338ca',
          'line-width': 2,
        },
      });
      if (isPointLike) {
        map.addLayer({
          id: 'product-bbox-point',
          type: 'circle',
          source: 'product-bbox',
          filter: ['==', ['geometry-type'], 'Point'],
          paint: {
            'circle-radius': 5,
            'circle-color': '#4338ca',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff',
          },
        });
      }

      if (isAntim) {
        map.fitBounds([[w, s], [e + 360, n]], { padding: 28, maxZoom: 6, duration: 0 });
      } else if (!isNearGlobal) {
        map.fitBounds([[w, s], [e, n]], { padding: 32, maxZoom: 7, duration: 0 });
      } else {
        map.jumpTo({ center: [0, 14], zoom: 0.45 });
      }
      map.resize();
    });

    map.on('error', () => {
      // Non-fatal tile/network warning; leave map container cleanly styled
    });

    state.mapResizeTimer = setTimeout(() => {
      if (state.map === map) map.resize();
    }, 220);
  } catch (err) {
    container.innerHTML = `<div class="h-full flex items-center justify-center text-[11px] text-slate-500 px-3 text-center">Interactive map unavailable (${esc(err.message)}). Bounds: W ${wRaw}°, S ${sRaw}°, E ${eRaw}°, N ${nRaw}°.</div>`;
  }
}

function kvItem(label, value, isMono = false) {
  const valStr = value === null || value === undefined || value === ''
    ? '<span class="text-slate-400 italic font-sans">null</span>'
    : `<span class="${isMono ? 'font-mono text-[11px]' : ''}" style="overflow-wrap:anywhere">${esc(value)}</span>`;
  return `<div class="min-w-0"><dt class="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">${esc(label)}</dt><dd class="mt-0.5 text-slate-800 min-w-0" style="overflow-wrap:anywhere">${valStr}</dd></div>`;
}

async function openDetailDrawer(filteredPos, historyMode = 'push') {
  const r = state.filtered[filteredPos];
  if (!r) return;
  await openDetailRecord(r, filteredPos, historyMode);
}

async function openDetailRecord(r, filteredPos, historyMode = 'push') {
  if (!r) return;
  destroyBboxMap();
  state.activeRecordId = r.record_id;
  state.activeRecordIdx = filteredPos;
  dom.drawerPrev.disabled = filteredPos < 0 || filteredPos <= 0;
  dom.drawerNext.disabled = filteredPos < 0 || filteredPos >= state.filtered.length - 1;
  writeUrlState(historyMode);

  const poolInfo = POOL_META[r.provider_pool] || { label: r.provider_pool || '—', cls: 'bg-slate-100 text-slate-700 ring-slate-500/20' };
  dom.drawerBadges.innerHTML = `
    <span class="inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${poolInfo.cls}">${esc(poolInfo.label)}</span>
    ${r.processing_level ? `<span class="inline-flex items-center rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-800">${esc(r.processing_level)}</span>` : ''}
    ${r.uses_sentinel ? `<span class="inline-flex items-center rounded-md bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">Sentinel ${esc(r.sentinel_missions || '')}</span>` : ''}
    ${!r.strict_eo ? '<span class="inline-flex items-center rounded-md bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-700 ring-1 ring-inset ring-amber-600/20">Non-strict sensitivity/ref</span>' : ''}
  `;
  dom.drawerTitle.textContent = r.title;
  dom.drawerSubtitle.textContent = `${r.record_id}${r.cluster_count > 1 ? ` · Canonical cluster of ${r.cluster_count} records` : ''}`;
  dom.drawerBody.innerHTML = '<div class="py-12 text-center text-slate-400">Loading full provenance metadata…</div>';

  dom.drawerBackdrop.classList.remove('hidden');
  dom.drawer.classList.remove('translate-x-full');

  let det = {};
  try {
    det = await fetchDetailRecord(r);
  } catch (err) {
    det = { _error: err.message };
  }
  if (state.activeRecordId !== r.record_id) return;

  const safeAccess = safeHref(r.access_url);
  const safeDoi = r.doi ? `https://doi.org/${encodeURI(r.doi)}` : null;
  const adjCitation = safeHref(det.adjudication_source_citation);
  const fullDesc = det.description || r.desc_snippet || 'No description provided upstream.';
  const kwList = (det.keywords || r.kw_snippet || '').split(/[;,]/).map((s) => s.trim()).filter(Boolean);

  dom.drawerBody.innerHTML = `
    ${det._error ? `<div class="rounded-lg bg-rose-50 border border-rose-200 p-3 text-rose-700">Failed to load detail shard: ${esc(det._error)}</div>` : ''}

    <!-- Quick Action Bar -->
    <div class="flex flex-wrap items-center gap-2">
      ${safeAccess ? `<a href="${esc(safeAccess)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-medium text-white hover:bg-slate-800 transition">Open Source Endpoint ↗</a>` : ''}
      ${safeDoi ? `<a href="${esc(safeDoi)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 rounded-lg bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 ring-1 ring-inset ring-indigo-600/20 hover:bg-indigo-100 transition">DOI: ${esc(r.doi)} ↗</a>` : ''}
      <button type="button" id="btn-copy-link" class="inline-flex items-center gap-1 rounded-lg bg-white px-2.5 py-2 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 transition">Copy Share Link</button>
      <button type="button" id="btn-copy-id" class="inline-flex items-center gap-1 rounded-lg bg-white px-2.5 py-2 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 transition">Copy Record ID</button>
    </div>

    <!-- Abstract / Description -->
    <section class="space-y-1.5">
      <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Description</h3>
      <p class="text-xs leading-relaxed text-slate-700 whitespace-pre-line bg-slate-50 rounded-lg p-3.5 border border-slate-200/70">${esc(fullDesc)}</p>
    </section>

    <!-- Spatial & Bounds -->
    <section class="space-y-2">
      <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Spatial Bounds &amp; Grid Resolution</h3>
      ${renderBboxContainer(r.bbox, r.antimeridian)}
      <dl class="grid grid-cols-2 sm:grid-cols-3 gap-3 bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">
        ${kvItem('Geographic Class', r.geographic_coverage_class)}
        ${kvItem('Observation Domain', r.observation_domain_class)}
        ${kvItem('Domain Mask', r.observation_domain_mask)}
        ${kvItem('Geo Provenance', det.geographic_coverage_provenance)}
        ${kvItem('Adjudicated Grid', formatGridMeters(r.grid_m) || (det.adjudicated_spatial_grid_value ? `${det.adjudicated_spatial_grid_value} ${det.adjudicated_spatial_grid_units || ''}` : null))}
        ${kvItem('Raw Spatial String', det.spatial_resolution_raw)}
        ${kvItem('Grid Source', det.adjudicated_spatial_resolution_source)}
        ${kvItem('Footprint (m)', det.spatial_footprint_meters)}
        ${kvItem('Title Token (m)', det.title_token_resolution_meters)}
      </dl>
      ${det.prose_spatial_evidence_sentence ? `<div class="text-[11px] text-slate-600 bg-slate-50 rounded-lg p-2.5 border border-slate-200/60"><span class="font-semibold text-slate-700">Prose Evidence (${esc(det.prose_spatial_role || 'note')}):</span> ${esc(det.prose_spatial_evidence_sentence)}</div>` : ''}
      ${det.spatial_resolution_conflict ? `<div class="text-[11px] text-amber-800 bg-amber-50 rounded-lg p-2.5 border border-amber-200"><strong>Spatial Conflict:</strong> ${esc(det.spatial_resolution_conflict_reason || 'Conflicting native metadata kept null')}</div>` : ''}
      ${adjCitation ? `<div class="text-[11px]"><span class="text-slate-500 font-medium">Citation:</span> <a href="${esc(adjCitation)}" target="_blank" rel="noopener noreferrer" class="text-indigo-600 underline break-all">${esc(adjCitation)}</a></div>` : ''}
    </section>

    <!-- Institutional & Classification Provenance -->
    <section class="space-y-2">
      <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Institutional Attribution &amp; Classification</h3>
      <dl class="grid grid-cols-1 sm:grid-cols-2 gap-3 bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">
        ${kvItem('Producing Institution', r.producing_institution)}
        ${kvItem('Hosting Platform', r.hosting_platform)}
        ${kvItem('Jurisdiction Tier', r.provider_jurisdiction_tier)}
        ${kvItem('Copernicus Service', r.copernicus_service)}
        ${kvItem('Ontology Class', r.product_ontology_class)}
        ${kvItem('Output Category', r.product_output_category)}
        ${kvItem('Processing Level', r.processing_level)}
        ${kvItem('Raw Level Tag', det.processing_level_raw)}
        ${kvItem('Levels Available', det.processing_levels_available)}
        ${kvItem('Level Provenance', det.processing_level_provenance)}
        ${kvItem('Canonical Key', det.canonical_product_key, true)}
        ${kvItem('Source Catalogues', det.all_source_catalogues || r.source_catalogue, true)}
      </dl>
      ${det.processing_level_conflict ? `<div class="text-[11px] text-amber-800 bg-amber-50 rounded-lg p-2.5 border border-amber-200"><strong>Level Conflict:</strong> ${esc(det.processing_level_conflict_reason || 'Conflicting upstream processing levels')}</div>` : ''}
    </section>

    <!-- Sentinel, Temporal & Licensing -->
    <section class="space-y-2">
      <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Sensors, Temporal Cadence &amp; Licensing</h3>
      <dl class="grid grid-cols-1 sm:grid-cols-2 gap-3 bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">
        ${kvItem('Sentinel Lineage Status', r.sentinel_lineage_status)}
        ${kvItem('Sentinel Missions', r.sentinel_missions)}
        ${kvItem('Platform / Sensor Raw', det.platform_sensor_raw)}
        ${kvItem('Storage Asset Type', det.asset_type_raw)}
        ${kvItem('Temporal Cadence', r.temporal_cadence_class)}
        ${kvItem('Output Cadences', det.temporal_output_cadences)}
        ${kvItem('Temporal Window Raw', `${det.temporal_start_raw || 'null'} -> ${det.temporal_end_raw || 'null'}`)}
        ${kvItem('Operational Maturity', r.operational_maturity_class)}
        ${kvItem('Span Provenance', det.temporal_span_provenance)}
        ${kvItem('Cadence by Epoch', det.temporal_cadence_by_epoch)}
        ${kvItem('License', r.license)}
        ${kvItem('DOI Source / Relation', det.doi_source ? `${det.doi_source} (${det.doi_relation_type || ''})` : null)}
      </dl>
      ${det.related_dois ? `<div class="text-[11px] text-slate-600"><span class="font-semibold text-slate-700">Related DOIs:</span> <span class="font-mono break-all">${esc(det.related_dois)}</span></div>` : ''}
      ${det.superseded_by_record ? `<div class="text-[11px] text-rose-700 bg-rose-50 rounded-lg p-2.5 border border-rose-200"><strong>Superseded by:</strong> <span class="font-mono">${esc(det.superseded_by_record)}</span></div>` : ''}
    </section>

    ${kwList.length > 0 ? `
      <section class="space-y-1.5">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Keywords (click to search)</h3>
        <div class="flex flex-wrap gap-1.5">
          ${kwList.slice(0, 40).map((kw) => `<button type="button" data-kw="${esc(kw)}" class="kw-chip rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] text-slate-700 hover:bg-indigo-100 hover:text-indigo-800 transition">${esc(kw)}</button>`).join('')}
        </div>
      </section>
    ` : ''}
  `;

  mountBboxMap(r.bbox, r.antimeridian);

  const copyLinkBtn = document.getElementById('btn-copy-link');
  if (copyLinkBtn) {
    copyLinkBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(window.location.href);
        copyLinkBtn.textContent = 'Link copied!';
        setTimeout(() => { copyLinkBtn.textContent = 'Copy Share Link'; }, 1400);
      } catch {
        copyLinkBtn.textContent = 'Clipboard blocked';
      }
    });
  }

  const copyBtn = document.getElementById('btn-copy-id');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(r.record_id);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy Record ID'; }, 1200);
      } catch {
        copyBtn.textContent = 'Clipboard blocked';
      }
    });
  }

  dom.drawerBody.querySelectorAll('.kw-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      dom.search.value = chip.dataset.kw || '';
      closeDetailDrawer('none');
      applyFiltersAndSort(true, 'push');
    });
  });
}

function closeDetailDrawer(historyMode = 'push') {
  destroyBboxMap();
  state.activeRecordId = null;
  state.activeRecordIdx = null;
  dom.drawer.classList.add('translate-x-full');
  dom.drawerBackdrop.classList.add('hidden');
  if (!state.loading && historyMode !== 'none') writeUrlState(historyMode);
}

function csvCell(val) {
  if (val === null || val === undefined) return '';
  let s = String(val);
  // Prevent CSV spreadsheet formula injection
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setDownloadMenuOpen(open) {
  if (!dom.downloadMenu || !dom.btnDownload) return;
  dom.downloadMenu.classList.toggle('hidden', !open);
  dom.btnDownload.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function xmlEsc(str) {
  return String(str)
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g, '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function colRef(idx) {
  let s = '';
  let n = idx + 1;
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

const CRC32_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    t[i] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC32_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function buildZipStoreBlob(entries, mimeType) {
  const te = new TextEncoder();
  const localParts = [];
  const centralParts = [];
  let offset = 0;

  entries.forEach(({ name, content }) => {
    const nameBytes = te.encode(name);
    const dataBytes = typeof content === 'string' ? te.encode(content) : content;
    const crc = crc32(dataBytes);
    const len = dataBytes.length;

    const lh = new Uint8Array(30 + nameBytes.length);
    const lhv = new DataView(lh.buffer);
    lhv.setUint32(0, 0x04034b50, true);
    lhv.setUint16(4, 20, true);
    lhv.setUint16(6, 0x0800, true);
    lhv.setUint16(8, 0, true);
    lhv.setUint32(14, crc, true);
    lhv.setUint32(18, len, true);
    lhv.setUint32(22, len, true);
    lhv.setUint16(26, nameBytes.length, true);
    lh.set(nameBytes, 30);
    localParts.push(lh, dataBytes);

    const ch = new Uint8Array(46 + nameBytes.length);
    const chv = new DataView(ch.buffer);
    chv.setUint32(0, 0x02014b50, true);
    chv.setUint16(4, 20, true);
    chv.setUint16(6, 20, true);
    chv.setUint16(8, 0x0800, true);
    chv.setUint16(10, 0, true);
    chv.setUint32(16, crc, true);
    chv.setUint32(20, len, true);
    chv.setUint32(24, len, true);
    chv.setUint16(28, nameBytes.length, true);
    chv.setUint32(42, offset, true);
    ch.set(nameBytes, 46);
    centralParts.push(ch);

    offset += lh.length + len;
  });

  const centralSize = centralParts.reduce((a, b) => a + b.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  return new Blob([...localParts, ...centralParts, eocd], { type: mimeType });
}

function buildXlsxBlob(columns, rows) {
  const letters = columns.map((_, i) => colRef(i));
  const rowXml = [];
  rowXml.push(
    `<row r="1">${columns.map(([name], ci) => `<c r="${letters[ci]}1" t="inlineStr"><is><t>${xmlEsc(name)}</t></is></c>`).join('')}</row>`
  );

  for (let ri = 0; ri < rows.length; ri++) {
    const r = rows[ri];
    const rn = ri + 2;
    const cells = [];
    for (let ci = 0; ci < columns.length; ci++) {
      const [name, type] = columns[ci];
      const v = r[name];
      if (v === null || v === undefined || v === '') continue;
      if (type === 'INT32' || type === 'DOUBLE') {
        if (typeof v === 'number' && Number.isFinite(v)) {
          cells.push(`<c r="${letters[ci]}${rn}"><v>${v}</v></c>`);
        }
      } else if (type === 'BOOLEAN') {
        cells.push(`<c r="${letters[ci]}${rn}" t="b"><v>${v ? 1 : 0}</v></c>`);
      } else {
        let s = String(v);
        if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
        cells.push(`<c r="${letters[ci]}${rn}" t="inlineStr"><is><t xml:space="preserve">${xmlEsc(s)}</t></is></c>`);
      }
    }
    rowXml.push(`<row r="${rn}">${cells.join('')}</row>`);
  }

  const sheetXml =
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' +
    rowXml.join('') +
    '</sheetData></worksheet>';

  const contentTypes =
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
    '<Default Extension="xml" ContentType="application/xml"/>' +
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' +
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' +
    '</Types>';

  const rootRels =
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>' +
    '</Relationships>';

  const workbookXml =
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">' +
    '<sheets><sheet name="Catalogue" sheetId="1" r:id="rId1"/></sheets>' +
    '</workbook>';

  const wbRels =
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>' +
    '</Relationships>';

  return buildZipStoreBlob(
    [
      { name: '[Content_Types].xml', content: contentTypes },
      { name: '_rels/.rels', content: rootRels },
      { name: 'xl/workbook.xml', content: workbookXml },
      { name: 'xl/_rels/workbook.xml.rels', content: wbRels },
      { name: 'xl/worksheets/sheet1.xml', content: sheetXml },
    ],
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  );
}

function exportFilteredCsv() {
  const cols = EXPORT_COLUMNS.map(([name]) => name);
  const lines = [cols.join(',')];
  state.filtered.forEach((r) => {
    lines.push(cols.map((c) => csvCell(r[c])).join(','));
  });
  const blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8;' });
  triggerBlobDownload(blob, `eo_l3_l4_catalogue_filtered_${state.filtered.length}.csv`);
}

function exportFilteredJson() {
  const cols = EXPORT_COLUMNS.map(([name]) => name);
  const payload = state.filtered.map((r) => {
    const obj = {};
    cols.forEach((c) => { obj[c] = r[c] === undefined ? null : r[c]; });
    return obj;
  });
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8;' });
  triggerBlobDownload(blob, `eo_l3_l4_catalogue_filtered_${state.filtered.length}.json`);
}

function exportFilteredXlsx() {
  const blob = buildXlsxBlob(EXPORT_COLUMNS, state.filtered);
  triggerBlobDownload(blob, `eo_l3_l4_catalogue_filtered_${state.filtered.length}.xlsx`);
}

async function exportFilteredParquet() {
  parquetWriterPromise ||= import('https://cdn.jsdelivr.net/npm/hyparquet-writer@0.18.0/+esm');
  const { parquetWriteBuffer } = await parquetWriterPromise;
  const rows = state.filtered;
  const columnData = EXPORT_COLUMNS.map(([name, type]) => ({
    name,
    type,
    nullable: true,
    data: rows.map((r) => {
      const v = r[name];
      if (v === null || v === undefined) return null;
      if (type === 'BOOLEAN') return Boolean(v);
      if (type === 'INT32' || type === 'DOUBLE') return Number.isNaN(v) ? null : v;
      return String(v);
    }),
  }));
  const buffer = parquetWriteBuffer({
    columnData,
    createdBy: 'global-EO-products-L3-L4',
  });
  const blob = new Blob([buffer], { type: 'application/vnd.apache.parquet' });
  triggerBlobDownload(blob, `eo_l3_l4_catalogue_filtered_${rows.length}.parquet`);
}

async function handleExportFormat(fmt) {
  setDownloadMenuOpen(false);
  if (state.loading || state.filtered.length === 0 || state.exporting) return;
  if (dom.errorBanner) {
    dom.errorBanner.textContent = '';
    dom.errorBanner.classList.add('hidden');
  }
  state.exporting = true;
  if (dom.btnDownload) dom.btnDownload.disabled = true;
  if (dom.btnDownloadLabel) dom.btnDownloadLabel.textContent = 'Exporting…';
  try {
    await new Promise((resolve) => setTimeout(resolve, 20));
    if (fmt === 'csv') exportFilteredCsv();
    else if (fmt === 'json') exportFilteredJson();
    else if (fmt === 'xlsx') exportFilteredXlsx();
    else if (fmt === 'parquet') await exportFilteredParquet();
  } catch (err) {
    if (fmt === 'parquet') parquetWriterPromise = null;
    dom.errorBanner.textContent = `${fmt.toUpperCase()} export failed (${err.message}). CSV, JSON, and Excel (.xlsx) exports remain available offline.`;
    dom.errorBanner.classList.remove('hidden');
  } finally {
    state.exporting = false;
    if (dom.btnDownload) dom.btnDownload.disabled = state.filtered.length === 0;
    if (dom.btnDownloadLabel) dom.btnDownloadLabel.textContent = 'Download';
  }
}

function resetAllFilters() {
  dom.search.value = '';
  dom.pool.value = '';
  dom.level.value = '';
  dom.sentinel.value = '';
  dom.geo.value = '';
  dom.strictEo.checked = false;
  dom.hasDoi.checked = false;
  dom.source.value = '';
  dom.tier.value = '';
  dom.category.value = '';
  dom.cadence.value = '';
  dom.maturity.value = '';
  dom.ontology.value = '';
  dom.grid.value = '';
  dom.year.value = '';
  dom.hasUrl.checked = false;
  dom.exclDeprecated.checked = false;
  state.sortCol = 'default';
  state.sortDir = 'asc';
  applyFiltersAndSort(true);
}

function syncDrawerToState(historyMode = 'none') {
  if (!state.activeRecordId) {
    if (!dom.drawer.classList.contains('translate-x-full')) closeDetailDrawer(historyMode);
    return;
  }
  const fIdx = state.filtered.findIndex((r) => r.record_id === state.activeRecordId);
  const rec = fIdx >= 0 ? state.filtered[fIdx] : state.records.find((r) => r.record_id === state.activeRecordId);
  if (rec) {
    openDetailRecord(rec, fIdx, historyMode);
  }
}

const PRESET_FILTERS = {
  non_eu_sentinel: {
    apply: () => { dom.pool.value = 'non_eu'; dom.sentinel.value = 'yes'; },
    clear: () => { dom.pool.value = ''; dom.sentinel.value = ''; },
  },
  copernicus_l3_l4: {
    apply: () => { dom.pool.value = 'copernicus_services'; },
    clear: () => { dom.pool.value = ''; },
  },
  global_ongoing: {
    apply: () => {
      dom.geo.value = 'global';
      dom.maturity.value = 'operational_ongoing_series';
      setMorePanelOpen(true, 'none');
    },
    clear: () => { dom.geo.value = ''; dom.maturity.value = ''; },
  },
  high_res_eo: {
    apply: () => {
      dom.strictEo.checked = true;
      dom.grid.value = 'le30';
      setMorePanelOpen(true, 'none');
    },
    clear: () => { dom.strictEo.checked = false; dom.grid.value = ''; },
  },
};

function wireEvents() {
  dom.search.addEventListener('input', () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => applyFiltersAndSort(true, 'replace'), 150);
  });
  dom.btnClearSearch.addEventListener('click', () => {
    dom.search.value = '';
    applyFiltersAndSort(true, 'push');
    dom.search.focus();
  });

  [dom.pool, dom.level, dom.sentinel, dom.geo, dom.strictEo, dom.hasDoi,
    dom.source, dom.tier, dom.category, dom.cadence, dom.maturity, dom.ontology,
    dom.grid, dom.hasUrl, dom.exclDeprecated].forEach((el) => {
    el.addEventListener('change', () => applyFiltersAndSort(true, 'push'));
  });

  dom.year.addEventListener('input', () => {
    clearTimeout(state.yearTimer);
    state.yearTimer = setTimeout(() => applyFiltersAndSort(true, 'replace'), 200);
  });

  dom.btnToggleMore.addEventListener('click', () => {
    const isOpen = !dom.morePanel.classList.contains('hidden');
    setMorePanelOpen(!isOpen, 'push');
  });

  dom.btnResetAll.addEventListener('click', () => resetAllFilters());

  dom.presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.preset;
      const spec = PRESET_FILTERS[p];
      if (!spec) return;
      const wasActive = btn.dataset.active === 'true';

      // Clear previously active preset filters without erasing user search text or unrelated filters.
      dom.presetBtns.forEach((other) => {
        if (other !== btn && other.dataset.active === 'true') {
          PRESET_FILTERS[other.dataset.preset]?.clear();
        }
      });

      if (wasActive) {
        spec.clear();
      } else {
        spec.apply();
      }
      applyFiltersAndSort(true, 'push');
    });
  });

  dom.sortThs.forEach((th) => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (state.sortCol === col) {
        if (state.sortDir === 'asc') state.sortDir = 'desc';
        else { state.sortCol = 'default'; state.sortDir = 'asc'; }
      } else {
        state.sortCol = col;
        state.sortDir = 'asc';
      }
      applyFiltersAndSort(false, 'push');
    });
  });

  dom.pageSize.addEventListener('change', () => {
    state.pageSize = Number.parseInt(dom.pageSize.value, 10) || 50;
    state.page = 1;
    renderTable();
    writeUrlState('push');
  });

  if (dom.btnShareUrl) {
    dom.btnShareUrl.addEventListener('click', async () => {
      writeUrlState('replace');
      try {
        await navigator.clipboard.writeText(window.location.href);
        dom.btnShareLabel.textContent = 'Copied!';
        setTimeout(() => { dom.btnShareLabel.textContent = 'Share URL'; }, 1400);
      } catch {
        dom.btnShareLabel.textContent = 'Copy from address bar';
        setTimeout(() => { dom.btnShareLabel.textContent = 'Share URL'; }, 1800);
      }
    });
  }

  window.addEventListener('popstate', () => {
    if (state.loading) return;
    hydrateFiltersFromUrl();
    applyFiltersAndSort(false, 'none');
    syncDrawerToState('none');
  });

  if (dom.btnDownload) {
    dom.btnDownload.addEventListener('click', (e) => {
      e.stopPropagation();
      const willOpen = dom.downloadMenu.classList.contains('hidden');
      setDownloadMenuOpen(willOpen);
      if (willOpen && dom.downloadOpts.length) dom.downloadOpts[0].focus();
    });
    dom.btnDownload.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setDownloadMenuOpen(true);
        if (dom.downloadOpts.length) dom.downloadOpts[0].focus();
      }
    });
  }
  dom.downloadOpts.forEach((btn, idx) => {
    btn.addEventListener('click', () => handleExportFormat(btn.dataset.exportFmt));
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        dom.downloadOpts[(idx + 1) % dom.downloadOpts.length].focus();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        dom.downloadOpts[(idx - 1 + dom.downloadOpts.length) % dom.downloadOpts.length].focus();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        setDownloadMenuOpen(false);
        dom.btnDownload.focus();
      }
    });
  });
  document.addEventListener('click', (e) => {
    if (dom.downloadWrap && !dom.downloadWrap.contains(e.target)) {
      setDownloadMenuOpen(false);
    }
  });
  dom.drawerClose.addEventListener('click', () => closeDetailDrawer('push'));
  dom.drawerBackdrop.addEventListener('click', () => closeDetailDrawer('push'));
  dom.drawerPrev.addEventListener('click', () => {
    if (state.activeRecordIdx !== null && state.activeRecordIdx > 0) openDetailDrawer(state.activeRecordIdx - 1, 'push');
  });
  dom.drawerNext.addEventListener('click', () => {
    if (state.activeRecordIdx !== null && state.activeRecordIdx < state.filtered.length - 1) openDetailDrawer(state.activeRecordIdx + 1, 'push');
  });

  dom.btnInfo.addEventListener('click', () => dom.infoModal.classList.remove('hidden'));
  dom.infoClose.addEventListener('click', () => dom.infoModal.classList.add('hidden'));
  dom.infoOk.addEventListener('click', () => dom.infoModal.classList.add('hidden'));
  dom.infoModal.addEventListener('click', (e) => {
    if (e.target === dom.infoModal) dom.infoModal.classList.add('hidden');
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement !== dom.search && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'SELECT') {
      e.preventDefault();
      dom.search.focus();
    } else if (e.key === 'Escape') {
      if (dom.downloadMenu && !dom.downloadMenu.classList.contains('hidden')) setDownloadMenuOpen(false);
      else if (!dom.infoModal.classList.contains('hidden')) dom.infoModal.classList.add('hidden');
      else if (state.activeRecordId !== null) closeDetailDrawer('push');
    } else if (state.activeRecordIdx !== null && state.activeRecordIdx >= 0 && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      e.preventDefault();
      const delta = e.key === 'ArrowDown' ? 1 : -1;
      const nextIdx = state.activeRecordIdx + delta;
      if (nextIdx >= 0 && nextIdx < state.filtered.length) openDetailDrawer(nextIdx, 'push');
    }
  });
}

async function init() {
  wireEvents();
  try {
    const metaResp = await fetch('data/index_meta.json');
    if (!metaResp.ok) throw new Error(`HTTP ${metaResp.status} fetching data/index_meta.json`);
    const metaDoc = await metaResp.json();
    state.meta = metaDoc.meta;
    state.enums = metaDoc.enums;
    dom.metaBadge.textContent = `${state.meta.total_canonical.toLocaleString()} canonical · ${state.meta.sources_count} sources`;

    populateSelectOptions(dom.source, state.enums.source_catalogue, (v) => v);
    populateSelectOptions(dom.tier, state.enums.provider_jurisdiction_tier);
    populateSelectOptions(dom.category, state.enums.product_output_category);
    populateSelectOptions(dom.cadence, state.enums.temporal_cadence_class, formatEnumLabel, true);
    populateSelectOptions(dom.maturity, state.enums.operational_maturity_class);
    populateSelectOptions(dom.ontology, state.enums.product_ontology_class);

    hydrateFiltersFromUrl();

    const parts = state.meta.index_parts || [];
    let completed = 0;
    const partArrays = await Promise.all(
      parts.map(async (partName) => {
        const resp = await fetch(`data/${partName}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} loading data/${partName}`);
        const json = await resp.json();
        completed++;
        dom.progressBar.style.width = `${Math.round(10 + (completed / parts.length) * 85)}%`;
        return json;
      })
    );

    const decoded = [];
    for (let p = 0; p < partArrays.length; p++) {
      const chunk = partArrays[p];
      for (let i = 0; i < chunk.length; i++) {
        decoded.push(decodeRow(chunk[i], state.enums));
      }
    }
    state.records = decoded;
    state.loading = false;
    dom.progressBar.style.width = '100%';
    setTimeout(() => dom.progressWrap.classList.add('opacity-0'), 250);

    applyFiltersAndSort(false, 'none');
    syncDrawerToState('none');
  } catch (err) {
    dom.errorBanner.textContent = `Could not load static catalogue index: ${err.message}. Serve docs/ over HTTP (e.g., python3 -m http.server -d docs 8080) rather than file://.`;
    dom.errorBanner.classList.remove('hidden');
    dom.metaBadge.textContent = 'Load failed';
    dom.progressWrap.classList.add('hidden');
  }
}

init();
