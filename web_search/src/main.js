import './style.css';
import { Map as MapLibreMap, NavigationControl, setWorkerUrl } from 'maplibre-gl';
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
import { exportResults } from './exporters.js';

setWorkerUrl(maplibreWorkerUrl);

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
  europe_regional: 'Europe Regional',
  europe: 'Europe (Global/BBox)',
  regional_or_local: 'Regional / Local',
  exclude_local: 'Excl. Regional/Local',
};

const MODE_LABELS = {
  bm25: 'Keyword (BM25)',
  hybrid: 'Hybrid (512 dimensions + BM25)',
  dense: 'Semantic only (512 dimensions)',
  'bm25+qwen3': 'Keyword + Qwen3-0.6B rerank',
  'hybrid+qwen3': 'Hybrid + Qwen3-0.6B rerank',
  'dense+qwen3': 'Semantic + Qwen3-0.6B rerank',
};

const MODE_HELPERS = {
  bm25: 'Instant keyword search across title, keywords, and description. Switch to Hybrid for EmbeddingGemma 2 with 512 dimensions (~175 MB cached once from Hugging Face).',
  hybrid: 'Combines EmbeddingGemma 2 with 512 dimensions (15 MB vectors plus ~175 MB model cached once from Hugging Face) and keyword BM25.',
  dense: 'Ranks strictly by semantic similarity using EmbeddingGemma 2 with 512 dimensions (~175 MB cached once from Hugging Face).',
};

const state = {
  ready: false,
  webgpuAvailable: false,
  manifest: null,
  mode: 'bm25',
  semanticReady: false,
  semanticLoading: false,
  semanticDevice: null,
  rerankerReady: false,
  rerankerLoading: false,
  rerankerDevice: null,
  searching: false,
  reranking: false,
  exporting: false,
  moreOpen: false,
  latestSearchSeq: 0,
  lastPayload: null,
  results: [],
  activeIdx: null,
  map: null,
  mapResizeTimer: null,
  productCache: new Map(),
  detailPending: new Map(),
  detailReqSeq: 0,
  prevNullSafe: true,
};

const dom = {
  statusBadge: document.getElementById('status-badge'),
  loadProgressWrap: document.getElementById('load-progress-wrap'),
  loadProgressBar: document.getElementById('load-progress-bar'),
  errorBanner: document.getElementById('error-banner'),

  modeTabs: Array.from(document.querySelectorAll('.mode-tab')),
  modeHelper: document.getElementById('mode-helper'),
  semanticStatus: document.getElementById('semantic-status'),
  devicePill: document.getElementById('device-pill'),
  semanticProgressWrap: document.getElementById('semantic-progress-wrap'),
  semanticProgressMsg: document.getElementById('semantic-progress-msg'),
  semanticProgressPct: document.getElementById('semantic-progress-pct'),
  semanticProgressBar: document.getElementById('semantic-progress-bar'),

  toggleReranker: document.getElementById('toggle-reranker'),
  rerankerOptions: document.getElementById('reranker-options'),
  rerankerStatus: document.getElementById('reranker-status'),
  rerankerWebgpuNote: document.getElementById('reranker-webgpu-note'),
  optRerankPool: document.getElementById('opt-rerank-pool'),
  optPure: document.getElementById('opt-pure'),
  rerankerProgressWrap: document.getElementById('reranker-progress-wrap'),
  rerankerProgressMsg: document.getElementById('reranker-progress-msg'),
  rerankerProgressPct: document.getElementById('reranker-progress-pct'),
  rerankerProgressBar: document.getElementById('reranker-progress-bar'),

  rerankLiveBanner: document.getElementById('rerank-live-banner'),
  rerankLiveText: document.getElementById('rerank-live-text'),
  btnCancelRerank: document.getElementById('btn-cancel-rerank'),

  form: document.getElementById('search-form'),
  queryInput: document.getElementById('query-input'),
  btnClearQuery: document.getElementById('btn-clear-query'),
  btnSearch: document.getElementById('btn-search'),
  btnSearchLabel: document.getElementById('btn-search-label'),

  optTopK: document.getElementById('opt-topk'),

  poolWrap: document.getElementById('pool-wrap'),
  btnPool: document.getElementById('btn-pool'),
  poolLabel: document.getElementById('pool-label'),
  poolMenu: document.getElementById('pool-menu'),
  btnPoolAll: document.getElementById('btn-pool-all'),
  btnSourcesClear: document.getElementById('btn-sources-clear'),
  poolChecks: Array.from(document.querySelectorAll('.pool-chk')),
  sourceOptions: document.getElementById('source-options'),

  level: document.getElementById('filter-level'),
  sentinel: document.getElementById('filter-sentinel'),
  geo: document.getElementById('filter-geo'),
  strictEo: document.getElementById('filter-strict-eo'),
  hasDoi: document.getElementById('filter-has-doi'),

  btnToggleMore: document.getElementById('btn-toggle-more'),
  morePanel: document.getElementById('more-filters-panel'),
  moreBadge: document.getElementById('more-filters-badge'),

  source: document.getElementById('filter-source'),
  tier: document.getElementById('filter-tier'),
  category: document.getElementById('filter-category'),
  cadence: document.getElementById('filter-cadence'),
  maturity: document.getElementById('filter-maturity'),
  ontology: document.getElementById('filter-ontology'),
  grid: document.getElementById('filter-grid'),
  year: document.getElementById('filter-year'),
  minEndYear: document.getElementById('filter-min-end-year'),
  hasUrl: document.getElementById('filter-has-url'),
  exclDeprecated: document.getElementById('filter-exclude-deprecated'),

  optNullSafe: document.getElementById('opt-null-safe'),
  optHardLimits: document.getElementById('opt-hard-limits'),
  optCollapse: document.getElementById('opt-collapse'),
  btnReset: document.getElementById('btn-reset'),
  activeChipsRow: document.getElementById('active-chips-row'),
  activeChips: document.getElementById('active-chips'),
  exampleBtns: Array.from(document.querySelectorAll('.example-btn')),

  resultsSummary: document.getElementById('results-summary'),
  resultsTiming: document.getElementById('results-timing'),
  resultsBody: document.getElementById('results-body'),

  btnShareUrl: document.getElementById('btn-share-url'),
  btnShareLabel: document.getElementById('btn-share-label'),
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
  infoModal: document.getElementById('info-modal'),
  infoClose: document.getElementById('info-close'),
  infoOk: document.getElementById('info-ok'),
};

const worker = new Worker(new URL('./search.worker.js', import.meta.url), { type: 'module' });

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

function badge(text, cls = 'bg-slate-100 text-slate-700 ring-slate-500/20') {
  return `<span class="inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${cls}">${esc(text)}</span>`;
}

function kvItem(label, value, isMono = false) {
  const valStr = value === null || value === undefined || value === ''
    ? '<span class="text-slate-400 italic font-sans">null</span>'
    : `<span class="${isMono ? 'font-mono text-[11px]' : ''}" style="overflow-wrap:anywhere">${esc(value)}</span>`;
  return `<div class="min-w-0"><dt class="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">${esc(label)}</dt><dd class="mt-0.5 text-slate-800 min-w-0" style="overflow-wrap:anywhere">${valStr}</dd></div>`;
}

function showError(msg) {
  if (!msg) {
    dom.errorBanner.classList.add('hidden');
    dom.errorBanner.textContent = '';
  } else {
    dom.errorBanner.textContent = msg;
    dom.errorBanner.classList.remove('hidden');
  }
}

function setActivityBusy(isBusy, label = 'Searching…') {
  state.searching = isBusy;
  dom.btnSearch.disabled = !state.ready || isBusy;
  dom.btnSearchLabel.textContent = !state.ready ? 'Loading…' : isBusy ? label : 'Search';
  if (isBusy) {
    dom.loadProgressBar.classList.add('bar-indeterminate');
    dom.loadProgressBar.style.width = '';
    dom.loadProgressWrap.classList.remove('opacity-0');
  } else if (state.ready) {
    dom.loadProgressBar.classList.remove('bar-indeterminate');
    dom.loadProgressBar.style.width = '100%';
    dom.loadProgressWrap.classList.add('opacity-0');
  }
}

function setModeUI(mode) {
  state.mode = ['bm25', 'hybrid', 'dense'].includes(mode) ? mode : 'bm25';
  dom.modeTabs.forEach((btn) => {
    btn.setAttribute('aria-pressed', btn.dataset.mode === state.mode ? 'true' : 'false');
  });
  dom.modeHelper.textContent = MODE_HELPERS[state.mode] || MODE_HELPERS.bm25;
  updateHeaderBadge();
}

function updateHeaderBadge() {
  if (!state.manifest) return;
  const nTotal = Number(state.manifest.n_docs || 0).toLocaleString();
  let modeDesc = 'BM25 ready';
  if (state.semanticReady) {
    const dev = state.semanticDevice === 'webgpu' ? 'WebGPU' : 'CPU';
    modeDesc = state.mode === 'bm25' ? `BM25 active, Hybrid ready (${dev})` : `Hybrid ready (${dev})`;
  }
  if (dom.toggleReranker.checked && state.rerankerReady) {
    modeDesc += ' + Qwen3';
  }
  dom.statusBadge.textContent = `${nTotal} products (${modeDesc})`;
  dom.statusBadge.className = state.semanticReady
    ? 'inline-flex items-center rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-800 ring-1 ring-inset ring-emerald-600/20 tabular-nums'
    : 'inline-flex items-center rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-200 tabular-nums';
}

function formatSpan(r) {
  if (r.t0 && r.t1) return `${r.t0}–${r.t1}`;
  if (r.t0) return `${r.t0}–present`;
  if (r.t1) return `–${r.t1}`;
  return '—';
}

function sortByNumericPrefix(a, b) {
  const na = Number.parseInt(String(a).match(/^(\d+)/)?.[1] ?? '9999', 10);
  const nb = Number.parseInt(String(b).match(/^(\d+)/)?.[1] ?? '9999', 10);
  return na !== nb ? na - nb : String(a).localeCompare(String(b));
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

function getSelectedPools() {
  return dom.poolChecks.filter((c) => c.checked).map((c) => c.value);
}

function getSelectedSources() {
  const checked = Array.from(dom.sourceOptions.querySelectorAll('.source-chk:checked')).map((c) => c.value);
  if (dom.source.value && !checked.includes(dom.source.value)) {
    checked.push(dom.source.value);
  }
  return checked;
}

function setPoolMenuOpen(open) {
  dom.poolMenu.classList.toggle('hidden', !open);
  dom.btnPool.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function setMorePanelOpen(open) {
  state.moreOpen = Boolean(open);
  dom.morePanel.classList.toggle('hidden', !state.moreOpen);
  dom.btnToggleMore.setAttribute('aria-expanded', String(state.moreOpen));
}

function updatePoolLabel() {
  const pools = getSelectedPools();
  const sources = getSelectedSources();
  dom.btnSourcesClear.classList.toggle('hidden', sources.length === 0);
  const parts = [];
  if (pools.length > 0 && pools.length < dom.poolChecks.length) {
    parts.push(pools.length === 1 ? (POOL_META[pools[0]]?.label || pools[0]) : `${pools.length} Pools`);
  }
  if (sources.length === 1) parts.push(sources[0]);
  else if (sources.length > 1) parts.push(`${sources.length} Sources`);
  dom.poolLabel.textContent = parts.length ? parts.join(', ') : 'All Pools';
}

function renderSourceOptions(sourceCounts) {
  const entries = Object.entries(sourceCounts || {}).sort(([a], [b]) => sortByNumericPrefix(a, b));
  dom.sourceOptions.innerHTML = entries
    .map(
      ([src, cnt]) => `
    <label class="flex items-center justify-between gap-2 px-2 py-1 rounded-lg hover:bg-slate-50 cursor-pointer text-xs text-slate-700 select-none">
      <span class="flex items-center gap-2 min-w-0">
        <input type="checkbox" value="${esc(src)}" class="source-chk rounded border-slate-300 text-indigo-600 focus:ring-indigo-500" />
        <span class="font-mono text-[11px] truncate" title="${esc(src)}">${esc(src)}</span>
      </span>
      <span class="text-[10px] font-mono text-slate-400 tabular-nums shrink-0">${Number(cnt).toLocaleString()}</span>
    </label>`,
    )
    .join('');
  dom.sourceOptions.querySelectorAll('.source-chk').forEach((chk) => {
    chk.addEventListener('change', () => {
      if (dom.source.value && !chk.checked && dom.source.value === chk.value) {
        dom.source.value = '';
      }
      updatePoolLabel();
      updateBadgesAndChips();
      if (dom.queryInput.value.trim()) runSearch();
    });
  });
}

function syncHardLimitsToggle(fromUserToggle = false) {
  const hard = dom.optHardLimits.checked;
  if (hard) {
    if (fromUserToggle) state.prevNullSafe = dom.optNullSafe.checked;
    dom.optNullSafe.checked = false;
  } else if (fromUserToggle) {
    dom.optNullSafe.checked = state.prevNullSafe;
  }
  dom.optNullSafe.disabled = hard;
  dom.optNullSafe.parentElement?.classList.toggle('opacity-50', hard);
}

function parseOptionalInt(raw, min, max) {
  const s = String(raw || '').trim();
  if (!s) return null;
  const n = Number.parseInt(s, 10);
  if (!Number.isFinite(n) || n < min || n > max) return null;
  return n;
}

function buildSearchPayload() {
  const geoVal = dom.geo.value;
  const levelVal = dom.level.value;
  const pools = getSelectedPools();
  const sources = getSelectedSources();
  const activePools = pools.length > 0 && pools.length < dom.poolChecks.length ? pools : null;
  const levels = levelVal
    ? (levelVal.includes(',') ? levelVal.split(',').map((s) => s.trim()) : [levelVal])
    : null;

  return {
    query: dom.queryInput.value.trim(),
    mode: state.mode,
    use_reranker: dom.toggleReranker.checked,
    rerank_top_n: Number(dom.optRerankPool.value || 25),
    rerank_pure: dom.optPure.checked,
    pools: activePools,
    sources: sources.length ? sources : null,
    levels,
    sentinel: dom.sentinel.value || null,
    geo: geoVal && geoVal !== 'exclude_local' ? geoVal : null,
    exclude_local: geoVal === 'exclude_local',
    strict_eo: dom.strictEo.checked,
    has_doi: dom.hasDoi.checked,
    tier: dom.tier.value || null,
    category: dom.category.value || null,
    cadence: dom.cadence.value || null,
    maturity: dom.maturity.value || null,
    ontology: dom.ontology.value || null,
    grid_rule: dom.grid.value || null,
    covers_year: parseOptionalInt(dom.year.value, 1950, 2035),
    min_end_year: parseOptionalInt(dom.minEndYear.value, 1970, 2040),
    has_url: dom.hasUrl.checked,
    exclude_superseded: dom.exclDeprecated.checked,
    null_safe: dom.optHardLimits.checked ? false : dom.optNullSafe.checked,
    hard_limits: dom.optHardLimits.checked,
    collapse: dom.optCollapse.checked,
    top_k: Number(dom.optTopK.value || 20),
  };
}

function updateBadgesAndChips() {
  const p = buildSearchPayload();
  const advCount = [
    dom.source.value,
    p.tier,
    p.category,
    p.cadence,
    p.maturity,
    p.ontology,
    p.grid_rule,
    p.covers_year !== null,
    p.min_end_year !== null,
    p.has_url,
    p.exclude_superseded,
  ].filter(Boolean).length;
  dom.moreBadge.textContent = String(advCount);
  dom.moreBadge.classList.toggle('hidden', advCount === 0);

  const chips = [];
  const addChip = (label, valText, clearFn) => chips.push({ label, valText, clearFn });

  if (p.pools) {
    addChip(
      'Pool',
      p.pools.map((x) => POOL_META[x]?.label || x).join(', '),
      () => {
        dom.poolChecks.forEach((c) => { c.checked = false; });
        updatePoolLabel();
      },
    );
  }
  if (p.sources) {
    addChip(
      'Source',
      p.sources.length === 1 ? p.sources[0] : `${p.sources.length} sources`,
      () => {
        dom.source.value = '';
        dom.sourceOptions.querySelectorAll('.source-chk').forEach((c) => { c.checked = false; });
        updatePoolLabel();
      },
    );
  }
  if (dom.level.value) {
    addChip('Level', dom.level.value === '__NULL__' ? 'Null' : dom.level.value, () => { dom.level.value = ''; });
  }
  if (p.sentinel) addChip('Sentinel', p.sentinel, () => { dom.sentinel.value = ''; });
  if (dom.geo.value) {
    addChip('Geo', dom.geo.value === '__NULL__' ? 'Null' : (GEO_LABELS[dom.geo.value] || dom.geo.value), () => { dom.geo.value = ''; });
  }
  if (p.strict_eo) addChip('Scope', 'Strict EO', () => { dom.strictEo.checked = false; });
  if (p.has_doi) addChip('DOI', 'Has DOI', () => { dom.hasDoi.checked = false; });
  if (p.tier) addChip('Tier', formatEnumLabel(p.tier), () => { dom.tier.value = ''; });
  if (p.category) addChip('Domain', formatEnumLabel(p.category), () => { dom.category.value = ''; });
  if (p.cadence) addChip('Cadence', p.cadence === '__NULL__' ? 'Null' : formatEnumLabel(p.cadence), () => { dom.cadence.value = ''; });
  if (p.maturity) addChip('Maturity', MATURITY_META[p.maturity]?.label || p.maturity, () => { dom.maturity.value = ''; });
  if (p.ontology) addChip('Ontology', formatEnumLabel(p.ontology), () => { dom.ontology.value = ''; });
  if (p.grid_rule) addChip('Grid', dom.grid.selectedOptions[0]?.textContent || p.grid_rule, () => { dom.grid.value = ''; });
  if (p.covers_year !== null) addChip('Year', String(p.covers_year), () => { dom.year.value = ''; });
  if (p.min_end_year !== null) addChip('Min End', `>= ${p.min_end_year}`, () => { dom.minEndYear.value = ''; });
  if (p.has_url) addChip('URL', 'Has Access URL', () => { dom.hasUrl.checked = false; });
  if (p.exclude_superseded) addChip('Filter', 'Hide Superseded (622)', () => { dom.exclDeprecated.checked = false; });
  if (p.hard_limits) {
    addChip('Mode', 'Hard Limits Verified', () => {
      dom.optHardLimits.checked = false;
      syncHardLimitsToggle(true);
    });
  }

  dom.activeChips.innerHTML = '';
  if (chips.length > 0) {
    dom.activeChipsRow.classList.remove('hidden');
    chips.forEach((ch) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'inline-flex items-center gap-1 rounded-md bg-indigo-50 px-2 py-0.5 text-[11px] font-medium text-indigo-700 ring-1 ring-inset ring-indigo-600/20 hover:bg-indigo-100 transition';
      btn.innerHTML = `<span class="text-indigo-500">${esc(ch.label)}:</span><span>${esc(ch.valText)}</span><span aria-hidden="true" class="ml-0.5 text-indigo-400 font-bold">&times;</span>`;
      btn.addEventListener('click', () => {
        ch.clearFn();
        updateBadgesAndChips();
        if (dom.queryInput.value.trim()) runSearch();
      });
      dom.activeChips.appendChild(btn);
    });
  } else {
    dom.activeChipsRow.classList.add('hidden');
  }
}

function syncUrlFromControls() {
  const params = new URLSearchParams();
  const q = dom.queryInput.value.trim();
  if (q) params.set('q', q);
  if (state.mode !== 'bm25') params.set('mode', state.mode);
  if (dom.toggleReranker.checked) params.set('rerank', '1');
  if (dom.optRerankPool.value !== '25') params.set('rpool', dom.optRerankPool.value);
  if (dom.optPure.checked) params.set('pure', '1');

  const pools = getSelectedPools();
  if (pools.length > 0 && pools.length < dom.poolChecks.length) params.set('pools', pools.join(','));
  const sources = getSelectedSources();
  if (sources.length > 0) params.set('sources', sources.join(','));

  if (dom.level.value) params.set('level', dom.level.value);
  if (dom.sentinel.value) params.set('sentinel', dom.sentinel.value);
  if (dom.geo.value) params.set('geo', dom.geo.value);
  if (dom.strictEo.checked) params.set('strict', '1');
  if (dom.hasDoi.checked) params.set('doi', '1');
  if (dom.tier.value) params.set('tier', dom.tier.value);
  if (dom.category.value) params.set('cat', dom.category.value);
  if (dom.cadence.value) params.set('cad', dom.cadence.value);
  if (dom.maturity.value) params.set('mat', dom.maturity.value);
  if (dom.ontology.value) params.set('ont', dom.ontology.value);
  if (dom.grid.value) params.set('grid', dom.grid.value);
  if (dom.year.value.trim()) params.set('year', dom.year.value.trim());
  if (dom.minEndYear.value.trim()) params.set('min_end', dom.minEndYear.value.trim());
  if (dom.hasUrl.checked) params.set('url', '1');
  if (dom.exclDeprecated.checked) params.set('no_dep', '1');
  if (dom.optTopK.value !== '20') params.set('k', dom.optTopK.value);
  if (dom.optHardLimits.checked) params.set('hard', '1');
  else if (!dom.optNullSafe.checked) params.set('null_safe', '0');
  if (!dom.optCollapse.checked) params.set('collapse', '0');
  if (state.moreOpen) params.set('adv', '1');

  const qs = params.toString();
  window.history.replaceState(null, '', qs ? `?${qs}` : window.location.pathname);
}

function setValidSelect(selectEl, val) {
  if (!val) {
    selectEl.value = '';
    return;
  }
  const ok = Array.from(selectEl.options).some((o) => o.value === val);
  if (ok) selectEl.value = val;
}

function applyUrlToControls() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('q')) dom.queryInput.value = params.get('q') || '';
  if (params.has('mode')) {
    const m = params.get('mode');
    if (['bm25', 'hybrid', 'dense'].includes(m)) setModeUI(m);
  }
  if (params.get('rerank') === '1') {
    dom.toggleReranker.checked = true;
    dom.rerankerOptions.classList.remove('hidden');
  }
  if (params.has('rpool') && ['15', '25', '50'].includes(params.get('rpool'))) {
    dom.optRerankPool.value = params.get('rpool');
  }
  if (params.get('pure') === '1') dom.optPure.checked = true;

  const rawPools = params.get('pools') || params.get('pool') || '';
  if (rawPools) {
    const wanted = new Set(rawPools.split(',').map((s) => s.trim()).filter(Boolean));
    dom.poolChecks.forEach((c) => { c.checked = wanted.has(c.value); });
  }
  const rawSources = params.get('sources') || params.get('source') || '';
  if (rawSources) {
    const wantedSrc = new Set(rawSources.split(',').map((s) => s.trim()).filter(Boolean));
    dom.sourceOptions.querySelectorAll('.source-chk').forEach((c) => { c.checked = wantedSrc.has(c.value); });
    if (wantedSrc.size === 1) setValidSelect(dom.source, Array.from(wantedSrc)[0]);
  }
  updatePoolLabel();

  setValidSelect(dom.level, params.get('level') || '');
  setValidSelect(dom.sentinel, params.get('sentinel') || '');
  setValidSelect(dom.geo, params.get('geo') || '');
  dom.strictEo.checked = params.get('strict') === '1' || params.get('strict_eo') === '1';
  dom.hasDoi.checked = params.get('doi') === '1';

  setValidSelect(dom.tier, params.get('tier') || '');
  setValidSelect(dom.category, params.get('cat') || '');
  setValidSelect(dom.cadence, params.get('cad') || '');
  setValidSelect(dom.maturity, params.get('mat') || '');
  setValidSelect(dom.ontology, params.get('ont') || '');

  const rawGrid = params.get('grid') || '';
  if (rawGrid) {
    const normalizedGrid = /^\d+$/.test(rawGrid) ? `le${rawGrid}` : rawGrid;
    setValidSelect(dom.grid, normalizedGrid);
  }
  const yr = parseOptionalInt(params.get('year'), 1950, 2035);
  if (yr !== null) dom.year.value = String(yr);
  const mey = parseOptionalInt(params.get('min_end'), 1970, 2040);
  if (mey !== null) dom.minEndYear.value = String(mey);

  dom.hasUrl.checked = params.get('url') === '1';
  dom.exclDeprecated.checked = params.get('no_dep') === '1';
  setValidSelect(dom.optTopK, params.get('k') || '20');
  if (params.get('hard') === '1') dom.optHardLimits.checked = true;
  if (params.get('null_safe') === '0') {
    dom.optNullSafe.checked = false;
    state.prevNullSafe = false;
  }
  syncHardLimitsToggle(false);
  if (params.get('collapse') === '0') dom.optCollapse.checked = false;

  const hasAdv = params.get('adv') === '1' || ['source', 'tier', 'cat', 'cad', 'mat', 'ont', 'grid', 'year', 'min_end', 'url', 'no_dep'].some((k) => params.has(k));
  setMorePanelOpen(hasAdv);

  dom.btnClearQuery.classList.toggle('hidden', !dom.queryInput.value);
  updateBadgesAndChips();
}

function triggerEnableSemantic() {
  if (state.semanticReady || state.semanticLoading) return;
  state.semanticLoading = true;
  dom.semanticStatus.textContent = 'Loading EmbeddingGemma 2…';
  dom.semanticProgressWrap.classList.remove('hidden');
  worker.postMessage({ type: 'enableSemantic' });
}

function triggerEnableReranker() {
  if (state.rerankerReady || state.rerankerLoading) return;
  state.rerankerLoading = true;
  dom.rerankerStatus.textContent = 'Downloading model…';
  dom.rerankerProgressWrap.classList.remove('hidden');
  worker.postMessage({ type: 'enableReranker' });
}

function runSearch() {
  if (!state.ready) return;
  const payload = buildSearchPayload();
  updateBadgesAndChips();
  if (!payload.query) {
    dom.queryInput.focus();
    return;
  }
  showError('');
  syncUrlFromControls();

  if ((payload.mode === 'hybrid' || payload.mode === 'dense') && !state.semanticReady) {
    triggerEnableSemantic();
  }
  if (payload.use_reranker && !state.rerankerReady) {
    triggerEnableReranker();
  }

  dom.rerankLiveBanner.classList.add('hidden');
  const busyLabel = payload.mode !== 'bm25' && !state.semanticReady ? 'Loading model…' : 'Searching…';
  setActivityBusy(true, busyLabel);
  worker.postMessage({ type: 'search', payload });
}

function renderResults(data) {
  const rows = data.results || [];
  dom.btnDownload.disabled = rows.length === 0 || state.exporting;

  const modeName = MODE_LABELS[data.mode_used] || data.mode_used;
  const details = [modeName];
  if (data.total_collapsed_in_results > 0) {
    details.push(`${data.total_collapsed_in_results} mirrors grouped`);
  }
  if (data.hard_limits && data.unknown_metadata_counts) {
    const u = data.unknown_metadata_counts;
    details.push(
      `hard limits verified (excluded unknown: grid ${Number(u.grid_meters || 0).toLocaleString()}, unverified grid ${Number(u.unverified_grid_source || 0).toLocaleString()}, level ${Number(u.processing_level || 0).toLocaleString()}, geo ${Number(u.geographic_coverage || 0).toLocaleString()}, end year ${Number(u.temporal_end_year || 0).toLocaleString()})`,
    );
  } else if (!data.null_safe_filter) {
    details.push('missing metadata excluded');
  }

  const supLabel = data.exclude_superseded ? 'active canonical' : 'canonical';
  dom.resultsSummary.textContent = `Showing ${rows.length} of ${data.eligible_after_filter.toLocaleString()} matching ${supLabel} products (${details.join(', ')})`;

  const timingParts = [`Search: ${Number(data.stage1_ms || 0).toFixed(0)} ms`];
  if (data.rerank_ms > 0) timingParts.push(`Rerank: ${(data.rerank_ms / 1000).toFixed(1)} s`);
  if (data.rerank_load_ms > 0) timingParts.push(`Warm-up: ${(data.rerank_load_ms / 1000).toFixed(1)} s`);
  dom.resultsTiming.textContent = timingParts.join(', ');

  if (rows.length === 0) {
    dom.resultsBody.innerHTML = `
      <tr>
        <td colspan="8" class="px-4 py-14 text-center text-sm text-slate-400">
          No matching products found under the current query and filters. Try broadening terms, switching to Hybrid mode, or relaxing filters.
        </td>
      </tr>`;
    return;
  }

  const maxScore = Math.max(...rows.map((r) => (typeof r.score === 'number' && r.score > 0 ? r.score : 0)), 1e-9);

  dom.resultsBody.innerHTML = rows
    .map((r, idx) => {
      const poolMeta = POOL_META[r.provider_pool] || { label: r.provider_pool || '—', cls: 'bg-slate-100 text-slate-600 ring-slate-400/20' };
      const matMeta = MATURITY_META[r.operational_maturity_class];
      const geoLabel = GEO_LABELS[r.geographic_coverage] || (r.geographic_coverage ? r.geographic_coverage : 'null');
      const url = safeHref(r.access_url);
      const doiHref = r.doi ? safeHref(r.doi.startsWith('http') ? r.doi : `https://doi.org/${r.doi}`) : null;

      let rankShift = '';
      if (r.rerank_rank !== null && r.rerank_rank !== undefined && r.stage1_rank) {
        const diff = r.stage1_rank - r.rank;
        if (diff > 0) {
          rankShift = `<div class="text-[10px] font-medium text-emerald-700 tabular-nums" title="Moved up from initial rank #${r.stage1_rank}">&uarr;${diff} <span class="text-slate-400">was #${r.stage1_rank}</span></div>`;
        } else if (diff < 0) {
          rankShift = `<div class="text-[10px] font-medium text-slate-400 tabular-nums" title="Moved down from initial rank #${r.stage1_rank}">&darr;${Math.abs(diff)} was #${r.stage1_rank}</div>`;
        } else {
          rankShift = `<div class="text-[10px] text-slate-400 tabular-nums">unchanged</div>`;
        }
      }

      const relPct = Math.max(8, Math.min(100, Math.round(((r.score > 0 ? r.score : 0) / maxScore) * 100)));

      const sentinelCell = r.uses_sentinel
        ? `<div class="font-medium text-emerald-700">${esc(r.sentinel_missions || 'Sentinel')}</div>`
        : '<span class="text-slate-400">—</span>';

      const mirrorsBadge = r.collapsed_siblings > 0
        ? `<span class="inline-flex items-center rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-medium text-indigo-700 ring-1 ring-inset ring-indigo-600/20">+${r.collapsed_siblings} mirror${r.collapsed_siblings === 1 ? '' : 's'}</span>`
        : '';

      return `
        <tr data-idx="${idx}" class="result-row hover:bg-indigo-50/40 cursor-pointer transition-colors">
          <td class="py-3 pl-4 pr-2 align-top whitespace-nowrap">
            <div class="text-sm font-semibold text-slate-900 tabular-nums">#${r.rank}</div>
            <div class="mt-1 h-1 w-10 rounded-full bg-slate-100 overflow-hidden" title="Relative score ${esc(r.score)}">
              <div class="h-full rounded-full bg-indigo-600" style="width: ${relPct}%;"></div>
            </div>
            ${rankShift}
          </td>
          <td class="px-3 py-3 align-top">
            <div class="flex items-start gap-2 flex-wrap">
              <span class="font-semibold text-slate-900 leading-snug">${esc(r.title)}</span>
              ${mirrorsBadge}
            </div>
            <div class="mt-0.5 text-[11px] text-slate-500 flex items-center flex-wrap gap-x-2">
              <span>${esc(r.producing_institution || r.source_catalogue)}</span>
              <span class="font-mono text-[10px] text-slate-400">${esc(r.source_catalogue)}</span>
            </div>
            ${r.snippet ? `<p class="mt-1 text-[11px] text-slate-600 leading-relaxed line-clamp-2">${esc(r.snippet)}</p>` : ''}
          </td>
          <td class="px-3 py-3 align-top whitespace-nowrap">
            ${r.processing_level ? badge(r.processing_level, 'bg-slate-100 text-slate-800 ring-slate-500/20 font-mono') : '<span class="text-slate-400 italic">null</span>'}
          </td>
          <td class="px-3 py-3 align-top whitespace-nowrap">
            ${badge(poolMeta.label, poolMeta.cls)}
            ${r.copernicus_service ? `<div class="mt-1 text-[10px] font-mono text-slate-500">${esc(r.copernicus_service)}</div>` : ''}
          </td>
          <td class="px-3 py-3 align-top whitespace-nowrap">${sentinelCell}</td>
          <td class="px-3 py-3 align-top whitespace-nowrap">
            <div class="font-medium text-slate-800 tabular-nums">${r.grid ? esc(r.grid) : '<span class="text-slate-400 italic font-normal">null</span>'}</div>
            <div class="text-[11px] text-slate-500">${esc(r.cadence ? formatEnumLabel(r.cadence) : '—')}</div>
          </td>
          <td class="px-3 py-3 align-top whitespace-nowrap">
            <div class="text-slate-800">${esc(geoLabel)}</div>
            <div class="mt-0.5 flex items-center gap-1.5">
              <span class="text-[11px] text-slate-500 tabular-nums">${esc(formatSpan(r))}</span>
              ${matMeta ? badge(matMeta.label, matMeta.cls) : ''}
            </div>
          </td>
          <td class="py-3 pl-3 pr-4 align-top text-right whitespace-nowrap space-x-1.5">
            ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()" class="inline-flex items-center rounded-md bg-white px-2 py-1 text-[11px] font-medium text-indigo-700 ring-1 ring-inset ring-indigo-200 hover:bg-indigo-50">Access</a>` : ''}
            ${doiHref ? `<a href="${esc(doiHref)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()" class="inline-flex items-center rounded-md bg-white px-2 py-1 text-[11px] font-medium text-slate-700 ring-1 ring-inset ring-slate-200 hover:bg-slate-50">DOI</a>` : ''}
          </td>
        </tr>`;
    })
    .join('');

  dom.resultsBody.querySelectorAll('.result-row').forEach((tr) => {
    tr.addEventListener('click', () => openDetailForResult(Number(tr.dataset.idx)));
  });
}

function destroyBboxMap() {
  if (state.mapResizeTimer) {
    clearTimeout(state.mapResizeTimer);
    state.mapResizeTimer = null;
  }
  if (state.map) {
    try { state.map.remove(); } catch { /* ignore */ }
    state.map = null;
  }
}

function mountBboxMap(bbox, crossesAntimeridian) {
  destroyBboxMap();
  const container = document.getElementById('drawer-bbox-map');
  if (!container) return;
  if (!bbox || bbox.length !== 4 || bbox.some((v) => v === null || Number.isNaN(Number(v)))) {
    container.innerHTML = '<div class="h-full flex items-center justify-center text-[11px] text-slate-400 italic">No bounding box provided by upstream API.</div>';
    return;
  }

  const [wRaw, sRaw, eRaw, nRaw] = bbox.map(Number);
  const clampLat = (lat) => Math.max(-85, Math.min(85, lat));
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

  try {
    const map = new MapLibreMap({
      container,
      style: 'https://tiles.openfreemap.org/styles/positron',
      center: isAntim ? [(w + (e + 360)) / 2 - 360, (s + n) / 2] : [(w + e) / 2, isNearGlobal ? 14 : (s + n) / 2],
      zoom: isNearGlobal ? 0.55 : 1.5,
      scrollZoom: false,
      attributionControl: { compact: true },
    });
    state.map = map;
    map.addControl(new NavigationControl({ showCompass: false }), 'top-right');

    map.on('load', () => {
      if (state.map !== map) return;
      map.addSource('product-bbox', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: geom, properties: {} }] },
      });
      map.addLayer({
        id: 'product-bbox-fill',
        type: 'fill',
        source: 'product-bbox',
        paint: { 'fill-color': '#4f46e5', 'fill-opacity': 0.18 },
      });
      map.addLayer({
        id: 'product-bbox-line',
        type: 'line',
        source: 'product-bbox',
        paint: { 'line-color': '#4338ca', 'line-width': 2 },
      });
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
      // Non-fatal tile/network error
    });
    state.mapResizeTimer = setTimeout(() => {
      if (state.map === map) map.resize();
    }, 220);
  } catch (err) {
    container.innerHTML = `<div class="h-full flex items-center justify-center text-[11px] text-slate-500 px-3 text-center">Map unavailable (${esc(err.message)}). Bounds: W ${wRaw}°, S ${sRaw}°, E ${eRaw}°, N ${nRaw}°.</div>`;
  }
}

function fetchProductDetail(recordId) {
  if (state.productCache.has(recordId)) return Promise.resolve(state.productCache.get(recordId));
  return new Promise((resolve, reject) => {
    const reqId = ++state.detailReqSeq;
    state.detailPending.set(reqId, { recordId, resolve, reject });
    worker.postMessage({ type: 'getProductDetail', recordId, reqId });
  });
}

async function openDetailForResult(idx) {
  const r = state.results[idx];
  if (!r) return;
  state.activeIdx = idx;
  await openDetailById(r.record_id, r);
}

async function openDetailById(recordId, rankCtx = null) {
  dom.drawerBackdrop.classList.remove('hidden');
  dom.drawer.classList.remove('translate-x-full');
  dom.drawerPrev.disabled = state.activeIdx === null || state.activeIdx <= 0;
  dom.drawerNext.disabled = state.activeIdx === null || state.activeIdx >= state.results.length - 1;

  dom.drawerBadges.innerHTML = '';
  dom.drawerTitle.textContent = rankCtx ? rankCtx.title : 'Loading product…';
  dom.drawerSubtitle.textContent = recordId;
  dom.drawerBody.innerHTML = '<div class="py-12 text-center text-slate-400">Loading full provenance…</div>';

  try {
    const det = await fetchProductDetail(recordId);
    const poolMeta = POOL_META[det.provider_pool] || { label: det.provider_pool || 'Unknown', cls: 'bg-slate-100 text-slate-700 ring-slate-500/20' };
    const matMeta = MATURITY_META[det.operational_maturity_class];

    const badges = [
      rankCtx ? badge(`Rank #${rankCtx.rank}`, 'bg-slate-900 text-white ring-slate-900') : '',
      det.processing_level ? badge(det.processing_level, 'bg-slate-100 text-slate-800 ring-slate-500/20 font-mono') : '',
      badge(poolMeta.label, poolMeta.cls),
      matMeta ? badge(matMeta.label, matMeta.cls) : '',
      det.strict_eo_c1_eligible ? badge('Strict EO', 'bg-emerald-50 text-emerald-700 ring-emerald-600/20') : badge('Sensitivity / Hybrid', 'bg-amber-50 text-amber-800 ring-amber-600/20'),
    ].filter(Boolean);

    dom.drawerBadges.innerHTML = badges.join('');
    dom.drawerTitle.textContent = det.title || recordId;
    dom.drawerSubtitle.textContent = det.record_id || recordId;

    const accessHref = safeHref(det.access_url);
    const doiHref = det.doi ? safeHref(det.doi.startsWith('http') ? det.doi : `https://doi.org/${det.doi}`) : null;
    const kwList = Array.from(
      new Set(
        String(det.keywords || '')
          .split(/[;|,]/)
          .map((s) => s.trim())
          .filter((s) => s.length >= 2 && s.length <= 60),
      ),
    );

    const rankingBlock = rankCtx
      ? `
      <section class="space-y-2">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Retrieval Scoring Signals</h3>
        <dl class="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-indigo-50/50 rounded-lg p-3 border border-indigo-100">
          ${kvItem('Final Score', rankCtx.score, true)}
          ${kvItem('Initial Rank', rankCtx.stage1_rank ? `#${rankCtx.stage1_rank}` : '—', true)}
          ${kvItem('Semantic Cosine / Rank', rankCtx.dense_score !== null && rankCtx.dense_score !== undefined ? `${rankCtx.dense_score} (#${rankCtx.dense_rank})` : '—', true)}
          ${kvItem('BM25 Score / Rank', rankCtx.bm25_rank ? `${rankCtx.bm25_score} (#${rankCtx.bm25_rank})` : '—', true)}
          ${rankCtx.rerank_score !== null && rankCtx.rerank_score !== undefined ? kvItem('Qwen3 Margin / Prob', `${rankCtx.rerank_score} (${Math.round((rankCtx.rerank_prob || 0) * 100)}%, #${rankCtx.rerank_rank})`, true) : ''}
        </dl>
      </section>`
      : '';

    const collapsedBlock =
      rankCtx && rankCtx.collapsed_items && rankCtx.collapsed_items.length > 0
        ? `
      <section class="space-y-2">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Grouped Mirrors and Editions (${rankCtx.collapsed_siblings})</h3>
        <div class="divide-y divide-slate-100 rounded-lg border border-slate-200/80 bg-slate-50/50">
          ${rankCtx.collapsed_items
            .map(
              (s) => `
            <button type="button" data-sib-id="${esc(s.record_id)}" class="sib-btn w-full text-left px-3 py-2 hover:bg-indigo-50/70 transition flex items-center justify-between gap-2">
              <div class="min-w-0">
                <div class="text-xs font-medium text-slate-800 truncate">${esc(s.title)}</div>
                <div class="text-[10px] font-mono text-slate-400 truncate">${esc(s.source_catalogue)} (${esc(s.record_id)})</div>
              </div>
              <span class="text-[11px] text-indigo-600 shrink-0">Inspect &rarr;</span>
            </button>`,
            )
            .join('')}
        </div>
      </section>`
        : '';

    dom.drawerBody.innerHTML = `
      <div class="flex flex-wrap items-center gap-2">
        ${accessHref ? `<a href="${esc(accessHref)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800 transition">Open Access URL &nearr;</a>` : ''}
        ${doiHref ? `<a href="${esc(doiHref)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 rounded-lg bg-indigo-50 px-3 py-1.5 text-xs font-medium text-indigo-700 ring-1 ring-inset ring-indigo-600/20 hover:bg-indigo-100 transition">DOI: ${esc(det.doi)}</a>` : ''}
        <button id="btn-copy-id" type="button" class="inline-flex items-center gap-1.5 rounded-lg bg-white px-3 py-1.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 transition">Copy Record ID</button>
      </div>

      ${rankingBlock}

      <section class="space-y-1.5">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Description</h3>
        <p class="text-xs text-slate-700 leading-relaxed whitespace-pre-line bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">${esc(det.description || det.desc_snippet || 'No description provided by upstream catalogue.')}</p>
      </section>

      ${collapsedBlock}

      <section class="space-y-2">
        <div class="flex items-center justify-between">
          <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Spatial Extent and Resolution</h3>
          ${det.bbox ? `<span class="text-[10px] font-mono text-slate-400 tabular-nums">W ${det.bbox[0]}°, S ${det.bbox[1]}°, E ${det.bbox[2]}°, N ${det.bbox[3]}°</span>` : ''}
        </div>
        <div id="drawer-bbox-map" class="h-48 w-full rounded-lg border border-slate-200 bg-slate-100 overflow-hidden"></div>
        <dl class="grid grid-cols-2 sm:grid-cols-3 gap-3 bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">
          ${kvItem('Geographic Class', det.geographic_coverage_class)}
          ${kvItem('Observation Domain', det.observation_domain_class)}
          ${kvItem('Adjudicated Grid', det.grid_human)}
          ${kvItem('Raw Spatial String', det.spatial_resolution_raw)}
          ${kvItem('Grid Source', det.adjudicated_spatial_resolution_source)}
          ${kvItem('Footprint (m)', det.spatial_footprint_meters)}
        </dl>
      </section>

      <section class="space-y-2">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Institutional Attribution and Sensors</h3>
        <dl class="grid grid-cols-1 sm:grid-cols-2 gap-3 bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">
          ${kvItem('Producing Institution', det.producing_institution)}
          ${kvItem('Hosting Platform', det.hosting_platform)}
          ${kvItem('Jurisdiction Tier', det.provider_jurisdiction_tier)}
          ${kvItem('Copernicus Service', det.copernicus_service)}
          ${kvItem('Ontology Class', det.product_ontology_class)}
          ${kvItem('Output Category', det.product_output_category)}
          ${kvItem('Sentinel Lineage', det.sentinel_lineage_status)}
          ${kvItem('Sentinel Missions', det.sentinel_missions)}
          ${kvItem('Platform / Sensor Raw', det.platform_sensor_raw)}
          ${kvItem('Temporal Cadence', det.temporal_cadence_class)}
          ${kvItem('Temporal Window', `${det.temporal_start_raw || 'null'} to ${det.temporal_end_raw || 'null'}`)}
          ${kvItem('License', det.license)}
        </dl>
      </section>

      ${
        kwList.length > 0
          ? `<section class="space-y-1.5">
              <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Keywords (click to search)</h3>
              <div class="flex flex-wrap gap-1.5">
                ${kwList.slice(0, 35).map((kw) => `<button type="button" data-kw="${esc(kw)}" class="kw-chip rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] text-slate-700 hover:bg-indigo-100 hover:text-indigo-800 transition">${esc(kw)}</button>`).join('')}
              </div>
            </section>`
          : ''
      }
    `;

    mountBboxMap(det.bbox, det.antimeridian);

    const copyBtn = document.getElementById('btn-copy-id');
    copyBtn?.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(det.record_id || recordId);
        copyBtn.textContent = 'Copied';
        setTimeout(() => { copyBtn.textContent = 'Copy Record ID'; }, 1200);
      } catch {
        copyBtn.textContent = 'Clipboard blocked';
      }
    });

    dom.drawerBody.querySelectorAll('.sib-btn').forEach((btn) => {
      btn.addEventListener('click', () => openDetailById(btn.dataset.sibId, null));
    });

    dom.drawerBody.querySelectorAll('.kw-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        dom.queryInput.value = chip.dataset.kw || '';
        dom.btnClearQuery.classList.toggle('hidden', !dom.queryInput.value);
        closeDetailDrawer();
        runSearch();
      });
    });
  } catch (err) {
    dom.drawerBody.innerHTML = `<div class="rounded-lg border border-rose-200 bg-rose-50 p-4 text-rose-800">Failed to load product details: ${esc(err.message)}</div>`;
  }
}

function closeDetailDrawer() {
  destroyBboxMap();
  state.activeIdx = null;
  dom.drawer.classList.add('translate-x-full');
  dom.drawerBackdrop.classList.add('hidden');
}

function setDownloadMenuOpen(open) {
  dom.downloadMenu.classList.toggle('hidden', !open);
  dom.btnDownload.setAttribute('aria-expanded', open ? 'true' : 'false');
}

async function handleExportFormat(fmt) {
  setDownloadMenuOpen(false);
  if (!state.results.length || state.exporting) return;
  state.exporting = true;
  dom.btnDownload.disabled = true;
  dom.btnDownloadLabel.textContent = 'Exporting…';
  showError('');
  try {
    const metadata = state.lastPayload
      ? {
          query: state.lastPayload.query,
          mode_used: state.lastPayload.mode_used,
          reranker_used: state.lastPayload.reranker_used,
          device: state.lastPayload.device,
          eligible_after_filter: state.lastPayload.eligible_after_filter,
          stage1_ms: state.lastPayload.stage1_ms,
          rerank_ms: state.lastPayload.rerank_ms,
        }
      : null;
    exportResults(fmt, state.results, metadata);
  } catch (err) {
    showError(`${fmt.toUpperCase()} export failed: ${err.message}`);
  } finally {
    state.exporting = false;
    dom.btnDownload.disabled = state.results.length === 0;
    dom.btnDownloadLabel.textContent = 'Download';
  }
}

worker.addEventListener('error', (ev) => {
  state.searching = false;
  state.reranking = false;
  state.semanticLoading = false;
  state.rerankerLoading = false;
  dom.rerankLiveBanner.classList.add('hidden');
  setActivityBusy(false);
  showError(`Search worker failed: ${ev.message || 'unexpected runtime error'}. Reload the page to reinitialize.`);
});

worker.addEventListener('messageerror', () => {
  setActivityBusy(false);
  showError('Search worker message deserialization error.');
});

worker.addEventListener('message', (ev) => {
  const msg = ev.data || {};
  if (msg.type === 'initProgress') {
    dom.loadProgressBar.style.width = `${msg.pct}%`;
    dom.resultsSummary.textContent = msg.message;
  } else if (msg.type === 'ready') {
    state.ready = true;
    state.webgpuAvailable = Boolean(msg.webgpuAvailable);
    state.manifest = msg.manifest;

    dom.toggleReranker.disabled = false;
    dom.devicePill.textContent = state.webgpuAvailable ? ' using WebGPU' : ' using CPU';
    dom.devicePill.classList.remove('hidden');
    if (!state.webgpuAvailable) {
      dom.rerankerWebgpuNote.classList.remove('hidden');
    }

    if (msg.enums) {
      const sortedSources = [...(msg.enums.source_catalogue || [])].sort(sortByNumericPrefix);
      populateSelectOptions(dom.source, sortedSources, (v) => v, false);
      populateSelectOptions(dom.tier, msg.enums.provider_jurisdiction_tier || [], formatEnumLabel, false);
      populateSelectOptions(dom.category, msg.enums.product_output_category || [], formatEnumLabel, false);
      populateSelectOptions(dom.cadence, msg.enums.temporal_cadence_class || [], formatEnumLabel, true);
      populateSelectOptions(dom.maturity, msg.enums.operational_maturity_class || [], (v) => MATURITY_META[v]?.label || formatEnumLabel(v), false);
      populateSelectOptions(dom.ontology, msg.enums.product_ontology_class || [], formatEnumLabel, false);
    }

    if (msg.manifest.pool_counts) {
      document.querySelectorAll('[data-pool-count]').forEach((el) => {
        const cnt = msg.manifest.pool_counts[el.dataset.poolCount];
        el.textContent = cnt !== undefined ? Number(cnt).toLocaleString() : '';
      });
    }
    if (msg.manifest.source_counts) {
      renderSourceOptions(msg.manifest.source_counts);
    }

    updateHeaderBadge();
    setActivityBusy(false);
    const totalStr = Number(msg.manifest.n_docs).toLocaleString();
    dom.resultsSummary.textContent = `Ready to search all ${totalStr} canonical L3/L4 products.`;
    dom.resultsBody.innerHTML = `
      <tr>
        <td colspan="8" class="px-4 py-14 text-center text-sm text-slate-400">
          Keyword index ready across all ${totalStr} products (${(msg.manifest.bm25_gz_bytes / 1e6).toFixed(1)} MB). Enter a query above or switch to <strong>Hybrid</strong> for semantic ranking.
        </td>
      </tr>`;

    applyUrlToControls();
    if (state.mode !== 'bm25') triggerEnableSemantic();
    if (dom.toggleReranker.checked) triggerEnableReranker();
    if (dom.queryInput.value.trim()) runSearch();
  } else if (msg.type === 'modelProgress') {
    if (msg.target === 'semantic') {
      dom.semanticProgressWrap.classList.remove('hidden');
      dom.semanticProgressMsg.textContent = msg.message;
      dom.semanticProgressPct.textContent = `${msg.pct}%`;
      dom.semanticProgressBar.style.width = `${msg.pct}%`;
    } else if (msg.target === 'reranker') {
      if (!dom.toggleReranker.checked) return;
      dom.rerankerProgressWrap.classList.remove('hidden');
      dom.rerankerProgressMsg.textContent = msg.message;
      dom.rerankerProgressPct.textContent = `${msg.pct}%`;
      dom.rerankerProgressBar.style.width = `${msg.pct}%`;
    }
  } else if (msg.type === 'semanticReady') {
    state.semanticReady = true;
    state.semanticLoading = false;
    state.semanticDevice = msg.device;
    const devLabel = msg.device === 'webgpu' ? 'WebGPU' : 'CPU';
    dom.semanticStatus.textContent = `EmbeddingGemma 2 loaded (${devLabel})`;
    dom.semanticStatus.className = 'text-[11px] text-emerald-400';
    dom.semanticProgressWrap.classList.add('hidden');
    updateHeaderBadge();
    if (dom.queryInput.value.trim() && !state.searching) {
      runSearch();
    }
  } else if (msg.type === 'rerankerReady') {
    state.rerankerReady = true;
    state.rerankerLoading = false;
    state.rerankerDevice = msg.device;
    const devLabel = msg.device === 'webgpu' ? 'WebGPU' : 'CPU';
    dom.rerankerStatus.textContent = `Qwen3-0.6B loaded (${devLabel})`;
    dom.rerankerStatus.className = 'text-[11px] text-emerald-400';
    dom.rerankerProgressWrap.classList.add('hidden');
    updateHeaderBadge();
  } else if (msg.type === 'rerankProgress') {
    if (msg.searchSeq < state.latestSearchSeq) return;
    state.reranking = true;
    dom.rerankLiveBanner.classList.remove('hidden');
    dom.rerankLiveText.textContent = `Reranking candidate ${msg.done} of ${msg.total} with Qwen3-0.6B (~${msg.etaSec}s remaining)…`;
  } else if (msg.type === 'rerankAborted') {
    if (msg.searchSeq < state.latestSearchSeq) return;
    state.reranking = false;
    dom.rerankLiveBanner.classList.add('hidden');
    setActivityBusy(false);
  } else if (msg.type === 'searchResult') {
    if (msg.searchSeq < state.latestSearchSeq) return;
    state.latestSearchSeq = msg.searchSeq;
    state.lastPayload = msg;
    state.results = msg.results || [];
    renderResults(msg);
    if (msg.reranking_pending) {
      state.reranking = true;
      dom.rerankLiveBanner.classList.remove('hidden');
      dom.rerankLiveText.textContent = state.rerankerReady
        ? `Reranking top ${msg.rerank_pool} candidates with Qwen3-0.6B…`
        : `Initial results ready; downloading Qwen3-0.6B (~995 MB) to rerank top ${msg.rerank_pool}…`;
      setActivityBusy(true, 'Reranking…');
    } else {
      state.reranking = false;
      dom.rerankLiveBanner.classList.add('hidden');
      setActivityBusy(false);
    }
  } else if (msg.type === 'productDetail') {
    const pending = state.detailPending.get(msg.reqId);
    if (pending) {
      state.detailPending.delete(msg.reqId);
      if (msg.error) pending.reject(new Error(msg.error));
      else {
        state.productCache.set(pending.recordId, msg.detail);
        pending.resolve(msg.detail);
      }
    }
  } else if (msg.type === 'error') {
    if (msg.source === 'getProductDetail' && msg.reqId) {
      const pending = state.detailPending.get(msg.reqId);
      if (pending) {
        state.detailPending.delete(msg.reqId);
        pending.reject(new Error(msg.message));
      }
      return;
    }
    if (msg.source === 'enableSemantic') {
      state.semanticLoading = false;
      dom.semanticStatus.textContent = 'Model download failed';
      dom.semanticStatus.className = 'text-[11px] text-rose-300';
      dom.semanticProgressWrap.classList.add('hidden');
      setModeUI('bm25');
    } else if (msg.source === 'enableReranker') {
      state.rerankerLoading = false;
      dom.toggleReranker.checked = false;
      dom.rerankerOptions.classList.add('hidden');
      dom.rerankerStatus.textContent = 'Model download failed';
      dom.rerankerProgressWrap.classList.add('hidden');
    }
    state.reranking = false;
    dom.rerankLiveBanner.classList.add('hidden');
    setActivityBusy(false);
    showError(`Browser search error (${msg.source || 'worker'}): ${msg.message}`);
  }
});

function wireEvents() {
  dom.form.addEventListener('submit', (e) => {
    e.preventDefault();
    runSearch();
  });

  dom.queryInput.addEventListener('input', () => {
    dom.btnClearQuery.classList.toggle('hidden', !dom.queryInput.value);
  });

  dom.btnClearQuery.addEventListener('click', () => {
    dom.queryInput.value = '';
    dom.btnClearQuery.classList.add('hidden');
    dom.queryInput.focus();
  });

  dom.modeTabs.forEach((btn) => {
    btn.addEventListener('click', () => {
      const nextMode = btn.dataset.mode;
      if (!nextMode || nextMode === state.mode) return;
      setModeUI(nextMode);
      if ((nextMode === 'hybrid' || nextMode === 'dense') && !state.semanticReady) {
        triggerEnableSemantic();
      }
      if (dom.queryInput.value.trim()) runSearch();
      else syncUrlFromControls();
    });
  });

  dom.toggleReranker.addEventListener('change', () => {
    const on = dom.toggleReranker.checked;
    dom.rerankerOptions.classList.toggle('hidden', !on);
    if (on) {
      if (!state.rerankerReady) {
        triggerEnableReranker();
      }
      if (dom.queryInput.value.trim()) runSearch();
    } else {
      worker.postMessage({ type: 'cancelRerank' });
      dom.rerankLiveBanner.classList.add('hidden');
      dom.rerankerProgressWrap.classList.add('hidden');
      if (dom.queryInput.value.trim()) runSearch();
    }
    updateHeaderBadge();
  });

  dom.btnCancelRerank.addEventListener('click', () => {
    worker.postMessage({ type: 'cancelRerank' });
    dom.rerankLiveBanner.classList.add('hidden');
    setActivityBusy(false);
  });

  dom.btnToggleMore.addEventListener('click', () => {
    setMorePanelOpen(!state.moreOpen);
    syncUrlFromControls();
  });

  dom.exampleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      dom.queryInput.value = btn.dataset.q || '';
      setValidSelect(dom.geo, btn.dataset.geo || '');
      setValidSelect(dom.grid, btn.dataset.grid || '');
      dom.btnClearQuery.classList.remove('hidden');
      runSearch();
    });
  });

  dom.source.addEventListener('change', () => {
    const val = dom.source.value;
    dom.sourceOptions.querySelectorAll('.source-chk').forEach((c) => {
      c.checked = val !== '' && c.value === val;
    });
    updatePoolLabel();
    updateBadgesAndChips();
    if (dom.queryInput.value.trim()) runSearch();
  });

  [
    dom.level,
    dom.sentinel,
    dom.geo,
    dom.strictEo,
    dom.hasDoi,
    dom.tier,
    dom.category,
    dom.cadence,
    dom.maturity,
    dom.ontology,
    dom.grid,
    dom.year,
    dom.minEndYear,
    dom.hasUrl,
    dom.exclDeprecated,
    dom.optTopK,
    dom.optNullSafe,
    dom.optCollapse,
    dom.optRerankPool,
    dom.optPure,
  ].forEach((el) => {
    el.addEventListener('change', () => {
      updateBadgesAndChips();
      if (dom.queryInput.value.trim()) runSearch();
    });
  });

  dom.optHardLimits.addEventListener('change', () => {
    syncHardLimitsToggle(true);
    updateBadgesAndChips();
    if (dom.queryInput.value.trim()) runSearch();
  });

  dom.btnPool.addEventListener('click', (e) => {
    e.stopPropagation();
    setPoolMenuOpen(dom.poolMenu.classList.contains('hidden'));
  });
  dom.poolChecks.forEach((chk) => {
    chk.addEventListener('change', () => {
      updatePoolLabel();
      updateBadgesAndChips();
      if (dom.queryInput.value.trim()) runSearch();
    });
  });
  dom.btnPoolAll.addEventListener('click', () => {
    dom.poolChecks.forEach((c) => { c.checked = false; });
    dom.source.value = '';
    dom.sourceOptions.querySelectorAll('.source-chk').forEach((c) => { c.checked = false; });
    updatePoolLabel();
    updateBadgesAndChips();
    if (dom.queryInput.value.trim()) runSearch();
  });
  dom.btnSourcesClear.addEventListener('click', () => {
    dom.source.value = '';
    dom.sourceOptions.querySelectorAll('.source-chk').forEach((c) => { c.checked = false; });
    updatePoolLabel();
    updateBadgesAndChips();
    if (dom.queryInput.value.trim()) runSearch();
  });

  dom.btnReset.addEventListener('click', () => {
    dom.poolChecks.forEach((c) => { c.checked = false; });
    dom.source.value = '';
    dom.sourceOptions.querySelectorAll('.source-chk').forEach((c) => { c.checked = false; });
    updatePoolLabel();
    dom.level.value = '';
    dom.sentinel.value = '';
    dom.geo.value = '';
    dom.strictEo.checked = false;
    dom.hasDoi.checked = false;
    dom.tier.value = '';
    dom.category.value = '';
    dom.cadence.value = '';
    dom.maturity.value = '';
    dom.ontology.value = '';
    dom.grid.value = '';
    dom.year.value = '';
    dom.minEndYear.value = '';
    dom.hasUrl.checked = false;
    dom.exclDeprecated.checked = false;
    dom.optTopK.value = '20';
    state.prevNullSafe = true;
    dom.optNullSafe.checked = true;
    dom.optHardLimits.checked = false;
    syncHardLimitsToggle(false);
    dom.optCollapse.checked = true;
    dom.optPure.checked = false;
    dom.optRerankPool.value = '25';
    setModeUI(state.semanticReady ? 'hybrid' : 'bm25');
    updateBadgesAndChips();
    if (dom.queryInput.value.trim()) runSearch();
  });

  dom.btnShareUrl.addEventListener('click', async () => {
    syncUrlFromControls();
    try {
      await navigator.clipboard.writeText(window.location.href);
      dom.btnShareLabel.textContent = 'Copied!';
      setTimeout(() => { dom.btnShareLabel.textContent = 'Share URL'; }, 1400);
    } catch {
      dom.btnShareLabel.textContent = 'Copy failed';
      setTimeout(() => { dom.btnShareLabel.textContent = 'Share URL'; }, 1400);
    }
  });

  dom.btnDownload.addEventListener('click', (e) => {
    e.stopPropagation();
    const willOpen = dom.downloadMenu.classList.contains('hidden');
    setDownloadMenuOpen(willOpen);
    if (willOpen && dom.downloadOpts.length) dom.downloadOpts[0].focus();
  });
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
    if (!dom.downloadWrap.contains(e.target)) setDownloadMenuOpen(false);
    if (!dom.poolWrap.contains(e.target)) setPoolMenuOpen(false);
  });

  dom.drawerClose.addEventListener('click', closeDetailDrawer);
  dom.drawerBackdrop.addEventListener('click', closeDetailDrawer);
  dom.drawerPrev.addEventListener('click', () => {
    if (state.activeIdx !== null && state.activeIdx > 0) openDetailForResult(state.activeIdx - 1);
  });
  dom.drawerNext.addEventListener('click', () => {
    if (state.activeIdx !== null && state.activeIdx < state.results.length - 1) openDetailForResult(state.activeIdx + 1);
  });

  const openInfo = () => dom.infoModal.classList.remove('hidden');
  const closeInfo = () => dom.infoModal.classList.add('hidden');
  dom.btnInfo.addEventListener('click', openInfo);
  dom.infoClose.addEventListener('click', closeInfo);
  dom.infoOk.addEventListener('click', closeInfo);
  dom.infoModal.addEventListener('click', (e) => { if (e.target === dom.infoModal) closeInfo(); });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!dom.poolMenu.classList.contains('hidden')) setPoolMenuOpen(false);
      else if (!dom.downloadMenu.classList.contains('hidden')) setDownloadMenuOpen(false);
      else if (!dom.infoModal.classList.contains('hidden')) closeInfo();
      else if (!dom.drawer.classList.contains('translate-x-full')) closeDetailDrawer();
      return;
    }
    if (e.key === '/' && document.activeElement !== dom.queryInput && !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      e.preventDefault();
      dom.queryInput.focus();
      return;
    }
    if (!dom.drawer.classList.contains('translate-x-full') && state.activeIdx !== null) {
      if (e.key === 'ArrowUp' && state.activeIdx > 0) {
        e.preventDefault();
        openDetailForResult(state.activeIdx - 1);
      } else if (e.key === 'ArrowDown' && state.activeIdx < state.results.length - 1) {
        e.preventDefault();
        openDetailForResult(state.activeIdx + 1);
      }
    }
  });
}

wireEvents();

const normalizedPageUrl =
  window.location.pathname.endsWith('/') || window.location.pathname.endsWith('.html')
    ? window.location.href
    : `${window.location.origin}${window.location.pathname}/${window.location.search}${window.location.hash}`;
const baseDirUrl = new URL('../', normalizedPageUrl).href;
worker.postMessage({ type: 'init', baseUrl: baseDirUrl });
