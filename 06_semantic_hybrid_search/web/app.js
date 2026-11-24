'use strict';

const state = {
  status: null,
  lastPayload: null,
  results: [],
  activeIdx: null,
  searching: false,
  exporting: false,
  map: null,
  mapResizeTimer: null,
  productCache: new Map(),
};

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

const MODE_LABELS = {
  'jina-v3.5': 'Jina v3.5 Top-50 Blend',
  'jina-v3.5-fast': 'Jina v3.5 Top-25 Pure',
  'qwen3-0.6b': 'Qwen3-0.6B Top-50 Blend',
  hybrid: 'Stage-1 Hybrid RRF',
  dense: 'Dense 768-d Only',
  bm25: 'BM25 Lexical Only',
};

const dom = {
  statusBadge: document.getElementById('status-badge'),
  activityBar: document.getElementById('activity-bar'),
  modeBanner: document.getElementById('mode-banner'),
  errorBanner: document.getElementById('error-banner'),

  form: document.getElementById('search-form'),
  queryInput: document.getElementById('query-input'),
  btnClearQuery: document.getElementById('btn-clear-query'),
  btnSearch: document.getElementById('btn-search'),
  btnSearchLabel: document.getElementById('btn-search-label'),

  optMode: document.getElementById('opt-mode'),
  optPool: document.getElementById('opt-pool'),
  optGeo: document.getElementById('opt-geo'),
  optGrid: document.getElementById('opt-grid'),
  optLevel: document.getElementById('opt-level'),
  optTopK: document.getElementById('opt-topk'),
  optStrictEo: document.getElementById('opt-strict-eo'),
  optNullSafe: document.getElementById('opt-null-safe'),
  optCollapse: document.getElementById('opt-collapse'),
  optPure: document.getElementById('opt-pure'),
  btnReset: document.getElementById('btn-reset'),
  exampleBtns: Array.from(document.querySelectorAll('.example-btn')),

  resultsSummary: document.getElementById('results-summary'),
  resultsTiming: document.getElementById('results-timing'),
  resultsBody: document.getElementById('results-body'),
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

function setSearching(isBusy, label = 'Searching…') {
  state.searching = isBusy;
  dom.btnSearch.disabled = isBusy;
  dom.btnSearchLabel.textContent = isBusy ? label : 'Search';
  dom.activityBar.classList.toggle('hidden', !isBusy);
}

function formatSpan(r) {
  const t0 = r.t0 ?? r.temporal_start_year;
  const t1 = r.t1 ?? r.temporal_end_year;
  if (t0 && t1) return `${t0}–${t1}`;
  if (t0) return `${t0}–present`;
  if (t1) return `–${t1}`;
  return '—';
}

function buildSearchPayload() {
  const geoVal = dom.optGeo.value;
  const levelVal = dom.optLevel.value;
  const gridVal = dom.optGrid.value;
  return {
    query: dom.queryInput.value.trim(),
    mode: dom.optMode.value,
    pool: dom.optPool.value || null,
    geo: geoVal === 'global' || geoVal === 'europe' ? geoVal : null,
    exclude_local: geoVal === 'exclude_local',
    max_grid_m: gridVal ? Number(gridVal) : null,
    eo_scope: dom.optStrictEo.checked ? 'strict_eo' : 'all',
    levels: levelVal ? levelVal.split(',') : null,
    null_safe: dom.optNullSafe.checked,
    collapse: dom.optCollapse.checked,
    rerank_pure: dom.optPure.checked,
    top_k: Number(dom.optTopK.value || 20),
  };
}

function syncUrlFromControls() {
  const params = new URLSearchParams();
  const q = dom.queryInput.value.trim();
  if (q) params.set('q', q);
  if (dom.optMode.value && state.status && dom.optMode.value !== state.status.default_mode) {
    params.set('mode', dom.optMode.value);
  }
  if (dom.optPool.value) params.set('pool', dom.optPool.value);
  if (dom.optGeo.value) params.set('geo', dom.optGeo.value);
  if (dom.optGrid.value) params.set('grid', dom.optGrid.value);
  if (dom.optLevel.value) params.set('level', dom.optLevel.value);
  if (dom.optTopK.value !== '20') params.set('k', dom.optTopK.value);
  if (dom.optStrictEo.checked) params.set('strict_eo', '1');
  if (!dom.optNullSafe.checked) params.set('null_safe', '0');
  if (!dom.optCollapse.checked) params.set('collapse', '0');
  if (dom.optPure.checked) params.set('pure', '1');
  const qs = params.toString();
  window.history.replaceState(null, '', qs ? `?${qs}` : window.location.pathname);
}

function applyUrlToControls() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('q')) dom.queryInput.value = params.get('q') || '';
  if (params.has('mode')) {
    const m = params.get('mode');
    if (Array.from(dom.optMode.options).some((o) => o.value === m && !o.disabled)) {
      dom.optMode.value = m;
    }
  }
  if (params.has('pool')) dom.optPool.value = params.get('pool') || '';
  if (params.has('geo')) dom.optGeo.value = params.get('geo') || '';
  if (params.has('grid')) dom.optGrid.value = params.get('grid') || '';
  if (params.has('level')) dom.optLevel.value = params.get('level') || '';
  if (params.has('k')) dom.optTopK.value = params.get('k') || '20';
  if (params.get('strict_eo') === '1') dom.optStrictEo.checked = true;
  if (params.get('null_safe') === '0') dom.optNullSafe.checked = false;
  if (params.get('collapse') === '0') dom.optCollapse.checked = false;
  if (params.get('pure') === '1') dom.optPure.checked = true;
  dom.btnClearQuery.classList.toggle('hidden', !dom.queryInput.value);
}

async function loadStatus() {
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const st = await resp.json();
    state.status = st;

    const devTagMap = { mlx: 'Hybrid · MLX', cuda: 'Hybrid · CUDA', mps: 'Hybrid · MPS', cpu: 'Hybrid · CPU' };
    const modeTag = st.hybrid_available ? (devTagMap[st.device] || `Hybrid · ${String(st.device || '').toUpperCase()}`) : 'BM25 lexical';
    dom.statusBadge.textContent = `${st.active_size.toLocaleString()} active · ${modeTag}`;
    dom.statusBadge.className = st.hybrid_available
      ? 'inline-flex items-center rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-800 ring-1 ring-inset ring-emerald-600/20 tabular-nums'
      : 'inline-flex items-center rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-inset ring-amber-600/20 tabular-nums';

    const allowed = new Set(st.available_modes || ['bm25']);
    Array.from(dom.optMode.options).forEach((opt) => {
      opt.disabled = !allowed.has(opt.value);
    });
    dom.optMode.value = st.default_mode;

    if (st.fallback_reason) {
      dom.modeBanner.textContent = st.fallback_reason;
      dom.modeBanner.classList.remove('hidden');
    }

    applyUrlToControls();
    if (dom.queryInput.value.trim()) {
      await runSearch();
    }
  } catch (err) {
    dom.statusBadge.textContent = 'Offline';
    showError(`Failed to reach local search server: ${err.message}`);
  }
}

async function runSearch() {
  const payload = buildSearchPayload();
  if (!payload.query) {
    dom.queryInput.focus();
    return;
  }
  showError('');
  syncUrlFromControls();

  const needsRerank = ['jina-v3.5', 'jina-v3.5-fast', 'qwen3-0.6b'].includes(payload.mode) && state.status?.hybrid_available;
  const coldRerank = needsRerank && !state.status?.loaded_reranker;
  setSearching(true, coldRerank ? 'Loading reranker…' : 'Searching…');

  try {
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.lastPayload = data;
    state.results = data.results || [];
    if (state.status && data.reranker && data.reranker !== 'none') {
      state.status.loaded_reranker = data.reranker;
    }
    renderResults(data);
  } catch (err) {
    state.lastPayload = null;
    state.results = [];
    if (dom.btnDownload) dom.btnDownload.disabled = true;
    dom.resultsBody.innerHTML = '<tr><td colspan="8" class="px-4 py-12 text-center text-sm text-rose-500">Search request failed. Check the error banner above.</td></tr>';
    showError(`Search failed: ${err.message}`);
  } finally {
    setSearching(false);
  }
}

function renderResults(data) {
  const rows = data.results || [];
  if (dom.btnDownload) dom.btnDownload.disabled = rows.length === 0 || state.exporting;

  const modeName = MODE_LABELS[data.mode_used] || data.mode_used;
  const colNote = data.total_collapsed_in_results > 0
    ? ` · ${data.total_collapsed_in_results} mirrors collapsed`
    : '';
  dom.resultsSummary.textContent = `Showing ${rows.length} of ${data.eligible_after_filter.toLocaleString()} eligible active products (${modeName}${colNote})`;

  const timingParts = [`Stage 1: ${data.stage1_ms.toFixed(0)} ms`];
  if (data.rerank_ms > 0) timingParts.push(`Rerank: ${data.rerank_ms.toFixed(0)} ms`);
  if (data.rerank_load_ms > 0) timingParts.push(`Model warm-up: ${(data.rerank_load_ms / 1000).toFixed(1)} s`);
  dom.resultsTiming.textContent = timingParts.join(' · ');

  if (rows.length === 0) {
    dom.resultsBody.innerHTML = `
      <tr>
        <td colspan="8" class="px-4 py-14 text-center text-sm text-slate-400">
          No matching active products found under the current filters. Try relaxing filters or enabling null-safe mode.
        </td>
      </tr>`;
    return;
  }

  dom.resultsBody.innerHTML = rows.map((r, idx) => {
    const poolMeta = POOL_META[r.provider_pool] || { label: r.provider_pool || '—', cls: 'bg-slate-100 text-slate-600 ring-slate-400/20' };
    const matMeta = MATURITY_META[r.operational_maturity_class];
    const geoLabel = GEO_LABELS[r.geographic_coverage] || (r.geographic_coverage ? r.geographic_coverage : 'null');
    const url = safeHref(r.access_url);
    const doiHref = r.doi ? safeHref(r.doi.startsWith('http') ? r.doi : `https://doi.org/${r.doi}`) : null;

    let rankShift = '';
    if (r.rerank_rank !== null && r.rerank_rank !== undefined && r.stage1_rank) {
      const diff = r.stage1_rank - r.rank;
      if (diff > 0) {
        rankShift = `<div class="text-[10px] font-medium text-emerald-700 tabular-nums" title="Moved up from Stage-1 rank #${r.stage1_rank}">&uarr;${diff} <span class="text-slate-400">S1 #${r.stage1_rank}</span></div>`;
      } else if (diff < 0) {
        rankShift = `<div class="text-[10px] font-medium text-slate-400 tabular-nums" title="Moved down from Stage-1 rank #${r.stage1_rank}">&darr;${Math.abs(diff)} S1 #${r.stage1_rank}</div>`;
      } else {
        rankShift = `<div class="text-[10px] text-slate-400 tabular-nums">S1 #${r.stage1_rank}</div>`;
      }
    }

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
          ${rankShift}
        </td>
        <td class="px-3 py-3 align-top">
          <div class="flex items-start gap-2 flex-wrap">
            <span class="font-semibold text-slate-900 leading-snug">${esc(r.title)}</span>
            ${mirrorsBadge}
          </div>
          <div class="mt-0.5 text-[11px] text-slate-500">
            ${esc(r.producing_institution || r.source_catalogue)}
            <span class="text-slate-300 mx-1">·</span>
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
          <div class="text-[11px] text-slate-500">${esc(r.cadence || '—')}</div>
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
  }).join('');

  dom.resultsBody.querySelectorAll('.result-row').forEach((tr) => {
    tr.addEventListener('click', () => {
      const idx = Number(tr.dataset.idx);
      openDetailForResult(idx);
    });
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
  if (!window.maplibregl) {
    container.innerHTML = `<div class="h-full flex items-center justify-center text-[11px] text-slate-500 px-3 text-center">Bounds: W ${bbox[0]}°, S ${bbox[1]}°, E ${bbox[2]}°, N ${bbox[3]}°.</div>`;
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
    state.mapResizeTimer = setTimeout(() => { if (state.map === map) map.resize(); }, 220);
  } catch (err) {
    container.innerHTML = `<div class="h-full flex items-center justify-center text-[11px] text-slate-500 px-3 text-center">Map unavailable (${esc(err.message)}). Bounds: W ${wRaw}°, S ${sRaw}°, E ${eRaw}°, N ${nRaw}°.</div>`;
  }
}

async function fetchProductDetail(recordId) {
  if (state.productCache.has(recordId)) return state.productCache.get(recordId);
  const resp = await fetch(`/api/product/${encodeURIComponent(recordId)}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  state.productCache.set(recordId, data);
  return data;
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
    const kwRaw = det.keywords || '';
    const kwList = Array.from(new Set(
      String(kwRaw).split(/[;|,]/).map((s) => s.trim()).filter((s) => s.length >= 2 && s.length <= 60)
    ));

    const rankingBlock = rankCtx ? `
      <section class="space-y-2">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Retrieval Scoring Signals</h3>
        <dl class="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-indigo-50/50 rounded-lg p-3 border border-indigo-100">
          ${kvItem('Final Score', rankCtx.score, true)}
          ${kvItem('Stage-1 Rank', rankCtx.stage1_rank ? `#${rankCtx.stage1_rank}` : '—', true)}
          ${kvItem('Dense Cosine / Rank', rankCtx.dense_score !== null && rankCtx.dense_score !== undefined ? `${rankCtx.dense_score} (#${rankCtx.dense_rank})` : '—', true)}
          ${kvItem('BM25 Rank', rankCtx.bm25_rank ? `#${rankCtx.bm25_rank}` : '—', true)}
          ${rankCtx.rerank_score !== null && rankCtx.rerank_score !== undefined ? kvItem('Reranker Score / Rank', `${rankCtx.rerank_score} (#${rankCtx.rerank_rank})`, true) : ''}
        </dl>
      </section>` : '';

    const collapsedBlock = rankCtx && rankCtx.collapsed_items && rankCtx.collapsed_items.length > 0 ? `
      <section class="space-y-2">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Collapsed Mirrors &amp; Editions (${rankCtx.collapsed_siblings})</h3>
        <div class="divide-y divide-slate-100 rounded-lg border border-slate-200/80 bg-slate-50/50">
          ${rankCtx.collapsed_items.map((s) => `
            <button type="button" data-sib-id="${esc(s.record_id)}" class="sib-btn w-full text-left px-3 py-2 hover:bg-indigo-50/70 transition flex items-center justify-between gap-2">
              <div class="min-w-0">
                <div class="text-xs font-medium text-slate-800 truncate">${esc(s.title)}</div>
                <div class="text-[10px] font-mono text-slate-400 truncate">${esc(s.source_catalogue)} · ${esc(s.record_id)}</div>
              </div>
              <span class="text-[11px] text-indigo-600 shrink-0">Inspect &rarr;</span>
            </button>`).join('')}
        </div>
      </section>` : '';

    dom.drawerBody.innerHTML = `
      <div class="flex flex-wrap items-center gap-2">
        ${accessHref ? `<a href="${esc(accessHref)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 transition">Open Access URL &nearr;</a>` : ''}
        ${doiHref ? `<a href="${esc(doiHref)}" target="_blank" rel="noopener noreferrer" class="inline-flex items-center gap-1.5 rounded-lg bg-white px-3 py-1.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 transition">DOI: ${esc(det.doi)}</a>` : ''}
        <button id="btn-copy-id" type="button" class="inline-flex items-center gap-1.5 rounded-lg bg-white px-3 py-1.5 text-xs font-medium text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 transition">Copy Record ID</button>
      </div>

      ${rankingBlock}

      <section class="space-y-1.5">
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Description</h3>
        <p class="text-xs text-slate-700 leading-relaxed whitespace-pre-line bg-slate-50/70 rounded-lg p-3 border border-slate-200/60">${esc(det.description || 'No description provided by upstream catalogue.')}</p>
      </section>

      ${collapsedBlock}

      <section class="space-y-2">
        <div class="flex items-center justify-between">
          <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Spatial Extent &amp; Resolution</h3>
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
        <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Institutional Attribution &amp; Sensors</h3>
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
          ${kvItem('Temporal Window', `${det.temporal_start_raw || 'null'} → ${det.temporal_end_raw || 'null'}`)}
          ${kvItem('License', det.license)}
        </dl>
      </section>

      ${kwList.length > 0 ? `
        <section class="space-y-1.5">
          <h3 class="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Keywords (click to search)</h3>
          <div class="flex flex-wrap gap-1.5">
            ${kwList.slice(0, 35).map((kw) => `<button type="button" data-kw="${esc(kw)}" class="kw-chip rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] text-slate-700 hover:bg-indigo-100 hover:text-indigo-800 transition">${esc(kw)}</button>`).join('')}
          </div>
        </section>` : ''}
    `;

    mountBboxMap(det.bbox, det.antimeridian);

    const copyBtn = document.getElementById('btn-copy-id');
    if (copyBtn) {
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(det.record_id || recordId);
          copyBtn.textContent = 'Copied';
          setTimeout(() => { copyBtn.textContent = 'Copy Record ID'; }, 1200);
        } catch {
          copyBtn.textContent = 'Clipboard blocked';
        }
      });
    }

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
  if (!dom.downloadMenu || !dom.btnDownload) return;
  dom.downloadMenu.classList.toggle('hidden', !open);
  dom.btnDownload.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function handleExportFormat(fmt) {
  setDownloadMenuOpen(false);
  if (!state.results.length || state.exporting) return;
  state.exporting = true;
  if (dom.btnDownload) dom.btnDownload.disabled = true;
  if (dom.btnDownloadLabel) dom.btnDownloadLabel.textContent = 'Exporting…';
  showError('');

  try {
    const metadata = state.lastPayload ? {
      query: state.lastPayload.query,
      device: state.lastPayload.device,
      mode_used: state.lastPayload.mode_used,
      eligible_after_filter: state.lastPayload.eligible_after_filter,
      stage1_ms: state.lastPayload.stage1_ms,
      rerank_ms: state.lastPayload.rerank_ms,
    } : null;

    const resp = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: fmt, results: state.results, metadata }),
    });
    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${resp.status}`);
    }
    const blob = await resp.blob();
    downloadBlob(blob, `eo_hybrid_search_top${state.results.length}.${fmt}`);
  } catch (err) {
    showError(`${fmt.toUpperCase()} export failed: ${err.message}`);
  } finally {
    state.exporting = false;
    if (dom.btnDownload) dom.btnDownload.disabled = state.results.length === 0;
    if (dom.btnDownloadLabel) dom.btnDownloadLabel.textContent = 'Download';
  }
}

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

  dom.exampleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      dom.queryInput.value = btn.dataset.q || '';
      dom.optGeo.value = btn.dataset.geo || '';
      dom.optGrid.value = btn.dataset.grid || '';
      dom.btnClearQuery.classList.remove('hidden');
      runSearch();
    });
  });

  [dom.optMode, dom.optPool, dom.optGeo, dom.optGrid, dom.optLevel, dom.optTopK,
    dom.optStrictEo, dom.optNullSafe, dom.optCollapse, dom.optPure].forEach((el) => {
    el.addEventListener('change', () => {
      if (dom.queryInput.value.trim()) runSearch();
    });
  });

  dom.btnReset.addEventListener('click', () => {
    dom.optPool.value = '';
    dom.optGeo.value = '';
    dom.optGrid.value = '';
    dom.optLevel.value = '';
    dom.optTopK.value = '20';
    dom.optStrictEo.checked = false;
    dom.optNullSafe.checked = true;
    dom.optCollapse.checked = true;
    dom.optPure.checked = false;
    if (state.status) dom.optMode.value = state.status.default_mode;
    if (dom.queryInput.value.trim()) runSearch();
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
      if (dom.downloadMenu && !dom.downloadMenu.classList.contains('hidden')) setDownloadMenuOpen(false);
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
loadStatus();
