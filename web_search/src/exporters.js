import { parquetWriteBuffer } from 'hyparquet-writer';

export const EXPORT_COLUMNS = [
  ['rank', 'INT32'],
  ['score', 'DOUBLE'],
  ['stage1_rank', 'INT32'],
  ['dense_rank', 'INT32'],
  ['dense_score', 'DOUBLE'],
  ['bm25_rank', 'INT32'],
  ['bm25_score', 'DOUBLE'],
  ['rerank_rank', 'INT32'],
  ['rerank_score', 'DOUBLE'],
  ['record_id', 'STRING'],
  ['title', 'STRING'],
  ['source_catalogue', 'STRING'],
  ['producing_institution', 'STRING'],
  ['provider_pool', 'STRING'],
  ['copernicus_service', 'STRING'],
  ['processing_level', 'STRING'],
  ['uses_sentinel', 'BOOLEAN'],
  ['sentinel_missions', 'STRING'],
  ['grid', 'STRING'],
  ['grid_m', 'DOUBLE'],
  ['cadence', 'STRING'],
  ['geographic_coverage', 'STRING'],
  ['t0', 'INT32'],
  ['t1', 'INT32'],
  ['operational_maturity_class', 'STRING'],
  ['collapsed_siblings', 'INT32'],
  ['doi', 'STRING'],
  ['access_url', 'STRING'],
];

function csvCell(val) {
  if (val === null || val === undefined) return '';
  let s = String(val);
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
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

export function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function exportResults(fmt, results, metadata = null) {
  const cols = EXPORT_COLUMNS.map(([name]) => name);
  const prefix = `eo_advanced_search_top${results.length}`;

  if (fmt === 'csv') {
    const lines = [cols.join(',')];
    for (const r of results) {
      lines.push(cols.map((c) => csvCell(r[c])).join(','));
    }
    triggerBlobDownload(new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8;' }), `${prefix}.csv`);
    return;
  }

  if (fmt === 'json') {
    const payload = {
      metadata,
      results: results.map((r) => {
        const obj = {};
        for (const c of cols) obj[c] = r[c] === undefined ? null : r[c];
        return obj;
      }),
    };
    triggerBlobDownload(
      new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8;' }),
      `${prefix}.json`,
    );
    return;
  }

  if (fmt === 'xlsx') {
    const letters = EXPORT_COLUMNS.map((_, i) => colRef(i));
    const rowXml = [
      `<row r="1">${EXPORT_COLUMNS.map(([name], ci) => `<c r="${letters[ci]}1" t="inlineStr"><is><t>${xmlEsc(name)}</t></is></c>`).join('')}</row>`,
    ];
    for (let ri = 0; ri < results.length; ri++) {
      const r = results[ri];
      const rn = ri + 2;
      const cells = [];
      for (let ci = 0; ci < EXPORT_COLUMNS.length; ci++) {
        const [name, type] = EXPORT_COLUMNS[ci];
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
      '<sheets><sheet name="AdvancedSearch" sheetId="1" r:id="rId1"/></sheets>' +
      '</workbook>';
    const wbRels =
      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
      '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>' +
      '</Relationships>';

    const blob = buildZipStoreBlob(
      [
        { name: '[Content_Types].xml', content: contentTypes },
        { name: '_rels/.rels', content: rootRels },
        { name: 'xl/workbook.xml', content: workbookXml },
        { name: 'xl/_rels/workbook.xml.rels', content: wbRels },
        { name: 'xl/worksheets/sheet1.xml', content: sheetXml },
      ],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    );
    triggerBlobDownload(blob, `${prefix}.xlsx`);
    return;
  }

  if (fmt === 'parquet') {
    const columnData = EXPORT_COLUMNS.map(([name, type]) => ({
      name,
      type,
      nullable: true,
      data: results.map((r) => {
        const v = r[name];
        if (v === null || v === undefined) return null;
        if (type === 'BOOLEAN') return Boolean(v);
        if (type === 'INT32' || type === 'DOUBLE') return Number.isFinite(Number(v)) ? Number(v) : null;
        return String(v);
      }),
    }));
    const buffer = parquetWriteBuffer({
      columnData,
      createdBy: 'global-EO-products-L3-L4-advanced-search',
    });
    triggerBlobDownload(new Blob([buffer], { type: 'application/vnd.apache.parquet' }), `${prefix}.parquet`);
  }
}
