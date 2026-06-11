// Run with: node --test
const { test } = require('node:test');
const assert = require('node:assert');

const {
  PolarQuantizedStore, MemoryStorageAdapter, normalize, cosineSim,
} = require('../js-vector-store');
const WasmPolarStore = require('../wasm-polar-store.cjs');
const WasmPolarQuantizedStore = require('../wasm-vector-store.cjs');

const DIM = 768;

function seededVec(n, dim = DIM) {
  const a = [];
  let s = n >>> 0;
  for (let i = 0; i < dim; i++) {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    a.push(s / 4294967296 - 0.5);
  }
  return normalize(a);
}

test('JS and WASM quantizers produce byte-identical output (parity)', () => {
  const js = new PolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3, seed: 42, silent: true });
  const wasm = new WasmPolarStore(DIM, 3, 42);
  for (let t = 0; t < 100; t++) {
    const v = seededVec(t + 1);
    const a = js._quantize(v);
    const b = wasm.quantize(v);
    assert.deepStrictEqual(Array.from(a), Array.from(b), `mismatch at vector ${t}`);
  }
});

test('cross-engine: a collection written by JS store reads correctly in WASM store', () => {
  const N = 300;
  const vecs = Array.from({ length: N }, (_, i) => seededVec(i + 1000));
  const adapter = new MemoryStorageAdapter();

  const js = new PolarQuantizedStore(adapter, DIM, { bits: 3, silent: true });
  for (let i = 0; i < N; i++) js.set('docs', 'id' + i, vecs[i]);
  js.flush();

  const wasm = new WasmPolarQuantizedStore(adapter, DIM, { bits: 3 });
  // The top-1 of a near-duplicate query must be the right vector.
  let top1 = 0;
  for (let q = 0; q < 30; q++) {
    const base = vecs[q];
    const query = normalize(base.map(x => x + 0.05 * (seededVec(q + 9000)[0])));
    const r = wasm.search('docs', query, 1);
    if (r[0] && r[0].id === 'id' + q) top1++;
  }
  assert.ok(top1 >= 27, `cross-engine top-1 too low: ${top1}/30`);
});

test('round-trip: set → flush → reload preserves count and metadata', () => {
  const adapter = new MemoryStorageAdapter();
  const s1 = new WasmPolarQuantizedStore(adapter, DIM, { bits: 3 });
  s1.set('c', 'a', seededVec(1), { text: 'alpha' });
  s1.set('c', 'b', seededVec(2), { text: 'beta' });
  s1.flush();

  const s2 = new WasmPolarQuantizedStore(adapter, DIM, { bits: 3 });
  assert.strictEqual(s2.count('c'), 2);
  const hit = s2.search('c', seededVec(1), 1)[0];
  assert.strictEqual(hit.metadata.text, 'alpha');
});

test('load guard: opening a collection with mismatched dim throws', () => {
  const adapter = new MemoryStorageAdapter();
  const s1 = new WasmPolarQuantizedStore(adapter, 768, { bits: 3 });
  s1.set('c', 'a', seededVec(1));
  s1.flush();

  const s2 = new WasmPolarQuantizedStore(adapter, 384, { bits: 3 });
  assert.throws(() => s2.count('c'), /quantized with dim=768/);
});

test('search() does not write to disk when there are pending vectors', () => {
  // A spy adapter that records writes. Search must not trigger any.
  const writes = [];
  const data = { bins: new Map(), jsons: new Map() };
  const adapter = {
    readBin: (k) => data.bins.get(k) ?? null,
    readJson: (k) => data.jsons.get(k) ?? null,
    writeBin: (k, v) => { writes.push(k); data.bins.set(k, v); },
    writeJson: (k, v) => { writes.push(k); data.jsons.set(k, v); },
    delete: (k) => { data.bins.delete(k); data.jsons.delete(k); },
  };
  const store = new WasmPolarQuantizedStore(adapter, DIM, { bits: 3 });
  store.set('c', 'a', seededVec(1), { text: 'alpha' });
  store.set('c', 'b', seededVec(2), { text: 'beta' });
  // No flush() — vectors are still pending.
  writes.length = 0;
  const r = store.search('c', seededVec(1), 1);
  assert.strictEqual(r[0].id, 'a', 'pending vector should be searchable');
  assert.deepStrictEqual(writes, [], `search wrote to disk: ${writes.join(', ')}`);

  // A real flush still persists.
  store.flush();
  assert.ok(writes.length > 0, 'flush should persist');
});

test('export → import round-trips a collection (replace mode)', () => {
  const src = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3, model: 'embeddinggemma' });
  for (let i = 0; i < 20; i++) src.set('docs', 'd' + i, seededVec(i + 1), { text: 'chunk ' + i });
  src.flush();
  const exported = src.exportCollection('docs');

  // Round-trip through JSON (as it would travel over the wire / a file).
  const wire = JSON.parse(JSON.stringify(exported));

  const dst = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  const r = dst.importCollection(wire, { mode: 'replace' });
  assert.strictEqual(r.total, 20);
  assert.strictEqual(dst.count('docs'), 20);

  // Same query returns the same top hit + preserved chunk text and model.
  const q = seededVec(5);
  assert.strictEqual(src.search('docs', q, 1)[0].id, dst.search('docs', q, 1)[0].id);
  assert.strictEqual(dst.search('docs', seededVec(5), 1)[0].metadata.text, src.search('docs', seededVec(5), 1)[0].metadata.text);
});

test('import rejects a collection name that could escape the data dir', () => {
  const store = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  const malicious = {
    format: 'potatorag-export', version: 1,
    collection: '../../evil', dim: DIM, bits: 3, seed: 42, model: null,
    count: 0, ids: [], meta: [], bin: '',
  };
  assert.throws(() => store.importCollection(malicious), /Invalid collection name/);
});

test('import rejects an export with incompatible quantizer params', () => {
  const src = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), 768, { bits: 3 });
  src.set('docs', 'a', seededVec(1));
  src.flush();
  const exported = src.exportCollection('docs');

  const dst = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), 384, { bits: 3 });
  assert.throws(() => dst.importCollection(exported), /Incompatible quantizer params/);
});

test('import merge mode appends and avoids id collisions', () => {
  const a = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  for (let i = 0; i < 5; i++) a.set('docs', 'x' + i, seededVec(i + 1), { text: 't' + i });
  a.flush();
  const exp = a.exportCollection('docs');

  const dst = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  dst.set('docs', 'x0', seededVec(99), { text: 'pre-existing' }); // collides with imported 'x0'
  dst.flush();
  const r = dst.importCollection(exp, { mode: 'merge' });
  assert.strictEqual(r.total, 6); // 1 existing + 5 imported
  assert.ok(dst.count('docs') === 6);
});

test('remove() deletes a record and keeps the rest searchable', () => {
  const store = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  for (let i = 0; i < 10; i++) store.set('c', 'm' + i, seededVec(i + 1), { text: 'mem ' + i });
  store.flush();
  assert.strictEqual(store.remove('c', 'm3'), true);
  assert.strictEqual(store.remove('c', 'nope'), false);
  assert.strictEqual(store.count('c'), 9);
  // surviving records still resolve and m3 is gone
  const ids = new Set(store.list('c').map(r => r.id));
  assert.ok(!ids.has('m3'));
  assert.ok(ids.has('m9'));
  const hit = store.search('c', seededVec(5), 1)[0];
  assert.strictEqual(hit.id, 'm4'); // seededVec(5) was stored under m4 (i+1)
});

test('search() honors a metadata filter', () => {
  const store = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  for (let i = 0; i < 20; i++) {
    store.set('c', 'm' + i, seededVec(i + 1), { text: 'mem ' + i, kind: i % 2 === 0 ? 'even' : 'odd' });
  }
  store.flush();
  const res = store.search('c', seededVec(3), 5, { kind: 'odd' });
  assert.ok(res.length > 0);
  assert.ok(res.every(r => r.metadata.kind === 'odd'), 'all results must match the filter');
});

test('list() returns records and supports filtering', () => {
  const store = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  store.set('c', 'a', seededVec(1), { text: 'x', tag: 'keep' });
  store.set('c', 'b', seededVec(2), { text: 'y', tag: 'drop' });
  store.flush();
  assert.strictEqual(store.list('c').length, 2);
  const kept = store.list('c', { tag: 'keep' });
  assert.strictEqual(kept.length, 1);
  assert.strictEqual(kept[0].id, 'a');
});

test('recall@1 floor on clustered data (sanity, not exhaustive)', () => {
  const N = 400, nClusters = 10;
  const centroids = Array.from({ length: nClusters }, (_, c) => seededVec(c + 1));
  const vecs = [];
  for (let i = 0; i < N; i++) {
    const c = centroids[i % nClusters];
    vecs.push(normalize(c.map((x, d) => x + 0.3 * (seededVec(i * 7 + d)[0]))));
  }
  const store = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  for (let i = 0; i < N; i++) store.set('c', String(i), vecs[i]);
  store.flush();

  let hit = 0;
  for (let q = 0; q < 40; q++) {
    const base = vecs[q];
    const query = normalize(base.map((x, d) => x + 0.1 * (seededVec(q * 13 + d)[0])));
    const exactTop = vecs
      .map((v, i) => ({ i, s: cosineSim(query, v) }))
      .sort((a, b) => b.s - a.s)[0].i;
    const approxTop = parseInt(store.search('c', query, 1)[0].id, 10);
    if (approxTop === exactTop) hit++;
  }
  assert.ok(hit >= 32, `recall@1 too low: ${hit}/40`);
});
