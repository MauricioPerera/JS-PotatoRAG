// Proves the "static index consumed by an agent" claim: an index written by the
// WASM store (what the server produces and what publish-index.js ships) can be
// read and searched by the pure-JS PolarQuantizedStore — the zero-dependency
// library an agent fetches from the static site (no WASM, no server).
const { test } = require('node:test');
const assert = require('node:assert');

const { PolarQuantizedStore, MemoryStorageAdapter, normalize } = require('../js-vector-store');
const WasmPolarQuantizedStore = require('../wasm-vector-store.cjs');

const DIM = 768;
function seededVec(n) {
  const a = []; let s = n >>> 0;
  for (let i = 0; i < DIM; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a.push(s / 4294967296 - 0.5); }
  return normalize(a);
}

test('agent (pure-JS store) consumes an index written by the WASM/server store', () => {
  const collection = 'docs_embeddinggemma_768';
  // A shared adapter stands in for "the two static files fetched into memory".
  const adapter = new MemoryStorageAdapter();

  // Publisher side: build + persist with the WASM store (same code path as the server).
  const publisher = new WasmPolarQuantizedStore(adapter, DIM, { bits: 3, model: 'embeddinggemma' });
  const vecs = [];
  for (let i = 0; i < 25; i++) { const v = seededVec(i + 1); vecs.push(v); publisher.set(collection, 'd' + i, v, { text: 'chunk ' + i }); }
  publisher.flush();

  // Agent side: only the pure-JS library, reading the same .p3.json/.p3.bin.
  const agentStore = new PolarQuantizedStore(adapter, DIM, { bits: 3, silent: true });

  // Querying with a document's own embedding returns that document + its text.
  for (const i of [0, 7, 24]) {
    const hit = agentStore.search(collection, vecs[i], 1)[0];
    assert.strictEqual(hit.id, 'd' + i, `expected d${i}, got ${hit.id}`);
    assert.strictEqual(hit.metadata.text, 'chunk ' + i);
  }

  // Top-k shape is what the SKILL recipe documents.
  const top3 = agentStore.search(collection, vecs[3], 3);
  assert.strictEqual(top3.length, 3);
  assert.ok(top3.every(r => typeof r.metadata.text === 'string'));
});

test('agent store refuses an index with mismatched dimension (load guard parity)', () => {
  const adapter = new MemoryStorageAdapter();
  const publisher = new WasmPolarQuantizedStore(adapter, DIM, { bits: 3 });
  publisher.set('docs_x_768', 'a', seededVec(1), { text: 'x' });
  publisher.flush();

  // Pure-JS store opened at the wrong dimension must not silently return garbage.
  const wrong = new PolarQuantizedStore(adapter, 384, { bits: 3, silent: true });
  assert.throws(() => wrong.search('docs_x_768', seededVec(1), 1));
});
