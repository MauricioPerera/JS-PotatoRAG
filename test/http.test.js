// HTTP smoke tests for routing/validation paths that don't require a model.
// Run with: node --test
const { test, before, after } = require('node:test');
const assert = require('node:assert');
const { spawn } = require('node:child_process');
const path = require('node:path');

let proc;
const PORT = 3099;
const BASE = `http://127.0.0.1:${PORT}`;

async function waitForServer(timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(`${BASE}/api/stats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: { embedSource: 'ollama', embedDimension: 768 } }),
      });
      if (r.ok) return;
    } catch { /* not up yet */ }
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error('server did not start in time');
}

before(async () => {
  proc = spawn(process.execPath, ['server.js'], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, PORT: String(PORT), HOST: '127.0.0.1' },
    stdio: 'ignore',
  });
  await waitForServer();
});

after(() => { if (proc) proc.kill(); });

test('GET / serves the PWA shell', async () => {
  const r = await fetch(`${BASE}/`);
  assert.strictEqual(r.status, 200);
  const body = await r.text();
  assert.match(body, /JS-PotatoRAG/);
});

test('POST /api/stats returns collection info without loading a model', async () => {
  const r = await fetch(`${BASE}/api/stats`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings: { embedSource: 'ollama', embedModel: 'x', embedDimension: 768 } }),
  });
  assert.strictEqual(r.status, 200);
  const j = await r.json();
  assert.strictEqual(j.dimension, 768);
  assert.strictEqual(j.vectorCount, 0);
});

test('POST /api/ingest rejects empty text with 400', async () => {
  const r = await fetch(`${BASE}/api/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: '   ' }),
  });
  assert.strictEqual(r.status, 400);
});

test('POST /api/query on empty collection returns no results (no embedding needed)', async () => {
  const r = await fetch(`${BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'hello', settings: { embedSource: 'ollama', embedModel: 'unused', embedDimension: 768 } }),
  });
  assert.strictEqual(r.status, 200);
  const j = await r.json();
  assert.deepStrictEqual(j.results, []);
});

test('POST /api/import then /api/export round-trips a collection (no model needed)', async () => {
  // Build a valid export in-memory using the store, then push it through HTTP.
  const WasmPolarQuantizedStore = require('../wasm-vector-store.cjs');
  const { MemoryStorageAdapter, normalize } = require('../js-vector-store');
  const dim = 768;
  const vec = (n) => {
    const a = []; let s = n >>> 0;
    for (let i = 0; i < dim; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a.push(s / 4294967296 - 0.5); }
    return normalize(a);
  };
  // Unique model name so this test never clobbers a real collection on disk.
  const model = 'httptest';
  const collection = 'docs_httptest_768';
  const tmp = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), dim, { bits: 3, model });
  for (let i = 0; i < 8; i++) tmp.set(collection, 'd' + i, vec(i + 1), { text: 'chunk ' + i });
  tmp.flush();
  const exportObj = tmp.exportCollection(collection);

  const imp = await fetch(`${BASE}/api/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: exportObj, mode: 'replace' }),
  });
  const impJson = await imp.json();
  assert.strictEqual(imp.status, 200, JSON.stringify(impJson));
  assert.strictEqual(impJson.total, 8);

  const settings = { embedSource: 'ollama', embedModel: model, embedDimension: 768 };
  const stats = await (await fetch(`${BASE}/api/stats`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ settings }),
  })).json();
  assert.strictEqual(stats.vectorCount, 8);

  const exp = await fetch(`${BASE}/api/export`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ settings }),
  });
  assert.strictEqual(exp.status, 200);
  const expJson = await exp.json();
  assert.strictEqual(expJson.count, 8);
  assert.strictEqual(expJson.bin, exportObj.bin, 'exported packed bytes must match what was imported');
});

test('CORS: a disallowed origin gets no Access-Control-Allow-Origin header', async () => {
  const r = await fetch(`${BASE}/api/stats`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Origin': 'http://evil.example' },
    body: JSON.stringify({ settings: {} }),
  });
  assert.strictEqual(r.headers.get('access-control-allow-origin'), null);
});
