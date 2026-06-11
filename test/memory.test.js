// Agent-memory API tests. These need an embedding backend, so they run against
// Ollama (embeddinggemma) and SKIP automatically if it isn't reachable — CI
// without Ollama stays green.
const { test, before, after } = require('node:test');
const assert = require('node:assert');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const PORT = 3098;
const BASE = `http://127.0.0.1:${PORT}`;
const NS = 'pottest';
const SETTINGS = { embedSource: 'ollama', embedModel: 'embeddinggemma', embedDimension: 768 };
const COLLECTION_FILES = [
  `mem_${NS}_embeddinggemma_768.p3.bin`,
  `mem_${NS}_embeddinggemma_768.p3.json`,
];

let proc;
let ollamaUp = false;

async function post(pathname, body) {
  const r = await fetch(`${BASE}${pathname}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: r.status, json: await r.json().catch(() => ({})) };
}

before(async () => {
  try {
    const r = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(2000) });
    ollamaUp = r.ok;
  } catch { ollamaUp = false; }
  if (!ollamaUp) return;

  proc = spawn(process.execPath, ['server.js'], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, PORT: String(PORT), HOST: '127.0.0.1' },
    stdio: 'ignore',
  });
  const start = Date.now();
  while (Date.now() - start < 15000) {
    try { if ((await fetch(`${BASE}/`)).ok) break; } catch {}
    await new Promise(r => setTimeout(r, 250));
  }
  // start from a clean collection
  await post('/api/memory/forget', { namespace: NS, filter: {}, settings: SETTINGS }).catch(() => {});
});

after(() => {
  if (proc) proc.kill();
  for (const f of COLLECTION_FILES) {
    const p = path.join(__dirname, '..', 'data', 'vectors', f);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  }
});

test('write → search recalls the right memory', async (t) => {
  if (!ollamaUp) return t.skip('Ollama not reachable');

  await post('/api/memory/write', { text: 'The deploy key for staging is rotated every Monday.', tags: ['ops', 'staging'], namespace: NS, settings: SETTINGS });
  await post('/api/memory/write', { text: 'The user prefers TypeScript over JavaScript for new services.', tags: ['prefs'], namespace: NS, settings: SETTINGS });
  await post('/api/memory/write', { text: 'Production database backups run nightly at 02:00 UTC.', tags: ['ops', 'prod'], namespace: NS, settings: SETTINGS });

  const { json } = await post('/api/memory/search', { query: 'when are staging credentials rotated?', k: 1, namespace: NS, settings: SETTINGS });
  assert.ok(json.results.length >= 1);
  assert.match(json.results[0].text, /deploy key for staging/i);
});

test('search filters by tag', async (t) => {
  if (!ollamaUp) return t.skip('Ollama not reachable');
  const { json } = await post('/api/memory/search', { query: 'operations', k: 5, namespace: NS, tags: ['prod'], settings: SETTINGS });
  assert.ok(json.results.length >= 1);
  assert.ok(json.results.every(r => r.tags.includes('prod')), 'every result must carry the prod tag');
});

test('list returns all memories in the namespace', async (t) => {
  if (!ollamaUp) return t.skip('Ollama not reachable');
  const { json } = await post('/api/memory/list', { namespace: NS, settings: SETTINGS });
  assert.strictEqual(json.count, 3);
});

test('forget by id removes one memory', async (t) => {
  if (!ollamaUp) return t.skip('Ollama not reachable');
  const { json: list } = await post('/api/memory/list', { namespace: NS, settings: SETTINGS });
  const target = list.items.find(i => i.tags.includes('prefs'));
  const { json } = await post('/api/memory/forget', { id: target.id, namespace: NS, settings: SETTINGS });
  assert.strictEqual(json.removed, 1);
  assert.strictEqual(json.total, 2);
});

test('forget by tag removes matching memories', async (t) => {
  if (!ollamaUp) return t.skip('Ollama not reachable');
  const { json } = await post('/api/memory/forget', { tags: ['ops'], namespace: NS, settings: SETTINGS });
  assert.ok(json.removed >= 1);
  const { json: list } = await post('/api/memory/list', { namespace: NS, settings: SETTINGS });
  assert.ok(list.items.every(i => !i.tags.includes('ops')));
});
