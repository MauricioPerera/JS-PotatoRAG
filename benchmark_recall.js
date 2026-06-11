// Recall benchmark: polar 3-bit (and other bit-widths) vs exact float32.
//
// Quality, not speed. Answers the only question the speed/memory benchmarks
// leave open: how much retrieval accuracy does the compression cost?
//
// Embeddings: uses Ollama (embeddinggemma) if reachable, otherwise falls back
// to CLUSTERED synthetic vectors (uniform-random vectors are an unrealistic
// worst case for cosine retrieval — real embeddings cluster, so we simulate
// that with gaussian blobs around random centroids).

const { PolarQuantizedStore, MemoryStorageAdapter, normalize, cosineSim } = require('./js-vector-store');

const DIM = 768;
const N = 1000;        // corpus size
const QUERIES = 100;   // number of probe queries
const KS = [1, 3, 10]; // recall@k to report
const BITS = [3, 4, 5];

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller gaussian from a uniform PRNG.
function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Clustered synthetic corpus: NCLUSTERS centroids, points = centroid + noise.
function makeClusteredVectors(rand, count, dim, nClusters = 20, noise = 0.35) {
  const centroids = [];
  for (let c = 0; c < nClusters; c++) {
    const v = new Array(dim);
    for (let d = 0; d < dim; d++) v[d] = gauss(rand);
    centroids.push(normalize(v));
  }
  const vecs = [];
  for (let i = 0; i < count; i++) {
    const c = centroids[i % nClusters];
    const v = new Array(dim);
    for (let d = 0; d < dim; d++) v[d] = c[d] + noise * gauss(rand);
    vecs.push(normalize(v));
  }
  return vecs;
}

async function tryOllamaEmbeddings() {
  let ollama;
  try {
    ollama = require('ollama').default || require('ollama');
  } catch {
    return null;
  }
  // Build a corpus of GENUINELY DISTINCT sentences so that embeddings are
  // well-separated and the exact top-1 is unambiguous. Each doc combines a
  // random subject + verb + object + modifier from disjoint vocabularies, so
  // no two docs are near-duplicates (which would make recall@1 a coin-flip).
  const subjects = ['the database', 'a neural network', 'the compiler', 'this algorithm', 'the spacecraft',
    'a glacier', 'the orchestra', 'her thesis', 'the volcano', 'a coral reef', 'the parliament', 'this enzyme',
    'the telescope', 'a hurricane', 'the marketplace', 'his sculpture', 'the migration', 'a vaccine',
    'the bridge', 'this protocol'];
  const verbs = ['accelerates', 'dissolves', 'illuminates', 'fractures', 'amplifies', 'condenses',
    'navigates', 'rejects', 'absorbs', 'reconstructs', 'destabilizes', 'synchronizes', 'encrypts',
    'germinates', 'oscillates', 'catalogs', 'erodes', 'replicates', 'suspends', 'negotiates'];
  const objects = ['the gradient', 'a continent', 'the harvest', 'three proteins', 'the signal',
    'an archive', 'the sediment', 'her argument', 'the lattice', 'a melody', 'the budget', 'the canopy',
    'this manuscript', 'the current', 'a frequency', 'the antibody', 'the coastline', 'his ledger',
    'the membrane', 'a treaty'];
  const modifiers = ['under extreme pressure', 'over several decades', 'during the monsoon season',
    'across the northern hemisphere', 'within a closed loop', 'beneath the ocean floor',
    'after the second iteration', 'before sunrise', 'against all predictions', 'through quantum tunneling',
    'in complete silence', 'without external power', 'around the central axis', 'despite the turbulence',
    'between two checkpoints', 'beyond the visible spectrum', 'along the fault line', 'amid the negotiations',
    'throughout the experiment', 'near the equilibrium point'];
  const pick = (arr, n) => arr[n % arr.length];
  const docs = [];
  for (let i = 0; i < N; i++) {
    docs.push(`${pick(subjects, i)} ${pick(verbs, (i * 7) + 1)} ${pick(objects, (i * 13) + 2)} ${pick(modifiers, (i * 17) + 3)}`);
  }
  try {
    const vecs = [];
    for (const d of docs) {
      const r = await ollama.embeddings({ model: 'embeddinggemma', prompt: d });
      vecs.push(normalize(r.embedding));
    }
    return { vecs, docs, dim: vecs[0].length, source: 'ollama:embeddinggemma' };
  } catch (e) {
    console.log(`   (Ollama not usable: ${e.message} — falling back to synthetic)`);
    return null;
  }
}

function recallAtK(store, vecs, queries, k) {
  let hit = 0, total = 0;
  for (const qv of queries) {
    const exact = vecs
      .map((v, i) => ({ i, s: cosineSim(qv, v) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map(x => x.i);
    const approx = new Set(store.search('docs', qv, k).map(r => parseInt(r.id, 10)));
    for (const i of exact) { total++; if (approx.has(i)) hit++; }
  }
  return hit / total;
}

async function main() {
  console.log('============================================================');
  console.log('  RECALL BENCHMARK — polar quantization vs exact float32');
  console.log('============================================================');

  const rand = mulberry32(12345);
  let vecs, dim, source;

  const ollamaResult = await tryOllamaEmbeddings();
  if (ollamaResult) {
    ({ vecs, dim, source } = ollamaResult);
  } else {
    vecs = makeClusteredVectors(rand, N, DIM);
    dim = DIM;
    source = 'synthetic:clustered';
  }

  // Queries: near-duplicates of random corpus members (realistic RAG queries
  // are close to a stored chunk, not arbitrary points).
  const queries = [];
  for (let q = 0; q < QUERIES; q++) {
    const base = vecs[Math.floor(rand() * vecs.length)];
    const v = base.map(x => x + 0.15 * gauss(rand));
    queries.push(normalize(v));
  }

  console.log(`Corpus: ${vecs.length} vectors, dim=${dim}, source=${source}`);
  console.log(`Queries: ${queries.length} (near-duplicate probes)\n`);

  const rawBytes = vecs.length * dim * 4;
  const header = ['bits', ...KS.map(k => `recall@${k}`), 'bytes/vec', 'compression'];
  console.log(header.join('\t'));

  const rows = [];
  for (const bits of BITS) {
    const store = new PolarQuantizedStore(new MemoryStorageAdapter(), dim, { bits, silent: true });
    for (let i = 0; i < vecs.length; i++) store.set('docs', String(i), vecs[i]);
    store.flush();
    const bpv = store.bytesPerVector();
    const recalls = KS.map(k => recallAtK(store, vecs, queries, k));
    const compression = (rawBytes / (vecs.length * bpv)).toFixed(1) + 'x';
    rows.push({ bits, recalls, bpv, compression });
    console.log([bits, ...recalls.map(r => (r * 100).toFixed(1) + '%'), bpv, compression].join('\t'));
  }

  const md = `# Recall Benchmark (polar quantization vs exact float32)

* **Date**: generated by benchmark_recall.js
* **Corpus**: ${vecs.length} vectors, dim=${dim}
* **Embedding source**: ${source}
* **Queries**: ${queries.length} near-duplicate probes

Recall@k = fraction of the exact float32 top-k that the quantized index also returns.

| Bits | ${KS.map(k => `Recall@${k}`).join(' | ')} | Bytes/vec | Compression |
| :--- | ${KS.map(() => ':---').join(' | ')} | :--- | :--- |
${rows.map(r => `| ${r.bits} | ${r.recalls.map(x => (x * 100).toFixed(1) + '%').join(' | ')} | ${r.bpv} | ${r.compression} |`).join('\n')}

> Note: with \`source=synthetic:clustered\` these numbers approximate real-embedding
> behavior. Run with Ollama + \`embeddinggemma\` available for ground-truth recall.
`;
  require('fs').writeFileSync('benchmark_recall_results.md', md, 'utf8');
  console.log('\nResults written to benchmark_recall_results.md');
}

main().catch(console.error);
