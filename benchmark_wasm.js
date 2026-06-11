const { PolarQuantizedStore, MemoryStorageAdapter, normalize } = require('./js-vector-store');
const WasmPolarQuantizedStore = require('./wasm-vector-store');

function hrMs(start) {
  const [s, ns] = process.hrtime(start);
  return s * 1000 + ns / 1e6;
}

function fakeVec(dim) {
  return normalize(Array.from({ length: dim }, () => Math.random() * 2 - 1));
}

function bar(value, max, width = 30) {
  const filled = Math.round((value / max) * width);
  return '#'.repeat(Math.max(0, filled)) + '-'.repeat(Math.max(0, width - filled));
}

async function runBenchmark() {
  const DIM = 768;
  const N = 10000;
  const ITERS = 100;
  
  console.log("============================================================");
  console.log(`🚀 WASM VS PURE JS BENCHMARK (N = ${N} vectors, dim = ${DIM})`);
  console.log("============================================================");

  // Generate synthetic vectors
  console.log("Generating synthetic vectors...");
  const vectors = Array.from({ length: N }, () => fakeVec(DIM));
  const queries = Array.from({ length: ITERS }, () => fakeVec(DIM));
  console.log("Vectors generated.");

  // ─── 1. Insertion Speed ─────────────────────────────────
  console.log("\n1. INSERTION BENCHMARK (Time to insert & quantize)");
  
  // Pure JS Ingestion
  const jsStore = new PolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3, silent: true });
  let t0 = process.hrtime();
  for (let i = 0; i < N; i++) {
    jsStore.set('bulk', `v-${i}`, vectors[i]);
  }
  jsStore.flush();
  const jsInsertTime = hrMs(t0);
  console.log(`   - Pure JS Store:   ${jsInsertTime.toFixed(2)} ms (${(jsInsertTime / N).toFixed(4)} ms/vec)`);

  // WASM Rust Ingestion
  const wasmStore = new WasmPolarQuantizedStore(new MemoryStorageAdapter(), DIM, { bits: 3 });
  t0 = process.hrtime();
  for (let i = 0; i < N; i++) {
    wasmStore.set('bulk', `v-${i}`, vectors[i]);
  }
  wasmStore.flush();
  const wasmInsertTime = hrMs(t0);
  console.log(`   - WASM Rust Store: ${wasmInsertTime.toFixed(2)} ms (${(wasmInsertTime / N).toFixed(4)} ms/vec)`);
  console.log(`   * WASM Ingestion Speedup: ${(jsInsertTime / wasmInsertTime).toFixed(1)}x`);

  // ─── 2. Search Speed ────────────────────────────────────
  console.log("\n2. SEARCH BENCHMARK (Avg time for top-10 cosine lookup)");

  // Pure JS search
  const jsSearchTimes = [];
  for (let i = 0; i < ITERS; i++) {
    const t = process.hrtime();
    jsStore.search('bulk', queries[i], 10);
    jsSearchTimes.push(hrMs(t));
  }
  const jsSearchAvg = jsSearchTimes.reduce((a, b) => a + b, 0) / ITERS;
  console.log(`   - Pure JS Store:   ${jsSearchAvg.toFixed(4)} ms/query`);

  // WASM search
  const wasmSearchTimes = [];
  for (let i = 0; i < ITERS; i++) {
    const t = process.hrtime();
    wasmStore.search('bulk', queries[i], 10);
    wasmSearchTimes.push(hrMs(t));
  }
  const wasmSearchAvg = wasmSearchTimes.reduce((a, b) => a + b, 0) / ITERS;
  console.log(`   - WASM Rust Store: ${wasmSearchAvg.toFixed(4)} ms/query`);
  console.log(`   * WASM Search Speedup: ${(jsSearchAvg / wasmSearchAvg).toFixed(1)}x`);

  console.log("\n============================================================");
  console.log("                      SUMMARY RESULTS                       ");
  console.log("============================================================");
  console.log(`  Insertion Speedup:   ${(jsInsertTime / wasmInsertTime).toFixed(1)}x`);
  console.log(`  Search Scan Speedup: ${(jsSearchAvg / wasmSearchAvg).toFixed(1)}x`);
  console.log("============================================================");

  // Write markdown comparison
  const report = `# Benchmark Results: WASM PolarStore vs. Pure JS PolarStore

Comparing insertion (quantization) and search speeds over **10,000 synthetic vectors** of **768 dimensions** on local hardware.

## Results Summary

| Metric | Pure JS Store | WASM Rust Store | Speedup |
| :--- | :--- | :--- | :--- |
| **Ingestion Time (10k vectors)** | ${jsInsertTime.toFixed(2)} ms | ${wasmInsertTime.toFixed(2)} ms | **${(jsInsertTime / wasmInsertTime).toFixed(1)}x** |
| **Avg Ingestion per Vector** | ${(jsInsertTime / N).toFixed(4)} ms | ${(wasmInsertTime / N).toFixed(4)} ms | **${(jsInsertTime / wasmInsertTime).toFixed(1)}x** |
| **Avg Search Latency (top-10)** | ${jsSearchAvg.toFixed(4)} ms | ${wasmSearchAvg.toFixed(4)} ms | **${(jsSearchAvg / wasmSearchAvg).toFixed(1)}x** |

## Key Takeaways
1. **Quantization speedup**: Rust compiles to native target code, enabling math-heavy coordinate-to-polar calculations during quantization to execute **${(jsInsertTime / wasmInsertTime).toFixed(1)}x faster** than interpreted JS loops.
2. **Search scan speedup**: The index scan in WebAssembly runs at near-native speed, achieving a **${(jsSearchAvg / wasmSearchAvg).toFixed(1)}x speedup** over JavaScript.
`;
  
  fs.writeFileSync('benchmark_wasm_results.md', report, 'utf8');
  console.log("WASM vs JS comparison results written to benchmark_wasm_results.md");
}

runBenchmark().catch(console.error);
