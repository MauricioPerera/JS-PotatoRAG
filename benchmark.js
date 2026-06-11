const { PolarQuantizedStore, FileStorageAdapter } = require('./js-vector-store');
const ollama = require('ollama').default || require('ollama');
const fs = require('fs');
const path = require('path');

// Technical corpus for testing RAG performance (identical to Python benchmark)
const sampleParagraphs = [
  "Google's TurboQuant is a data-oblivious vector quantization algorithm developed to enable high compression for high-dimensional vector embeddings, particularly for Large Language Model Serving and nearest neighbor search engines.",
  "Unlike traditional Product Quantization methods, TurboQuant does not require any codebook training or calibration dataset, making it ideal for online, dynamic ingestion pipelines where data is added incrementally.",
  "TurboQuant works by performing random orthogonal rotations on the input vectors, which concentrates the distribution of coordinates. It then applies optimized scalar quantizers to compress each coordinate to low bit-widths, such as 4-bit representation.",
  "By using Matryoshka Representation Learning (MRL), Google's EmbeddingGemma model allows flexible output dimensions ranging from 128 to 768, enabling developers to trade off accuracy for storage and retrieval speed.",
  "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM generation by retrieving relevant context documents from a vector database and passing them to the generator model alongside the user query.",
  "EmbeddingGemma is a specialized embedding model with 300 million parameters, built on the Gemma 3 architecture, and optimized for generating high-quality semantic representations on consumer-grade local hardware.",
  "Streamlit is a popular open-source Python framework used to build interactive, web-based data applications quickly. In PotatoRAG, it provides the frontend chat interface and displays search latencies to the user.",
  "SIMD (Single Instruction, Multiple Data) execution, using AVX-512 or NEON instructions, allows turbovec to perform millions of vector dot product comparisons per second directly on compressed, quantized representations on standard CPUs.",
  "Local LLM serving using Ollama allows running model architectures like Llama, Gemma, and Qwen entirely on-device, offering high privacy, zero latency variance from internet connection, and zero subscription costs.",
  "PotatoRAG is designed to run efficiently on low-resource machines with as little as 4GB of RAM by leveraging the extreme compression of TurboQuant 4-bit indexing combined with lightweight 1.3B parameter LLMs."
];

// Duplicate paragraphs to simulate a larger document (50 chunks total)
const testCorpus = [];
for (let i = 0; i < 5; i++) {
  testCorpus.push(...sampleParagraphs);
}

// Sample queries for evaluation
const testQueries = [
  "What is TurboQuant?",
  "Does TurboQuant require training?",
  "How does TurboQuant compress vectors?",
  "What is the default dimension of EmbeddingGemma?",
  "What framework is used for the PotatoRAG frontend?",
  "Why run local LLMs with Ollama?",
  "Explain RAG retrieval workflow.",
  "How does SIMD speed up vector search?",
  "What are the benefits of Matryoshka Representation Learning?",
  "Can PotatoRAG run on 4GB RAM?"
];

async function runBenchmark() {
  console.log("============================================================");
  console.log("STARTING JS-POTATORAG (POLAR 3-BIT VDB) BENCHMARK");
  console.log("============================================================");

  const benchDir = path.join(__dirname, 'data', 'bench-vectors');
  // Clear any existing bench data
  if (fs.existsSync(benchDir)) {
    fs.rmSync(benchDir, { recursive: true, force: true });
  }

  const store = new PolarQuantizedStore(new FileStorageAdapter(benchDir), 768, { bits: 3, silent: true });

  // 1. Ingestion Benchmark
  console.log(`\nPhase 1: Ingesting ${testCorpus.length} text chunks...`);
  
  const ingestTimes = [];
  const startIngest = process.hrtime();

  for (let i = 0; i < testCorpus.length; i++) {
    const chunk = testCorpus[i];
    const t0 = process.hrtime();
    
    const response = await ollama.embeddings({
      model: 'embeddinggemma',
      prompt: chunk
    });
    
    store.set('docs', `doc-${i}`, response.embedding, { text: chunk });
    
    const [diffSec, diffNsec] = process.hrtime(t0);
    ingestTimes.push(diffSec * 1000 + diffNsec / 1000000);
  }
  
  store.flush();
  
  const [totalSec, totalNsec] = process.hrtime(startIngest);
  const totalIngestTime = totalSec + totalNsec / 1000000000;
  
  const avgIngestTime = ingestTimes.reduce((a, b) => a + b, 0) / ingestTimes.length;
  const variance = ingestTimes.reduce((a, b) => a + Math.pow(b - avgIngestTime, 2), 0) / ingestTimes.length;
  const stdIngestTime = Math.sqrt(variance);

  console.log(`OK: Ingestion complete.`);
  console.log(`   - Total Ingestion Time: ${totalIngestTime.toFixed(3)} s`);
  console.log(`   - Avg Time per Chunk: ${avgIngestTime.toFixed(2)} ms (plus/minus ${stdIngestTime.toFixed(2)} ms)`);

  // 2. Search/Retrieval Benchmark
  console.log(`\nPhase 2: Running ${testQueries.length} search queries (k=3)...`);
  
  const totalQueryTimes = [];
  const polarStoreTimes = [];
  
  for (const query of testQueries) {
    const startTotal = process.hrtime();
    
    const response = await ollama.embeddings({
      model: 'embeddinggemma',
      prompt: query
    });
    const queryEmb = response.embedding;
    
    const startVdb = process.hrtime();
    const results = store.search('docs', queryEmb, 3, 0, 'cosine');
    const [vdbSec, vdbNsec] = process.hrtime(startVdb);
    
    const [totalSec, totalNsec] = process.hrtime(startTotal);
    
    totalQueryTimes.push(totalSec * 1000 + totalNsec / 1000000);
    polarStoreTimes.push(vdbSec * 1000 + vdbNsec / 1000000);
  }
  
  const avgTotalQuery = totalQueryTimes.reduce((a, b) => a + b, 0) / totalQueryTimes.length;
  const avgPolarStore = polarStoreTimes.reduce((a, b) => a + b, 0) / polarStoreTimes.length;

  console.log(`OK: Search benchmark complete.`);
  console.log(`   - Avg Total Query Time (including Ollama Embedding): ${avgTotalQuery.toFixed(2)} ms`);
  console.log(`   - Avg PolarStore Search Time (vector index scan only): ${avgPolarStore.toFixed(4)} ms`);

  // 3. Memory Compression Analysis
  const dim = 768;
  const numVectors = testCorpus.length;
  const bytesPerVec = store.bytesPerVector();
  
  const rawSizeBytes = numVectors * dim * 4;
  const quantizedSizeBytes = numVectors * bytesPerVec;
  
  const compressionRatio = rawSizeBytes / quantizedSizeBytes;
  const savingsPct = (1 - (quantizedSizeBytes / rawSizeBytes)) * 100;

  console.log(`\nPhase 3: Memory compression analysis (${numVectors} vectors, dim=${dim})`);
  console.log(`   - Raw float32 Memory: ${(rawSizeBytes / 1024).toFixed(2)} KB`);
  console.log(`   - Quantized 3-bit Memory: ${(quantizedSizeBytes / 1024).toFixed(2)} KB`);
  console.log(`   - Compression Ratio: ${compressionRatio.toFixed(1)}x`);
  console.log(`   - Memory Savings: ${savingsPct.toFixed(1)}%`);
  console.log("============================================================");

  // Clean up benchmark database files
  fs.rmSync(benchDir, { recursive: true, force: true });
  
  // Write the markdown results
  const report = `# JS-PotatoRAG Benchmark Results (Polar 3-bit)

* **Date**: 2026-06-11
* **Embedding Model**: \`embeddinggemma\`
* **Database**: \`PolarQuantizedStore\` (3-bit)

## Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Total Chunks** | ${numVectors} |
| **Total Ingest Time** | ${totalIngestTime.toFixed(3)} s |
| **Avg Ingest per Chunk** | ${avgIngestTime.toFixed(2)} ms |
| **Avg Total Search Latency** | ${avgTotalQuery.toFixed(2)} ms |
| **Avg Polar VDB Latency (Index Only)** | ${avgPolarStore.toFixed(4)} ms |
| **Quantized Memory Footprint** | ${(quantizedSizeBytes / 1024).toFixed(2)} KB |
| **Compression Ratio** | ${compressionRatio.toFixed(1)}x |
`;
  
  fs.writeFileSync('benchmark_results.md', report, 'utf8');
  console.log("Benchmark results written to benchmark_results.md");
}

runBenchmark().catch(console.error);
