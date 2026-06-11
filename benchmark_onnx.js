const fs = require('fs');

async function testONNX() {
  console.log("Loading local EmbeddingGemma ONNX model...");
  const { pipeline } = await import('@huggingface/transformers');
  const startLoad = process.hrtime();
  const extractor = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-300m-ONNX', {
    dtype: 'q8'
  });
  const [loadS, loadNs] = process.hrtime(startLoad);
  console.log(`Model loaded in ${(loadS * 1000 + loadNs / 1e6).toFixed(2)} ms.`);

  const text = "JS-PotatoRAG is an ultra-low memory local RAG application. It utilizes a 3-bit Polar Quantized vector store written in Rust and compiled to WebAssembly (WASM).";
  
  const N = 50;
  console.log(`Generating ${N} embeddings sequentially...`);
  
  // Warmup
  await extractor(text, { pooling: 'mean', normalize: true });
  
  const startEmbed = process.hrtime();
  for (let i = 0; i < N; i++) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    // Convert to standard JS array to simulate actual usage
    const arr = Array.from(output.data);
  }
  const [embedS, embedNs] = process.hrtime(startEmbed);
  const totalMs = embedS * 1000 + embedNs / 1e6;
  const avgMs = totalMs / N;
  
  console.log(`Total time for ${N} embeddings: ${totalMs.toFixed(2)} ms`);
  console.log(`Average time per embedding: ${avgMs.toFixed(2)} ms`);
  
  const results = {
    model: 'onnx-community/embeddinggemma-300m-ONNX (q8)',
    count: N,
    totalTimeMs: totalMs,
    avgTimeMs: avgMs
  };
  
  fs.writeFileSync('benchmark_onnx_results.json', JSON.stringify(results, null, 2), 'utf8');
}

testONNX().catch(console.error);
