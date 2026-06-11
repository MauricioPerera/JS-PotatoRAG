const fs = require('fs');

async function testGemma3() {
  console.log("Loading local Gemma-3-270M-IT ONNX model (q4)...");
  const { pipeline } = await import('@huggingface/transformers');
  const startLoad = process.hrtime();
  const generator = await pipeline('text-generation', 'onnx-community/gemma-3-270m-it-ONNX', {
    dtype: 'q4'
  });
  const [loadS, loadNs] = process.hrtime(startLoad);
  console.log(`Model loaded in ${(loadS * 1000 + loadNs / 1e6).toFixed(2)} ms.`);

  const messages = [
    { role: 'user', content: "Explain Progressive Web Apps in one short sentence." }
  ];
  
  // Warmup run
  console.log("Running warmup generation...");
  await generator(messages, { max_new_tokens: 32 });
  
  console.log("Running benchmark generation...");
  const startGen = process.hrtime();
  const output = await generator(messages, {
    max_new_tokens: 64,
    return_full_text: false
  });
  const [genS, genNs] = process.hrtime(startGen);
  const genMs = genS * 1000 + genNs / 1e6;
  
  // Decoded text response
  const text = output[0]?.generated_text || '';
  const lastMsg = Array.isArray(text) ? text[text.length - 1]?.content : text;
  
  const cleanText = typeof lastMsg === 'string' ? lastMsg : JSON.stringify(lastMsg);
  
  // Estimate tokens by splitting by whitespace / punctuation
  const wordCount = cleanText.split(/\s+/).filter(Boolean).length;
  const tokenCount = Math.max(1, Math.round(wordCount * 1.3)); 
  const tokensPerSec = (tokenCount / (genMs / 1000)).toFixed(2);
  
  console.log(`Generated text: "${cleanText.trim()}"`);
  console.log(`Tokens generated (estimated): ${tokenCount}`);
  console.log(`Time taken: ${genMs.toFixed(2)} ms`);
  console.log(`Speed: ${tokensPerSec} tokens/sec`);

  const results = {
    model: 'onnx-community/gemma-3-270m-it-ONNX (q4)',
    generationTimeMs: genMs,
    estimatedTokens: tokenCount,
    tokensPerSecond: parseFloat(tokensPerSec),
    generatedText: cleanText.trim()
  };
  
  fs.writeFileSync('benchmark_gemma3_results.json', JSON.stringify(results, null, 2), 'utf8');
}

testGemma3().catch(console.error);
