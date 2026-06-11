const express = require('express');
const cors = require('cors');
const path = require('path');
const WasmPolarQuantizedStore = require('./wasm-vector-store.cjs');

const app = express();
const PORT = process.env.PORT || 3005;
const VDB_PATH = path.join(__dirname, 'data', 'vectors');
const DIMENSION_DEFAULT = 768;

// Initialize Express middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Dynamic Vector Stores factory
const stores = new Map();
function getStore(dim) {
  const d = parseInt(dim) || DIMENSION_DEFAULT;
  if (d % 2 !== 0) throw new Error('Dimension must be an even integer.');
  if (!stores.has(d)) {
    console.log(`Instantiating new WasmPolarQuantizedStore for dimension ${d}`);
    stores.set(d, new WasmPolarQuantizedStore(VDB_PATH, d, { bits: 3 }));
  }
  return stores.get(d);
}

let extractor = null;
let generator = null;

async function initEmbedder() {
  console.log('Loading local EmbeddingGemma ONNX model...');
  const { pipeline } = await import('@huggingface/transformers');
  extractor = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-300m-ONNX', {
    dtype: 'q8'
  });
  console.log('Local embedding model initialized.');
}

async function getGenerator() {
  if (!generator) {
    console.log('Loading local Gemma-3-270M-IT ONNX model (q4)...');
    const { pipeline } = await import('@huggingface/transformers');
    generator = await pipeline('text-generation', 'onnx-community/gemma-3-270m-it-ONNX', {
      dtype: 'q4'
    });
    console.log('Local Gemma-3-270M IT model loaded.');
  }
  return generator;
}

function getSettings(payload) {
  const settings = payload || {};
  return {
    embedSource: settings.embedSource || 'local',
    embedModel: settings.embedModel || 'embeddinggemma',
    embedDimension: parseInt(settings.embedDimension) || 768,
    llmProvider: settings.llmProvider || 'ollama',
    llmModel: settings.llmModel || 'qwen2.5:1.5b',
    llmUrl: settings.llmUrl || 'http://localhost:11434'
  };
}

function getCollectionName(modelName, dimension) {
  const cleanModel = (modelName || 'local').replace(/[^a-zA-Z0-9]/g, '_');
  return `docs_${cleanModel}_${dimension}`;
}

async function getEmbedding(text, config) {
  if (config.embedSource === 'local') {
    if (!extractor) {
      throw new Error('Local embedding extractor is not initialized yet.');
    }
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
  }

  // External API call (Ollama or LM Studio/OpenAI)
  let url = config.llmUrl;
  
  if (config.embedSource === 'ollama') {
    if (!url.endsWith('/api/embeddings') && !url.endsWith('/v1/embeddings')) {
      url = url.replace(/\/$/, '') + '/api/embeddings';
    }
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: config.embedModel, prompt: text })
    });
    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Ollama Embedding failed (${response.status}): ${err}`);
    }
    const data = await response.json();
    if (!data.embedding) throw new Error('Ollama response missing embedding array.');
    return data.embedding;
  } else {
    // OpenAI Compatible / LM Studio
    let base = url.replace(/\/$/, '');
    if (base.endsWith('/v1')) {
      url = base + '/embeddings';
    } else {
      url = base + '/v1/embeddings';
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: config.embedModel, input: text })
    });
    if (!response.ok) {
      const err = await response.text();
      throw new Error(`OpenAI Compatible Embedding failed (${response.status}): ${err}`);
    }
    const data = await response.json();
    if (!data.data || !data.data[0] || !data.data[0].embedding) {
      throw new Error('OpenAI Compatible response missing embedding array.');
    }
    return data.data[0].embedding;
  }
}

// Chunking helper: splits text into overlapping chunks of word size
function chunkText(text, chunkSize = 200, overlap = 50) {
  const words = text.split(/\s+/);
  const chunks = [];
  if (words.length === 0 || words[0] === "") return chunks;
  
  const step = Math.max(1, chunkSize - overlap);
  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    chunks.push(chunk);
    if (i + chunkSize >= words.length) break;
  }
  return chunks;
}

// Endpoint: Ingest text document
app.post('/api/ingest', async (req, res) => {
  try {
    const { text, settings } = req.body;
    if (!text || text.trim() === "") {
      return res.status(400).json({ error: 'Text content is required' });
    }

    const config = getSettings(settings);
    const chunks = chunkText(text);
    if (chunks.length === 0) {
      return res.json({ success: true, count: 0 });
    }

    const collection = getCollectionName(config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);

    console.log(`Ingesting ${chunks.length} chunks into collection '${collection}'...`);
    
    let count = 0;
    const existingCount = store.count(collection);
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = await getEmbedding(chunk, config);
      const docId = `doc-${existingCount + count}`;
      
      store.set(collection, docId, embedding, { text: chunk });
      count++;
    }

    store.flush();
    console.log(`Ingested ${count} chunks. Total database size: ${store.count(collection)}`);
    
    res.json({ success: true, count, total: store.count(collection) });
  } catch (err) {
    console.error('Ingestion error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Endpoint: Search/Query context
app.post('/api/query', async (req, res) => {
  try {
    const { query, settings } = req.body;
    if (!query) {
      return res.status(400).json({ error: 'Query text is required' });
    }

    const config = getSettings(settings);
    const collection = getCollectionName(config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);

    if (store.count(collection) === 0) {
      return res.json({ results: [], searchTimeMs: 0 });
    }

    const queryEmb = await getEmbedding(query, config);

    const start = process.hrtime();
    const matches = store.search(collection, queryEmb, 3, 0, 'cosine');
    const [seconds, nanoseconds] = process.hrtime(start);
    const searchTimeMs = seconds * 1000 + nanoseconds / 1000000;

    res.json({ results: matches, searchTimeMs });
  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Endpoint: Chat completion (streams SSE)
app.post('/api/chat', async (req, res) => {
  try {
    const { query, context, settings } = req.body;
    if (!query) {
      return res.status(400).json({ error: 'Query text is required' });
    }

    const config = getSettings(settings);

    const systemPrompt = 
      "You are JS-PotatoRAG, a direct, hyper-optimized assistant. Use the provided context to answer the user query. " +
      "IMPORTANT: Do NOT use <think> tags, thinking blocks, or reasoning chains. Output only the final answer. " +
      "Keep it concise, accurate, and direct.";

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Context:\n${context}\n\nQuery: ${query}` }
    ];

    if (config.llmProvider === 'local-onnx') {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      console.log('Using local Gemma-3-270M-IT ONNX model for completions...');
      const gen = await getGenerator();
      const { TextStreamer } = await import('@huggingface/transformers');
      
      class ResponseStreamer extends TextStreamer {
        on_finalized_text(text, stream_end) {
          if (text) {
            res.write(text);
          }
        }
      }

      const streamer = new ResponseStreamer(gen.tokenizer, {
        skip_prompt: true,
        decode_kwargs: { skip_special_tokens: true }
      });

      await gen(messages, {
        max_new_tokens: 512,
        streamer: streamer
      });
      
      res.end();
      return;
    }

    let url = config.llmUrl;
    let payload = {};
    let isOllamaNative = false;

    if (config.llmProvider === 'ollama') {
      if (!url.endsWith('/api/chat') && !url.endsWith('/v1/chat/completions')) {
        url = url.replace(/\/$/, '') + '/api/chat';
        isOllamaNative = true;
      } else if (url.endsWith('/api/chat')) {
        isOllamaNative = true;
      }
    } else {
      if (!url.endsWith('/v1/chat/completions') && !url.endsWith('/chat/completions')) {
        url = url.replace(/\/$/, '') + '/v1/chat/completions';
      }
    }

    payload = {
      model: config.llmModel,
      messages: messages,
      stream: true
    };

    console.log(`Connecting to LLM server at ${url} using model ${config.llmModel}...`);

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`LLM API failed (${response.status}): ${errorText}`);
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader = response.body;
    const decoder = new TextDecoder();

    if (isOllamaNative) {
      for await (const chunk of reader) {
        const chunkText = decoder.decode(chunk, { stream: true });
        const lines = chunkText.split('\n');
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const json = JSON.parse(line);
            const content = json.message?.content || '';
            if (content) res.write(content);
          } catch (e) {
            // ignore partial line parse errors
          }
        }
      }
    } else {
      let buffer = '';
      for await (const chunk of reader) {
        const chunkText = decoder.decode(chunk, { stream: true });
        buffer += chunkText;
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep last incomplete line in buffer

        for (const line of lines) {
          const cleaned = line.trim();
          if (cleaned.startsWith('data: ')) {
            const dataStr = cleaned.slice(6).trim();
            if (dataStr === '[DONE]') continue;
            try {
              const json = JSON.parse(dataStr);
              const content = json.choices[0]?.delta?.content || '';
              if (content) res.write(content);
            } catch (e) {
              // ignore partial line parsing errors
            }
          }
        }
      }
      if (buffer.trim().startsWith('data: ')) {
        const dataStr = buffer.trim().slice(6).trim();
        if (dataStr !== '[DONE]') {
          try {
            const json = JSON.parse(dataStr);
            const content = json.choices[0]?.delta?.content || '';
            if (content) res.write(content);
          } catch (e) {}
        }
      }
    }

    res.end();
  } catch (err) {
    console.error('Chat stream error:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: err.message });
    } else {
      res.end();
    }
  }
});

// Endpoint: DB Stats (receives dynamic settings)
app.post('/api/stats', (req, res) => {
  try {
    const { settings } = req.body;
    const config = getSettings(settings);
    const collection = getCollectionName(config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);

    const count = store.count(collection);
    res.json({
      collection: collection,
      vectorCount: count,
      dimension: config.embedDimension,
      compressionFormat: 'Polar 3-bit (Google TurboQuant-inspired)',
      model: config.embedModel,
      provider: config.llmProvider
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

initEmbedder().then(() => {
  app.listen(PORT, () => {
    console.log(`=======================================================`);
    console.log(`🚀 JS-PotatoRAG backend running on http://localhost:${PORT}`);
    console.log(`Database directory: ${VDB_PATH}`);
    console.log(`Embedding model: Dynamic (Local ONNX/WASM or Remote API)`);
    console.log(`=======================================================`);
  });
}).catch(err => {
  console.error('Failed to initialize local embedding model:', err);
  process.exit(1);
});
