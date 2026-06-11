const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const path = require('path');
const WasmPolarQuantizedStore = require('./wasm-vector-store.cjs');

const app = express();
const PORT = process.env.PORT || 3005;
const HOST = process.env.HOST || '127.0.0.1'; // localhost-only by default; set HOST=0.0.0.0 to expose
const VDB_PATH = path.join(__dirname, 'data', 'vectors');
const DIMENSION_DEFAULT = 768;

// CORS: the UI is served same-origin from this app, so only allow the local
// origin. An open `cors()` lets any web page in the browser read/write the
// vector DB and drive the LLM. Override with ALLOWED_ORIGIN if you proxy it.
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || `http://localhost:${PORT}`;
app.use(cors({ origin: [ALLOWED_ORIGIN, `http://127.0.0.1:${PORT}`] }));
app.use(express.json({ limit: process.env.JSON_LIMIT || '8mb' }));
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
let extractorPromise = null;
let generator = null;

// Lazy: only load the ~300MB local embedder when a request actually needs
// `embedSource: 'local'`. Loading it eagerly at boot blocks startup (and, if it
// failed, killed the process) even when the user configured Ollama/LM Studio.
async function getExtractor() {
  if (extractor) return extractor;
  if (!extractorPromise) {
    extractorPromise = (async () => {
      console.log('Loading local EmbeddingGemma ONNX model...');
      const { pipeline } = await import('@huggingface/transformers');
      extractor = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-300m-ONNX', {
        dtype: 'q8'
      });
      console.log('Local embedding model initialized.');
      return extractor;
    })().catch(err => {
      extractorPromise = null; // allow retry on a later request
      throw err;
    });
  }
  return extractorPromise;
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
    const ext = await getExtractor();
    const output = await ext(text, { pooling: 'mean', normalize: true });
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

// Serialize ingests: the in-memory stores are mutated without locking, and the
// handler awaits between reads and writes. Two concurrent ingests would
// interleave and (with the old `doc-${count}` scheme) collide IDs, overwriting
// each other's chunks. A simple promise chain makes ingest atomic per process.
let ingestQueue = Promise.resolve();

app.post('/api/ingest', async (req, res) => {
  try {
    const { text, settings } = req.body;
    if (!text || text.trim() === "") {
      return res.status(400).json({ error: 'Text content is required' });
    }

    const result = await (ingestQueue = ingestQueue.then(
      () => doIngest(text, settings),
      () => doIngest(text, settings) // keep the chain alive even if a prior ingest threw
    ));

    res.json({ success: true, count: result.count, total: result.total });
  } catch (err) {
    console.error('Ingestion error:', err);
    res.status(500).json({ error: err.message });
  }
});

async function doIngest(text, settings) {
  const config = getSettings(settings);
  const chunks = chunkText(text);
  if (chunks.length === 0) return { count: 0, total: 0 };

  const collection = getCollectionName(config.embedModel, config.embedDimension);
  const store = getStore(config.embedDimension);

  console.log(`Ingesting ${chunks.length} chunks into collection '${collection}'...`);

  let count = 0;
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = await getEmbedding(chunk, config);
    const docId = `doc-${crypto.randomUUID()}`;
    store.set(collection, docId, embedding, { text: chunk });
    count++;
  }

  store.flush();
  const total = store.count(collection);
  console.log(`Ingested ${count} chunks. Total database size: ${total}`);
  return { count, total };
}

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
    // The WASM store always scores by polar-cosine; its search(col, query, limit)
    // takes only 3 args (the extra dimSlice/metric the JS store accepts are N/A here).
    const matches = store.search(collection, queryEmb, 3);
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

// Endpoint: Export a collection as a portable, self-contained JSON artifact.
app.post('/api/export', (req, res) => {
  try {
    const { settings } = req.body;
    const config = getSettings(settings);
    const collection = getCollectionName(config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);

    if (store.count(collection) === 0) {
      return res.status(404).json({ error: `Collection '${collection}' is empty or does not exist.` });
    }

    const data = store.exportCollection(collection);
    res.setHeader('Content-Disposition', `attachment; filename="${collection}.potatorag.json"`);
    res.json(data);
  } catch (err) {
    console.error('Export error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Endpoint: Import a previously exported collection.
app.post('/api/import', (req, res) => {
  try {
    const { data, mode } = req.body;
    if (!data || data.format !== 'potatorag-export') {
      return res.status(400).json({ error: 'Body must include a valid potatorag-export object in `data`.' });
    }
    const dim = parseInt(data.dim) || DIMENSION_DEFAULT;
    const store = getStore(dim);
    const result = store.importCollection(data, { mode: mode === 'merge' ? 'merge' : 'replace' });
    res.json({ success: true, ...result });
  } catch (err) {
    console.error('Import error:', err);
    res.status(400).json({ error: err.message });
  }
});

// ===========================================================================
// AGENT MEMORY API
// A namespaced, filterable read/write/delete surface over the same vector store,
// designed for an agent to persist and recall memories locally. Each memory is
// one discrete record (text + tags + createdAt), addressable by id.
// ===========================================================================

function getMemoryCollection(namespace, model, dim) {
  const ns = (namespace || 'default').replace(/[^a-zA-Z0-9]/g, '_');
  const m = (model || 'local').replace(/[^a-zA-Z0-9]/g, '_');
  return `mem_${ns}_${m}_${dim}`;
}

function tagIntersect(metaTags, queryTags) {
  if (!Array.isArray(queryTags) || queryTags.length === 0) return true;
  if (!Array.isArray(metaTags)) return false;
  return queryTags.some(t => metaTags.includes(t));
}

let memoryQueue = Promise.resolve();

// Write a memory. Body: { text, tags?, namespace?, id?, metadata?, settings? }
app.post('/api/memory/write', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || String(text).trim() === '') {
      return res.status(400).json({ error: 'Memory text is required.' });
    }
    const result = await (memoryQueue = memoryQueue.then(
      () => doMemoryWrite(req.body),
      () => doMemoryWrite(req.body)
    ));
    res.json({ success: true, ...result });
  } catch (err) {
    console.error('Memory write error:', err);
    res.status(500).json({ error: err.message });
  }
});

async function doMemoryWrite(body) {
  const { text, tags, namespace, id, metadata } = body;
  const config = getSettings(body.settings);
  const collection = getMemoryCollection(namespace, config.embedModel, config.embedDimension);
  const store = getStore(config.embedDimension);

  const embedding = await getEmbedding(String(text), config);
  const memId = id || `mem-${crypto.randomUUID()}`;
  const meta = {
    ...(metadata || {}),
    text: String(text),
    tags: Array.isArray(tags) ? tags : [],
    namespace: namespace || 'default',
    createdAt: new Date().toISOString(),
  };
  store.set(collection, memId, embedding, meta);
  store.flush();
  return { id: memId, namespace: namespace || 'default', collection, total: store.count(collection) };
}

// Search memories. Body: { query, k?, namespace?, tags?, filter?, settings? }
app.post('/api/memory/search', async (req, res) => {
  try {
    const { query, k, namespace, tags, filter } = req.body;
    if (!query) return res.status(400).json({ error: 'query is required.' });

    const config = getSettings(req.body.settings);
    const collection = getMemoryCollection(namespace, config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);
    if (store.count(collection) === 0) return res.json({ results: [] });

    const limit = parseInt(k) || 5;
    const queryEmb = await getEmbedding(String(query), config);

    // Fetch extra when tag-filtering (tags use array-intersection, not matchFilter).
    const wantTags = Array.isArray(tags) && tags.length > 0;
    const fetchN = wantTags ? Math.max(limit * 5, 50) : limit;
    let matches = store.search(collection, queryEmb, fetchN, filter || null);
    if (wantTags) matches = matches.filter(m => tagIntersect(m.metadata.tags, tags)).slice(0, limit);

    res.json({
      results: matches.map(m => ({
        id: m.id,
        score: m.score,
        text: m.metadata.text,
        tags: m.metadata.tags || [],
        createdAt: m.metadata.createdAt,
        metadata: m.metadata,
      })),
    });
  } catch (err) {
    console.error('Memory search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Forget memories. Body: { id?, namespace?, tags?, filter?, settings? }
app.post('/api/memory/forget', async (req, res) => {
  try {
    const { id, namespace, tags, filter } = req.body;
    const config = getSettings(req.body.settings);
    const collection = getMemoryCollection(namespace, config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);

    let removed = 0;
    if (id) {
      if (store.remove(collection, id)) removed = 1;
    } else if (filter || (Array.isArray(tags) && tags.length > 0)) {
      const candidates = store.list(collection, filter || null)
        .filter(r => tagIntersect(r.metadata.tags, tags));
      for (const c of candidates) if (store.remove(collection, c.id)) removed++;
    } else {
      return res.status(400).json({ error: 'Provide an id, tags, or a filter to forget.' });
    }
    store.flush();
    res.json({ success: true, removed, total: store.count(collection) });
  } catch (err) {
    console.error('Memory forget error:', err);
    res.status(500).json({ error: err.message });
  }
});

// List memories. Body: { namespace?, limit?, filter?, tags?, settings? }
app.post('/api/memory/list', (req, res) => {
  try {
    const { namespace, limit, filter, tags } = req.body;
    const config = getSettings(req.body.settings);
    const collection = getMemoryCollection(namespace, config.embedModel, config.embedDimension);
    const store = getStore(config.embedDimension);

    let items = store.list(collection, filter || null)
      .filter(r => tagIntersect(r.metadata.tags, tags));
    if (limit) items = items.slice(0, parseInt(limit));

    res.json({
      count: items.length,
      items: items.map(r => ({
        id: r.id,
        text: r.metadata.text,
        tags: r.metadata.tags || [],
        createdAt: r.metadata.createdAt,
      })),
    });
  } catch (err) {
    console.error('Memory list error:', err);
    res.status(500).json({ error: err.message });
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

app.listen(PORT, HOST, () => {
  console.log(`=======================================================`);
  console.log(`🚀 JS-PotatoRAG backend running on http://${HOST}:${PORT}`);
  console.log(`Database directory: ${VDB_PATH}`);
  console.log(`Embedding model: Dynamic (Local ONNX/WASM or Remote API)`);
  console.log(`Local embedder loads lazily on first 'local' embed request.`);
  console.log(`=======================================================`);
});
