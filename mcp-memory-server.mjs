#!/usr/bin/env node
// MCP server exposing JS-PotatoRAG as a local agent memory.
//
// Tools: memory_write, memory_search, memory_forget, memory_list.
// It is a thin client over the HTTP API (single source of truth for storage,
// persistence and locking) — start `npm start` first, then point your MCP
// client at this server.
//
// Configuration (environment variables):
//   MEMORY_API_URL   base URL of the running server   (default http://127.0.0.1:3005)
//   MEMORY_NAMESPACE default namespace for memories   (default "default")
//   EMBED_SOURCE     "ollama" | "local" | "openai"    (default "ollama")
//   EMBED_MODEL      embedding model name             (default "embeddinggemma")
//   EMBED_DIM        embedding dimension              (default 768)
//   EMBED_URL        embedding API url (ollama/openai) (optional)

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const API_URL = (process.env.MEMORY_API_URL || 'http://127.0.0.1:3005').replace(/\/$/, '');
const DEFAULT_NS = process.env.MEMORY_NAMESPACE || 'default';
const SETTINGS = {
  embedSource: process.env.EMBED_SOURCE || 'ollama',
  embedModel: process.env.EMBED_MODEL || 'embeddinggemma',
  embedDimension: parseInt(process.env.EMBED_DIM || '768', 10),
  ...(process.env.EMBED_URL ? { llmUrl: process.env.EMBED_URL } : {}),
};

async function api(pathname, body) {
  let res;
  try {
    res = await fetch(`${API_URL}${pathname}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, settings: SETTINGS }),
    });
  } catch (e) {
    throw new Error(`Cannot reach the memory server at ${API_URL} (is "npm start" running?): ${e.message}`);
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
  return json;
}

const ok = (obj) => ({ content: [{ type: 'text', text: JSON.stringify(obj, null, 2) }] });
const fail = (e) => ({ content: [{ type: 'text', text: `Error: ${e.message}` }], isError: true });

const server = new McpServer({ name: 'potatorag-memory', version: '1.0.0' });

server.registerTool(
  'memory_write',
  {
    title: 'Write memory',
    description: 'Persist a discrete memory (a fact, preference, or note) into the local vector store so it can be recalled semantically later. Returns the memory id.',
    inputSchema: {
      text: z.string().describe('The memory content to store.'),
      tags: z.array(z.string()).optional().describe('Optional tags for filtering (e.g. ["prefs","ops"]).'),
      namespace: z.string().optional().describe(`Namespace to isolate memories (default "${DEFAULT_NS}").`),
      id: z.string().optional().describe('Optional explicit id; if it exists it is overwritten.'),
    },
  },
  async ({ text, tags, namespace, id }) => {
    try {
      const r = await api('/api/memory/write', { text, tags, namespace: namespace || DEFAULT_NS, id });
      return ok({ id: r.id, namespace: r.namespace, total: r.total });
    } catch (e) { return fail(e); }
  }
);

server.registerTool(
  'memory_search',
  {
    title: 'Search memory',
    description: 'Recall the most relevant memories for a query using local semantic (vector) search. Optionally filter by tags.',
    inputSchema: {
      query: z.string().describe('What to recall.'),
      k: z.number().int().positive().optional().describe('How many memories to return (default 5).'),
      namespace: z.string().optional().describe(`Namespace to search (default "${DEFAULT_NS}").`),
      tags: z.array(z.string()).optional().describe('Only return memories carrying at least one of these tags.'),
    },
  },
  async ({ query, k, namespace, tags }) => {
    try {
      const r = await api('/api/memory/search', { query, k: k || 5, namespace: namespace || DEFAULT_NS, tags });
      return ok(r.results.map(m => ({ id: m.id, score: Number(m.score?.toFixed?.(4) ?? m.score), text: m.text, tags: m.tags, createdAt: m.createdAt })));
    } catch (e) { return fail(e); }
  }
);

server.registerTool(
  'memory_forget',
  {
    title: 'Forget memory',
    description: 'Delete memories by id, or by tags (any match). Provide either id or tags.',
    inputSchema: {
      id: z.string().optional().describe('Delete a single memory by id.'),
      tags: z.array(z.string()).optional().describe('Delete all memories carrying at least one of these tags.'),
      namespace: z.string().optional().describe(`Namespace to operate on (default "${DEFAULT_NS}").`),
    },
  },
  async ({ id, tags, namespace }) => {
    try {
      if (!id && !(Array.isArray(tags) && tags.length)) throw new Error('Provide an id or tags to forget.');
      const r = await api('/api/memory/forget', { id, tags, namespace: namespace || DEFAULT_NS });
      return ok({ removed: r.removed, total: r.total });
    } catch (e) { return fail(e); }
  }
);

server.registerTool(
  'memory_list',
  {
    title: 'List memories',
    description: 'List stored memories in a namespace (most useful for inspection/debugging). Optionally filter by tags.',
    inputSchema: {
      namespace: z.string().optional().describe(`Namespace to list (default "${DEFAULT_NS}").`),
      limit: z.number().int().positive().optional().describe('Max items to return.'),
      tags: z.array(z.string()).optional().describe('Only list memories carrying at least one of these tags.'),
    },
  },
  async ({ namespace, limit, tags }) => {
    try {
      const r = await api('/api/memory/list', { namespace: namespace || DEFAULT_NS, limit, tags });
      return ok({ count: r.count, items: r.items });
    } catch (e) { return fail(e); }
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error(`[potatorag-memory] MCP server ready · API=${API_URL} · namespace=${DEFAULT_NS} · embed=${SETTINGS.embedSource}/${SETTINGS.embedModel}`);
