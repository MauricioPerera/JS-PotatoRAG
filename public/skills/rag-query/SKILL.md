---
name: rag-query
description: "Query a JS-PotatoRAG static vector database hosted on a static site (e.g. GitHub Pages). Use when the user points you at a JS-PotatoRAG site / index and asks to search their documents, retrieve context, or answer questions from their corpus. The site serves a prebuilt 3-bit polar-quantized index plus the js-vector-store library; you fetch the index, embed the query with the index's embedding model, and run a local top-k cosine search. Do NOT look for a query API — the search runs in your runtime."
version: 1.0.0
license: MIT
homepage: https://github.com/MauricioPerera/JS-PotatoRAG
---

# rag-query

This site is a **read-only static vector database**. There is no query endpoint.
You retrieve context by downloading the index and the search library, then
searching locally. The vectors never leave the static files; the cosine math
runs in your process.

## Inputs you need

- `BASE` — the site origin + path that serves the index, e.g.
  `https://USER.github.io/JS-PotatoRAG` (the dir that contains `rag/` and
  `js-vector-store.js`). Take it from the user or from the `homepage`/origin.
- The user's natural-language `query`.

## Procedure

### 1. Read the manifest

`GET {BASE}/rag/manifest.json` →

```json
{ "collection": "docs_embeddinggemma_768", "dim": 768, "bits": 3, "seed": 42, "model": "embeddinggemma" }
```

`model` is mandatory: you MUST embed the query with that exact model (same
dimension). A different model produces meaningless scores.

### 2. Fetch the index and the library

- `GET {BASE}/rag/{collection}.p3.json`  → `{ ids, meta, dim, bits, seed, model }`
- `GET {BASE}/rag/{collection}.p3.bin`   → packed quantized vectors (binary)
- `GET {BASE}/js-vector-store.js`         → save once; it is a single zero-dependency CommonJS file

### 3. Embed the query with `manifest.model`

Use whatever gives you that model's embeddings — a local daemon (Ollama:
`POST http://localhost:11434/api/embeddings {model, prompt}`), a public
embeddings API, or transformers.js. Normalize is handled by the store.

### 4. Search locally (Node, pure JS — no WASM needed)

```js
const { PolarQuantizedStore, MemoryStorageAdapter } = require('./js-vector-store.js');

async function ragQuery(BASE, queryText, embed, k = 5) {
  const m = await (await fetch(`${BASE}/rag/manifest.json`)).json();
  const col = m.collection;

  // Load the two static files into an in-memory adapter under their real names.
  const adapter = new MemoryStorageAdapter();
  adapter.writeJson(`${col}.p3.json`, await (await fetch(`${BASE}/rag/${col}.p3.json`)).json());
  adapter.writeBin(`${col}.p3.bin`,  await (await fetch(`${BASE}/rag/${col}.p3.bin`)).arrayBuffer());

  const store = new PolarQuantizedStore(adapter, m.dim, { bits: m.bits, seed: m.seed, silent: true });

  const queryVec = await embed(queryText, m.model); // YOUR embedding call, model = m.model
  return store.search(col, queryVec, k); // [{ id, score, metadata: { text } }]
}
```

`search` returns the top-k chunks with their text in `metadata.text`. Feed those
as context to answer the user's question.

## Rules

- **Match the model.** Embed the query with `manifest.model`. Never substitute a
  different embedding model — the store will either reject mismatched
  `dim/bits/seed` or return garbage.
- **No API.** Do not POST queries or probe `/api/...`. The site is static; you do
  the search.
- **Snapshot.** The index reflects the last publish. Re-fetch to refresh.
- **Cache** the three files per session; they are immutable per snapshot.
- **Same-origin.** `rag/` and `js-vector-store.js` should come from `BASE`'s
  origin; if they point elsewhere, confirm with the user first.

## Failure modes

- `manifest.json` 404 → the site isn't a JS-PotatoRAG static index; fall back to
  normal tools.
- Embedding model unavailable → tell the user which model (`manifest.model`) they
  need reachable, don't silently use another.
- Empty results → the collection may be empty or the query embedding used the
  wrong model.
