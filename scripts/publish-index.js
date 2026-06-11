#!/usr/bin/env node
// Publish a collection as a static index that an agent can consume from a static
// host (e.g. GitHub Pages). Copies the collection's two files into public/rag/
// and writes public/rag/manifest.json describing it.
//
// Usage:
//   node scripts/publish-index.js <collection> [embeddingModelName]
//
// Example:
//   node scripts/publish-index.js docs_embeddinggemma_768 embeddinggemma
//
// The collection lives in data/vectors/<collection>.p3.{json,bin} (created by
// ingesting through the server). The agent reads manifest.json to learn which
// embedding model to use for the query — pass the real model name as the 2nd
// arg, since the collection name only carries a sanitized version of it.

const fs = require('fs');
const path = require('path');

const collection = process.argv[2];
if (!collection) {
  console.error('Usage: node scripts/publish-index.js <collection> [embeddingModelName]');
  process.exit(1);
}

const VDB = path.join(__dirname, '..', 'data', 'vectors');
const OUT = path.join(__dirname, '..', 'public', 'rag');

const jsonSrc = path.join(VDB, `${collection}.p3.json`);
const binSrc = path.join(VDB, `${collection}.p3.bin`);

if (!fs.existsSync(jsonSrc) || !fs.existsSync(binSrc)) {
  console.error(`Collection not found: ${jsonSrc} (and .p3.bin). Ingest it first.`);
  process.exit(1);
}

const manifestSrc = JSON.parse(fs.readFileSync(jsonSrc, 'utf8'));
// Recover the embedding model: explicit arg wins; else the value stored in the
// manifest; else best-effort from the collection name (docs_<model>_<dim>).
const parsedModel = (() => {
  const m = /^docs_(.+)_(\d+)$/.exec(collection);
  return m ? m[1] : null;
})();
const model = process.argv[3] || manifestSrc.model || parsedModel || 'unknown';

fs.mkdirSync(OUT, { recursive: true });
fs.copyFileSync(jsonSrc, path.join(OUT, `${collection}.p3.json`));
fs.copyFileSync(binSrc, path.join(OUT, `${collection}.p3.bin`));

const manifest = {
  format: 'js-potatorag-index',
  version: 1,
  collection,
  dim: manifestSrc.dim,
  bits: manifestSrc.bits,
  seed: manifestSrc.seed,
  model,
  count: Array.isArray(manifestSrc.ids) ? manifestSrc.ids.length : 0,
};
fs.writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2));

console.log(`Published '${collection}' (${manifest.count} chunks, model=${model}) to public/rag/`);
console.log('Files: manifest.json, ' + `${collection}.p3.json, ${collection}.p3.bin`);
console.log('Commit public/rag/ (and public/js-vector-store.js) to your static host.');
