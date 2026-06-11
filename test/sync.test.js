// Guards against drift between the canonical library and the copy served to the
// browser / fetched by agents from the static site. They MUST be byte-identical:
// the agent's pure-JS search (public/js-vector-store.js) has to use the exact
// same quantizer (PRNG, packing) as the server that built the index, or results
// are silently wrong.
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

test('public/js-vector-store.js is identical to the root js-vector-store.js', () => {
  const root = fs.readFileSync(path.join(__dirname, '..', 'js-vector-store.js'), 'utf8');
  const pub = fs.readFileSync(path.join(__dirname, '..', 'public', 'js-vector-store.js'), 'utf8');
  assert.strictEqual(
    pub, root,
    'public/js-vector-store.js drifted from the root copy. Run: cp js-vector-store.js public/js-vector-store.js'
  );
});
