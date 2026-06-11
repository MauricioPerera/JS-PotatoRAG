const WasmPolarStore = require('./wasm-polar-store.cjs');
const { FileStorageAdapter, matchFilter } = require('./js-vector-store');

class WasmPolarQuantizedStore {
  constructor(dirOrAdapter, dim = 768, opts = {}) {
    if (dim % 2 !== 0) throw new Error('WasmPolarQuantizedStore: dim must be even');
    this.dim = dim;
    this.bits = opts.bits || 3;
    this.seed = opts.seed ?? 42;
    this.defaultModel = opts.model || null;
    
    this._adapter = typeof dirOrAdapter === 'string'
      ? new FileStorageAdapter(dirOrAdapter)
      : dirOrAdapter;
      
    this._collections = new Map();
    this.wasmStore = new WasmPolarStore(dim, this.bits, this.seed);
    this.bytesPerVec = this.wasmStore.bytesPerVector();
  }

  _binFile(col)  { return `${col}.p3.bin`; }
  _jsonFile(col) { return `${col}.p3.json`; }

  _load(col) {
    if (this._collections.has(col)) return this._collections.get(col);
    const manifest = this._adapter.readJson(this._jsonFile(col));
    // Guard: the polar quantizer is parametrized by (dim, bits, seed). Reading a
    // collection quantized with different params yields silently-garbage scores.
    if (manifest) {
      const mDim = manifest.dim, mBits = manifest.bits, mSeed = manifest.seed;
      if ((mDim !== undefined && mDim !== this.dim) ||
          (mBits !== undefined && mBits !== this.bits) ||
          (mSeed !== undefined && mSeed !== this.seed)) {
        throw new Error(
          `Collection '${col}' was quantized with dim=${mDim} bits=${mBits} seed=${mSeed}, ` +
          `but this store uses dim=${this.dim} bits=${this.bits} seed=${this.seed}. ` +
          `Re-ingest the collection or open it with matching parameters.`
        );
      }
    }
    const ids   = manifest ? manifest.ids  : [];
    const meta  = manifest ? manifest.meta : [];
    const model = manifest?.model || this.defaultModel || null;
    
    const idMap = new Map();
    for (let i = 0; i < ids.length; i++) idMap.set(ids[i], i);
    
    const bin = this._adapter.readBin(this._binFile(col));
    const entry = { ids, meta, idMap, bin, model, pending: [], dirty: false };
    this._collections.set(col, entry);
    return entry;
  }

  set(col, id, vector, metadata = {}) {
    const entry = this._load(col);
    const existing = entry.idMap.get(id);
    const packed = this.wasmStore.quantize(vector);

    if (existing !== undefined) {
      const committed = entry.ids.length - entry.pending.length;
      if (existing < committed && entry.bin) {
        new Uint8Array(entry.bin).set(packed, existing * this.bytesPerVec);
      } else if (existing >= committed) {
        entry.pending[existing - committed].packed = packed;
      }
      entry.meta[existing] = metadata;
    } else {
      const idx = entry.ids.length;
      entry.ids.push(id);
      entry.meta.push(metadata);
      entry.idMap.set(id, idx);
      entry.pending.push({ id, packed, metadata });
    }
    entry.dirty = true;
  }

  // Materialize pending vectors into the in-memory packed buffer WITHOUT any
  // disk I/O. Search needs an up-to-date `bin` but must not write to disk — a
  // read causing a write is surprising, breaks on a read-only FS, and races
  // with concurrent persistence.
  _materialize(entry) {
    if (entry.pending.length === 0) return;
    const committed = entry.ids.length - entry.pending.length;
    const total = entry.ids.length;
    const bpv = this.bytesPerVec;
    const newBuf = new ArrayBuffer(total * bpv);
    const dst = new Uint8Array(newBuf);

    if (entry.bin && committed > 0) {
      dst.set(new Uint8Array(entry.bin, 0, committed * bpv));
    }

    for (let p = 0; p < entry.pending.length; p++) {
      dst.set(entry.pending[p].packed, (committed + p) * bpv);
    }
    entry.bin = newBuf;
    entry.pending = [];
  }

  _flushCol(col, entry) {
    this._materialize(entry);
    if (entry.bin) this._adapter.writeBin(this._binFile(col), entry.bin);

    const manifest = { ids: entry.ids, meta: entry.meta, dim: this.dim, bits: this.bits, seed: this.seed };
    if (entry.model) manifest.model = entry.model;

    this._adapter.writeJson(this._jsonFile(col), manifest);
    entry.dirty = false;
  }

  flush() {
    for (const [col, entry] of this._collections) {
      if (entry.dirty) this._flushCol(col, entry);
    }
  }

  count(col) {
    return this._load(col).ids.length;
  }

  // Export a collection as a single self-contained, portable object:
  // quantizer params + ids + chunk metadata + the packed vectors (base64).
  // This is the whole DB for that collection — no embeddings need regenerating
  // to move it to another machine.
  exportCollection(col) {
    const entry = this._load(col);
    this._materialize(entry); // bring pending into bin (no disk write)
    const binBytes = entry.bin ? new Uint8Array(entry.bin) : new Uint8Array(0);
    return {
      format: 'potatorag-export',
      version: 1,
      collection: col,
      dim: this.dim,
      bits: this.bits,
      seed: this.seed,
      model: entry.model || this.defaultModel || null,
      count: entry.ids.length,
      ids: entry.ids.slice(),
      meta: entry.meta.slice(),
      bin: Buffer.from(binBytes).toString('base64'),
    };
  }

  // Import a previously exported collection. `mode`:
  //   'replace' (default) — overwrite the target collection wholesale.
  //   'merge'             — append the imported vectors (new ids on collision).
  importCollection(data, { mode = 'replace' } = {}) {
    if (!data || data.format !== 'potatorag-export') {
      throw new Error('Unrecognized export format.');
    }
    // `collection` becomes a filename (`${col}.p3.bin`). Reject anything that
    // isn't a plain token to prevent path traversal from an untrusted export.
    if (typeof data.collection !== 'string' || !/^[a-zA-Z0-9_]+$/.test(data.collection)) {
      throw new Error('Invalid collection name in export (must match [a-zA-Z0-9_]).');
    }
    if (data.dim !== this.dim || data.bits !== this.bits || data.seed !== this.seed) {
      throw new Error(
        `Incompatible quantizer params: export has dim=${data.dim} bits=${data.bits} seed=${data.seed}, ` +
        `this store uses dim=${this.dim} bits=${this.bits} seed=${this.seed}.`
      );
    }
    const col = data.collection;
    const binBuf = Buffer.from(data.bin || '', 'base64');
    const expected = data.count * this.bytesPerVec;
    if (binBuf.length !== expected) {
      throw new Error(`Corrupt export: packed length ${binBuf.length} != expected ${expected}.`);
    }
    const bpv = this.bytesPerVec;

    if (mode === 'merge') {
      const entry = this._load(col);
      const seen = entry.idMap;
      for (let i = 0; i < data.count; i++) {
        let id = data.ids[i];
        if (seen.has(id)) id = `${id}-imp-${i}`;
        const packed = new Uint8Array(binBuf.buffer, binBuf.byteOffset + i * bpv, bpv).slice();
        const idx = entry.ids.length;
        entry.ids.push(id);
        entry.meta.push(data.meta[i]);
        entry.idMap.set(id, idx);
        entry.pending.push({ id, packed, metadata: data.meta[i] });
      }
      entry.dirty = true;
      this._flushCol(col, entry);
      return { collection: col, imported: data.count, total: entry.ids.length, mode };
    }

    // replace
    const ab = binBuf.buffer.slice(binBuf.byteOffset, binBuf.byteOffset + binBuf.byteLength);
    const idMap = new Map();
    for (let i = 0; i < data.ids.length; i++) idMap.set(data.ids[i], i);
    const entry = {
      ids: data.ids.slice(),
      meta: data.meta.slice(),
      idMap,
      bin: ab,
      model: data.model || this.defaultModel || null,
      pending: [],
      dirty: true,
    };
    this._collections.set(col, entry);
    this._flushCol(col, entry);
    return { collection: col, imported: data.count, total: entry.ids.length, mode: 'replace' };
  }

  // search(col, query, limit, filter?)
  // `filter` is an optional metadata filter (see matchFilter: $and/$or/$gt/$in/…).
  // When a filter is present we fetch the full ranking from WASM and filter in JS,
  // since the WASM top-k can't know about metadata.
  search(col, query, limit = 5, filter = null) {
    const entry = this._load(col);
    // Bring pending vectors into the searchable buffer in memory only; the data
    // is already persisted (ingest flushes) and `dirty` stays set so an explicit
    // flush() still writes it. No disk I/O on the read path.
    if (entry.pending.length > 0) this._materialize(entry);

    if (entry.ids.length === 0 || !entry.bin) return [];

    const rotated = this.wasmStore.rotateQuery(query);
    const packedData = new Uint8Array(entry.bin);

    const fetchN = filter ? entry.ids.length : limit;
    const { indices, scores } = this.wasmStore.search(rotated, packedData, fetchN);

    const results = [];
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const meta = entry.meta[idx];
      if (filter && !matchFilter(meta, filter)) continue;
      results.push({ id: entry.ids[idx], score: scores[i], metadata: meta });
      if (results.length >= limit) break;
    }

    return results;
  }

  // Remove a record by id (swap-with-last on the packed buffer). No disk write;
  // caller flushes. Returns true if the id existed.
  remove(col, id) {
    const entry = this._load(col);
    const idx = entry.idMap.get(id);
    if (idx === undefined) return false;
    this._materialize(entry);

    const bpv = this.bytesPerVec;
    const lastIdx = entry.ids.length - 1;
    if (entry.bin) {
      const u8 = new Uint8Array(entry.bin);
      if (idx !== lastIdx) u8.copyWithin(idx * bpv, lastIdx * bpv, (lastIdx + 1) * bpv);
      entry.bin = entry.bin.slice(0, lastIdx * bpv);
    }
    if (idx !== lastIdx) {
      const lastId = entry.ids[lastIdx];
      entry.ids[idx] = lastId;
      entry.meta[idx] = entry.meta[lastIdx];
      entry.idMap.set(lastId, idx);
    }
    entry.ids.pop();
    entry.meta.pop();
    entry.idMap.delete(id);
    entry.dirty = true;
    return true;
  }

  // List records (optionally filtered). Returns [{ id, metadata }].
  list(col, filter = null) {
    const entry = this._load(col);
    const out = [];
    for (let i = 0; i < entry.ids.length; i++) {
      if (filter && !matchFilter(entry.meta[i], filter)) continue;
      out.push({ id: entry.ids[i], metadata: entry.meta[i] });
    }
    return out;
  }
}

module.exports = WasmPolarQuantizedStore;
