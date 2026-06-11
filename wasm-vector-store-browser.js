class BrowserIndexedDbAdapter {
  constructor(dbName = 'PotatoRAG_DB', storeName = 'files') {
    this.dbName = dbName;
    this.storeName = storeName;
    this.cache = new Map();
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onerror = () => reject(request.error);
      request.onupgradeneeded = () => {
        const db = request.result;
        db.createObjectStore(this.storeName);
      };
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction(this.storeName, 'readonly');
        const store = tx.objectStore(this.storeName);
        const cursorReq = store.openCursor();
        cursorReq.onsuccess = (e) => {
          const cursor = e.target.result;
          if (cursor) {
            this.cache.set(cursor.key, cursor.value);
            cursor.continue();
          } else {
            resolve();
          }
        };
        cursorReq.onerror = () => reject(cursorReq.error);
      };
    });
  }

  readBin(filename) {
    const val = this.cache.get(filename);
    return val || null;
  }

  writeBin(filename, buffer) {
    this.cache.set(filename, buffer);
    this._persist(filename, buffer);
  }

  readJson(filename) {
    const val = this.cache.get(filename);
    return val || null;
  }

  writeJson(filename, data) {
    this.cache.set(filename, data);
    this._persist(filename, data);
  }

  delete(filename) {
    this.cache.delete(filename);
    this._deletePersisted(filename);
  }

  _persist(key, val) {
    const request = indexedDB.open(this.dbName, 1);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      store.put(val, key);
    };
  }

  _deletePersisted(key) {
    const request = indexedDB.open(this.dbName, 1);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      store.delete(key);
    };
  }
}

class BrowserWasmVectorStore {
  constructor(adapter, dim = 768, opts = {}) {
    if (dim % 2 !== 0) throw new Error('BrowserWasmVectorStore: dim must be even');
    this.dim = dim;
    this.bits = opts.bits || 3;
    this.seed = opts.seed ?? 42;
    this.defaultModel = opts.model || null;
    
    this._adapter = adapter;
    this._collections = new Map();
    this.wasmStore = new WasmPolarStore(dim, this.bits, this.seed);
    this.bytesPerVec = this.wasmStore.bytesPerVector();
  }

  _binFile(col)  { return `${col}.p3.bin`; }
  _jsonFile(col) { return `${col}.p3.json`; }

  _load(col) {
    if (this._collections.has(col)) return this._collections.get(col);
    const manifest = this._adapter.readJson(this._jsonFile(col));
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

  _flushCol(col, entry) {
    if (entry.pending.length > 0) {
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

  search(col, query, limit = 5) {
    const entry = this._load(col);
    if (entry.pending.length > 0) this._flushCol(col, entry);
    
    if (entry.ids.length === 0 || !entry.bin) return [];
    
    const rotated = this.wasmStore.rotateQuery(query);
    const packedData = new Uint8Array(entry.bin);
    
    const { indices, scores } = this.wasmStore.search(rotated, packedData, limit);
    
    const results = [];
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      results.push({
        id: entry.ids[idx],
        score: scores[i],
        metadata: entry.meta[idx]
      });
    }
    
    return results;
  }
}
