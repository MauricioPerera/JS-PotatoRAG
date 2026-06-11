class WasmPolarStore {
  static async init(wasmUrl = 'rust_polar.wasm') {
    if (WasmPolarStore.exports) return;
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`Failed to fetch WASM from ${wasmUrl}`);
    const wasmBuffer = await response.arrayBuffer();
    const wasmModule = await WebAssembly.compile(wasmBuffer);
    const wasmInstance = await WebAssembly.instantiate(wasmModule, {});
    WasmPolarStore.exports = wasmInstance.exports;
  }

  constructor(dim, bits, seed = 42) {
    const exports = WasmPolarStore.exports;
    if (!exports) {
      throw new Error('WasmPolarStore is not initialized. Run await WasmPolarStore.init() first.');
    }
    if (dim % 2 !== 0) throw new Error('WasmPolarStore: dim must be even');
    this.dim = dim;
    this.bits = bits;
    this.seed = seed;
    
    // Create the Rust PolarStore instance
    this.storePtr = exports.create_store(dim, bits, seed);
    this.bytesPerVec = exports.get_bytes_per_vector(this.storePtr);
  }

  free() {
    const exports = WasmPolarStore.exports;
    if (this.storePtr && exports) {
      exports.free_store(this.storePtr);
      this.storePtr = null;
    }
  }

  bytesPerVector() {
    return this.bytesPerVec;
  }

  quantize(vector) {
    const exports = WasmPolarStore.exports;
    const dim = this.dim;
    const bpv = this.bytesPerVec;
    
    const vecPtr = exports.alloc(dim * 4);
    const vecView = new Float32Array(exports.memory.buffer, vecPtr, dim);
    for (let i = 0; i < dim; i++) vecView[i] = vector[i] ?? 0;
    
    const packedPtr = exports.alloc(bpv);
    exports.quantize(this.storePtr, vecPtr, packedPtr);
    
    const result = new Uint8Array(exports.memory.buffer, packedPtr, bpv).slice();
    
    exports.dealloc(vecPtr, dim * 4);
    exports.dealloc(packedPtr, bpv);
    
    return result;
  }

  rotateQuery(query) {
    const exports = WasmPolarStore.exports;
    const dim = this.dim;
    
    const queryPtr = exports.alloc(dim * 4);
    const queryView = new Float32Array(exports.memory.buffer, queryPtr, dim);
    for (let i = 0; i < dim; i++) queryView[i] = query[i] ?? 0;
    
    const rotatedPtr = exports.alloc(dim * 4);
    exports.rotate_query(this.storePtr, queryPtr, rotatedPtr);
    
    const result = new Float32Array(exports.memory.buffer, rotatedPtr, dim).slice();
    
    exports.dealloc(queryPtr, dim * 4);
    exports.dealloc(rotatedPtr, dim * 4);
    
    return result;
  }

  search(queryRotated, packedData, limit) {
    const exports = WasmPolarStore.exports;
    const dim = this.dim;
    const bpv = this.bytesPerVec;
    const numVectors = packedData.length / bpv;
    const actualLimit = Math.min(limit, numVectors);
    
    if (actualLimit === 0) {
      return { indices: new Uint32Array(0), scores: new Float32Array(0) };
    }
    
    const qRotPtr = exports.alloc(dim * 4);
    new Float32Array(exports.memory.buffer, qRotPtr, dim).set(queryRotated);
    
    const dataSize = packedData.length;
    const dataPtr = exports.alloc(dataSize);
    new Uint8Array(exports.memory.buffer, dataPtr, dataSize).set(packedData);
    
    const outIndicesPtr = exports.alloc(actualLimit * 4);
    const outScoresPtr = exports.alloc(actualLimit * 4);
    
    exports.cosine_polar_search(
      this.storePtr,
      qRotPtr,
      dataPtr,
      numVectors,
      actualLimit,
      outIndicesPtr,
      outScoresPtr
    );
    
    const indices = new Uint32Array(exports.memory.buffer, outIndicesPtr, actualLimit).slice();
    const scores = new Float32Array(exports.memory.buffer, outScoresPtr, actualLimit).slice();
    
    exports.dealloc(qRotPtr, dim * 4);
    exports.dealloc(dataPtr, dataSize);
    exports.dealloc(outIndicesPtr, actualLimit * 4);
    exports.dealloc(outScoresPtr, actualLimit * 4);
    
    return { indices, scores };
  }
}

WasmPolarStore.exports = null;
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WasmPolarStore;
}
