const fs = require('fs');
const path = require('path');

// Load WASM module synchronously
const wasmBuffer = fs.readFileSync(path.join(__dirname, 'rust_polar.wasm'));
const wasmModule = new WebAssembly.Module(wasmBuffer);
const wasmInstance = new WebAssembly.Instance(wasmModule, {});
const wasmExports = wasmInstance.exports;

class WasmPolarStore {
  constructor(dim, bits, seed = 42) {
    if (dim % 2 !== 0) throw new Error('WasmPolarStore: dim must be even');
    this.dim = dim;
    this.bits = bits;
    this.seed = seed;
    
    // Create the Rust PolarStore instance
    this.storePtr = wasmExports.create_store(dim, bits, seed);
    this.bytesPerVec = wasmExports.get_bytes_per_vector(this.storePtr);
  }

  free() {
    if (this.storePtr) {
      wasmExports.free_store(this.storePtr);
      this.storePtr = null;
    }
  }

  bytesPerVector() {
    return this.bytesPerVec;
  }

  // Quantizes a Float32 vector into a Uint8Array using WASM Rust engine
  quantize(vector) {
    const dim = this.dim;
    const bpv = this.bytesPerVec;
    
    // Allocate WASM memory for input vector
    const vecPtr = wasmExports.alloc(dim * 4);
    const vecView = new Float32Array(wasmExports.memory.buffer, vecPtr, dim);
    for (let i = 0; i < dim; i++) vecView[i] = vector[i] ?? 0;
    
    // Allocate WASM memory for output packed buffer
    const packedPtr = wasmExports.alloc(bpv);
    
    // Call Rust quantization
    wasmExports.quantize(this.storePtr, vecPtr, packedPtr);
    
    // Copy output buffer back to JS
    const result = new Uint8Array(wasmExports.memory.buffer, packedPtr, bpv).slice();
    
    // Clean up WASM allocations
    wasmExports.dealloc(vecPtr, dim * 4);
    wasmExports.dealloc(packedPtr, bpv);
    
    return result;
  }

  // Normalizes and rotates query vector in WASM
  rotateQuery(query) {
    const dim = this.dim;
    
    const queryPtr = wasmExports.alloc(dim * 4);
    const queryView = new Float32Array(wasmExports.memory.buffer, queryPtr, dim);
    for (let i = 0; i < dim; i++) queryView[i] = query[i] ?? 0;
    
    const rotatedPtr = wasmExports.alloc(dim * 4);
    
    wasmExports.rotate_query(this.storePtr, queryPtr, rotatedPtr);
    
    const result = new Float32Array(wasmExports.memory.buffer, rotatedPtr, dim).slice();
    
    wasmExports.dealloc(queryPtr, dim * 4);
    wasmExports.dealloc(rotatedPtr, dim * 4);
    
    return result;
  }

  // Performs similarity search in WASM
  search(queryRotated, packedData, limit) {
    const dim = this.dim;
    const bpv = this.bytesPerVec;
    const numVectors = packedData.length / bpv;
    const actualLimit = Math.min(limit, numVectors);
    
    if (actualLimit === 0) {
      return { indices: new Uint32Array(0), scores: new Float32Array(0) };
    }
    
    // Allocate and copy queryRotated
    const qRotPtr = wasmExports.alloc(dim * 4);
    new Float32Array(wasmExports.memory.buffer, qRotPtr, dim).set(queryRotated);
    
    // Allocate and copy packedData
    const dataSize = packedData.length;
    const dataPtr = wasmExports.alloc(dataSize);
    new Uint8Array(wasmExports.memory.buffer, dataPtr, dataSize).set(packedData);
    
    // Allocate output buffers
    const outIndicesPtr = wasmExports.alloc(actualLimit * 4);
    const outScoresPtr = wasmExports.alloc(actualLimit * 4);
    
    // Call Rust search
    wasmExports.cosine_polar_search(
      this.storePtr,
      qRotPtr,
      dataPtr,
      numVectors,
      actualLimit,
      outIndicesPtr,
      outScoresPtr
    );
    
    // Copy results
    const indices = new Uint32Array(wasmExports.memory.buffer, outIndicesPtr, actualLimit).slice();
    const scores = new Float32Array(wasmExports.memory.buffer, outScoresPtr, actualLimit).slice();
    
    // Free allocations
    wasmExports.dealloc(qRotPtr, dim * 4);
    wasmExports.dealloc(dataPtr, dataSize);
    wasmExports.dealloc(outIndicesPtr, actualLimit * 4);
    wasmExports.dealloc(outScoresPtr, actualLimit * 4);
    
    return { indices, scores };
  }
}

module.exports = WasmPolarStore;
