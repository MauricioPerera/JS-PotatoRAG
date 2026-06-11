# Benchmark Results: WASM PolarStore vs. Pure JS PolarStore

Comparing insertion (quantization) and search speeds over **10,000 synthetic vectors** of **768 dimensions** on local hardware.

## Results Summary

| Metric | Pure JS Store | WASM Rust Store | Speedup |
| :--- | :--- | :--- | :--- |
| **Ingestion Time (10k vectors)** | 208.01 ms | 160.64 ms | **1.3x** |
| **Avg Ingestion per Vector** | 0.0208 ms | 0.0161 ms | **1.3x** |
| **Avg Search Latency (top-10)** | 105.6877 ms | 14.9884 ms | **7.1x** |

## Key Takeaways
1. **Quantization speedup**: Rust compiles to native target code, enabling math-heavy coordinate-to-polar calculations during quantization to execute **1.3x faster** than interpreted JS loops.
2. **Search scan speedup**: The index scan in WebAssembly runs at near-native speed, achieving a **7.1x speedup** over JavaScript.
