# Polar Quantization — how it works, what it costs, and when it doesn't

This documents the `PolarQuantizedStore` (JS, [`js-vector-store.js`](../js-vector-store.js))
and its Rust/WASM twin ([`rust_polar/src/lib.rs`](../rust_polar/src/lib.rs)),
which power the vector database. Numbers below were **measured** on this repo
(see [Reproducing](#reproducing)), not estimated.

## Attribution

This is **not an original algorithm**. The code labels it *"PolarQuant-inspired"*
and the server reports the format as *"Polar 3-bit (Google TurboQuant-inspired)"*.
The building blocks are established:

- **Random rotation before quantization** — standard preprocessing (e.g. Google
  TurboQuant, optimized product quantization).
- **Polar quantization** — quantizing 2D samples by angle, an old signal-processing idea.
- **Discarding magnitude, keeping direction for cosine** — the multi-bit
  generalization of sign-based binary quantization (see `BinaryQuantizedStore` in
  the same file).

What is local to this project is the *implementation*: a lightweight pure-JS +
Rust/WASM store, the 3-bit-per-pair packing, and the air-gapped packaging.

## The algorithm

Per vector, at `bits = 3` (8 angular levels), `dim = 768`:

1. **Normalize** to unit length.
2. **"Rotate"** with a deterministic, seeded transform.
3. For each **pair** of coordinates `(a, b)`, store only the **angle**
   `θ = atan2(b, a)`, quantized to `bits` levels in `[-π, π]`. The **magnitude**
   `√(a²+b²)` is discarded.
4. **Pack** the angle indices MSB-first → `ceil(dim/2 · bits / 8)` bytes
   (**144 B/vector** at 768-D → **~21.3×** vs float32).

Search reconstructs a unit vector **per pair** from the stored angles and takes a
dot product against the (rotated) query.

## Two structural properties worth knowing

### 1. The "rotation" is a signed permutation — it does **not** mix coordinates

`_rotate` computes `out[i] = vec[perm[i]] · signs[perm[i]]`. That is orthogonal
(norm-preserving, verified) but it only **shuffles coordinates and flips signs**.
It does **not** decorrelate or spread energy the way a true Hadamard/FWHT or random
rotation would. Older comments calling it *"Haar-like / distributes energy"* were
inaccurate and have been corrected.

### 2. Per-pair magnitude is discarded → every pair is weighted equally

Each stored pair is reconstructed on the unit circle `(cosθ, sinθ)`, so a
near-zero (noise) pair is weighted the same as a dominant one. Consequences:

- The reconstructed stored vector has norm `√(pairs) = √384 ≈ 19.6`, **not 1**.
- The similarity score is therefore **not bounded by [0,1]** (observed scores up
  to ~9). It is a monotonic ranking signal, not a calibrated cosine. The UI labels
  it "Similarity" — treat it as a rank score, not a probability.

## Measured quality (real `embeddinggemma`, 768-D, 3-bit)

| Metric | Value |
| :--- | :--- |
| recall@1 vs exact float32 (well-separated docs) | ~99–100% |
| recall@10 vs exact float32 | 100% |
| Reconstruction fidelity `cos(original, reconstructed)` | ~0.85 |
| Fidelity ceiling (random unit vectors) | ~0.94 |

**More bits barely helps.** Mean `cos(original, reconstructed)` on random 768-D
unit vectors: 2-bit 0.866 · 3-bit 0.911 · 5-bit 0.936 · 8-bit 0.9375. The ceiling
is set by the two structural properties above, **not** by angular granularity —
so `bits = 3` is a reasonable default and `bits > 4` mostly wastes space.

## When it works, and when it doesn't: energy concentration

The method's accuracy depends on how a vector's energy (`Σ xᵢ²`) is spread across
its dimensions, measured by the **Participation Ratio** `PR = (Σxᵢ²)² / Σxᵢ⁴`
(effective number of dimensions carrying energy; `PR ≈ dim` = uniform,
`PR ≈ k` = concentrated on k dims).

- **`embeddinggemma`: PR ≈ 199 / 768** (top-10 dims hold 14%, top-50 hold 37%).
  Energy is well spread → discarding per-pair magnitude hurts little → the method
  is near its practical optimum. **No useful headroom here.**
- **Concentrated embeddings (low PR):** raw/unnormalized LLM activations with
  outlier dimensions, sparse vectors (SPLADE/TF-IDF), etc. There most pairs are
  noise and accuracy collapses (synthetic PR≈11 vector measured: fidelity 0.54).

## Optimization notes (measured, not assumed)

| Lever | Effect on `embeddinggemma` | Notes |
| :--- | :--- | :--- |
| **FWHT instead of signed permutation** | fidelity +0.017, recall +0 | Needs power-of-2 dim → pad 768→1024 = **+33% memory** (21.3×→16×). Not worth it here. **Big win only for concentrated/low-PR embeddings** (synthetic: 0.54→0.88). |
| **Store 1–2 magnitude bits per pair** | recovers per-pair weighting | +~3 B/vec; only matters when energy is uneven. |
| **Exact rerank of top-K** | guarantees recall | Keep float32 sidecar; the only lever that reliably lifts recall@k>1, at a memory cost. Largely unnecessary given recall is already ~100%. |
| **Normalize score to [0,1]** | UX only | Current score reaches ~±19.6. |

**Bottom line for this stack (embeddinggemma + RAG): leave it as is.** It is close
to its practical optimum; optimization only pays off if you switch to an embedding
model whose vectors have concentrated energy (low PR), or if you measure
insufficient recall@k on your own corpus.

## Compatibility note

The JS and WASM quantizers are kept **byte-identical** (parity test in
[`test/store.test.js`](../test/store.test.js)). Any change to the rotation, the
PRNG, or the packing must preserve that parity and the committed
[`rust_polar.wasm`](../rust_polar.wasm) — rebuild it with
[`rust_polar/build.sh`](../rust_polar/build.sh) if you touch `lib.rs`. A collection
is only readable by a store with the same `dim/bits/seed` (enforced by a load guard).

## Reproducing

```bash
node benchmark_recall.js     # recall@k vs exact float32 (Ollama embeddings or synthetic)
node benchmark_wasm.cjs      # WASM vs pure-JS search/insert speed
npm test                     # parity, recall floor, round-trip, guards
```
