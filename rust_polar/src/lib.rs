use std::mem;
use std::os::raw::c_void;

fn xorshift(mut state: u32) -> u32 {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    state
}

struct Rotation {
    signs: Vec<f32>,
    perm: Vec<usize>,
}

fn generate_rotation(dim: usize, seed: u32) -> Rotation {
    let mut signs = vec![0.0f32; dim];
    let mut state = if seed == 0 { 42 } else { seed };
    for i in 0..dim {
        state = xorshift(state);
        signs[i] = if (state & 1) != 0 { 1.0 } else { -1.0 };
    }
    let mut perm = (0..dim).collect::<Vec<usize>>();
    state = seed * 7 + 13;
    for i in (1..dim).rev() {
        state = xorshift(state);
        let j = (state as usize) % (i + 1);
        perm.swap(i, j);
    }
    Rotation { signs, perm }
}

pub struct PolarStore {
    dim: usize,
    bits: usize,
    levels: usize,
    pairs: usize,
    pub bytes_per_vec: usize,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    rotation: Rotation,
}

impl PolarStore {
    pub fn new(dim: usize, bits: usize, seed: u32) -> Self {
        let levels = 1 << bits;
        let pairs = dim / 2;
        let bytes_per_vec = ((pairs * bits) as f32 / 8.0).ceil() as usize;

        let mut cos_table = vec![0.0f32; levels];
        let mut sin_table = vec![0.0f32; levels];
        for i in 0..levels {
            let theta = -std::f32::consts::PI + (i as f32 + 0.5) * (2.0 * std::f32::consts::PI / levels as f32);
            cos_table[i] = theta.cos();
            sin_table[i] = theta.sin();
        }

        let rotation = generate_rotation(dim, seed);

        PolarStore {
            dim,
            bits,
            levels,
            pairs,
            bytes_per_vec,
            cos_table,
            sin_table,
            rotation,
        }
    }

    fn rotate(&self, vec: &[f32], out: &mut [f32]) {
        for i in 0..self.dim {
            let p_idx = self.rotation.perm[i];
            out[i] = vec[p_idx] * self.rotation.signs[p_idx];
        }
    }

    pub fn quantize(&self, vector: &[f32], out_packed: &mut [u8]) {
        // 1. Normalize
        let mut norm_sq = 0.0f32;
        for &x in vector {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let mut unit = vec![0.0f32; self.dim];
        if norm > 0.0 {
            for i in 0..self.dim {
                unit[i] = vector[i] / norm;
            }
        }

        // 2. Rotate
        let mut rotated = vec![0.0f32; self.dim];
        self.rotate(&unit, &mut rotated);

        // 3. Coordinate pairs to polar angles (levels)
        let mut indices = vec![0u8; self.pairs];
        for p in 0..self.pairs {
            let a = rotated[p * 2];
            let b = rotated[p * 2 + 1];
            let theta = b.atan2(a); // [-PI, PI]
            let mut level = (((theta + std::f32::consts::PI) / (2.0 * std::f32::consts::PI)) * self.levels as f32).floor() as usize;
            if level >= self.levels {
                level = self.levels - 1;
            }
            indices[p] = level as u8;
        }

        // 4. Pack bits
        out_packed.fill(0);
        let mut bit_pos = 0;
        for p in 0..self.pairs {
            let val = indices[p];
            for b in (0..self.bits).rev() {
                if (val & (1 << b)) != 0 {
                    out_packed[bit_pos >> 3] |= 1 << (7 - (bit_pos & 7));
                }
                bit_pos += 1;
            }
        }
    }

    fn unpack_bits(&self, packed: &[u8], offset: usize, out_indices: &mut [u8]) {
        let mut bit_pos = 0;
        for p in 0..self.pairs {
            let mut val = 0u8;
            for b in (0..self.bits).rev() {
                let byte_idx = offset + (bit_pos >> 3);
                let bit_idx = 7 - (bit_pos & 7);
                if (packed[byte_idx] & (1 << bit_idx)) != 0 {
                    val |= 1 << b;
                }
                bit_pos += 1;
            }
            out_indices[p] = val;
        }
    }

    pub fn cosine_polar(&self, query_rotated: &[f32], packed: &[u8], offset: usize) -> f32 {
        let mut indices = vec![0u8; self.pairs];
        self.unpack_bits(packed, offset, &mut indices);

        let mut dot = 0.0f32;
        let mut nq = 0.0f32;
        for p in 0..self.pairs {
            let qa = query_rotated[p * 2];
            let qb = query_rotated[p * 2 + 1];
            let idx = indices[p] as usize;
            dot += qa * self.cos_table[idx] + qb * self.sin_table[idx];
            nq += qa * qa + qb * qb;
        }
        let denom_q = nq.sqrt();
        if denom_q == 0.0 {
            0.0
        } else {
            dot / denom_q
        }
    }
}

// ===========================================================================
// WebAssembly FFI Exports
// ===========================================================================

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut c_void {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    mem::forget(buf);
    ptr as *mut c_void
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut c_void, size: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr as *mut u8, 0, size);
    }
}

#[no_mangle]
pub extern "C" fn create_store(dim: usize, bits: usize, seed: u32) -> *mut PolarStore {
    let store = Box::new(PolarStore::new(dim, bits, seed));
    Box::into_raw(store)
}

#[no_mangle]
pub extern "C" fn free_store(store_ptr: *mut PolarStore) {
    if !store_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(store_ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn get_bytes_per_vector(store_ptr: *mut PolarStore) -> usize {
    let store = unsafe { &*store_ptr };
    store.bytes_per_vec
}

#[no_mangle]
pub extern "C" fn quantize(store_ptr: *mut PolarStore, vector_ptr: *const f32, out_packed_ptr: *mut u8) {
    let store = unsafe { &*store_ptr };
    let vector = unsafe { std::slice::from_raw_parts(vector_ptr, store.dim) };
    let out_packed = unsafe { std::slice::from_raw_parts_mut(out_packed_ptr, store.bytes_per_vec) };
    store.quantize(vector, out_packed);
}

#[no_mangle]
pub extern "C" fn rotate_query(store_ptr: *mut PolarStore, query_ptr: *const f32, out_rotated_ptr: *mut f32) {
    let store = unsafe { &*store_ptr };
    let query = unsafe { std::slice::from_raw_parts(query_ptr, store.dim) };
    // Normalize query
    let mut norm_sq = 0.0f32;
    for &x in query {
        norm_sq += x * x;
    }
    let norm = norm_sq.sqrt();
    let mut unit = vec![0.0f32; store.dim];
    if norm > 0.0 {
        for i in 0..store.dim {
            unit[i] = query[i] / norm;
        }
    }
    let out_rotated = unsafe { std::slice::from_raw_parts_mut(out_rotated_ptr, store.dim) };
    store.rotate(&unit, out_rotated);
}

#[no_mangle]
pub extern "C" fn cosine_polar_search(
    store_ptr: *mut PolarStore,
    query_rotated_ptr: *const f32,
    packed_data_ptr: *const u8,
    num_vectors: usize,
    k: usize,
    out_indices_ptr: *mut u32,
    out_scores_ptr: *mut f32,
) {
    let store = unsafe { &*store_ptr };
    let query_rotated = unsafe { std::slice::from_raw_parts(query_rotated_ptr, store.dim) };
    let bpv = store.bytes_per_vec;
    let packed_data = unsafe { std::slice::from_raw_parts(packed_data_ptr, num_vectors * bpv) };

    let mut candidates = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        let score = store.cosine_polar(query_rotated, packed_data, i * bpv);
        candidates.push((i as u32, score));
    }

    // Sort candidates descending by score
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let limit = std::cmp::min(k, num_vectors);
    let out_indices = unsafe { std::slice::from_raw_parts_mut(out_indices_ptr, limit) };
    let out_scores = unsafe { std::slice::from_raw_parts_mut(out_scores_ptr, limit) };

    for i in 0..limit {
        out_indices[i] = candidates[i].0;
        out_scores[i] = candidates[i].1;
    }
}
