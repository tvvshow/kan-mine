// Pearl (PRL) PlainProof generator - faithful C++ port of zk-pow Rust reference.
// CPU-only. Generates A,B, runs commitments/noise/jackpot/merkle pipeline using
// the golden config+header, and on a winning tile prints base64(bincode(PlainProof)).
//
// Build (separate from CUDA Makefile):
//   g++ -O2 -std=c++17 -Iblake3 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 \
//       -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 \
//       plainproof_gen.cpp blake3/blake3.c blake3/blake3_portable.c \
//       blake3/blake3_dispatch.c -o plainproof_gen
//
// Usage: ./plainproof_gen [seed] [--dump]
//   stdout : base64 PlainProof
//   stderr : HEADER_HEX + chosen tile + (with --dump) writes A/B/tile debug files

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#include "blake3.h"
#include "blake3_impl.h"
}
// gpu_draw.h deliberately NOT included: gpu_produce_draw is unused here until
// Week4 GPU-RNG lands, and pulling <cuda_runtime.h> into this TU breaks the
// host-only build (CUDA-13 headers are C++-template-heavy).

#include "prover.h"

using std::vector;
using std::array;
using Digest = array<uint8_t, 32>;

// ---------------------------------------------------------------------------
// BLAKE3 helpers
// ---------------------------------------------------------------------------

// Full-data keyed/unkeyed one-shot hash (== blake3_digest in Rust).
static Digest blake3_digest(const uint8_t* data, size_t len, const Digest* key) {
    blake3_hasher h;
    if (key) blake3_hasher_init_keyed(&h, key->data());
    else     blake3_hasher_init(&h);
    blake3_hasher_update(&h, data, len);
    Digest out;
    blake3_hasher_finalize(&h, out.data(), 32);
    return out;
}
static Digest blake3_digest(const vector<uint8_t>& d, const Digest* key) {
    return blake3_digest(d.data(), d.size(), key);
}

static void key_to_words(const Digest& key, uint32_t w[8]) {
    for (int i = 0; i < 8; i++)
        w[i] = (uint32_t)key[i*4] | ((uint32_t)key[i*4+1] << 8) |
               ((uint32_t)key[i*4+2] << 16) | ((uint32_t)key[i*4+3] << 24);
}

static const uint32_t IV_WORDS[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

static void cv_to_bytes(const uint32_t cv[8], Digest& out) {
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(cv[i]);
        out[i*4+1] = (uint8_t)(cv[i] >> 8);
        out[i*4+2] = (uint8_t)(cv[i] >> 16);
        out[i*4+3] = (uint8_t)(cv[i] >> 24);
    }
}
static void bytes_to_cv(const Digest& in, uint32_t cv[8]) {
    for (int i = 0; i < 8; i++)
        cv[i] = (uint32_t)in[i*4] | ((uint32_t)in[i*4+1] << 8) |
                ((uint32_t)in[i*4+2] << 16) | ((uint32_t)in[i*4+3] << 24);
}

// Keyed BLAKE3 chunk CV (non-root) for a single chunk (<=1024 bytes) at chunk
// counter = chunk_index. Mirrors Blake3Hasher::chunk_cv (KEYED_HASH base flag).
static Digest chunk_cv(const Digest& key, const uint8_t* data, size_t len, uint64_t chunk_index) {
    uint32_t cv[8];
    key_to_words(key, cv);
    const uint8_t base_flags = KEYED_HASH;
    size_t pos = 0;
    bool first = true;
    // Process full 64-byte blocks except keep the final block for finalize.
    while (len - pos > BLAKE3_BLOCK_LEN) {
        uint8_t flags = base_flags;
        if (first) { flags |= CHUNK_START; first = false; }
        blake3_compress_in_place(cv, data + pos, BLAKE3_BLOCK_LEN, chunk_index, flags);
        pos += BLAKE3_BLOCK_LEN;
    }
    // Final block (may be partial; for our 1024-byte chunks it's a full 64 bytes).
    uint8_t lastblock[BLAKE3_BLOCK_LEN] = {0};
    size_t lastlen = len - pos;
    memcpy(lastblock, data + pos, lastlen);
    uint8_t flags = base_flags | CHUNK_END;
    if (first) flags |= CHUNK_START;
    blake3_compress_in_place(cv, lastblock, (uint8_t)lastlen, chunk_index, flags);
    Digest out;
    cv_to_bytes(cv, out);
    return out;
}

// parent merge (non-root): compress(key_words, left||right, 64, 0, KEYED|PARENT)
static Digest parent_cv(const Digest& key, const Digest& left, const Digest& right) {
    uint32_t cv[8];
    key_to_words(key, cv);
    uint8_t block[BLAKE3_BLOCK_LEN];
    memcpy(block, left.data(), 32);
    memcpy(block + 32, right.data(), 32);
    blake3_compress_in_place(cv, block, BLAKE3_BLOCK_LEN, 0, KEYED_HASH | PARENT);
    Digest out;
    cv_to_bytes(cv, out);
    return out;
}

// root merge: compress_xof(key_words, left||right, 64, 0, KEYED|PARENT|ROOT)[..32]
static Digest root_cv(const Digest& key, const Digest& left, const Digest& right) {
    uint32_t cv[8];
    key_to_words(key, cv);
    uint8_t block[BLAKE3_BLOCK_LEN];
    memcpy(block, left.data(), 32);
    memcpy(block + 32, right.data(), 32);
    uint8_t xof[64];
    blake3_compress_xof(cv, block, BLAKE3_BLOCK_LEN, 0, KEYED_HASH | PARENT | ROOT, xof);
    Digest out;
    memcpy(out.data(), xof, 32);
    return out;
}

// ---------------------------------------------------------------------------
// Merkle tree (faithful port of pearl-blake3 MerkleTree)
// ---------------------------------------------------------------------------
static const size_t CHUNK_LEN = 1024;

struct MerkleTree {
    Digest key;
    vector<vector<Digest>> layers;
    const uint8_t* data;
    size_t data_len;

    MerkleTree(const uint8_t* d, size_t dl, const Digest& k) : key(k), data(d), data_len(dl) {
        if (dl == 0) { layers.push_back({}); return; }
        if (dl <= CHUNK_LEN) {
            layers.push_back({ blake3_digest(d, dl, &k) });
            return;
        }
        size_t num_chunks = (dl + CHUNK_LEN - 1) / CHUNK_LEN;
        vector<Digest> chunk_cvs(num_chunks);
        for (size_t i = 0; i < num_chunks; i++) {
            size_t start = i * CHUNK_LEN;
            size_t len = std::min(CHUNK_LEN, dl - start);
            chunk_cvs[i] = chunk_cv(k, d + start, len, (uint64_t)i);
        }
        layers.push_back(std::move(chunk_cvs));
        while (layers.back().size() > 2) {
            const vector<Digest>& prev = layers.back();
            vector<Digest> next;
            next.reserve((prev.size() + 1) / 2);
            for (size_t i = 0; i + 1 < prev.size(); i += 2)
                next.push_back(parent_cv(k, prev[i], prev[i+1]));
            if (prev.size() % 2 == 1) next.push_back(prev.back());
            layers.push_back(std::move(next));
        }
        const vector<Digest>& last = layers.back();
        if (last.size() == 2) {
            Digest root = root_cv(k, last[0], last[1]);
            layers.push_back({ root });
        }
    }

    size_t num_leaves() const { return layers[0].size(); }
    Digest root() const { return layers.back()[0]; }
};

struct MerkleProof {
    vector<array<uint8_t, CHUNK_LEN>> leaf_data;
    vector<size_t> leaf_indices;
    size_t total_leaves;
    Digest root;
    vector<Digest> siblings;
};

static MerkleProof get_multileaf_proof(const MerkleTree& tree, const vector<size_t>& idxs) {
    std::set<size_t> unique(idxs.begin(), idxs.end());
    size_t total_leaves = tree.num_leaves();
    vector<size_t> sorted(unique.begin(), unique.end());

    MerkleProof p;
    p.total_leaves = total_leaves;
    p.root = tree.root();
    p.leaf_indices = sorted;
    for (size_t i : sorted) {
        array<uint8_t, CHUNK_LEN> chunk{};
        size_t start = i * CHUNK_LEN;
        size_t end = std::min(start + CHUNK_LEN, tree.data_len);
        memcpy(chunk.data(), tree.data + start, end - start);
        p.leaf_data.push_back(chunk);
    }

    std::set<size_t> current_set = unique;
    size_t level_len = total_leaves;
    size_t level = 0;
    while (level_len > 1 && !current_set.empty()) {
        const vector<Digest>& level_nodes = tree.layers[level];
        // ascending iteration over current_set (std::set is ordered)
        for (size_t i : current_set) {
            if (i % 2 == 1) {
                if (current_set.find(i - 1) == current_set.end())
                    p.siblings.push_back(level_nodes[i - 1]);
            } else {
                if (current_set.find(i + 1) == current_set.end() && (i + 1) < level_len)
                    p.siblings.push_back(level_nodes[i + 1]);
            }
        }
        std::set<size_t> next;
        for (size_t i : current_set) next.insert(i / 2);
        current_set = std::move(next);
        level_len = (level_len + 1) / 2;
        level++;
    }
    return p;
}

static vector<size_t> compute_leaf_indices_from_rows(const vector<size_t>& rows, size_t /*num_rows*/, size_t cols) {
    std::set<size_t> idx;
    for (size_t row : rows) {
        size_t first = (row * cols) / CHUNK_LEN;
        size_t last = ((row + 1) * cols - 1) / CHUNK_LEN;
        for (size_t i = first; i <= last; i++) idx.insert(i);
    }
    return vector<size_t>(idx.begin(), idx.end());
}

// ---------------------------------------------------------------------------
// PeriodicPattern (only what we need: from_list, period, size, offset_is_valid,
//  to_list, to_bytes)
// ---------------------------------------------------------------------------
struct PeriodicPattern {
    array<std::pair<uint32_t,uint32_t>, 3> shape; // (stride, length)

    static PeriodicPattern from_list(const vector<uint32_t>& pattern) {
        // assume valid input (golden config)
        vector<uint32_t> p = pattern;
        vector<std::pair<uint32_t,uint32_t>> shape_vec;
        while (p.size() > 1) {
            bool found = false;
            for (size_t period = 1; period < p.size(); period++) {
                if (p.size() % period == 0) {
                    uint32_t s = p[period];
                    bool is_periodic = true;
                    for (size_t i = 0; i + period < p.size(); i++)
                        if (p[i] + s != p[i + period]) { is_periodic = false; break; }
                    if (is_periodic) {
                        shape_vec.push_back({s, (uint32_t)(p.size() / period)});
                        p.resize(period);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) { fprintf(stderr, "pattern not periodic\n"); exit(1); }
        }
        std::reverse(shape_vec.begin(), shape_vec.end());
        uint32_t period = shape_vec.empty() ? 1 : shape_vec.back().first * shape_vec.back().second;
        while (shape_vec.size() < 3) shape_vec.push_back({period, 1});
        PeriodicPattern pp;
        for (int i = 0; i < 3; i++) pp.shape[i] = shape_vec[i];
        return pp;
    }

    uint32_t period() const {
        auto& last = shape[2];
        return last.first * last.second;
    }
    uint32_t size() const {
        uint32_t s = 1;
        for (auto& d : shape) s *= d.second;
        return s;
    }
    uint32_t max() const {
        auto l = to_list();
        return *std::max_element(l.begin(), l.end());
    }
    bool offset_is_valid(uint32_t offset) const {
        for (int i = 2; i >= 0; i--) {
            uint32_t stride = shape[i].first, length = shape[i].second;
            offset %= stride * length;
            if (offset >= stride) return false;
        }
        return true;
    }
    vector<uint32_t> to_list() const {
        vector<uint32_t> res = {0};
        for (auto& d : shape) {
            uint32_t stride = d.first, length = d.second;
            vector<uint32_t> nr;
            nr.reserve(res.size() * length);
            for (uint32_t i = 0; i < length; i++)
                for (uint32_t r : res) nr.push_back(r + i * stride);
            res = std::move(nr);
        }
        return res;
    }
    array<uint8_t,6> to_bytes() const {
        array<uint8_t,6> data{};
        uint32_t min_stride = 1;
        for (int i = 0; i < 3; i++) {
            uint32_t stride = shape[i].first, length = shape[i].second;
            uint32_t factor = stride / min_stride;
            data[2*i] = (uint8_t)(factor - 1);
            data[2*i+1] = (uint8_t)(length - 1);
            min_stride = stride * length;
        }
        return data;
    }
};

// threads_partition: for each valid offset, base_indices + offset
static vector<vector<size_t>> threads_partition(const PeriodicPattern& pat, size_t total) {
    vector<uint32_t> base = pat.to_list();
    vector<vector<size_t>> out;
    for (size_t i = 0; i < total; i++) {
        if (pat.offset_is_valid((uint32_t)i)) {
            vector<size_t> t;
            for (uint32_t d : base) t.push_back(i + d);
            out.push_back(std::move(t));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Config / header serialization
// ---------------------------------------------------------------------------
struct Header {
    uint32_t version = 0;
    array<uint8_t,32> prev_block;
    array<uint8_t,32> merkle_root;
    uint32_t timestamp = 0x66666666;
    uint32_t nbits = 0x1D2FFFFF;
    Header() { prev_block.fill(1); merkle_root.fill(2); }
    vector<uint8_t> to_bytes() const {
        vector<uint8_t> b;
        for (int i=0;i<4;i++) b.push_back((version >> (8*i)) & 0xFF);
        for (int i=31;i>=0;i--) b.push_back(prev_block[i]);   // reversed
        for (int i=31;i>=0;i--) b.push_back(merkle_root[i]);  // reversed
        for (int i=0;i<4;i++) b.push_back((timestamp >> (8*i)) & 0xFF);
        for (int i=0;i<4;i++) b.push_back((nbits >> (8*i)) & 0xFF);
        return b;
    }
};

struct Config {
    uint32_t common_dim;
    uint16_t rank;
    uint16_t mma_type = 0;
    PeriodicPattern rows_pattern, cols_pattern;
    vector<uint8_t> to_bytes() const {
        vector<uint8_t> b;
        for (int i=0;i<4;i++) b.push_back((common_dim >> (8*i)) & 0xFF);
        for (int i=0;i<2;i++) b.push_back((rank >> (8*i)) & 0xFF);
        for (int i=0;i<2;i++) b.push_back((mma_type >> (8*i)) & 0xFF);
        auto rp = rows_pattern.to_bytes();
        b.insert(b.end(), rp.begin(), rp.end());
        auto cp = cols_pattern.to_bytes();
        b.insert(b.end(), cp.begin(), cp.end());
        for (int i=0;i<32;i++) b.push_back(0); // reserved
        return b;
    }
};

// ---------------------------------------------------------------------------
// Noise (faithful port of pearl_noise.rs)
// ---------------------------------------------------------------------------
static const size_t BLAKE3_DIGEST_SIZE = 32;

static Digest get_random_hash(size_t index, const Digest& seed, const Digest& key, size_t prepend_index) {
    uint8_t message[64] = {0};
    int32_t prepend_value = (int32_t)(1 + index);
    // little-endian i32 at prepend_index*4
    message[prepend_index*4 + 0] = (uint8_t)(prepend_value);
    message[prepend_index*4 + 1] = (uint8_t)(prepend_value >> 8);
    message[prepend_index*4 + 2] = (uint8_t)(prepend_value >> 16);
    message[prepend_index*4 + 3] = (uint8_t)(prepend_value >> 24);
    memcpy(message + 32, seed.data(), 32);
    return blake3_digest(message, 64, &key);
}

// NOISE_RANGE=128, IDXS_PER_COL=2 -> UNIFORM_NOISE_RANGE=64,
// ZERO_POINT_TRANSLATION=32, RANGE_MASK=63
static const uint8_t RANGE_MASK = 63;
static const int8_t ZERO_POINT = 32;

// generate one uniform random row (length num_cols) for a given row_idx
static vector<int8_t> uniform_row(const Digest& seed, const Digest& key, size_t row_idx, size_t num_cols) {
    size_t start_idx = row_idx * num_cols;
    size_t block_first = start_idx / BLAKE3_DIGEST_SIZE;
    size_t block_last = (start_idx + num_cols + BLAKE3_DIGEST_SIZE - 1) / BLAKE3_DIGEST_SIZE; // div_ceil, exclusive
    vector<int8_t> row;
    row.reserve(num_cols);
    for (size_t block = block_first; block < block_last; block++) {
        Digest h = get_random_hash(block, seed, key, 0);
        for (size_t kk = 0; kk < 32; kk++) {
            size_t idx = block * BLAKE3_DIGEST_SIZE + kk;
            if (idx >= start_idx && idx < start_idx + num_cols)
                row.push_back((int8_t)((h[kk] & RANGE_MASK)) - ZERO_POINT);
        }
    }
    return row;
}

static uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

// permutation matrix: k pairs [first, second]
static vector<array<uint32_t,2>> generate_permutation_matrix(const Digest& seed, const Digest& key, size_t k, size_t noise_rank) {
    const size_t LINES_PER_HASH = BLAKE3_DIGEST_SIZE / 4; // 8
    uint32_t rank_mask = (uint32_t)(noise_rank - 1);
    vector<array<uint32_t,2>> res(k);
    size_t num_hashes = (k + LINES_PER_HASH - 1) / LINES_PER_HASH;
    for (size_t i = 0; i < num_hashes; i++) {
        Digest h = get_random_hash(i, seed, key, 1);
        size_t base = i * LINES_PER_HASH;
        for (size_t j = 0; j < LINES_PER_HASH && base + j < k; j++) {
            uint32_t ru = (uint32_t)h[j*4] | ((uint32_t)h[j*4+1] << 8) |
                          ((uint32_t)h[j*4+2] << 16) | ((uint32_t)h[j*4+3] << 24);
            uint32_t first_idx = ru & rank_mask;
            uint32_t second_idx = first_idx ^ (1 + mul_hi_u32(rank_mask, ru));
            res[base + j] = {first_idx, second_idx};
        }
    }
    return res;
}

// matvec_sparse_perm: result[i] = vec[perm[i][0]] - vec[perm[i][1]]
static vector<int8_t> matvec_sparse_perm(const vector<array<uint32_t,2>>& perm, const vector<int8_t>& vec) {
    vector<int8_t> out(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        int32_t pos = vec[perm[i][0]];
        int32_t neg = vec[perm[i][1]];
        out[i] = (int8_t)(pos - neg);
    }
    return out;
}

static const uint8_t SEED_LABEL_A[32] = {'A','_','t','e','n','s','o','r',0};
static const uint8_t SEED_LABEL_B[32] = {'B','_','t','e','n','s','o','r',0};

// ---------------------------------------------------------------------------
// U256 little-endian compare (jackpot_hash <= bound)
// bound = nbits_to_difficulty(nbits) * (h*w*dot_len). We compute bound as
// 256-bit (in 32 bytes little-endian) using simple big integer arithmetic.
// ---------------------------------------------------------------------------
struct U256 {
    uint8_t b[32]; // little-endian
    U256() { memset(b, 0, 32); }
    static U256 from_le(const uint8_t* d) { U256 u; memcpy(u.b, d, 32); return u; }
    bool le(const U256& o) const { // this <= o
        for (int i = 31; i >= 0; i--) {
            if (b[i] != o.b[i]) return b[i] < o.b[i];
        }
        return true;
    }
};

// nbits_to_difficulty -> 32-byte little-endian
static U256 nbits_to_difficulty(uint32_t nbits) {
    U256 r;
    uint32_t exponent = nbits >> 24;
    uint32_t mantissa = nbits & 0x00ffffff;
    if (mantissa == 0 || exponent == 0) return r;
    if (mantissa & 0x00800000) return r;
    // target = mantissa * 256^(exponent-3)
    // place mantissa (24-bit) then shift bytes
    // represent as big number in base-256 little-endian
    uint8_t tmp[40] = {0};
    tmp[0] = mantissa & 0xFF;
    tmp[1] = (mantissa >> 8) & 0xFF;
    tmp[2] = (mantissa >> 16) & 0xFF;
    if (exponent <= 3) {
        // shift right by 8*(3-exponent) bits = drop (3-exponent) bytes
        uint32_t drop = 3 - exponent;
        for (int i = 0; i < 32; i++)
            r.b[i] = (i + drop < 40) ? tmp[i + drop] : 0;
    } else {
        // shift left by (exponent-3) bytes
        uint32_t sh = exponent - 3;
        for (int i = 0; i < 32; i++) {
            int src = (int)i - (int)sh;
            r.b[i] = (src >= 0 && src < 40) ? tmp[src] : 0;
        }
    }
    return r;
}

// multiply a 256-bit LE number by a u64 multiplier; if overflow -> return MAX.
// Mirrors: if target > U256::MAX / factor { MAX } else target*factor.
static U256 mul_u256_u64_saturate(const U256& a, uint64_t factor) {
    if (factor == 0) { U256 z; return z; }
    // compute product into 40 bytes, detect overflow beyond 32 bytes
    uint8_t prod[48] = {0};
    uint64_t carry = 0;
    // multiply byte-by-byte: treat factor as up to 64-bit
    // do schoolbook: for each byte of a, multiply by factor
    // accumulate into prod with proper shifting
    // simpler: iterate over 32 bytes of a, prod is base-256
    uint64_t acc[48] = {0};
    for (int i = 0; i < 32; i++) {
        uint64_t av = a.b[i];
        if (av == 0) continue;
        uint64_t f = factor;
        // av*f can be up to 255 * 2^64 -> need 128-bit; split factor into bytes
        // do: for each byte position of factor
        for (int jf = 0; jf < 8; jf++) {
            uint64_t fb = (f >> (8*jf)) & 0xFF;
            if (fb == 0) continue;
            uint64_t mul = av * fb;
            int pos = i + jf;
            uint64_t c = mul;
            int p = pos;
            while (c && p < 48) {
                acc[p] += c & 0xFF; // will normalize later
                c >>= 8;
                // but acc may exceed 255; handle via full normalization after
                // Instead accumulate full value:
                p++;
            }
            // The above is wrong for multi-byte; do a cleaner accumulation:
        }
    }
    // The byte-split approach above is error-prone. Reimplement cleanly below.
    memset(acc, 0, sizeof(acc));
    for (int i = 0; i < 32; i++) {
        uint64_t av = a.b[i];
        if (!av) continue;
        // multiply av (<=255) by factor (64-bit) -> up to ~72-bit; split into bytes
        // value = av * factor
        unsigned __int128 v = (unsigned __int128)av * (unsigned __int128)factor;
        int p = i;
        while (v && p < 48) {
            acc[p] += (uint64_t)(v & 0xFF);
            v >>= 8;
            p++;
        }
    }
    // normalize carries
    uint64_t carry2 = 0;
    for (int i = 0; i < 48; i++) {
        uint64_t cur = acc[i] + carry2;
        prod[i] = (uint8_t)(cur & 0xFF);
        carry2 = cur >> 8;
    }
    (void)carry; (void)carry2;
    // overflow if any byte beyond index 31 is nonzero
    for (int i = 32; i < 48; i++) {
        if (prod[i] != 0) {
            U256 maxv;
            memset(maxv.b, 0xFF, 32);
            return maxv;
        }
    }
    U256 r;
    memcpy(r.b, prod, 32);
    return r;
}

static Digest compute_jackpot_hash(const uint32_t jackpot[16], const Digest& key) {
    uint8_t msg[64];
    for (int i = 0; i < 64; i++)
        msg[i] = (uint8_t)(jackpot[i/4] >> (8*(i%4)));
    return blake3_digest(msg, 64, &key);
}

static inline uint32_t rotl32(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }

// CPU ground-truth recomputation for ONE candidate tile.  This is cheap even at
// the live Kryptex dimensions (only h+w strips, not the full m*n search) and is
// deliberately run before printing/submitting a mined proof.  It catches every
// class of false positive that would otherwise become a pool reject: GPU math
// bugs, row/col pattern mixups, target endian mistakes, and stale A/B/noise
// state after the redraw loop.
static Digest compute_tile_jackpot_hash_cpu(
    const vector<int8_t>& A,
    const vector<int8_t>& Bt,
    size_t k,
    size_t rank,
    const vector<size_t>& a_rows,
    const vector<size_t>& b_cols,
    const vector<array<uint32_t,2>>& e_ar_t,
    const vector<array<uint32_t,2>>& e_bl,
    const Digest& seed_a_label,
    const Digest& seed_b_label,
    const Digest& a_noise_seed,
    const Digest& b_noise_seed,
    uint32_t jackpot_out[16])
{
    memset(jackpot_out, 0, 16 * sizeof(uint32_t));
    const size_t tile_h = a_rows.size();
    const size_t tile_w = b_cols.size();

    vector<vector<int8_t>> na_rows(tile_h), nb_cols(tile_w);
    for (size_t u = 0; u < tile_h; u++) {
        vector<int8_t> e_al = uniform_row(seed_a_label, a_noise_seed, a_rows[u], rank);
        na_rows[u] = matvec_sparse_perm(e_ar_t, e_al);
    }
    for (size_t v = 0; v < tile_w; v++) {
        vector<int8_t> e_br = uniform_row(seed_b_label, b_noise_seed, b_cols[v], rank);
        nb_cols[v] = matvec_sparse_perm(e_bl, e_br);
    }

    vector<vector<int32_t>> jackpot_tile(tile_h, vector<int32_t>(tile_w, 0));
    for (size_t ll = rank; ll <= k; ll += rank) {
        for (size_t u = 0; u < tile_h; u++) {
            const size_t a_idx = a_rows[u];
            const vector<int8_t>& na = na_rows[u];
            for (size_t v = 0; v < tile_w; v++) {
                const size_t b_idx = b_cols[v];
                const vector<int8_t>& nb = nb_cols[v];
                int32_t acc = jackpot_tile[u][v];
                for (size_t l = ll - rank; l < ll; l++) {
                    const int32_t a_no = (int32_t)A[a_idx*k + l] + (int32_t)na[l];
                    const int32_t b_no = (int32_t)Bt[b_idx*k + l] + (int32_t)nb[l];
                    acc += a_no * b_no;
                }
                jackpot_tile[u][v] = acc;
            }
        }

        uint32_t xored = 0;
        for (size_t u = 0; u < tile_h; u++)
            for (size_t v = 0; v < tile_w; v++)
                xored ^= (uint32_t)jackpot_tile[u][v];
        const size_t tid = (ll / rank - 1) % 16;
        jackpot_out[tid] = rotl32(jackpot_out[tid], 13) ^ xored;
    }

    return compute_jackpot_hash(jackpot_out, a_noise_seed);
}

static void fprint_u256_be(FILE* f, const U256& u) {
    for (int i = 31; i >= 0; i--) fprintf(f, "%02x", u.b[i]);
}

static void fprint_digest_be_as_u256(FILE* f, const Digest& d) {
    for (int i = 31; i >= 0; i--) fprintf(f, "%02x", d[(size_t)i]);
}

// ---------------------------------------------------------------------------
// base64 STANDARD
// ---------------------------------------------------------------------------
static std::string base64_encode(const vector<uint8_t>& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    while (i + 3 <= in.size()) {
        uint32_t v = (in[i]<<16)|(in[i+1]<<8)|in[i+2];
        out.push_back(T[(v>>18)&63]); out.push_back(T[(v>>12)&63]);
        out.push_back(T[(v>>6)&63]);  out.push_back(T[v&63]);
        i += 3;
    }
    if (in.size() - i == 1) {
        uint32_t v = in[i]<<16;
        out.push_back(T[(v>>18)&63]); out.push_back(T[(v>>12)&63]);
        out.push_back('='); out.push_back('=');
    } else if (in.size() - i == 2) {
        uint32_t v = (in[i]<<16)|(in[i+1]<<8);
        out.push_back(T[(v>>18)&63]); out.push_back(T[(v>>12)&63]);
        out.push_back(T[(v>>6)&63]); out.push_back('=');
    }
    return out;
}

// ---------------------------------------------------------------------------
// bincode serialization (1.3.3 default: LE fixint, usize->u64, Vec->u64 count)
// ---------------------------------------------------------------------------
struct BinWriter {
    vector<uint8_t> buf;
    void u64(uint64_t v) { for (int i=0;i<8;i++) buf.push_back((v>>(8*i))&0xFF); }
    void bytes(const uint8_t* p, size_t n) { buf.insert(buf.end(), p, p+n); }
};

static void write_merkle_proof(BinWriter& w, const MerkleProof& mp) {
    // leaf_data: serde_chunk_vec -> Vec<&[u8]> -> u64 outer_count, per leaf u64(len)+bytes
    w.u64(mp.leaf_data.size());
    for (auto& ld : mp.leaf_data) {
        w.u64(CHUNK_LEN);
        w.bytes(ld.data(), CHUNK_LEN);
    }
    // leaf_indices: Vec<usize> -> u64 count, each usize->u64
    w.u64(mp.leaf_indices.size());
    for (size_t x : mp.leaf_indices) w.u64((uint64_t)x);
    // total_leaves: usize -> u64
    w.u64((uint64_t)mp.total_leaves);
    // root: [u8;32] -> 32 raw bytes (no prefix)
    w.bytes(mp.root.data(), 32);
    // siblings: Vec<[u8;32]> -> u64 count, each 32 raw bytes
    w.u64(mp.siblings.size());
    for (auto& s : mp.siblings) w.bytes(s.data(), 32);
}

static void write_matrix_proof(BinWriter& w, const MerkleProof& mp, const vector<size_t>& row_indices) {
    write_merkle_proof(w, mp);
    w.u64(row_indices.size());
    for (size_t x : row_indices) w.u64((uint64_t)x);
}

// ---------------------------------------------------------------------------
// padding helper
// ---------------------------------------------------------------------------
static vector<uint8_t> pad_to_chunk_boundary(const vector<int8_t>& flat) {
    size_t padded = ((flat.size() + CHUNK_LEN - 1) / CHUNK_LEN) * CHUNK_LEN;
    vector<uint8_t> out(padded, 0);
    for (size_t i = 0; i < flat.size(); i++) out[i] = (uint8_t)flat[i];
    return out;
}

// ---------------------------------------------------------------------------
static vector<uint8_t> parse_hex(const std::string& s) {
    vector<uint8_t> out;
    auto nyb = [](char c)->int{
        if (c>='0'&&c<='9') return c-'0';
        if (c>='a'&&c<='f') return c-'a'+10;
        if (c>='A'&&c<='F') return c-'A'+10;
        return -1;
    };
    for (size_t i = 0; i + 1 < s.size(); i += 2) {
        int hi = nyb(s[i]), lo = nyb(s[i+1]);
        if (hi < 0 || lo < 0) break;
        out.push_back((uint8_t)((hi<<4)|lo));
    }
    return out;
}

// ---------------------------------------------------------------------------
// dp4a search (gpu_jackpot_search / gpu_mine_*) retired — tc_block.cu's
// tc_jackpot_search is the one verified GPU kernel. See _archive/dead-kernels/.

extern "C" int tc_jackpot_search(
    const signed char* a_noised, const signed char* b_noised_t,
    int m,int n,int k,int rank,
    const int* pat_rows, const int* pat_cols, int h,int w,
    const int* row_off, int nrow_off, const int* col_off, int ncol_off,
    const unsigned char* a_noise_seed32, const unsigned char* bound_le32,
    unsigned int* out_hashes_host, unsigned int* dbg_j0_host,
    int* out_rt, int* out_ct);

// GPU-resident draw pipeline (tc_cutlass_v2.cu + gpu_prep.cu). WEAK so builds
// that link other kernels (tc_block, sweep harnesses) still resolve; the mine
// loop only takes the GPU path when all three symbols are present.
extern "C" int tc_alloc_bufs(int m,int n,int k,int h,int w,int nrow_off,int ncol_off,
                             signed char** dA, signed char** dBt) __attribute__((weak));
extern "C" int gpu_prep_phase1(
    signed char* dA, signed char* dBt, int m,int n,int k,
    uint64_t seed, uint64_t draw, const unsigned char* job_key32,
    unsigned char* hash_a32, unsigned char* hash_b32,
    double* ms_rng, double* ms_hash) __attribute__((weak));
extern "C" int gpu_prep_phase2(
    signed char* dA, signed char* dBt, int m,int n,int k,int rank,
    const unsigned char* a_noise_seed32, const unsigned char* b_noise_seed32,
    const unsigned char* label_a32, const unsigned char* label_b32,
    const unsigned int* perm_a, const unsigned int* perm_b,
    double* ms_noise) __attribute__((weak));

int mine_plain_proof(const MineParams& P, MineResult& R, std::atomic<bool>* stop_flag) {
    R.found = false;
    // Unpack params into the locals the pipeline below already uses (faithful
    // move of the former main() body; same names, same logic).
    uint64_t seed = P.seed;
    bool dump = P.dump;
    bool probe = P.probe;
    bool use_tc = P.use_tc;
    bool mine = P.mine;
    bool real_cfg = P.real_cfg;            // REAL kryptex config (m=n=131072,k=4096,r=256)
    bool breakdown = P.breakdown;
    uint64_t maxdraws = P.maxdraws;
    std::string header_hex_override = P.header_hex;  // "" => golden default header
    std::string target_hex = P.target_hex;

    // -----------------------------------------------------------------------
    // Config selection. GOLDEN (default) = small CPU-winnable toy instance used
    // for fast oracle cross-checks. REAL (--cfg real) = the live kryptex network
    // config captured from lpminer --pearl-share-dump (m=n=131072,k=4096,r=256;
    // patterns verified to reproduce the oracle's 52-byte mining_config.bin).
    // The whole pipeline below (noise/jackpot/merkle/bincode) is dimension- and
    // pattern-agnostic, so only these few values change between configs.
    // -----------------------------------------------------------------------
    size_t m, n, k, rank;
    Header header;
    Config config;
    if (real_cfg) {
        m = 131072; n = 131072; k = 4096; rank = 256;
        config.rows_pattern = PeriodicPattern::from_list({0,8,32,40,64,72,96,104});
        config.cols_pattern = PeriodicPattern::from_list(
            {0,1,32,33,64,65,96,97,128,129,160,161,192,193,224,225});
    } else {
        m = 6144; n = 4096; k = 2240; rank = 128;
        config.rows_pattern = PeriodicPattern::from_list({0,1,8,9,64,65,72,73});
        config.cols_pattern = PeriodicPattern::from_list({0,1,8,9,64,65,72,73});
    }
    config.common_dim = (uint32_t)k;
    config.rank = (uint16_t)rank;

    // header bytes: golden by default, or raw 76B from --header <152hex>
    vector<uint8_t> header_bytes = header.to_bytes();
    if (!header_hex_override.empty()) {
        header_bytes = parse_hex(header_hex_override);
        if (header_bytes.size() != 76) {
            fprintf(stderr, "ERROR: --header must be 152 hex chars (76 bytes), got %zu bytes\n", header_bytes.size());
            return 1;
        }
    }
    uint32_t nbits = (uint32_t)header_bytes[72] | ((uint32_t)header_bytes[73] << 8) |
                     ((uint32_t)header_bytes[74] << 16) | ((uint32_t)header_bytes[75] << 24);

    // job_key = blake3(header_bytes || config.to_bytes())
    vector<uint8_t> jk_input = header_bytes;
    {
        auto cb = config.to_bytes();
        jk_input.insert(jk_input.end(), cb.begin(), cb.end());
    }
    Digest job_key = blake3_digest(jk_input, nullptr);

    // A/B buffers are shared by the single-shot path and by the redraw loop.
    // In mine mode do NOT run the historical one-shot A/B/hash/noise setup here:
    // each draw immediately regenerates fresh A/B and seeds below. Running this
    // setup before entering the loop cost ~20 seconds at the real config and made
    // pool jobs stale before the first actual draw could even start.
    vector<int8_t> A(m * k);     // A[i*k + j]
    vector<int8_t> Bt(n * k);    // Bt[i*k + j] = b_matrix[j][i]  (transpose)
    auto chunk_padded_len = [](size_t len) -> size_t {
        const size_t rem = len % BLAKE3_CHUNK_LEN;
        return rem ? (len + (BLAKE3_CHUNK_LEN - rem)) : len;
    };
    vector<uint8_t> a_padded(chunk_padded_len(A.size()));
    vector<uint8_t> bt_padded(chunk_padded_len(Bt.size()));
    Digest hash_a{}, hash_b{}, b_noise_seed{}, a_noise_seed{};

    Digest seed_a_label; memcpy(seed_a_label.data(), SEED_LABEL_A, 32);
    Digest seed_b_label; memcpy(seed_b_label.data(), SEED_LABEL_B, 32);

    vector<array<uint32_t,2>> e_ar_t;
    vector<array<uint32_t,2>> e_bl;

    if (!mine) {
        // RNG: signal in [-64, 64] inclusive (65 values). Reproducible.
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> dist(-64, 64);

        // Generate A (m x k) row-major, and B (k x n), then transpose B into Bt.
        // Generate in the SAME order as Rust: A first (row-major), then B
        // (row-major k x n). This path is kept for CPU/oracle self-tests.
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < k; j++)
                A[i*k + j] = (int8_t)dist(rng);
        vector<int8_t> B(k * n); // B[i*n + j]
        for (size_t i = 0; i < k; i++)
            for (size_t j = 0; j < n; j++)
                B[i*n + j] = (int8_t)dist(rng);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < k; j++)
                Bt[i*k + j] = B[j*n + i];

        // commitments
        a_padded = pad_to_chunk_boundary(A);
        bt_padded = pad_to_chunk_boundary(Bt);
        hash_a = blake3_digest(a_padded, &job_key);
        hash_b = blake3_digest(bt_padded, &job_key);

        // b_noise_seed = blake3(job_key || hash_b); a_noise_seed = blake3(b_noise_seed || hash_a)
        uint8_t seed_in[64];
        memcpy(seed_in, job_key.data(), 32); memcpy(seed_in+32, hash_b.data(), 32);
        b_noise_seed = blake3_digest(seed_in, 64, nullptr);
        memcpy(seed_in, b_noise_seed.data(), 32); memcpy(seed_in+32, hash_a.data(), 32);
        a_noise_seed = blake3_digest(seed_in, 64, nullptr);

        // Permutation matrices are global (k pairs each), generate once.
        e_ar_t = generate_permutation_matrix(seed_a_label, a_noise_seed, k, rank);
        e_bl   = generate_permutation_matrix(seed_b_label, b_noise_seed, k, rank);
    }

    U256 bound;
    {
        U256 td = nbits_to_difficulty(nbits);
        size_t h = config.rows_pattern.size();
        size_t wsz = config.cols_pattern.size();
        size_t dot_len = k - (k % rank);
        uint64_t factor = (uint64_t)h * wsz * dot_len;
        bound = mul_u256_u64_saturate(td, factor);
    }
    // --target overrides the nbits-derived bound. Pool sends 32-byte target in
    // BIG-ENDIAN hex; jackpot hash is compared LITTLE-ENDIAN, so reverse bytes.
    // bound = mul_u256_u64_saturate(target_LE, h*w*dot_len).
    if (!target_hex.empty()) {
        vector<uint8_t> tb = parse_hex(target_hex);
        if (tb.size() != 32) {
            fprintf(stderr, "ERROR: --target must be 64 hex chars (32 bytes), got %zu bytes\n", tb.size());
            return 1;
        }
        U256 target_le;
        for (int i = 0; i < 32; i++) target_le.b[i] = tb[31 - i]; // big-endian -> little-endian
        size_t h = config.rows_pattern.size();
        size_t wsz = config.cols_pattern.size();
        size_t dot_len = k - (k % rank);
        uint64_t factor = (uint64_t)h * wsz * dot_len;
        bound = mul_u256_u64_saturate(target_le, factor);
        fprintf(stderr, "TARGET bound (LE hex)=");
        for (int i = 31; i >= 0; i--) fprintf(stderr, "%02x", bound.b[i]);
        fprintf(stderr, " (factor=%llu)\n", (unsigned long long)factor);
    }

    // We need noise rows of A only for a_rows we test, and noise cols of B for b_cols.
    // To match try_mine_one we iterate threads_partition over rows then cols.
    auto row_parts = threads_partition(config.rows_pattern, m);
    auto col_parts = threads_partition(config.cols_pattern, n);

    // Cache noise rows for A (E_AL row -> noise_a row) and noise cols for B.
    // noise_a[row] = matvec_sparse_perm(e_ar_t, e_al_row)  (length k)
    // e_al_row = uniform_row(seed_a_label, a_noise_seed, row, rank)  (length rank)
    // noise_b_t[col] = matvec_sparse_perm(e_bl, e_br_t_col) (length k)
    // e_br_t_col = uniform_row(seed_b_label, b_noise_seed, col, rank) (length rank)
    std::map<size_t, vector<int8_t>> noise_a_cache, noise_b_cache;
    auto get_noise_a = [&](size_t row) -> const vector<int8_t>& {
        auto it = noise_a_cache.find(row);
        if (it != noise_a_cache.end()) return it->second;
        vector<int8_t> e_al = uniform_row(seed_a_label, a_noise_seed, row, rank);
        vector<int8_t> na = matvec_sparse_perm(e_ar_t, e_al);
        return noise_a_cache.emplace(row, std::move(na)).first->second;
    };
    auto get_noise_b = [&](size_t col) -> const vector<int8_t>& {
        auto it = noise_b_cache.find(col);
        if (it != noise_b_cache.end()) return it->second;
        vector<int8_t> e_br = uniform_row(seed_b_label, b_noise_seed, col, rank);
        vector<int8_t> nb = matvec_sparse_perm(e_bl, e_br);
        return noise_b_cache.emplace(col, std::move(nb)).first->second;
    };

    bool found = false;
    vector<size_t> win_rows, win_cols;

    // row_off/col_off (base offsets) and pattern arrays are draw-invariant.
    // pat_rows / pat_cols = within-tile index offsets (PeriodicPattern.to_list()).
    // GOLDEN has identical row/col patterns; REAL differs (h=8 rows, w=16 cols),
    // so the two arrays are distinct and must NOT be conflated. The legacy dp4a /
    // tc kernels below assume pat_rows==pat_cols and are GOLDEN-only; the
    // real-config search uses the CPU path here (and, later, the fused kernel).
    std::vector<uint32_t> _pr = config.rows_pattern.to_list();
    std::vector<uint32_t> _pc = config.cols_pattern.to_list();
    std::vector<int> pat_rows(_pr.begin(), _pr.end());
    std::vector<int> pat_cols(_pc.begin(), _pc.end());
    int hh=(int)config.rows_pattern.size(), ww=(int)config.cols_pattern.size();
    std::vector<int> row_off(row_parts.size()), col_off(col_parts.size());
    for (size_t i=0;i<row_parts.size();i++) row_off[i]=(int)row_parts[i][0];
    for (size_t j=0;j<col_parts.size();j++) col_off[j]=(int)col_parts[j][0];

    if (mine) {
        // -------------------------------------------------------------------
        // Internal redraw loop, CPU/GPU PIPELINED. INVARIANTS hoisted above:
        // header/config, job_key, bound, pattern, row_parts/col_parts,
        // row_off/col_off. PER DRAW: fresh A/Bt -> hash_a/hash_b -> noise seeds
        // -> noise -> a_noised/b_noised_t -> GPU search. job_key is keyed by
        // header+config ONLY (constant); hash_a/hash_b and the noise seeds
        // depend on A/Bt, so they are recomputed every draw.
        //
        // OVERLAP: the ~1.4s CPU prep (RNG+blake3+noise+build) for draw N+1 runs
        // on a producer thread WHILE the ~2.3s GPU search for draw N runs on this
        // thread, hiding the CPU cost (~1.6x throughput). Only the GPU handoff
        // (a_noised/b_noised_t/a_noise_seed) is double-buffered; A/Bt, the padded
        // copies, the permutation matrices and the noise scratch are shared and
        // touched by the single producer only, so there is no race with the GPU
        // (which reads just the slot it was handed). On a win the producer has
        // already advanced the shared scratch to draw N+1, so we re-run
        // produce_draw(N) once (a pure function of (seed,draw) -> byte identical)
        // to restore draw N's state for the POSTCHECK + Merkle proof below.
        // tc_jackpot_search (tc_block.cu) manages its own device memory per call.
        // -------------------------------------------------------------------

        // Double-buffered GPU handoff (two slots). Everything else is shared
        // scratch, allocated once and written only inside produce_draw.
        std::vector<signed char> a_noised[2]   = { std::vector<signed char>((size_t)m*k), std::vector<signed char>((size_t)m*k) };
        std::vector<signed char> b_noised_t[2] = { std::vector<signed char>((size_t)n*k), std::vector<signed char>((size_t)n*k) };
        Digest a_noise_seed_slot[2] = {};
        std::vector<int8_t> noise_a((size_t)m*k), noise_b((size_t)n*k);

        using clk = std::chrono::high_resolution_clock;
        auto t_start = clk::now();
        auto t_window = t_start;
        uint64_t draw = 0;
        uint64_t window_draw0 = 0;
        // Match SRBMiner-MULTI/lpminer/pool display units: TH/s.  One displayed
        // PRL "hash" here is the same work unit used by the jackpot bound:
        // tile_count * pattern_rows * pattern_cols * dot_len.  Do not report
        // raw tile/s or tensor-MAC-specific labels in user-facing hashrate.
        const double work_per_draw =
            (double)row_parts.size() * (double)col_parts.size() *
            (double)hh * (double)ww * (double)(k - (k % rank));
        R.work_per_draw = work_per_draw;
        auto ths_for = [&](uint64_t draws_done, double seconds) -> double {
            return seconds > 0.0 ? (double)draws_done * work_per_draw / seconds / 1e12 : 0.0;
        };
        auto stopping = [&]() -> bool {
            return stop_flag && stop_flag->load(std::memory_order_relaxed);
        };

        // ONE canonical production path (stages a-d) -> shared A/Bt/a_padded/
        // bt_padded/e_ar_t/e_bl/hash_*/seeds AND the per-slot GPU handoff. Pure
        // function of (seed,d): safe to re-run for the winning draw to restore
        // shared state after the producer has moved on. Runs on the producer
        // thread (overlapping the GPU) and, once, on this thread to re-derive.
        auto produce_draw = [&](uint64_t d, int slot, bool tells) {
            double ms_rng=0, ms_hash=0, ms_noise=0, ms_build=0;
            auto tic = clk::now();
            auto lap = [&](double& dst){ auto now=clk::now(); dst=std::chrono::duration<double,std::milli>(now-tic).count(); tic=now; };

            // (a) RNG fill of A and Bt. Each row gets its own splitmix64 stream
            // seeded deterministically from (d, row_index) so OpenMP is safe and
            // order-independent. A used for hash_a == A used in GEMM/strip.
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < (long)m; i++) {
                uint64_t st=(seed^0x9E3779B97F4A7C15ULL)+d*1000003ULL+(uint64_t)i*0x100000001B3ULL;
                int8_t* rowp=&A[(size_t)i*k];
                for(size_t j=0;j<k;j+=8){uint64_t z=(st+=0x9E3779B97F4A7C15ULL);z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;z=z^(z>>31);for(int b=0;b<8&&j+(size_t)b<k;b++)rowp[j+b]=(int8_t)((int)((((uint32_t)((z>>(8*b))&0xFF))*129u)>>8)-64);}
            }
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < (long)n; i++) {
                uint64_t st=(seed^0xD1B54A32D192ED03ULL)+d*1000003ULL+(uint64_t)i*0x100000001B3ULL;
                int8_t* rowp=&Bt[(size_t)i*k];
                for(size_t j=0;j<k;j+=8){uint64_t z=(st+=0x9E3779B97F4A7C15ULL);z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;z=z^(z>>31);for(int b=0;b<8&&j+(size_t)b<k;b++)rowp[j+b]=(int8_t)((int)((((uint32_t)((z>>(8*b))&0xFF))*129u)>>8)-64);}
            }
            if (tells) lap(ms_rng);

            // (b) blake3 hash_a + hash_b (over padded A / padded Bt) + chained seeds.
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < (long)m; i++)
                for (size_t j=0;j<k;j++) a_padded[(size_t)i*k+j]=(uint8_t)A[(size_t)i*k+j];
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < (long)n; i++)
                for (size_t j=0;j<k;j++) bt_padded[(size_t)i*k+j]=(uint8_t)Bt[(size_t)i*k+j];
            hash_a = blake3_digest(a_padded, &job_key);
            hash_b = blake3_digest(bt_padded, &job_key);
            { uint8_t si[64];
              memcpy(si, job_key.data(),32); memcpy(si+32, hash_b.data(),32);
              b_noise_seed = blake3_digest(si,64,nullptr);
              memcpy(si, b_noise_seed.data(),32); memcpy(si+32, hash_a.data(),32);
              a_noise_seed = blake3_digest(si,64,nullptr); }
            if (tells) lap(ms_hash);

            // (c) noise generation: permutation matrices (depend on seeds) +
            // per-row/per-col uniform noise rows -> noise_a[row]/noise_b[col].
            e_ar_t = generate_permutation_matrix(seed_a_label, a_noise_seed, k, rank);
            e_bl   = generate_permutation_matrix(seed_b_label, b_noise_seed, k, rank);
            #pragma omp parallel for schedule(static)
            for (long row = 0; row < (long)m; row++) {
                vector<int8_t> e_al = uniform_row(seed_a_label, a_noise_seed, (size_t)row, rank);
                vector<int8_t> na = matvec_sparse_perm(e_ar_t, e_al);
                for (size_t l=0;l<k;l++) noise_a[(size_t)row*k+l]=na[l];
            }
            #pragma omp parallel for schedule(static)
            for (long col = 0; col < (long)n; col++) {
                vector<int8_t> e_br = uniform_row(seed_b_label, b_noise_seed, (size_t)col, rank);
                vector<int8_t> nb = matvec_sparse_perm(e_bl, e_br);
                for (size_t l=0;l<k;l++) noise_b[(size_t)col*k+l]=nb[l];
            }
            if (tells) lap(ms_noise);

            // (d) build a_noised[slot] + b_noised_t[slot] int8 + snapshot the seed.
            #pragma omp parallel for schedule(static)
            for (long row=0; row<(long)m; row++)
                for(size_t l=0;l<k;l++) a_noised[slot][(size_t)row*k+l]=(signed char)((int)A[(size_t)row*k+l]+(int)noise_a[(size_t)row*k+l]);
            #pragma omp parallel for schedule(static)
            for (long col=0; col<(long)n; col++)
                for(size_t l=0;l<k;l++) b_noised_t[slot][(size_t)col*k+l]=(signed char)((int)Bt[(size_t)col*k+l]+(int)noise_b[(size_t)col*k+l]);
            a_noise_seed_slot[slot] = a_noise_seed;
            if (tells) { lap(ms_build);
                fprintf(stderr, "CPUPREP draw=%llu (ms): RNG=%.2f blake3=%.2f noise=%.2f build=%.2f\n",
                        (unsigned long long)d, ms_rng, ms_hash, ms_noise, ms_build);
            }
        };

        // -------- GPU-RESIDENT draw pipeline (Phase2, DESIGN_speedup.md) -----
        // When the GPU prep kernels are linked in (tc_cutlass_v2 + gpu_prep),
        // generate RNG + commitments + noise directly in the search kernel's
        // persistent dA/dBt and skip BOTH the 1.5s CPU prep and the 1GB H2D.
        // Host work per draw shrinks to 2 seed hashes + 2 permutation matrices
        // (~1 ms). Correctness: on a win, produce_draw(draw) re-derives the
        // whole draw on the CPU and POSTCHECK independently recomputes the
        // winning tile -> any GPU/CPU divergence fails ok=1 and is refused.
        // Requires power-of-two chunk counts (real config: 512MiB = 2^19) and
        // rank 256; otherwise falls back to the CPU producer path below.
        bool gpu_pipe = false;
        signed char *g_dA = nullptr, *g_dBt = nullptr;
        if (real_cfg && rank == 256 &&
            tc_alloc_bufs && gpu_prep_phase1 && gpu_prep_phase2 &&
            tc_alloc_bufs((int)m,(int)n,(int)k,hh,ww,
                          (int)row_parts.size(),(int)col_parts.size(),
                          &g_dA,&g_dBt) == 0) {
            gpu_pipe = true;
            fprintf(stderr, "MINE: GPU-resident draw pipeline ACTIVE (RNG+hash+noise on GPU, no per-draw H2D)\n");
        }
        if (gpu_pipe) {
            std::vector<unsigned int> perm_fa((size_t)k*2), perm_fb((size_t)k*2);
            for (draw = 0; draw < maxdraws; draw++) {
                if (stopping()) { fprintf(stderr, "MINE abort: stop requested at draw=%llu\n", (unsigned long long)draw); break; }
                double ms_rng=0, ms_hash=0, ms_noise=0;
                if (gpu_prep_phase1(g_dA, g_dBt, (int)m,(int)n,(int)k, seed, draw,
                                    job_key.data(), hash_a.data(), hash_b.data(),
                                    &ms_rng, &ms_hash)) {
                    fprintf(stderr, "MINE: gpu_prep_phase1 failed at draw=%llu\n", (unsigned long long)draw);
                    return 2;
                }
                { uint8_t si[64];
                  memcpy(si, job_key.data(),32); memcpy(si+32, hash_b.data(),32);
                  b_noise_seed = blake3_digest(si,64,nullptr);
                  memcpy(si, b_noise_seed.data(),32); memcpy(si+32, hash_a.data(),32);
                  a_noise_seed = blake3_digest(si,64,nullptr); }
                e_ar_t = generate_permutation_matrix(seed_a_label, a_noise_seed, k, rank);
                e_bl   = generate_permutation_matrix(seed_b_label, b_noise_seed, k, rank);
                for (size_t i=0;i<k;i++) { perm_fa[2*i]=e_ar_t[i][0]; perm_fa[2*i+1]=e_ar_t[i][1];
                                           perm_fb[2*i]=e_bl[i][0];   perm_fb[2*i+1]=e_bl[i][1]; }
                if (gpu_prep_phase2(g_dA, g_dBt, (int)m,(int)n,(int)k,(int)rank,
                                    a_noise_seed.data(), b_noise_seed.data(),
                                    seed_a_label.data(), seed_b_label.data(),
                                    perm_fa.data(), perm_fb.data(), &ms_noise)) {
                    fprintf(stderr, "MINE: gpu_prep_phase2 failed at draw=%llu\n", (unsigned long long)draw);
                    return 2;
                }
                if (breakdown || draw == 5)
                    fprintf(stderr, "GPUPREP draw=%llu (ms): rng=%.1f hash=%.1f noise=%.1f\n",
                            (unsigned long long)draw, ms_rng, ms_hash, ms_noise);

                int rt=-1, ct=-1;
                int ok = tc_jackpot_search(nullptr, nullptr,   // data already in dA/dBt
                                           (int)m,(int)n,(int)k,(int)rank,
                                           pat_rows.data(),pat_cols.data(),hh,ww,
                                           row_off.data(),(int)row_parts.size(),
                                           col_off.data(),(int)col_parts.size(),
                                           a_noise_seed.data(), bound.b,
                                           nullptr, nullptr, &rt, &ct);
                if (ok==1 && rt>=0 && ct>=0) {
                    // Re-derive draw N fully on the CPU: restores A/Bt/padded/
                    // seeds/perms for POSTCHECK + Merkle. POSTCHECK is the
                    // GPU-vs-CPU equivalence gate for this whole pipeline.
                    produce_draw(draw, 0, false);
                    found=true; win_rows=row_parts[(size_t)rt]; win_cols=col_parts[(size_t)ct];
                    fprintf(stderr, "MINE WIN draw=%llu tile rt=%d ct=%d\n",
                            (unsigned long long)(draw+1), rt, ct);
                    break;
                }
                if (ok < 0) { fprintf(stderr, "MINE: kernel error at draw=%llu\n", (unsigned long long)draw); return 2; }

                uint64_t done = draw + 1;
                auto t_now = clk::now();
                double win_el = std::chrono::duration<double>(t_now-t_window).count();
                if (done % 100 == 0 || win_el >= 60.0) {
                    double el = std::chrono::duration<double>(t_now-t_start).count();
                    uint64_t win_draws = done - window_draw0;
                    fprintf(stderr,
                            "draw %llu, elapsed %.2fs, %.2f draws/sec, avg %.2f TH/s, window %.2f TH/s\n",
                            (unsigned long long)done, el, (double)done/el,
                            ths_for(done, el), ths_for(win_draws, win_el));
                    if (win_el >= 60.0) { t_window = t_now; window_draw0 = done; }
                }
            }
        } else {
        // Prime the pipeline: produce draw 0 into slot 0 (synchronous).
        int cur = 0;
        bool primed = (maxdraws > 0) && !stopping();
        if (primed) produce_draw(0, cur, breakdown);

        for (draw = 0; primed && draw < maxdraws; draw++) {
            // Cooperative abort: pool/solo set stop_flag when a newer job arrives.
            // We finish the in-flight GPU draw (uninterruptible) then break.
            if (stopping()) { fprintf(stderr, "MINE abort: stop requested at draw=%llu\n", (unsigned long long)draw); break; }
            bool do_breakdown = breakdown || (draw == 5);

            // Kick the producer for the NEXT draw; it fills the other slot and the
            // shared scratch while this thread runs the GPU on the current slot.
            uint64_t nd = draw + 1;
            int nslot = 1 - cur;
            bool have_next = (nd < maxdraws) && !stopping();
            std::thread prod;
            if (have_next) prod = std::thread([&, nd, nslot, do_breakdown]{ produce_draw(nd, nslot, do_breakdown); });

            // (e) fused tensor-core search on the CURRENT slot (draw N). rt/ct
            // return as row_off/col_off indices mapping back via row_parts/col_parts.
            auto tg = clk::now();
            int rt=-1, ct=-1; float kms=0;
            int ok = tc_jackpot_search(a_noised[cur].data(), b_noised_t[cur].data(),
                                       (int)m,(int)n,(int)k,(int)rank,
                                       pat_rows.data(),pat_cols.data(),hh,ww,
                                       row_off.data(),(int)row_parts.size(),
                                       col_off.data(),(int)col_parts.size(),
                                       a_noise_seed_slot[cur].data(), bound.b,
                                       nullptr, nullptr, &rt, &ct);
            double ms_gpu = std::chrono::duration<double,std::milli>(clk::now()-tg).count();

            // Producer owns the shared scratch -> must complete before we either
            // start the next iteration or re-derive draw N on a win.
            if (prod.joinable()) prod.join();
            if (do_breakdown)
                fprintf(stderr, "BREAKDOWN draw=%llu GPU=%.2f ms (kernel=%.2f) [CPU(N+1) overlapped]\n",
                        (unsigned long long)draw, ms_gpu, (double)kms);

            if (ok==1 && rt>=0 && ct>=0) {
                // The producer already advanced shared A/Bt/e_ar_t/e_bl/seeds to
                // draw nd; re-run draw N to restore byte-exact state for the
                // POSTCHECK + Merkle proof assembly that follows the loop.
                produce_draw(draw, cur, false);
                found=true; win_rows=row_parts[(size_t)rt]; win_cols=col_parts[(size_t)ct];
                fprintf(stderr, "MINE WIN draw=%llu tile rt=%d ct=%d\n",
                        (unsigned long long)(draw+1), rt, ct);
                break;
            }

            cur = nslot;

            uint64_t done = draw + 1;
            auto t_now = clk::now();
            double win_el = std::chrono::duration<double>(t_now-t_window).count();
            if (done % 100 == 0 || win_el >= 60.0) {
                double el = std::chrono::duration<double>(t_now-t_start).count();
                uint64_t win_draws = done - window_draw0;
                fprintf(stderr,
                        "draw %llu, elapsed %.2fs, %.2f draws/sec, avg %.2f TH/s, window %.2f TH/s\n",
                        (unsigned long long)done, el, (double)done/el,
                        ths_for(done, el), ths_for(win_draws, win_el));
                if (win_el >= 60.0) {
                    t_window = t_now;
                    window_draw0 = done;
                }
            }
        }
        } // end CPU-producer path (gpu_pipe else)
        double el = std::chrono::duration<double>(clk::now()-t_start).count();
        uint64_t draws_done = found ? (draw + 1) : draw;
        R.draws = draws_done;
        R.elapsed_s = el;
        fprintf(stderr, "MINE done: draws=%llu elapsed=%.2fs %.2f draws/sec %.2f TH/s found=%d\n",
                (unsigned long long)draws_done, el,
                (double)draws_done/(el>0?el:1),
                ths_for(draws_done, el), found?1:0);
    } else if (use_tc) {
        std::vector<signed char> a_noised((size_t)m*k), b_noised_t((size_t)n*k);
        for (size_t row=0; row<m; row++){ const auto& na=get_noise_a(row); for(size_t l=0;l<k;l++) a_noised[row*k+l]=(signed char)((int)A[row*k+l]+(int)na[l]); }
        for (size_t col=0; col<n; col++){ const auto& nb=get_noise_b(col); for(size_t l=0;l<k;l++) b_noised_t[col*k+l]=(signed char)((int)Bt[col*k+l]+(int)nb[l]); }
        int rt=-1, ct=-1;
        int ok = tc_jackpot_search(a_noised.data(), b_noised_t.data(),
                                   (int)m,(int)n,(int)k,(int)rank,
                                   pat_rows.data(),pat_cols.data(),hh,ww,
                                   row_off.data(),(int)row_parts.size(),
                                   col_off.data(),(int)col_parts.size(),
                                   a_noise_seed.data(), bound.b,
                                   nullptr, nullptr, &rt, &ct);
        if (ok==1 && rt>=0 && ct>=0) {
            found=true; win_rows=row_parts[(size_t)rt]; win_cols=col_parts[(size_t)ct];
            fprintf(stderr, "TC WIN tile rt=%d ct=%d\n", rt, ct);
        } else {
            fprintf(stderr, "TC search: no win (ok=%d)\n", ok);
        }
    } else if (probe) {
        // Probe: emit a structurally-valid proof for the FIRST tile without searching
        // for a winning jackpot. Used to test pool format/dims acceptance at real
        // difficulty, where finding a CPU win is infeasible.
        found = true;
        win_rows = row_parts[0];
        win_cols = col_parts[0];
    } else {
    for (auto& a_rows : row_parts) {
        for (auto& b_cols : col_parts) {
            size_t tile_h = a_rows.size(), tile_w = b_cols.size();
            // precompute noised rows for this tile
            // a_noised[a_idx][l] = A[a_idx*k+l] + noise_a[a_idx][l]
            // b_noised_t[b_idx][l] = Bt[b_idx*k+l]? NO: b_noised_t[b_idx][l]
            //   = b_matrix[l][b_idx] + noise.b[b_idx][l] = Bt[b_idx*k+l] + noise_b[b_idx][l]
            uint32_t jackpot[16] = {0};
            // jackpot_tile accumulates over the whole k (not reset per ll)
            vector<vector<int32_t>> jackpot_tile(tile_h, vector<int32_t>(tile_w, 0));

            // gather noise rows
            vector<const vector<int8_t>*> na_rows(tile_h), nb_cols(tile_w);
            for (size_t u = 0; u < tile_h; u++) na_rows[u] = &get_noise_a(a_rows[u]);
            for (size_t v = 0; v < tile_w; v++) nb_cols[v] = &get_noise_b(b_cols[v]);

            for (size_t ll = rank; ll <= k; ll += rank) {
                for (size_t u = 0; u < tile_h; u++) {
                    size_t a_idx = a_rows[u];
                    const vector<int8_t>& na = *na_rows[u];
                    for (size_t v = 0; v < tile_w; v++) {
                        size_t b_idx = b_cols[v];
                        const vector<int8_t>& nb = *nb_cols[v];
                        int32_t acc = jackpot_tile[u][v];
                        for (size_t l = ll - rank; l < ll; l++) {
                            int32_t a_no = (int32_t)A[a_idx*k + l] + (int32_t)na[l];
                            int32_t b_no = (int32_t)Bt[b_idx*k + l] + (int32_t)nb[l];
                            acc += a_no * b_no;
                        }
                        jackpot_tile[u][v] = acc;
                    }
                }
                uint32_t xored = 0;
                for (size_t u = 0; u < tile_h; u++)
                    for (size_t v = 0; v < tile_w; v++)
                        xored ^= (uint32_t)jackpot_tile[u][v];
                size_t tid = (ll / rank - 1) % 16;
                jackpot[tid] = rotl32(jackpot[tid], 13) ^ xored;
            }

            Digest jh = compute_jackpot_hash(jackpot, a_noise_seed);
            U256 jhv = U256::from_le(jh.data());
            if (jhv.le(bound)) {
                found = true;
                win_rows = a_rows;
                win_cols = b_cols;
                break;
            }
        }
        if (found) break;
    }
    }

    if (!found) {
        fprintf(stderr, "No winning tile found with seed %llu\n", (unsigned long long)seed);
        return 2;
    }

    fprintf(stderr, "WIN rows=[");
    for (size_t i=0;i<win_rows.size();i++) fprintf(stderr, "%zu%s", win_rows[i], i+1<win_rows.size()?", ":"");
    fprintf(stderr, "] cols=[");
    for (size_t i=0;i<win_cols.size();i++) fprintf(stderr, "%zu%s", win_cols[i], i+1<win_cols.size()?", ":"");
    fprintf(stderr, "]\n");

    // Final CPU ground-truth gate for any real mined win.  --probe intentionally
    // emits a structurally-valid but non-winning proof, so it is excluded.
    if (!probe) {
        uint32_t post_jackpot[16];
        Digest post_hash = compute_tile_jackpot_hash_cpu(
            A, Bt, k, rank, win_rows, win_cols, e_ar_t, e_bl,
            seed_a_label, seed_b_label, a_noise_seed, b_noise_seed, post_jackpot);
        U256 post_v = U256::from_le(post_hash.data());
        const bool post_ok = post_v.le(bound);
        fprintf(stderr, "POSTCHECK jackpot=");
        for (int i = 0; i < 16; i++) fprintf(stderr, "%08x%s", post_jackpot[i], i+1<16?",":"");
        fprintf(stderr, " hash_u256_be=");
        fprint_digest_be_as_u256(stderr, post_hash);
        fprintf(stderr, " bound_be=");
        fprint_u256_be(stderr, bound);
        fprintf(stderr, " ok=%d\n", post_ok ? 1 : 0);
        if (!post_ok) {
            fprintf(stderr, "ERROR: GPU/driver reported a win, but CPU postcheck says it does not meet the active target; refusing to emit a pool-rejected proof.\n");
            return 4;
        }
    }

    // Build Merkle proofs
    MerkleTree tree_a(a_padded.data(), a_padded.size(), job_key);
    MerkleTree tree_bt(bt_padded.data(), bt_padded.size(), job_key);
    vector<size_t> a_leaf_idx = compute_leaf_indices_from_rows(win_rows, m, k);
    vector<size_t> bt_leaf_idx = compute_leaf_indices_from_rows(win_cols, n, k);
    MerkleProof a_proof = get_multileaf_proof(tree_a, a_leaf_idx);
    MerkleProof bt_proof = get_multileaf_proof(tree_bt, bt_leaf_idx);

    // Serialize PlainProof via bincode
    BinWriter w;
    w.u64((uint64_t)m);
    w.u64((uint64_t)n);
    w.u64((uint64_t)k);
    w.u64((uint64_t)rank); // noise_rank
    write_matrix_proof(w, a_proof, win_rows);
    write_matrix_proof(w, bt_proof, win_cols);

    std::string b64 = base64_encode(w.buf);
    R.proof_b64 = b64;
    R.found = true;
    R.win_rows = win_rows;
    R.win_cols = win_cols;

    // header hex for verifier
    {
        fprintf(stderr, "HEADER_HEX=");
        for (uint8_t x : header_bytes) fprintf(stderr, "%02x", x);
        fprintf(stderr, "\n");
    }

    if (dump) {
        FILE* fa = fopen("/tmp/A.bin", "wb"); fwrite(A.data(), 1, A.size(), fa); fclose(fa);
        FILE* fb = fopen("/tmp/Bt.bin", "wb"); fwrite(Bt.data(), 1, Bt.size(), fb); fclose(fb);
        FILE* ft = fopen("/tmp/tile.txt", "w");
        fprintf(ft, "%zu", win_rows.size());
        for (size_t r : win_rows) fprintf(ft, " %zu", r);
        fprintf(ft, "\n%zu", win_cols.size());
        for (size_t c : win_cols) fprintf(ft, " %zu", c);
        fprintf(ft, "\n");
        fclose(ft);
        // also dump intermediates
        FILE* fi = fopen("/tmp/cpp_intermediates.txt", "w");
        auto hx = [&](FILE* f, const char* nm, const Digest& d){ fprintf(f, "%s=", nm); for (uint8_t b : d) fprintf(f, "%02x", b); fprintf(f, "\n"); };
        hx(fi, "job_key", job_key);
        hx(fi, "hash_a", hash_a);
        hx(fi, "hash_b", hash_b);
        hx(fi, "b_noise_seed", b_noise_seed);
        hx(fi, "a_noise_seed", a_noise_seed);
        hx(fi, "merkle_root_a", a_proof.root);
        hx(fi, "merkle_root_bt", bt_proof.root);
        fprintf(fi, "a_siblings=%zu bt_siblings=%zu\n", a_proof.siblings.size(), bt_proof.siblings.size());
        fclose(fi);
    }

    return 0;
}

#ifndef PROVER_LIB
// Standalone CLI: argv -> MineParams -> mine_plain_proof -> base64 proof on stdout.
// Preserved verbatim from the historical interface (seed | --dump | --probe |
// --tc | --mine [N] | --target <64hex> | --header <152hex> | --cfg real).
int main(int argc, char** argv) {
    MineParams P;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            fprintf(stderr,
                "plainproof_gen - Pearl(PRL) PlainProof generator/miner\n\n"
                "usage:\n"
                "  plainproof_gen [seed]\n"
                "  plainproof_gen --mine [N] [--cfg golden|real] [--header 152hex] [--target 64hex]\n\n"
                "options:\n"
                "  --mine [N]        mine up to N draws and emit base64 PlainProof on success (fused tensor-core)\n"
                "  --tc              force the tensor-core kernel for the single-shot self-test path\n"
                "  --cfg real        use live pool dimensions (default is golden test config)\n"
                "  --header HEX      76-byte block header as 152 hex chars\n"
                "  --target HEX      32-byte pool target as 64 big-endian hex chars\n"
                "  --breakdown       print per-draw timing breakdown to stderr\n"
                "  --probe           emit a probe proof shape without jackpot search\n"
                "  --dump            dump debug artifacts for oracle comparison\n");
            return 0;
        }
        else if (a == "--dump") P.dump = true;
        else if (a == "--probe") P.probe = true;
        else if (a == "--breakdown") P.breakdown = true;
        else if (a == "--tc") P.use_tc = true;
        else if (a == "--mine") {
            P.mine = true;
            if (i + 1 < argc) {
                char* end = nullptr;
                unsigned long long v = strtoull(argv[i+1], &end, 10);
                if (end && *end == '\0' && argv[i+1][0] != '\0') { P.maxdraws = (uint64_t)v; i++; }
            }
        }
        else if (a == "--target" && i + 1 < argc) P.target_hex = argv[++i];
        else if (a == "--header" && i + 1 < argc) P.header_hex = argv[++i];
        else if (a == "--cfg" && i + 1 < argc) P.real_cfg = (std::string(argv[++i]) == "real");
        else P.seed = strtoull(argv[i], nullptr, 10);
    }
    MineResult R;
    int rc = mine_plain_proof(P, R, nullptr);
    if (rc == 0 && R.found) printf("%s\n", R.proof_b64.c_str());
    return rc;
}
#endif  // PROVER_LIB
