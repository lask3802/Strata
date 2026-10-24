// A K-slice [ff0, ff0 + ffn) of an expert's n_ff as bytes (expert parallelism, tensor parallel members): the gate and
// up rows [ff0, ..) - contiguous - and in each of the dim down rows the blocks of the inputs [ff0, ..) - strided.  The
// whole expert's blob and a slice's blob each have their matrices at their own offsets (gate, up, down); a slice's down
// rows are packed (ds bytes apart), so the device kernels read it as an expert of n_ff = ffn.  No GPU:
// tests/dsv41/ep_slice_test.cpp; the engine's wrappers are Impl::mx_blob_to_slice / mx_slice_to_blob (ds4_model.cu).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::ds4 {

struct FfSlice {
    int dim = 0;          // down rows (n_embd)
    int ff0 = 0, ffn = 0; // the slice of n_ff
    size_t gr = 0;        // bytes a gate / up row (dim values)
    size_t dr = 0;        // bytes a whole down row (n_ff values)
    size_t ds = 0;        // bytes a slice's down row (ffn values)
    size_t d_off = 0;     // bytes of the values [0, ff0) in a down row
};

/// whole (matrices at src[3]) -> the slice's blob (matrices at dst[3])
inline void ff_blob_to_slice(const FfSlice& s, const uint8_t* blob, const size_t src[3], uint8_t* dst, const size_t dsto[3]) {
    for (int r = 0; r < 2; ++r)
        std::memcpy(dst + dsto[r], blob + src[r] + (size_t) s.ff0 * s.gr, (size_t) s.ffn * s.gr);
    for (size_t row = 0; row < (size_t) s.dim; ++row)
        std::memcpy(dst + dsto[2] + row * s.ds, blob + src[2] + row * s.dr + s.d_off, s.ds);
}
/// the slice's blob -> its bytes of the whole one (the rest of the whole blob left as it is)
inline void ff_slice_to_blob(const FfSlice& s, const uint8_t* src, const size_t srco[3], uint8_t* blob, const size_t dst[3]) {
    for (int r = 0; r < 2; ++r)
        std::memcpy(blob + dst[r] + (size_t) s.ff0 * s.gr, src + srco[r], (size_t) s.ffn * s.gr);
    for (size_t row = 0; row < (size_t) s.dim; ++row)
        std::memcpy(blob + dst[2] + row * s.dr + s.d_off, src + srco[2] + row * s.ds, s.ds);
}

}  // namespace strata::ds4
