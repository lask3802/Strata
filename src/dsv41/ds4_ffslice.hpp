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

/// Layer::ep_ff0 / ep_ffn as the load leaves them: ep_ffn -1 (the default) is the whole n_ff
inline void ep_resolve(int& ff0, int& ffn, int n_ff) {
    if (ffn < 0) {
        ff0 = 0;
        ffn = n_ff;
    }
}
/// (U2's rule, kept for its test; the engine no longer calls it - H' gives each cpu_experts call its range by the pick's
/// lane, ds4_rx_split.hpp, so the host-routed path and the fallback never take a layer's slice from its fields)
/// The n_ff range a call of the CPU lane computes: the layer's EP slice in an EP window (ep: decode, a draft's
/// verification - both PCs' slices sum to the whole), the whole n_ff in every other (the prompt's windows and chunks,
/// feed_mark's last token, calibrate_paths: the main's alone until prompts split too), whatever the layer's fields say.
/// The window's kind, not the lane it takes (serve_lane or the host-routed path): the lane cannot change the sum
inline void ep_range(bool ep, int ep_ff0, int ep_ffn, int n_ff, int& ff0, int& ffn) {
    ff0 = 0;
    ffn = n_ff;
    if (!ep) return;
    ff0 = ep_ff0;
    ffn = ep_ffn;
    ep_resolve(ff0, ffn, n_ff);   // (a layer the load did not see: whole)
}
/// Why a layer cannot take the n_ff slice [ff0, ff0 + ffn) of its routed experts (nullptr: it can, or it has none).  Only
/// the GGUF i-quant CPU lane (cpu_experts_iq) computes a slice; any other path computes an expert whole, which the other
/// PC's part would then add a second time - so a layer with one of them is refused at setup until it takes slices too.
/// So are the drafter's layers (mtp: they run on the main alone), a remote stage's worker (stage_worker: the link does not
/// say a window's kind) and STRATA_DS4_DISK_SKIP_W (disk_skip: it drops experts by what is in this PC's RAM, which the
/// other PC's differs from).  tagged (H', DS4_EP_K: the lane slices only a pick it sent tagged slice in the same request,
/// ds4_rx_split.hpp): a VRAM tier or the store's GPU-visible slots are no reason - their picks are never sent, so never
/// sliced; computed whole on the device, they are added once
inline const char* ep_slice_refusal(int ff0, int ffn, int n_ff, bool iq_cpu_lane, bool vram_tier, bool store_zc, bool mtp,
                                    bool stage_worker, bool disk_skip, bool tagged = false) {
    ep_resolve(ff0, ffn, n_ff);   // (fields the load has not resolved yet: -1 is the whole)
    if (ff0 == 0 && ffn == n_ff) return nullptr;
    if (n_ff % 256 != 0 || ff0 < 0 || ffn <= 0 || ff0 % 256 != 0 || ffn % 256 != 0 || ff0 + ffn > n_ff)
        return "not whole 256-value blocks inside n_ff";
    if (mtp) return "a drafter (DSpark) layer: the main runs them alone, whole";
    if (stage_worker) return "a remote stage's worker: the link does not carry a window's kind (EP or whole) yet";
    if (disk_skip) return "STRATA_DS4_DISK_SKIP_W skips experts by this PC's RAM, which the other PC's differs from";
    if (!iq_cpu_lane) return "its CPU lane computes these experts whole (MXFP4 / EXL3: only the GGUF i-quants take slices)";
    if (vram_tier && !tagged) return "its VRAM tier computes experts whole (--vram-experts 0 for the layers that take slices)";
    if (store_zc && !tagged) return "the RAM store's GPU-visible slots are computed whole on the device";
    return nullptr;
}

}  // namespace strata::ds4
