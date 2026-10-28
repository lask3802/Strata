// tests/dsv41/bc_pull_test.cu - DS4_BC_ONCE=3's pull (bc_pull, src/dsv41/ds4_bc_pull.cuh): over a run of layers, each
// layer's batch bounds written into a mapped, pinned host table (bc_pin as hmap makes it: cudaHostAllocMapped |
// cudaHostAllocPortable, not write-combined) and pulled into the device table (bc_dev) on a stream, the layer's runs
// of regions (bc_runs: one, or two where they wrap) in one launch, in flow_bc_upfront's order - the stream
// synchronized before the host writes (the layer's bounds wait), the launch issued right after them with nothing
// between.  After each layer the
// whole device table must be the host's mirror: the layer's regions the host's values, every other region what it held
// (the layers before, or the sentinel).  Runs of up to all bc_n regions (more ints than a block's threads), runs of
// one region, an empty layer (nothing launched).  Needs a GPU, no model.  bc_pull_test
#include "../../src/dsv41/ds4_bc_pull.cuh"
#include "../../src/dsv41/ds4_flow_batches.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace sd = strata::ds4;

namespace {

int g_fail = 0, g_cases = 0;
std::mt19937 rng(331);

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

/// layers of a ring of R slots (bc_w = R + 1 ints a region) and bc_n regions; each layer nb batches from n_batch on,
// their runs pulled in one launch
void run(int R, int bc_n, int layers, const char* what) {
    const size_t w = (size_t) R + 1, all = (size_t) bc_n * w;
    int* pin = nullptr;
    CU(cudaHostAlloc((void**) &pin, all * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(pin, 0, all * 4);
    void* m = nullptr;
    CU(cudaHostGetDevicePointer(&m, pin, 0));
    int* map = (int*) m;
    int* dev = nullptr;
    CU(cudaMalloc((void**) &dev, all * 4));
    CU(cudaMemset(dev, 0xff, all * 4));   // (the sentinel: -1 everywhere)
    std::vector<int> want(all, -1), got(all);
    cudaStream_t s;
    CU(cudaStreamCreate(&s));
    uint64_t n_batch = (uint64_t) (rng() % 100000);
    bool ok = true, wrapped = false, empty = false, full = false;
    size_t pulled = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const int pick = (int) (rng() % 8);   // (an empty layer, all bc_n regions, else any count)
        const size_t nb = pick == 0 ? 0 : pick == 1 ? (size_t) bc_n : rng() % ((size_t) bc_n + 1);
        CU(cudaStreamSynchronize(s));   // (the layer's bounds wait: s done with every region before the host writes)
        for (size_t j = 0; j < nb; ++j) {
            const size_t reg = sd::bc_region(n_batch + j, bc_n);
            for (size_t q = 0; q < w; ++q) {
                const int v = (int) (rng() & 0x7fffffff);
                pin[reg * w + q] = v;
                want[reg * w + q] = v;
            }
        }
        sd::BcRun r[2];
        const int nr = sd::bc_runs(n_batch, nb, bc_n, r);
        CU(sd::bc_pull(map, dev, w, r, nr, s));   // (one launch; no sync, no fence between the host's writes and it)
        for (int i = 0; i < nr; ++i) pulled += r[i].n * w;
        wrapped = wrapped || nr == 2;
        empty = empty || nb == 0;
        full = full || nb == (size_t) bc_n;
        n_batch += nb;
        CU(cudaMemcpyAsync(got.data(), dev, all * 4, cudaMemcpyDeviceToHost, s));
        CU(cudaStreamSynchronize(s));
        ok = ok && got == want;
    }
    ++g_cases;
    const bool cover = wrapped && empty && full;
    if (!ok || !cover) ++g_fail;
    std::printf("%-44s R %2d, bc_n %3d, %4d layers, %8zu ints pulled%s%s%s -> %s\n", what, R, bc_n, layers, pulled,
                wrapped ? ", wraps" : "", empty ? ", empty layers" : "", full ? ", all bc_n" : "",
                ok && cover ? "ok" : !ok ? "FAIL (the device table is not the host's)" : "FAIL (a case not met)");
    CU(cudaStreamDestroy(s));
    CU(cudaFree(dev));
    CU(cudaFreeHost(pin));
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU\n");
        return 1;
    }
    CU(sd::bc_pull(nullptr, nullptr, 0, nullptr, 0, nullptr));   // (no runs: nothing launched, no error)
    run(40, 256 + 2, 400, "the engine's shape (a 40-slot ring, 256 experts)");
    run(1, 3, 400, "a ring of 1, 3 regions (runs under a warp)");
    run(255, 9, 200, "a region of 256 ints (a block's threads)");
    run(300, 40, 200, "regions longer than a block's threads");
    std::printf("%d of %d cases failed\n", g_fail, g_cases);
    return g_fail != 0;
}
