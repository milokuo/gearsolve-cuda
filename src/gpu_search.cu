// CUDA exact solver, v1 (naive): one grid-stride thread walk over the whole
// 3.7e12-combo space, one eval_combo<float> per combination.
//
//   gpu_search validate                        eval the embedded ground-truth
//                                              combos on device, compare
//   gpu_search search [--threshold X] [--limit N]
//        every feasible combo scoring >= X is appended to a candidate buffer
//        (atomic); the host re-scores candidates in double and prints top-16.
//        The global maximum is also tracked with an atomicMax so an overflow
//        or a too-high threshold is always detected, never silent.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "eval.h"

using namespace gs;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err_), __FILE__, __LINE__);      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

constexpr int TOP_K = 16;
constexpr uint32_t BUFFER_CAP = 1u << 22;   // 4M candidates * 12 B = 48 MB

__constant__ Ctx d_ctx;
__constant__ uint32_t d_counts[N_PARTS];
__constant__ uint32_t d_offsets[N_PARTS];

struct Candidate { float dmg; uint32_t lo, hi; };   // 64-bit combo index split

// ---------------------------------------------------------------------------
__global__ void validate_kernel(const float* __restrict__ stats,
                                const int32_t* __restrict__ set_id,
                                const uint32_t* __restrict__ val_idx,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto r = eval_combo<float>(d_ctx, stats, set_id, d_offsets, val_idx + (size_t)i * N_PARTS);
    out[i] = r.dmg;
}

// ---------------------------------------------------------------------------
// One slice [start, end) of the space per launch: Windows' display watchdog
// (TDR) kills any single kernel that runs for ~2 s, so the host loops over
// ~2e9-combo slices instead of launching one multi-minute kernel.
__global__ void search_kernel(const float* __restrict__ stats,
                              const int32_t* __restrict__ set_id,
                              uint64_t start, uint64_t end, float threshold,
                              Candidate* __restrict__ buffer, uint32_t* __restrict__ buf_count,
                              unsigned int* __restrict__ best_bits,
                              unsigned long long* __restrict__ feasible_count) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long local_feasible = 0;
    float local_best = -1.0f;

    for (uint64_t lin = start + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         lin < end; lin += stride) {
        // mixed-radix decode, last part fastest (matches the CPU loop order)
        uint32_t idx[N_PARTS];
        uint64_t rest = lin;
        for (int p = N_PARTS - 1; p > 0; p--) {
            uint32_t n = d_counts[p];
            idx[p] = (uint32_t)(rest % n);
            rest /= n;
        }
        idx[0] = (uint32_t)rest;

        auto r = eval_combo<float>(d_ctx, stats, set_id, d_offsets, idx);
        if (!r.feasible) continue;
        local_feasible++;
        local_best = fmaxf(local_best, r.dmg);
        if (r.dmg >= threshold) {
            uint32_t slot = atomicAdd(buf_count, 1u);
            if (slot < BUFFER_CAP) {
                buffer[slot].dmg = r.dmg;
                buffer[slot].lo = (uint32_t)(lin & 0xffffffffu);
                buffer[slot].hi = (uint32_t)(lin >> 32);
            }
        }
    }
    if (local_feasible) atomicAdd(feasible_count, local_feasible);
    if (local_best > 0.0f)
        atomicMax(best_bits, __float_as_uint(local_best));  // positive floats: bit order = value order
}

// ---------------------------------------------------------------------------
static void decode(uint64_t lin, const uint32_t* counts, uint32_t* idx) {
    for (int p = N_PARTS - 1; p > 0; p--) { idx[p] = (uint32_t)(lin % counts[p]); lin /= counts[p]; }
    idx[0] = (uint32_t)lin;
}

int main(int argc, char** argv) {
    const char* path = "data/problem.bin";
    std::string mode = argc > 1 ? argv[1] : "validate";
    float threshold = 0.0f;
    uint64_t limit = 0;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--limit") && i + 1 < argc) limit = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--problem") && i + 1 < argc) path = argv[++i];
    }
    Problem p = load_problem(path);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device         %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::printf("problem        %u items, space %.3e\n", p.total_items, (double)p.space);

    CUDA_CHECK(cudaMemcpyToSymbol(d_ctx, &p.ctx, sizeof(Ctx)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_counts, p.counts, sizeof(p.counts)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_offsets, p.offsets, sizeof(p.offsets)));
    float* d_stats; int32_t* d_set;
    CUDA_CHECK(cudaMalloc(&d_stats, p.stats.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_set, p.set_id.size() * 4));
    CUDA_CHECK(cudaMemcpy(d_stats, p.stats.data(), p.stats.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_set, p.set_id.data(), p.set_id.size() * 4, cudaMemcpyHostToDevice));

    if (mode == "validate") {
        int n = (int)p.validation.size();
        std::vector<uint32_t> vidx(n * N_PARTS);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < N_PARTS; k++) vidx[i * N_PARTS + k] = p.validation[i].idx[k];
        uint32_t* d_vidx; float* d_out;
        CUDA_CHECK(cudaMalloc(&d_vidx, vidx.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_out, n * 4));
        CUDA_CHECK(cudaMemcpy(d_vidx, vidx.data(), vidx.size() * 4, cudaMemcpyHostToDevice));
        validate_kernel<<<(n + 255) / 256, 256>>>(d_stats, d_set, d_vidx, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(n);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
        double max_rel = 0; int worst = -1;
        for (int i = 0; i < n; i++) {
            double rel = std::abs((double)out[i] - p.validation[i].expected)
                       / std::max(1.0, std::abs(p.validation[i].expected));
            if (rel > max_rel) { max_rel = rel; worst = i; }
        }
        std::printf("validation     %d combos, GPU f32 vs python f64: max rel err %.3e (combo %d)\n",
                    n, max_rel, worst);
        std::printf("  %s\n", max_rel < 5e-4 ? "PASS" : "FAIL");
        return max_rel < 5e-4 ? 0 : 1;
    }

    // ---- search ------------------------------------------------------------
    uint64_t space = limit ? std::min(limit, p.space) : p.space;
    Candidate* d_buf; uint32_t* d_count; unsigned int* d_best; unsigned long long* d_feas;
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(Candidate) * BUFFER_CAP));
    CUDA_CHECK(cudaMalloc(&d_count, 4));
    CUDA_CHECK(cudaMalloc(&d_best, 4));
    CUDA_CHECK(cudaMalloc(&d_feas, 8));
    CUDA_CHECK(cudaMemset(d_count, 0, 4));
    CUDA_CHECK(cudaMemset(d_best, 0, 4));
    CUDA_CHECK(cudaMemset(d_feas, 0, 8));

    int block = 256;
    int grid = prop.multiProcessorCount * 64;
    const uint64_t chunk = 1ull << 31;          // ~2.1e9 combos per launch, well under TDR
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    uint64_t n_chunks = (space + chunk - 1) / chunk;
    for (uint64_t c0 = 0, ci = 0; c0 < space; c0 += chunk, ci++) {
        uint64_t c1 = std::min(space, c0 + chunk);
        search_kernel<<<grid, block>>>(d_stats, d_set, c0, c1, threshold,
                                       d_buf, d_count, d_best, d_feas);
        if ((ci & 63) == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());   // also surfaces async errors early
            std::fprintf(stderr, "\r  chunk %llu / %llu",
                         (unsigned long long)ci, (unsigned long long)n_chunks);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::fprintf(stderr, "\r%40s\r", "");
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    uint32_t n_cand; unsigned int best_bits; unsigned long long feasible;
    CUDA_CHECK(cudaMemcpy(&n_cand, d_count, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&best_bits, d_best, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&feasible, d_feas, 8, cudaMemcpyDeviceToHost));
    float best = 0; std::memcpy(&best, &best_bits, 4);

    std::printf("searched       %.3e combos in %.2f s   (%.3e combos/s)\n",
                (double)space, ms / 1e3, (double)space / (ms / 1e3));
    std::printf("feasible       %llu\n", feasible);
    std::printf("global best    %.3f (f32)\n", best);
    if (n_cand > BUFFER_CAP) {
        std::printf("BUFFER OVERFLOW: %u candidates >= threshold %.1f; raise --threshold\n",
                    n_cand, threshold);
        return 1;
    }
    std::printf("candidates     %u >= threshold %.1f\n", n_cand, threshold);

    std::vector<Candidate> cand(n_cand);
    CUDA_CHECK(cudaMemcpy(cand.data(), d_buf, sizeof(Candidate) * n_cand, cudaMemcpyDeviceToHost));
    std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.dmg > b.dmg; });

    // exact double re-score of the survivors, so FP32 near-ties cannot misrank
    int n_out = std::min<size_t>(cand.size(), 4096);
    std::vector<std::pair<double, uint64_t>> exact(n_out);
    for (int i = 0; i < n_out; i++) {
        uint64_t lin = ((uint64_t)cand[i].hi << 32) | cand[i].lo;
        uint32_t idx[N_PARTS];
        decode(lin, p.counts, idx);
        auto r = eval_combo<double>(p.ctx, p.stats.data(), p.set_id.data(), p.offsets, idx);
        exact[i] = { r.dmg, lin };
    }
    std::sort(exact.begin(), exact.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; i < std::min(TOP_K, n_out); i++) {
        uint32_t idx[N_PARTS];
        decode(exact[i].second, p.counts, idx);
        std::printf("  #%-2d %12.3f   [%u %u %u %u %u %u]\n", i + 1, exact[i].first,
                    idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
    }
    return 0;
}
