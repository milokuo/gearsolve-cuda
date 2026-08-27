// CPU exact solver: the honest baseline the GPU has to beat.
//
//   reference validate                    check eval_combo against the Python
//                                         ground truth embedded in problem.bin
//   reference search [--pairs N] [--threads N] [--f32]
//                                         exhaustive exact top-16 over the full
//                                         space (or N weapon*helmet pairs,
//                                         evenly strided, for a timed sample)
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"
#include "eval.h"

using namespace gs;

constexpr int TOP_K = 16;

struct Hit { double dmg; uint32_t idx[N_PARTS]; };

static void push_top(std::vector<Hit>& top, const Hit& h) {
    if ((int)top.size() < TOP_K) {
        top.push_back(h);
        std::sort(top.begin(), top.end(), [](const Hit& a, const Hit& b) { return a.dmg > b.dmg; });
    } else if (h.dmg > top.back().dmg) {
        top.back() = h;
        std::sort(top.begin(), top.end(), [](const Hit& a, const Hit& b) { return a.dmg > b.dmg; });
    }
}

static int validate(const Problem& p) {
    double max_rel64 = 0, max_rel32 = 0;
    int bad64 = 0, bad_feas = 0;
    for (const auto& v : p.validation) {
        auto r64 = eval_combo<double>(p.ctx, p.stats.data(), p.set_id.data(), p.offsets, v.idx);
        auto r32 = eval_combo<float>(p.ctx, p.stats.data(), p.set_id.data(), p.offsets, v.idx);
        double rel64 = std::abs(r64.dmg - v.expected) / std::max(1.0, std::abs(v.expected));
        double rel32 = std::abs((double)r32.dmg - v.expected) / std::max(1.0, std::abs(v.expected));
        max_rel64 = std::max(max_rel64, rel64);
        max_rel32 = std::max(max_rel32, rel32);
        // Residual vs the Python tool is f32 quantization of the exported
        // constants (skill rates etc.), not math drift -- see README.
        if (rel64 > 1e-6) bad64++;
        if ((r64.feasible ? 1u : 0u) != v.feasible) bad_feas++;
    }
    std::printf("validation     %zu combos\n", p.validation.size());
    std::printf("  f64 vs python   max rel err %.3e   (%d beyond 1e-6)\n", max_rel64, bad64);
    std::printf("  f32 vs python   max rel err %.3e\n", max_rel32);
    std::printf("  feasibility     %d mismatches\n", bad_feas);
    bool ok = bad64 == 0 && bad_feas == 0 && max_rel32 < 5e-4;
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

template <typename T>
static void search(const Problem& p, int pairs_wanted, int threads) {
    const uint32_t* c = p.counts;
    const uint64_t all_pairs = (uint64_t)c[0] * c[1];
    const uint64_t inner = (uint64_t)c[2] * c[3] * c[4] * c[5];
    uint64_t pairs = pairs_wanted > 0 ? std::min<uint64_t>(pairs_wanted, all_pairs) : all_pairs;
    // Evenly strided pair sample so a timed subset still sees the whole pool.
    double stride = (double)all_pairs / (double)pairs;

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
    std::vector<std::vector<Hit>> tops;
    uint64_t feasible = 0;
    auto t0 = std::chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
#ifdef _OPENMP
#pragma omp single
#endif
        tops.resize(nth);
        std::vector<Hit> top;
        uint64_t local_feasible = 0;
        uint32_t idx[N_PARTS];

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (int pi = 0; pi < (int)pairs; pi++) {
            uint64_t pair = (uint64_t)((double)pi * stride);
            idx[0] = (uint32_t)(pair / c[1]);
            idx[1] = (uint32_t)(pair % c[1]);
            for (idx[2] = 0; idx[2] < c[2]; idx[2]++)
             for (idx[3] = 0; idx[3] < c[3]; idx[3]++)
              for (idx[4] = 0; idx[4] < c[4]; idx[4]++)
               for (idx[5] = 0; idx[5] < c[5]; idx[5]++) {
                    auto r = eval_combo<T>(p.ctx, p.stats.data(), p.set_id.data(), p.offsets, idx);
                    if (!r.feasible) continue;
                    local_feasible++;
                    if ((int)top.size() < TOP_K || (double)r.dmg > top.back().dmg) {
                        Hit h; h.dmg = (double)r.dmg;
                        std::memcpy(h.idx, idx, sizeof(h.idx));
                        push_top(top, h);
                    }
               }
        }
        tops[tid] = std::move(top);
#ifdef _OPENMP
#pragma omp critical
#endif
        feasible += local_feasible;
    }

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    uint64_t done = pairs * inner;

    std::vector<Hit> merged;
    for (auto& t : tops) for (auto& h : t) push_top(merged, h);

    std::printf("searched       %llu combos (%llu of %llu pairs) in %.2f s\n",
                (unsigned long long)done, (unsigned long long)pairs,
                (unsigned long long)all_pairs, sec);
    std::printf("throughput     %.3e combos/s   full space ETA %.1f min\n",
                done / sec, (double)p.space / (done / sec) / 60.0);
    std::printf("feasible       %llu\n", (unsigned long long)feasible);
    for (size_t i = 0; i < merged.size(); i++) {
        const auto& h = merged[i];
        std::printf("  #%-2zu %12.3f   [%u %u %u %u %u %u]\n", i + 1, h.dmg,
                    h.idx[0], h.idx[1], h.idx[2], h.idx[3], h.idx[4], h.idx[5]);
    }
}

int main(int argc, char** argv) {
    const char* path = "data/problem.bin";
    std::string mode = argc > 1 ? argv[1] : "validate";
    int pairs = 0, threads = 0;
    bool f32 = false;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--pairs") && i + 1 < argc) pairs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--f32")) f32 = true;
        else if (!std::strcmp(argv[i], "--problem") && i + 1 < argc) path = argv[++i];
    }
    Problem p = load_problem(path);
    std::printf("problem        %u items, space %.3e, spd_min %.0f crit_min %.0f\n",
                p.total_items, (double)p.space, p.ctx.spd_min, p.ctx.crit_min);
    if (mode == "validate") return validate(p);
    if (mode == "search") { f32 ? search<float>(p, pairs, threads) : search<double>(p, pairs, threads); return 0; }
    fail("mode must be validate or search");
}
