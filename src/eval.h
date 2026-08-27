// The exact objective, shared verbatim by the CPU reference and the CUDA
// kernels: one template instantiated as double (ground truth, matches the
// Python exporter to ~1e-12) and as float (what the GPU runs).
//
// Faithfulness notes, learned the hard way from the game's own rounding:
//  - every percent-of-base set bonus is rounded PER SOURCE, and Python's
//    round() is round-half-to-EVEN -- so this uses rint (default FE_TONEAREST),
//    not round(), whose halfway cases go away from zero.
//  - crit_rate caps at 100 and crit_dmg at 350 on the panel itself; buffs at
//    damage time are capped again by the same limits.
#pragma once
#include <cmath>
#include "common.h"

#if defined(__CUDACC__)
#define GS_HD __host__ __device__ __forceinline__
#else
#define GS_HD inline
#endif

namespace gs {

GS_HD double gs_rint(double x) { return rint(x); }
GS_HD float  gs_rint(float x)  { return rintf(x); }

template <typename T>
struct EvalResult {
    T dmg;
    T spd;
    T crit_rate;
    bool feasible;
};

// stats: flat [total_items][8]; set_id: flat [total_items];
// offsets: first flat index of each part; idx: local item index per part.
template <typename T>
GS_HD EvalResult<T> eval_combo(const Ctx& ctx,
                               const float* stats, const int32_t* set_id,
                               const uint32_t* offsets, const uint32_t* idx) {
    // --- panel: fixed + the six item vectors (percentages were resolved and
    // rounded per source at export time; items are plain adds here) ----------
    T panel[N_STATS];
    for (int s = 0; s < N_STATS; s++) panel[s] = (T)ctx.fixed[s];
    uint32_t flat[N_PARTS];
    int32_t sid[N_PARTS];
    for (int p = 0; p < N_PARTS; p++) {
        flat[p] = offsets[p] + idx[p];
        sid[p] = set_id[flat[p]];
        const float* v = stats + (size_t)flat[p] * N_STATS;
        for (int s = 0; s < N_STATS; s++) panel[s] += (T)v[s];
    }

    // --- set effects: the only coupling between slots -----------------------
    T other = (T)ctx.base_other;      // damage-multiplier addends land here
    T pen_extra = (T)0;               // extra DEF penetration (single target)
    for (int i = 0; i < N_PARTS; i++) {
        int32_t s = sid[i];
        if (s <= 0) continue;
        bool seen = false;
        for (int j = 0; j < i; j++) seen |= (sid[j] == s);
        if (seen) continue;
        int count = 0;
        for (int j = i; j < N_PARTS; j++) count += (sid[j] == s);
        const SetEntry& e = ctx.sets[s];
        if ((uint32_t)count < e.pieces) continue;

        if (e.stat_idx >= 0) {        // stat-granting set: largest met tier
            for (int t = 2; t >= 0; t--) {
                float amt = e.amounts[t];
                if (amt != 0.0f && count >= 2 * (t + 1)) {
                    T add = e.of_base
                        ? gs_rint((T)ctx.raw4[e.stat_idx] * (T)amt / (T)100)
                        : (T)amt;
                    panel[e.stat_idx] += add;
                    break;
                }
            }
        }
        if (s == 17) {                // riptide: damage addend by tier
            for (int t = 2; t >= 0; t--)
                if (count >= 2 * (t + 1)) { other += (T)ctx.riptide[t]; break; }
        }
        if (s == 13 && ctx.target_debuffed)   // rage vs a debuffed target
            other += (T)ctx.rage_addend;
        if (s == 14)                  // penetration: single-target DEF pierce
            pen_extra = (T)ctx.pen_single;
    }

    // --- panel caps and the build constraints -------------------------------
    if (panel[4] > (T)100) panel[4] = (T)100;
    if (panel[5] > (T)350) panel[5] = (T)350;
    EvalResult<T> r;
    r.spd = panel[3];
    r.crit_rate = panel[4];
    r.feasible = panel[3] >= (T)ctx.spd_min - (T)1e-6
              && panel[4] >= (T)ctx.crit_min - (T)1e-6;

    // --- exact damage (skills.damage + optimize.damage_value) ---------------
    T atk = panel[0] * (T)ctx.atk_mult;
    T cr = panel[4] + (T)ctx.crit_add;
    if (cr > (T)100) cr = (T)100;
    if (ctx.force_crit) cr = (T)100;
    T cd = panel[5] + (T)ctx.crit_dmg_add;
    if (cd > (T)350) cd = (T)350;

    T total = (T)0;
    for (int k = 0; k < ctx.n_skills; k++) {
        const Skill& sk = ctx.skills[k];
        T tdef = (T)ctx.target_def;
        if (sk.single) tdef *= (T)1 - pen_extra;

        T flat_dmg = panel[2] * (T)sk.hp_scale + panel[1] * (T)sk.def_scale
                   + (T)ctx.target_hp * (T)sk.target_hp_scale;
        T base = atk * (T)sk.rate + flat_dmg;

        T miss = ctx.edge < 0 ? (T)0.5 : (T)0;
        T crit = (T)0;
        if (!sk.no_crit) {
            crit = (cr + (ctx.edge > 0 ? (T)15 : (T)0) - (T)ctx.target_crit_res) / (T)100;
            if (crit < (T)0) crit = (T)0;
            if (crit > (T)1) crit = (T)1;
        }
        T strike = ctx.edge > 0 ? (T)0.8 : (T)0.3;
        T landed = crit * (cd / (T)100)
                 + ((T)1 - crit) * (strike * (T)1.3 + ((T)1 - strike));
        T hit = miss * (T)0.75 + ((T)1 - miss) * landed;

        T outgoing = base * ((T)1.871 * (T)sk.pow)
                   * (ctx.edge > 0 ? (T)1.1 : (T)1.0)
                   * hit * (T)sk.enh * (T)sk.bonus * other;
        T defence = tdef * ((T)1 - (T)sk.pen);
        total += (T)sk.weight * (outgoing / (defence / (T)300 + (T)1) + (T)sk.fixed);
    }
    r.dmg = total;
    return r;
}

} // namespace gs
