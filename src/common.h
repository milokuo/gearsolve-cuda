// Problem blob loader. Layout must match export_problem.py exactly.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gs {

constexpr int N_PARTS = 6;
constexpr int N_STATS = 8;             // atk, def, hp, spd, crit_rate, crit_dmg, hit, res
constexpr int MAX_SETS = 32;
constexpr int MAX_SKILLS = 4;

struct SetEntry {
    uint32_t pieces;                   // copies needed for the effect (2 or 4)
    int32_t stat_idx;                  // index into STATS, -1 = no stat bonus
    uint32_t of_base;                  // 1: percent of the wearer's raw base
    float amounts[3];                  // value at thresholds 2 / 4 / 6, 0 = none
};

struct Skill {
    float rate, pow, enh, bonus;
    float hp_scale, def_scale, target_hp_scale;
    float pen, fixed, weight;
    uint32_t no_crit, single;
};

// Everything the evaluator needs besides the item arrays. POD, device-copyable.
struct Ctx {
    float fixed[N_STATS];
    float raw4[4];                     // atk, def, hp, spd raw base
    float spd_min, crit_min;
    int32_t n_sets;
    SetEntry sets[MAX_SETS];
    float riptide[3];                  // damage addend at 2 / 4 / 6 pieces
    float rage_addend, pen_single;
    uint32_t target_debuffed;
    float atk_mult, crit_add, crit_dmg_add;
    uint32_t force_crit;
    float base_other;
    float target_hp, target_def, target_crit_res;
    int32_t edge;
    int32_t n_skills;
    Skill skills[MAX_SKILLS];
};

struct Validation {
    uint32_t idx[N_PARTS];
    double expected;
    uint32_t feasible;
};

struct Problem {
    Ctx ctx{};
    uint32_t counts[N_PARTS]{};
    uint32_t offsets[N_PARTS]{};       // prefix offsets into the flat item arrays
    uint32_t total_items = 0;
    std::vector<float> stats;          // total_items * N_STATS
    std::vector<int32_t> set_id;       // total_items
    std::vector<Validation> validation;
    uint64_t space = 1;
};

inline void fail(const char* msg) { std::fprintf(stderr, "error: %s\n", msg); std::exit(1); }

inline Problem load_problem(const char* path) {
    FILE* fh = std::fopen(path, "rb");
    if (!fh) fail("cannot open problem.bin");
    auto rd = [&](void* dst, size_t n) { if (std::fread(dst, 1, n, fh) != n) fail("truncated problem.bin"); };

    char magic[4];
    rd(magic, 4);
    if (std::memcmp(magic, "GSV1", 4) != 0) fail("bad magic");

    Problem p;
    rd(p.counts, sizeof(p.counts));
    rd(p.ctx.fixed, sizeof(p.ctx.fixed));
    rd(p.ctx.raw4, sizeof(p.ctx.raw4));
    rd(&p.ctx.spd_min, 4); rd(&p.ctx.crit_min, 4);

    uint32_t n_sets; rd(&n_sets, 4);
    if (n_sets > MAX_SETS) fail("too many sets");
    p.ctx.n_sets = (int32_t)n_sets;
    for (uint32_t i = 0; i < n_sets; i++) rd(&p.ctx.sets[i], sizeof(SetEntry));
    rd(p.ctx.riptide, sizeof(p.ctx.riptide));
    rd(&p.ctx.rage_addend, 4); rd(&p.ctx.pen_single, 4);
    rd(&p.ctx.target_debuffed, 4);
    rd(&p.ctx.atk_mult, 4); rd(&p.ctx.crit_add, 4); rd(&p.ctx.crit_dmg_add, 4);
    rd(&p.ctx.force_crit, 4);
    rd(&p.ctx.base_other, 4);
    rd(&p.ctx.target_hp, 4); rd(&p.ctx.target_def, 4); rd(&p.ctx.target_crit_res, 4);
    rd(&p.ctx.edge, 4);

    uint32_t n_skills; rd(&n_skills, 4);
    if (n_skills > MAX_SKILLS) fail("too many skills");
    p.ctx.n_skills = (int32_t)n_skills;
    for (uint32_t i = 0; i < n_skills; i++) rd(&p.ctx.skills[i], sizeof(Skill));

    p.total_items = 0;
    for (int i = 0; i < N_PARTS; i++) { p.offsets[i] = p.total_items; p.total_items += p.counts[i]; }
    p.stats.resize((size_t)p.total_items * N_STATS);
    p.set_id.resize(p.total_items);
    for (uint32_t i = 0; i < p.total_items; i++) {
        rd(&p.stats[(size_t)i * N_STATS], N_STATS * 4);
        rd(&p.set_id[i], 4);
    }

    uint32_t n_val; rd(&n_val, 4);
    p.validation.resize(n_val);
    for (uint32_t i = 0; i < n_val; i++) {
        rd(p.validation[i].idx, sizeof(uint32_t) * N_PARTS);
        rd(&p.validation[i].expected, 8);
        rd(&p.validation[i].feasible, 4);
    }
    std::fclose(fh);

    p.space = 1;
    for (int i = 0; i < N_PARTS; i++) p.space *= p.counts[i];
    return p;
}

} // namespace gs
