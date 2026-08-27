"""Export one gear-search problem from ArkBuildPicker into a solver-ready blob.

Everything game-specific is resolved HERE, by importing ArkBuildPicker's own
modules: each warehouse item becomes a fixed 8-float stat vector for the chosen
wearer (percentages resolve against the raw base and round per source), the set
tables become small numeric arrays, and the damage objective becomes a handful
of constants. The C++/CUDA solvers then see a pure combinatorial problem:

    pick one item per part (6 parts) maximizing exact_damage(panel(combo)),
    subject to panel.spd >= spd_min and panel.crit_rate >= crit_min.

Outputs (into data/):
    problem.bin   binary blob the solvers mmap/fread (see src/common.h)
    problem.json  human-readable sidecar: item names/uids, metadata,
                  the CPU tool's own surrogate top-N for later comparison
    validation is embedded in problem.bin: 512 random combos evaluated by
    THIS script through ArkBuildPicker's exact code path.

Usage (from this directory):
    python export_problem.py 歐貝恩絲 --buffed --debuffed
    python export_problem.py 蓋兒 --spd-min 228 --crit-min 64
"""

import argparse
import json
import os
import random
import struct
import sys

STATS = ["atk", "def", "hp", "spd", "crit_rate", "crit_dmg", "hit", "res"]
PARTS = ["武器", "頭盔", "鎧甲", "項鍊", "戒指", "鞋子"]
N_VALIDATION = 512
MAGIC = b"GSV1"


def f32(x):
    """Round-trip through float32 so Python-side validation sees exactly the
    numbers the solvers will read."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def stage(tools_dir, args):
    sys.path.insert(0, tools_dir)
    import arkdata
    import effects
    import skills as skills_mod
    from panel import Roster
    import optimize

    roster = Roster()
    match = [c for c in roster.characters.values()
             if c["name_zh"] == args.character or c["static_id"] == args.character]
    if not match:
        sys.exit(f"characters.csv has no {args.character!r}")
    static_id = match[0]["static_id"]

    pool = optimize.Pool(potential=not args.current)
    search = optimize.Search(roster, pool, static_id)
    staged = pool.staged(search.raw)
    by_part = {p: [c for c in staged if c["item"]["part_zh"] == p] for p in PARTS}
    for p in PARTS:
        if not by_part[p]:
            sys.exit(f"warehouse has no legendary/reincarnated {p}")

    fixed = search._fixed_stats()
    raw = search.raw

    rows = [r for r in skills_mod.load() if r["name"] == match[0]["name_zh"]]
    if not rows:
        sys.exit(f"skills.csv has no rows for {match[0]['name_zh']}")
    picked = [r for r in rows if r["skill"].startswith("S3")] or rows
    spec = {
        "kind": "damage",
        "skills": [{"row": picked[-1], "weight": 1}],
        "atk_mult": 1 + effects.ATK_UP if args.buffed else 1.0,
        "target_def_mult": 1 - effects.DEF_DOWN if args.debuffed else 1.0,
        "target_debuffed": args.debuffed,
    }
    mods = {"arkdata": arkdata, "effects": effects, "skills_mod": skills_mod,
            "optimize": optimize, "roster": roster}
    return mods, static_id, match[0], by_part, fixed, raw, spec, search


def set_tables(arkdata, effects):
    """SET_* tables -> numeric arrays indexed by set_id (0 unused)."""
    n = max(arkdata.SET_ID.values()) + 1
    tables = []
    for sid in range(n):
        key = next((k for k, v in arkdata.SET_ID.items() if v == sid), None)
        pieces = arkdata.SET_PIECES.get(key, 99)
        stat_idx, of_base, amounts = -1, 0, [0.0, 0.0, 0.0]
        entry = arkdata.SET_STAT.get(key)
        if entry:
            stat, ob, tiers = entry
            stat_idx = STATS.index(stat) if stat in STATS else -1
            of_base = 1 if ob else 0
            for t, amt in tiers.items():
                amounts[{2: 0, 4: 1, 6: 2}[t]] = float(amt)
        tables.append({"key": key or "", "pieces": pieces, "stat_idx": stat_idx,
                       "of_base": of_base, "amounts": amounts})
    riptide = [effects.RIPTIDE_DMG.get(t, 0.0) for t in (2, 4, 6)]
    return tables, riptide, effects.RAGE_VS_DEBUFF, effects.PENETRATION_SINGLE


def evaluate(combo, vecs, set_ids, mods, fixed, raw, spec, spd_min, crit_min):
    """Exact objective for one combo, through ArkBuildPicker's own functions.
    This is the ground truth every solver must reproduce."""
    arkdata, optimize = mods["arkdata"], mods["optimize"]
    panel = dict(fixed)
    for part_i, item_i in enumerate(combo):
        for s_i, s in enumerate(STATS):
            panel[s] += vecs[part_i][item_i][s_i]
    fake = []
    for part_i, item_i in enumerate(combo):
        sid = set_ids[part_i][item_i]
        key = next((k for k, v in arkdata.SET_ID.items() if v == sid), "")
        fake.append({"set": key})
    for s, amt in arkdata.set_stats(fake, raw).items():
        if s in panel:
            panel[s] += amt
    panel["crit_rate"] = min(panel["crit_rate"], 100.0)
    panel["crit_dmg"] = min(panel["crit_dmg"], 350.0)
    feasible = panel["spd"] >= spd_min - 1e-6 and panel["crit_rate"] >= crit_min - 1e-6
    dmg = optimize.damage_value(panel, arkdata.active_sets(fake), spec)
    return dmg, feasible


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("character")
    ap.add_argument("--tools", default=r"C:\software\ArkBuildPicker\tools")
    ap.add_argument("--buffed", action="store_true")
    ap.add_argument("--debuffed", action="store_true")
    ap.add_argument("--current", action="store_true",
                    help="score gear as-is instead of at its +15 expectation")
    ap.add_argument("--spd-min", type=float, default=0.0)
    ap.add_argument("--crit-min", type=float, default=0.0)
    ap.add_argument("--cpu-tool-top", type=int, default=12,
                    help="also record the surrogate tool's own top-N (0 = skip)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    mods, static_id, char, by_part, fixed_d, raw_d, spec, search = stage(
        os.path.abspath(args.tools), args)
    arkdata = mods["arkdata"]

    # --- stage numeric arrays (round-tripped through f32) -------------------
    counts = [len(by_part[p]) for p in PARTS]
    vecs, set_ids, meta_items = [], [], []
    for p in PARTS:
        vv, ss, mm = [], [], []
        for c in by_part[p]:
            vv.append([f32(c["eff"][s]) for s in STATS])
            ss.append(c["item"]["set_id"])
            it = c["item"]
            mm.append({"uid": it["uid"], "name": it["name"], "set": it["set"],
                       "set_zh": it["set_zh"], "enhance": it["enhance"],
                       "item_level": it["item_level"], "rarity": it["rarity"],
                       "main_stat": it["main_stat"],
                       "holder_name": it.get("holder_name", "")})
        vecs.append(vv); set_ids.append(ss); meta_items.append(mm)
    fixed = [f32(fixed_d[s]) for s in STATS]
    raw4 = [f32(raw_d[s]) for s in ("atk", "def", "hp", "spd")]
    fixed_d32 = dict(zip(STATS, fixed))
    raw_d32 = dict(zip(("atk", "def", "hp", "spd"), raw4))
    tables, riptide, rage, pen_single = set_tables(arkdata, mods["effects"])

    row = spec["skills"][0]["row"]
    tgt = dict(mods["skills_mod"].DUMMY)
    target_def_scaled = tgt["def"] * spec.get("target_def_mult", 1.0)
    skills_bin = [{
        "rate": row["rate"], "pow": row["pow"], "enh": row["enh"],
        "bonus": row["bonus"], "hp_scale": row["hp_scale"],
        "def_scale": row["def_scale"], "target_hp_scale": row["target_hp_scale"],
        "pen": row["pen"], "fixed": row["fixed"],
        "weight": spec["skills"][0].get("weight", 1),
        "no_crit": 1 if row["no_crit"] else 0,
        "single": 0 if "全體" in (row.get("target") or "") else 1,
    }]

    space = 1
    for n in counts:
        space *= n

    # --- validation set through the exact Python path -----------------------
    rng = random.Random(42)
    val = []
    for _ in range(N_VALIDATION):
        combo = [rng.randrange(counts[i]) for i in range(6)]
        dmg, feas = evaluate(combo, vecs, set_ids, mods, fixed_d32, raw_d32,
                             spec, args.spd_min, args.crit_min)
        val.append((combo, dmg, feas))

    # --- the surrogate tool's own answer, for the accuracy story ------------
    cpu_tool = []
    if args.cpu_tool_top:
        import time
        t0 = time.perf_counter()
        results = search.run(spec, spd_min=args.spd_min, crit_min=args.crit_min,
                             top=args.cpu_tool_top)
        cpu_tool = [{"value": r["value"],
                     "uids": sorted(i["uid"] for i in r["items"])}
                    for r in results]
        cpu_tool_seconds = time.perf_counter() - t0

    # --- write problem.bin ---------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "problem.bin")
    with open(out, "wb") as fh:
        w = fh.write
        w(MAGIC)
        w(struct.pack("<6I", *counts))
        w(struct.pack("<8f", *fixed))
        w(struct.pack("<4f", *raw4))
        w(struct.pack("<2f", args.spd_min, args.crit_min))
        w(struct.pack("<I", len(tables)))
        for t in tables:
            w(struct.pack("<IiI3f", t["pieces"], t["stat_idx"], t["of_base"],
                          *t["amounts"]))
        w(struct.pack("<3f", *riptide))
        w(struct.pack("<2f", rage, pen_single))
        w(struct.pack("<I", 1 if spec["target_debuffed"] else 0))
        w(struct.pack("<3fI", spec["atk_mult"], 0.0, 0.0, 0))  # crit_add/crit_dmg_add/force_crit unused by optimize.main
        w(struct.pack("<f", 1.0))                              # base_other = 1 + extra_addends
        w(struct.pack("<3fi", tgt["hp"], target_def_scaled, tgt.get("crit_res", 0), 0))
        w(struct.pack("<I", len(skills_bin)))
        for s in skills_bin:
            w(struct.pack("<10f2I", s["rate"], s["pow"], s["enh"], s["bonus"],
                          s["hp_scale"], s["def_scale"], s["target_hp_scale"],
                          s["pen"], s["fixed"], s["weight"],
                          s["no_crit"], s["single"]))
        for part_i in range(6):
            for item_i in range(counts[part_i]):
                w(struct.pack("<8f", *vecs[part_i][item_i]))
                w(struct.pack("<I", set_ids[part_i][item_i]))
        w(struct.pack("<I", len(val)))
        for combo, dmg, feas in val:
            w(struct.pack("<6I", *combo))
            w(struct.pack("<dI", dmg, 1 if feas else 0))

    sidecar = {
        "character": char["name_zh"], "static_id": static_id,
        "objective": "damage", "buffed": args.buffed, "debuffed": args.debuffed,
        "potential": not args.current,
        "spd_min": args.spd_min, "crit_min": args.crit_min,
        "skill": {"name": row["skill"], "target": row.get("target", "")},
        "counts": dict(zip(PARTS, counts)), "search_space": space,
        "cpu_tool_top": cpu_tool,
        "cpu_tool_seconds": cpu_tool_seconds if args.cpu_tool_top else None,
        "items": {p: meta_items[i] for i, p in enumerate(PARTS)},
    }
    with open(os.path.join(args.out, "problem.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=1)

    print(f"character      {char['name_zh']} ({static_id})  skill {row['skill']}")
    print(f"pool           {counts}  -> {space:.3e} combos")
    print(f"validation     {len(val)} combos through the exact Python path")
    if cpu_tool:
        print(f"cpu tool       top-{len(cpu_tool)} best value {cpu_tool[0]['value']:,.0f}"
              f"  in {cpu_tool_seconds:.1f}s (surrogate + rerank)")
    print(f"wrote          {out} ({os.path.getsize(out):,} bytes) + problem.json")


if __name__ == "__main__":
    main()
