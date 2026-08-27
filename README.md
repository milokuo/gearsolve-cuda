# gearsolve-cuda

Exhaustive-exact GPU search over a 3.7×10¹² gear-combination space — the
brute-force answer to a problem I previously had to approximate on CPU.

## The problem

My gear optimizer for a turn-based RPG ([ArkBuildPicker]) picks the best six
equipment pieces for one character. Per-item stats are additive (every
percentage resolves against the character's raw base, rounded per source), but
**set bonuses couple the slots** and **the damage objective is nonlinear**
(crit caps, expected-hit mixing, DEF mitigation), so the CPU tool searches a
*linearized surrogate* — the damage gradient at a reference panel — inside each
enumerated set plan, prunes each slot to ~15–30 candidates, and exact-rescores
only the top 120 finalists.

That design exists because the exact exhaustive search is out of reach on CPU:

| solver | exact? | full-space result |
|---|---|---|
| Python tool (surrogate + rerank) | no — linearized | 0.7 s |
| C++ naive, 1 thread | yes | ~43 h (measured 2.4×10⁷ evals/s) |
| C++ naive, OpenMP ×16 | yes | ~14 h (measured 7.4×10⁷ evals/s) |
| CUDA v1 naive | yes | 199.6 s |
| CUDA v2 hoisted partials | yes | 165.1 s |
| CUDA v3 + admissible pruning | yes | 0.51 s |
| C++ pruned, 1 thread | yes | 1.89 s |
| **C++ pruned, OpenMP ×16** | **yes** | **0.38 s** |

GPU: RTX 4070 Ti Super. The attribution these rows force is the actual point
of this repo — see "Where the speed really came from" below. Either exact
pruned solver **beats the 0.7 s surrogate outright**: the approximation is no
longer even the fast option on this instance.

The GPU version evaluates the **exact objective on every one of the
3,713,477,331,600 combinations** and answers the question the surrogate never
could: *did the approximation find the true optimum?*

## Design

- `export_problem.py` — resolves everything game-specific *once*, via the CPU
  tool's own modules: each warehouse item becomes a fixed 8-float stat vector
  for the chosen wearer, set tables become numeric arrays, the objective
  becomes a handful of constants. The solvers see a pure combinatorial problem.
  512 random combos are evaluated through the CPU tool's exact Python path and
  embedded as ground truth.
- `src/eval.h` — the objective as **one `__host__ __device__` template**,
  instantiated as `double` (CPU reference, matches Python to ~5×10⁻⁸) and
  `float` (GPU). One implementation, zero transcription drift.
- `src/reference.cpp` — exact CPU exhaustive baseline (OpenMP), top-16.
- `src/gpu_search.cu` — v1 kernel: grid-stride walk, mixed-radix index decode,
  per-thread eval; feasible combos above a threshold are appended to a global
  candidate buffer (atomic), which the host re-scores in `double` so FP32
  near-ties can never misrank the winners. A global `atomicMax` on the raw
  float bits (positive floats are monotonic in their bit patterns) tracks the
  true maximum independently, so a mis-set threshold is detected, not silent.

### Faithfulness details that actually bit

- Python's `round()` is round-half-to-**even**; C's `round()` is half-away.
  The per-source stat rounding must use `rint()` to match the game (and the
  tool) on `.5` boundaries.
- Windows' display watchdog (TDR) kills any kernel running ≳2 s, so the full
  sweep is issued as ~1,700 slices of 2³¹ combinations, each well under the
  limit.
- The exported constants are FP32; the CPU reference in `double` therefore
  agrees with the Python tool to ~5×10⁻⁸ relative (quantization, not drift),
  and GPU FP32 to ~2×10⁻⁷ — three orders of magnitude below the gap between
  adjacent builds.

## Results (v1)

All 3,713,477,331,600 combinations evaluated exactly in **199.6 s** —
**776× a single CPU core, 251× sixteen OpenMP threads**. Only 163,108 combos
(4.4×10⁻⁸ of the space) satisfy the spd ≥ 228 constraint.

**Verdict on the surrogate:** its #1 build *is* the true global optimum
(9881.50, 速度4+暴擊2), and its #2 is the true #2. The linearization + top-120
rerank earned an exhaustive-search accuracy certificate it could never issue
for itself. (The tool's displayed 9876.8 vs the exact 9875.7 on rank 2 traces
to its *display* path rounding projected substats of a +12 item to 0.1 before
re-scoring — the solvers here all score the canonical staged vectors.)

So "is the heuristic actually right?" changed from *unanswerable* to *a
half-second GPU job*.

## Optimization log (measure → change → measure)

**Profiling note.** Nsight Compute hit `ERR_NVGPUCTRPERM` (the driver
restricts GPU performance counters to admin on Windows), so attribution was
done by ablation instead: a kernel that runs *only* v1's per-combo 64-bit
mixed-radix decode measured **32.9 ms vs 107.1 ms for the full v1 slice — the
decode alone is 31% of v1**. Pure bookkeeping, no work.

**v2 — hoist partial sums, kill the 64-bit decode.** One thread owns a 4-slot
prefix and loops rings×shoes (9,922 combos) itself: the 4-slot panel is summed
once per prefix, the 5-slot panel once per ring, and the decode shrinks to
three 32-bit divmods per 9,922 combos. Predicted a big win; **measured only
1.21× (199.6 s → 165.1 s)**. Lesson: with the bookkeeping gone, the bottleneck
moved to `finish_eval` — the set-counting branches and the damage math that
every combo still runs. The prediction was wrong and the measurement caught
it; that is the entire point of measuring.

**v3 — admissible speed bound (`search3`).** Only 163,108 of 3.7×10¹² combos
(4.4×10⁻⁸) can satisfy spd ≥ 228, so almost every eval was doomed before it
started. v3 skips a whole prefix (9,922 combos) or a whole ring (82 combos)
when even an *upper bound* on reachable speed — max remaining item spd plus
the largest spd any single set completion can add, `rint(16% × raw spd)`;
two 4-piece spd sets cannot fit in six slots — cannot reach the constraint.
The same suffix-bound idea the CPU tool's DFS uses, made admissible so a
pruned branch provably contains no feasible combo. **Measured: 165.1 s →
0.51 s (324×)**, and the pruned run reproduces the unpruned run's exact
feasible count (163,108), best (9881.498) and candidate set — the safety
argument, checked empirically.

**Not pursued, deliberately.** Shared-memory staging of the item table was on
the roadmap; two measurements killed it. v2's mere 1.21× showed the bottleneck
was `finish_eval` compute, not item loads (the whole table is 27 KB and the
inner working set 3 KB — L1 already holds it), and after v3 the loads it
would stage mostly never happen. Recorded because *deciding not to optimize*
is also a measurement-driven decision.

## Where the speed really came from

After the GPU hit 0.51 s I wrote the strongest CPU baseline I could to attack
my own result (`reference search_pruned`): the same admissible bound, but
placed at **every** level of a nested loop — a bound after the weapon/helmet
pair alone discards 1.24M-combo subtrees — with fully hoisted partial panels.
Single-threaded it covers the full space in **1.89 s**; sixteen threads,
**0.38 s — faster than the GPU's 0.51 s**. All invariants reproduced
(163,108 feasible; only 6.05×10⁷ evals actually run, 0.00163% of the space).

The honest attribution:

- **The admissible bound is worth ~4 orders of magnitude.** It dominates
  everything else in this repo.
- **The GPU is worth ~250–300× on brute force** (199.6 s vs 43 h naive;
  165 s vs 14 h) — parallel hardware pays when the work is dense and uniform.
- **On the pruned workload the GPU loses to a multicore CPU**, and the reason
  is structural, not incidental: the GPU's flat prefix decomposition must
  bound-check all 374M prefixes, while the CPU's loop nest prunes whole
  subtrees at the pair and armor levels (22,344 pair checks kill most of the
  tree before it exists). The best optimization *changed the shape of the
  workload* — from dense-uniform (GPU territory) to sparse-hierarchical
  (CPU territory). Winning it back on the GPU has a name: two-phase stream
  compaction of surviving prefixes. Identified, not yet implemented.

Practical fallout: the CPU tool's surrogate is now obsolete for this query
shape — `search_pruned` is exact *and* faster (0.38 s vs 0.7 s), and could
replace the surrogate as ArkBuildPicker's engine.

**Constraint caveat.** Every pruned number feeds on spd ≥ 228 being brutally
selective. An unconstrained query prunes nothing and runs at v2 speed on the
GPU (~165 s) — where the GPU's 250× over CPU naive is the story again. Which
solver wins depends on the instance; knowing that is the job.

## Usage

```
python export_problem.py 歐貝恩絲 --buffed --debuffed --spd-min 228
build.bat
build\reference.exe validate          # CPU f64/f32 vs Python ground truth
build\gpu_search.exe validate         # GPU f32 vs Python ground truth
build\gpu_search.exe ablate           # decode-cost ablation
build\gpu_search.exe search  --threshold 9800   # v1 naive
build\gpu_search.exe search2 --threshold 9800   # v2 hoisted partials
build\gpu_search.exe search3 --threshold 9800   # v3 + admissible pruning
python report.py <six winning indices> --compare
```

[ArkBuildPicker]: ../ArkBuildPicker
