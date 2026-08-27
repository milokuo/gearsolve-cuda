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

| solver | exact? | throughput | full-space time |
|---|---|---|---|
| Python tool (surrogate + rerank) | no — linearized | — | 0.7 s |
| C++ single thread, exact | yes | 2.4×10⁷ evals/s | ~43 h (extrapolated) |
| C++ OpenMP ×16, exact | yes | 7.4×10⁷ evals/s | ~14 h (extrapolated) |
| **CUDA v1 naive, RTX 4070 Ti Super** | **yes** | **1.86×10¹⁰ evals/s** | **199.6 s, measured** |

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

So: the surrogate stays the interactive path (0.7 s), but "is the heuristic
actually right?" changed from *unanswerable* to *a three-minute GPU job*.

### Roadmap (v2)

Measured next steps, in profile-first order: pin the two innermost slots to
the thread and hoist the 4-slot partial panel out of their loops (turns most
of the eval into a handful of adds); kill the per-combo 64-bit div/mod decode;
stage item vectors in shared memory; skip evals that a per-slot speed bound
already proves infeasible; Nsight Compute before/after for each.

## Usage

```
python export_problem.py 歐貝恩絲 --buffed --debuffed --spd-min 228
build.bat
build\reference.exe validate
build\gpu_search.exe validate
build\gpu_search.exe search --threshold 9800
python report.py <six winning indices> --compare
```

[ArkBuildPicker]: ../ArkBuildPicker
