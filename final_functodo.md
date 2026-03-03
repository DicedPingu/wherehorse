# TODO - Final FuncFind Roadmap

## Completed (Current State)
- [x] Add subtree crossover (recombination) in reproduction flow.
- [x] Add moderate targeted stagnation escape (constant/operator/subtree edits + partial refresh).
- [x] Add multi-seed mode (`--multi-seed N`) with repeatable seed sequencing.
- [x] Add aggregate multi-seed reporting (median/mean/std raw error + best seed/expression).
- [x] Add crossover/stagnation tuning flags:
  - `--crossover-rate`
  - `--stagnation-window`
  - `--stagnation-threshold`
  - `--stagnation-refresh`
- [x] Keep Python 3.14 strict Numba policy.
- [x] Add single startup warning for Python 3.15 fallback mode (+ `runtime_warning` field in JSON mode).
- [x] Keep pure math operator set (no trig/cos/sin path).

## Quality Gate Status
- [x] Baseline comparison run at fixed budget (`population=500`, `generations=1500`, workers=1).
- [x] Medium sweep completed (20 seeds).
- [x] Median raw-error improvement target met (>10%).
- [x] Latest measured result:
  - Baseline median raw error: `110.0`
  - New median raw error: `57.5`
  - Improvement: `47.73%`

## Next Priority (High Impact)
- [ ] Add a true benchmark command that compares `numba-jit` vs Python fallback on the same seeds and config.
- [ ] Add checkpoint save/resume (`--checkpoint-out`, `--checkpoint-in`) for long runs.
- [ ] Add automatic A/B mode for strategy tuning (mutation-only vs crossover+escape) with summary output.
- [ ] Add significance reporting for multi-seed comparisons (median delta + confidence indicator).

## Search Quality Backlog
- [ ] Add adaptive fitness-weight scheduling across generations (exploration -> exploitation).
- [ ] Add operator probability scheduling by generation.
- [ ] Add novelty archive to preserve diverse high-value structures.
- [ ] Add Pareto mode (raw error vs complexity) to avoid single-scalar overfitting.
- [ ] Add optional semantic dedupe/simplification before scoring duplicates.

## Performance Backlog
- [ ] Cache compiled postfix programs by subtree hash to reduce recompilation overhead.
- [ ] Add incremental fitness updates for small local mutations where valid.
- [ ] Add batch scoring mode that evaluates many individuals in one Numba call.
- [ ] Add optional process-based parallel mode for non-free-threaded runtimes.
- [ ] Add profiling mode (`--profile`) with per-stage timing breakdown.
- [ ] Add adaptive worker policy (auto-tune based on population size + runtime characteristics).

## UX and CLI Backlog
- [ ] Add progress ETA + moving-average generation speed.
- [ ] Add `--quiet` mode for final-summary-only output.
- [ ] Add `--report-file` for human-readable report export.
- [ ] Add `--csv-log` for per-generation metrics.
- [ ] Add config file support (`--config path`) for reproducible presets.
- [ ] Add quality presets (`--preset speed|balanced|quality`) that map to known good parameters.

## Reliability and Testing Backlog
- [ ] Add unit tests for postfix compiler correctness (`tree.evaluate` parity with VM scoring).
- [ ] Add unit tests for safe-math guards (div/mod by zero, exponent clamps, clipping).
- [ ] Add reproducibility tests for fixed-seed single and multi-seed runs.
- [ ] Add regression tests for known target grids and expected fitness/raw-error thresholds.
- [ ] Add compatibility CI matrix for Python 3.14 and Python 3.15 fallback mode.
- [ ] Add non-regression performance check for key workloads (guard against accidental slowdowns).

## Python 3.15+ Forward Work
- [ ] Add optional alternative acceleration backend for 3.15 while Numba support is incomplete.
- [ ] Keep fallback warning concise and informative; include measured slowdown guidance in docs.
- [ ] Re-enable strict Numba requirement on 3.15 once stable wheels are available.
- [ ] Add one command that validates both runtime paths in one sweep:
  - Python 3.14 + Numba
  - Python 3.15 + fallback
- [ ] Track upstream `numba`/`llvmlite` releases and update pins when 3.15 support lands.
