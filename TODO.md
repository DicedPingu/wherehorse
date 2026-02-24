# TODO - Final FuncFind Improvements

## High Priority
- [ ] Add crossover (recombination) between parents, not only mutation.
- [ ] Add a true benchmark command that compares `numba-jit` vs Python fallback on the same seed.
- [ ] Add checkpoint save/resume (`--checkpoint-out`, `--checkpoint-in`) for long runs.
- [ ] Add stagnation escape strategy with targeted operator/constant mutation instead of random reset.
- [ ] Add multi-seed run mode (`--multi-seed N`) and report best/mean/std fitness.

## Accuracy and Search Quality
- [ ] Add adaptive fitness weights over time (early exploration, late exploitation).
- [ ] Add operator probability scheduling by generation.
- [ ] Add optional semantic simplification pass before scoring duplicate expressions.
- [ ] Add novelty archive to preserve diverse high-value structures.
- [ ] Add Pareto mode (error vs complexity) to avoid overfitting to one scalar score.

## Performance
- [ ] Cache compiled postfix programs by subtree hash to reduce recompilation overhead.
- [ ] Add incremental fitness update for local mutations where possible.
- [ ] Add batch-scoring mode that evaluates many individuals in one Numba call.
- [ ] Add optional process-based parallel mode for non-free-threaded runtimes.
- [ ] Add profiling mode (`--profile`) that outputs per-stage timing breakdown.

## UX and CLI
- [ ] Add progress ETA and moving-average generation speed.
- [ ] Add `--quiet` mode that only prints final summary.
- [ ] Add `--report-file` for saving final human-readable report.
- [ ] Add `--csv-log` to stream per-generation metrics to CSV.
- [ ] Add config file support (`--config path`) for reproducible experiment presets.

## Reliability and Testing
- [ ] Add unit tests for postfix compiler correctness (tree eval == VM eval).
- [ ] Add unit tests for safe math behavior (div/mod by zero, exponent clamps, clipping).
- [ ] Add reproducibility tests for fixed seed runs.
- [ ] Add compatibility CI matrix for Python 3.14 and 3.15.
- [ ] Add regression tests with known target grids and expected fitness thresholds.

## Python 3.15 Forward Work
- [ ] Add optional alternative JIT backend path for 3.15 while Numba support lags.
- [ ] Add explicit runtime warning that explains why fallback mode is slower on 3.15.
- [ ] Re-enable strict Numba requirement on 3.15 once stable wheels are available.
- [ ] Add one command that validates 3.14 (Numba) and 3.15 (fallback) in one sweep.
