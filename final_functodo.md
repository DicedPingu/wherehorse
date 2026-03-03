# TODO - Final FuncFind (Speed-First)

## Current Baseline
- [x] Single-file runner (`final_funcfind.py`) with minimal CLI (`--seed`, `--fast`).
- [x] Always-on JSON logging to `runs/<run-id>/run.json`.
- [x] Python 3.15+ project baseline.
- [x] Optional Numba acceleration path with safe Python fallback.
- [x] Fast preset tuned for short iteration loops.

## Next Work (Highest Priority)
- [ ] Add deterministic micro-benchmark mode (`--bench`) that reports evals/sec and gen/sec.
- [ ] Emit per-stage timing (`selection`, `crossover`, `mutation`, `scoring`) into `run.json`.
- [ ] Add adaptive population sizing based on measured scoring throughput.
- [ ] Add early-stop logic from fitness slope and stagnation confidence.
- [ ] Add warm-start checkpoint restore to skip cold-start evolution.

## Runtime Performance
- [ ] Reduce object churn in hot paths (node creation/mutation/crossover).
- [ ] Add node-pool reuse for temporary trees during evolution.
- [ ] Cache compiled postfix programs keyed by structural hash.
- [ ] Add batch scoring kernels for whole cohorts in a single call.
- [ ] Replace repeated Python list growth with preallocated typed buffers.
- [ ] Profile and remove branch-heavy math paths in evaluator loops.

## Numba / Acceleration
- [ ] Auto-detect whether Numba is installed and emit one concise acceleration status line.
- [ ] Add `--require-accel` hard-fail mode for benchmarking environments.
- [ ] Add optional `--no-accel` to force fallback for parity testing.
- [ ] Keep a tiny JIT warmup benchmark in logs to detect degraded toolchains.
- [ ] Evaluate alternative acceleration backend if Numba is unavailable.

## Search Quality Per Time
- [ ] Add adaptive mutation/crossover schedule by generation progress.
- [ ] Add novelty pressure term that decays as raw error improves.
- [ ] Add Pareto selection mode (raw error vs complexity) behind a flag.
- [ ] Add symmetry-aware operator priors for the current target grid family.
- [ ] Add elite replay buffer to prevent regressions during aggressive exploration.

## Reliability and Testing
- [ ] Add repeatable smoke test script for both `--fast` and default mode.
- [ ] Add seed-stability test (same seed, same expression/fitness within tolerance).
- [ ] Add evaluator parity test (tree eval vs postfix VM output).
- [ ] Add regression corpus of expected max raw-error thresholds by preset.
- [ ] Add CI matrix for CPython `3.15` and free-threaded `3.15t`.
- [ ] Add performance guardrail test that fails on major throughput regressions.

## Output and DX
- [ ] Add compact summary line suitable for shell history comparisons.
- [ ] Add optional `--run-tag` for easier run grouping under `runs/`.
- [ ] Add lightweight `runs/index.json` append-only registry.
- [ ] Add command to compare two run directories and print delta summary.
