# `bench/tasks-suite`

> A benchmark suite `tasks` that isolates the runtime's own per-task cost.

**Builds on:** master

## Why

The array/linalg/sparse/stencil suites measure tile compute and data movement. None isolates the per-task cost that limits latency-bound programs (many small tasks, short dependent phases) and, under MPI, what every rank pays for tasks it never runs.

## What it adds

* `benchmark/suites/tasks.jl`, opt-in via `BENCHMARK=tasks:dagger`: microsecond task bodies with explicit `ExactScope` placement: round-robin and single-owner pinned spawns, unscoped spawns, a cross-owner dependent chain (hop latency), a pairwise fan-in tree, a multi-phase exchange step, and a rank-local variant on a communicator of size one.
* Every benchmark is checked once at a tiny size with `fetch` when the suite is built (`wait` does not rethrow a failed task, so a benchmark whose tasks all fail would 'run' in microseconds); a benchmark that does not produce the expected value is skipped with a warning.
* Variants for tasks the benchmark process does not run and for bodies that may throw.

## Measured effect

Provides the numbers quoted in the other branches (e.g. foreign-rank tasks 46.5 -> 5.0 ms per 256 tasks with owner elision).

## Tests

The suite is its own test: the build-time check above.

## Limits and notes

Not in the default benchmark set. Same suite is used for both revisions when comparing.

## Main files

`benchmark/suites/tasks.jl`, `benchmark/benchmarks.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
