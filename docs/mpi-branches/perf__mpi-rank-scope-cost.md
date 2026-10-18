# `perf/mpi-rank-scope-cost`

> Fix of a regression in `elide_task!`: the scope-chain check is computed only for chunk arguments that carry a restriction.

**Builds on:** feat/mpi-elide-darray-blocks

## Why

The check added for scope-restricted chunk arguments computed `constrain(compute scope, result scope)` for every task. For a task scoped to a whole rank that is a union scope with one entry per thread, and `constrain` on it costs time proportional to the thread count.

## What it adds

* Start the chain from the compute scope, and only for the arguments that carry a restriction, in a function of its own (`mpi_chunk_scopes_fit`). The result scope is always nothing for a task that gets there.

## Measured effect

Foreign-rank planning of a rank-scoped task: 69-76 us -> 38-43 us at 16 threads; 123-130 us -> ~60 us at 32 threads (an `ExactScope` task costs ~20 us).

## Tests

`test/mpi/elide.jl` unchanged and green; probe `LLM_info/bench/scope_cost.jl` (16 and 32 threads per rank).

## Limits and notes

A rank scope still costs ~3x an `ExactScope` for a rank that only plans it; a structural fix (a dedicated rank scope type) is blocked by untestable MPI x GPU suites. Build `Dagger.scope(mpi_rank=r)` once and reuse it (~0.35 ms per call at 32 threads).

## Main files

`ext/MPIExt.jl`, `AGENTS.md` (lesson 87)

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
