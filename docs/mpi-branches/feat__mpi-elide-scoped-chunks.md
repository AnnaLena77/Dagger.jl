# `feat/mpi-elide-scoped-chunks`

> Elision also for tasks whose chunk arguments carry a scope restriction (the usage guide `MPI_FEATURES.md` starts here).

**Builds on:** feat/mpi-local-datadeps

## Why

A rank skipped elision for any task with a chunk argument whose scope was not `AnyScope`, and a `DArray` block has one: every task that reads a block was planned in full by every rank (181 us on a rank that neither runs the task nor owns the block, against 28 us for an unrestricted chunk, measured in a Barnes-Hut code).

## What it adds

* A chunk's scope only matters through `constrain(task scope, chunk scope)` in the scheduler. The elision code computes that chain on the rank that elides (the same one the scheduler computes) and elides when it is valid; an invalid one keeps the ordinary protocol, which reports the failure as before.
* Also in this branch (cherry-picked): the cheap cost estimate of `perf/sched-cost-estimate`, the space-hash fix of `fix/mpi-space-hash`, and `MPI_FEATURES.md` (switches, elision rules, rank-local graphs, pitfalls).

## Measured effect

Foreign-rank planning of a task reading a restricted chunk: 181 us -> ~28 us.

## Tests

`test/mpi/elide.jl` 'chunk arguments with a scope restriction' (a chunk of the target rank with a rank scope, also with a pinned task; a contradicting scope is not elided and fails). Fails on the old rule, passes on the new one (3 ranks x 2 threads).

## Limits and notes

Requires the elision switch (off by default).

## Main files

`ext/MPIExt.jl`, `test/mpi/elide.jl`, `MPI_FEATURES.md`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
