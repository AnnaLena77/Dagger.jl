# `feat/mpi-comm-self`

> Rank-local task graphs: a graph that only this rank plans and runs, on a communicator of size one (`MPI.COMM_SELF`).

**Builds on:** feat/mpi-owner-elision

## Why

Data-dependent work (sorting, tree build, force walk) has a different number of tasks on every rank and cannot go through the SPMD graph without breaking the uniform ids and tags.

## What it adds

* `Dagger.with_rank_local(f)`: every task spawned in `f` belongs to this rank's own graph (task-local acceleration = `MPIAcceleration(COMM_SELF)`), nothing process-global changes. Ids of such a graph come from the acceleration's own counter and start above `MPI.tag_ub()`, so they can never become a world tag and different numbers of local tasks cannot shift the world's ids.
* `Dagger.per_rank(f, T)` (SPMD): every rank runs `f` as a local graph and the result (type `T`) is returned as one shared chunk per rank, chunk `r` owned by rank `r`, usable as argument of shared tasks.
* A local task may read a world chunk its own rank owns, in place. `accel_matches_proc` compares the communicator, `poolget` of a size-one communicator does not broadcast, uniformity checks are skipped for local graphs, elision and failure-only status stay world-only.

## Measured effect

A local task costs what a shared pinned task costs (40.6 / 40.3 / 45.2 ms per 256 tasks against 40.7 ms); with every switch off the world's tasks equal `perf/mpi-constants` within noise and allocate at most 1.5 % more.

## Tests

`test/mpi/comm_self.jl` (a different number of tasks per rank, world ids unaffected, failures stay local, `per_rank` hand-over).

## Limits and notes

Spawn and fetch inside the block; a chunk of another rank cannot be read by a local task (move it with a shared task first); GPU chunks in `per_rank` not supported. Barnes-Hut: rank-local tree build (-3 %).

## Main files

`ext/MPIExt.jl`, `src/acceleration.jl`, `test/mpi/comm_self.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
