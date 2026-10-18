# `feat/mpi-group-graphs`

> Group graphs: a task graph on a communicator of k ranks that only its members plan and run, with elision inside the group.

**Builds on:** feat/mpi-elision-stats

## Why

In the shared graph every rank plans every task (~25 us even when elided, growing linearly with the rank count: 7 % of a 1M-body Barnes-Hut step at 16 ranks). A graph on a sub-communicator makes planning per rank independent of the total rank count.

## What it adds

* `Dagger.with_group(f, comm)`: tasks spawned in `f` belong to a graph on `comm`; every member runs the same program inside the block; the group has its own id counter (`MPI_GROUP_ID_BASE + (index << 24) + counter`, index chosen without coordination between groups), processors and message space. `Dagger.free_group!(comm)` releases it.
* `Dagger.per_group(f, comm, T)`: `per_rank` on a group. Data crosses between communicators only where one process is a member of both and owns the chunk (read in place). `MPIRefID` ids are 64 bit now.
* Elision inside groups: participation and owner are judged per communicator; a `DTask` remembers the acceleration it was spawned under (`DTask.accel`).

## Measured effect

Synthetic LET-like pattern, groups of 4, 240 tasks per group and step, 8 ranks x 8 threads/node, Fulda, ms per step at 8 / 16 / 32 / 64 ranks: shared graph 31.9 / 40.3 / 63.8 / 108.6, group graph 26.9 / 27.1 / 27.8 / 30.4 (flat). Rank 0 plans 240 instead of 3840 tasks at 64 ranks.

## Tests

`test/mpi/groups.jl` (overlapping groups by pairs and by parity, a different number of tasks per group, `per_group` hand-over, failures stay in the group, world ids uniform, elision inside a group, `Dagger.allgather` and Datadeps in a group in later branches); 2, 3 and 4 ranks.

## Limits and notes

Keep live groups few: every group has a broadcast relay of its own (world hop 50 us -> 56 / 91 / 255 / 897 us with 1 / 8 / 32 / 128 idle groups; `free_group!` restores it). The application's LET exchange crosses groups and does not map onto disjoint groups. GPU chunks in groups untested.

## Main files

`ext/MPIExt.jl`, `src/acceleration.jl`, `src/dtask.jl`, `src/submission.jl`, `test/mpi/groups.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
