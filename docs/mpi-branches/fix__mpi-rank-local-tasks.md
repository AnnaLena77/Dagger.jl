# `fix/mpi-rank-local-tasks`

> Early approach to rank-local work: tasks with `acceleration=DistributedAcceleration()` stay on their rank. Superseded by `feat/mpi-comm-self`.

**Builds on:** master

## Why

Work that only one rank should plan and run (data-dependent number of tasks) cannot go through the shared SPMD graph. This branch kept such tasks local by giving them the `DistributedAcceleration`.

## What it adds

* `DTask`/`submission` changes so a task spawned with `acceleration=DistributedAcceleration()` is not mirrored on the other ranks; `test/mpi/rank_local.jl`.

## Measured effect

Functional only. It did not give the rank-local graph its own id/tag space, so the world's uniform ids were exposed to the number of local tasks; the maintainers (J. Samaroo) preferred a convergent-per-communicator design.

## Tests

`test/mpi/rank_local.jl` (33 lines).

## Limits and notes

Kept for reference only; use `feat/mpi-comm-self` (`Dagger.with_rank_local`). Not part of `integration/mpi-all`.

## Main files

`ext/MPIExt.jl`, `src/dtask.jl`, `src/submission.jl`, `src/acceleration.jl`, `test/mpi/rank_local.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
