# `feat/mpi-owner-elision`

> Owner elision: a rank that neither runs a task nor owns data the task reads does not build the task at all. Plus free MPI threads.

**Builds on:** perf/mpi-constants

## Why

Under MPI every rank plans every task (SPMD). A rank that only has to account for a task paid ~170 us for it, and the owner broadcast `:ok`/`:error` for every task that may throw, so every rank was coupled to every task.

## What it adds

* Core hook `elide_task!(accel, spec, task)` (default `false`): an acceleration may complete a task without building a thunk. Off by default: `JULIA_DAGGER_ELIDE_FOREIGN_TASKS=1` or `Dagger.elide_foreign_tasks!(true)` on every rank.
* MPI implementation: a rank elides when the task is scoped to one rank (`ExactScope` of one `MPIProcessor`, or `Dagger.scope(mpi_rank=r)`), it owns none of the arguments (plain values, `MPIRef` chunks, earlier tasks of known owner), the return type is concrete, uniformity checks are off and there is no tag/Datadeps. It takes the task id as before and stamps exactly the placeholder `execute!` stamps on a non-owner (ref id included).
* Failure-only task status: the owner reports only failures (`Sch.task_failed!` sends the error on the task's tag); a rank that later needs the data (argument move, `fetch`) stops waiting when the failure arrives (`POISON_REF`); `fetch` stays uniform (`fetch_failed_uniform!`).
* Tasks without a scope are placed at spawn time by the deterministic rule (owner of the first MPI argument, then rotation by task id) and pinned with an `ExactScope`, so they can be elided too.
* Free MPI threads (opt-in `JULIA_DAGGER_FREE_MPI_THREADS=1` / `Dagger.free_mpi_threads!(true)`): a task whose scope does not name exactly one processor runs on any free thread of its rank instead of waiting behind a non-yielding task on the thread the scheduler happened to name.

## Measured effect

Fulda, 4 ranks x 16 threads/node, suite `tasks`, switch on vs off, 1 / 2 / 4 nodes: tasks pinned to another rank, 256 tasks 46.5 / 44.5 / 66.9 -> 5.0 / 4.8 / 4.8 ms (~19 us per foreign task); round robin 44.6 / 58.9 / 47.8 -> 15.3 / 10.6 / 8.1 ms; fan-in tree 95.5 / 98.4 / 134.2 -> 39.9 / 29.1 / 20.5 ms; all-to-all exchange 15.4 / 36.3 / 124.9 -> 9.3 / 15.3 / 33.3 ms; unscoped 64 tasks 7.5 / 10.9 / 12.1 -> 5.1 / 1.8 / 2.0 ms. Free threads: a task placed on a busy thread waited 3.32 s of 4 s without the switch, all eight finished within 0.23 s with it.

## Tests

`test/mpi/elide.jl` (rank/thread scopes, spawn-time placement, chains, failing tasks, uniform ids and chunk identity; 84 assertions per rank at 3 ranks x 4 threads), `test/mpi/free_threads.jl`.

## Limits and notes

Off by default; a task is elided only when every condition holds, otherwise the old protocol runs (never wrong, just not faster). A rank learns of a failure only when it needs the data. Barnes-Hut at 1M bodies, 16 ranks: step 0.65 s (no switches) -> 0.49 s (elision).

## Main files

`src/submission.jl`, `src/dtask.jl`, `src/acceleration.jl`, `ext/MPIExt.jl`, `src/sch/Sch.jl`, `test/mpi/elide.jl`, `test/mpi/free_threads.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
