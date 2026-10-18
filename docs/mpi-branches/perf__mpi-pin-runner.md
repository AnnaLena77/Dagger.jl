# `perf/mpi-pin-runner`

> MPI tasks start pinned to the thread of their processor, so a dependent task no longer pays a second task and a hand-over.

**Builds on:** feat/mpi-comm-self

## Why

`task_tid_for_processor` had no method for `MPIProcessor`, so MPI tasks started migratable. `execute!(::ThreadProc)` then missed its inline fast path and, for every task, created a second task pinned to the thread, scheduled it and waited: 160-190 us per task (measured inside `execute!`), 7 us once the task starts pinned.

## What it adds

* `task_tid_for_processor(::MPIProcessor{ThreadProc}, task)` returns the inner processor's tid for processors of this rank, and nothing for a task that asked for no thread while free MPI threads are on.
* The pooled task slots (pinned or made migratable once, when a slot is created) are kept in two caches, `:start_processor_runner!_task_cache` and `..._migratable`, because that decision now differs per task: a migratable task that reused a pinned slot stayed bound to that thread.

## Measured effect

Fulda, 4 ranks x 16 threads, dependent chain of 3000 tasks, us per link: same rank 458 / 414 -> 284 / 254, alternating between two ranks 572 / 557 -> 296 / 307 (~40 % faster). Independent tasks unchanged within noise (160.0 -> 156.5 allocations per pinned task).

## Tests

`test/mpi/free_threads.jl` (pinned-thread unit test; asserts when the tasks ran, not when `fetch` returned them).

## Limits and notes

Always on. A benchmark chain pinned to thread 1 shares that thread with the driver and can hide the gain: pin to another thread.

## Main files

`src/sch/Sch.jl`, `ext/MPIExt.jl`, `test/mpi/free_threads.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
