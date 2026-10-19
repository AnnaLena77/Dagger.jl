# `perf/sched-targeted-kicks`

> After an enqueue, wake only the runners that can take the task, at most one per enqueued task. Shared memory and MPI alike.

**Builds on:** `integration/mpi-all` (`e710a8d`); the change itself is independent of MPI and applies to a plain Dagger process.

## Why

After every `do_tasks`, every other processor's doorbell was rung "to make it steal". Each woken runner then scanned the queues of all T processors, locking each one: T wake-ups and about T^2 lock operations per enqueued task. This is one of the reasons the task throughput of one process falls above 32 threads. For a task whose scope is an `ExactScope` nobody else can take it at all (`proc_in_scope` is `proc == scope.processor`), and for an unscoped task one thief is enough.

## What it adds

* No kick when every enqueued task has an `ExactScope`.
* Otherwise at most one runner per enqueued task is woken, chosen among runners that are **parked** (`ProcessorInternalState.parked`, set before the runner looks for work and cleared when it takes a task; one that is not parked looks at the queues again by itself, so skipping it cannot lose a wake-up) and that could take a task (scope, acceleration, free occupancy: the test the steal scan applies). The starting point rotates (`KICK_CURSOR`) so the same runners are not always the thieves. The MPI rule (`stealing_permitted`) stays.

## Measured effect

One process, no MPI, Fulda (Opteron 6272), tasks/s with 50 us bodies, before -> after at 8 / 16 / 32 / 64 threads (single runs, `LLM_info/bench/shm_scaling.jl`): unscoped 6.6k / 4.9k / 5.7k / 3.3k -> 6.0k / 5.1k / 6.0k / 5.6k (64 threads +69 %); pinned 7.0k / 7.2k / 8.3k / 5.5k -> 7.5k / 8.0k / 8.8k / 7.2k (64 threads +32 %). With 500 us bodies at 64 threads: unscoped 3.7k -> 4.2k, pinned 5.8k -> 7.5k. Within noise at 8 and 16 threads, as expected: the wasted work grows with T^2.

## Tests

`thunk`, `scheduler`, `processors`, `uniform-placement`, `scopes`, `options`, `task-queues`, `task-affinity` and `mutation` pass through `test/runtests.jl` (`-p 2 -t 2`).

## Limits and notes

The throughput no longer falls between 32 and 64 threads but stays at about 6-8k tasks/s: what remains is the serial per-task cost of the submitting thread (about 125-165 us on the Opterons; `schedule_one!` 53 %, `fire_tasks!` 17 %, spawn and options 19 %, a flat profile). Batching with `spawn_bulk` does not help in one process. Past that ceiling: coarser tasks, or more schedulers per node (several MPI ranks on one node).

## Main files

`src/sch/Sch.jl`
