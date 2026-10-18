# `perf/sched-cost-estimate`

> Cheap cost estimate for a task with many candidate processors; independent of MPI.

**Builds on:** master (shares two commits with perf/mpi-constants)

## Why

An unscoped task on a process with many threads is placed by `estimate_task_costs!`, which used dictionaries keyed by the abstract type `Processor` and a dynamic dispatch per element, so the per-task cost grew with the thread count of one process: 126 us at 8 threads and 553 us at 64 (empty task, single process, Fulda).

## What it adds

* Costs live in a vector aligned with the candidates instead of a `Dict{Processor,Float64}`; the dictionary is only filled for callers that ask (`estimate_task_costs`, used by Datadeps and tests).
* Per-candidate invariants are computed once (no transfer-cost table without Chunk arguments; per-worker pressure and transfer-rate lookups once per run of candidates of one worker).
* The two per-candidate loops are function barriers specialised on the candidates' common concrete type; mixed candidates keep the generic path; ties keep their random order.

## Measured effect

Single process without MPI, Fulda (Opteron 6272), unscoped empty task, us per task at 8 / 16 / 32 / 64 threads: 125.7 / 158.0 / 251.6 / 552.7 -> 101.0 / 110.0 / 151.0 / 331.2 (-20 / -30 / -40 / -40 %).

## Tests

`test/scheduler.jl` 'Scheduler algorithms' (cost values, per-processor pressure, transfer rate, signatures; 82 assertions) unchanged and green; `test/allocations.jl` unchanged.

## Limits and notes

The cost still rises with the thread count (at 64 threads the scheduler lock and the dispatch dominate), but much more slowly. Contains the same two commits as `perf/mpi-constants` (placement sort, no wake-up of non-stealers).

## Main files

`src/sch/util.jl`, `src/sch/Sch.jl`, `src/acceleration.jl`, `test/`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
