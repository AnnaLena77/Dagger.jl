# `perf/mpi-elide-typed-proc`

> 13-14 % cheaper planning of a task this rank only plans (two small fixes in the elision path).

**Builds on:** feat/mpi-group-graphs

## Why

`mpi_scope_processor` returns an entry of a `Dict{UInt,Tuple{Any,Any}}`, so its inferred type was `Any` and everything `elide_task!` did with the processor (field reads, `memory_space`, `tochunk(...; type=T)`) was a dynamic dispatch per planned task. `take_ref_id!` also looked a scoped value up twice, and it runs twice per elided task.

## What it adds

* Type assertion `Union{Nothing,MPIProcessor}` at the two return sites (`mpi_task_processor`, `mpi_scope_processor`).
* `take_ref_id!` reads `DATADEPS_THUNK_ID[]` once.

## Measured effect

Fulda (Opteron 6272), one rank plans 20000 tasks another rank runs: 4 ranks x 16 threads 19.3 -> 16.75 us per task, 2 x 32 threads 20.6 -> 17.6 us (-13..-14 %), 2135 -> 1831 B per task. (Same loop on a laptop: 1.4 -> 1.16 us; every primitive is 7-20x slower on the Opterons.) At 104 ranks that is ~8 ms of the application's 188 ms `let.plan`.

## Tests

`test/mpi` comm_self, elide, spawn, groups, free_threads at 3 ranks. Argument shape does not change the cost (plain 16.9, two task results 19.9, export-shaped task 17.65 us).

## Limits and notes

The rest of the cost is inherent to building a typed stub chunk per task; lazily created stubs or batched planning would save at most about a third of it (not started). GC is not the cause (0 ms in the timed region).

## Main files

`ext/MPIExt.jl`, `AGENTS.md` (lesson 88)

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
