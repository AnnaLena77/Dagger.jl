# `fix/mpi-elide-stub-processor`

> With owner elision, the placeholder a rank stamps for a task it only plans names the processor the owner really runs the task on.

**Builds on:** `perf/mpi-rank-scope-planning` (its address-keyed union-scope cache)

## Why

A rank that elides a task (it neither runs it nor sources its data) stamps a placeholder result. For a task scoped to a whole rank, the placeholder's processor came from `mpi_union_scope_processor`, a fixed member of the union, while the owner's scheduler picks the thread inside the rank by rotation (`place_processors_uniform!`). So the ranks disagreed on `chunk.processor` (BUGS B9; old, also on the snapshot). On CPU only the uniformity checks notice it; with several GPUs per rank the placeholder would name another device.

## What it changes

`ext/MPIExt.jl`: under uniform execution the owner takes the candidate at `mod(id - 1, n) + 1` of the scope's processors in canonical order (the owner rank is the whole candidate list or not in it, so no preferred block shortens it). The placeholder now uses the same index, from an address-keyed, lock-free cache of the ordered processors per union scope (`mpi_union_scope_ordered`, `mpi_planned_processor`), so the per-task cost does not grow with the thread count.

## Measured effect

* `test/mpi/spawn.jl` with free threads and elision on, 2 ranks x 4 threads: 35 failures -> 0; full MPI suite with free threads + elision (Fulda, 2 x 4): 2227 / 2180 assertions, 0 failures (it had 62-119 placement failures before).
* Planning cost unchanged or lower: a rank scope built per task, planned by another rank, 225-231 -> 161-176 us (together with `perf/mpi-rank-scope-planning`, interleaved A/B on Fulda, 2 x 32).

## Tests

`test/mpi/elide.jl`: the processors stamped on 16 tasks of one rank scope agree across ranks; fails without the change.

## Main files

`ext/MPIExt.jl`, `test/mpi/elide.jl`
