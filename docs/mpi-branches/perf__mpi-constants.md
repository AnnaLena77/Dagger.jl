# `perf/mpi-constants`

> Three scheduler fixes that make the per-task cost of MPI independent of the number of processors.

**Builds on:** master

## Why

Under MPI (uniform execution) every task re-sorted all candidate processors with a key that formats a string on every comparison, and every enqueue woke every other processor, which then found out it could not steal. Unscoped `@spawn` therefore cost O(P log P) string constructions per task (~85 ms per task at 832 processors).

## What it adds

* `sort_processors_canonical!` / `compatible_processors_ordered`: the canonical candidate order is computed once per candidate set and cached; `place_processors_uniform!` does only the per-task part (owner preference, rotation by task id), allocation-free. The resulting order is identical to the old algorithm's, which uniformity across ranks depends on.
* `do_tasks` no longer wakes processors that cannot steal (every `MPIProcessor`).
* `compat_entry` replaces a stale cache entry instead of keeping it (`BasicLFUCache.get!` keeps the old value, so every later lookup missed and recomputed).

## Measured effect

Fulda, 4 ranks x 16 threads/node, suite `tasks`, median of 7: unscoped spawn of 64 tasks at 64 / 128 / 256 processors 244 / 547 / 1258 ms -> 11.7 / 12.2 / 14.0 ms (21x / 45x / 90x), allocations 0.75 M -> 13 k; no waking of non-stealing runners: all 27 benchmark/size cells faster or equal, allocations -20..-65 %; the stale-cache fix removed a 5.3x slowdown (7.5 -> 40.1 ms per 64 tasks) that appeared when a one-rank communicator registered processors mid-run.

## Tests

`test/uniform-placement.jl` (new: compares the order with a copy of the old implementation on randomized candidate sets, 5770 assertions, plus the stale-entry case), `test/mpi/spawn.jl`, `test/allocations.jl` unchanged (183 allocations, bound 500).

## Limits and notes

No switch: always on. The pinned (ExactScope) path has one candidate and is unchanged.

## Main files

`src/acceleration.jl`, `src/sch/Sch.jl`, `src/sch/util.jl`, `test/uniform-placement.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
