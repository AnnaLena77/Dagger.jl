# `perf/mpi-rank-scope-planning`

> A task scoped to a whole rank (`Dagger.scope(mpi_rank=r)`) costs a rank that only plans it what an `ExactScope` task costs, and costs its owner less than an `ExactScope` task.

**Builds on:** integration/mpi-all (523d0ac)

## Why

The Barnes-Hut application wants to name only the rank of a task and leave the thread to the scheduler (an AMT should place work, not the application). It measured 6 % worse at 22 ranks than with pinned threads (4M bodies: 0.940 s default, 0.999 s with a rank scope for every task) and traced it to planning: a rank that only plans such a task spent ~100 us on it against ~25 us for an `ExactScope` task, and at 100 ranks planning and collectives were ~650 of 1209 ms per step.

A rank scope is a `UnionScope` with one `ExactScope` per thread, held in a `Tuple`. Two places treated that size as free:

1. **Elision.** `elide_task!` asked what the scope names three times per task (`mpi_task_processor` twice, `mpi_failure_only` once), each through a cache keyed by `objectid(scope)` behind a lock. `objectid` of a `UnionScope` hashes the whole tuple, so the lookup cost grew with the thread count. Profile of a rank that only plans: 1526 of ~2350 samples in `mpi_union_scope_processor`.
2. **The scheduler on the owner.** `schedule_one!` does `constrain(compute_scope, AnyScope())` for every task without a result scope. `constrain(::UnionScope, y)` is generic: it wraps `y` in a union and rebuilds a new `UnionScope` of every member (a `Set`, hashing, a fresh tuple), ~90 us and ~11 KB per task, to return what it was given. Profile of the owner: 1754 of 4875 samples; 19 KB/task and 246 ms GC per 20000 tasks against 8 KB/task for an `ExactScope` task.

## What it adds

* `ext/MPIExt.jl`: the union-scope cache is keyed by the scope's **address** (`jl_value_ptr`, the scope is a heap object because it is not a bits type; the entry keeps it alive, the GC does not move objects, and `===` on a hit takes the pointer-equal fast path) and read **without a lock** (the table is replaced as a whole under a lock; at most 256 entries). `elide_task!` resolves the processor once and hands it to `mpi_failure_only(opts, proc)`.
* `src/scopes.jl`: `constrain(x::UnionScope, ::AnyScope) = x`. This helps every `UnionScope` user, with or without MPI.
* `src/acceleration.jl`: `preferred_processor_count(procs, accel, rank)`, the helper of upstream PR #771. The fork already contains the *behaviour* of PR #771 (rotation of the preferred rank's processors by task id, implemented as `place_processors_uniform!` with a binary search for the rank's range), so merging the PR would change nothing; the application probes for the helper by name to detect the feature and refused `--placement=rank/any` because it was missing.

No new scope type was needed: `Dagger.scope(mpi_rank=r)` is unchanged for the scheduler, so nothing that consumes scopes (constraints, candidate caches, GPU scopes) had to learn a new one.

## Measured effect (Fulda, 2 ranks x 32 threads, microseconds per task)

| | before | after |
|---|---|---|
| rank that only plans an `ExactScope` task | 18-24 | 19-29 |
| rank that only plans a rank-scoped task | **57-60** | **19-24** |
| plain task with an argument owned by the other rank (planned by a spectator) | 61 | 36 |
| owner of a rank-scoped task (spawn + wait) | 196-260 | **154** (an `ExactScope` task: 201-239) |
| owner of a plain task without arguments | 148 | 106 |

## Tests

* `test/mpi/elide.jl`: the cache is correct by object, not by content (two equal scopes resolve alike, scopes of different ranks never share an answer, 20 rounds of fresh scope objects with `GC.gc(false)`); 3 ranks: 208 and 212 assertions together with `free_threads.jl` and `comm_self.jl`.
* `test/scopes.jl`: `constrain(union, AnyScope()) === union` in both orders.
* `test/uniform-placement.jl`: `preferred_processor_count` against a direct count.
* Probes: `LLM_info/bench/scope_cost.jl`, `plain_elide_why.jl`, `elide_cost_scope.jl` (+ `.sbatch`; `PROFILE=1`, `SCOPE=exact|rank`, `TARGET=<rank>`).

## Limits and notes

* Building `Dagger.scope(mpi_rank=r)` costs ~340 us at 32 threads (a `Set` over all processors); it is not memoized. Build it once per rank and reuse the object; a scope built per task costs ~170 us more than a prebuilt one.
* A plain (unscoped) task is **not** cheaper to plan than a rank-scoped one: `mpi_preplace!` turns it into a rank scope anyway (rank of the first MPI argument, rotation by task id). A plain task without an MPI argument is spread over all ranks, so half of the tasks are the rank's own.

## Main files

`ext/MPIExt.jl`, `src/scopes.jl`, `src/acceleration.jl`, `test/mpi/elide.jl`, `test/scopes.jl`, `test/uniform-placement.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
