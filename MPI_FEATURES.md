# MPI features of the `integration/mpi-all` branch: how to use them

For application authors (and agents) writing SPMD code on Dagger's MPI backend. This branch is a **testing branch**
that stacks several independent changes; each one is also on its own branch (listed at the end). Everything below was
measured on a cluster with 4 ranks x 16 threads per node, Open MPI, Julia 1.13.

## 0. Setup

```julia
] add https://github.com/AnnaLena77/Dagger.jl#integration/mpi-all     # needs DataStructures 0.19 (not 0.18)
using Dagger, MPI
Dagger.accelerate!(:mpi)          # on every rank, as before
```

Nothing about SPMD changes: **every rank runs the same sequence of `@spawn`s** (same functions, same order, same
scopes). The features below make a rank cheaper when it does not take part in a task, or give it work the others
never see (rank-local graphs). They do not let ranks spawn different shared tasks.

## 1. Two switches (both OFF by default)

| Switch | Environment variable (read at startup, set it on **every** rank, e.g. `mpiexec -x VAR`) | Function |
|---|---|---|
| owner elision | `JULIA_DAGGER_ELIDE_FOREIGN_TASKS=1` | `Dagger.elide_foreign_tasks!(true)` |
| free MPI threads | `JULIA_DAGGER_FREE_MPI_THREADS=1` | `Dagger.free_mpi_threads!(true)` |

Fixes that are always on (no switch): cached placement of unscoped tasks, no wake-up of processors that cannot steal,
cheap cost estimate, MPI tasks start pinned to their thread (dependent tasks ~40 % faster), ref-id leak fix, Open MPI
hash fix.

## 2. Owner elision (`JULIA_DAGGER_ELIDE_FOREIGN_TASKS=1`)

Under SPMD every rank plans every task. With the switch on, a rank that neither runs a task nor owns data the task
reads does not build the task at all (about 20 us instead of about 170 us per such task).

A rank elides a task when **all** of these hold (otherwise it silently behaves as before, never wrongly):
* the task is scoped to **one rank**: `scope=Dagger.scope(mpi_rank=r)` (or an `ExactScope` of one of its processors);
* this rank is not that rank and owns none of the task's arguments;
* every argument is a plain value, an `MPIRef` chunk (a scope restriction on the chunk is fine as long as it does not
  contradict the task's own scope), an earlier task whose owner is known, or **a block of a `DArray`** (the block grid is the
  same on every rank, so the rank of each block is known when the task is spawned);
* the function's **return type is concrete** (write type-stable functions; a task whose result type cannot be inferred
  needs the type broadcast to everyone);
* uniformity checking is off (`Dagger.check_uniformity!(true)` disables elision) and it is not a Datadeps task.

Tasks **without** a scope are placed at spawn time by a deterministic rule (the owner of the first MPI argument is
preferred, otherwise rotation by task id) and can then be elided on the other ranks like scoped ones.

Failure semantics change slightly: for such a task the owner reports **only failures**. A rank learns of a failure when it
later needs the task's data (an argument move or a `fetch`); a rank that never touches the result never sees it.
`fetch` stays uniform: every rank takes the same ids.

`DArray` blocks do not defeat elision any more (they did before: every task reading a block was planned in full by every
rank, 190-250 us; a rank that only plans such a task now pays about 50 us, measured on 4 ranks x 8 threads).

**Why is my task not elided? Ask Dagger:** `Dagger.elision_stats()` returns, for this rank, how many of the tasks it planned were
elided (`elided`), run here (`own_task`), or planned in full, with the reason: `no_owner_known` (the scope names no single rank),
`not_failure_only`, `syncdeps_or_restore`, `return_type_not_concrete`, `function_not_plain`, `argument_not_positional`,
`argument_participates` (this rank owns an argument, or its owner is unknown) and `scope_conflict`; `not_applicable` counts tasks seen
while uniformity checks are on or in a rank-local graph. The counts are cumulative; call `Dagger.reset_elision_stats!()` at the start of a
step and `Dagger.elision_stats()` at its end to get per-step numbers. They only count while the elision switch is on.

Cost of the scope you pass: build `Dagger.scope(mpi_rank=r)` **once** and reuse the object (one call costs ~0.35 ms at 32 threads);
a task with a rank scope costs a rank that only plans it the same as an `ExactScope` task (19-24 vs 19-29 us at 32 threads, branch `perf/mpi-rank-scope-planning`; before it was ~60 us) and costs its owner less (154 vs 201-239 us). Name the rank, not the thread, unless you need a particular thread: the scheduler (and `free_mpi_threads!`) then place it.

Practical rule: give each task an owner rank (`scope=Dagger.scope(mpi_rank=r)`), keep its inputs as chunks owned by
that rank, and make the functions type-stable. Measured effect (1 node, 4 ranks): pinned round robin 40.9 -> 13.9 ms
per 256 tasks, pinned to one other rank 46 -> 5.3 ms, fan-in tree 98 -> 40 ms.

## 3. Free MPI threads (`JULIA_DAGGER_FREE_MPI_THREADS=1`)

Without the switch an MPI task runs on the thread of the processor the scheduler named (a rotation by task id), so it can
wait behind a long non-yielding task on that thread while other threads of the rank are idle (measured: 3.3 s block).
With it, a task whose scope does **not** name exactly one processor runs on any free thread of its rank.
Tasks pinned with an `ExactScope(proc)` (and Datadeps tasks) stay bound to their thread.

Use `scope=Dagger.scope(mpi_rank=r)` for work whose thread does not matter (tree walks, merges), `ExactScope` only when
you really want one thread.

Limits (measured): (1) the switch makes tasks **start** early; their results can still reach *other ranks* up to ~0.45 s
late while all but one thread of the rank are busy with non-yielding code (the broadcast relay shares the free thread).
Keep one or two threads of headroom per rank. (2) A non-allocating loop never reaches a GC safepoint, so a collection
triggered anywhere waits for it: put `GC.safepoint()` in long kernels that do not allocate.

## 4. Rank-local task graphs

A graph that **only this rank plans and runs**, with a data-dependent number of tasks that may differ per rank (sorting,
tree build, force walk). Internally it is an ordinary graph on a communicator of size one (`MPI.COMM_SELF`), so nothing
the other ranks do can disagree with it and the world's task ids/tags are never shifted by it.

```julia
# inside a rank-local block, every @spawn belongs to this rank's own graph
total = Dagger.with_rank_local() do
    parts = [Dagger.@spawn(work(i)) for i in 1:n_this_rank_wants]     # n may differ per rank
    fetch(Dagger.@spawn(combine(parts)))                               # fetch inside the block
end

# hand a rank-local result to shared tasks: SPMD call, every rank calls it, every rank runs f as its own local graph
chunks = Dagger.per_rank(Vector{Float64}) do        # the type must be given (it must be known on every rank)
    Dagger.@spawn(build_local(rank_data))           # return a task or chunk of the local graph
end                                                  # -> Vector of P chunks, chunks[r+1] owned by rank r
s = fetch(Dagger.@spawn(scope=Dagger.scope(mpi_rank=0), reduce_two(chunks[1], chunks[end])))   # shared task, data moves
```

Rules:
* spawn and fetch local tasks **inside** the `with_rank_local` block; to give a local result to shared tasks use
  `per_rank` (passing a task of a local graph directly as an argument of a shared task is not supported and untested);
* a local task may read a world chunk **owned by its own rank** in place; a chunk of another rank is an error
  (move it with a shared task first);
* a failing local task fails only on its rank; the world is unaffected;
* `Dagger.spawn_datadeps()` works inside a local graph;
* not supported yet: GPU chunks in `per_rank`, a non-blocking `per_rank`.

A local task costs what a shared pinned task costs (about 40 ms per 256 tasks), so use local graphs for *data-dependent
or rank-specific* work, not to save planning time (elision does that for shared tasks).

## 4b. Group graphs (branch `feat/mpi-group-graphs`, included in `integration/mpi-all`)

A rank-local graph generalized to **k ranks**: a task graph on a communicator of k ranks (`MPI.Comm_split`) that only
its members plan and run. Ranks outside the group do not build its tasks at all, and the group has its own task ids and
tags, so the world's ids stay in agreement however many tasks each group spawns (different groups may even spawn different numbers).

```julia
g = MPI.Comm_split(MPI.COMM_WORLD, color, key)         # e.g. color = rank ÷ 4: groups of 4 ranks
res = Dagger.with_group(g) do                          # every member runs the same code inside the block
    ts = [Dagger.@spawn(scope=my_group_scope(i), work(i)) for i in 1:n]     # n may differ between groups
    fetch.(ts)                                         # fetch inside the block
end
chunks = Dagger.per_group(g, Vector{Float64}) do       # like per_rank, on the group: one chunk per member
    Dagger.@spawn(build_local(data))
end
Dagger.free_group!(g)                                  # collective over the members, when the group is no longer used
```

Rules:
* every member of `g` runs the same sequence of `@spawn`s inside the block (SPMD **within the group**);
* a processor of a member of the group is named through the group's own processor list
  (`Dagger.get_processors(MPIClusterProc(g))`, see `test/mpi/groups.jl`; `ExactScope` of one of them);
* data crosses between the world and a group, or between two groups, only for a rank that is a member of both and
  owns the chunk (it is then used in place); anything else is an error: hand data over with `per_group` or a shared task;
* with the elision switch on, a member that neither runs a group task nor owns its data skips it exactly like in the world graph;
* a failing group task is seen by every member that fetches it; the world and other groups are unaffected;
* `Dagger.spawn_datadeps()` works inside a group: the data of a region is owned by the group's rank 0, tasks run on any member, the value is written back at the end of the region (tested at 3 and 4 ranks, CPU only); not supported yet: GPU chunks inside a group;
* **keep the number of live groups per rank small and call `free_group!` when a group is done.** Each group has a broadcast
  relay of its own that polls and sleeps by itself, and idle relays take thread time from the workers. Measured on one machine
  with 2 ranks x 4 threads (a dependent task hop in the *world* graph, 50 us without groups): 1 live group 56 us, 4 groups 67 us,
  8 groups 91 us, 32 groups 255 us, 128 groups 897 us; after `free_group!` of all of them it is back at 49 us. A rank that is in a
  few disjoint groups pays almost nothing; hundreds of live overlapping groups would. (Serving all groups from one relay was tried
  and was slower: a probe per group per sweep costs more than the sleeping relays it replaces.)

Measured (synthetic LET-like pattern, groups of 4, 240 tasks per group and step, 8 ranks x 8 threads per node, owner elision on,
best of 5 steps), *before* elision inside groups: the shared graph costs 29.5 / 41.6 / 63.5 / 104.6 ms per step at 8 / 16 / 32 / 64
ranks (about 1.3 ms more per added rank: every rank still pays ~25 us for each foreign task), the group graph stays flat at 45-47 ms
(rank 0 plans 240 tasks instead of 3840 at 64 ranks). **Use it only when planning is a large part of your step**: the Barnes-Hut application at 16-26 ranks
spends 3-7 % of its step planning, and there a rank-local graph with unpinned workers was even slower (placement, not planning).
It is for large rank counts.

## 4c. `Dagger.allgather`: every rank gets the values of tasks owned by all ranks (branch `feat/mpi-allgather`)

The pattern "one task per rank (a summary of its own data) -> one gather task with P arguments -> `fetch`" moves every result to one
rank and then broadcasts the gathered vector. `Dagger.allgather` is the exchange itself:

```julia
summaries = [Dagger.@spawn(scope=owner_scope(r), return_type=Summary, summarize(part[r])) for r in 0:P-1]
all_summaries = Dagger.allgather(summaries)        # Vector{Summary} on EVERY rank, in the order of the list
```

* SPMD collective: every rank calls it with the same list in the same order (inside `with_group` it works on the group);
* each rank contributes the values of the tasks whose result it owns; by default the ranks pass them around in ceil(log_5 P) rounds of
  point-to-point messages with cooperative waits (so it never parks the thread that runs the broadcast relay);
  `JULIA_DAGGER_MPI_ALLGATHER=native` (on every rank) uses the library's non-blocking `MPI_Iallgather` / `MPI_Iallgatherv` instead.
  The rounds are better for small contributions (up to ~10 KB per rank, what the Barnes-Hut summaries are: 20 against 40 ms at 104 ranks),
  the native collective from about 16 KB per rank on Open MPI (96 ranks, 64 KB: 171 against 258 ms). Try both on a new machine;
* the result is a `Vector` with the narrowest concrete element type the values share (`Vector{Any}` only for an empty list or mixed
  types); `Dagger.allgather(tasks, T)` returns a `Vector{T}`;
* call it from the driver, one call at a time (an MPI rule for collectives on one communicator);
* a failed task makes **every** rank throw its exception, after all have taken part; the next call works;
* it takes no task or reference ids, so it can be called wherever every rank calls it;
* results must be MPI chunks (the normal case under `accelerate!(:mpi)`); elsewhere `allgather(tasks)` is `fetch.(tasks)`.

Measured on Fulda (Open MPI, 24 / 48 / 96 ranks, everything but the exchange finished, slowest rank, ms; noisy, single runs):

| per rank | gather task + `fetch` | `Dagger.allgather`, rounds (default) | `Dagger.allgather`, native |
|---|---|---|---|
| 4 KB | 10 / 18 / 48 | 5.0 / 6.3 / 19 | 7.8 / 11 / 20 |
| 16 KB | 14 / 40 / 166 | 7.5 / 22 / 121 | 14 / 18 / 39 |
| 64 KB | 63 / 209 / 408 | 34 / 43 / 258 | 17 / 28 / 171 |

So 2-7x faster than the pattern it replaces; a raw `MPI.Allgather` is still faster (1-14 ms at 4 KB), the rest is the cooperative waiting
and the (de)serialization of the contributions.

## 4d. Raw transport of structs and the scheduler wake-up (always on)

* **Structs with bit arrays travel as raw buffers.** A concrete struct that holds at least one plain `Array` of bits (directly or inside an
  immutable struct field) is sent between ranks as raw buffers plus a small header instead of through `serialize`; `String`, `Dict`,
  `Vector{Any}` and other fields ride in the header as before, a value in which one array is reachable twice stays on the old path (the
  sharing survives), arrays of arrays (`Vector{Vector{T}}`) are not covered. Hooks you define yourself (`Dagger.inplace_mpi_parts` /
  `inplace_mpi_alloc` / `inplace_mpi_build`) take precedence. 40 MB between two Fulda nodes: 580 -> 210 ms.
* **Fewer wake-ups in the scheduler.** After an enqueue only parked runners that can take the task are woken, at most one per task, and
  none for a task with an `ExactScope`. One process, 64 threads: unscoped 3.3k -> 5.6k tasks/s, pinned 5.5k -> 7.2k. The ceiling of one
  process stays at ~7-10k tasks/s (about 140 us of serial work per task in the submitting thread): tasks shorter than about
  `threads x 140 us` cannot keep all threads busy, so use coarser tasks or several ranks per node.

## 5. Measured costs (4 ranks x 16 threads, one node)

| | plain | with both switches |
|---|---:|---:|
| task this rank runs (pinned) | ~160 us | ~160 us |
| task this rank only plans (foreign) | ~170 us | ~20 us |
| dependent task (one link of a chain) | ~290 us (was ~410) | same |
| extra per cross-rank hop | +~40 us | same |
| rank-local task | ~160 us | ~160 us |

## 6. Pitfalls

* The **first `fetch`** of a new task type compiles for ~1 s on slow nodes; warm up before timing. Stencil/sparse
  kernels can compile for minutes on the first call (compile latency, not scheduling).
* `wait(task)` does **not** throw for a failed task; `fetch` does.
* In a script, **do not pass non-constant globals as task arguments** (unknown type -> every task takes the slow
  protocol that broadcasts the result type); spawn from inside a function.
* "Hit probable hang on recv" is a warning after ~10 s without progress, often just a slow first compile; "Hit hang" aborts.
* Do not wait with `yield()` loops (sticky tasks starve the thread's spawned work); use `sleep` or `wait`.
* A benchmark chain pinned to thread 1 shares that thread with the driver that plans the tasks; pin to another thread.
* The repository's own MPI suite (`test/mpi/cpu.jl`) takes about 43 minutes at 2 ranks x 2 threads on the Fulda nodes (Open MPI) and passes there on
  `integration/mpi-all` (1263 and 1270 assertions on the two ranks at `a3d4650`, 1315 and 1323 with the allgather and struct-transport tests, no failures); long silent phases (first-call compilation of the stencil and sparse
  kernels, minutes) print "Hit probable hang on recv" warnings that are only timers. Give a job at least an hour, and do not cancel on the warning.

## 7. What to report back when something looks wrong

`git rev-parse HEAD` of the Dagger checkout, which switches were set, ranks x threads, the smallest program that
shows it, and the log lines containing `Non-uniform value`, `Hit probable hang` or `Hit hang`.

## 8. Where each change lives (all on the fork `AnnaLena77/Dagger.jl`)

`perf/mpi-constants` (placement cache, kick loop, stale-cache fix) · `perf/sched-cost-estimate` ·
`fix/mpi-space-hash` · `feat/mpi-owner-elision` (elision + free threads) · `feat/mpi-comm-self` (rank-local graphs) ·
`perf/mpi-pin-runner` (pinned MPI tasks) · `fix/mpi-reftid-leak` · `feat/mpi-local-datadeps` ·
`feat/mpi-elide-scoped-chunks` · `feat/mpi-elide-darray-blocks` (elision for scope-restricted chunks and `DArray` blocks) ·
`feat/mpi-group-graphs` (group graphs, section 4b) · `perf/mpi-elide-typed-proc` (13-14 % cheaper planning of a task this rank only plans) · `bench/tasks-suite` (the `tasks` benchmark suite, `benchmark/suites/tasks.jl`) · `integration/mpi-all` (all of it).
