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
* every argument is a plain value, an `MPIRef` chunk without a scope restriction, or an earlier task whose owner is known;
* the function's **return type is concrete** (write type-stable functions; a task whose result type cannot be inferred
  needs the type broadcast to everyone);
* uniformity checking is off (`Dagger.check_uniformity!(true)` disables elision) and it is not a Datadeps task.

Tasks **without** a scope are placed at spawn time by a deterministic rule (the owner of the first MPI argument is
preferred, otherwise rotation by task id) and can then be elided on the other ranks like scoped ones.

Failure semantics change slightly: for such a task the owner reports **only failures**. A rank learns of a failure when it
later needs the task's data (an argument move or a `fetch`); a rank that never touches the result never sees it.
`fetch` stays uniform: every rank takes the same ids.

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
* Known unrelated failures of the repository's own MPI suite on Open MPI at 2 x 2: the `check_uniform` test stalls
  after its deliberate failure, and the stencil tests need many minutes of first-call compilation.

## 7. What to report back when something looks wrong

`git rev-parse HEAD` of the Dagger checkout, which switches were set, ranks x threads, the smallest program that
shows it, and the log lines containing `Non-uniform value`, `Hit probable hang` or `Hit hang`.

## 8. Where each change lives (all on the fork `AnnaLena77/Dagger.jl`)

`perf/mpi-constants` (placement cache, kick loop, stale-cache fix) · `perf/sched-cost-estimate` ·
`fix/mpi-space-hash` · `feat/mpi-owner-elision` (elision + free threads) · `feat/mpi-comm-self` (rank-local graphs) ·
`perf/mpi-pin-runner` (pinned MPI tasks) · `fix/mpi-reftid-leak` · `feat/mpi-local-datadeps` ·
`bench/tasks-suite` (the `tasks` benchmark suite, `benchmark/suites/tasks.jl`) · `integration/mpi-all` (all of it).
