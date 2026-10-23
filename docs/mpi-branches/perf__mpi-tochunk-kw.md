# `perf/mpi-tochunk-kw`

> Every MPI task result is built without rebuilding its keyword arguments.

**Builds on:** integration/mpi-all (0f1f488)

## Why

`tochunk` always passes `type=` to `tochunk_pset`. The MPI method removed it from `kwargs` by building a new `NamedTuple` through a generator, a dynamic operation, on every call, and every MPI task result goes through it. Stock Dagger 0.22.7 has the same code (BUGS B11).

## What it changes

`ext/MPIExt.jl`: `tochunk_pset(x, space::MPIMemorySpace; device, force_nonlocal, type=nothing, kwargs...)` takes `type` as a declared keyword (core's `tochunk_pset` already does), so the remaining keywords go to `poolset` unchanged.

## Measured effect (laptop, MPICH, one rank)

| | before | after |
|---|---|---|
| `tochunk_pset(nothing, space; type=T)` | 826 ns, 17 allocations | 246 ns, 12 |
| `tochunk(nothing, proc, space; type=T)` | 1067 ns, 25 allocations | 450 ns, 20 |
| plain `@spawn` of 1000 tiny tasks, 4 threads | 36.5 us per task | 26.4 us (shared memory 21.2) |
| Fulda, MPI Cholesky 4096^2, 1 rank x 64 (separate runs) | 0.78-0.83 s | 0.72-0.75 s |

## Tests

The whole MPI suite exercises it (every result chunk); targeted subset (spawn, elision, struct transport, rank-local and group graphs) passes. Probe: `LLM_info/bench/task_overhead.jl` (not in git).

## Main files

`ext/MPIExt.jl`
