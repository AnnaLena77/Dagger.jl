# `feat/mpi-allgather`

> `Dagger.allgather(tasks)`: every rank gets the values of tasks owned by all ranks with one exchange, instead of a gather task plus `fetch`. Also: Datadeps tests in groups and the measured cost of live groups.

**Builds on:** perf/mpi-elide-typed-proc (= integration/mpi-all)

## Why

The application collects per-partition summaries with 'P tasks -> one gather task with P arguments -> `fetch`': every result moves to one rank, then the gathered vector is broadcast. `tree.summarize` was 39 / 54 / 83 ms at 26 / 52 / 104 ranks.

## What it adds

* `Dagger.allgather(tasks)` (SPMD collective, same list on every rank, inside `with_group` it works on the group): each rank contributes the values of the tasks whose result it owns; a failure travels as data so every rank throws after all took part; it takes no task or ref ids.
* Default: the library's non-blocking `MPI_Iallgather` / `MPI_Iallgatherv` (MPI.jl 0.20 wraps only the blocking one, so `MPI.API` is used), waited for cooperatively with `MPI.Test` so no thread is parked inside MPI. `JULIA_DAGGER_MPI_ALLGATHER=rounds` selects hand-written dissemination rounds with k partners per round (`MPI_ALLGATHER_FANOUT`, default 4: 3 rounds instead of 7 at 96 ranks). Elsewhere `allgather(tasks)` is `fetch.(tasks)`.
* Test of Datadeps regions inside a group; usage-guide sections 4b/4c and AGENTS lessons 89-90.

## Measured effect

Fulda (Open MPI), 24 / 48 / 96 ranks, producers finished, slowest rank, ms (noisy single runs): 4 KB per rank gather+fetch 10 / 18 / 48, rounds 5.0 / 6.3 / 19, native 7.8 / 11 / 20; 16 KB 14 / 40 / 166, rounds 7.5 / 22 / 121, native 14 / 18 / 39; 64 KB 63 / 209 / 408, rounds 34 / 43 / 258, native 17 / 28 / 171. So 2-7x faster than the pattern it replaces; expected `tree.summarize` 83 -> ~30-40 ms at 104 ranks (~4 % of the step), to be confirmed in the application. On MPICH (laptop) the rounds were 2x faster than native.

## Tests

`test/mpi/allgather.jl` (uneven owners, order, empty list, contributions from empty to 20 MB, failing task, id agreement afterwards, both paths and fanouts 1-4, in a group) at 2, 3, 4 and 7 ranks on MPICH 5.0.2 and Open MPI 5.0.10; `test/mpi/groups.jl` 'Datadeps regions inside a group'. The complete `test/mpi/cpu.jl` passes on `integration/mpi-all` at 2 ranks x 2 threads on Fulda (1263 / 1270 assertions, 43 min).

## Limits and notes

A raw `MPI.Allgather` is still faster than `Dagger.allgather` (cooperative waits, serialization, a second collective for the sizes). Call it from the driver, one call at a time. Results must be MPI chunks.

## Main files

`ext/MPIExt.jl`, `src/acceleration.jl`, `test/mpi/allgather.jl`, `test/mpi/groups.jl`, `test/mpi/cpu.jl`, `MPI_FEATURES.md`, `AGENTS.md`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
