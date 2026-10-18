# `fix/mpi-space-hash`

> Bug fix: ranks that agree no longer report "Non-uniform value" under Open MPI.

**Builds on:** master

## Why

`MPIMemorySpace` hashed the raw communicator handle (`space.comm.val`). In MPICH that is a small integer, equal on every rank; in Open MPI it is a pointer, different in every process. The uniformity check hashes a `Vector{MemorySpace}` on each rank and compares the results, so ranks that agreed reported `Non-uniform value`: 19 failures at 2 ranks x 4 threads and 47 at 2 x 2 on the Fulda cluster, on unchanged master.

## What it adds

* `hash(::MPIMemorySpace)` leaves the handle out. `==` still distinguishes communicators, and a coarser hash is allowed.

## Measured effect

19 / 47 spurious failures of the MPI test suite on Open MPI -> 0. No performance change.

## Tests

The existing `test/mpi` uniformity tests; on Open MPI they failed before and pass after.

## Limits and notes

One-line fix, no switch.

## Main files

`ext/MPIExt.jl` (5 lines)

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
