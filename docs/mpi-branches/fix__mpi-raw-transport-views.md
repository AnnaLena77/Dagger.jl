# `fix/mpi-raw-transport-views`

> An array view travels between ranks as its elements, not as the whole matrix it views.

**Builds on:** integration/mpi-all (523d0ac)

## Why

`feat/mpi-struct-raw-transport` (commit `5154237`) sends an immutable struct that holds plain bit arrays as raw buffers. A `SubArray` (and `ReshapedArray`, `PermutedDimsArray`, ...) is such a struct, but the array it holds is the **whole parent**, which it only views. So a 256 x 256 view of a 2048 x 2048 matrix travelled as all 32 MB of the parent.

`copyto!(::DArray, ::Matrix)` copies from one view per block, and the blocks of the other rank are sent, so the bytes grew with blocks x matrix size: the call went quadratic. An MPI Cholesky benchmark at 8192^2 never finished a 15-minute case because of its `copyto!`. One rank does not show it (no transfer). Stock Dagger does not have the bug: it has raw transport only for `DenseArray` and `SparseMatrixCSC` and serializes everything else, which sends a view as its elements.

## What it changes

`src/memory-spaces.jl`: `raw_has_array` excludes every `AbstractArray` subtype (it already excluded `Array`, which has its own hook). An array wrapper is not decomposed, neither at the top level nor as a field of a struct; it travels through `serialize`. Types with their own hooks (`Array`, `SparseMatrixCSC`) are unchanged.

## Measured effect (laptop, MPICH, 2 ranks x 2 threads, 256 x 256 blocks, `darray_copyto!` from a matrix)

| matrix | before | after |
|---|---|---|
| 1024^2 | 0.15 s | 0.023-0.027 s |
| 2048^2 | 2.1-2.9 s | 0.054-0.105 s |

## Tests

`test/mpi/struct_transport.jl`: a view, a reshaped view and a `PermutedDimsArray` are not decomposed; a struct holding an array and a view sends only the array raw. Fails without the change. Probe: `LLM_info/bench/darray_io.jl` (not in git).

## Main files

`src/memory-spaces.jl`, `test/mpi/struct_transport.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
