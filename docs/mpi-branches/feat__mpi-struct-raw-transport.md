# `feat/mpi-struct-raw-transport`

> A struct that holds arrays of bits is sent between ranks as raw buffers automatically, instead of through Julia's `serialize`.

**Builds on:** `integration/mpi-all` (`e710a8d`)

## Why

A struct with an array travelled through `MPI.isend(serialize(...))`: Serialization copies the array into a fresh buffer, the receiver reads that into another fresh buffer and deserializes it into a third. Fresh memory is expensive on the Opterons, so a 40 MB struct of three arrays moved at ~80 MB/s. Dagger already had a raw-buffer path (`inplace_mpi_parts` / `inplace_mpi_alloc` / `inplace_mpi_build`), but only for dense arrays and for types that define the three hooks by hand.

## What it adds

* Types that define nothing get the raw path automatically when they qualify: a concrete struct with at least one plain `Array` of bits, directly or inside an immutable struct field (nested structs are flattened).
* The arrays travel as raw buffers; every other field (bits, `String`, `Dict`, `Vector{Any}`, ...) travels in the header, which is serialized as before, so such a field costs what it did.
* The value is rebuilt like `Serialization` does (`jl_new_structv`, no constructor call), so a type with an inner constructor round-trips unchanged.
* A value in which one array is reachable twice (a shared field) opts out, so the sharing, which `serialize` keeps, survives.
* More specific methods still win (`DenseArray`, `SparseMatrixCSC`, hooks a user defines); arrays of arrays (`Vector{Vector{T}}`) are not covered yet.

## Measured effect

Fulda (Open MPI), a struct of three arrays made on rank 0 and consumed on rank 1 (the time includes making it), median of 5, serialize -> automatic raw buffers: two nodes 0.4 MB 6.6 -> 3.1 ms, 4 MB 53.4 -> 19.5 ms, 40 MB 580 -> 210 ms; one node 4 MB 37.3 -> 13.4 ms, 40 MB 580 -> 174 ms (0.4 MB is dominated by task latency, noise). Equal to defining the three hooks by hand (172 / 208 ms at 40 MB). Laptop, MPICH: 40 MB 119 -> 84 ms.

## Tests

`test/mpi/struct_transport.jl` at 2 and 3 ranks on MPICH 5.0.2 and Open MPI 5.0.10 (24 assertions): the decomposition, an eligible struct, a struct with a `String` and a `Dict`, a nested struct, a shared array (sharing must survive), an inner constructor with an invariant; a struct without arrays stays on the old path. `comm_self`, `elide`, `spawn`, `groups`, `allgather`, `free_threads` pass at 3 ranks.

## Limits and notes

Applies to values moved between ranks as task arguments or results; `fetch` of a result to every rank uses a different path. The header serializes the non-array fields as before, so a struct that is mostly a huge `Vector{Any}` gains nothing. Benchmark: `LLM_info/bench/bundle_transport.jl`.

## Main files

`src/memory-spaces.jl`, `test/mpi/struct_transport.jl`, `test/mpi/cpu.jl`
