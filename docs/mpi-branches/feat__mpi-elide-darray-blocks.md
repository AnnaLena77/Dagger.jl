# `feat/mpi-elide-darray-blocks`

> Elision for tasks that read a block of a `DArray`: the rank of every block is recorded when the array is placed.

**Builds on:** feat/mpi-elide-scoped-chunks

## Why

Under MPI the tasks that make the blocks of a `DArray` run inside a Datadeps region (they carry a tag, are not elidable and have no owner), so every task reading a block counted its argument as of unknown owner and was planned in full by every rank: 181 us per task on a rank that neither runs it nor owns the block, against 28 us for a `per_rank` chunk (the application had to stop using `DArray`).

## What it adds

* `DTask.placed` (where a result lives) is set by the placement code through a new hook `note_chunk_placement!` (called by `emit_chunk_tasks!` and the staging path of `darray.jl`; no-op by default), and `mpi_arg_participates` / `mpi_arg_owner` read it. It is separate from `DTask.owner` on purpose: `owner` also switches `fetch` to the failure-only path, which is wrong for Datadeps tasks.
* A task without a scope uses a scope-restricted chunk's rank as its placement preference, after checking that the chosen processor is inside every chunk argument's scope.

## Measured effect

DArray block argument, foreign-rank planning, 4 ranks x 8 threads: 181 -> 35.9 us. Barnes-Hut 1M bodies, 16 ranks: `--rank_blocks` no longer matters (it gave -8 % before).

## Tests

`test/mpi/elide.jl` 'DArray blocks ...' (recorded rank equals the holder, consumers elided on all other ranks, a task on a third rank, uniform ids; 3 ranks x 2 threads).

## Limits and notes

Requires the elision switch. When probing, take the block's owner from the real chunk (`fetch(t; raw=true).handle.rank`), not a guessed round robin.

## Main files

`src/dtask.jl`, `src/acceleration.jl`, `src/procgrid.jl`, `src/array/darray.jl`, `ext/MPIExt.jl`, `test/mpi/elide.jl`, `AGENTS.md` (lesson 86)

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
