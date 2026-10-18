# `fix/mpi-reftid-leak`

> Memory leak fix: a plain task's ref-id counter is dropped when the task is done.

**Builds on:** perf/mpi-pin-runner

## Why

`_MPIREF_TID` holds one sub-counter per task id for the ref ids a task takes while it is moved and executed. It was reclaimed only at the end of a Datadeps region, so a program that only uses `@spawn` grew it by one entry per task for the whole session.

## What it adds

* A task that is not part of a Datadeps region (no tag option) takes ids only during its own move and `execute!`, never during planning, so its counter is dead once `execute!` returns: `mpi_cleanup_tid` deletes it there. Tagged (Datadeps) tasks keep the later cleanup, since their planning shares the key with their execution.

## Measured effect

300 plain tasks leave the table within 20 entries (before: 300).

## Tests

`test/mpi/spawn.jl` (ref-id table size after 300 plain tasks).

## Limits and notes

Always on.

## Main files

`ext/MPIExt.jl`, `test/mpi/spawn.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
