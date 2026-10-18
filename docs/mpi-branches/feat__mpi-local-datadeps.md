# `feat/mpi-local-datadeps`

> `Dagger.spawn_datadeps()` works inside a rank-local graph without disturbing the world.

**Builds on:** fix/mpi-reftid-leak

## Why

Datadeps inside a rank-local graph plans on the driver, outside any task, and took its generic ref ids from the world's counter. The ranks run different numbers of local operations, so the world's uniform ids and tags stopped agreeing: after local Datadeps regions a world task got id 31 on one rank and 42 on the other, and the next world Datadeps region failed with 'Non-uniform value'.

## What it adds

* When the current acceleration is a rank-local one, those ids come from the acceleration's own counter (above the tag range) and the collective uniformity check is skipped for them.

## Measured effect

Correctness: a different number of regions and tasks per rank, overlapping views, world ids and a following world Datadeps region all work (the test failed before, passes after).

## Tests

`test/mpi/comm_self.jl` 'Datadeps inside a rank-local graph'.

## Limits and notes

CPU only; GPU chunks in local graphs not covered.

## Main files

`ext/MPIExt.jl`, `test/mpi/comm_self.jl`, `AGENTS.md`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
