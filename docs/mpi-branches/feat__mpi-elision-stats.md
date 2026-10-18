# `feat/mpi-elision-stats`

> `Dagger.elision_stats()`: counters that say why a task was or was not elided.

**Builds on:** perf/mpi-rank-scope-cost

## Why

A task planned in full on a rank that neither runs it nor owns its data costs ~170 us instead of ~20 us, and the only way to see why was to read `elide_task!`.

## What it adds

* Every task that reaches `elide_task!` is counted once per outcome with one atomic increment: `elided`, `own_task`, `no_owner_known`, `not_failure_only`, `syncdeps_or_restore`, `return_type_not_concrete`, `function_not_plain`, `argument_not_positional`, `argument_participates`, `scope_conflict`, `not_applicable`.
* `Dagger.elision_stats()` returns a NamedTuple, `Dagger.reset_elision_stats!()` resets (generic in core, nothing for accelerations that do not elide). Counts accumulate only while the elision switch is on.

## Measured effect

Used on Barnes-Hut (1M bodies) to measure the planned tasks per rank and step: 201 / 193 / 185 / 910 / 1613 / 2764 at 2 / 4 / 8 / 26 / 52 / 104 ranks; none of the 'not elided because' reasons occurs there.

## Tests

`test/mpi/elide.jl` 'elision statistics ...' (an elided task, a non-concrete return type, a task that reads data the rank owns, every task counted once; 3 ranks x 2 threads).

## Limits and notes

Cumulative: call `reset_elision_stats!()` at the start of a step and `elision_stats()` at its end.

## Main files

`ext/MPIExt.jl`, `src/acceleration.jl`, `test/mpi/elide.jl`

See `docs/mpi-branches/README.md` (on `integration/mpi-all`) for how the branches fit together.
