# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Every Python invocation goes through the project's `.venv`; never system Python, never
`uv run`.

```bash
source .venv/bin/activate

# Full suite.
python -m pytest

# Single test.
python -m pytest tests/test_registry.py::test_fringefit_parameters_match_catalogue -v

# Lint and format.
ruff check . && ruff format .

# Execute the notebook end to end (uses the "gain-skeletons" Jupyter kernel).
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=gain-skeletons \
  notebooks/gain_skeletons_demo.ipynb
```

The suite passes clean under bare `-W error`, not just the default filters — use
`python -m pytest -W error` when checking for stray warnings. `-W error::UserWarning`
would miss a real regression: the zarr read-side warning (see below) is a
`RuntimeWarning`.

## A calibration type is data, not code

`registry.py` holds a dict of `CalSpec` objects, one per calibration type.
`builder.py` is a generic interpreter over any `CalSpec` — it has no branch for `G`
versus `B` versus `fringefit`. Adding a calibration type means adding a `CalSpec` to
`registry.py`; it should never require touching `builder.py`. If you find yourself editing
a builder to special-case a new type, stop — the type belongs in the registry with the
right axes and `default_sizes`, not as a new code path.

## Two layouts, one spec

`make_gain_xds` and `make_split_gain_xds` both consume the same `CalSpec`. They agree
exactly for the nine calibration types with a single parameter, and
`tests/test_builder_split.py` pins that agreement with `xds.identical()`. Preserving it
depends on both builders drawing parameter values, and then flags, from an identically
seeded `numpy.random.default_rng` in the same order — reorder that draw sequence in one
builder and the equivalence test will catch it by producing different random values, not
by erroring.

## Axis presence versus axis extent

An absent axis and an axis of length one are never interchangeable here.
`ParamSpec.axes` declares
which axes exist; `CalSpec.default_sizes` declares how long the sized ones (`direction`,
`time`, `antenna_name`, `frequency`) default to. Never represent an absent axis as a
length-one axis, and never assume a `default_sizes` entry implies presence — presence is
`axes`, extent is `default_sizes`.

## `tests/test_registry.py` duplicates `registry.py` on purpose

Its expected-value tables restate the catalogue independently rather than reflecting
`registry.py`. That way the test checks the registry against a statement of intent rather
than against itself, and changing a type's axes, units or dtype has to be a deliberate act
in two places. Do not "simplify" it by importing or generating either table from the
other.

## `FLAG` never carries `parameter_label`

A flag marks a whole solution bad; the components of one solution — `antpos`'s `dX` versus
its `dY`, or one Fringefit quantity versus another within the same dataset — are not
independently valid. `FLAG`'s dimensions are always the parameter array's dimensions minus
`parameter_label`, in both layouts.

## zarr specifics

Always pass `consolidated=False` on both `to_zarr` and `open_dataset(engine="zarr")`. Zarr
format 3 does not specify consolidated metadata: omitting the flag on write triggers one
warning, and omitting it on read triggers another — a `RuntimeWarning`, since it goes
looking for metadata that was never written. There is deliberately no I/O wrapper around
either call.

## Why the consolidated layout exists

The obvious layout gives each differently-united quantity its own array, so `units` stays a
scalar attribute — that is `make_split_gain_xds`. The consolidated layout trades that for
memory and disk locality: the parameters needed to evaluate one Jones term sit adjacent in
one chunked array rather than scattered across several, and one `FLAG` describes one solve.
The cost is a `parameter_units` coordinate instead of a scalar attribute, and redundant
broadcasting when a type mixes polarised and unpolarised quantities — `fringefit`'s
`DISP_DELAY` is the only case in the registry. Neither layout is the correct one; keep both
working.

## Scope

Everything here is a demonstrator with random values. Nothing computes, applies, or
validates calibration, and no dataset this package writes should be treated as a schema
anyone has committed to. The registry is illustrative rather than exhaustive.
