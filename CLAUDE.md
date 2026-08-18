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
python -m pytest tests/test_registry.py::test_fringe_fit_parameters_match_catalogue -v

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
`builder.py` is a generic interpreter over any `CalSpec` — it has no branch for
`antenna_gain` versus `bandpass` versus `fringe_fit`. Adding a calibration type means adding a `CalSpec` to
`registry.py`; it should never require touching `builder.py`. If you find yourself editing
a builder to special-case a new type, stop — the type belongs in the registry with the
right axes and `default_sizes`, not as a new code path.

## One array per quantity, and why

`make_gain_xds` is the only builder. It gives one solve one `xr.Dataset`: one data array
per parameter, named for the parameter, each with exactly the axes its `ParamSpec` declares
and a scalar `units` attribute, plus one `FLAG`.

A consolidated layout — every quantity stacked into one array indexed by `parameter_label`
— existed alongside this one and was removed deliberately. It forced units out of a scalar
attribute and into either a `parameter_units` coordinate or a per-label mapping in attrs.
The coordinate worked but meant a reader had to check two places depending on the data; the
mapping did not subset, so after `.sel(parameter_label="DELAY")` the attrs still described
all four labels. One array per parameter means one unit per array, which makes the whole
question disappear. Do not reintroduce a layout that stacks differently-united quantities
into one array.

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

## `FLAG` never carries `parameter_label` or `receptor_label`

A flag marks a whole solution bad, and these two axes index the components of one solution
rather than distinct solutions: the components a single quantity is made of —
`antenna_positions`'s `dX` versus its `dY` — and the receptors solved together. If one
component of a solution cannot be trusted, neither can the rest of it. The pair lives in
`builder.UNFLAGGED_AXES`; `time`, `antenna_name`, `frequency` and `direction` index
genuinely separate solutions and stay.

There is exactly one `FLAG` per dataset, however many arrays that dataset holds. Its
dimensions are every axis some parameter uses, less the component axes, so a quantity
defined over fewer axes than its neighbours — `fringe_fit`'s unpolarised `DISP_DELAY` — is
still covered. `tests/test_builder.py` spells the expected dimensions out rather than
deriving them from `UNFLAGGED_AXES`, so widening that tuple cannot quietly widen its own
test.

## zarr specifics

Always pass `consolidated=False` on both `to_zarr` and `open_dataset(engine="zarr")`. Zarr
format 3 does not specify consolidated metadata: omitting the flag on write triggers one
warning, and omitting it on read triggers another — a `RuntimeWarning`, since it goes
looking for metadata that was never written. There is deliberately no I/O wrapper around
either call.

## `parameter_label` has one job

It distinguishes components within one parameter — `antenna_positions`' `dX` from its `dY`,
the two columns of a Jones term. It never distinguishes one parameter from another, because
each parameter is its own array carrying its own name. So `delay` and `fringe_fit`, the two
multi-parameter entries, declare no parameter axis at all; only the three single-parameter
entries with several components do.

Because a dataset holds one `parameter_label` dimension with one coordinate, every
parameter declaring that axis indexes the same labels, and `CalSpec.__post_init__` requires
them to agree. Note the direction: the rule is agreement, not uniqueness. Two parameters
sharing the axis must ask for identical labels; asking for different ones describes an axis
that cannot exist. `CalSpec.parameter_labels` reports the agreed labels, or `None` when no
parameter declares the axis.

## Scope

Everything here is a demonstrator with random values. Nothing computes, applies, or
validates calibration, and no dataset this package writes should be treated as a schema
anyone has committed to. The registry is illustrative rather than exhaustive.
