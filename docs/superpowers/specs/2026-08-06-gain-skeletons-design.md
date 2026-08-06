# gain-skeletons: design

**Date:** 2026-08-06
**Status:** approved

## Purpose

`gain_skeletons` builds mock xarray datasets that scaffold radio interferometric
calibration ("gain") solutions, writes them to zarr, and reads them back. It exists so
that users can see what such datasets look like on disk and how their coordinate axes
behave, before any production schema is settled.

This is a demonstrator, not production software. All array values are randomly
generated; nothing here computes or applies calibration.

Source material: George Moellenbrock, *Calibration Dataset Coordinate Dimensions*,
2026-07-30, archived at
`docs/reference/calibration-dataset-coordinate-dimensions.pdf`. Slides 6 and 7 give the
catalogue of calibration types that the registry transcribes.

## Non-goals

- No calibration mathematics: no Jones matrix evaluation, no application to visibilities.
- No dependency on `xradio`. The package borrows MSv4 *naming*, not MSv4 machinery.
- No schema validation framework. The registry is data, and tests check it against the
  slides.
- No dask. Datasets are small and numpy-backed.

## Design principles taken from the deck

Slide 5 states four computing principles, which this package honours as follows:

| Principle | How it is honoured |
| --- | --- |
| Alignment with XRADIO MSv4 | Coordinate names and coordinate attributes follow MSv4 vocabulary exactly. |
| Adopted coordinate dimensions impose common mutual sampling on all axes | A dataset has one set of coordinates; every array in it indexes those same coordinates. |
| Uniform units per data array | Preserved wherever it holds. See [Units](#units) for the one case where it cannot. |
| Omit unnecessary axes per calibration type | Axis *presence* is per calibration type. See [Optional axes](#optional-axes). |

## Architecture

Three layers, in dependency order. Each is usable without the ones above it.

```
axes.py       coordinate factories        (no internal dependencies)
    ^
spec.py       ParamSpec, CalSpec          (validation only)
registry.py   the slide 6-7 catalogue
    ^
builder.py    make_gain_xds, make_split_gain_xds
```

`__init__.py` re-exports the public names. There is no I/O module: writing and reading are
`xds.to_zarr(path, consolidated=False)` and
`xr.open_dataset(path, engine="zarr", consolidated=False)`, called directly by the
notebook and tests. Wrapping a one-liner would obscure the very thing the package sets out
to show.

`consolidated=False` is required on **both** calls, not just the write. Zarr format 3 does
not specify consolidated metadata: omitting the flag on write makes xarray emit a warning
about writing metadata that format 3 does not support, and omitting it on read makes
xarray emit a *different* warning — a `RuntimeWarning`, not a `UserWarning` — because it
goes looking for consolidated metadata that was never written and does not find it. Both
warnings are silenced only by passing the flag on the corresponding call; passing it on
just one leaves the other's warning live.

### `axes.py` — coordinate factories

One factory per axis. Each returns an `xr.DataArray` carrying MSv4-style coordinate
attributes, so a factory's output can be dropped straight into a dataset or inspected on
its own.

| Factory | Values | Defaults |
| --- | --- | --- |
| `time_coord(n, start=..., interval=8.0)` | Seconds since the Unix epoch, `start + interval * arange(n)` | `interval=8.0` s |
| `frequency_coord(n, start=856e6, end=1712e6)` | `linspace(start, end, n)` in Hz | MeerKAT L-band |
| `antenna_name_coord(n, prefix="m")` | `f"{prefix}{i:03d}"` | `m000`, `m001`, ... |
| `receptor_label_coord(labels=("X", "Y"))` | Receptor labels verbatim | dual linear |
| `parameter_label_coord(labels)` | Parameter labels verbatim | no default |
| `direction_coord(n)` | `arange(n)`, integer index | slide 8: "indexed directions" |

Ranged factories are `linspace`-based with overridable endpoints, as required. `n=1` is
valid and yields the range start, not its midpoint.

Coordinate attributes follow MSv4: `time` carries `type="time"`, `units="s"`,
`scale="utc"`, `format="unix"`; `frequency` carries `type="spectral_coord"`,
`units="Hz"`, `observer="topo"`. Label axes carry only `long_name`.

Defaults are chosen so output resembles real MeerKAT data. This is cosmetic but makes
the notebook legible.

### `spec.py` — the declarative model

```python
@dataclass(frozen=True)
class ParamSpec:
    name: str                              # e.g. "DELAY"
    units: str                             # e.g. "s"
    axes: tuple[str, ...]                  # axes this parameter is defined over
    dtype: str                              # "complex64" or "float64"
    labels: tuple[str, ...] | None = None  # parameter_label values; None means (name,)
    scale: float = 1.0                     # magnitude hint for random float generation

@dataclass(frozen=True)
class CalSpec:
    name: str
    parameters: tuple[ParamSpec, ...]
    default_sizes: Mapping[str, int]        # required for every sized axis in use
    consolidated_name: str | None = None    # array name in consolidated layout; None
                                             # defaults to the sole parameter's name
    jones_structure: str | None = None      # "diagonal" | "off-diagonal" | "scalar" | "full"
    description: str = ""
```

`dtype` lives on `ParamSpec`, not `CalSpec`. A single dtype on `CalSpec` would apply to
every parameter uniformly, which makes the mixed-dtype `ValueError` documented under
[Trade-off between the layouts](#trade-off-between-the-layouts) unreachable: nothing could
ever construct the calibration type the error is supposed to reject. Putting `dtype` on
`ParamSpec` lets one `CalSpec` hold parameters of different dtypes, so the check has
something to check.

`CalSpec` validates on construction: axis names must be known, axes must be given in
canonical order, a parameter declaring `labels` longer than one must include
`parameter_label` in its `axes`, labels must be unique across the whole `CalSpec`, a
multi-parameter type must set `consolidated_name`, and `default_sizes` must correspond
exactly to the sized axes in use. It does *not* validate that a multi-parameter type has a
`parameter_label` axis to consolidate onto — that is a precondition of `make_gain_xds`, not
of a well-formed `CalSpec`, since `make_split_gain_xds` has no such need. See
[`builder.py`](#builderpy--two-layouts) below.

`CalSpec.direction_dependent` is a derived property, not a declared field: it is `True`
exactly when `"direction"` appears in the union of the parameters' axes. There is no way to
declare a `CalSpec` as direction-dependent while omitting the `direction` axis, or vice
versa, because there is nothing separate to declare.

`default_sizes` covers only the sized axes — `direction`, `time`, `antenna_name`,
`frequency` — that the calibration type actually uses, and it is *required* for every one
of them, not optional. `receptor_label` and `parameter_label` are *label* axes, not sized
ones: their extent follows from the labels they carry, not from a size a caller supplies.
`CalSpec.__post_init__` checks `default_sizes` three ways: it raises if a label axis
(`receptor_label` or `parameter_label`) appears in `default_sizes` at all, since its
extent is never configurable; it raises if `default_sizes` names an axis the calibration
type does not use, sized or not; and it raises if any sized axis the type *does* use is
missing from `default_sizes`. The extent of `parameter_label` is always `len(labels)`,
because each position along it denotes a specific named parameter, and the extent of
`receptor_label` is always `len(receptor_labels)` for the same reason.

`__post_init__` also snapshots the validated `default_sizes` into a read-only
`types.MappingProxyType` and rebinds the attribute to that snapshot via
`object.__setattr__`. `frozen=True` stops a caller from doing `spec.default_sizes = ...`,
but it does nothing to stop them mutating the dict they passed in after construction —
without the snapshot, that mutation would silently invalidate an already-validated spec.
One consequence is that `CalSpec` is not hashable: dataclass-generated `__hash__` would hash
the field tuple, and a mapping is not hashable regardless of whether it is proxied. This is
a deliberate YAGNI call rather than an oversight — nothing in the package ever puts a
`CalSpec` in a set or uses one as a dict key — and it can be revisited if that changes.

#### Where `parameter_label` values come from

`ParamSpec.labels` is the single source, which makes one rule cover both senses of
"parameter":

| Entry | Parameters | Resulting labels |
| --- | --- | --- |
| `J` | one `ParamSpec`, `labels=("aligned", "cross")` | `("aligned", "cross")` |
| `antpos` | one `ParamSpec`, `labels=("dX", "dY", "dZ")` | `("dX", "dY", "dZ")` |
| `fringefit` | four `ParamSpec`s, all `labels=None` | `("PHASE", "DELAY", "RATE", "DISP_DELAY")` |

In the consolidated layout, `parameter_label` is the concatenation of every parameter's
labels in declaration order. In the split layout, each dataset gets only its own
parameter's labels — length one for each Fringefit quantity, matching the deck's
`nPar=1`.

`ParamSpec.scale` is a magnitude hint for random float generation: a delay in seconds and
a phase in degrees should not come out looking the same size, so each float parameter
carries its own scale (see [Random data](#random-data)). Complex parameters ignore it,
since they are generated near unit amplitude regardless.

`CalSpec.consolidated_name` defaults to `None`, which `resolved_consolidated_name`
resolves to the sole parameter's name when there is exactly one parameter. This is what
makes the two layouts produce data arrays of the same name for the nine single-parameter
types — `make_gain_xds("B")` and `make_split_gain_xds("B")` both call the array `GAIN`
without either function needing to be told to. Only `fringefit`, with four parameters,
must set `consolidated_name` explicitly (to `"PARAMETER"`).

Whether `parameter_label` exists at all is declared per parameter in `axes`, exactly like
every other optional axis: `G` omits it, `J` and `antpos` and all four Fringefit
quantities include it.

The two senses of "parameter" that the deck runs together are separated here:

- **Same-unit parameters within one quantity** — `J`'s `nPar=2` (aligned, cross; both
  `[rel]`), `antpos`'s `nPar=3` (`dX, dY, dZ`; all `[m]`). One `ParamSpec` with
  `size > 1`. The deck already consolidates these along the parameter axis.
- **Different-unit quantities within one calibration type** — Fringefit's `PHASE [deg]`,
  `DELAY [s]`, `RATE [s/s]`, `DISP_DELAY [s]`. Several `ParamSpec`s.

### Canonical axis order

```
(direction, time, antenna_name, frequency, receptor_label, parameter_label)
```

Taken from the order in which the slides list dimensions. Every array the package builds
uses this order, restricted to the axes present.

### Optional axes

The deck's brace notation encodes *presence*, not extent:

- `nFreq=1` — the `frequency` axis exists and has length one.
- `{nFreq=0}` — the `frequency` axis does not exist at all.

These are materially different and the package keeps them apart. A `ParamSpec.axes`
tuple lists exactly the axes that exist; anything absent from it is genuinely absent from
the dataset — no length-one placeholder. The `nFreq=1` versus `nFreq=nCh` distinction is
carried by `CalSpec.default_sizes` instead: `G` defaults `frequency=1`, `B` defaults
`frequency=64`.

A bare `nFreq` with neither `=1` nor braces (only `J` on slide 6) denotes an arbitrary,
deliberately unspecified extent. `J` therefore defaults to `frequency=64`, matching `B`
and `D`, but nothing about the type constrains it.

### `builder.py` — two layouts

Both builders accept the same `CalSpec` (or a registry name) plus per-axis size
overrides, so switching layout is a one-line change.

| | `make_gain_xds(spec, ...)` | `make_split_gain_xds(spec, ...)` |
| --- | --- | --- |
| Returns | one `xr.Dataset` | `dict[str, xr.Dataset]`, keyed by parameter name |
| Data arrays | `spec.resolved_consolidated_name`, plus `FLAG` | one named array plus `FLAG`, per dataset |
| `parameter_label` | labels every parameter of every quantity | that quantity's own labels |
| Axes | union over all parameters, broadcast as needed | each quantity's exact axes |
| Units | `units` attr when uniform, else `parameter_units` coord | scalar `units` attr always |

Two functions rather than one `layout=` keyword, because a keyword that flips the return
type between `Dataset` and `dict` is worse to consume than two names. Neither layout is
privileged as the correct one.

**Nine of the ten registry entries declare a single `ParamSpec`.** For those the two
builders produce identical datasets, and a test pins that. The layouts diverge only for
`fringefit`.

#### Trade-off between the layouts

Recorded here because the notebook demonstrates it rather than resolving it.

Consolidated favours locality and flag correctness:

- Evaluating a Jones term needs every parameter at a given
  `(time, antenna_name, frequency, receptor_label)`. Consolidated keeps them adjacent in
  memory, and on disk as one chunked zarr array rather than several.
- A fringe fit is a *single* solve emitting all four parameters together; if it fails
  they are all bad. One `FLAG` states that. Four independent flags imply an independence
  that does not exist.

Split favours exact axes and uniform units:

- `DISP_DELAY` is unpolarised while `PHASE`, `DELAY` and `RATE` are not. Consolidating
  forces `DISP_DELAY` to broadcast redundantly over `receptor_label`. Slide 5's own
  principle — adopted dimensions *impose* common mutual sampling — arguably endorses
  exactly this, but the redundancy is real and should be visible.
- Consolidation requires a common dtype, so a calibration type mixing complex and float
  quantities cannot consolidate at all. `make_gain_xds` raises a `ValueError` naming the
  offending parameters in that case.

`make_gain_xds` also raises `ValueError` for a multi-parameter `CalSpec` with no
`parameter_label` axis. Without the guard, the consolidated fill loop has no way to tell
one parameter's slice from another's and silently keeps only the last one written — a
real bug caught during implementation, not a hypothetical. Such a spec is not invalid: it
is only unconsolidatable. `make_split_gain_xds` has no such restriction, since each
parameter gets its own dataset regardless of whether a parameter axis exists. This is why
the check lives in `make_gain_xds` rather than in `CalSpec.__post_init__` — see
[`spec.py`](#specpy--the-declarative-model) above.

### Units

Every parameter has exactly one unit. The question is only whether a *data array* can
also have exactly one.

In the split layout, and in the consolidated layout whenever all parameters share a unit,
the array carries a scalar `units` attribute and the deck's uniform-units principle holds
unchanged. That covers all nine single-quantity entries.

Only `fringefit` consolidated is genuinely heterogeneous. There the scalar `units`
attribute is **omitted** — asserting a single unit would be false — and units are carried
instead as a non-dimension coordinate `parameter_units` aligned to `parameter_label`.
This keeps units queryable and makes them travel with selections:

```python
xds.sel(parameter_label="DELAY").parameter_units.item()   # "s"
```

`parameter_units` is present only when needed, so its absence signals that `units` is
authoritative.

### Flags

`FLAG` is boolean and always present, with **the parameter array's dimensions minus
`parameter_label`**.

One rule, uniformly applied. Where there is no parameter axis, `FLAG` matches the
parameter array's dimensions exactly. The rationale is that a flag marks a *solution* as
bad, and the individual components of one solution are not independently valid — flagging
`antpos`'s `dX` while trusting its `dY` is meaningless.

### Random data

A single `numpy.random.default_rng(seed)` per build, `seed` being a keyword argument, so
every dataset is reproducible.

- Complex parameters: unit amplitude perturbed by a small normal deviate, with uniform
  random phase. A uniform-random complex gain of arbitrary magnitude would be
  physically nonsensical and would undercut the demonstration.
- Float parameters: standard normal, multiplied by the parameter's `ParamSpec.scale`, so
  a delay in seconds and a phase in degrees do not come out the same size.
- `FLAG`: Bernoulli with a `flag_fraction` keyword, default `0.05`, so flags are neither
  empty nor dominant. `flag_fraction=0.0` gives a clean dataset.

### Dataset attributes

`cal_type`, `direction_dependent`, and `jones_structure` where the slides specify one.
The `(on-diag only)`, `(off-diag only)` and `(scalar, unpol!)` annotations become
`jones_structure` values rather than extra axes or arrays: per slide 5 they describe the
*origin* of the data, not an instruction for its use.

`jones_structure` is omitted from the attributes dict entirely when the calibration type
has none — `opacity`, `antpos` and `ionosphere` — rather than stored as an explicit null.
A `jones_structure: None` attribute would round-trip through zarr as a claim that the type
was checked against the deck's Jones-matrix structure and found to have none; omission
says only that the question does not apply.

## The registry

Transcribed from slides 6 and 7. Sizes shown are defaults; all are overridable.

The registry has **ten** entries, one row per table below. `fringefit` is a single entry
holding four `ParamSpec`s, not four entries — counting `ParamSpec`s instead of registry
keys gives nine single-parameter entries plus Fringefit's four, thirteen in total, which
is a count of a different thing and not the size of the registry.

Direction-independent (slide 6):

| Key | Axes present | dtype | Parameters (units) | Notes |
| --- | --- | --- | --- | --- |
| `J` | time, antenna_name, frequency, receptor_label, parameter_label | complex64 | `GAIN` (rel), labels: aligned, cross | frequency=64; full Jones |
| `G` | time, antenna_name, frequency, receptor_label | complex64 | `GAIN` (rel) | frequency=1; diagonal |
| `T` | time, antenna_name, frequency | complex64 | `GAIN` (rel) | frequency=1; scalar, unpolarised |
| `opacity` | time, antenna_name, frequency | float64 | `OPAC` (nepers) | frequency=1; unpolarised |
| `B` | time, antenna_name, frequency, receptor_label | complex64 | `GAIN` (rel) | frequency=64; diagonal |
| `D` | time, antenna_name, frequency, receptor_label | complex64 | `GAIN` (rel) | frequency=64; off-diagonal |
| `antpos` | time, antenna_name, parameter_label | float64 | `ANTENNA_POSITION_OFFSET` (m), labels: dX, dY, dZ | no frequency, no receptor_label |
| `fringefit` | time, antenna_name, frequency, receptor_label, parameter_label | float64 | `PHASE` (deg), `DELAY` (s), `RATE` (s/s) — all polarised; `DISP_DELAY` (s) — unpolarised | frequency=1; the only multi-quantity entry |

Direction-dependent (slide 7):

| Key | Axes present | dtype | Parameters (units) | Notes |
| --- | --- | --- | --- | --- |
| `dd_gain` | direction, time, antenna_name, frequency, receptor_label | complex64 | `GAIN` (rel) | frequency=1; diagonal |
| `ionosphere` | direction, time, antenna_name | float64 | `TEC` (TECU) | no frequency, no receptor_label |

Global size defaults are small — `time=4`, `antenna_name=8`, `direction=3` — so notebook
output stays readable.

## Testing

`pytest`, tests in `tests/`.

- **Registry against source.** A parametrised table transcribed from slides 6 and 7
  *independently of* `registry.py`, asserting each entry's dimensions, dtype and units.
  The registry is checked against the deck, not against itself. This is the test that
  carries real value; the transcription is deliberately duplicated.
- **Optional axes.** For every entry, axes absent from the spec are absent from the built
  dataset — asserting absence, not length one. Complements the presence assertions above.
- **Layout equivalence.** For all nine single-quantity entries, `make_gain_xds` and
  `make_split_gain_xds` agree on dimensions, dtype, attributes and values.
- **Flag rule.** `FLAG` dimensions equal the parameter array's minus `parameter_label`,
  across every entry and both layouts.
- **Parameter labels.** `J` yields `("aligned", "cross")`, `antpos` yields
  `("dX", "dY", "dZ")`, and `fringefit` consolidated yields the four quantity names in
  declaration order, while each split Fringefit dataset yields its own single label.
- **Coordinate factories.** Endpoints honoured, lengths correct, `n=1` yields the range
  start, MSv4 attributes present.
- **Consolidated units.** `fringefit` consolidated has no scalar `units` attribute and a
  correct `parameter_units` coordinate; every other entry has the scalar attribute and no
  `parameter_units`.
- **Mixed-dtype rejection.** A hand-built `CalSpec` mixing complex and float raises
  `ValueError` from `make_gain_xds`.
- **Reproducibility.** Equal seeds give equal values; different seeds differ.
- **Zarr round-trip.** `to_zarr(consolidated=False)` then
  `open_dataset(engine="zarr", consolidated=False)` preserves dimensions, complex dtypes,
  coordinate values, string coordinates, `parameter_units`, and both dataset and variable
  attributes.

Everything is fast and in-memory; zarr round-trips use `tmp_path`. No `slow` markers
expected.

## The notebook

`notebooks/gain_skeletons_demo.ipynb` demonstrates the package; it defines no schema
logic of its own. Narrative:

1. Coordinate factories on their own, including overridden ranges.
2. `G` — the simplest case; read the repr against slide 6.
3. `B` versus `G` — `nFreq=nCh` against `nFreq=1`, same axes.
4. `antpos` — axes genuinely absent, and a parameter axis with meaningful labels.
5. `ionosphere` — the direction axis appearing.
6. `fringefit` both ways, side by side: consolidated versus split, the `DISP_DELAY`
   broadcast, `parameter_units`, and one flag versus four.
7. Round-trip to zarr and show the on-disk tree for both Fringefit layouts — one array
   against four.
8. The escape hatch: a hand-written `CalSpec` for a calibration type not in the deck.

## Packaging

- `src/` layout, package `gain_skeletons`, distribution `gain-skeletons`.
- `requires-python = ">=3.11"`; runtime dependencies `xarray`, `zarr`, `numpy`.
- Extras: `dev` (`pytest`), `notebook` (`jupyter`, `ipykernel`).
- `ruff` for lint and format, line length 100, `target-version` matching
  `requires-python`, at minimum `E501` and `I` enabled.
- Development environment is `.venv`, created with `uv venv`, populated with
  `uv pip install -e ".[dev,notebook]"`.

## Delivered alongside this design

- `README.md` — install, the minimal example, the three layers, the two layouts, and the
  registry table.
- `CLAUDE.md` — orientation for future work in this repository: the commands, the
  data-not-code registry idea, the presence-versus-extent rule, and why
  `tests/test_registry.py` duplicates `registry.py` on purpose.

Both were deliberately written once the package existed rather than guessed at up front.
Nothing from the original design remains deferred.
