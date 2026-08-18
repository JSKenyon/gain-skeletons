# gain-skeletons

`gain_skeletons` builds mock [xarray](https://docs.xarray.dev/) datasets that scaffold
radio interferometric calibration ("gain") solutions, writes them to
[zarr](https://zarr.dev/), and reads them back.

**This is a demonstrator, not production software.** Every array value it produces is
randomly generated; nothing here computes or applies calibration. It exists so you can see
what such datasets look like on disk and how their coordinate axes behave, before any
production schema is settled.

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,notebook]"
```

## Minimal example

```python
import gain_skeletons as gs

xds = gs.make_gain_xds("bandpass")
print(xds.GAIN.dims, xds.GAIN.shape, xds.GAIN.dtype)
# ('time', 'antenna_name', 'frequency', 'receptor_label') (4, 8, 64, 2) complex64
```

## The three layers

- **`axes.py`** — one coordinate factory per axis (`time_coord`, `frequency_coord`, and so
  on), each returning a standalone `xr.DataArray` with MSv4-style attributes.
- **`spec.py` and `registry.py`** — `spec.py` defines `ParamSpec` and `CalSpec`, the
  frozen, self-validating dataclasses that declare a calibration type without any code of
  its own; `registry.py` holds the catalogue of `CalSpec` objects, built on top of them.
- **`builder.py`** — `make_gain_xds` and `make_split_gain_xds` turn any `CalSpec`
  (registered or hand-built) into datasets. It imports `registry.py`, so it sits alone on
  top rather than sharing a layer with it.

## Two layouts

Every calibration type can be built two ways, and neither is privileged as the correct
one. Both give one solve one dataset with one `FLAG`; they differ in how that dataset
holds the parameters:

- **`make_gain_xds`** puts every parameter into one data array, indexed by an explicit
  `parameter_label` axis. This keeps the parameters needed to evaluate a Jones term
  adjacent in memory and, once written, in one chunked zarr array rather than several
  that chunk and compress independently. Putting a polarised and an unpolarised parameter
  into the same array means broadcasting the unpolarised one redundantly over
  `receptor_label` — visible in `fringe_fit`, where `DISP_DELAY` is broadcast this way —
  and units that vary between parameters move from a scalar attribute to a
  `parameter_units` coordinate.
- **`make_split_gain_xds`** gives each parameter its own data array within that dataset,
  named for the parameter, with its own exact axes and a scalar `units` attribute. Nothing
  is broadcast and no parameter is padded out over an axis it does not need. The cost is
  that the quantities are no longer adjacent.

For the nine calibration types with a single parameter, the two functions produce
identical datasets — `tests/test_builder_split.py` pins this. The layouts diverge for
`delay` and `fringe_fit`, the two types with several differently-united parameters, and
only `fringe_fit` pays the broadcasting cost, since both of `delay`'s parameters are
polarised.

`parameter_label` does two jobs, and only one of them survives splitting. Where it
distinguishes components within a single parameter — `antenna_positions`' `dX`, `dY` and
`dZ` — it is present in both layouts. Where it would merely restate an array's own name,
as it would for each of `fringe_fit`'s four quantities, the split layout drops it: the
array name already carries that.

## Flagging

Every dataset carries one boolean `FLAG`, in both layouts. It never carries
`parameter_label` or `receptor_label`, because those two index the components of a single
solution rather than distinct solutions — the quantities one solve produced, and the
receptors it solved together. A solution whose one component is untrustworthy is not a
solution you can use the rest of. `time`, `antenna_name`, `frequency` and `direction` do
index separate solutions, so `FLAG` keeps them:

```python
gs.make_gain_xds("bandpass").FLAG.dims
# ('time', 'antenna_name', 'frequency')
```

## The registry

| Key | Conventional | dtype | Parameters | Direction-dependent | Notes |
| --- | --- | --- | --- | --- | --- |
| `phenomenological_gain` | `J` | complex64 | `GAIN` (rel), labels: aligned, cross | no | full Jones; channel-resolved |
| `antenna_gain` | `G` | complex64 | `GAIN` (rel) | no | on-diagonal only; single channel |
| `tropospheric_gain` | `T` | complex64 | `GAIN` (rel) | no | scalar, unpolarised; single channel |
| `opacity` | | float64 | `OPAC` (nepers) | no | unpolarised; single channel |
| `bandpass` | `B` | complex64 | `GAIN` (rel) | no | on-diagonal only; channel-resolved |
| `leakage` | `D` | complex64 | `GAIN` (rel) | no | off-diagonal only; channel-resolved |
| `delay` | `K` | float64 | `PHASE` (deg), `DELAY` (s) | no | multi-parameter; both polarised |
| `antenna_positions` | | float64 | `ANTENNA_POSITION_OFFSET` (m), labels: dX, dY, dZ | no | no frequency or receptor axis |
| `fringe_fit` | | float64 | `PHASE` (deg), `DELAY` (s), `RATE` (s/s), `DISP_DELAY` (s, unpolarised) | no | multi-parameter; mixes polarised and unpolarised |
| `dd_phenomenological_gain` | | complex64 | `GAIN` (rel), labels: aligned, cross | yes | full Jones; single channel |
| `ionosphere` | | float64 | `TEC` (TECU) | yes | no frequency or receptor axis |

Keys are spelled out rather than abbreviated, since the conventional single letters carry
their meaning only by convention. The second column gives the letter where one is in
common use, for anyone arriving from CASA or AIPS; it is not accepted as a key.

The catalogue is illustrative, not exhaustive — it covers the range of coordinate shapes
these datasets take. Look up any entry with `gs.get_spec("fringe_fit")`, or list all keys
with `gs.list_cal_types()`. A hand-written `CalSpec` works everywhere a registered one
does, so a type the registry does not carry needs no change to the package.

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

## Further reading

[`notebooks/gain_skeletons_demo.ipynb`](notebooks/gain_skeletons_demo.ipynb) — a guided
tour through the coordinate factories, several calibration types, both layouts side by
side, and a zarr round-trip.
