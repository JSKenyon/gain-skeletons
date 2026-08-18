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
- **`builder.py`** — `make_gain_xds` turns any `CalSpec` (registered or hand-built) into a
  dataset. It imports `registry.py`, so it sits alone on top rather than sharing a layer
  with it.

## One array per quantity

`make_gain_xds` gives one solve one dataset: one data array per parameter, named for the
parameter, plus one `FLAG` describing the solve. Each array carries exactly the axes its
`ParamSpec` declares and a scalar `units` attribute.

That last point is why the layout is what it is. One array per parameter means one unit
per array, so units are always a scalar attribute — they never have to move to a
coordinate or a per-label mapping, and they survive subsetting for free because they
describe the array rather than positions along an axis:

```python
xds = gs.make_gain_xds("fringe_fit")
{name: xds[name].attrs["units"] for name in ("PHASE", "DELAY", "RATE", "DISP_DELAY")}
# {'PHASE': 'deg', 'DELAY': 's', 'RATE': 's/s', 'DISP_DELAY': 's'}
```

Nothing is broadcast either. `fringe_fit`'s `DISP_DELAY` is unpolarised while its three
siblings are not, so its array simply carries one axis fewer — no padding out over a
`receptor_label` axis it does not need.

The trade is locality: a type with several quantities holds them in several arrays, which
chunk and compress independently once written, rather than in one array a reader can slice
across.

`parameter_label` has exactly one job here: distinguishing components within a single
quantity, as with `antenna_positions`' `dX`, `dY` and `dZ`. It never distinguishes one
quantity from another, because the array names already do that. So the axis appears only
where a parameter declares it, and only three registry entries do.

## Flagging

Every dataset carries one boolean `FLAG`. It never carries
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
| `phenomenological_gain` | `J` | complex64 | `GAIN` (rel), labels: gain_X, gain_Y | no | full Jones; channel-resolved |
| `antenna_gain` | `G` | complex64 | `GAIN` (rel) | no | on-diagonal only; single channel |
| `tropospheric_gain` | `T` | complex64 | `GAIN` (rel) | no | scalar, unpolarised; single channel |
| `opacity` | | float64 | `OPAC` (nepers) | no | unpolarised; single channel |
| `bandpass` | `B` | complex64 | `GAIN` (rel) | no | on-diagonal only; channel-resolved |
| `leakage` | `D` | complex64 | `GAIN` (rel) | no | off-diagonal only; channel-resolved |
| `delay` | `K` | float64 | `PHASE` (deg), `DELAY` (s) | no | two arrays; both polarised |
| `antenna_positions` | | float64 | `ANTENNA_POSITION_OFFSET` (m), labels: dX, dY, dZ | no | no frequency or receptor axis |
| `fringe_fit` | | float64 | `PHASE` (deg), `DELAY` (s), `RATE` (s/s), `DISP_DELAY` (s, unpolarised) | no | four arrays; mixes polarised and unpolarised |
| `dd_phenomenological_gain` | | complex64 | `GAIN` (rel), labels: gain_X, gain_Y | yes | full Jones; single channel |
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
tour through the coordinate factories, several calibration types, flagging, and a zarr
round-trip.
