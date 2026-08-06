# gain-skeletons

`gain_skeletons` builds mock [xarray](https://docs.xarray.dev/) datasets that scaffold
radio interferometric calibration ("gain") solutions, writes them to
[zarr](https://zarr.dev/), and reads them back.

**This is a demonstrator, not production software.** Every array value it produces is
randomly generated; nothing here computes or applies calibration. It exists so you can see
what such datasets look like on disk and how their coordinate axes behave, before any
production schema is settled.

The calibration types it builds are transcribed from George Moellenbrock's *Calibration
Dataset Coordinate Dimensions* (2026-07-30), archived at
[`docs/reference/calibration-dataset-coordinate-dimensions.pdf`](docs/reference/calibration-dataset-coordinate-dimensions.pdf).

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,notebook]"
```

## Minimal example

```python
import gain_skeletons as gs

xds = gs.make_gain_xds("B")
print(xds.GAIN.dims, xds.GAIN.shape, xds.GAIN.dtype)
# ('time', 'antenna_name', 'frequency', 'receptor_label') (4, 8, 64, 2) complex64
```

## The three layers

- **`axes.py`** — one coordinate factory per axis (`time_coord`, `frequency_coord`, and so
  on), each returning a standalone `xr.DataArray` with MSv4-style attributes.
- **`spec.py`** — `ParamSpec` and `CalSpec`, the frozen, self-validating dataclasses that
  declare a calibration type without any code of its own.
- **`registry.py` and `builder.py`** — `registry.py` holds the catalogue of `CalSpec`
  objects transcribed from the deck; `builder.py`'s `make_gain_xds` and
  `make_split_gain_xds` turn any `CalSpec` (registered or hand-built) into datasets.

## Two layouts

Every calibration type can be built two ways, and neither is privileged as the correct
one:

- **`make_gain_xds`** puts every parameter of a calibration type into one data array,
  indexed by an explicit `parameter_label` axis. This keeps the parameters needed to
  evaluate a Jones term adjacent in memory and, once written, in one chunked zarr array
  rather than several, and lets a single `FLAG` describe a single solve. Putting a
  polarised and an unpolarised parameter into the same array means broadcasting the
  unpolarised one redundantly over `receptor_label` — visible in `fringefit`, where
  `DISP_DELAY` is broadcast this way.
- **`make_split_gain_xds`** gives each parameter its own dataset, with its own exact axes
  and a scalar `units` attribute. Nothing is broadcast and no parameter is padded out over
  an axis it does not need. The cost is fragmentation: a calibration type produced by one
  solve, such as `fringefit`, is spread over four datasets with four independent flags.

For the nine calibration types with a single parameter, the two functions produce
identical datasets — `tests/test_builder_split.py` pins this. The layouts diverge only for
`fringefit`, the one type with several differently-united parameters.

## The registry

| Key | dtype | Parameters | Direction-dependent | Notes |
| --- | --- | --- | --- | --- |
| `J` | complex64 | `GAIN` (rel), labels: aligned, cross | no | full Jones; channel-resolved |
| `G` | complex64 | `GAIN` (rel) | no | on-diagonal only; single channel |
| `T` | complex64 | `GAIN` (rel) | no | scalar, unpolarised; single channel |
| `opacity` | float64 | `OPAC` (nepers) | no | unpolarised; single channel |
| `B` | complex64 | `GAIN` (rel) | no | on-diagonal only; channel-resolved |
| `D` | complex64 | `GAIN` (rel) | no | off-diagonal only; channel-resolved |
| `antpos` | float64 | `ANTENNA_POSITION_OFFSET` (m), labels: dX, dY, dZ | no | no frequency or receptor axis |
| `fringefit` | float64 | `PHASE` (deg), `DELAY` (s), `RATE` (s/s), `DISP_DELAY` (s, unpolarised) | no | the only multi-parameter entry |
| `dd_gain` | complex64 | `GAIN` (rel) | yes | on-diagonal only; single channel |
| `ionosphere` | float64 | `TEC` (TECU) | yes | no frequency or receptor axis |

Look up any entry with `gs.get_spec("fringefit")`, or list all keys with
`gs.list_cal_types()`.

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

## Further reading

- [`docs/reference/calibration-dataset-coordinate-dimensions.pdf`](docs/reference/calibration-dataset-coordinate-dimensions.pdf)
  — the source deck; slides 6 and 7 catalogue the calibration types the registry
  transcribes.
- [`notebooks/gain_skeletons_demo.ipynb`](notebooks/gain_skeletons_demo.ipynb) — a guided
  tour through the coordinate factories, several calibration types, both layouts side by
  side, and a zarr round-trip.
