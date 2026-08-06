# gain-skeletons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `gain_skeletons`, a demonstrator package that constructs mock xarray datasets scaffolding radio interferometric calibration solutions, writes them to zarr, and reads them back.

**Architecture:** Three layers in dependency order — `axes.py` (coordinate factories), `spec.py` + `registry.py` (a declarative catalogue transcribed from the source deck), and `builder.py` (two builders producing random-filled datasets). No I/O module: zarr access is `xds.to_zarr(path)` and `xr.open_dataset(path, engine="zarr")` called directly.

**Tech Stack:** Python 3.12, xarray 2026.7, zarr 3.3, numpy 2.5, pytest, ruff, jupyter.

**Spec:** `docs/superpowers/specs/2026-08-06-gain-skeletons-design.md`. Read it before starting. Source material is `docs/reference/calibration-dataset-coordinate-dimensions.pdf` (slides 6 and 7 hold the catalogue).

## Global Constraints

- Virtual environment is `.venv`, already created. **Every** Python invocation must use it: `source .venv/bin/activate` first. Never use system Python, never `uv run`.
- Use `uv pip`, never bare `pip`.
- `requires-python = ">=3.11"`; ruff `target-version = "py311"`; ruff `line-length = 100`; ruff lint selects at minimum `E501` and `I`.
- Python 3.12 is in the venv but the floor is 3.11, so builtin generics and `X | Y` unions are available without `from __future__ import annotations`. Do not add that import.
- Google-style docstrings on public classes, methods and functions, with `Args:`, `Returns:`, `Raises:` as appropriate.
- 4-space indent, trailing commas in multi-line collections and signatures, f-strings for all interpolation including logger calls.
- Floats in tests use `numpy.testing` or `pytest.approx`, never `==`.
- Canonical axis order, used everywhere without exception:
  `("direction", "time", "antenna_name", "frequency", "receptor_label", "parameter_label")`
- Run `ruff check` and `ruff format` before every commit.
- Pass `consolidated=False` on **both** write and read: `xds.to_zarr(path, consolidated=False)` and `xr.open_dataset(path, engine="zarr", consolidated=False)`. Zarr format 3 does not specify consolidated metadata, so xarray emits a `ZarrUserWarning` if asked to write it, and a `RuntimeWarning` if asked to look for metadata that was never written. Omitting it on the read side is easy to miss because the warning is a `RuntimeWarning`, not a `UserWarning`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/gain_skeletons/__init__.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `gain_skeletons` with `__version__: str`. A working `pytest` and `ruff` setup that all later tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/test_package.py`:

```python
"""Tests that the package is installed and importable."""

import gain_skeletons


def test_package_exposes_version():
    assert isinstance(gain_skeletons.__version__, str)
    assert gain_skeletons.__version__
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_package.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gain_skeletons'`.

- [ ] **Step 3: Write minimal implementation**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gain-skeletons"
version = "0.1.0"
description = "Mock xarray/zarr dataset scaffolds for radio interferometric gain solutions"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "xarray",
    "zarr>=3",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]
notebook = [
    "ipykernel",
    "jupyter",
]

[tool.hatch.build.targets.wheel]
packages = ["src/gain_skeletons"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `src/gain_skeletons/__init__.py`:

```python
"""Mock xarray/zarr dataset scaffolds for radio interferometric gain solutions.

This package is a demonstrator. Every array value it produces is randomly
generated; nothing here computes or applies calibration.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

Create a placeholder `README.md` so the build backend can resolve `readme` (Task 9 fills it in):

```markdown
# gain-skeletons

Mock xarray/zarr dataset scaffolds for radio interferometric gain solutions.
```

- [ ] **Step 4: Install and run the test**

```bash
source .venv/bin/activate
uv pip install -e ".[dev,notebook]"
python -m pytest tests/test_package.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify lint is clean**

```bash
source .venv/bin/activate && ruff check . && ruff format --check .
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md src/gain_skeletons/__init__.py tests/test_package.py
git commit -m "Add project scaffolding for gain_skeletons"
```

---

### Task 2: Coordinate factories

**Files:**
- Create: `src/gain_skeletons/axes.py`
- Create: `tests/test_axes.py`
- Modify: `src/gain_skeletons/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `CANONICAL_AXES: tuple[str, ...]` — the six axis names in canonical order.
  - `SIZED_AXES: tuple[str, ...]` — `("direction", "time", "antenna_name", "frequency")`, the axes whose extent is a free integer.
  - `LABEL_AXES: tuple[str, ...]` — `("receptor_label", "parameter_label")`, the axes whose extent follows from their labels.
  - `time_coord(size: int, *, start: float = 1_700_000_000.0, interval: float = 8.0) -> xr.DataArray`
  - `frequency_coord(size: int, *, start: float = 856e6, end: float = 1712e6) -> xr.DataArray`
  - `antenna_name_coord(size: int, *, prefix: str = "m") -> xr.DataArray`
  - `direction_coord(size: int) -> xr.DataArray`
  - `receptor_label_coord(labels: Sequence[str] = ("X", "Y")) -> xr.DataArray`
  - `parameter_label_coord(labels: Sequence[str]) -> xr.DataArray`
  - `sorted_axes(axes: Iterable[str]) -> tuple[str, ...]` — returns `axes` in canonical order; raises `ValueError` on an unknown name.
  - `SIZED_AXIS_FACTORIES: dict[str, Callable[..., xr.DataArray]]` — maps each sized axis name to its factory.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_axes.py`:

```python
"""Tests for the coordinate factories."""

import numpy as np
import pytest

from gain_skeletons.axes import (
    CANONICAL_AXES,
    antenna_name_coord,
    direction_coord,
    frequency_coord,
    parameter_label_coord,
    receptor_label_coord,
    sorted_axes,
    time_coord,
)


def test_canonical_axes_order():
    assert CANONICAL_AXES == (
        "direction",
        "time",
        "antenna_name",
        "frequency",
        "receptor_label",
        "parameter_label",
    )


def test_time_coord_is_regularly_spaced_from_start():
    coord = time_coord(4, start=100.0, interval=8.0)
    assert coord.dims == ("time",)
    np.testing.assert_allclose(coord.values, [100.0, 108.0, 116.0, 124.0])


def test_time_coord_carries_msv4_attributes():
    coord = time_coord(2)
    assert coord.attrs["type"] == "time"
    assert coord.attrs["units"] == "s"
    assert coord.attrs["scale"] == "utc"
    assert coord.attrs["format"] == "unix"


def test_frequency_coord_spans_the_requested_range():
    coord = frequency_coord(3, start=1.0e9, end=2.0e9)
    assert coord.dims == ("frequency",)
    np.testing.assert_allclose(coord.values, [1.0e9, 1.5e9, 2.0e9])


def test_frequency_coord_carries_msv4_attributes():
    coord = frequency_coord(2)
    assert coord.attrs["type"] == "spectral_coord"
    assert coord.attrs["units"] == "Hz"
    assert coord.attrs["observer"] == "topo"


# A single channel must sit at the range start, not its midpoint: a size-one
# frequency axis means "one solution for the whole band", conventionally
# labelled by where the band begins.
def test_single_channel_frequency_sits_at_range_start():
    coord = frequency_coord(1, start=856e6, end=1712e6)
    np.testing.assert_allclose(coord.values, [856e6])


def test_single_time_sits_at_range_start():
    np.testing.assert_allclose(time_coord(1, start=42.0).values, [42.0])


def test_antenna_name_coord_is_zero_padded():
    coord = antenna_name_coord(3)
    assert coord.dims == ("antenna_name",)
    assert list(coord.values) == ["m000", "m001", "m002"]


def test_antenna_name_coord_honours_prefix():
    assert list(antenna_name_coord(2, prefix="ea").values) == ["ea000", "ea001"]


def test_direction_coord_is_an_integer_index():
    coord = direction_coord(3)
    assert coord.dims == ("direction",)
    assert np.issubdtype(coord.dtype, np.integer)
    assert list(coord.values) == [0, 1, 2]


def test_receptor_label_coord_defaults_to_dual_linear():
    coord = receptor_label_coord()
    assert coord.dims == ("receptor_label",)
    assert list(coord.values) == ["X", "Y"]


def test_receptor_label_coord_accepts_circular_labels():
    assert list(receptor_label_coord(("R", "L")).values) == ["R", "L"]


def test_parameter_label_coord_preserves_label_order():
    coord = parameter_label_coord(("dX", "dY", "dZ"))
    assert coord.dims == ("parameter_label",)
    assert list(coord.values) == ["dX", "dY", "dZ"]


@pytest.mark.parametrize("size", [0, -1])
def test_sized_factories_reject_non_positive_sizes(size):
    with pytest.raises(ValueError, match="must be positive"):
        time_coord(size)


def test_parameter_label_coord_rejects_empty_labels():
    with pytest.raises(ValueError, match="at least one label"):
        parameter_label_coord(())


def test_sorted_axes_returns_canonical_order():
    assert sorted_axes(("frequency", "time", "direction")) == (
        "direction",
        "time",
        "frequency",
    )


def test_sorted_axes_rejects_unknown_axis():
    with pytest.raises(ValueError, match="unknown axis"):
        sorted_axes(("time", "baseline_id"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_axes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gain_skeletons.axes'`.

- [ ] **Step 3: Write the implementation**

Create `src/gain_skeletons/axes.py`:

```python
"""Coordinate factories for calibration dataset axes.

Each factory returns a one-dimensional :class:`xarray.DataArray` whose single
dimension is the axis name and whose attributes follow the MSv4 vocabulary, so
that a factory's output can be inspected on its own or dropped straight into a
dataset.

Sized axes take an integer extent and generate values with ``linspace`` or
``arange``, with overridable endpoints. Label axes take their values verbatim.
"""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import xarray as xr

# Canonical axis order, taken from the order in which the source deck lists
# dimensions on slides 6 and 7. Every array this package builds orders its
# dimensions this way, restricted to the axes actually present.
CANONICAL_AXES = (
    "direction",
    "time",
    "antenna_name",
    "frequency",
    "receptor_label",
    "parameter_label",
)

# Axes whose extent is a free integer, versus axes whose extent follows from
# the labels they carry.
SIZED_AXES = ("direction", "time", "antenna_name", "frequency")
LABEL_AXES = ("receptor_label", "parameter_label")

# Default time origin. A fixed constant rather than the current clock, so that
# every dataset this package builds is byte-for-byte reproducible.
DEFAULT_TIME_START = 1_700_000_000.0

# MeerKAT L-band, so that generated datasets resemble real data.
DEFAULT_FREQUENCY_START = 856.0e6
DEFAULT_FREQUENCY_END = 1712.0e6


def _check_size(axis: str, size: int) -> None:
    """Validate the requested extent of a sized axis.

    Args:
        axis: Name of the axis, used in the error message.
        size: Requested extent.

    Raises:
        ValueError: If size is not a positive integer.
    """
    if size < 1:
        raise ValueError(f"{axis} size must be positive, got {size}")


def time_coord(
    size: int,
    *,
    start: float = DEFAULT_TIME_START,
    interval: float = 8.0,
) -> xr.DataArray:
    """Build a regularly sampled time coordinate.

    Args:
        size: Number of time samples.
        start: Time of the first sample, in seconds since the Unix epoch.
        interval: Spacing between samples, in seconds.

    Returns:
        The time coordinate, with MSv4 time attributes.

    Raises:
        ValueError: If size is not positive.
    """
    _check_size("time", size)
    values = start + interval * np.arange(size, dtype=np.float64)
    return xr.DataArray(
        values,
        dims=("time",),
        name="time",
        attrs={
            "type": "time",
            "units": "s",
            "scale": "utc",
            "format": "unix",
            "integration_time": interval,
        },
    )


def frequency_coord(
    size: int,
    *,
    start: float = DEFAULT_FREQUENCY_START,
    end: float = DEFAULT_FREQUENCY_END,
) -> xr.DataArray:
    """Build a frequency coordinate spanning a band.

    A size-one axis yields the range start rather than its midpoint: one
    solution for the whole band is conventionally labelled by where the band
    begins.

    Args:
        size: Number of channels.
        start: Frequency of the first channel, in Hz.
        end: Frequency of the last channel, in Hz.

    Returns:
        The frequency coordinate, with MSv4 spectral attributes.

    Raises:
        ValueError: If size is not positive.
    """
    _check_size("frequency", size)
    values = np.linspace(start, end, size, dtype=np.float64)
    channel_width = (end - start) / (size - 1) if size > 1 else (end - start)
    return xr.DataArray(
        values,
        dims=("frequency",),
        name="frequency",
        attrs={
            "type": "spectral_coord",
            "units": "Hz",
            "observer": "topo",
            "reference_frequency": float(values[0]),
            "channel_width": float(channel_width),
        },
    )


def antenna_name_coord(size: int, *, prefix: str = "m") -> xr.DataArray:
    """Build an antenna name coordinate.

    Args:
        size: Number of antennas.
        prefix: Name prefix. The default yields MeerKAT-style names.

    Returns:
        The antenna_name coordinate, holding zero-padded names.

    Raises:
        ValueError: If size is not positive.
    """
    _check_size("antenna_name", size)
    names = [f"{prefix}{index:03d}" for index in range(size)]
    return xr.DataArray(
        np.array(names),
        dims=("antenna_name",),
        name="antenna_name",
        attrs={"long_name": "Antenna name"},
    )


def direction_coord(size: int) -> xr.DataArray:
    """Build a direction coordinate.

    Slide 8 of the source deck specifies indexed directions, so this is a plain
    integer index rather than a sky position. The index is intended to be
    resolved against a direction list held elsewhere.

    Args:
        size: Number of directions.

    Returns:
        The direction coordinate, holding integer indices.

    Raises:
        ValueError: If size is not positive.
    """
    _check_size("direction", size)
    return xr.DataArray(
        np.arange(size, dtype=np.int64),
        dims=("direction",),
        name="direction",
        attrs={"long_name": "Direction index"},
    )


def _label_coord(axis: str, labels: Sequence[str], long_name: str) -> xr.DataArray:
    """Build a label coordinate from verbatim labels.

    Args:
        axis: Axis name, used as the dimension name.
        labels: Label values, used in the order given.
        long_name: Value for the long_name attribute.

    Returns:
        The label coordinate.

    Raises:
        ValueError: If labels is empty.
    """
    if len(labels) == 0:
        raise ValueError(f"{axis} needs at least one label")
    return xr.DataArray(
        np.array(list(labels)),
        dims=(axis,),
        name=axis,
        attrs={"long_name": long_name},
    )


def receptor_label_coord(labels: Sequence[str] = ("X", "Y")) -> xr.DataArray:
    """Build a receptor label coordinate.

    Args:
        labels: Receptor labels. The default is dual linear; pass ("R", "L")
            for dual circular.

    Returns:
        The receptor_label coordinate.

    Raises:
        ValueError: If labels is empty.
    """
    return _label_coord("receptor_label", labels, "Receptor label")


def parameter_label_coord(labels: Sequence[str]) -> xr.DataArray:
    """Build a parameter label coordinate.

    Args:
        labels: Parameter labels, one per position along the axis.

    Returns:
        The parameter_label coordinate.

    Raises:
        ValueError: If labels is empty.
    """
    return _label_coord("parameter_label", labels, "Parameter label")


SIZED_AXIS_FACTORIES: dict[str, Callable[..., xr.DataArray]] = {
    "direction": direction_coord,
    "time": time_coord,
    "antenna_name": antenna_name_coord,
    "frequency": frequency_coord,
}


def sorted_axes(axes: Iterable[str]) -> tuple[str, ...]:
    """Return axes in canonical order.

    Args:
        axes: Axis names, in any order, without duplicates.

    Returns:
        The same axis names, ordered as in CANONICAL_AXES.

    Raises:
        ValueError: If any name is not a known axis.
    """
    requested = tuple(axes)
    unknown = [axis for axis in requested if axis not in CANONICAL_AXES]
    if unknown:
        raise ValueError(f"unknown axis names {unknown}; expected some of {list(CANONICAL_AXES)}")
    return tuple(axis for axis in CANONICAL_AXES if axis in requested)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_axes.py -v
```

Expected: all PASS.

- [ ] **Step 5: Export from the package**

Replace `src/gain_skeletons/__init__.py` with:

```python
"""Mock xarray/zarr dataset scaffolds for radio interferometric gain solutions.

This package is a demonstrator. Every array value it produces is randomly
generated; nothing here computes or applies calibration.
"""

from gain_skeletons.axes import (
    CANONICAL_AXES,
    antenna_name_coord,
    direction_coord,
    frequency_coord,
    parameter_label_coord,
    receptor_label_coord,
    time_coord,
)

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_AXES",
    "__version__",
    "antenna_name_coord",
    "direction_coord",
    "frequency_coord",
    "parameter_label_coord",
    "receptor_label_coord",
    "time_coord",
]
```

- [ ] **Step 6: Lint, test and commit**

```bash
source .venv/bin/activate
ruff check --fix . && ruff format .
python -m pytest -v
git add src/gain_skeletons/axes.py src/gain_skeletons/__init__.py tests/test_axes.py
git commit -m "Add coordinate factories for calibration dataset axes"
```

---

### Task 3: Specification dataclasses

**Files:**
- Create: `src/gain_skeletons/spec.py`
- Create: `tests/test_spec.py`
- Modify: `src/gain_skeletons/__init__.py`

**Interfaces:**
- Consumes: `CANONICAL_AXES`, `LABEL_AXES`, `SIZED_AXES`, `sorted_axes` from `gain_skeletons.axes`.
- Produces:
  - `ParamSpec(name, units, axes, dtype, labels=None, scale=1.0)`, frozen dataclass. Property `resolved_labels: tuple[str, ...]` returns `labels` or `(name,)`.
  - `CalSpec(name, parameters, default_sizes, consolidated_name=None, jones_structure=None, description="")`, frozen dataclass. Properties: `axes: tuple[str, ...]` (union of parameter axes, canonical order), `direction_dependent: bool`, `resolved_consolidated_name: str`, `all_labels: tuple[str, ...]` (concatenation of every parameter's `resolved_labels`), `uniform_units: str | None` (the shared unit, or None if heterogeneous), `uniform_dtype: str | None` (the shared dtype, or None if heterogeneous).

**Note on `dtype`:** the spec document places `dtype` on `CalSpec`. Put it on `ParamSpec` instead. The spec also requires `make_gain_xds` to raise `ValueError` for a calibration type mixing complex and float quantities — which is unreachable if a single dtype is a property of the whole `CalSpec`. Per-parameter dtype makes that documented error case expressible. Update the spec document accordingly in Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec.py`:

```python
"""Tests for the declarative specification dataclasses."""

import pytest

from gain_skeletons.spec import CalSpec, ParamSpec


def make_gain_param(**overrides) -> ParamSpec:
    """Build a plausible single-parameter ParamSpec, overridable per test."""
    kwargs = {
        "name": "GAIN",
        "units": "rel",
        "axes": ("time", "antenna_name", "frequency", "receptor_label"),
        "dtype": "complex64",
    }
    kwargs.update(overrides)
    return ParamSpec(**kwargs)


def test_resolved_labels_defaults_to_the_parameter_name():
    assert make_gain_param().resolved_labels == ("GAIN",)


def test_resolved_labels_uses_explicit_labels():
    param = make_gain_param(
        axes=("time", "antenna_name", "frequency", "receptor_label", "parameter_label"),
        labels=("aligned", "cross"),
    )
    assert param.resolved_labels == ("aligned", "cross")


def test_param_spec_rejects_unknown_axis():
    with pytest.raises(ValueError, match="unknown axis"):
        make_gain_param(axes=("time", "baseline_id"))


def test_param_spec_rejects_non_canonical_axis_order():
    with pytest.raises(ValueError, match="canonical order"):
        make_gain_param(axes=("antenna_name", "time"))


def test_param_spec_rejects_duplicate_axes():
    with pytest.raises(ValueError, match="duplicate"):
        make_gain_param(axes=("time", "time"))


# Multiple labels need somewhere to live, so the parameter axis must be
# declared. Declaring labels without it is a spec error, not something to
# silently repair.
def test_param_spec_rejects_multiple_labels_without_parameter_axis():
    with pytest.raises(ValueError, match="parameter_label"):
        make_gain_param(labels=("aligned", "cross"))


def test_param_spec_rejects_empty_labels():
    with pytest.raises(ValueError, match="at least one label"):
        make_gain_param(labels=())


def test_param_spec_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="dtype"):
        make_gain_param(dtype="int32")


def test_cal_spec_axes_is_the_union_in_canonical_order():
    spec = CalSpec(
        name="fringefit",
        parameters=(
            ParamSpec("PHASE", "deg", ("time", "receptor_label", "parameter_label"), "float64"),
            ParamSpec("DISP_DELAY", "s", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    assert spec.axes == ("time", "receptor_label", "parameter_label")


def test_cal_spec_all_labels_concatenates_in_declaration_order():
    spec = CalSpec(
        name="fringefit",
        parameters=(
            ParamSpec("PHASE", "deg", ("time", "parameter_label"), "float64"),
            ParamSpec("DELAY", "s", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    assert spec.all_labels == ("PHASE", "DELAY")


# default_sizes must name every sized axis the type uses, so the helper below
# supplies a complete set for make_gain_param's default axes.
GAIN_SIZES = {"time": 4, "antenna_name": 8, "frequency": 1}


def test_cal_spec_consolidated_name_defaults_to_the_sole_parameter():
    spec = CalSpec(name="G", parameters=(make_gain_param(),), default_sizes=GAIN_SIZES)
    assert spec.resolved_consolidated_name == "GAIN"


def test_cal_spec_requires_consolidated_name_when_multi_parameter():
    with pytest.raises(ValueError, match="consolidated_name"):
        CalSpec(
            name="fringefit",
            parameters=(
                ParamSpec("PHASE", "deg", ("time",), "float64"),
                ParamSpec("DELAY", "s", ("time",), "float64"),
            ),
            default_sizes={"time": 4},
        )


def test_cal_spec_reports_uniform_units():
    spec = CalSpec(
        name="antpos",
        parameters=(
            ParamSpec(
                "ANTENNA_POSITION_OFFSET",
                "m",
                ("time", "antenna_name", "parameter_label"),
                "float64",
                labels=("dX", "dY", "dZ"),
            ),
        ),
        default_sizes={"time": 4, "antenna_name": 8},
    )
    assert spec.uniform_units == "m"


def test_cal_spec_reports_heterogeneous_units_as_none():
    spec = CalSpec(
        name="fringefit",
        parameters=(
            ParamSpec("PHASE", "deg", ("time", "parameter_label"), "float64"),
            ParamSpec("DELAY", "s", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    assert spec.uniform_units is None
    assert spec.uniform_dtype == "float64"


def test_cal_spec_reports_heterogeneous_dtype_as_none():
    spec = CalSpec(
        name="mixed",
        parameters=(
            ParamSpec("A", "rel", ("time", "parameter_label"), "complex64"),
            ParamSpec("B", "rel", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    assert spec.uniform_dtype is None


def test_cal_spec_direction_dependent_follows_the_direction_axis():
    di = CalSpec(name="G", parameters=(make_gain_param(),), default_sizes=GAIN_SIZES)
    dd = CalSpec(
        name="dd_gain",
        parameters=(make_gain_param(axes=("direction", "time", "antenna_name")),),
        default_sizes={"direction": 3, "time": 4, "antenna_name": 8},
    )
    assert di.direction_dependent is False
    assert dd.direction_dependent is True


def test_cal_spec_rejects_duplicate_labels_across_parameters():
    with pytest.raises(ValueError, match="duplicate"):
        CalSpec(
            name="clash",
            parameters=(
                ParamSpec("A", "s", ("time", "parameter_label"), "float64", labels=("x",)),
                ParamSpec("B", "s", ("time", "parameter_label"), "float64", labels=("x",)),
            ),
            default_sizes={"time": 4},
            consolidated_name="PARAMETER",
        )


def test_cal_spec_rejects_no_parameters():
    with pytest.raises(ValueError, match="at least one parameter"):
        CalSpec(name="empty", parameters=(), default_sizes={})


# default_sizes exists to distinguish "nFreq=1" from "nFreq=nCh". Naming an
# axis the calibration type does not have is a spec error.
def test_cal_spec_rejects_default_size_for_absent_axis():
    with pytest.raises(ValueError, match="not an axis of"):
        CalSpec(
            name="G",
            parameters=(make_gain_param(axes=("time", "antenna_name")),),
            default_sizes={"time": 4, "frequency": 64},
        )


def test_cal_spec_rejects_default_size_for_label_axis():
    with pytest.raises(ValueError, match="never configurable"):
        CalSpec(
            name="G",
            parameters=(make_gain_param(),),
            default_sizes={"time": 4, "receptor_label": 2},
        )


def test_cal_spec_requires_default_size_for_every_sized_axis():
    with pytest.raises(ValueError, match="missing default size"):
        CalSpec(
            name="G",
            parameters=(make_gain_param(axes=("time", "antenna_name")),),
            default_sizes={"time": 4},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_spec.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gain_skeletons.spec'`.

- [ ] **Step 3: Write the implementation**

Create `src/gain_skeletons/spec.py`:

```python
"""Declarative description of a calibration type.

A :class:`ParamSpec` describes one named physical quantity: its units, its
dtype, the axes over which it is defined, and the labels it occupies along the
parameter axis. A :class:`CalSpec` gathers the parameters belonging to one
calibration type.

Both are frozen and validate on construction, so an invalid calibration type
cannot be represented.

The source deck runs two senses of "parameter" together, and this module keeps
them apart under a single mechanism:

- Same-unit parameters within one quantity, such as the aligned and cross gains
  of a full Jones term, are one ParamSpec with several labels.
- Different-unit quantities within one calibration type, such as a fringe fit's
  phase, delay and rate, are several ParamSpecs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from gain_skeletons.axes import CANONICAL_AXES, LABEL_AXES, SIZED_AXES, sorted_axes

# Calibration parameters are either complex, when they are gains, or real, when
# they are physical quantities from which gains are generated. Integer and
# boolean parameters have no meaning here; FLAG is generated separately.
SUPPORTED_DTYPE_KINDS = ("c", "f")


def _validate_axes(owner: str, axes: Sequence[str]) -> None:
    """Validate an axis tuple for a parameter.

    Args:
        owner: Description of the owner, used in error messages.
        axes: Axis names, expected in canonical order without duplicates.

    Raises:
        ValueError: If an axis is unknown, duplicated, or out of canonical order.
    """
    if len(set(axes)) != len(axes):
        raise ValueError(f"{owner} declares duplicate axes: {list(axes)}")
    # sorted_axes raises on unknown names, so this validates membership too.
    canonical = sorted_axes(axes)
    if canonical != tuple(axes):
        raise ValueError(
            f"{owner} declares axes out of canonical order: got {list(axes)}, "
            f"expected {list(canonical)}"
        )


@dataclass(frozen=True)
class ParamSpec:
    """One named calibration parameter.

    Attributes:
        name: Data array name in the split layout, and the default parameter
            label. Conventionally upper case, following MSv4.
        units: Units of the parameter. One parameter always has exactly one unit.
        axes: Axes over which the parameter is defined, in canonical order.
            Axes absent from this tuple are absent from the dataset entirely,
            which is how the deck's brace notation is represented.
        dtype: Numpy dtype name. Must be a complex or floating kind.
        labels: Values this parameter occupies along the parameter axis. None
            means a single label equal to name.
        scale: Magnitude hint for random generation, so that a delay in seconds
            and a phase in degrees do not come out the same size. Ignored for
            complex parameters, which are generated around unit amplitude.
    """

    name: str
    units: str
    axes: tuple[str, ...]
    dtype: str
    labels: tuple[str, ...] | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the parameter.

        Raises:
            ValueError: If the axes are invalid, the labels are empty, several
                labels are declared without a parameter axis, or the dtype is
                not a complex or floating kind.
        """
        owner = f"parameter {self.name!r}"
        _validate_axes(owner, self.axes)

        if self.labels is not None:
            if len(self.labels) == 0:
                raise ValueError(f"{owner} needs at least one label")
            if len(self.labels) > 1 and "parameter_label" not in self.axes:
                raise ValueError(
                    f"{owner} declares {len(self.labels)} labels but no parameter_label axis"
                )

        if np.dtype(self.dtype).kind not in SUPPORTED_DTYPE_KINDS:
            raise ValueError(
                f"{owner} has unsupported dtype {self.dtype!r}; expected a complex or float dtype"
            )

    @property
    def resolved_labels(self) -> tuple[str, ...]:
        """Labels this parameter occupies along the parameter axis."""
        return self.labels if self.labels is not None else (self.name,)


@dataclass(frozen=True)
class CalSpec:
    """A calibration type: one or more parameters sharing a coordinate system.

    Attributes:
        name: Registry key for the calibration type.
        parameters: The parameters, in declaration order. That order fixes the
            order of the consolidated parameter axis.
        default_sizes: Default extent of every sized axis the type uses. This
            is what distinguishes a deliberately single-channel axis from a
            channel-resolved one.
        consolidated_name: Data array name in the consolidated layout. May be
            omitted only when there is exactly one parameter, in which case
            that parameter's name is used.
        jones_structure: Which part of the Jones matrix the type populates, if
            the deck specifies one. Describes the origin of the data, not an
            instruction for its use.
        description: Human-readable summary.
    """

    name: str
    parameters: tuple[ParamSpec, ...]
    default_sizes: Mapping[str, int] = field(default_factory=dict)
    consolidated_name: str | None = None
    jones_structure: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the calibration type.

        Raises:
            ValueError: If there are no parameters, labels collide across
                parameters, a multi-parameter type omits consolidated_name, or
                default_sizes does not correspond exactly to the sized axes in
                use.
        """
        if not self.parameters:
            raise ValueError(f"calibration type {self.name!r} needs at least one parameter")

        labels = self.all_labels
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"calibration type {self.name!r} has duplicate parameter labels: {list(labels)}"
            )

        if self.consolidated_name is None and len(self.parameters) > 1:
            raise ValueError(
                f"calibration type {self.name!r} has {len(self.parameters)} parameters "
                "and so must set consolidated_name"
            )

        axes = self.axes
        for axis in self.default_sizes:
            if axis in LABEL_AXES:
                raise ValueError(
                    f"the extent of {axis!r} is never configurable; it follows from its labels"
                )
            if axis not in axes:
                raise ValueError(f"{axis!r} is not an axis of calibration type {self.name!r}")

        missing = [axis for axis in axes if axis in SIZED_AXES and axis not in self.default_sizes]
        if missing:
            raise ValueError(
                f"calibration type {self.name!r} is missing default size for {missing}"
            )

    @property
    def axes(self) -> tuple[str, ...]:
        """Union of every parameter's axes, in canonical order."""
        union = {axis for param in self.parameters for axis in param.axes}
        return tuple(axis for axis in CANONICAL_AXES if axis in union)

    @property
    def direction_dependent(self) -> bool:
        """Whether this calibration type resolves direction within a field of view."""
        return "direction" in self.axes

    @property
    def resolved_consolidated_name(self) -> str:
        """Data array name to use in the consolidated layout."""
        if self.consolidated_name is not None:
            return self.consolidated_name
        return self.parameters[0].name

    @property
    def all_labels(self) -> tuple[str, ...]:
        """Every parameter label, concatenated in declaration order."""
        return tuple(label for param in self.parameters for label in param.resolved_labels)

    @property
    def uniform_units(self) -> str | None:
        """The units shared by every parameter, or None if they differ."""
        units = {param.units for param in self.parameters}
        return units.pop() if len(units) == 1 else None

    @property
    def uniform_dtype(self) -> str | None:
        """The dtype shared by every parameter, or None if they differ."""
        dtypes = {param.dtype for param in self.parameters}
        return dtypes.pop() if len(dtypes) == 1 else None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_spec.py -v
```

Expected: all PASS.

- [ ] **Step 5: Export from the package**

Add to the imports in `src/gain_skeletons/__init__.py`:

```python
from gain_skeletons.spec import CalSpec, ParamSpec
```

and add `"CalSpec"` and `"ParamSpec"` to `__all__`, keeping it sorted.

- [ ] **Step 6: Lint, test and commit**

```bash
source .venv/bin/activate
ruff check --fix . && ruff format .
python -m pytest -v
git add src/gain_skeletons/spec.py src/gain_skeletons/__init__.py tests/test_spec.py
git commit -m "Add ParamSpec and CalSpec with construction-time validation"
```

---

### Task 4: The registry

**Files:**
- Create: `src/gain_skeletons/registry.py`
- Create: `tests/test_registry.py`
- Modify: `src/gain_skeletons/__init__.py`

**Interfaces:**
- Consumes: `ParamSpec`, `CalSpec` from `gain_skeletons.spec`.
- Produces:
  - `REGISTRY: dict[str, CalSpec]` — ten entries keyed `"J"`, `"G"`, `"T"`, `"opacity"`, `"B"`, `"D"`, `"antpos"`, `"fringefit"`, `"dd_gain"`, `"ionosphere"`.
  - `get_spec(name: str) -> CalSpec` — raises `KeyError` listing the available keys.
  - `list_cal_types() -> tuple[str, ...]` — registry keys in insertion order.

**Note:** the test transcribes slides 6 and 7 a second time, deliberately duplicating `registry.py`. That duplication is the point: it checks the registry against the source material rather than against itself. Do not refactor the two into one shared table.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_registry.py`:

```python
"""Tests that the registry matches the source deck.

The expected values below are transcribed independently from slides 6 and 7 of
docs/reference/calibration-dataset-coordinate-dimensions.pdf. The duplication
against registry.py is deliberate: it checks the registry against the source
material rather than against itself. Do not factor these tables together.

Deck notation: "nFreq=1" means the frequency axis exists with length one, while
"{nFreq=0}" means it does not exist at all.
"""

import pytest

from gain_skeletons.registry import REGISTRY, get_spec, list_cal_types

TIME_ANT = ("time", "antenna_name")

# (key, axes, parameter name, units, dtype, labels, jones_structure)
SLIDE_6_DIRECTION_INDEPENDENT = [
    # General "J": [nTime, nAnt, nFreq, nPol=2, nPar=2] (Complex) GAIN in [rel]
    (
        "J",
        (*TIME_ANT, "frequency", "receptor_label", "parameter_label"),
        "GAIN",
        "rel",
        "complex64",
        ("aligned", "cross"),
        "full",
    ),
    # Standard "G": [nTime, nAnt, nFreq=1, nPol=2] (Complex) GAIN in [rel] (on-diag only)
    (
        "G",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Standard "T": [nTime, nAnt, nFreq=1, {nPol=0}] (Complex) GAIN in [rel] (scalar, unpol!)
    ("T", (*TIME_ANT, "frequency"), "GAIN", "rel", "complex64", ("GAIN",), "scalar"),
    # Opacity: [nTime, nAnt, nFreq=1, {nPol=0}] (Float) OPAC in [nepers] (unpol!)
    ("opacity", (*TIME_ANT, "frequency"), "OPAC", "nepers", "float64", ("OPAC",), None),
    # Standard "B": [nTime, nAnt, nFreq=nCh, nPol=2] (Complex) GAIN in [rel] (on-diag only)
    (
        "B",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Standard "D": [nTime, nAnt, nFreq=nCh, nPol=2] (Complex) GAIN in [rel] (off-diag only)
    (
        "D",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "off-diagonal",
    ),
    # Antpos: [nTime, nAnt, {nFreq=0}, {nPol=0}, nPar=3] (Float) (dX,dY,dZ) in [m]
    (
        "antpos",
        (*TIME_ANT, "parameter_label"),
        "ANTENNA_POSITION_OFFSET",
        "m",
        "float64",
        ("dX", "dY", "dZ"),
        None,
    ),
]

# (key, axes, parameter name, units, dtype, labels, jones_structure)
SLIDE_7_DIRECTION_DEPENDENT = [
    # Generic gain: [nDir, nTime, nAnt, nFreq=1, nPol=2] (Complex) GAIN in [rel] (on-diag only)
    (
        "dd_gain",
        ("direction", *TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Ionosphere: [nDir, nTime, nAnt, {nFreq=0}, {nPol=0}] (Float) TEC in [TECU]
    ("ionosphere", ("direction", *TIME_ANT), "TEC", "TECU", "float64", ("TEC",), None),
]

SINGLE_PARAMETER_CASES = SLIDE_6_DIRECTION_INDEPENDENT + SLIDE_7_DIRECTION_DEPENDENT

# Fringefit is the only multi-parameter entry. Slide 6 gives four lines:
#   [nTime, nAnt, nFreq=1,  nPol=2,   nPar=1] (Float) PHASE      in [deg]
#   [nTime, nAnt, nFreq=1,  nPol=2,   nPar=1] (Float) DELAY      in [s]
#   [nTime, nAnt, nFreq=1,  nPol=2,   nPar=1] (Float) RATE       in [s/s]
#   [nTime, nAnt, nFreq=1, {nPol=0},  nPar=1] (Float) DISP_DELAY in [s]
POLARISED = (*TIME_ANT, "frequency", "receptor_label", "parameter_label")
UNPOLARISED = (*TIME_ANT, "frequency", "parameter_label")
FRINGEFIT_PARAMETERS = [
    ("PHASE", "deg", POLARISED),
    ("DELAY", "s", POLARISED),
    ("RATE", "s/s", POLARISED),
    ("DISP_DELAY", "s", UNPOLARISED),
]


def test_registry_has_exactly_the_deck_entries():
    expected = {key for key, *_ in SINGLE_PARAMETER_CASES} | {"fringefit"}
    assert set(REGISTRY) == expected


def test_list_cal_types_matches_registry_keys():
    assert list_cal_types() == tuple(REGISTRY)


@pytest.mark.parametrize(
    ("key", "axes", "param_name", "units", "dtype", "labels", "jones_structure"),
    SINGLE_PARAMETER_CASES,
    ids=[case[0] for case in SINGLE_PARAMETER_CASES],
)
def test_single_parameter_entry_matches_deck(
    key, axes, param_name, units, dtype, labels, jones_structure
):
    spec = get_spec(key)
    assert len(spec.parameters) == 1
    param = spec.parameters[0]
    assert spec.axes == axes
    assert param.axes == axes
    assert param.name == param_name
    assert param.units == units
    assert param.dtype == dtype
    assert param.resolved_labels == labels
    assert spec.jones_structure == jones_structure


def test_fringefit_parameters_match_deck():
    spec = get_spec("fringefit")
    actual = [(param.name, param.units, param.axes) for param in spec.parameters]
    assert actual == FRINGEFIT_PARAMETERS


def test_fringefit_is_the_only_multi_parameter_entry():
    multi = {key for key, spec in REGISTRY.items() if len(spec.parameters) > 1}
    assert multi == {"fringefit"}


def test_fringefit_consolidated_labels_are_the_quantity_names():
    assert get_spec("fringefit").all_labels == ("PHASE", "DELAY", "RATE", "DISP_DELAY")


def test_fringefit_units_are_heterogeneous():
    assert get_spec("fringefit").uniform_units is None


# Slide 6 marks G, T, opacity and every fringefit quantity as single-channel,
# while B and D are channel-resolved. That distinction lives in default_sizes.
@pytest.mark.parametrize("key", ["G", "T", "opacity", "fringefit", "dd_gain"])
def test_single_channel_entries_default_to_one_channel(key):
    assert get_spec(key).default_sizes["frequency"] == 1


@pytest.mark.parametrize("key", ["B", "D", "J"])
def test_channel_resolved_entries_default_to_many_channels(key):
    assert get_spec(key).default_sizes["frequency"] == 64


# "{nFreq=0}" on slide 6 and 7 means the axis is genuinely absent, which is
# materially different from a length-one axis.
@pytest.mark.parametrize("key", ["antpos", "ionosphere"])
def test_frequency_independent_entries_have_no_frequency_axis(key):
    assert "frequency" not in get_spec(key).axes


@pytest.mark.parametrize("key", ["T", "opacity", "antpos", "ionosphere"])
def test_unpolarised_entries_have_no_receptor_axis(key):
    assert "receptor_label" not in get_spec(key).axes


@pytest.mark.parametrize("key", ["dd_gain", "ionosphere"])
def test_direction_dependent_entries_are_flagged(key):
    assert get_spec(key).direction_dependent is True


@pytest.mark.parametrize(
    "key", ["J", "G", "T", "opacity", "B", "D", "antpos", "fringefit"]
)
def test_direction_independent_entries_are_flagged(key):
    assert get_spec(key).direction_dependent is False


def test_get_spec_rejects_unknown_name_and_lists_alternatives():
    with pytest.raises(KeyError, match="ionosphere"):
        get_spec("not_a_cal_type")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gain_skeletons.registry'`.

- [ ] **Step 3: Write the implementation**

Create `src/gain_skeletons/registry.py`:

```python
"""The calibration type catalogue, transcribed from the source deck.

Every entry corresponds to a line on slide 6 (direction-independent) or slide 7
(direction-dependent) of
docs/reference/calibration-dataset-coordinate-dimensions.pdf.

The deck's brace notation is represented by axis presence: "nFreq=1" is a
frequency axis of length one, recorded in default_sizes, while "{nFreq=0}" is no
frequency axis at all, recorded by its absence from the parameter's axes.

Sizes here are defaults only, and every builder accepts overrides. They are kept
small so that notebook output stays readable.
"""

from gain_skeletons.spec import CalSpec, ParamSpec

# Global size defaults, small enough to read in a notebook repr.
N_TIME = 4
N_ANTENNA = 8
N_DIRECTION = 3

# A deliberately single-channel axis, versus a channel-resolved one. Slide 6
# writes these "nFreq=1" and "nFreq=nCh" respectively.
N_CHANNEL_ONE = 1
N_CHANNEL_MANY = 64

# Axis groupings that recur across entries.
_TIME_ANT = ("time", "antenna_name")
_GAIN_AXES = (*_TIME_ANT, "frequency", "receptor_label")
_UNPOL_AXES = (*_TIME_ANT, "frequency")

# Slide 4 names the two columns of the Jones matrix the "aligned" and "cross"
# gains, which is what the general J term's two parameters per receptor are.
_JONES_LABELS = ("aligned", "cross")


def _gain_spec(
    name: str,
    *,
    jones_structure: str,
    n_frequency: int,
    axes: tuple[str, ...] = _GAIN_AXES,
    labels: tuple[str, ...] | None = None,
    description: str = "",
    n_direction: int | None = None,
) -> CalSpec:
    """Build a CalSpec for a complex gain calibration type.

    Args:
        name: Registry key.
        jones_structure: Which part of the Jones matrix the type populates.
        n_frequency: Default extent of the frequency axis.
        axes: Axes of the single GAIN parameter, in canonical order.
        labels: Parameter labels, or None for a single label named GAIN.
        description: Human-readable summary.
        n_direction: Default extent of the direction axis, if the type has one.

    Returns:
        The calibration type specification.
    """
    default_sizes = {"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": n_frequency}
    if n_direction is not None:
        default_sizes["direction"] = n_direction
    return CalSpec(
        name=name,
        parameters=(
            ParamSpec(
                name="GAIN",
                units="rel",
                axes=axes,
                dtype="complex64",
                labels=labels,
            ),
        ),
        default_sizes=default_sizes,
        jones_structure=jones_structure,
        description=description,
    )


# Slide 6: direction-independent calibration types.
_J = _gain_spec(
    "J",
    jones_structure="full",
    n_frequency=N_CHANNEL_MANY,
    axes=(*_GAIN_AXES, "parameter_label"),
    labels=_JONES_LABELS,
    description=(
        "General Jones term. Two complex gains per receptor, the aligned and cross "
        "responses, so a parameter axis is required. The deck leaves its frequency "
        "extent unspecified, so it defaults to channel-resolved."
    ),
)

_G = _gain_spec(
    "G",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_ONE,
    description="Standard electronic gain, on-diagonal only, one solution per band.",
)

_T = _gain_spec(
    "T",
    jones_structure="scalar",
    n_frequency=N_CHANNEL_ONE,
    axes=_UNPOL_AXES,
    description=(
        "Standard tropospheric gain. Scalar and unpolarised, so it carries no "
        "receptor axis."
    ),
)

_B = _gain_spec(
    "B",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_MANY,
    description="Standard bandpass, on-diagonal only, resolved in frequency.",
)

_D = _gain_spec(
    "D",
    jones_structure="off-diagonal",
    n_frequency=N_CHANNEL_MANY,
    description="Standard polarisation leakage, off-diagonal only, resolved in frequency.",
)

_OPACITY = CalSpec(
    name="opacity",
    parameters=(
        ParamSpec(
            name="OPAC",
            units="nepers",
            axes=_UNPOL_AXES,
            dtype="float64",
            scale=0.05,
        ),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": N_CHANNEL_ONE},
    description="Atmospheric opacity. Unpolarised, so it carries no receptor axis.",
)

_ANTPOS = CalSpec(
    name="antpos",
    parameters=(
        ParamSpec(
            name="ANTENNA_POSITION_OFFSET",
            units="m",
            axes=(*_TIME_ANT, "parameter_label"),
            dtype="float64",
            labels=("dX", "dY", "dZ"),
            scale=0.01,
        ),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA},
    description=(
        "Antenna position correction. Three same-unit components on the parameter "
        "axis, and neither frequency- nor polarisation-dependent, so both of those "
        "axes are absent."
    ),
)

# Slide 6, Fringefit: the only entry with several quantities. Their units
# differ, and DISP_DELAY is unpolarised while the other three are not, so the
# two layouts genuinely diverge here.
_FRINGEFIT = CalSpec(
    name="fringefit",
    parameters=(
        ParamSpec("PHASE", "deg", (*_GAIN_AXES, "parameter_label"), "float64", scale=30.0),
        ParamSpec("DELAY", "s", (*_GAIN_AXES, "parameter_label"), "float64", scale=1.0e-9),
        ParamSpec("RATE", "s/s", (*_GAIN_AXES, "parameter_label"), "float64", scale=1.0e-12),
        ParamSpec(
            "DISP_DELAY",
            "s",
            (*_UNPOL_AXES, "parameter_label"),
            "float64",
            scale=1.0e-9,
        ),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": N_CHANNEL_ONE},
    consolidated_name="PARAMETER",
    description=(
        "Fringe fit. Four quantities with differing units, produced by a single "
        "solve. DISP_DELAY is unpolarised while the others are not, so the "
        "consolidated layout must broadcast it over the receptor axis."
    ),
)

# Slide 7: direction-dependent calibration types.
_DD_GAIN = _gain_spec(
    "dd_gain",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_ONE,
    axes=("direction", *_GAIN_AXES),
    n_direction=N_DIRECTION,
    description=(
        "Generic direction-dependent gain, on-diagonal only. The direction axis "
        "indexes facets within a single field of view."
    ),
)

_IONOSPHERE = CalSpec(
    name="ionosphere",
    parameters=(
        ParamSpec(
            name="TEC",
            units="TECU",
            axes=("direction", *_TIME_ANT),
            dtype="float64",
            scale=5.0,
        ),
    ),
    default_sizes={"direction": N_DIRECTION, "time": N_TIME, "antenna_name": N_ANTENNA},
    description=(
        "Ionospheric total electron content. Direction-dependent, but neither "
        "frequency- nor polarisation-dependent, so both of those axes are absent."
    ),
)

REGISTRY: dict[str, CalSpec] = {
    spec.name: spec
    for spec in (
        _J,
        _G,
        _T,
        _OPACITY,
        _B,
        _D,
        _ANTPOS,
        _FRINGEFIT,
        _DD_GAIN,
        _IONOSPHERE,
    )
}


def list_cal_types() -> tuple[str, ...]:
    """List the registered calibration type names.

    Returns:
        The registry keys, in the order the deck presents them.
    """
    return tuple(REGISTRY)


def get_spec(name: str) -> CalSpec:
    """Look up a calibration type by name.

    Args:
        name: Registry key, such as "B" or "fringefit".

    Returns:
        The calibration type specification.

    Raises:
        KeyError: If the name is not registered. The message lists the
            available names.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown calibration type {name!r}; available types are {list(REGISTRY)}"
        ) from None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_registry.py -v
```

Expected: all PASS. If a `test_single_parameter_entry_matches_deck` case fails, trust the test — it is the transcription of the deck — and fix `registry.py`.

- [ ] **Step 5: Export from the package**

Add to the imports in `src/gain_skeletons/__init__.py`:

```python
from gain_skeletons.registry import REGISTRY, get_spec, list_cal_types
```

and add `"REGISTRY"`, `"get_spec"` and `"list_cal_types"` to `__all__`, keeping it sorted.

- [ ] **Step 6: Lint, test and commit**

```bash
source .venv/bin/activate
ruff check --fix . && ruff format .
python -m pytest -v
git add src/gain_skeletons/registry.py src/gain_skeletons/__init__.py tests/test_registry.py
git commit -m "Add calibration type registry transcribed from source deck"
```

---

### Task 5: Consolidated builder

**Files:**
- Create: `src/gain_skeletons/builder.py`
- Create: `tests/test_builder_consolidated.py`
- Modify: `src/gain_skeletons/__init__.py`

**Interfaces:**
- Consumes: `SIZED_AXIS_FACTORIES`, `parameter_label_coord`, `receptor_label_coord` from `axes`; `CalSpec`, `ParamSpec` from `spec`; `get_spec` from `registry`.
- Produces:
  - `make_gain_xds(spec, *, n_direction=None, n_time=None, n_antenna=None, n_frequency=None, receptor_labels=("X", "Y"), coord_kwargs=None, seed=0, flag_fraction=0.05) -> xr.Dataset`
    - `spec` accepts a `CalSpec` or a registry name.
    - Returns a dataset with one data array named `spec.resolved_consolidated_name` plus a boolean `FLAG`.
  - `DEFAULT_FLAG_FRACTION: float = 0.05`
  - `DEFAULT_AMPLITUDE_JITTER: float = 0.1`
  - Private helpers `_resolve_spec`, `_resolve_sizes`, `_build_coords`, `_generate_values`, `_broadcast_to_axes`, `_flag_dims`, `_dataset_attrs`, reused by Task 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_builder_consolidated.py`:

```python
"""Tests for the consolidated dataset builder."""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds
from gain_skeletons.registry import get_spec, list_cal_types
from gain_skeletons.spec import CalSpec, ParamSpec


def test_accepts_a_registry_name():
    xds = make_gain_xds("G")
    assert isinstance(xds, xr.Dataset)
    assert xds.attrs["cal_type"] == "G"


def test_accepts_a_cal_spec_object():
    assert make_gain_xds(get_spec("G")).attrs["cal_type"] == "G"


def test_g_has_the_dimensions_from_slide_6():
    xds = make_gain_xds("G", n_time=4, n_antenna=8)
    assert xds.GAIN.dims == ("time", "antenna_name", "frequency", "receptor_label")
    assert xds.GAIN.shape == (4, 8, 1, 2)
    assert xds.GAIN.dtype == np.complex64


def test_data_array_carries_units_when_uniform():
    assert make_gain_xds("G").GAIN.attrs["units"] == "rel"


def test_size_overrides_are_honoured():
    xds = make_gain_xds("B", n_time=2, n_antenna=3, n_frequency=16)
    assert xds.GAIN.shape == (2, 3, 16, 2)


def test_direction_size_override_is_honoured():
    xds = make_gain_xds("dd_gain", n_direction=5)
    assert xds.sizes["direction"] == 5


# Silently ignoring a size for an axis the calibration type does not have would
# hide a real user error, so it raises.
def test_size_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_gain_xds("antpos", n_frequency=64)


def test_direction_size_for_direction_independent_type_is_rejected():
    with pytest.raises(ValueError, match="has no 'direction' axis"):
        make_gain_xds("G", n_direction=3)


@pytest.mark.parametrize("key", ["antpos", "ionosphere"])
def test_absent_frequency_axis_is_genuinely_absent(key):
    xds = make_gain_xds(key)
    assert "frequency" not in xds.dims
    assert "frequency" not in xds.coords


@pytest.mark.parametrize("key", ["T", "opacity", "antpos", "ionosphere"])
def test_absent_receptor_axis_is_genuinely_absent(key):
    xds = make_gain_xds(key)
    assert "receptor_label" not in xds.dims
    assert "receptor_label" not in xds.coords


@pytest.mark.parametrize("key", ["G", "B", "D", "T", "opacity"])
def test_types_without_a_parameter_axis_do_not_gain_one(key):
    assert "parameter_label" not in make_gain_xds(key).dims


def test_antpos_parameter_labels():
    xds = make_gain_xds("antpos")
    assert list(xds.parameter_label.values) == ["dX", "dY", "dZ"]
    assert xds.ANTENNA_POSITION_OFFSET.dims == ("time", "antenna_name", "parameter_label")


def test_j_parameter_labels_are_the_jones_columns():
    assert list(make_gain_xds("J").parameter_label.values) == ["aligned", "cross"]


def test_receptor_labels_are_overridable():
    xds = make_gain_xds("G", receptor_labels=("R", "L"))
    assert list(xds.receptor_label.values) == ["R", "L"]


def test_coord_kwargs_reach_the_factories():
    xds = make_gain_xds(
        "B",
        n_frequency=3,
        coord_kwargs={"frequency": {"start": 1.0e9, "end": 2.0e9}},
    )
    np.testing.assert_allclose(xds.frequency.values, [1.0e9, 1.5e9, 2.0e9])


def test_coord_kwargs_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_gain_xds("antpos", coord_kwargs={"frequency": {"start": 1.0e9}})


# FLAG marks a whole solution bad. The components of one solution are not
# independently valid, so FLAG never carries the parameter axis.
@pytest.mark.parametrize("key", list_cal_types())
def test_flag_dims_are_the_parameter_dims_without_the_parameter_axis(key):
    xds = make_gain_xds(key)
    spec = get_spec(key)
    parameter = xds[spec.resolved_consolidated_name]
    expected = tuple(dim for dim in parameter.dims if dim != "parameter_label")
    assert xds.FLAG.dims == expected


@pytest.mark.parametrize("key", list_cal_types())
def test_flag_is_boolean(key):
    assert make_gain_xds(key).FLAG.dtype == np.bool_


def test_flag_fraction_zero_gives_a_clean_dataset():
    assert not make_gain_xds("B", flag_fraction=0.0).FLAG.values.any()


def test_flag_fraction_one_flags_everything():
    assert make_gain_xds("B", flag_fraction=1.0).FLAG.values.all()


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_invalid_flag_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="flag_fraction"):
        make_gain_xds("G", flag_fraction=fraction)


# Complex gains sit near unit amplitude. A uniform-random complex number of
# arbitrary magnitude would be physically nonsensical.
def test_complex_gains_are_generated_near_unit_amplitude():
    amplitude = np.abs(make_gain_xds("B", n_time=8, n_frequency=64).GAIN.values)
    assert 0.5 < amplitude.mean() < 1.5


def test_float_parameters_respect_their_scale():
    # DELAY has scale 1e-9, so values should be nanosecond-ish, not order unity.
    delay = make_gain_xds("fringefit").PARAMETER.sel(parameter_label="DELAY").values
    assert np.abs(delay).max() < 1.0e-6


def test_equal_seeds_give_equal_values():
    a = make_gain_xds("B", seed=7)
    b = make_gain_xds("B", seed=7)
    assert a.identical(b)


def test_different_seeds_give_different_values():
    a = make_gain_xds("B", seed=1)
    b = make_gain_xds("B", seed=2)
    assert not np.array_equal(a.GAIN.values, b.GAIN.values)


def test_dataset_attributes_record_cal_type_and_direction_dependence():
    xds = make_gain_xds("dd_gain")
    assert xds.attrs["cal_type"] == "dd_gain"
    assert xds.attrs["direction_dependent"] is True
    assert xds.attrs["jones_structure"] == "diagonal"


# Storing a null attribute is worse than omitting it: it asserts that the
# calibration type has a Jones structure whose value happens to be nothing.
@pytest.mark.parametrize("key", ["opacity", "antpos", "ionosphere", "fringefit"])
def test_jones_structure_is_omitted_when_the_deck_gives_none(key):
    assert "jones_structure" not in make_gain_xds(key).attrs


def test_time_coord_carries_msv4_attributes():
    assert make_gain_xds("G").time.attrs["type"] == "time"


class TestFringefitConsolidation:
    """The one entry where consolidation genuinely does something."""

    def test_all_four_quantities_share_one_array(self):
        xds = make_gain_xds("fringefit")
        assert "PARAMETER" in xds.data_vars
        assert set(xds.data_vars) == {"PARAMETER", "FLAG"}
        assert list(xds.parameter_label.values) == ["PHASE", "DELAY", "RATE", "DISP_DELAY"]

    def test_parameter_axis_is_last(self):
        xds = make_gain_xds("fringefit")
        assert xds.PARAMETER.dims == (
            "time",
            "antenna_name",
            "frequency",
            "receptor_label",
            "parameter_label",
        )

    # Heterogeneous units cannot be asserted by a scalar attribute without
    # lying, so they move to a coordinate aligned to parameter_label.
    def test_units_move_to_a_coordinate(self):
        xds = make_gain_xds("fringefit")
        assert "units" not in xds.PARAMETER.attrs
        assert list(xds.parameter_units.values) == ["deg", "s", "s/s", "s"]

    def test_units_travel_with_a_selection(self):
        xds = make_gain_xds("fringefit")
        assert xds.sel(parameter_label="DELAY").parameter_units.item() == "s"

    # DISP_DELAY is unpolarised, so consolidating forces it to repeat across
    # the receptor axis. That redundancy is the documented cost of this layout.
    def test_unpolarised_quantity_is_broadcast_across_receptors(self):
        xds = make_gain_xds("fringefit", receptor_labels=("X", "Y"))
        disp = xds.PARAMETER.sel(parameter_label="DISP_DELAY")
        np.testing.assert_array_equal(
            disp.sel(receptor_label="X").values,
            disp.sel(receptor_label="Y").values,
        )

    def test_polarised_quantities_are_not_broadcast(self):
        xds = make_gain_xds("fringefit", n_time=4, n_antenna=8)
        phase = xds.PARAMETER.sel(parameter_label="PHASE")
        assert not np.array_equal(
            phase.sel(receptor_label="X").values,
            phase.sel(receptor_label="Y").values,
        )

    def test_one_flag_covers_the_whole_solve(self):
        xds = make_gain_xds("fringefit")
        assert xds.FLAG.dims == ("time", "antenna_name", "frequency", "receptor_label")


@pytest.mark.parametrize("key", list_cal_types())
def test_uniform_unit_entries_have_no_parameter_units_coord(key):
    xds = make_gain_xds(key)
    if get_spec(key).uniform_units is not None:
        assert "parameter_units" not in xds.coords


def test_mixed_dtype_spec_cannot_consolidate():
    spec = CalSpec(
        name="mixed",
        parameters=(
            ParamSpec("A", "rel", ("time", "parameter_label"), "complex64"),
            ParamSpec("B", "m", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    with pytest.raises(ValueError, match="cannot be consolidated"):
        make_gain_xds(spec)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_builder_consolidated.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gain_skeletons.builder'`.

- [ ] **Step 3: Write the implementation**

Create `src/gain_skeletons/builder.py`:

```python
"""Builders that turn a calibration type specification into a dataset.

Two layouts are offered, and neither is privileged as the correct one. The
source deck stores each differently-united quantity in its own data array, which
:func:`make_split_gain_xds` reproduces. Consolidating every quantity into one
array indexed by an explicit parameter axis, which :func:`make_gain_xds` does,
keeps the parameters needed to evaluate a Jones term adjacent in memory and on
disk, and lets a single flag describe a single solve.

Nine of the ten registry entries declare one parameter, and for those the two
builders produce identical datasets. The layouts diverge only for fringefit.

All values are random. Complex parameters are generated near unit amplitude,
since a uniform-random complex gain of arbitrary magnitude would be physically
nonsensical.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from gain_skeletons.axes import (
    SIZED_AXIS_FACTORIES,
    parameter_label_coord,
    receptor_label_coord,
)
from gain_skeletons.registry import get_spec
from gain_skeletons.spec import CalSpec, ParamSpec

DEFAULT_FLAG_FRACTION = 0.05

# Fractional spread of complex gain amplitude about unity.
DEFAULT_AMPLITUDE_JITTER = 0.1

FLAG_NAME = "FLAG"
PARAMETER_UNITS_COORD = "parameter_units"

# Maps the builders' keyword arguments to the axes they size. Keeping the
# public names short (n_antenna, not n_antenna_name) at the cost of this table.
_SIZE_KEYWORDS = {
    "n_direction": "direction",
    "n_time": "time",
    "n_antenna": "antenna_name",
    "n_frequency": "frequency",
}


def _resolve_spec(spec: CalSpec | str) -> CalSpec:
    """Accept either a specification or a registry name.

    Args:
        spec: A CalSpec, or the name of a registered calibration type.

    Returns:
        The calibration type specification.

    Raises:
        KeyError: If a name is given that is not registered.
    """
    return get_spec(spec) if isinstance(spec, str) else spec


def _resolve_sizes(spec: CalSpec, overrides: Mapping[str, int | None]) -> dict[str, int]:
    """Combine a specification's default sizes with caller overrides.

    Args:
        spec: The calibration type.
        overrides: Keyword name to requested size, where None means "use the
            default". Keys are the names in _SIZE_KEYWORDS.

    Returns:
        Axis name to extent, for every sized axis the calibration type has.

    Raises:
        ValueError: If a size is given for an axis the calibration type does
            not have, or if a size is not positive.
    """
    sizes = {axis: size for axis, size in spec.default_sizes.items()}
    for keyword, size in overrides.items():
        if size is None:
            continue
        axis = _SIZE_KEYWORDS[keyword]
        if axis not in spec.axes:
            raise ValueError(
                f"calibration type {spec.name!r} has no {axis!r} axis, so {keyword} "
                "cannot be set"
            )
        if size < 1:
            raise ValueError(f"{keyword} must be positive, got {size}")
        sizes[axis] = size
    return sizes


def _build_coords(
    spec: CalSpec,
    axes: Sequence[str],
    sizes: Mapping[str, int],
    labels: Sequence[str],
    receptor_labels: Sequence[str],
    coord_kwargs: Mapping[str, Mapping[str, Any]],
) -> dict[str, xr.DataArray]:
    """Build every coordinate for a dataset.

    Args:
        spec: The calibration type, used for error messages.
        axes: Axes the dataset has, in canonical order.
        sizes: Extent of each sized axis.
        labels: Parameter labels, used when axes includes parameter_label.
        receptor_labels: Receptor labels, used when axes includes receptor_label.
        coord_kwargs: Axis name to extra keyword arguments for that axis's
            factory, such as frequency start and end.

    Returns:
        Axis name to coordinate.

    Raises:
        ValueError: If coord_kwargs names an axis the calibration type lacks.
    """
    unknown = [axis for axis in coord_kwargs if axis not in axes]
    if unknown:
        raise ValueError(
            f"calibration type {spec.name!r} has no {unknown[0]!r} axis, so coord_kwargs "
            f"cannot configure it; its axes are {list(axes)}"
        )

    coords: dict[str, xr.DataArray] = {}
    for axis in axes:
        if axis == "receptor_label":
            coords[axis] = receptor_label_coord(receptor_labels)
        elif axis == "parameter_label":
            coords[axis] = parameter_label_coord(labels)
        else:
            factory = SIZED_AXIS_FACTORIES[axis]
            coords[axis] = factory(sizes[axis], **coord_kwargs.get(axis, {}))
    return coords


def _generate_values(
    param: ParamSpec,
    shape: tuple[int, ...],
    rng: np.random.Generator,
    amplitude_jitter: float,
) -> np.ndarray:
    """Generate random values for one parameter.

    Complex parameters are gains, so they are generated as a unit amplitude
    perturbed by a small normal deviate, with uniform random phase. Real
    parameters are standard normal scaled by the parameter's magnitude hint.

    Args:
        param: The parameter being generated.
        shape: Shape to generate.
        rng: Random number generator.
        amplitude_jitter: Fractional spread of complex amplitude about unity.

    Returns:
        An array of param.dtype and the requested shape.
    """
    dtype = np.dtype(param.dtype)
    if dtype.kind == "c":
        amplitude = 1.0 + amplitude_jitter * rng.standard_normal(shape)
        phase = rng.uniform(-np.pi, np.pi, shape)
        return (amplitude * np.exp(1j * phase)).astype(dtype)
    return (param.scale * rng.standard_normal(shape)).astype(dtype)


def _broadcast_to_axes(
    values: np.ndarray,
    own_axes: Sequence[str],
    target_axes: Sequence[str],
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Broadcast a parameter's values onto a wider set of axes.

    Used by the consolidated layout when a parameter lacks an axis that its
    siblings have, such as a fringe fit's unpolarised dispersive delay. The
    values are repeated along the missing axes, which is redundant, and
    deliberately visible as such.

    Args:
        values: Values defined over own_axes.
        own_axes: Axes the values are defined over, in canonical order.
        target_axes: Axes to broadcast onto, in canonical order. Must be a
            superset of own_axes.
        target_shape: Shape corresponding to target_axes.

    Returns:
        A read-only view of shape target_shape.
    """
    reshaped = values.reshape(
        tuple(values.shape[own_axes.index(axis)] if axis in own_axes else 1 for axis in target_axes)
    )
    return np.broadcast_to(reshaped, target_shape)


def _flag_dims(dims: Sequence[str]) -> tuple[str, ...]:
    """Derive the flag dimensions for a parameter array.

    A flag marks a whole solution bad, and the components of one solution are
    not independently valid, so the parameter axis is dropped.

    Args:
        dims: Dimensions of the parameter array, in canonical order.

    Returns:
        The same dimensions without parameter_label.
    """
    return tuple(dim for dim in dims if dim != "parameter_label")


def _generate_flags(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    flag_fraction: float,
) -> np.ndarray:
    """Generate a boolean flag array.

    Args:
        shape: Shape to generate.
        rng: Random number generator.
        flag_fraction: Probability that any given solution is flagged.

    Returns:
        A boolean array of the requested shape.
    """
    if flag_fraction <= 0.0:
        return np.zeros(shape, dtype=bool)
    if flag_fraction >= 1.0:
        return np.ones(shape, dtype=bool)
    return rng.random(shape) < flag_fraction


def _dataset_attrs(spec: CalSpec) -> dict[str, Any]:
    """Build the dataset-level attributes.

    jones_structure is omitted rather than stored as null when the deck does not
    specify one: a null attribute asserts that the calibration type has a Jones
    structure whose value happens to be nothing.

    Args:
        spec: The calibration type.

    Returns:
        The dataset attributes.
    """
    attrs: dict[str, Any] = {
        "cal_type": spec.name,
        "direction_dependent": spec.direction_dependent,
    }
    if spec.jones_structure is not None:
        attrs["jones_structure"] = spec.jones_structure
    if spec.description:
        attrs["description"] = spec.description
    return attrs


def _check_flag_fraction(flag_fraction: float) -> None:
    """Validate the flag fraction.

    Args:
        flag_fraction: Probability that any given solution is flagged.

    Raises:
        ValueError: If it is not between zero and one inclusive.
    """
    if not 0.0 <= flag_fraction <= 1.0:
        raise ValueError(f"flag_fraction must be between 0 and 1, got {flag_fraction}")


def make_gain_xds(
    spec: CalSpec | str,
    *,
    n_direction: int | None = None,
    n_time: int | None = None,
    n_antenna: int | None = None,
    n_frequency: int | None = None,
    receptor_labels: Sequence[str] = ("X", "Y"),
    coord_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int | None = 0,
    flag_fraction: float = DEFAULT_FLAG_FRACTION,
    amplitude_jitter: float = DEFAULT_AMPLITUDE_JITTER,
) -> xr.Dataset:
    """Build a calibration dataset with every parameter in one array.

    Parameters share a single data array, indexed by an explicit parameter axis.
    This keeps the parameters needed to evaluate a Jones term adjacent in memory
    and, once written, in one chunked zarr array. A single flag then describes a
    single solve.

    Where the parameters do not all share an axis, the ones that lack it are
    broadcast over it. Where they do not all share units, units move from a
    scalar attribute to a coordinate aligned to the parameter axis.

    Args:
        spec: A CalSpec, or the name of a registered calibration type.
        n_direction: Override the direction extent.
        n_time: Override the time extent.
        n_antenna: Override the antenna extent.
        n_frequency: Override the frequency extent.
        receptor_labels: Receptor labels, if the type has a receptor axis.
        coord_kwargs: Axis name to extra keyword arguments for that axis's
            coordinate factory, such as ``{"frequency": {"start": 1e9}}``.
        seed: Seed for the random generator, for reproducibility.
        flag_fraction: Probability that any given solution is flagged.
        amplitude_jitter: Fractional spread of complex amplitude about unity.

    Returns:
        A dataset holding one parameter array, named after the calibration
        type's consolidated name, and a boolean FLAG.

    Raises:
        ValueError: If the parameters do not share a dtype, if a size or
            coord_kwargs entry names an axis the type lacks, or if
            flag_fraction is out of range.
        KeyError: If spec is a name that is not registered.
    """
    spec = _resolve_spec(spec)
    _check_flag_fraction(flag_fraction)

    dtype = spec.uniform_dtype
    if dtype is None:
        dtypes = {param.name: param.dtype for param in spec.parameters}
        raise ValueError(
            f"calibration type {spec.name!r} cannot be consolidated because its "
            f"parameters do not share a dtype: {dtypes}. Use make_split_gain_xds instead."
        )

    sizes = _resolve_sizes(
        spec,
        {
            "n_direction": n_direction,
            "n_time": n_time,
            "n_antenna": n_antenna,
            "n_frequency": n_frequency,
        },
    )
    axes = spec.axes
    coords = _build_coords(
        spec,
        axes,
        sizes,
        spec.all_labels,
        receptor_labels,
        coord_kwargs or {},
    )
    shape = tuple(coords[axis].size for axis in axes)
    rng = np.random.default_rng(seed)

    # The parameter axis is last in canonical order, so each parameter occupies a
    # contiguous trailing slice. Types without a parameter axis have exactly one
    # parameter filling the whole array, and must not be sliced at all: the
    # trailing axis would be receptor_label.
    has_parameter_axis = "parameter_label" in axes
    values = np.empty(shape, dtype=np.dtype(dtype))
    offset = 0
    for param in spec.parameters:
        width = len(param.resolved_labels)
        own_shape = tuple(
            width if axis == "parameter_label" else coords[axis].size for axis in param.axes
        )
        generated = _generate_values(param, own_shape, rng, amplitude_jitter)
        target_shape = tuple(
            width if axis == "parameter_label" else coords[axis].size for axis in axes
        )
        broadcast = _broadcast_to_axes(generated, param.axes, axes, target_shape)
        if has_parameter_axis:
            values[..., offset : offset + width] = broadcast
            offset += width
        else:
            values[...] = broadcast

    flag_dims = _flag_dims(axes)
    flag_shape = tuple(coords[axis].size for axis in flag_dims)

    units = spec.uniform_units
    parameter_attrs: dict[str, Any] = {} if units is None else {"units": units}

    xds = xr.Dataset(
        data_vars={
            spec.resolved_consolidated_name: (axes, values, parameter_attrs),
            FLAG_NAME: (
                flag_dims,
                _generate_flags(flag_shape, rng, flag_fraction),
                {"long_name": "Solution flag"},
            ),
        },
        coords=coords,
        attrs=_dataset_attrs(spec),
    )

    if units is None:
        # Units differ between parameters, so a scalar attribute would be a
        # false claim. Carry them alongside the parameter labels instead, where
        # they travel with any selection.
        xds = xds.assign_coords(
            {
                PARAMETER_UNITS_COORD: (
                    "parameter_label",
                    np.array(
                        [
                            param.units
                            for param in spec.parameters
                            for _ in param.resolved_labels
                        ]
                    ),
                )
            }
        )
        xds[PARAMETER_UNITS_COORD].attrs["long_name"] = "Parameter units"

    return xds
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_builder_consolidated.py -v
```

Expected: all PASS.

- [ ] **Step 5: Export from the package**

Add to the imports in `src/gain_skeletons/__init__.py`:

```python
from gain_skeletons.builder import make_gain_xds
```

and add `"make_gain_xds"` to `__all__`, keeping it sorted.

- [ ] **Step 6: Lint, test and commit**

```bash
source .venv/bin/activate
ruff check --fix . && ruff format .
python -m pytest -v
git add src/gain_skeletons/builder.py src/gain_skeletons/__init__.py tests/test_builder_consolidated.py
git commit -m "Add consolidated dataset builder"
```

---

### Task 6: Split builder and layout equivalence

**Files:**
- Modify: `src/gain_skeletons/builder.py`
- Create: `tests/test_builder_split.py`
- Modify: `src/gain_skeletons/__init__.py`

**Interfaces:**
- Consumes: every private helper from Task 5, plus `make_gain_xds`.
- Produces: `make_split_gain_xds(spec, *, <same keywords as make_gain_xds>) -> dict[str, xr.Dataset]`, keyed by parameter name. Each dataset holds one array named after its parameter, with a scalar `units` attribute, plus a boolean `FLAG`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_builder_split.py`:

```python
"""Tests for the split dataset builder, and for equivalence between layouts."""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds, make_split_gain_xds
from gain_skeletons.registry import get_spec, list_cal_types

SINGLE_PARAMETER_KEYS = [key for key in list_cal_types() if len(get_spec(key).parameters) == 1]


def test_returns_a_mapping_keyed_by_parameter_name():
    datasets = make_split_gain_xds("fringefit")
    assert set(datasets) == {"PHASE", "DELAY", "RATE", "DISP_DELAY"}
    assert all(isinstance(xds, xr.Dataset) for xds in datasets.values())


def test_single_parameter_type_yields_one_dataset():
    datasets = make_split_gain_xds("G")
    assert set(datasets) == {"GAIN"}


def test_each_dataset_holds_one_parameter_and_a_flag():
    for name, xds in make_split_gain_xds("fringefit").items():
        assert set(xds.data_vars) == {name, "FLAG"}


# Splitting exists so that units can be a scalar attribute on every array.
def test_every_array_carries_scalar_units():
    datasets = make_split_gain_xds("fringefit")
    assert datasets["PHASE"].PHASE.attrs["units"] == "deg"
    assert datasets["DELAY"].DELAY.attrs["units"] == "s"
    assert datasets["RATE"].RATE.attrs["units"] == "s/s"
    assert datasets["DISP_DELAY"].DISP_DELAY.attrs["units"] == "s"


def test_no_dataset_needs_a_parameter_units_coord():
    for xds in make_split_gain_xds("fringefit").values():
        assert "parameter_units" not in xds.coords


# The deck gives each fringefit quantity nPar=1, so the parameter axis is
# present but length one.
def test_each_fringefit_dataset_has_a_length_one_parameter_axis():
    for name, xds in make_split_gain_xds("fringefit").items():
        assert xds.sizes["parameter_label"] == 1
        assert list(xds.parameter_label.values) == [name]


# Splitting keeps each quantity's exact axes, so the unpolarised dispersive
# delay is not padded out over receptors.
def test_unpolarised_quantity_keeps_no_receptor_axis():
    datasets = make_split_gain_xds("fringefit")
    assert "receptor_label" not in datasets["DISP_DELAY"].dims
    assert "receptor_label" in datasets["PHASE"].dims


def test_flag_drops_the_parameter_axis_in_every_dataset():
    for name, xds in make_split_gain_xds("fringefit").items():
        expected = tuple(dim for dim in xds[name].dims if dim != "parameter_label")
        assert xds.FLAG.dims == expected
        assert xds.FLAG.dtype == np.bool_


def test_each_quantity_gets_its_own_flag():
    datasets = make_split_gain_xds("fringefit", flag_fraction=0.5, seed=3)
    phase_flags = datasets["PHASE"].FLAG.values
    rate_flags = datasets["RATE"].FLAG.values
    assert not np.array_equal(phase_flags, rate_flags)


def test_dataset_attributes_are_carried_through():
    xds = make_split_gain_xds("dd_gain")["GAIN"]
    assert xds.attrs["cal_type"] == "dd_gain"
    assert xds.attrs["direction_dependent"] is True


def test_size_overrides_are_honoured():
    xds = make_split_gain_xds("B", n_time=2, n_antenna=3, n_frequency=16)["GAIN"]
    assert xds.GAIN.shape == (2, 3, 16, 2)


def test_size_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_split_gain_xds("antpos", n_frequency=64)


def test_invalid_flag_fraction_is_rejected():
    with pytest.raises(ValueError, match="flag_fraction"):
        make_split_gain_xds("G", flag_fraction=2.0)


# A calibration type mixing dtypes cannot consolidate, but splitting it is
# perfectly well defined.
def test_mixed_dtype_spec_splits_successfully():
    from gain_skeletons.spec import CalSpec, ParamSpec

    spec = CalSpec(
        name="mixed",
        parameters=(
            ParamSpec("A", "rel", ("time", "parameter_label"), "complex64"),
            ParamSpec("B", "m", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    datasets = make_split_gain_xds(spec)
    assert datasets["A"].A.dtype == np.complex64
    assert datasets["B"].B.dtype == np.float64


# Nine of the ten registry entries declare a single parameter. For those the
# layout choice is a distinction without a difference, and this pins that
# claim so neither builder can drift from the other.
@pytest.mark.parametrize("key", SINGLE_PARAMETER_KEYS)
def test_layouts_agree_for_single_parameter_types(key):
    consolidated = make_gain_xds(key, seed=11)
    split = make_split_gain_xds(key, seed=11)
    assert len(split) == 1
    (only,) = split.values()
    assert consolidated.identical(only)


def test_fringefit_is_the_only_type_where_layouts_differ():
    differing = []
    for key in list_cal_types():
        split = make_split_gain_xds(key, seed=11)
        if len(split) > 1 or not make_gain_xds(key, seed=11).identical(next(iter(split.values()))):
            differing.append(key)
    assert differing == ["fringefit"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_builder_split.py -v
```

Expected: FAIL with `ImportError: cannot import name 'make_split_gain_xds'`.

- [ ] **Step 3: Write the implementation**

Append to `src/gain_skeletons/builder.py`:

```python
def make_split_gain_xds(
    spec: CalSpec | str,
    *,
    n_direction: int | None = None,
    n_time: int | None = None,
    n_antenna: int | None = None,
    n_frequency: int | None = None,
    receptor_labels: Sequence[str] = ("X", "Y"),
    coord_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int | None = 0,
    flag_fraction: float = DEFAULT_FLAG_FRACTION,
    amplitude_jitter: float = DEFAULT_AMPLITUDE_JITTER,
) -> dict[str, xr.Dataset]:
    """Build one calibration dataset per parameter.

    This is the layout the source deck describes: each quantity lives in its own
    data array so that units can be a scalar attribute, and each keeps only the
    axes it is actually defined over. Nothing is broadcast, and no parameter is
    padded out over an axis it does not need.

    The cost is fragmentation. A calibration type whose quantities come from a
    single solve is spread over several datasets, each with its own flag.

    For a calibration type with one parameter, which is nine of the ten
    registered types, this returns a single dataset identical to the one
    :func:`make_gain_xds` produces.

    Args:
        spec: A CalSpec, or the name of a registered calibration type.
        n_direction: Override the direction extent.
        n_time: Override the time extent.
        n_antenna: Override the antenna extent.
        n_frequency: Override the frequency extent.
        receptor_labels: Receptor labels, for parameters with a receptor axis.
        coord_kwargs: Axis name to extra keyword arguments for that axis's
            coordinate factory.
        seed: Seed for the random generator, for reproducibility.
        flag_fraction: Probability that any given solution is flagged.
        amplitude_jitter: Fractional spread of complex amplitude about unity.

    Returns:
        Parameter name to dataset. Each dataset holds one array of that name
        and a boolean FLAG.

    Raises:
        ValueError: If a size or coord_kwargs entry names an axis the type
            lacks, or if flag_fraction is out of range.
        KeyError: If spec is a name that is not registered.
    """
    spec = _resolve_spec(spec)
    _check_flag_fraction(flag_fraction)

    sizes = _resolve_sizes(
        spec,
        {
            "n_direction": n_direction,
            "n_time": n_time,
            "n_antenna": n_antenna,
            "n_frequency": n_frequency,
        },
    )
    # Validate coord_kwargs against the whole calibration type, not each
    # parameter, so that configuring an axis only some parameters use is not an
    # error.
    _build_coords(spec, spec.axes, sizes, spec.all_labels, receptor_labels, coord_kwargs or {})

    attrs = _dataset_attrs(spec)
    rng = np.random.default_rng(seed)
    datasets: dict[str, xr.Dataset] = {}

    for param in spec.parameters:
        axes = param.axes
        coords = _build_coords(
            spec,
            axes,
            sizes,
            param.resolved_labels,
            receptor_labels,
            {axis: kwargs for axis, kwargs in (coord_kwargs or {}).items() if axis in axes},
        )
        shape = tuple(coords[axis].size for axis in axes)
        values = _generate_values(param, shape, rng, amplitude_jitter)

        flag_dims = _flag_dims(axes)
        flag_shape = tuple(coords[axis].size for axis in flag_dims)

        datasets[param.name] = xr.Dataset(
            data_vars={
                param.name: (axes, values, {"units": param.units}),
                FLAG_NAME: (
                    flag_dims,
                    _generate_flags(flag_shape, rng, flag_fraction),
                    {"long_name": "Solution flag"},
                ),
            },
            coords=coords,
            attrs=dict(attrs),
        )

    return datasets
```

**Note on RNG ordering:** for a single-parameter type both builders draw the parameter values first and the flags second, from a generator seeded identically, so `test_layouts_agree_for_single_parameter_types` passes. Do not reorder those draws in either builder.

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_builder_split.py -v
```

Expected: all PASS. If `test_layouts_agree_for_single_parameter_types` fails, compare the two builders' random draw order before changing anything else.

- [ ] **Step 5: Export from the package**

Add `make_split_gain_xds` to the `gain_skeletons.builder` import in `src/gain_skeletons/__init__.py` and to `__all__`, keeping it sorted.

- [ ] **Step 6: Lint, test and commit**

```bash
source .venv/bin/activate
ruff check --fix . && ruff format .
python -m pytest -v
git add src/gain_skeletons/builder.py src/gain_skeletons/__init__.py tests/test_builder_split.py
git commit -m "Add split dataset builder and pin layout equivalence"
```

---

### Task 7: Zarr round-trip tests

**Files:**
- Create: `tests/test_zarr_roundtrip.py`

**Interfaces:**
- Consumes: `make_gain_xds`, `make_split_gain_xds`, `list_cal_types`.
- Produces: no source changes. This task proves the datasets survive zarr, which is the package's reason for existing.

**Note:** always pass `consolidated=False` to `to_zarr`. Zarr format 3 does not specify consolidated metadata, and xarray's default emits a `ZarrUserWarning`.

- [ ] **Step 1: Write the tests**

Create `tests/test_zarr_roundtrip.py`:

```python
"""Tests that generated datasets survive a zarr round-trip.

Reading and writing is plain xarray: xds.to_zarr(path) and
xr.open_dataset(path, engine="zarr"). The package deliberately wraps neither.

consolidated=False throughout: zarr format 3 does not specify consolidated
metadata and xarray warns if asked to write it.
"""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds, make_split_gain_xds
from gain_skeletons.registry import list_cal_types


def roundtrip(xds: xr.Dataset, path) -> xr.Dataset:
    """Write a dataset to zarr, read it back, and load it into memory.

    Args:
        xds: Dataset to write.
        path: Destination store path.

    Returns:
        The dataset as read back from disk.
    """
    xds.to_zarr(path, consolidated=False)
    return xr.open_dataset(path, engine="zarr").load()


@pytest.mark.parametrize("key", list_cal_types())
def test_consolidated_layout_survives_roundtrip(key, tmp_path):
    xds = make_gain_xds(key)
    assert xds.identical(roundtrip(xds, tmp_path / f"{key}.zarr"))


@pytest.mark.parametrize("key", list_cal_types())
def test_split_layout_survives_roundtrip(key, tmp_path):
    for name, xds in make_split_gain_xds(key).items():
        assert xds.identical(roundtrip(xds, tmp_path / f"{key}_{name}.zarr"))


# Complex gains are the case most likely to be mangled by a storage layer, so
# it gets its own assertion rather than relying on identical() alone.
def test_complex_dtype_is_preserved_exactly(tmp_path):
    xds = make_gain_xds("B")
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert read.GAIN.dtype == np.complex64
    np.testing.assert_array_equal(read.GAIN.values, xds.GAIN.values)


def test_boolean_flags_are_preserved(tmp_path):
    xds = make_gain_xds("B", flag_fraction=0.5)
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert read.FLAG.dtype == np.bool_
    np.testing.assert_array_equal(read.FLAG.values, xds.FLAG.values)


def test_string_coordinates_are_preserved(tmp_path):
    xds = make_gain_xds("B")
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert list(read.antenna_name.values) == list(xds.antenna_name.values)
    assert list(read.receptor_label.values) == list(xds.receptor_label.values)


# parameter_units is a non-dimension coordinate, which is the part of the
# consolidated layout most at risk of being demoted to a data variable.
def test_parameter_units_survives_as_a_coordinate(tmp_path):
    xds = make_gain_xds("fringefit")
    read = roundtrip(xds, tmp_path / "fringefit.zarr")
    assert "parameter_units" in read.coords
    assert list(read.parameter_units.values) == ["deg", "s", "s/s", "s"]


def test_coordinate_attributes_are_preserved(tmp_path):
    xds = make_gain_xds("B")
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert read.time.attrs["type"] == "time"
    assert read.frequency.attrs["units"] == "Hz"


def test_dataset_and_variable_attributes_are_preserved(tmp_path):
    xds = make_gain_xds("dd_gain")
    read = roundtrip(xds, tmp_path / "dd.zarr")
    assert read.attrs["cal_type"] == "dd_gain"
    assert read.attrs["direction_dependent"] is True
    assert read.attrs["jones_structure"] == "diagonal"
    assert read.GAIN.attrs["units"] == "rel"


# The layouts differ on disk as well as in memory: one chunked array against
# four separate stores. This is the difference the notebook exists to show.
def test_consolidated_stores_one_array_where_split_stores_four(tmp_path):
    consolidated_path = tmp_path / "consolidated.zarr"
    make_gain_xds("fringefit").to_zarr(consolidated_path, consolidated=False)
    parameter_arrays = {
        entry.name
        for entry in consolidated_path.iterdir()
        if entry.is_dir() and entry.name not in {"time", "antenna_name", "frequency"}
    }
    assert "PARAMETER" in parameter_arrays

    split = make_split_gain_xds("fringefit")
    for name, xds in split.items():
        xds.to_zarr(tmp_path / f"split_{name}.zarr", consolidated=False)
    assert len(split) == 4
    assert all((tmp_path / f"split_{name}.zarr" / name).is_dir() for name in split)
```

- [ ] **Step 2: Run the tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_zarr_roundtrip.py -v -W error::UserWarning
```

Expected: all PASS, with no warnings escalated to errors. If a `ZarrUserWarning` about consolidated metadata surfaces, a `to_zarr` call is missing `consolidated=False`.

- [ ] **Step 3: Confirm the whole suite is fast**

```bash
source .venv/bin/activate && python -m pytest --durations=5
```

Expected: all PASS, total well under 30 seconds. If any single test exceeds a second, reduce the sizes it requests rather than marking it slow.

- [ ] **Step 4: Commit**

```bash
source .venv/bin/activate
ruff check --fix . && ruff format .
git add tests/test_zarr_roundtrip.py
git commit -m "Add zarr round-trip tests for both layouts"
```

---

### Task 8: Demonstrator notebook

**Files:**
- Create: `notebooks/gain_skeletons_demo.ipynb`

**Interfaces:**
- Consumes: the entire public API.
- Produces: nothing importable. The notebook defines no schema logic of its own; every dataset comes from the package.

- [ ] **Step 1: Register the kernel**

```bash
source .venv/bin/activate
python -m ipykernel install --user --name gain-skeletons --display-name "gain-skeletons"
```

- [ ] **Step 2: Write the notebook**

Build it with `nbformat` so the file is generated rather than hand-edited, then execute it. Create `/tmp/build_notebook.py` and run it with the venv Python. The notebook needs these cells, in order, each markdown cell explaining what the following code cell shows:

1. **Title and framing.** Markdown. State that the package is a demonstrator, that all values are random, and cite the deck at `docs/reference/calibration-dataset-coordinate-dimensions.pdf` with slides 6 and 7 as the catalogue.

2. **Imports and the catalogue.**
```python
import numpy as np
import xarray as xr

import gain_skeletons as gs

gs.list_cal_types()
```

3. **Coordinate factories on their own.** Show that each is usable standalone and that ranges are overridable.
```python
print(gs.time_coord(3, start=0.0, interval=8.0).values)
print(gs.frequency_coord(5).values)
print(gs.frequency_coord(5, start=1.0e9, end=2.0e9).values)
print(gs.antenna_name_coord(4).values)
gs.frequency_coord(4)
```

4. **The simplest case, `G`.** Markdown should quote slide 6's line
   `Standard "G": [nTime, nAnt, nFreq=1, nPol=2] (Complex) GAIN in [rel] (on-diag only)`
   and invite the reader to check the repr against it.
```python
gs.make_gain_xds("G")
```

5. **`B` against `G`: `nFreq=nCh` against `nFreq=1`.** Same axes, different extent.
```python
b = gs.make_gain_xds("B", n_frequency=64)
g = gs.make_gain_xds("G")
print("B dims:", dict(b.sizes))
print("G dims:", dict(g.sizes))
print("same axes:", b.GAIN.dims == g.GAIN.dims)
```

6. **`antpos`: axes genuinely absent, and meaningful parameter labels.** Markdown must explain that `{nFreq=0}` in the deck means the axis does not exist, which is materially different from a length-one axis.
```python
antpos = gs.make_gain_xds("antpos")
print("axes present:", antpos.ANTENNA_POSITION_OFFSET.dims)
print("frequency absent:", "frequency" not in antpos.dims)
print("parameter labels:", list(antpos.parameter_label.values))
antpos
```

7. **`ionosphere`: the direction axis.** Note that direction is an index into a direction list, per slide 8, and appears only for genuine direction dependence.
```python
gs.make_gain_xds("ionosphere", n_direction=4)
```

8. **Fringefit both ways.** The heart of the notebook. Markdown must lay out the trade-off: consolidated keeps one solve's parameters adjacent and needs one flag, but broadcasts the unpolarised `DISP_DELAY` over receptors; split keeps exact axes and scalar units, but fragments one solve into four datasets with four flags.
```python
consolidated = gs.make_gain_xds("fringefit")
split = gs.make_split_gain_xds("fringefit")

print("consolidated arrays:", list(consolidated.data_vars))
print("split datasets     :", list(split))
consolidated
```

9. **Units in the consolidated layout.**
```python
print("scalar units attr:", consolidated.PARAMETER.attrs.get("units", "<absent>"))
print("parameter_units  :", list(consolidated.parameter_units.values))
print("selecting DELAY  :", consolidated.sel(parameter_label="DELAY").parameter_units.item())
print("split units      :", {k: v[k].attrs["units"] for k, v in split.items()})
```

10. **The cost of consolidating.**
```python
disp = consolidated.PARAMETER.sel(parameter_label="DISP_DELAY")
print("DISP_DELAY repeated across receptors:",
      np.array_equal(disp.sel(receptor_label="X").values, disp.sel(receptor_label="Y").values))
print("consolidated flag dims:", consolidated.FLAG.dims)
print("split flag dims       :", {k: v.FLAG.dims for k, v in split.items()})
print("split DISP_DELAY axes :", split["DISP_DELAY"].DISP_DELAY.dims)
```

11. **Round-trip to zarr and show the on-disk layout.** Use a `tempfile.TemporaryDirectory` so the notebook leaves nothing behind, and print the directory tree for both layouts so the one-array-against-four difference is visible.
```python
import tempfile
from pathlib import Path

def show_tree(path: Path, limit: int = 24) -> None:
    """Print the top two levels of a zarr store."""
    entries = sorted(p.relative_to(path) for p in path.rglob("*") if len(p.relative_to(path).parts) <= 2)
    for entry in entries[:limit]:
        print("  ", entry)
    if len(entries) > limit:
        print(f"   ... and {len(entries) - limit} more")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    consolidated.to_zarr(root / "fringefit_consolidated.zarr", consolidated=False)
    print("consolidated store:")
    show_tree(root / "fringefit_consolidated.zarr")

    for name, xds in split.items():
        xds.to_zarr(root / f"fringefit_{name}.zarr", consolidated=False)
    print("\nsplit stores:", sorted(p.name for p in root.glob("fringefit_*.zarr")))

    reread = xr.open_dataset(
        root / "fringefit_consolidated.zarr", engine="zarr", consolidated=False
    ).load()
    print("\nround-trip identical:", reread.identical(consolidated))
```

12. **The escape hatch.** A hand-written `CalSpec` for a calibration type not in the deck, to show the registry is a convenience rather than a limit.
```python
pointing = gs.CalSpec(
    name="pointing_offset",
    parameters=(
        gs.ParamSpec(
            name="POINTING_OFFSET",
            units="rad",
            axes=("time", "antenna_name", "parameter_label"),
            dtype="float64",
            labels=("dAZ", "dEL"),
            scale=1.0e-4,
        ),
    ),
    default_sizes={"time": 4, "antenna_name": 8},
    description="Antenna pointing correction; not in the source deck.",
)

gs.make_gain_xds(pointing)
```

- [ ] **Step 3: Execute the notebook end to end**

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=gain-skeletons \
  notebooks/gain_skeletons_demo.ipynb
```

Expected: exits zero with no traceback in any cell. If a cell errors, fix the notebook, not the package, unless the error reveals a genuine package bug — in which case add a failing test to the relevant test file first.

- [ ] **Step 4: Commit**

```bash
git add notebooks/gain_skeletons_demo.ipynb
git commit -m "Add demonstrator notebook"
```

---

### Task 9: README, CLAUDE.md, and spec reconciliation

**Files:**
- Modify: `README.md`
- Create: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-08-06-gain-skeletons-design.md`

**Interfaces:**
- Consumes: the finished package.
- Produces: documentation only.

- [ ] **Step 1: Reconcile the spec with what was built**

Edit `docs/superpowers/specs/2026-08-06-gain-skeletons-design.md`:

- In the `spec.py` section, move `dtype` from `CalSpec` to `ParamSpec` and add a sentence explaining why: a single dtype on `CalSpec` makes the documented mixed-dtype `ValueError` unreachable.
- Note that `CalSpec.direction_dependent` is derived from the presence of the `direction` axis rather than declared.
- Note that `default_sizes` is required for every sized axis in use, not optional.
- Add `consolidated=False` to the zarr guidance, on **both** write and read, with the zarr format 3 reason. The design document currently shows the read as `xr.open_dataset(path, engine="zarr")`, which emits a `RuntimeWarning` because it hunts for consolidated metadata that was never written.
- Note that `jones_structure` is omitted from dataset attributes when None rather than stored as null.

- [ ] **Step 2: Write the README**

Replace `README.md`. Cover, in this order: what the package is and that it is a demonstrator with random values; install (`uv venv`, `uv pip install -e ".[dev,notebook]"`); a minimal example (`gs.make_gain_xds("B")`); the three layers with one sentence each; the two layouts and when each is preferable; a table of the ten registry entries; how to run the tests; and a pointer to the deck and the notebook. Keep it under 150 lines.

- [ ] **Step 3: Write CLAUDE.md**

Create `CLAUDE.md`, prefixed exactly:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

Then cover only what is not discoverable from a glance at the tree:

- **Commands.** Activate `.venv` for every Python call, never system Python, never `uv run`. Full suite: `source .venv/bin/activate && python -m pytest`. Single test: `python -m pytest tests/test_registry.py::test_fringefit_parameters_match_deck -v`. Lint: `ruff check . && ruff format .`. Execute the notebook: the `nbconvert` command from Task 8.
- **The one architectural idea that needs several files to see.** A calibration type is data, not code: `registry.py` holds `CalSpec` objects, and `builder.py` is a generic interpreter of them. Adding a calibration type means adding a `CalSpec`, never touching the builders.
- **Two layouts, one spec.** `make_gain_xds` consolidates every parameter into one array; `make_split_gain_xds` gives each its own dataset. They agree for the nine single-parameter types, and `test_builder_split.py` pins that. Both builders must draw parameter values before flags from an identically seeded generator, or that equivalence breaks.
- **Axis presence versus axis extent.** The deck's `{nFreq=0}` means the axis is absent; `nFreq=1` means present with length one. `ParamSpec.axes` encodes presence, `CalSpec.default_sizes` encodes extent. Never substitute a length-one axis for an absent one.
- **`tests/test_registry.py` duplicates `registry.py` on purpose.** It is an independent transcription of slides 6 and 7, so it checks the registry against the source deck rather than against itself. Do not factor the two tables together. When they disagree, verify against the PDF.
- **`FLAG` never carries `parameter_label`.** A flag marks a solution bad, and one solution's components are not independently valid.
- **zarr specifics.** Always `consolidated=False`, on **both** write and read; zarr format 3 warns otherwise, and the read-side warning is a `RuntimeWarning` that a `-W error::UserWarning` check will not catch. There is deliberately no I/O wrapper — use `xds.to_zarr(path, consolidated=False)` and `xr.open_dataset(path, engine="zarr", consolidated=False)` directly.
- **Where the source of truth lives.** `docs/reference/calibration-dataset-coordinate-dimensions.pdf`, slides 6 and 7. The design rationale, including the two deliberate departures from the deck, is in `docs/superpowers/specs/2026-08-06-gain-skeletons-design.md`.

Do not restate the user's global rules, and do not list every file.

- [ ] **Step 4: Verify the README example actually runs**

```bash
source .venv/bin/activate
python -c "
import gain_skeletons as gs
xds = gs.make_gain_xds('B')
print(xds.GAIN.dims, xds.GAIN.shape, xds.GAIN.dtype)
"
```

Expected: `('time', 'antenna_name', 'frequency', 'receptor_label') (4, 8, 64, 2) complex64`.

- [ ] **Step 5: Final full check**

```bash
source .venv/bin/activate
ruff check . && ruff format --check .
python -m pytest -v
```

Expected: lint clean, all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/superpowers/specs/2026-08-06-gain-skeletons-design.md
git commit -m "Add README and CLAUDE.md, reconcile spec with implementation"
```

---

## Validation already performed

Two things were checked against the real toolchain before this plan was written, so
neither is an assumption:

- **Zarr round-trip.** On xarray 2026.7.0 / zarr 3.3.0 / numpy 2.5.1, a dataset with a
  `complex64` data array, a `bool` FLAG, `<U4` string coordinates, a non-dimension
  `parameter_units` coordinate, and both dataset and coordinate attributes round-trips
  with `xds.identical(reread)` True. `consolidated=False` is required to avoid a
  `ZarrUserWarning`; zarr format 3 does not specify consolidated metadata.
- **Builder core.** The consolidated fill loop, the broadcast helper, and the
  layout-equivalence claim were prototyped and run. Equivalence holds for `G`, `B`, `T`,
  `J` and `antpos` — including the two types whose single parameter spans several
  labels — and Fringefit correctly yields four split datasets, broadcasts `DISP_DELAY`
  across receptors while leaving `PHASE` unbroadcast, and moves units to
  `parameter_units`.

Two bugs were found and fixed during that validation, both already corrected in the task
code above. They are called out here because they are easy to reintroduce:

1. The consolidated fill loop must not slice the trailing axis when the calibration type
   has no `parameter_label`. For `G`, `B`, `D`, `T` and `opacity` the trailing axis is
   `receptor_label`, and slicing it raises a shape mismatch. Hence the `has_parameter_axis`
   branch.
2. `CalSpec` requires a default size for every sized axis in use, so test helpers must
   supply a complete `default_sizes`, not a partial one.

## Self-review notes

Spec coverage checked section by section:

| Spec section | Task |
| --- | --- |
| Purpose, non-goals | 9 (README, CLAUDE.md) |
| Deck principles table | 4 (registry), 5 and 6 (builders) |
| `axes.py` factory table | 2 |
| `spec.py` dataclasses, label sourcing | 3 |
| Canonical axis order | 2 (`CANONICAL_AXES`, `sorted_axes`) |
| Optional axes, presence versus extent | 3 (validation), 4 (`default_sizes`), 5 (absence tests) |
| Two layouts, trade-off | 5, 6 |
| Units, `parameter_units` | 5 |
| Flags | 5, 6 |
| Random data, reproducibility | 5 |
| Dataset attributes | 5 |
| Registry, all ten entries | 4 |
| Every listed test | 2, 3, 4, 5, 6, 7 |
| Notebook, all eight narrative beats | 8 |
| Packaging, ruff, venv | 1 |
| Deferred `CLAUDE.md` and `README.md` | 9 |

One deliberate divergence from the spec, recorded in Task 3 and reconciled in Task 9 Step 1: `dtype` sits on `ParamSpec`, not `CalSpec`, because the spec's own mixed-dtype `ValueError` is otherwise unreachable.
