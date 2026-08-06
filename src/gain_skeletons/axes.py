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
