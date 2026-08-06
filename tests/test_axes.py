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
