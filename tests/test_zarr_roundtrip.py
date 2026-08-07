"""Tests that generated datasets survive a zarr round-trip.

Reading and writing is plain xarray: xds.to_zarr(path) and
xr.open_dataset(path, engine="zarr"). The package deliberately wraps neither.

consolidated=False on both write and read: zarr format 3 does not specify
consolidated metadata, so xarray warns if asked to write it, and warns again if
asked to look for metadata that was never written.
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
    return xr.open_dataset(path, engine="zarr", consolidated=False).load()


@pytest.mark.parametrize("key", list_cal_types())
def test_consolidated_layout_survives_roundtrip(key, tmp_path):
    xds = make_gain_xds(key)
    assert xds.identical(roundtrip(xds, tmp_path / f"{key}.zarr"))


@pytest.mark.parametrize("key", list_cal_types())
def test_split_layout_survives_roundtrip(key, tmp_path):
    xds = make_split_gain_xds(key)
    assert xds.identical(roundtrip(xds, tmp_path / f"{key}_split.zarr"))


# Complex gains are the case most likely to be mangled by a storage layer, so
# it gets its own assertion rather than relying on identical() alone.
def test_complex_dtype_is_preserved_exactly(tmp_path):
    xds = make_gain_xds("bandpass")
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert read.GAIN.dtype == np.complex64
    np.testing.assert_array_equal(read.GAIN.values, xds.GAIN.values)


def test_boolean_flags_are_preserved(tmp_path):
    xds = make_gain_xds("bandpass", flag_fraction=0.5)
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert read.FLAG.dtype == np.bool_
    np.testing.assert_array_equal(read.FLAG.values, xds.FLAG.values)


def test_string_coordinates_are_preserved(tmp_path):
    xds = make_gain_xds("bandpass")
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert list(read.antenna_name.values) == list(xds.antenna_name.values)
    assert list(read.receptor_label.values) == list(xds.receptor_label.values)


# parameter_units is a non-dimension coordinate, which is the part of the
# consolidated layout most at risk of being demoted to a data variable.
def test_parameter_units_survives_as_a_coordinate(tmp_path):
    xds = make_gain_xds("fringe_fit")
    read = roundtrip(xds, tmp_path / "fringe_fit.zarr")
    assert "parameter_units" in read.coords
    assert list(read.parameter_units.values) == ["deg", "s", "s/s", "s"]


def test_coordinate_attributes_are_preserved(tmp_path):
    xds = make_gain_xds("bandpass")
    read = roundtrip(xds, tmp_path / "b.zarr")
    assert read.time.attrs["type"] == "time"
    assert read.frequency.attrs["units"] == "Hz"


def test_dataset_and_variable_attributes_are_preserved(tmp_path):
    xds = make_gain_xds("dd_phenomenological_gain")
    read = roundtrip(xds, tmp_path / "dd.zarr")
    assert read.attrs["cal_type"] == "dd_phenomenological_gain"
    assert read.attrs["direction_dependent"] is True
    assert read.attrs["jones_structure"] == "full"
    assert read.GAIN.attrs["units"] == "rel"


def stored_arrays(xds: xr.Dataset, path) -> set[str]:
    """Write a dataset to zarr and report the arrays the store holds.

    Args:
        xds: Dataset to write.
        path: Destination store path.

    Returns:
        Names of the top-level arrays in the store, data variables and
        coordinates alike.
    """
    xds.to_zarr(path, consolidated=False)
    return {entry.name for entry in path.iterdir() if entry.is_dir()}


# The layouts differ on disk as well as in memory: one chunked parameter array
# against four that chunk and compress independently. Each is one store either
# way. This is the difference the notebook exists to show.
def test_consolidated_stores_one_parameter_array_where_split_stores_four(tmp_path):
    consolidated = make_gain_xds("fringe_fit")
    split = make_split_gain_xds("fringe_fit")

    # A zarr store holds one top-level directory per array, whether that array
    # is a data variable or a coordinate.
    on_disk = stored_arrays(consolidated, tmp_path / "consolidated.zarr")
    assert on_disk == set(consolidated.data_vars) | set(consolidated.coords)
    assert set(consolidated.data_vars) == {"PARAMETER", "FLAG"}

    split_on_disk = stored_arrays(split, tmp_path / "split.zarr")
    assert split_on_disk == set(split.data_vars) | set(split.coords)
    assert set(split.data_vars) == {"PHASE", "DELAY", "RATE", "DISP_DELAY", "FLAG"}

    # Consolidating buys locality and pays for it with two extra coordinate
    # arrays: the parameter labels, and the units that vary along them.
    assert on_disk - split_on_disk == {"PARAMETER", "parameter_label", "parameter_units"}
