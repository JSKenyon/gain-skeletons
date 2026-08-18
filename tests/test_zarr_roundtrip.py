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

from gain_skeletons.builder import make_gain_xds
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
def test_every_registered_type_survives_roundtrip(key, tmp_path):
    xds = make_gain_xds(key)
    assert xds.identical(roundtrip(xds, tmp_path / f"{key}.zarr"))


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


# Each quantity's units ride on its own array, so a type whose quantities are
# differently united needs nothing beyond per-array attributes to survive.
def test_per_array_units_survive_for_a_multi_parameter_type(tmp_path):
    xds = make_gain_xds("fringe_fit")
    read = roundtrip(xds, tmp_path / "fringe_fit.zarr")
    assert {
        name: read[name].attrs["units"] for name in ("PHASE", "DELAY", "RATE", "DISP_DELAY")
    } == {
        "PHASE": "deg",
        "DELAY": "s",
        "RATE": "s/s",
        "DISP_DELAY": "s",
    }
    assert "parameter_units" not in read.coords


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


# One solve is one store, and within it one array per quantity plus one per
# coordinate. A multi-parameter type therefore chunks and compresses each of its
# quantities independently, which is the trade the layout accepts in exchange
# for every array keeping its own exact axes and units.
def test_each_quantity_gets_its_own_array_in_the_store(tmp_path):
    xds = make_gain_xds("fringe_fit")

    # A zarr store holds one top-level directory per array, whether that array
    # is a data variable or a coordinate.
    on_disk = stored_arrays(xds, tmp_path / "fringe_fit.zarr")
    assert on_disk == set(xds.data_vars) | set(xds.coords)
    assert set(xds.data_vars) == {"PHASE", "DELAY", "RATE", "DISP_DELAY", "FLAG"}
    assert "parameter_units" not in on_disk
