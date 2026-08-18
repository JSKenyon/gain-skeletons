"""Tests for the dataset builder."""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds
from gain_skeletons.spec import CalSpec, ParamSpec


def test_returns_one_dataset_holding_an_array_per_parameter():
    xds = make_gain_xds("fringe_fit")
    assert isinstance(xds, xr.Dataset)
    assert set(xds.data_vars) == {"PHASE", "DELAY", "RATE", "DISP_DELAY", "FLAG"}


def test_single_parameter_type_yields_one_array_and_a_flag():
    assert set(make_gain_xds("antenna_gain").data_vars) == {"GAIN", "FLAG"}


# One array per parameter means one unit per array, so units are always a
# scalar attribute. There is no case where they have to move elsewhere.
def test_every_array_carries_scalar_units():
    xds = make_gain_xds("fringe_fit")
    assert xds.PHASE.attrs["units"] == "deg"
    assert xds.DELAY.attrs["units"] == "s"
    assert xds.RATE.attrs["units"] == "s/s"
    assert xds.DISP_DELAY.attrs["units"] == "s"


def test_units_never_become_a_coordinate():
    assert "parameter_units" not in make_gain_xds("fringe_fit").coords


# Units survive subsetting for free, because they describe the array rather
# than positions along an axis. This is the property a per-label mapping could
# not offer.
def test_units_survive_subsetting():
    subset = make_gain_xds("fringe_fit").isel(time=slice(0, 2), frequency=0)
    assert subset.DELAY.attrs["units"] == "s"
    assert subset.RATE.attrs["units"] == "s/s"


# A parameter axis appears only where a parameter declares one. Fringe fit's
# four quantities are four arrays named for themselves, so none of them needs
# an axis to say which quantity it holds.
def test_a_type_whose_parameters_declare_no_parameter_axis_has_none():
    xds = make_gain_xds("fringe_fit")
    assert "parameter_label" not in xds.dims
    assert "parameter_label" not in xds.coords


# The axis appears where it distinguishes components within one parameter.
def test_multi_label_parameter_axis_is_built():
    xds = make_gain_xds("antenna_positions")
    assert list(xds.parameter_label.values) == ["dX", "dY", "dZ"]
    assert xds.ANTENNA_POSITION_OFFSET.dims == ("time", "antenna_name", "parameter_label")


# Presence is what a ParamSpec declares, not something the builder derives. A
# parameter asking for the axis with one label gets it, at length one.
def test_a_declared_single_label_parameter_axis_is_kept_at_length_one():
    spec = CalSpec(
        name="one_label",
        parameters=(ParamSpec("VAL", "m", ("time", "parameter_label"), "float64"),),
        default_sizes={"time": 3},
    )
    xds = make_gain_xds(spec)
    assert xds.VAL.dims == ("time", "parameter_label")
    assert list(xds.parameter_label.values) == ["VAL"]


# Each array keeps its parameter's exact axes, so the unpolarised dispersive
# delay is not padded out over receptors.
def test_unpolarised_quantity_keeps_no_receptor_axis():
    xds = make_gain_xds("fringe_fit")
    assert "receptor_label" not in xds.DISP_DELAY.dims
    assert "receptor_label" in xds.PHASE.dims


# One solve, one flag. The flag spans every axis some parameter uses, less the
# component axes, so it still covers a quantity defined over fewer of them.
def test_one_flag_spans_every_axis_any_parameter_uses():
    xds = make_gain_xds("fringe_fit")
    assert xds.FLAG.dims == ("time", "antenna_name", "frequency")
    assert xds.FLAG.dtype == np.bool_


def test_flag_carries_neither_component_axis():
    assert make_gain_xds("antenna_positions").FLAG.dims == ("time", "antenna_name")
    assert make_gain_xds("bandpass").FLAG.dims == ("time", "antenna_name", "frequency")


def test_dataset_attributes_are_carried_through():
    xds = make_gain_xds("dd_phenomenological_gain")
    assert xds.attrs["cal_type"] == "dd_phenomenological_gain"
    assert xds.attrs["direction_dependent"] is True


def test_size_overrides_are_honoured():
    xds = make_gain_xds("bandpass", n_time=2, n_antenna=3, n_frequency=16)
    assert xds.GAIN.shape == (2, 3, 16, 2)


def test_size_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_gain_xds("antenna_positions", n_frequency=64)


# coord_kwargs configures a generated range. Label axes take their values from
# labels, so naming one is a user error rather than something to discard.
def test_coord_kwargs_for_label_axis_is_rejected():
    with pytest.raises(ValueError, match="receptor_labels"):
        make_gain_xds("antenna_gain", coord_kwargs={"receptor_label": {"labels": ("R", "L")}})


def test_invalid_flag_fraction_is_rejected():
    with pytest.raises(ValueError, match="flag_fraction"):
        make_gain_xds("antenna_gain", flag_fraction=2.0)


# Separate arrays are under no obligation to share a dtype, so a calibration
# type mixing complex and real quantities is well defined.
def test_mixed_dtype_spec_builds_successfully():
    spec = CalSpec(
        name="mixed",
        parameters=(
            ParamSpec("A", "rel", ("time",), "complex64"),
            ParamSpec("B", "m", ("time",), "float64"),
        ),
        default_sizes={"time": 4},
    )
    xds = make_gain_xds(spec)
    assert xds.A.dtype == np.complex64
    assert xds.B.dtype == np.float64


# Several parameters need no axis to tell them apart: their array names do it.
def test_multi_parameter_spec_without_parameter_axis_builds():
    spec = CalSpec(
        name="no_parameter_axis",
        parameters=(
            ParamSpec("A", "m", ("time", "antenna_name"), "float64"),
            ParamSpec("B", "m", ("time", "antenna_name"), "float64"),
        ),
        default_sizes={"time": 4, "antenna_name": 8},
    )
    xds = make_gain_xds(spec)
    assert set(xds.data_vars) == {"A", "B", "FLAG"}
    assert xds.A.dims == ("time", "antenna_name")
    assert "parameter_label" not in xds.dims


def test_delay_yields_two_arrays_with_scalar_units():
    xds = make_gain_xds("delay")
    assert set(xds.data_vars) == {"PHASE", "DELAY", "FLAG"}
    assert xds.PHASE.attrs["units"] == "deg"
    assert xds.DELAY.attrs["units"] == "s"
    assert xds.PHASE.dims == xds.DELAY.dims


def test_the_same_seed_reproduces_the_same_dataset():
    assert make_gain_xds("fringe_fit", seed=7).identical(make_gain_xds("fringe_fit", seed=7))
