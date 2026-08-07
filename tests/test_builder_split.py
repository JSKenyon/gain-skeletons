"""Tests for the split dataset builder, and for equivalence between layouts."""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds, make_split_gain_xds
from gain_skeletons.registry import get_spec, list_cal_types
from gain_skeletons.spec import CalSpec, ParamSpec

SINGLE_PARAMETER_KEYS = [key for key in list_cal_types() if len(get_spec(key).parameters) == 1]


def test_returns_one_dataset_holding_an_array_per_parameter():
    xds = make_split_gain_xds("fringe_fit")
    assert isinstance(xds, xr.Dataset)
    assert set(xds.data_vars) == {"PHASE", "DELAY", "RATE", "DISP_DELAY", "FLAG"}


def test_single_parameter_type_yields_one_array_and_a_flag():
    assert set(make_split_gain_xds("antenna_gain").data_vars) == {"GAIN", "FLAG"}


# Splitting exists so that units can be a scalar attribute on every array.
def test_every_array_carries_scalar_units():
    xds = make_split_gain_xds("fringe_fit")
    assert xds.PHASE.attrs["units"] == "deg"
    assert xds.DELAY.attrs["units"] == "s"
    assert xds.RATE.attrs["units"] == "s/s"
    assert xds.DISP_DELAY.attrs["units"] == "s"


def test_no_parameter_units_coord_is_needed():
    assert "parameter_units" not in make_split_gain_xds("fringe_fit").coords


# Each array carries its parameter's name, so a parameter axis holding one
# label that restates that name would say nothing. It is dropped rather than
# kept at length one.
def test_single_label_parameter_axis_is_dropped():
    xds = make_split_gain_xds("fringe_fit")
    assert "parameter_label" not in xds.dims
    assert "parameter_label" not in xds.coords


# The axis survives where it distinguishes components within one parameter,
# which is a different job from distinguishing one parameter from another.
def test_multi_label_parameter_axis_survives():
    xds = make_split_gain_xds("antenna_positions")
    assert list(xds.parameter_label.values) == ["dX", "dY", "dZ"]
    assert xds.ANTENNA_POSITION_OFFSET.dims == ("time", "antenna_name", "parameter_label")


# Splitting keeps each quantity's exact axes, so the unpolarised dispersive
# delay is not padded out over receptors.
def test_unpolarised_quantity_keeps_no_receptor_axis():
    xds = make_split_gain_xds("fringe_fit")
    assert "receptor_label" not in xds.DISP_DELAY.dims
    assert "receptor_label" in xds.PHASE.dims


# One solve, one flag, in this layout as in the consolidated one. The flag
# spans every axis some parameter uses, so it still covers a quantity defined
# over fewer of them.
def test_one_flag_spans_every_axis_any_parameter_uses():
    xds = make_split_gain_xds("fringe_fit")
    assert xds.FLAG.dims == ("time", "antenna_name", "frequency", "receptor_label")
    assert xds.FLAG.dtype == np.bool_


def test_flag_drops_the_parameter_axis():
    xds = make_split_gain_xds("antenna_positions")
    assert xds.FLAG.dims == ("time", "antenna_name")


def test_dataset_attributes_are_carried_through():
    xds = make_split_gain_xds("dd_phenomenological_gain")
    assert xds.attrs["cal_type"] == "dd_phenomenological_gain"
    assert xds.attrs["direction_dependent"] is True


def test_size_overrides_are_honoured():
    xds = make_split_gain_xds("bandpass", n_time=2, n_antenna=3, n_frequency=16)
    assert xds.GAIN.shape == (2, 3, 16, 2)


def test_size_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_split_gain_xds("antenna_positions", n_frequency=64)


# coord_kwargs configures a generated range. Label axes take their values from
# labels, so naming one is a user error rather than something to silently
# discard, and this must hold for the split builder too.
def test_coord_kwargs_for_label_axis_is_rejected():
    with pytest.raises(ValueError, match="receptor_labels"):
        make_split_gain_xds("antenna_gain", coord_kwargs={"receptor_label": {"labels": ("R", "L")}})


def test_invalid_flag_fraction_is_rejected():
    with pytest.raises(ValueError, match="flag_fraction"):
        make_split_gain_xds("antenna_gain", flag_fraction=2.0)


# A calibration type mixing dtypes cannot consolidate, but splitting it is
# perfectly well defined: the arrays are separate, so nothing forces a common
# dtype on them.
def test_mixed_dtype_spec_splits_successfully():
    spec = CalSpec(
        name="mixed",
        parameters=(
            ParamSpec("A", "rel", ("time", "parameter_label"), "complex64"),
            ParamSpec("B", "m", ("time", "parameter_label"), "float64"),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    xds = make_split_gain_xds(spec)
    assert xds.A.dtype == np.complex64
    assert xds.B.dtype == np.float64


# Consolidating a multi-parameter spec with no parameter axis is rejected,
# because there would be nowhere to put the parameters. Splitting it is well
# defined: the parameters are separate arrays, which need no axis to tell them
# apart.
def test_multi_parameter_spec_without_parameter_axis_splits_successfully():
    spec = CalSpec(
        name="no_parameter_axis",
        parameters=(
            ParamSpec("A", "m", ("time", "antenna_name"), "float64"),
            ParamSpec("B", "m", ("time", "antenna_name"), "float64"),
        ),
        default_sizes={"time": 4, "antenna_name": 8},
        consolidated_name="PARAMETER",
    )
    with pytest.raises(ValueError, match="no parameter_label axis"):
        make_gain_xds(spec)

    xds = make_split_gain_xds(spec)
    assert set(xds.data_vars) == {"A", "B", "FLAG"}
    assert xds.A.dims == ("time", "antenna_name")
    assert "parameter_label" not in xds.dims


# Arrays sharing one parameter axis cannot disagree about its labels. No
# registered type does this, but a hand-written spec can, and silently keeping
# one parameter's labels would mislabel the other's components.
def test_conflicting_multi_label_parameters_cannot_split():
    spec = CalSpec(
        name="conflicting_labels",
        parameters=(
            ParamSpec("A", "m", ("time", "parameter_label"), "float64", labels=("dX", "dY", "dZ")),
            ParamSpec("B", "m", ("time", "parameter_label"), "float64", labels=("dAZ", "dEL")),
        ),
        default_sizes={"time": 4},
        consolidated_name="PARAMETER",
    )
    with pytest.raises(ValueError, match="different labels"):
        make_split_gain_xds(spec)


# Nine of the eleven registry entries declare a single parameter. For those the
# layout choice is a distinction without a difference, and this pins that
# claim so neither builder can drift from the other.
@pytest.mark.parametrize("key", SINGLE_PARAMETER_KEYS)
def test_layouts_agree_for_single_parameter_types(key):
    assert make_gain_xds(key, seed=11).identical(make_split_gain_xds(key, seed=11))


# The layouts agree for single-parameter types because the consolidated array
# takes its name from the sole parameter. A spec that overrides
# consolidated_name breaks that, which is why the docstring qualifies it.
def test_custom_consolidated_name_breaks_single_parameter_equivalence():
    spec = CalSpec(
        name="odd",
        parameters=(ParamSpec("VAL", "m", ("time",), "float64"),),
        default_sizes={"time": 3},
        consolidated_name="PARAMETER",
    )
    consolidated = make_gain_xds(spec, seed=5)
    split = make_split_gain_xds(spec, seed=5)
    assert "PARAMETER" in consolidated.data_vars
    assert "VAL" in split.data_vars
    assert not consolidated.identical(split)


def test_only_the_multi_parameter_types_differ_between_layouts():
    differing = [
        key
        for key in list_cal_types()
        if not make_gain_xds(key, seed=11).identical(make_split_gain_xds(key, seed=11))
    ]
    assert differing == ["delay", "fringe_fit"]


# Delay splits like fringe fit, but with nothing to un-broadcast: both of its
# quantities already carry the receptor axis. What splitting buys it is the
# scalar units attribute alone.
def test_delay_splits_into_two_arrays_with_scalar_units():
    xds = make_split_gain_xds("delay")
    assert set(xds.data_vars) == {"PHASE", "DELAY", "FLAG"}
    assert xds.PHASE.attrs["units"] == "deg"
    assert xds.DELAY.attrs["units"] == "s"
    assert xds.PHASE.dims == xds.DELAY.dims
    assert "parameter_units" not in xds.coords
