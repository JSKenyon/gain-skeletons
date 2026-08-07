"""Tests for the split dataset builder, and for equivalence between layouts."""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds, make_split_gain_xds
from gain_skeletons.registry import get_spec, list_cal_types
from gain_skeletons.spec import CalSpec, ParamSpec

SINGLE_PARAMETER_KEYS = [key for key in list_cal_types() if len(get_spec(key).parameters) == 1]


def test_returns_a_mapping_keyed_by_parameter_name():
    datasets = make_split_gain_xds("fringe_fit")
    assert set(datasets) == {"PHASE", "DELAY", "RATE", "DISP_DELAY"}
    assert all(isinstance(xds, xr.Dataset) for xds in datasets.values())


def test_single_parameter_type_yields_one_dataset():
    datasets = make_split_gain_xds("antenna_gain")
    assert set(datasets) == {"GAIN"}


def test_each_dataset_holds_one_parameter_and_a_flag():
    for name, xds in make_split_gain_xds("fringe_fit").items():
        assert set(xds.data_vars) == {name, "FLAG"}


# Splitting exists so that units can be a scalar attribute on every array.
def test_every_array_carries_scalar_units():
    datasets = make_split_gain_xds("fringe_fit")
    assert datasets["PHASE"].PHASE.attrs["units"] == "deg"
    assert datasets["DELAY"].DELAY.attrs["units"] == "s"
    assert datasets["RATE"].RATE.attrs["units"] == "s/s"
    assert datasets["DISP_DELAY"].DISP_DELAY.attrs["units"] == "s"


def test_no_dataset_needs_a_parameter_units_coord():
    for xds in make_split_gain_xds("fringe_fit").values():
        assert "parameter_units" not in xds.coords


# Each fringe fit quantity is one parameter, so in the split layout the
# parameter axis is present but length one.
def test_each_fringe_fit_dataset_has_a_length_one_parameter_axis():
    for name, xds in make_split_gain_xds("fringe_fit").items():
        assert xds.sizes["parameter_label"] == 1
        assert list(xds.parameter_label.values) == [name]


# Splitting keeps each quantity's exact axes, so the unpolarised dispersive
# delay is not padded out over receptors.
def test_unpolarised_quantity_keeps_no_receptor_axis():
    datasets = make_split_gain_xds("fringe_fit")
    assert "receptor_label" not in datasets["DISP_DELAY"].dims
    assert "receptor_label" in datasets["PHASE"].dims


def test_flag_drops_the_parameter_axis_in_every_dataset():
    for name, xds in make_split_gain_xds("fringe_fit").items():
        expected = tuple(dim for dim in xds[name].dims if dim != "parameter_label")
        assert xds.FLAG.dims == expected
        assert xds.FLAG.dtype == np.bool_


def test_each_quantity_gets_its_own_flag():
    datasets = make_split_gain_xds("fringe_fit", flag_fraction=0.5, seed=3)
    phase_flags = datasets["PHASE"].FLAG.values
    rate_flags = datasets["RATE"].FLAG.values
    assert not np.array_equal(phase_flags, rate_flags)


def test_dataset_attributes_are_carried_through():
    xds = make_split_gain_xds("dd_phenomenological_gain")["GAIN"]
    assert xds.attrs["cal_type"] == "dd_phenomenological_gain"
    assert xds.attrs["direction_dependent"] is True


def test_size_overrides_are_honoured():
    xds = make_split_gain_xds("bandpass", n_time=2, n_antenna=3, n_frequency=16)["GAIN"]
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
# perfectly well defined.
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
    datasets = make_split_gain_xds(spec)
    assert datasets["A"].A.dtype == np.complex64
    assert datasets["B"].B.dtype == np.float64


# Consolidating a multi-parameter spec with no parameter axis is rejected,
# because there would be nowhere to put the parameters. Splitting it is well
# defined: each parameter gets its own dataset.
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

    datasets = make_split_gain_xds(spec)
    assert set(datasets) == {"A", "B"}
    assert datasets["A"].A.dims == ("time", "antenna_name")
    assert "parameter_label" not in datasets["A"].dims


# Nine of the eleven registry entries declare a single parameter. For those the
# layout choice is a distinction without a difference, and this pins that
# claim so neither builder can drift from the other.
@pytest.mark.parametrize("key", SINGLE_PARAMETER_KEYS)
def test_layouts_agree_for_single_parameter_types(key):
    consolidated = make_gain_xds(key, seed=11)
    split = make_split_gain_xds(key, seed=11)
    assert len(split) == 1
    (only,) = split.values()
    assert consolidated.identical(only)


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
    (only,) = make_split_gain_xds(spec, seed=5).values()
    assert "PARAMETER" in consolidated.data_vars
    assert "VAL" in only.data_vars
    assert not consolidated.identical(only)


def test_only_the_multi_parameter_types_differ_between_layouts():
    differing = []
    for key in list_cal_types():
        split = make_split_gain_xds(key, seed=11)
        if len(split) > 1 or not make_gain_xds(key, seed=11).identical(next(iter(split.values()))):
            differing.append(key)
    assert differing == ["delay", "fringe_fit"]


# Delay splits like fringe fit, but with nothing to un-broadcast: both of its
# quantities already carry the receptor axis.
def test_delay_splits_into_two_datasets_with_scalar_units():
    datasets = make_split_gain_xds("delay")
    assert set(datasets) == {"PHASE", "DELAY"}
    assert datasets["PHASE"].PHASE.attrs["units"] == "deg"
    assert datasets["DELAY"].DELAY.attrs["units"] == "s"
    for xds in datasets.values():
        assert "receptor_label" in xds.dims
        assert "parameter_units" not in xds.coords
