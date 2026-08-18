"""Tests for the consolidated dataset builder."""

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds
from gain_skeletons.registry import get_spec, list_cal_types
from gain_skeletons.spec import CalSpec, ParamSpec


def test_accepts_a_registry_name():
    xds = make_gain_xds("antenna_gain")
    assert isinstance(xds, xr.Dataset)
    assert xds.attrs["cal_type"] == "antenna_gain"


def test_accepts_a_cal_spec_object():
    assert make_gain_xds(get_spec("antenna_gain")).attrs["cal_type"] == "antenna_gain"


def test_g_has_the_dimensions_from_the_catalogue():
    xds = make_gain_xds("antenna_gain", n_time=4, n_antenna=8)
    assert xds.GAIN.dims == ("time", "antenna_name", "frequency", "receptor_label")
    assert xds.GAIN.shape == (4, 8, 1, 2)
    assert xds.GAIN.dtype == np.complex64


def test_data_array_carries_units_when_uniform():
    assert make_gain_xds("antenna_gain").GAIN.attrs["units"] == "rel"


def test_size_overrides_are_honoured():
    xds = make_gain_xds("bandpass", n_time=2, n_antenna=3, n_frequency=16)
    assert xds.GAIN.shape == (2, 3, 16, 2)


def test_direction_size_override_is_honoured():
    xds = make_gain_xds("dd_phenomenological_gain", n_direction=5)
    assert xds.sizes["direction"] == 5


# Silently ignoring a size for an axis the calibration type does not have would
# hide a real user error, so it raises.
def test_size_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_gain_xds("antenna_positions", n_frequency=64)


def test_direction_size_for_direction_independent_type_is_rejected():
    with pytest.raises(ValueError, match="has no 'direction' axis"):
        make_gain_xds("antenna_gain", n_direction=3)


@pytest.mark.parametrize("key", ["antenna_positions", "ionosphere"])
def test_absent_frequency_axis_is_genuinely_absent(key):
    xds = make_gain_xds(key)
    assert "frequency" not in xds.dims
    assert "frequency" not in xds.coords


@pytest.mark.parametrize(
    "key",
    ["tropospheric_gain", "opacity", "antenna_positions", "ionosphere"],
)
def test_absent_receptor_axis_is_genuinely_absent(key):
    xds = make_gain_xds(key)
    assert "receptor_label" not in xds.dims
    assert "receptor_label" not in xds.coords


@pytest.mark.parametrize(
    "key",
    ["antenna_gain", "bandpass", "leakage", "tropospheric_gain", "opacity"],
)
def test_types_without_a_parameter_axis_do_not_gain_one(key):
    assert "parameter_label" not in make_gain_xds(key).dims


def test_antenna_positions_parameter_labels():
    xds = make_gain_xds("antenna_positions")
    assert list(xds.parameter_label.values) == ["dX", "dY", "dZ"]
    assert xds.ANTENNA_POSITION_OFFSET.dims == ("time", "antenna_name", "parameter_label")


@pytest.mark.parametrize("key", ["phenomenological_gain", "dd_phenomenological_gain"])
def test_phenomenological_parameter_labels_are_the_jones_columns(key):
    assert list(make_gain_xds(key).parameter_label.values) == ["aligned", "cross"]


def test_receptor_labels_are_overridable():
    xds = make_gain_xds("antenna_gain", receptor_labels=("R", "L"))
    assert list(xds.receptor_label.values) == ["R", "L"]


def test_coord_kwargs_reach_the_factories():
    xds = make_gain_xds(
        "bandpass",
        n_frequency=3,
        coord_kwargs={"frequency": {"start": 1.0e9, "end": 2.0e9}},
    )
    np.testing.assert_allclose(xds.frequency.values, [1.0e9, 1.5e9, 2.0e9])


def test_coord_kwargs_for_absent_axis_is_rejected():
    with pytest.raises(ValueError, match="has no 'frequency' axis"):
        make_gain_xds("antenna_positions", coord_kwargs={"frequency": {"start": 1.0e9}})


# coord_kwargs configures a generated range. Label axes take their values from
# labels, and direction_coord takes no keywords, so naming either is a user
# error rather than something to silently discard.
def test_coord_kwargs_for_label_axis_is_rejected():
    with pytest.raises(ValueError, match="receptor_labels"):
        make_gain_xds("antenna_gain", coord_kwargs={"receptor_label": {"labels": ("R", "L")}})


def test_coord_kwargs_for_parameter_label_is_rejected():
    with pytest.raises(ValueError, match="no coordinate configuration"):
        make_gain_xds(
            "antenna_positions",
            coord_kwargs={"parameter_label": {"labels": ("a", "b", "c")}},
        )


def test_coord_kwargs_for_direction_is_rejected():
    with pytest.raises(ValueError, match="no coordinate configuration"):
        make_gain_xds("dd_phenomenological_gain", coord_kwargs={"direction": {"start": 1}})


@pytest.mark.parametrize("axis", ["time", "antenna_name", "frequency"])
def test_coord_kwargs_accepted_for_configurable_axes(axis):
    kwargs = {
        "time": {"interval": 4.0},
        "antenna_name": {"prefix": "ea"},
        "frequency": {"start": 1.0e9},
    }
    xds = make_gain_xds("bandpass", coord_kwargs={axis: kwargs[axis]})
    assert axis in xds.coords


# FLAG marks a whole solution bad. The components of one solution are not
# independently valid, so FLAG carries neither the parameter axis nor the
# receptor axis: a solve's quantities, and the receptors it solved together,
# stand or fall as one. The expected dims are spelled out rather than derived
# from UNFLAGGED_AXES, so widening that tuple cannot quietly widen the test.
@pytest.mark.parametrize("key", list_cal_types())
def test_flag_carries_neither_the_parameter_nor_the_receptor_axis(key):
    xds = make_gain_xds(key)
    spec = get_spec(key)
    parameter = xds[spec.resolved_consolidated_name]
    expected = tuple(
        dim for dim in parameter.dims if dim not in ("parameter_label", "receptor_label")
    )
    assert xds.FLAG.dims == expected


@pytest.mark.parametrize("key", list_cal_types())
def test_flag_is_boolean(key):
    assert make_gain_xds(key).FLAG.dtype == np.bool_


def test_flag_fraction_zero_gives_a_clean_dataset():
    assert not make_gain_xds("bandpass", flag_fraction=0.0).FLAG.values.any()


def test_flag_fraction_one_flags_everything():
    assert make_gain_xds("bandpass", flag_fraction=1.0).FLAG.values.all()


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_invalid_flag_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="flag_fraction"):
        make_gain_xds("antenna_gain", flag_fraction=fraction)


# Complex gains sit near unit amplitude. A uniform-random complex number of
# arbitrary magnitude would be physically nonsensical.
def test_complex_gains_are_generated_near_unit_amplitude():
    amplitude = np.abs(make_gain_xds("bandpass", n_time=8, n_frequency=64).GAIN.values)
    assert 0.5 < amplitude.mean() < 1.5


def test_float_parameters_respect_their_scale():
    # DELAY has scale 1e-9, so values should be nanosecond-ish, not order unity.
    delay = make_gain_xds("fringe_fit").PARAMETER.sel(parameter_label="DELAY").values
    assert np.abs(delay).max() < 1.0e-6


def test_equal_seeds_give_equal_values():
    a = make_gain_xds("bandpass", seed=7)
    b = make_gain_xds("bandpass", seed=7)
    assert a.identical(b)


def test_different_seeds_give_different_values():
    a = make_gain_xds("bandpass", seed=1)
    b = make_gain_xds("bandpass", seed=2)
    assert not np.array_equal(a.GAIN.values, b.GAIN.values)


def test_dataset_attributes_record_cal_type_and_direction_dependence():
    xds = make_gain_xds("dd_phenomenological_gain")
    assert xds.attrs["cal_type"] == "dd_phenomenological_gain"
    assert xds.attrs["direction_dependent"] is True
    assert xds.attrs["jones_structure"] == "full"


# Storing a null attribute is worse than omitting it: it asserts that the
# calibration type has a Jones structure whose value happens to be nothing.
# Every real-valued parameterised type leaves it unset: what such a type stores
# has to be evaluated before it is a Jones term at all.
@pytest.mark.parametrize(
    "key",
    ["opacity", "antenna_positions", "ionosphere", "delay", "fringe_fit"],
)
def test_jones_structure_is_omitted_when_the_type_has_none(key):
    assert "jones_structure" not in make_gain_xds(key).attrs


def test_time_coord_carries_msv4_attributes():
    assert make_gain_xds("antenna_gain").time.attrs["type"] == "time"


class TestFringeFitConsolidation:
    """The one entry where consolidation costs something as well as buying it."""

    def test_all_four_quantities_share_one_array(self):
        xds = make_gain_xds("fringe_fit")
        assert "PARAMETER" in xds.data_vars
        assert set(xds.data_vars) == {"PARAMETER", "FLAG"}
        assert list(xds.parameter_label.values) == ["PHASE", "DELAY", "RATE", "DISP_DELAY"]

    def test_parameter_axis_is_last(self):
        xds = make_gain_xds("fringe_fit")
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
        xds = make_gain_xds("fringe_fit")
        assert "units" not in xds.PARAMETER.attrs
        assert list(xds.parameter_units.values) == ["deg", "s", "s/s", "s"]

    def test_units_travel_with_a_selection(self):
        xds = make_gain_xds("fringe_fit")
        assert xds.sel(parameter_label="DELAY").parameter_units.item() == "s"

    # DISP_DELAY is unpolarised, so consolidating forces it to repeat across
    # the receptor axis. That redundancy is the documented cost of this layout.
    def test_unpolarised_quantity_is_broadcast_across_receptors(self):
        xds = make_gain_xds("fringe_fit", receptor_labels=("X", "Y"))
        disp = xds.PARAMETER.sel(parameter_label="DISP_DELAY")
        np.testing.assert_array_equal(
            disp.sel(receptor_label="X").values,
            disp.sel(receptor_label="Y").values,
        )

    def test_polarised_quantities_are_not_broadcast(self):
        xds = make_gain_xds("fringe_fit", n_time=4, n_antenna=8)
        phase = xds.PARAMETER.sel(parameter_label="PHASE")
        assert not np.array_equal(
            phase.sel(receptor_label="X").values,
            phase.sel(receptor_label="Y").values,
        )

    def test_one_flag_covers_the_whole_solve(self):
        xds = make_gain_xds("fringe_fit")
        assert xds.FLAG.dims == ("time", "antenna_name", "frequency")


class TestDelayConsolidation:
    """Consolidation without the broadcasting cost fringe fit pays."""

    def test_both_quantities_share_one_array(self):
        xds = make_gain_xds("delay")
        assert set(xds.data_vars) == {"PARAMETER", "FLAG"}
        assert list(xds.parameter_label.values) == ["PHASE", "DELAY"]

    # The offset and the slope are in different units, so a scalar attribute
    # cannot describe both.
    def test_units_move_to_a_coordinate(self):
        xds = make_gain_xds("delay")
        assert "units" not in xds.PARAMETER.attrs
        assert list(xds.parameter_units.values) == ["deg", "s"]

    # Both quantities are polarised, so unlike fringe fit nothing is padded out
    # over an axis it does not need.
    def test_neither_quantity_is_broadcast_across_receptors(self):
        xds = make_gain_xds("delay", receptor_labels=("X", "Y"))
        for label in ("PHASE", "DELAY"):
            values = xds.PARAMETER.sel(parameter_label=label)
            assert not np.array_equal(
                values.sel(receptor_label="X").values,
                values.sel(receptor_label="Y").values,
            )

    # The delay is a per-band solution, not a per-channel one, so the frequency
    # axis is present and length one rather than absent.
    def test_frequency_axis_is_present_and_single_channel(self):
        xds = make_gain_xds("delay")
        assert xds.sizes["frequency"] == 1
        assert "frequency" in xds.coords

    def test_frequency_extent_is_overridable(self):
        assert make_gain_xds("delay", n_frequency=8).sizes["frequency"] == 8

    def test_one_flag_covers_the_whole_solve(self):
        xds = make_gain_xds("delay")
        assert xds.FLAG.dims == ("time", "antenna_name", "frequency")


# Restricted to the types whose parameters actually share a unit, so every
# parametrized case makes a real assertion. delay and fringe_fit are the types
# this excludes, and both are covered by the classes above.
UNIFORM_UNIT_KEYS = [key for key in list_cal_types() if get_spec(key).uniform_units is not None]


@pytest.mark.parametrize("key", UNIFORM_UNIT_KEYS)
def test_uniform_unit_entries_have_no_parameter_units_coord(key):
    xds = make_gain_xds(key)
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


# Consolidating several parameters needs an axis to distinguish them. Without
# one there is nowhere to put them, and silently keeping only the last would
# discard data.
def test_multi_parameter_spec_without_parameter_axis_cannot_consolidate():
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


# A single parameter with no parameter axis is the normal case for G, B, D, T
# and opacity, and must keep working.
def test_single_parameter_spec_without_parameter_axis_still_builds():
    xds = make_gain_xds("antenna_gain")
    assert "parameter_label" not in xds.dims
    assert xds.GAIN.shape == (4, 8, 1, 2)
