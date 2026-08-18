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
        name="fringe_fit",
        parameters=(
            ParamSpec("PHASE", "deg", ("time", "receptor_label"), "float64"),
            ParamSpec("DISP_DELAY", "s", ("time",), "float64"),
        ),
        default_sizes={"time": 4},
    )
    assert spec.axes == ("time", "receptor_label")


def test_cal_spec_parameter_labels_is_none_without_a_parameter_axis():
    spec = CalSpec(
        name="fringe_fit",
        parameters=(
            ParamSpec("PHASE", "deg", ("time",), "float64"),
            ParamSpec("DELAY", "s", ("time",), "float64"),
        ),
        default_sizes={"time": 4},
    )
    assert spec.parameter_labels is None


def test_cal_spec_parameter_labels_reports_the_shared_axis():
    spec = CalSpec(
        name="antenna_positions",
        parameters=(
            ParamSpec(
                "ANTENNA_POSITION_OFFSET",
                "m",
                ("time", "parameter_label"),
                "float64",
                labels=("dX", "dY", "dZ"),
            ),
        ),
        default_sizes={"time": 4},
    )
    assert spec.parameter_labels == ("dX", "dY", "dZ")


# default_sizes must name every sized axis the type uses, so the helper below
# supplies a complete set for make_gain_param's default axes.
GAIN_SIZES = {"time": 4, "antenna_name": 8, "frequency": 1}


def test_cal_spec_direction_dependent_follows_the_direction_axis():
    di = CalSpec(name="antenna_gain", parameters=(make_gain_param(),), default_sizes=GAIN_SIZES)
    dd = CalSpec(
        name="dd_phenomenological_gain",
        parameters=(make_gain_param(axes=("direction", "time", "antenna_name")),),
        default_sizes={"direction": 3, "time": 4, "antenna_name": 8},
    )
    assert di.direction_dependent is False
    assert dd.direction_dependent is True


# One dataset holds one parameter_label coordinate, so parameters declaring
# that axis must agree on its labels. Two parameters wanting different labels
# describe an axis that cannot exist, and that is a spec error rather than
# something for a builder to reconcile.
def test_cal_spec_rejects_parameters_that_disagree_about_parameter_labels():
    with pytest.raises(ValueError, match="disagree"):
        CalSpec(
            name="conflicting_labels",
            parameters=(
                ParamSpec(
                    "A", "m", ("time", "parameter_label"), "float64", labels=("dX", "dY", "dZ")
                ),
                ParamSpec("B", "m", ("time", "parameter_label"), "float64", labels=("dAZ", "dEL")),
            ),
            default_sizes={"time": 4},
        )


# Agreement, not uniqueness: sharing the axis means sharing its labels.
def test_cal_spec_accepts_parameters_sharing_identical_labels():
    spec = CalSpec(
        name="agreeing_labels",
        parameters=(
            ParamSpec("A", "m", ("time", "parameter_label"), "float64", labels=("dX", "dY")),
            ParamSpec("B", "s", ("time", "parameter_label"), "float64", labels=("dX", "dY")),
        ),
        default_sizes={"time": 4},
    )
    assert spec.parameter_labels == ("dX", "dY")


# A parameter that does not declare the axis is unconstrained by the labels of
# one that does.
def test_cal_spec_ignores_parameters_without_the_axis_when_checking_labels():
    spec = CalSpec(
        name="partial",
        parameters=(
            ParamSpec("A", "m", ("time", "parameter_label"), "float64", labels=("dX", "dY")),
            ParamSpec("B", "s", ("time",), "float64"),
        ),
        default_sizes={"time": 4},
    )
    assert spec.parameter_labels == ("dX", "dY")


def test_cal_spec_rejects_no_parameters():
    with pytest.raises(ValueError, match="at least one parameter"):
        CalSpec(name="empty", parameters=(), default_sizes={})


# default_sizes exists to give an axis that is present an extent — one channel
# or many. Naming an axis the calibration type does not have is a spec error.
def test_cal_spec_rejects_default_size_for_absent_axis():
    with pytest.raises(ValueError, match="not an axis of"):
        CalSpec(
            name="antenna_gain",
            parameters=(make_gain_param(axes=("time", "antenna_name")),),
            default_sizes={"time": 4, "frequency": 64},
        )


def test_cal_spec_rejects_default_size_for_label_axis():
    with pytest.raises(ValueError, match="never configurable"):
        CalSpec(
            name="antenna_gain",
            parameters=(make_gain_param(),),
            default_sizes={"time": 4, "receptor_label": 2},
        )


def test_cal_spec_requires_default_size_for_every_sized_axis():
    with pytest.raises(ValueError, match="missing default size"):
        CalSpec(
            name="antenna_gain",
            parameters=(make_gain_param(axes=("time", "antenna_name")),),
            default_sizes={"time": 4},
        )


def test_default_sizes_cannot_be_mutated_after_construction():
    sizes = {"time": 4, "antenna_name": 8, "frequency": 1}
    spec = CalSpec(name="antenna_gain", parameters=(make_gain_param(),), default_sizes=sizes)
    with pytest.raises(TypeError):
        spec.default_sizes["frequency"] = 999
    assert spec.default_sizes["frequency"] == 1


# The snapshot must also be insulated from later edits to the caller's dict.
def test_default_sizes_snapshot_is_independent_of_caller_dict():
    sizes = {"time": 4, "antenna_name": 8, "frequency": 1}
    spec = CalSpec(name="antenna_gain", parameters=(make_gain_param(),), default_sizes=sizes)
    sizes["frequency"] = 999
    assert spec.default_sizes["frequency"] == 1
