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


def test_default_sizes_cannot_be_mutated_after_construction():
    sizes = {"time": 4, "antenna_name": 8, "frequency": 1}
    spec = CalSpec(name="G", parameters=(make_gain_param(),), default_sizes=sizes)
    with pytest.raises(TypeError):
        spec.default_sizes["frequency"] = 999
    assert spec.default_sizes["frequency"] == 1


# The snapshot must also be insulated from later edits to the caller's dict.
def test_default_sizes_snapshot_is_independent_of_caller_dict():
    sizes = {"time": 4, "antenna_name": 8, "frequency": 1}
    spec = CalSpec(name="G", parameters=(make_gain_param(),), default_sizes=sizes)
    sizes["frequency"] = 999
    assert spec.default_sizes["frequency"] == 1
