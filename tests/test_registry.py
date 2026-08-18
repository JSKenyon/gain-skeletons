"""Tests that the registry holds the intended calibration type catalogue.

The expected values below restate every catalogue entry independently of
registry.py. The duplication is deliberate: it pins each type's axes, units and
dtype as a statement of intent, so that changing the registry has to be a
deliberate act rather than something a test silently ratifies. Do not factor
these tables together, and do not generate either from the other.

Presence and extent are distinguished throughout. A frequency axis of length one
is present; a type with no frequency dependence has no frequency axis at all.
"""

import pytest

from gain_skeletons.registry import REGISTRY, get_spec, list_cal_types

TIME_ANT = ("time", "antenna_name")

# (key, axes, parameter name, units, dtype, labels, jones_structure)
DIRECTION_INDEPENDENT_CASES = [
    # General Jones term: complex, channel-resolved, polarised, and carrying two
    # gains per receptor on the parameter axis.
    (
        "phenomenological_gain",
        (*TIME_ANT, "frequency", "receptor_label", "parameter_label"),
        "GAIN",
        "rel",
        "complex64",
        ("gain_X", "gain_Y"),
        "full",
    ),
    # Standard electronic gain: complex, one solution per band, on-diagonal only.
    (
        "antenna_gain",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Tropospheric gain: complex and scalar, so unpolarised — no receptor axis.
    (
        "tropospheric_gain",
        (*TIME_ANT, "frequency"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "scalar",
    ),
    # Atmospheric opacity: real, in nepers, unpolarised.
    (
        "opacity",
        (*TIME_ANT, "frequency"),
        "OPAC",
        "nepers",
        "float64",
        ("OPAC",),
        None,
    ),
    # Bandpass: as antenna_gain but resolved per channel rather than one
    # solution per band.
    (
        "bandpass",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Polarisation leakage: channel-resolved, and off-diagonal rather than on.
    (
        "leakage",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "off-diagonal",
    ),
    # Antenna position offset: three same-unit components on the parameter axis,
    # with neither frequency nor polarisation dependence.
    (
        "antenna_positions",
        (*TIME_ANT, "parameter_label"),
        "ANTENNA_POSITION_OFFSET",
        "m",
        "float64",
        ("dX", "dY", "dZ"),
        None,
    ),
]

# (key, axes, parameter name, units, dtype, labels, jones_structure)
DIRECTION_DEPENDENT_CASES = [
    # Direction-dependent general Jones term: phenomenological_gain with a
    # leading direction axis, and single-channel by default rather than
    # channel-resolved.
    (
        "dd_phenomenological_gain",
        ("direction", *TIME_ANT, "frequency", "receptor_label", "parameter_label"),
        "GAIN",
        "rel",
        "complex64",
        ("gain_X", "gain_Y"),
        "full",
    ),
    # Ionospheric TEC: direction-dependent, but neither frequency- nor
    # polarisation-dependent, so both of those axes are absent.
    ("ionosphere", ("direction", *TIME_ANT), "TEC", "TECU", "float64", ("TEC",), None),
]

SINGLE_PARAMETER_CASES = DIRECTION_INDEPENDENT_CASES + DIRECTION_DEPENDENT_CASES

# Two entries hold several quantities. Delay holds the two parameters of a phase
# ramp, both polarised. Fringe fit holds four, of which DISP_DELAY alone is
# unpolarised — the catalogue's only mix of the two. Neither declares a parameter
# axis: each quantity is its own array, named for itself, so there is nothing for
# such an axis to distinguish.
POLARISED = (*TIME_ANT, "frequency", "receptor_label")
UNPOLARISED = (*TIME_ANT, "frequency")
DELAY_PARAMETERS = [
    ("PHASE", "deg", POLARISED),
    ("DELAY", "s", POLARISED),
]
FRINGE_FIT_PARAMETERS = [
    ("PHASE", "deg", POLARISED),
    ("DELAY", "s", POLARISED),
    ("RATE", "s/s", POLARISED),
    ("DISP_DELAY", "s", UNPOLARISED),
]
MULTI_PARAMETER_KEYS = {"delay", "fringe_fit"}


def test_registry_has_exactly_the_catalogue_entries():
    expected = {key for key, *_ in SINGLE_PARAMETER_CASES} | MULTI_PARAMETER_KEYS
    assert set(REGISTRY) == expected


def test_list_cal_types_matches_registry_keys():
    assert list_cal_types() == tuple(REGISTRY)


@pytest.mark.parametrize(
    ("key", "axes", "param_name", "units", "dtype", "labels", "jones_structure"),
    SINGLE_PARAMETER_CASES,
    ids=[case[0] for case in SINGLE_PARAMETER_CASES],
)
def test_single_parameter_entry_matches_catalogue(
    key, axes, param_name, units, dtype, labels, jones_structure
):
    spec = get_spec(key)
    assert len(spec.parameters) == 1
    param = spec.parameters[0]
    assert spec.axes == axes
    assert param.axes == axes
    assert param.name == param_name
    assert param.units == units
    assert param.dtype == dtype
    assert param.resolved_labels == labels
    assert spec.jones_structure == jones_structure


def test_delay_parameters_match_catalogue():
    spec = get_spec("delay")
    actual = [(param.name, param.units, param.axes) for param in spec.parameters]
    assert actual == DELAY_PARAMETERS


def test_fringe_fit_parameters_match_catalogue():
    spec = get_spec("fringe_fit")
    actual = [(param.name, param.units, param.axes) for param in spec.parameters]
    assert actual == FRINGE_FIT_PARAMETERS


def test_delay_and_fringe_fit_are_the_only_multi_parameter_entries():
    multi = {key for key, spec in REGISTRY.items() if len(spec.parameters) > 1}
    assert multi == MULTI_PARAMETER_KEYS


# Delay is multi-parameter but wholly polarised, so fringe fit is the only entry
# whose arrays do not all share a shape.
def test_fringe_fit_is_the_only_entry_mixing_polarised_and_unpolarised():
    mixed = {
        key
        for key, spec in REGISTRY.items()
        if len({"receptor_label" in param.axes for param in spec.parameters}) > 1
    }
    assert mixed == {"fringe_fit"}


# The parameter axis distinguishes components within one quantity, never one
# quantity from another. The multi-parameter entries therefore have no such axis.
@pytest.mark.parametrize("key", sorted(MULTI_PARAMETER_KEYS))
def test_multi_parameter_entries_declare_no_parameter_axis(key):
    assert get_spec(key).parameter_labels is None
    assert "parameter_label" not in get_spec(key).axes


# The entries that do carry the axis carry it for components of a single
# quantity, which is why they have exactly one parameter each.
@pytest.mark.parametrize(
    ("key", "labels"),
    [
        ("phenomenological_gain", ("gain_X", "gain_Y")),
        ("antenna_positions", ("dX", "dY", "dZ")),
        ("dd_phenomenological_gain", ("gain_X", "gain_Y")),
    ],
)
def test_parameter_axis_entries_are_single_quantity_components(key, labels):
    spec = get_spec(key)
    assert len(spec.parameters) == 1
    assert spec.parameter_labels == labels


@pytest.mark.parametrize("key", sorted(MULTI_PARAMETER_KEYS))
def test_multi_parameter_entries_have_heterogeneous_units(key):
    units = {param.units for param in get_spec(key).parameters}
    assert len(units) > 1


# Several entries solve once per band, while bandpass, leakage and the general
# Jones term resolve every channel. That distinction lives in default_sizes.
@pytest.mark.parametrize(
    "key",
    [
        "antenna_gain",
        "tropospheric_gain",
        "opacity",
        "delay",
        "fringe_fit",
        "dd_phenomenological_gain",
    ],
)
def test_single_channel_entries_default_to_one_channel(key):
    assert get_spec(key).default_sizes["frequency"] == 1


@pytest.mark.parametrize("key", ["bandpass", "leakage", "phenomenological_gain"])
def test_channel_resolved_entries_default_to_many_channels(key):
    assert get_spec(key).default_sizes["frequency"] == 64


# These types have no frequency dependence at all, so the axis is genuinely
# absent — materially different from a length-one axis.
@pytest.mark.parametrize("key", ["antenna_positions", "ionosphere"])
def test_frequency_independent_entries_have_no_frequency_axis(key):
    assert "frequency" not in get_spec(key).axes


@pytest.mark.parametrize(
    "key",
    ["tropospheric_gain", "opacity", "antenna_positions", "ionosphere"],
)
def test_unpolarised_entries_have_no_receptor_axis(key):
    assert "receptor_label" not in get_spec(key).axes


@pytest.mark.parametrize("key", ["dd_phenomenological_gain", "ionosphere"])
def test_direction_dependent_entries_are_flagged(key):
    assert get_spec(key).direction_dependent is True


@pytest.mark.parametrize(
    "key",
    [
        "phenomenological_gain",
        "antenna_gain",
        "tropospheric_gain",
        "opacity",
        "bandpass",
        "leakage",
        "delay",
        "antenna_positions",
        "fringe_fit",
    ],
)
def test_direction_independent_entries_are_flagged(key):
    assert get_spec(key).direction_dependent is False


def test_get_spec_rejects_unknown_name_and_lists_alternatives():
    with pytest.raises(KeyError, match="ionosphere"):
        get_spec("not_a_cal_type")
