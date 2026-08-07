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
        "J",
        (*TIME_ANT, "frequency", "receptor_label", "parameter_label"),
        "GAIN",
        "rel",
        "complex64",
        ("aligned", "cross"),
        "full",
    ),
    # Standard electronic gain: complex, one solution per band, on-diagonal only.
    (
        "G",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Tropospheric gain: complex and scalar, so unpolarised — no receptor axis.
    (
        "T",
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
    # Bandpass: as G but resolved per channel rather than one solution per band.
    (
        "B",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Polarisation leakage: channel-resolved, and off-diagonal rather than on.
    (
        "D",
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
        "antpos",
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
    # Generic direction-dependent gain: G with a leading direction axis.
    (
        "dd_gain",
        ("direction", *TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Ionospheric TEC: direction-dependent, but neither frequency- nor
    # polarisation-dependent, so both of those axes are absent.
    ("ionosphere", ("direction", *TIME_ANT), "TEC", "TECU", "float64", ("TEC",), None),
]

SINGLE_PARAMETER_CASES = DIRECTION_INDEPENDENT_CASES + DIRECTION_DEPENDENT_CASES

# Fringefit is the only multi-parameter entry: four real quantities with
# differing units, of which DISP_DELAY alone is unpolarised.
POLARISED = (*TIME_ANT, "frequency", "receptor_label", "parameter_label")
UNPOLARISED = (*TIME_ANT, "frequency", "parameter_label")
FRINGEFIT_PARAMETERS = [
    ("PHASE", "deg", POLARISED),
    ("DELAY", "s", POLARISED),
    ("RATE", "s/s", POLARISED),
    ("DISP_DELAY", "s", UNPOLARISED),
]


def test_registry_has_exactly_the_catalogue_entries():
    expected = {key for key, *_ in SINGLE_PARAMETER_CASES} | {"fringefit"}
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


def test_fringefit_parameters_match_catalogue():
    spec = get_spec("fringefit")
    actual = [(param.name, param.units, param.axes) for param in spec.parameters]
    assert actual == FRINGEFIT_PARAMETERS


def test_fringefit_is_the_only_multi_parameter_entry():
    multi = {key for key, spec in REGISTRY.items() if len(spec.parameters) > 1}
    assert multi == {"fringefit"}


def test_fringefit_consolidated_labels_are_the_quantity_names():
    assert get_spec("fringefit").all_labels == ("PHASE", "DELAY", "RATE", "DISP_DELAY")


def test_fringefit_units_are_heterogeneous():
    assert get_spec("fringefit").uniform_units is None


# G, T, opacity and every fringefit quantity are single-channel, while B and D
# are channel-resolved. That distinction lives in default_sizes.
@pytest.mark.parametrize("key", ["G", "T", "opacity", "fringefit", "dd_gain"])
def test_single_channel_entries_default_to_one_channel(key):
    assert get_spec(key).default_sizes["frequency"] == 1


@pytest.mark.parametrize("key", ["B", "D", "J"])
def test_channel_resolved_entries_default_to_many_channels(key):
    assert get_spec(key).default_sizes["frequency"] == 64


# These types have no frequency dependence at all, so the axis is genuinely
# absent — materially different from a length-one axis.
@pytest.mark.parametrize("key", ["antpos", "ionosphere"])
def test_frequency_independent_entries_have_no_frequency_axis(key):
    assert "frequency" not in get_spec(key).axes


@pytest.mark.parametrize("key", ["T", "opacity", "antpos", "ionosphere"])
def test_unpolarised_entries_have_no_receptor_axis(key):
    assert "receptor_label" not in get_spec(key).axes


@pytest.mark.parametrize("key", ["dd_gain", "ionosphere"])
def test_direction_dependent_entries_are_flagged(key):
    assert get_spec(key).direction_dependent is True


@pytest.mark.parametrize("key", ["J", "G", "T", "opacity", "B", "D", "antpos", "fringefit"])
def test_direction_independent_entries_are_flagged(key):
    assert get_spec(key).direction_dependent is False


def test_get_spec_rejects_unknown_name_and_lists_alternatives():
    with pytest.raises(KeyError, match="ionosphere"):
        get_spec("not_a_cal_type")
