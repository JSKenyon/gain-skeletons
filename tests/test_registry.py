"""Tests that the registry matches the source deck.

The expected values below are transcribed independently from slides 6 and 7 of
docs/reference/calibration-dataset-coordinate-dimensions.pdf. The duplication
against registry.py is deliberate: it checks the registry against the source
material rather than against itself. Do not factor these tables together.

Deck notation: "nFreq=1" means the frequency axis exists with length one, while
"{nFreq=0}" means it does not exist at all.
"""

import pytest

from gain_skeletons.registry import REGISTRY, get_spec, list_cal_types

TIME_ANT = ("time", "antenna_name")

# (key, axes, parameter name, units, dtype, labels, jones_structure)
SLIDE_6_DIRECTION_INDEPENDENT = [
    # General "J": [nTime, nAnt, nFreq, nPol=2, nPar=2] (Complex) GAIN in [rel]
    (
        "J",
        (*TIME_ANT, "frequency", "receptor_label", "parameter_label"),
        "GAIN",
        "rel",
        "complex64",
        ("aligned", "cross"),
        "full",
    ),
    # Standard "G": [nTime, nAnt, nFreq=1, nPol=2] (Complex) GAIN in [rel] (on-diag
    # only)
    (
        "G",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Standard "T": [nTime, nAnt, nFreq=1, {nPol=0}] (Complex) GAIN in [rel]
    # (scalar, unpol!)
    (
        "T",
        (*TIME_ANT, "frequency"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "scalar",
    ),
    # Opacity: [nTime, nAnt, nFreq=1, {nPol=0}] (Float) OPAC in [nepers] (unpol!)
    (
        "opacity",
        (*TIME_ANT, "frequency"),
        "OPAC",
        "nepers",
        "float64",
        ("OPAC",),
        None,
    ),
    # Standard "B": [nTime, nAnt, nFreq=nCh, nPol=2] (Complex) GAIN in [rel]
    # (on-diag only)
    (
        "B",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Standard "D": [nTime, nAnt, nFreq=nCh, nPol=2] (Complex) GAIN in [rel]
    # (off-diag only)
    (
        "D",
        (*TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "off-diagonal",
    ),
    # Antpos: [nTime, nAnt, {nFreq=0}, {nPol=0}, nPar=3] (Float) (dX,dY,dZ) in [m]
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
SLIDE_7_DIRECTION_DEPENDENT = [
    # Generic gain: [nDir, nTime, nAnt, nFreq=1, nPol=2] (Complex) GAIN in [rel]
    # (on-diag only)
    (
        "dd_gain",
        ("direction", *TIME_ANT, "frequency", "receptor_label"),
        "GAIN",
        "rel",
        "complex64",
        ("GAIN",),
        "diagonal",
    ),
    # Ionosphere: [nDir, nTime, nAnt, {nFreq=0}, {nPol=0}] (Float) TEC in [TECU]
    ("ionosphere", ("direction", *TIME_ANT), "TEC", "TECU", "float64", ("TEC",), None),
]

SINGLE_PARAMETER_CASES = SLIDE_6_DIRECTION_INDEPENDENT + SLIDE_7_DIRECTION_DEPENDENT

# Fringefit is the only multi-parameter entry. Slide 6 gives four lines:
#   [nTime, nAnt, nFreq=1,  nPol=2,   nPar=1] (Float) PHASE      in [deg]
#   [nTime, nAnt, nFreq=1,  nPol=2,   nPar=1] (Float) DELAY      in [s]
#   [nTime, nAnt, nFreq=1,  nPol=2,   nPar=1] (Float) RATE       in [s/s]
#   [nTime, nAnt, nFreq=1, {nPol=0},  nPar=1] (Float) DISP_DELAY in [s]
POLARISED = (*TIME_ANT, "frequency", "receptor_label", "parameter_label")
UNPOLARISED = (*TIME_ANT, "frequency", "parameter_label")
FRINGEFIT_PARAMETERS = [
    ("PHASE", "deg", POLARISED),
    ("DELAY", "s", POLARISED),
    ("RATE", "s/s", POLARISED),
    ("DISP_DELAY", "s", UNPOLARISED),
]


def test_registry_has_exactly_the_deck_entries():
    expected = {key for key, *_ in SINGLE_PARAMETER_CASES} | {"fringefit"}
    assert set(REGISTRY) == expected


def test_list_cal_types_matches_registry_keys():
    assert list_cal_types() == tuple(REGISTRY)


@pytest.mark.parametrize(
    ("key", "axes", "param_name", "units", "dtype", "labels", "jones_structure"),
    SINGLE_PARAMETER_CASES,
    ids=[case[0] for case in SINGLE_PARAMETER_CASES],
)
def test_single_parameter_entry_matches_deck(
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


def test_fringefit_parameters_match_deck():
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


# Slide 6 marks G, T, opacity and every fringefit quantity as single-channel,
# while B and D are channel-resolved. That distinction lives in default_sizes.
@pytest.mark.parametrize("key", ["G", "T", "opacity", "fringefit", "dd_gain"])
def test_single_channel_entries_default_to_one_channel(key):
    assert get_spec(key).default_sizes["frequency"] == 1


@pytest.mark.parametrize("key", ["B", "D", "J"])
def test_channel_resolved_entries_default_to_many_channels(key):
    assert get_spec(key).default_sizes["frequency"] == 64


# "{nFreq=0}" on slide 6 and 7 means the axis is genuinely absent, which is
# materially different from a length-one axis.
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
