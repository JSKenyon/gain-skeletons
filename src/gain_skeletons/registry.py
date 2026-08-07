"""The calibration type catalogue, transcribed from the source deck.

Every entry corresponds to a line on slide 6 (direction-independent) or slide 7
(direction-dependent) of George Moellenbrock's "Calibration Dataset Coordinate
Dimensions" (2026-07-30). The deck itself is not committed to this repository.

The deck's brace notation is represented by axis presence: "nFreq=1" is a
frequency axis of length one, recorded in default_sizes, while "{nFreq=0}" is no
frequency axis at all, recorded by its absence from the parameter's axes.

Sizes here are defaults only, and every builder accepts overrides. They are kept
small so that notebook output stays readable.
"""

from gain_skeletons.spec import CalSpec, ParamSpec

# Global size defaults, small enough to read in a notebook repr.
N_TIME = 4
N_ANTENNA = 8
N_DIRECTION = 3

# A deliberately single-channel axis, versus a channel-resolved one. Slide 6
# writes these "nFreq=1" and "nFreq=nCh" respectively.
N_CHANNEL_ONE = 1
N_CHANNEL_MANY = 64

# Axis groupings that recur across entries.
_TIME_ANT = ("time", "antenna_name")
_GAIN_AXES = (*_TIME_ANT, "frequency", "receptor_label")
_UNPOL_AXES = (*_TIME_ANT, "frequency")

# Slide 4 names the two columns of the Jones matrix the "aligned" and "cross"
# gains, which is what the general J term's two parameters per receptor are.
_JONES_LABELS = ("aligned", "cross")


def _gain_spec(
    name: str,
    *,
    jones_structure: str,
    n_frequency: int,
    axes: tuple[str, ...] = _GAIN_AXES,
    labels: tuple[str, ...] | None = None,
    description: str = "",
    n_direction: int | None = None,
) -> CalSpec:
    """Build a CalSpec for a complex gain calibration type.

    Args:
        name: Registry key.
        jones_structure: Which part of the Jones matrix the type populates.
        n_frequency: Default extent of the frequency axis.
        axes: Axes of the single GAIN parameter, in canonical order.
        labels: Parameter labels, or None for a single label named GAIN.
        description: Human-readable summary.
        n_direction: Default extent of the direction axis, if the type has one.

    Returns:
        The calibration type specification.
    """
    default_sizes = {"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": n_frequency}
    if n_direction is not None:
        default_sizes["direction"] = n_direction
    return CalSpec(
        name=name,
        parameters=(
            ParamSpec(
                name="GAIN",
                units="rel",
                axes=axes,
                dtype="complex64",
                labels=labels,
            ),
        ),
        default_sizes=default_sizes,
        jones_structure=jones_structure,
        description=description,
    )


# Slide 6: direction-independent calibration types.
_J = _gain_spec(
    "J",
    jones_structure="full",
    n_frequency=N_CHANNEL_MANY,
    axes=(*_GAIN_AXES, "parameter_label"),
    labels=_JONES_LABELS,
    description=(
        "General Jones term. Two complex gains per receptor, the aligned and cross "
        "responses, so a parameter axis is required. The deck leaves its frequency "
        "extent unspecified, so it defaults to channel-resolved."
    ),
)

_G = _gain_spec(
    "G",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_ONE,
    description="Standard electronic gain, on-diagonal only, one solution per band.",
)

_T = _gain_spec(
    "T",
    jones_structure="scalar",
    n_frequency=N_CHANNEL_ONE,
    axes=_UNPOL_AXES,
    description=(
        "Standard tropospheric gain. Scalar and unpolarised, so it carries no receptor axis."
    ),
)

_B = _gain_spec(
    "B",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_MANY,
    description="Standard bandpass, on-diagonal only, resolved in frequency.",
)

_D = _gain_spec(
    "D",
    jones_structure="off-diagonal",
    n_frequency=N_CHANNEL_MANY,
    description="Standard polarisation leakage, off-diagonal only, resolved in frequency.",
)

_OPACITY = CalSpec(
    name="opacity",
    parameters=(
        ParamSpec(
            name="OPAC",
            units="nepers",
            axes=_UNPOL_AXES,
            dtype="float64",
            scale=0.05,
        ),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": N_CHANNEL_ONE},
    description="Atmospheric opacity. Unpolarised, so it carries no receptor axis.",
)

_ANTPOS = CalSpec(
    name="antpos",
    parameters=(
        ParamSpec(
            name="ANTENNA_POSITION_OFFSET",
            units="m",
            axes=(*_TIME_ANT, "parameter_label"),
            dtype="float64",
            labels=("dX", "dY", "dZ"),
            scale=0.01,
        ),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA},
    description=(
        "Antenna position correction. Three same-unit components on the parameter "
        "axis, and neither frequency- nor polarisation-dependent, so both of those "
        "axes are absent."
    ),
)

# Slide 6, Fringefit: the only entry with several quantities. Their units
# differ, and DISP_DELAY is unpolarised while the other three are not, so the
# two layouts genuinely diverge here.
_FRINGEFIT = CalSpec(
    name="fringefit",
    parameters=(
        ParamSpec("PHASE", "deg", (*_GAIN_AXES, "parameter_label"), "float64", scale=30.0),
        ParamSpec("DELAY", "s", (*_GAIN_AXES, "parameter_label"), "float64", scale=1.0e-9),
        ParamSpec("RATE", "s/s", (*_GAIN_AXES, "parameter_label"), "float64", scale=1.0e-12),
        ParamSpec(
            "DISP_DELAY",
            "s",
            (*_UNPOL_AXES, "parameter_label"),
            "float64",
            scale=1.0e-9,
        ),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": N_CHANNEL_ONE},
    consolidated_name="PARAMETER",
    description=(
        "Fringe fit. Four quantities with differing units, produced by a single "
        "solve. DISP_DELAY is unpolarised while the others are not, so the "
        "consolidated layout must broadcast it over the receptor axis."
    ),
)

# Slide 7: direction-dependent calibration types.
_DD_GAIN = _gain_spec(
    "dd_gain",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_ONE,
    axes=("direction", *_GAIN_AXES),
    n_direction=N_DIRECTION,
    description=(
        "Generic direction-dependent gain, on-diagonal only. The direction axis "
        "indexes facets within a single field of view."
    ),
)

_IONOSPHERE = CalSpec(
    name="ionosphere",
    parameters=(
        ParamSpec(
            name="TEC",
            units="TECU",
            axes=("direction", *_TIME_ANT),
            dtype="float64",
            scale=5.0,
        ),
    ),
    default_sizes={"direction": N_DIRECTION, "time": N_TIME, "antenna_name": N_ANTENNA},
    description=(
        "Ionospheric total electron content. Direction-dependent, but neither "
        "frequency- nor polarisation-dependent, so both of those axes are absent."
    ),
)

REGISTRY: dict[str, CalSpec] = {
    spec.name: spec
    for spec in (
        _J,
        _G,
        _T,
        _OPACITY,
        _B,
        _D,
        _ANTPOS,
        _FRINGEFIT,
        _DD_GAIN,
        _IONOSPHERE,
    )
}


def list_cal_types() -> tuple[str, ...]:
    """List the registered calibration type names.

    Returns:
        The registry keys, in the order the deck presents them.
    """
    return tuple(REGISTRY)


def get_spec(name: str) -> CalSpec:
    """Look up a calibration type by name.

    Args:
        name: Registry key, such as "B" or "fringefit".

    Returns:
        The calibration type specification.

    Raises:
        KeyError: If the name is not registered. The message lists the
            available names.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown calibration type {name!r}; available types are {list(REGISTRY)}"
        ) from None
