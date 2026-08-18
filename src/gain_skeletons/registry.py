"""The calibration type catalogue.

Eleven representative calibration types, split into direction-independent
entries and direction-dependent ones. The catalogue is illustrative rather than
exhaustive: it exists to show the range of coordinate shapes such datasets take,
and hand-written CalSpec objects work everywhere a registered one does.

Axis presence and axis extent are recorded separately. A frequency axis of
length one is a frequency axis, recorded in default_sizes; a type with no
frequency dependence at all has no frequency axis, recorded by its absence from
the parameter's axes.

Sizes here are defaults only, and every builder accepts overrides. They are kept
small so that notebook output stays readable.
"""

from gain_skeletons.spec import CalSpec, ParamSpec

# Global size defaults, small enough to read in a notebook repr.
N_TIME = 4
N_ANTENNA = 8
N_DIRECTION = 3

# A deliberately single-channel axis, versus a channel-resolved one: one
# solution for the whole band, or one per channel.
N_CHANNEL_ONE = 1
N_CHANNEL_MANY = 64

# Axis groupings that recur across entries.
_TIME_ANT = ("time", "antenna_name")
_GAIN_AXES = (*_TIME_ANT, "frequency", "receptor_label")
_UNPOL_AXES = (*_TIME_ANT, "frequency")

# The two columns of the Jones matrix, named for the receptor each column maps
# from, are what a phenomenological term's two parameters per receptor are. The
# naming is provisional; nothing outside this constant depends on the spelling.
_JONES_LABELS = ("gain_X", "gain_Y")


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


# Direction-independent calibration types.
_PHENOMENOLOGICAL_GAIN = _gain_spec(
    "phenomenological_gain",
    jones_structure="full",
    n_frequency=N_CHANNEL_MANY,
    axes=(*_GAIN_AXES, "parameter_label"),
    labels=_JONES_LABELS,
    description=(
        "General Jones term, describing the response without attributing it to a "
        "physical cause. Two complex gains per receptor, one per column of the "
        "Jones matrix, so a parameter axis is required. Its frequency extent is "
        "arbitrary, so it defaults to channel-resolved."
    ),
)

_ANTENNA_GAIN = _gain_spec(
    "antenna_gain",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_ONE,
    description="Standard electronic gain, on-diagonal only, one solution per band.",
)

_TROPOSPHERIC_GAIN = _gain_spec(
    "tropospheric_gain",
    jones_structure="scalar",
    n_frequency=N_CHANNEL_ONE,
    axes=_UNPOL_AXES,
    description=(
        "Standard tropospheric gain. Scalar and unpolarised, so it carries no receptor axis."
    ),
)

_BANDPASS = _gain_spec(
    "bandpass",
    jones_structure="diagonal",
    n_frequency=N_CHANNEL_MANY,
    description="Standard bandpass, on-diagonal only, resolved in frequency.",
)

_LEAKAGE = _gain_spec(
    "leakage",
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

# Delay: a linear phase ramp across frequency, stored as the parameters of that
# ramp rather than sampled channel by channel. Two differently-united quantities
# from one solve, so two data arrays.
_DELAY = CalSpec(
    name="delay",
    parameters=(
        ParamSpec("PHASE", "deg", _GAIN_AXES, "float64", scale=30.0),
        ParamSpec("DELAY", "s", _GAIN_AXES, "float64", scale=1.0e-9),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": N_CHANNEL_ONE},
    description=(
        "Delay. A phase offset and a slope in seconds per receptor, which together "
        "parameterise a phase ramp across frequency. The frequency axis is present "
        "but single-channel: one ramp is solved per band, not per channel."
    ),
)

_ANTENNA_POSITIONS = CalSpec(
    name="antenna_positions",
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

# Fringe fit: the only entry that mixes polarised and unpolarised quantities.
# DISP_DELAY carries no receptor axis while the other three do, so it is the one
# entry whose arrays do not all share a shape.
_FRINGE_FIT = CalSpec(
    name="fringe_fit",
    parameters=(
        ParamSpec("PHASE", "deg", _GAIN_AXES, "float64", scale=30.0),
        ParamSpec("DELAY", "s", _GAIN_AXES, "float64", scale=1.0e-9),
        ParamSpec("RATE", "s/s", _GAIN_AXES, "float64", scale=1.0e-12),
        ParamSpec("DISP_DELAY", "s", _UNPOL_AXES, "float64", scale=1.0e-9),
    ),
    default_sizes={"time": N_TIME, "antenna_name": N_ANTENNA, "frequency": N_CHANNEL_ONE},
    description=(
        "Fringe fit. Four quantities with differing units, produced by a single "
        "solve. DISP_DELAY is unpolarised while the others are not, so its array "
        "carries one axis fewer than its siblings'."
    ),
)

# Direction-dependent calibration types.
_DD_PHENOMENOLOGICAL_GAIN = _gain_spec(
    "dd_phenomenological_gain",
    jones_structure="full",
    n_frequency=N_CHANNEL_ONE,
    axes=("direction", *_GAIN_AXES, "parameter_label"),
    labels=_JONES_LABELS,
    n_direction=N_DIRECTION,
    description=(
        "Direction-dependent general Jones term. The same two gains per receptor "
        "as phenomenological_gain, with a leading direction axis indexing facets "
        "within a single field of view. Single-channel by default, since directions "
        "multiply the array size."
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
        _PHENOMENOLOGICAL_GAIN,
        _ANTENNA_GAIN,
        _TROPOSPHERIC_GAIN,
        _OPACITY,
        _BANDPASS,
        _LEAKAGE,
        _DELAY,
        _ANTENNA_POSITIONS,
        _FRINGE_FIT,
        _DD_PHENOMENOLOGICAL_GAIN,
        _IONOSPHERE,
    )
}


def list_cal_types() -> tuple[str, ...]:
    """List the registered calibration type names.

    Returns:
        The registry keys, direction-independent types first.
    """
    return tuple(REGISTRY)


def get_spec(name: str) -> CalSpec:
    """Look up a calibration type by name.

    Args:
        name: Registry key, such as "bandpass" or "fringe_fit".

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
