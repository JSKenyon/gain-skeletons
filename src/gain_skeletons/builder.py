"""Builder that turns a calibration type specification into a dataset.

:func:`make_gain_xds` produces one dataset per solve: one data array per
parameter, named for it, plus one flag describing the solve. Each array carries
exactly the axes its parameter declares and a scalar ``units`` attribute, so
nothing is broadcast and no quantity is padded out over an axis it does not
need.

All values are random. Complex parameters are generated near unit amplitude,
since a uniform-random complex gain of arbitrary magnitude would be physically
nonsensical.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from gain_skeletons.axes import (
    DEFAULT_RECEPTOR_LABELS,
    SIZED_AXIS_FACTORIES,
    parameter_label_coord,
    receptor_label_coord,
)
from gain_skeletons.registry import get_spec
from gain_skeletons.spec import CalSpec, ParamSpec

DEFAULT_FLAG_FRACTION = 0.05

# Fractional spread of complex gain amplitude about unity.
DEFAULT_AMPLITUDE_JITTER = 0.1

FLAG_NAME = "FLAG"

# Only these axes have factories taking extra keyword arguments. direction_coord
# takes none, and the label axes get their values from labels rather than from a
# generated range.
COORD_KWARGS_AXES = ("time", "antenna_name", "frequency")

# Maps the builders' keyword arguments to the axes they size. Keeping the
# public names short (n_antenna, not n_antenna_name) at the cost of this table.
_SIZE_KEYWORDS = {
    "n_direction": "direction",
    "n_time": "time",
    "n_antenna": "antenna_name",
    "n_frequency": "frequency",
}


def _resolve_spec(spec: CalSpec | str) -> CalSpec:
    """Accept either a specification or a registry name.

    Args:
        spec: A CalSpec, or the name of a registered calibration type.

    Returns:
        The calibration type specification.

    Raises:
        KeyError: If a name is given that is not registered.
    """
    return get_spec(spec) if isinstance(spec, str) else spec


def _resolve_sizes(spec: CalSpec, overrides: Mapping[str, int | None]) -> dict[str, int]:
    """Combine a specification's default sizes with caller overrides.

    Args:
        spec: The calibration type.
        overrides: Keyword name to requested size, where None means "use the
            default". Keys are the names in _SIZE_KEYWORDS.

    Returns:
        Axis name to extent, for every sized axis the calibration type has.

    Raises:
        ValueError: If a size is given for an axis the calibration type does
            not have, or if a size is not positive.
    """
    sizes = {axis: size for axis, size in spec.default_sizes.items()}
    for keyword, size in overrides.items():
        if size is None:
            continue
        axis = _SIZE_KEYWORDS[keyword]
        if axis not in spec.axes:
            raise ValueError(
                f"calibration type {spec.name!r} has no {axis!r} axis, so {keyword} cannot be set"
            )
        if size < 1:
            raise ValueError(f"{keyword} must be positive, got {size}")
        sizes[axis] = size
    return sizes


def _build_coords(
    spec: CalSpec,
    axes: Sequence[str],
    sizes: Mapping[str, int],
    labels: Sequence[str],
    receptor_labels: Sequence[str],
    coord_kwargs: Mapping[str, Mapping[str, Any]],
) -> dict[str, xr.DataArray]:
    """Build every coordinate for a dataset.

    Args:
        spec: The calibration type, used for error messages.
        axes: Axes the dataset has, in canonical order.
        sizes: Extent of each sized axis.
        labels: Parameter labels, used when axes includes parameter_label.
        receptor_labels: Receptor labels, used when axes includes receptor_label.
        coord_kwargs: Axis name to extra keyword arguments for that axis's
            factory, such as frequency start and end. Only accepted for
            ``time``, ``antenna_name``, and ``frequency`` (see
            COORD_KWARGS_AXES); direction_coord takes no keywords at all, and
            the label axes (receptor_label, parameter_label) get their values
            from labels rather than from a generated range.

    Returns:
        Axis name to coordinate.

    Raises:
        ValueError: If coord_kwargs names an axis the calibration type lacks,
            or names an axis that takes no coordinate configuration.
    """
    unknown = [axis for axis in coord_kwargs if axis not in axes]
    if unknown:
        raise ValueError(
            f"calibration type {spec.name!r} has no {unknown[0]!r} axis, so coord_kwargs "
            f"cannot configure it; its axes are {list(axes)}"
        )

    # Only the sized, ranged axes have factories that accept extra keywords.
    # Naming a label axis or direction here would otherwise be silently
    # discarded (label axes) or blow up inside the factory with an unrelated
    # TypeError (direction_coord), neither of which is the clear error a
    # genuine user mistake deserves.
    not_configurable = [axis for axis in coord_kwargs if axis not in COORD_KWARGS_AXES]
    if not_configurable:
        axis = not_configurable[0]
        hint = (
            " Receptor labels are set with the receptor_labels parameter, not coord_kwargs."
            if axis == "receptor_label"
            else ""
        )
        raise ValueError(
            f"{axis!r} takes no coordinate configuration; coord_kwargs only configures "
            f"{list(COORD_KWARGS_AXES)}.{hint}"
        )

    coords: dict[str, xr.DataArray] = {}
    for axis in axes:
        if axis == "receptor_label":
            coords[axis] = receptor_label_coord(receptor_labels)
        elif axis == "parameter_label":
            coords[axis] = parameter_label_coord(labels)
        else:
            factory = SIZED_AXIS_FACTORIES[axis]
            coords[axis] = factory(sizes[axis], **coord_kwargs.get(axis, {}))
    return coords


def _generate_values(
    param: ParamSpec,
    shape: tuple[int, ...],
    rng: np.random.Generator,
    amplitude_jitter: float,
) -> np.ndarray:
    """Generate random values for one parameter.

    Complex parameters are gains, so they are generated as a unit amplitude
    perturbed by a small normal deviate, with uniform random phase. Real
    parameters are standard normal scaled by the parameter's magnitude hint.

    Args:
        param: The parameter being generated.
        shape: Shape to generate.
        rng: Random number generator.
        amplitude_jitter: Fractional spread of complex amplitude about unity.

    Returns:
        An array of param.dtype and the requested shape.
    """
    dtype = np.dtype(param.dtype)
    if dtype.kind == "c":
        amplitude = 1.0 + amplitude_jitter * rng.standard_normal(shape)
        phase = rng.uniform(-np.pi, np.pi, shape)
        return (amplitude * np.exp(1j * phase)).astype(dtype)
    return (param.scale * rng.standard_normal(shape)).astype(dtype)


#: Axes a flag never carries. A flag marks a whole solution bad, and these two
#: axes index the components of one solution rather than distinct solutions: the
#: quantities a single solve produced, and the receptors it solved together. If
#: one component of a solution cannot be trusted, neither can the rest of it.
UNFLAGGED_AXES = ("receptor_label", "parameter_label")


def _flag_dims(dims: Sequence[str]) -> tuple[str, ...]:
    """Derive the flag dimensions for a parameter array.

    Args:
        dims: Dimensions of the parameter array, in canonical order.

    Returns:
        The same dimensions without the component axes (see UNFLAGGED_AXES).
    """
    return tuple(dim for dim in dims if dim not in UNFLAGGED_AXES)


def _generate_flags(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    flag_fraction: float,
) -> np.ndarray:
    """Generate a boolean flag array.

    Args:
        shape: Shape to generate.
        rng: Random number generator.
        flag_fraction: Probability that any given solution is flagged.

    Returns:
        A boolean array of the requested shape.
    """
    # These two short circuits skip drawing from rng entirely, rather than
    # drawing and thresholding as usual. Both builders draw every parameter
    # value before drawing any flag, so this cannot shift a parameter's values;
    # it does mean the flags themselves are not a thresholded version of the
    # same draws at the extremes.
    if flag_fraction <= 0.0:
        return np.zeros(shape, dtype=bool)
    if flag_fraction >= 1.0:
        return np.ones(shape, dtype=bool)
    return rng.random(shape) < flag_fraction


def _dataset_attrs(spec: CalSpec) -> dict[str, Any]:
    """Build the dataset-level attributes.

    jones_structure is omitted rather than stored as null when the calibration
    type does not have one: a null attribute asserts that the type has a Jones
    structure whose value happens to be nothing.

    Args:
        spec: The calibration type.

    Returns:
        The dataset attributes.
    """
    attrs: dict[str, Any] = {
        "cal_type": spec.name,
        "direction_dependent": spec.direction_dependent,
    }
    if spec.jones_structure is not None:
        attrs["jones_structure"] = spec.jones_structure
    if spec.description:
        attrs["description"] = spec.description
    return attrs


def _check_flag_fraction(flag_fraction: float) -> None:
    """Validate the flag fraction.

    Args:
        flag_fraction: Probability that any given solution is flagged.

    Raises:
        ValueError: If it is not between zero and one inclusive.
    """
    if not 0.0 <= flag_fraction <= 1.0:
        raise ValueError(f"flag_fraction must be between 0 and 1, got {flag_fraction}")


def make_gain_xds(
    spec: CalSpec | str,
    *,
    n_direction: int | None = None,
    n_time: int | None = None,
    n_antenna: int | None = None,
    n_frequency: int | None = None,
    receptor_labels: Sequence[str] = DEFAULT_RECEPTOR_LABELS,
    coord_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int | None = 0,
    flag_fraction: float = DEFAULT_FLAG_FRACTION,
    amplitude_jitter: float = DEFAULT_AMPLITUDE_JITTER,
) -> xr.Dataset:
    """Build one calibration dataset holding one array per parameter.

    Each quantity lives in its own data array, named for the parameter, with a
    scalar ``units`` attribute and exactly the axes its ParamSpec declares. One
    solve is one dataset with one flag.

    A parameter axis appears only where some parameter declares one, which in
    the registry means the several components of a single quantity: an antenna
    position offset's dX, dY and dZ, or the two columns of a Jones term. Every
    parameter declaring that axis shares its coordinate, which CalSpec has
    already checked they agree on.

    Args:
        spec: A CalSpec, or the name of a registered calibration type.
        n_direction: Override the direction extent.
        n_time: Override the time extent.
        n_antenna: Override the antenna extent.
        n_frequency: Override the frequency extent.
        receptor_labels: Receptor labels, for parameters with a receptor axis.
        coord_kwargs: Axis name to extra keyword arguments for that axis's
            coordinate factory. Only ``time``, ``antenna_name``, and
            ``frequency`` accept extra keywords; direction_coord takes none,
            and the label axes get their values from receptor_labels or the
            parameter's own labels instead.
        seed: Seed for the random generator, for reproducibility.
        flag_fraction: Probability that any given solution is flagged.
        amplitude_jitter: Fractional spread of complex amplitude about unity.

    Returns:
        A dataset holding one array per parameter, each named for it, plus a
        single boolean FLAG covering the solve.

    Raises:
        ValueError: If a size or coord_kwargs entry names an axis the type
            lacks, if coord_kwargs names an axis that takes no coordinate
            configuration, or if flag_fraction is out of range.
        KeyError: If spec is a name that is not registered.
    """
    spec = _resolve_spec(spec)
    _check_flag_fraction(flag_fraction)

    sizes = _resolve_sizes(
        spec,
        {
            "n_direction": n_direction,
            "n_time": n_time,
            "n_antenna": n_antenna,
            "n_frequency": n_frequency,
        },
    )
    # coord_kwargs is validated against the calibration type as a whole rather
    # than parameter by parameter, so configuring an axis only some parameters
    # use is not an error.
    axes = spec.axes
    coords = _build_coords(
        spec,
        axes,
        sizes,
        spec.parameter_labels or (),
        receptor_labels,
        coord_kwargs or {},
    )

    rng = np.random.default_rng(seed)
    data_vars: dict[str, Any] = {}

    # Every parameter's values are drawn before any flag, so the flag values a
    # given seed produces do not depend on how many parameters precede them.
    for param in spec.parameters:
        shape = tuple(coords[axis].size for axis in param.axes)
        values = _generate_values(param, shape, rng, amplitude_jitter)
        data_vars[param.name] = (param.axes, values, {"units": param.units})

    # One solve, one flag. Its dimensions span every axis some parameter uses,
    # less the component axes, so a quantity defined over fewer axes than its
    # neighbours is still covered.
    flag_dims = _flag_dims(axes)
    flag_shape = tuple(coords[axis].size for axis in flag_dims)
    data_vars[FLAG_NAME] = (
        flag_dims,
        _generate_flags(flag_shape, rng, flag_fraction),
        {"long_name": "Solution flag"},
    )

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=_dataset_attrs(spec))
