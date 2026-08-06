"""Builders that turn a calibration type specification into a dataset.

Two layouts are offered, and neither is privileged as the correct one. The
source deck stores each differently-united quantity in its own data array, which
:func:`make_split_gain_xds` reproduces. Consolidating every quantity into one
array indexed by an explicit parameter axis, which :func:`make_gain_xds` does,
keeps the parameters needed to evaluate a Jones term adjacent in memory and on
disk, and lets a single flag describe a single solve.

Nine of the ten registry entries declare one parameter, and for those the two
builders produce identical datasets. The layouts diverge only for fringefit.

All values are random. Complex parameters are generated near unit amplitude,
since a uniform-random complex gain of arbitrary magnitude would be physically
nonsensical.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from gain_skeletons.axes import (
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
PARAMETER_UNITS_COORD = "parameter_units"

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
            factory, such as frequency start and end.

    Returns:
        Axis name to coordinate.

    Raises:
        ValueError: If coord_kwargs names an axis the calibration type lacks.
    """
    unknown = [axis for axis in coord_kwargs if axis not in axes]
    if unknown:
        raise ValueError(
            f"calibration type {spec.name!r} has no {unknown[0]!r} axis, so coord_kwargs "
            f"cannot configure it; its axes are {list(axes)}"
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


def _broadcast_to_axes(
    values: np.ndarray,
    own_axes: Sequence[str],
    target_axes: Sequence[str],
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Broadcast a parameter's values onto a wider set of axes.

    Used by the consolidated layout when a parameter lacks an axis that its
    siblings have, such as a fringe fit's unpolarised dispersive delay. The
    values are repeated along the missing axes, which is redundant, and
    deliberately visible as such.

    Args:
        values: Values defined over own_axes.
        own_axes: Axes the values are defined over, in canonical order.
        target_axes: Axes to broadcast onto, in canonical order. Must be a
            superset of own_axes.
        target_shape: Shape corresponding to target_axes.

    Returns:
        A read-only view of shape target_shape.
    """
    reshaped = values.reshape(
        tuple(values.shape[own_axes.index(axis)] if axis in own_axes else 1 for axis in target_axes)
    )
    return np.broadcast_to(reshaped, target_shape)


def _flag_dims(dims: Sequence[str]) -> tuple[str, ...]:
    """Derive the flag dimensions for a parameter array.

    A flag marks a whole solution bad, and the components of one solution are
    not independently valid, so the parameter axis is dropped.

    Args:
        dims: Dimensions of the parameter array, in canonical order.

    Returns:
        The same dimensions without parameter_label.
    """
    return tuple(dim for dim in dims if dim != "parameter_label")


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
    if flag_fraction <= 0.0:
        return np.zeros(shape, dtype=bool)
    if flag_fraction >= 1.0:
        return np.ones(shape, dtype=bool)
    return rng.random(shape) < flag_fraction


def _dataset_attrs(spec: CalSpec) -> dict[str, Any]:
    """Build the dataset-level attributes.

    jones_structure is omitted rather than stored as null when the deck does not
    specify one: a null attribute asserts that the calibration type has a Jones
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
    receptor_labels: Sequence[str] = ("X", "Y"),
    coord_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int | None = 0,
    flag_fraction: float = DEFAULT_FLAG_FRACTION,
    amplitude_jitter: float = DEFAULT_AMPLITUDE_JITTER,
) -> xr.Dataset:
    """Build a calibration dataset with every parameter in one array.

    Parameters share a single data array, indexed by an explicit parameter axis.
    This keeps the parameters needed to evaluate a Jones term adjacent in memory
    and, once written, in one chunked zarr array. A single flag then describes a
    single solve.

    Where the parameters do not all share an axis, the ones that lack it are
    broadcast over it. Where they do not all share units, units move from a
    scalar attribute to a coordinate aligned to the parameter axis.

    Args:
        spec: A CalSpec, or the name of a registered calibration type.
        n_direction: Override the direction extent.
        n_time: Override the time extent.
        n_antenna: Override the antenna extent.
        n_frequency: Override the frequency extent.
        receptor_labels: Receptor labels, if the type has a receptor axis.
        coord_kwargs: Axis name to extra keyword arguments for that axis's
            coordinate factory, such as ``{"frequency": {"start": 1e9}}``.
        seed: Seed for the random generator, for reproducibility.
        flag_fraction: Probability that any given solution is flagged.
        amplitude_jitter: Fractional spread of complex amplitude about unity.

    Returns:
        A dataset holding one parameter array, named after the calibration
        type's consolidated name, and a boolean FLAG.

    Raises:
        ValueError: If the parameters do not share a dtype, if a size or
            coord_kwargs entry names an axis the type lacks, or if
            flag_fraction is out of range.
        KeyError: If spec is a name that is not registered.
    """
    spec = _resolve_spec(spec)
    _check_flag_fraction(flag_fraction)

    dtype = spec.uniform_dtype
    if dtype is None:
        dtypes = {param.name: param.dtype for param in spec.parameters}
        raise ValueError(
            f"calibration type {spec.name!r} cannot be consolidated because its "
            f"parameters do not share a dtype: {dtypes}. Use make_split_gain_xds instead."
        )

    sizes = _resolve_sizes(
        spec,
        {
            "n_direction": n_direction,
            "n_time": n_time,
            "n_antenna": n_antenna,
            "n_frequency": n_frequency,
        },
    )
    axes = spec.axes

    # Consolidating several parameters requires somewhere to put them. Without
    # a parameter_label axis, the fill loop below has no way to distinguish
    # one parameter's slice from another's, and would silently keep only the
    # last one written.
    if len(spec.parameters) > 1 and "parameter_label" not in axes:
        names = [param.name for param in spec.parameters]
        raise ValueError(
            f"calibration type {spec.name!r} cannot be consolidated because it has "
            f"{len(spec.parameters)} parameters but no parameter_label axis to "
            f"distinguish them: {names}. Give its parameters a parameter_label axis, "
            "or use make_split_gain_xds instead."
        )

    coords = _build_coords(
        spec,
        axes,
        sizes,
        spec.all_labels,
        receptor_labels,
        coord_kwargs or {},
    )
    shape = tuple(coords[axis].size for axis in axes)
    rng = np.random.default_rng(seed)

    # The parameter axis is last in canonical order, so each parameter occupies a
    # contiguous trailing slice. Types without a parameter axis have exactly one
    # parameter filling the whole array, and must not be sliced at all: the
    # trailing axis would be receptor_label.
    has_parameter_axis = "parameter_label" in axes
    values = np.empty(shape, dtype=np.dtype(dtype))
    offset = 0
    for param in spec.parameters:
        width = len(param.resolved_labels)
        own_shape = tuple(
            width if axis == "parameter_label" else coords[axis].size for axis in param.axes
        )
        generated = _generate_values(param, own_shape, rng, amplitude_jitter)
        target_shape = tuple(
            width if axis == "parameter_label" else coords[axis].size for axis in axes
        )
        broadcast = _broadcast_to_axes(generated, param.axes, axes, target_shape)
        if has_parameter_axis:
            values[..., offset : offset + width] = broadcast
            offset += width
        else:
            values[...] = broadcast

    flag_dims = _flag_dims(axes)
    flag_shape = tuple(coords[axis].size for axis in flag_dims)

    units = spec.uniform_units
    parameter_attrs: dict[str, Any] = {} if units is None else {"units": units}

    xds = xr.Dataset(
        data_vars={
            spec.resolved_consolidated_name: (axes, values, parameter_attrs),
            FLAG_NAME: (
                flag_dims,
                _generate_flags(flag_shape, rng, flag_fraction),
                {"long_name": "Solution flag"},
            ),
        },
        coords=coords,
        attrs=_dataset_attrs(spec),
    )

    if units is None:
        # Units differ between parameters, so a scalar attribute would be a
        # false claim. Carry them alongside the parameter labels instead, where
        # they travel with any selection.
        xds = xds.assign_coords(
            {
                PARAMETER_UNITS_COORD: (
                    "parameter_label",
                    np.array(
                        [param.units for param in spec.parameters for _ in param.resolved_labels]
                    ),
                )
            }
        )
        xds[PARAMETER_UNITS_COORD].attrs["long_name"] = "Parameter units"

    return xds
