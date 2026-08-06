"""Declarative description of a calibration type.

A :class:`ParamSpec` describes one named physical quantity: its units, its
dtype, the axes over which it is defined, and the labels it occupies along the
parameter axis. A :class:`CalSpec` gathers the parameters belonging to one
calibration type.

Both are frozen and validate on construction, so an invalid calibration type
cannot be represented.

The source deck runs two senses of "parameter" together, and this module keeps
them apart under a single mechanism:

- Same-unit parameters within one quantity, such as the aligned and cross gains
  of a full Jones term, are one ParamSpec with several labels.
- Different-unit quantities within one calibration type, such as a fringe fit's
  phase, delay and rate, are several ParamSpecs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from gain_skeletons.axes import CANONICAL_AXES, LABEL_AXES, SIZED_AXES, sorted_axes

# Calibration parameters are either complex, when they are gains, or real, when
# they are physical quantities from which gains are generated. Integer and
# boolean parameters have no meaning here; FLAG is generated separately.
SUPPORTED_DTYPE_KINDS = ("c", "f")


def _validate_axes(owner: str, axes: Sequence[str]) -> None:
    """Validate an axis tuple for a parameter.

    Args:
        owner: Description of the owner, used in error messages.
        axes: Axis names, expected in canonical order without duplicates.

    Raises:
        ValueError: If an axis is unknown, duplicated, or out of canonical order.
    """
    if len(set(axes)) != len(axes):
        raise ValueError(f"{owner} declares duplicate axes: {list(axes)}")
    # sorted_axes raises on unknown names, so this validates membership too.
    canonical = sorted_axes(axes)
    if canonical != tuple(axes):
        raise ValueError(
            f"{owner} declares axes out of canonical order: got {list(axes)}, "
            f"expected {list(canonical)}"
        )


@dataclass(frozen=True)
class ParamSpec:
    """One named calibration parameter.

    Attributes:
        name: Data array name in the split layout, and the default parameter
            label. Conventionally upper case, following MSv4.
        units: Units of the parameter. One parameter always has exactly one unit.
        axes: Axes over which the parameter is defined, in canonical order.
            Axes absent from this tuple are absent from the dataset entirely,
            which is how the deck's brace notation is represented.
        dtype: Numpy dtype name. Must be a complex or floating kind.
        labels: Values this parameter occupies along the parameter axis. None
            means a single label equal to name.
        scale: Magnitude hint for random generation, so that a delay in seconds
            and a phase in degrees do not come out the same size. Ignored for
            complex parameters, which are generated around unit amplitude.
    """

    name: str
    units: str
    axes: tuple[str, ...]
    dtype: str
    labels: tuple[str, ...] | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the parameter.

        Raises:
            ValueError: If the axes are invalid, the labels are empty, several
                labels are declared without a parameter axis, or the dtype is
                not a complex or floating kind.
        """
        owner = f"parameter {self.name!r}"
        _validate_axes(owner, self.axes)

        if self.labels is not None:
            if len(self.labels) == 0:
                raise ValueError(f"{owner} needs at least one label")
            if len(self.labels) > 1 and "parameter_label" not in self.axes:
                raise ValueError(
                    f"{owner} declares {len(self.labels)} labels but no parameter_label axis"
                )

        if np.dtype(self.dtype).kind not in SUPPORTED_DTYPE_KINDS:
            raise ValueError(
                f"{owner} has unsupported dtype {self.dtype!r}; expected a complex or float dtype"
            )

    @property
    def resolved_labels(self) -> tuple[str, ...]:
        """Labels this parameter occupies along the parameter axis."""
        return self.labels if self.labels is not None else (self.name,)


@dataclass(frozen=True)
class CalSpec:
    """A calibration type: one or more parameters sharing a coordinate system.

    Attributes:
        name: Registry key for the calibration type.
        parameters: The parameters, in declaration order. That order fixes the
            order of the consolidated parameter axis.
        default_sizes: Default extent of every sized axis the type uses. This
            is what distinguishes a deliberately single-channel axis from a
            channel-resolved one.
        consolidated_name: Data array name in the consolidated layout. May be
            omitted only when there is exactly one parameter, in which case
            that parameter's name is used.
        jones_structure: Which part of the Jones matrix the type populates, if
            the deck specifies one. Describes the origin of the data, not an
            instruction for its use.
        description: Human-readable summary.
    """

    name: str
    parameters: tuple[ParamSpec, ...]
    default_sizes: Mapping[str, int] = field(default_factory=dict)
    consolidated_name: str | None = None
    jones_structure: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the calibration type.

        Raises:
            ValueError: If there are no parameters, labels collide across
                parameters, a multi-parameter type omits consolidated_name, or
                default_sizes does not correspond exactly to the sized axes in
                use.
        """
        if not self.parameters:
            raise ValueError(f"calibration type {self.name!r} needs at least one parameter")

        labels = self.all_labels
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"calibration type {self.name!r} has duplicate parameter labels: {list(labels)}"
            )

        if self.consolidated_name is None and len(self.parameters) > 1:
            raise ValueError(
                f"calibration type {self.name!r} has {len(self.parameters)} parameters "
                "and so must set consolidated_name"
            )

        axes = self.axes
        for axis in self.default_sizes:
            if axis in LABEL_AXES:
                raise ValueError(
                    f"the extent of {axis!r} is never configurable; it follows from its labels"
                )
            if axis not in axes:
                raise ValueError(f"{axis!r} is not an axis of calibration type {self.name!r}")

        missing = [axis for axis in axes if axis in SIZED_AXES and axis not in self.default_sizes]
        if missing:
            raise ValueError(
                f"calibration type {self.name!r} is missing default size for {missing}"
            )

    @property
    def axes(self) -> tuple[str, ...]:
        """Union of every parameter's axes, in canonical order."""
        union = {axis for param in self.parameters for axis in param.axes}
        return tuple(axis for axis in CANONICAL_AXES if axis in union)

    @property
    def direction_dependent(self) -> bool:
        """Whether this calibration type resolves direction within a field of view."""
        return "direction" in self.axes

    @property
    def resolved_consolidated_name(self) -> str:
        """Data array name to use in the consolidated layout."""
        if self.consolidated_name is not None:
            return self.consolidated_name
        return self.parameters[0].name

    @property
    def all_labels(self) -> tuple[str, ...]:
        """Every parameter label, concatenated in declaration order."""
        return tuple(label for param in self.parameters for label in param.resolved_labels)

    @property
    def uniform_units(self) -> str | None:
        """The units shared by every parameter, or None if they differ."""
        units = {param.units for param in self.parameters}
        return units.pop() if len(units) == 1 else None

    @property
    def uniform_dtype(self) -> str | None:
        """The dtype shared by every parameter, or None if they differ."""
        dtypes = {param.dtype for param in self.parameters}
        return dtypes.pop() if len(dtypes) == 1 else None
