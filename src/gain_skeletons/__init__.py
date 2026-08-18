"""Mock xarray/zarr dataset scaffolds for radio interferometric gain solutions.

This package is a demonstrator. Every array value it produces is randomly
generated; nothing here computes or applies calibration.
"""

from gain_skeletons.axes import (
    CANONICAL_AXES,
    antenna_name_coord,
    direction_coord,
    frequency_coord,
    parameter_label_coord,
    receptor_label_coord,
    time_coord,
)
from gain_skeletons.builder import make_gain_xds
from gain_skeletons.registry import REGISTRY, get_spec, list_cal_types
from gain_skeletons.spec import CalSpec, ParamSpec

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_AXES",
    "CalSpec",
    "ParamSpec",
    "REGISTRY",
    "__version__",
    "antenna_name_coord",
    "direction_coord",
    "frequency_coord",
    "get_spec",
    "list_cal_types",
    "make_gain_xds",
    "parameter_label_coord",
    "receptor_label_coord",
    "time_coord",
]
