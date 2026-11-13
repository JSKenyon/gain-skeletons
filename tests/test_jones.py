"""Tests that a phenomenological gain converts to MSv4 polarization products.

A phenomenological GAIN is a Jones matrix stored over two label axes: rows on
receptor_label, as positions (pos_0, pos_1), and columns on parameter_label, as
receptors (X, Y). Naming the rows by the feed's receptors and flattening the
pair gives one ``polarization`` axis labelled as in an MSv4 visibility dataset
(XX, XY, YX, YY), so a gain lines up with the data it would calibrate.

The conversion lives here rather than in the package: the package builds
datasets and never applies calibration.
"""

from collections.abc import Sequence

import numpy as np
import pytest
import xarray as xr

from gain_skeletons.builder import make_gain_xds

POSITION_PREFIX = "pos_"


def to_jones_polarization(gain: xr.DataArray, receptors: Sequence[str]) -> xr.DataArray:
    """Flatten a Jones-matrix gain onto an MSv4 polarization axis.

    Args:
        gain: Array with receptor_label (positions) and parameter_label
            (receptors) as its two trailing axes.
        receptors: The feed's receptors in position order, so pos_n names
            receptors[n]. Held outside the gain dataset in the schema.

    Returns:
        The same values with the two label axes replaced by a single
        polarization axis, labelled row receptor then column receptor.

    Raises:
        ValueError: If the matrix is not square over the given receptors, or
            its columns are not those receptors in feed order.
    """
    receptors = tuple(receptors)
    columns = tuple(gain.parameter_label.values)
    if gain.sizes["receptor_label"] != len(receptors) or columns != receptors:
        raise ValueError(
            f"gain is {gain.sizes['receptor_label']}x{len(columns)} over columns "
            f"{list(columns)}, which is not a Jones matrix over receptors {list(receptors)}"
        )

    rows = [
        receptors[int(str(label).removeprefix(POSITION_PREFIX))]
        for label in gain.receptor_label.values
    ]
    named = gain.assign_coords(receptor_label=rows)
    stacked = named.stack(polarization=("receptor_label", "parameter_label"), create_index=False)
    labels = np.char.add(
        stacked.receptor_label.values.astype(str), stacked.parameter_label.values.astype(str)
    )
    return stacked.drop_vars(["receptor_label", "parameter_label"]).assign_coords(
        polarization=labels
    )


LINEAR = ("X", "Y")
CIRCULAR = ("R", "L")


def test_linear_feed_gives_msv4_polarization_products():
    gain = make_gain_xds("phenomenological_gain").GAIN
    jones = to_jones_polarization(gain, LINEAR)
    assert jones.dims == ("time", "antenna_name", "frequency", "polarization")
    assert list(jones.polarization.values) == ["XX", "XY", "YX", "YY"]


# The core property: row position n and column receptor p land on the product
# named receptors[n] followed by p.
@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("column", LINEAR)
def test_every_element_lands_on_its_product(row, column):
    gain = make_gain_xds("phenomenological_gain").GAIN
    jones = to_jones_polarization(gain, LINEAR)
    expected = gain.sel(receptor_label=f"pos_{row}", parameter_label=column, drop=True)
    actual = jones.sel(polarization=f"{LINEAR[row]}{column}", drop=True)
    xr.testing.assert_identical(actual, expected)


# Canonical order puts receptor_label before parameter_label, so the stored
# layout is already a row-major Jones matrix and the conversion is a reshape.
def test_conversion_is_a_row_major_reshape():
    gain = make_gain_xds("phenomenological_gain").GAIN
    jones = to_jones_polarization(gain, LINEAR)
    leading = gain.shape[:-2]
    np.testing.assert_array_equal(jones.values, gain.values.reshape(*leading, 4))
    np.testing.assert_array_equal(jones.values.reshape(*leading, 2, 2), gain.values)


def test_circular_feed_gives_circular_products():
    gain = make_gain_xds("phenomenological_gain", parameter_labels=CIRCULAR).GAIN
    jones = to_jones_polarization(gain, CIRCULAR)
    assert list(jones.polarization.values) == ["RR", "RL", "LR", "LL"]


def test_direction_dependent_gain_keeps_its_leading_axes():
    gain = make_gain_xds("dd_phenomenological_gain").GAIN
    jones = to_jones_polarization(gain, LINEAR)
    assert jones.dims == ("direction", "time", "antenna_name", "frequency", "polarization")
    assert jones.sizes["polarization"] == 4


def test_receptors_that_do_not_match_the_matrix_are_rejected():
    gain = make_gain_xds("phenomenological_gain").GAIN
    with pytest.raises(ValueError, match="not a Jones matrix"):
        to_jones_polarization(gain, ("X", "Y", "Z"))
    with pytest.raises(ValueError, match="not a Jones matrix"):
        to_jones_polarization(gain, CIRCULAR)
