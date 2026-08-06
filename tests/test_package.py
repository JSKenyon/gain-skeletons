"""Tests that the package is installed and importable."""

import gain_skeletons


def test_package_exposes_version():
    assert isinstance(gain_skeletons.__version__, str)
    assert gain_skeletons.__version__
