"""Tests that the package is installed and importable."""

import gain_skeletons


def test_package_exposes_version():
    assert isinstance(gain_skeletons.__version__, str)
    assert gain_skeletons.__version__


def test_all_exported_names_are_importable():
    """Every name in __all__ must actually exist on the package."""
    missing = [name for name in gain_skeletons.__all__ if not hasattr(gain_skeletons, name)]
    assert not missing
