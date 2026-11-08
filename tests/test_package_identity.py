"""Regression checks for the clean bandedqr package/native rename."""
from pathlib import Path

import pytest

import bandedqr
from bandedqr import _native


def test_package_and_native_names():
    assert Path(bandedqr.__file__).parent.name == "bandedqr"
    assert "bandedqr" in Path(_native.lib._name).name
    assert hasattr(_native.lib, "bandedqr_factor_d")
    assert hasattr(_native.lib, "bandedqr_factor_z")
    with pytest.raises(AttributeError):
        getattr(_native.lib, "bandqr_factor_d")
    assert bandedqr.BandQR.__name__ == "BandQR"


def test_native_library_override_uses_new_prefix(monkeypatch, tmp_path):
    current = str(tmp_path / "current.so")
    legacy = str(tmp_path / "legacy.so")
    monkeypatch.setenv("BANDEDQR_LIBRARY", current)
    monkeypatch.setenv("BANDQR_LIBRARY", legacy)
    candidates = list(_native._candidates())
    assert candidates[0] == current
    assert legacy not in candidates
    monkeypatch.delenv("BANDEDQR_LIBRARY")
    assert legacy not in list(_native._candidates())
