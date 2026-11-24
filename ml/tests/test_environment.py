"""Smoke tests: the toolchain imports and the shared constants are consistent."""

import importlib

import numpy as np
import pytest

from ishara_ml import config


@pytest.mark.parametrize("module", ["mediapipe", "tensorflow", "sklearn", "pandas", "dtaidistance", "cv2"])
def test_dependency_imports(module: str) -> None:
    importlib.import_module(module)


def test_feature_dim_matches_spec() -> None:
    assert config.FEATURE_DIM == 63
    assert config.LANDMARK_SPEC_VERSION >= 1


def test_set_seeds_is_reproducible() -> None:
    config.set_seeds()
    a = np.random.rand(3)
    config.set_seeds()
    b = np.random.rand(3)
    assert np.array_equal(a, b)
