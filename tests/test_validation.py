import numpy as np
import pytest

from autogmm import AutoGMM


def test_bad_params_raise():
    with pytest.raises(ValueError):
        AutoGMM(min_components=5, max_components=3)


def test_determinism_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    a = AutoGMM(min_components=1, max_components=3, random_state=42).fit_predict(X)
    b = AutoGMM(min_components=1, max_components=3, random_state=42).fit_predict(X)
    assert (a == b).all()
