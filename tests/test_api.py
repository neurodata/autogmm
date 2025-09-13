import numpy as np

from autogmm import AutoGMM


def test_import_and_defaults():
    mdl = AutoGMM(min_components=1, max_components=3, random_state=0)
    assert hasattr(mdl, "fit")
    assert hasattr(mdl, "fit_predict")


def test_fit_small_blob():
    X = np.random.default_rng(0).normal(size=(50, 2))
    mdl = AutoGMM(min_components=1, max_components=2, random_state=0)
    mdl.fit(X)
    assert mdl.labels_ is not None
    assert mdl.n_components_ >= 1
