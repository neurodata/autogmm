from sklearn.datasets import make_blobs

from autogmm import AutoGMM


def test_fit_predict_shape():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.6, random_state=0)
    mdl = AutoGMM(min_components=1, max_components=5, random_state=0)
    y = mdl.fit_predict(X)
    assert y.shape == (X.shape[0],)
