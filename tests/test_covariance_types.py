from sklearn.datasets import make_blobs

from autogmm import AutoGMM


def test_covariance_grid_runs():
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=0)
    for cov in ["spherical", "diag", "tied", "full"]:
        mdl = AutoGMM(
            min_components=2, max_components=4, covariances=[cov], random_state=0
        )
        y = mdl.fit_predict(X)
        assert y.size == X.shape[0]
