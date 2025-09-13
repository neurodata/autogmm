from sklearn.datasets import make_blobs

from autogmm import AutoGMM


def test_inits_run_all():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.7, random_state=0)
    mdl = AutoGMM(
        min_components=2,
        max_components=4,
        init_agglomerative=True,
        n_init_kmeans=10,
        random_state=0,
    )
    mdl.fit(X)
    assert mdl.labels_ is not None
