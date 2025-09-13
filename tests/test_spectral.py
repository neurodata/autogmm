from sklearn.datasets import make_moons

from autogmm import KernelAutoGMM


def test_spectral_frontend_smoketest():
    X, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
    mdl = KernelAutoGMM(
        min_components=2, max_components=6, kernel_embedding=True, random_state=0
    )
    y = mdl.fit_predict(X)
    assert y.shape[0] == X.shape[0]
