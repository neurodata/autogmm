# AutoGMM

Automatic Gaussian Mixture Modeling in Python.

### Install
``` bash
pip install autogmm
```

### Quick start
``` python
from autogmm import AutoGMM
from sklearn.datasets import make_blobs

X, _ = make_blobs(
                 n_samples=1000,
                 centers=4,
                 cluster_std=1.2,
                 random_state=0
)

labels = AutoGMM(
                 min_components=1,
                 max_components=10, # unknown K
                 criterion="bic",
                 random_state=0
).fit_predict(X)
```


### Features
- Initializations: KMeans, Ward–Euclidean, Ward–Mahalanobis

- EM with eigenvalue flooring and covariance constraints (spherical, diag, tied, full)

- Model selection via BIC/AIC (unknown *K*)

- Optional spectral front-end (ASE/LSE) for nonconvex structure

- Parallel evaluation, clean API, reproducible scripts



### Documentation
- API & Guides: https://github.com/neurodata/autogmm/
- Examples: [examples/](examples) (benchmarks; stress tests; runtime scaling)
- Reproducibility: [scripts/reproduce.sh](scripts/reproduce.sh) (regenerates all figures with fixed seeds)

### Legacy & Independence
AutoGMM was originally developed in the [graspologic](https://github.com/graspologic-org/graspologic/) library.
As of v1.0, it is a standalone package with no dependency on graspologic.

### Contributing
Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Citation
``` bibtex
@software{autogmm,
  title   = {AutoGMM: Automatic Gaussian Mixture Modeling in Python},
  author  = {Liu, Tingshan and Athey, Thomas L. and Pedigo, Benjamin D. and Vogelstein, Joshua T.},
  year    = {2025},
  url     = {https://github.com/neurodata/autogmm}
}
```

### License
BSD 3-Clause License. See [LICENSE](LICENSE).
