import logging
import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import inv
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.linalg import eigh, pinvh
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import SpectralEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_array


def ward_mahalanobis_linkage(X, method="ward"):
    # 1) centre
    Xc = X - np.mean(X, axis=0)

    # 2) PCA reduction
    pca = PCA(n_components=0.99, svd_solver="full").fit(Xc)
    Xp = pca.transform(Xc)

    # 2) OAS shrinkage covariance Σ̂ and its inverse Σ̂⁻¹
    cov_oas = OAS(assume_centered=True).fit(Xp).covariance_
    VI = inv(cov_oas)  # OAS is well-conditioned

    # 3) Mahalanobis pairwise distances + linkage
    D = pdist(Xp, metric="mahalanobis", VI=VI)
    return scipy_linkage(D, method=method)


def _initialize_params(X, labels, covariance_type, eigen_thres=False):
    uniq, labels = np.unique(labels, return_inverse=True)
    K = len(uniq)
    D = X.shape[1]
    means, covs, weights = [], [], []
    for k in range(K):
        Xk = X[labels == k]
        if Xk.shape[0] < 2:
            var = np.var(X, axis=0).mean()
            Ck = np.eye(D) * var
        else:
            Ck = np.cov(Xk, rowvar=False)
            if eigen_thres:
                ev, evec = eigh(Ck)
                ev = np.maximum(ev, np.sqrt(np.finfo(float).eps))
                Ck = evec @ np.diag(ev) @ evec.T
        if covariance_type == "spherical":
            Ck = np.eye(D) * np.trace(Ck) / D
        means.append(Xk.mean(0))
        covs.append(Ck)
        weights.append(len(Xk) / len(X))
    return means, covs, weights


def get_precisions_init(covs, cov_type, n_features, eigen_thres=False, reg_covar=1e-6):
    if eigen_thres:

        def invf(C):
            try:
                return np.linalg.inv(C)
            except np.linalg.LinAlgError:
                return pinvh(C + reg_covar * np.eye(n_features))

    else:

        def invf(C):
            return pinvh(C + reg_covar * np.eye(n_features))

    if cov_type == "full":
        P = []
        for C in covs:
            Pi = invf(C)
            Pi = 0.5 * (Pi + Pi.T)
            P.append(Pi)
        return P

    if cov_type == "tied":
        avg = sum(covs) / len(covs)
        Pi = invf(avg)
        return 0.5 * (Pi + Pi.T)

    if cov_type == "diag":
        return [1.0 / np.diag(C) for C in covs]

    # spherical
    return [1.0 / (np.trace(C) / n_features + reg_covar) for C in covs]


class MclustStyleGM(GaussianMixture):
    def __init__(self, eigen_thres=True, reg_covar=1e-6, **kwargs):
        self.eigen_thres = eigen_thres
        self.eps = np.sqrt(np.finfo(float).eps)
        super().__init__(reg_covar=(self.eps if eigen_thres else reg_covar), **kwargs)

    def _m_step(self, X, log_resp):
        super()._m_step(X, log_resp)
        if self.eigen_thres and self.covariance_type in ("full", "tied"):
            if self.covariance_type == "tied":
                ev, evec = eigh(self.covariances_)
                ev = np.maximum(ev, self.eps)
                self.covariances_ = evec @ np.diag(ev) @ evec.T
            else:
                for k in range(self.n_components):
                    ev, evec = eigh(self.covariances_[k])
                    ev = np.maximum(ev, self.eps)
                    self.covariances_[k] = evec @ np.diag(ev) @ evec.T


class AutoGMM(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        min_components=1,
        max_components=10,
        init_agglomerative=True,
        agglom_linkages=None,
        agglom_affinities=None,
        n_init_kmeans=10,
        eigen_thres=True,
        reg_covar=1e-6,
        criterion="bic",
        random_state=None,
        verbose=False,
        n_jobs=1,
        early_stop_delta=1e-3,
        y=None,
        covariances=None,
    ):
        self.min_components = min_components
        self.max_components = max_components
        self.init_agglomerative = init_agglomerative
        self.agglom_linkages = agglom_linkages or [
            "ward",
            "complete",
            "average",
            "single",
        ]
        self.agglom_affinities = agglom_affinities or [
            "euclidean",
            "cityblock",
            "cosine",
            "mahalanobis",
        ]
        self.n_init_kmeans = n_init_kmeans
        self.eigen_thres = eigen_thres
        self.reg_covar = reg_covar
        self.criterion = criterion
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.early_stop_delta = early_stop_delta
        self.y = y

        _allowed_covs = ("spherical", "diag", "tied", "full")
        if covariances is None:
            self.covariances = list(_allowed_covs)
        else:
            if not isinstance(covariances, (list, tuple)):
                raise TypeError("covariances must be a list/tuple of strings or None.")
            covs = list(covariances)
            bad = [c for c in covs if c not in _allowed_covs]
            if bad:
                raise ValueError(
                    f"Unknown covariance types: {bad}. Allowed: {_allowed_covs}"
                )
            if not covs:
                raise ValueError("covariances must be non-empty or None.")
            self.covariances = covs

        if not isinstance(self.min_components, int) or self.min_components < 1:
            raise ValueError("min_components must be a positive integer.")
        if not isinstance(self.max_components, int) or self.max_components < 1:
            raise ValueError("max_components must be a positive integer.")
        if self.min_components > self.max_components:
            raise ValueError("min_components must be <= max_components.")

        if not isinstance(self.n_init_kmeans, int) or self.n_init_kmeans < 1:
            raise ValueError("n_init_kmeans must be >= 1.")

        if not isinstance(self.reg_covar, (int, float)) or self.reg_covar < 0:
            raise ValueError("reg_covar must be non-negative.")

        if (
            not isinstance(self.early_stop_delta, (int, float))
            or self.early_stop_delta < 0
        ):
            raise ValueError("early_stop_delta must be >= 0.")

    def _fit_single(self, X, cov_type, k, labels_init):
        if len(np.unique(labels_init)) != k:
            if self.verbose:
                logging.warning(
                    f"Skipping init cov={cov_type},k={k}: got {len(np.unique(labels_init))}"
                )
            return None, np.inf

        means, covs, weights = _initialize_params(
            X, labels_init, cov_type, self.eigen_thres
        )
        precs = get_precisions_init(
            covs, cov_type, X.shape[1], self.eigen_thres, self.reg_covar
        )

        G = MclustStyleGM if self.eigen_thres else GaussianMixture
        gmm = G(
            n_components=k,
            covariance_type=cov_type,
            means_init=means,
            weights_init=weights,
            precisions_init=precs,
            random_state=self.random_state,
            max_iter=1000,
        )
        gmm.fit(X)
        if not gmm.converged_:
            return None, np.inf

        score = gmm.bic(X) if self.criterion == "bic" else gmm.aic(X)
        return gmm, score

    def fit(self, X, y=None):
        X = check_array(X)

        # — SciPy agglom inits via ONE linkage + fcluster(…maxclust)… —
        labels_agglom = {}
        if self.init_agglomerative:
            dc = {}
            # 1× Euclidean‐Ward is free to precompute:
            if "ward" in self.agglom_linkages and "euclidean" in self.agglom_affinities:
                Z_euclid_ward = scipy_linkage(X, method="ward")
            else:
                Z_euclid_ward = None

            for link in self.agglom_linkages:
                for aff in self.agglom_affinities:
                    if link == "ward" and aff not in ("euclidean", "mahalanobis"):
                        continue
                    if link != "ward" and aff == "mahalanobis":
                        continue

                    if aff == "mahalanobis":
                        Z = ward_mahalanobis_linkage(X, method=link)
                    elif link == "ward":
                        # reuse the one Euclidean‐Ward you already computed
                        Z = Z_euclid_ward
                    else:
                        m = "cityblock" if aff == "cityblock" else aff
                        if m not in dc:
                            dc[m] = pdist(X, metric=m)
                        Z = scipy_linkage(dc[m], method=link)

                    for k in range(self.min_components, self.max_components + 1):
                        labs = fcluster(Z, k, criterion="maxclust")
                        if len(np.unique(labs)) == k:
                            labels_agglom[(link, aff, k)] = labs

        # — kmeans inits —
        labels_km = {}
        for k in range(self.min_components, self.max_components + 1):
            for i in range(self.n_init_kmeans):
                seed_i = (0 if self.random_state is None else self.random_state) + i
                km = KMeans(n_clusters=k, n_init=1, random_state=seed_i)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    labels_km[(k, i)] = km.fit_predict(X)

        # — assemble all inits —
        tasks = []
        for cov in self.covariances:
            if self.init_agglomerative:
                for (_, _, k), labs in labels_agglom.items():
                    tasks.append((cov, k, labs))
            for k in range(self.min_components, self.max_components + 1):
                for i in range(self.n_init_kmeans):
                    tasks.append((cov, k, labels_km[(k, i)]))

        # — fit in parallel —
        self.best_score_, self.best_model_ = np.inf, None
        for gmm, score in Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single)(X, cov, k, lbl) for cov, k, lbl in tasks
        ):
            if gmm is not None and score + self.early_stop_delta < self.best_score_:
                self.best_score_, self.best_model_ = score, gmm

        self.labels_ = self.best_model_.predict(X)
        self.n_components_ = getattr(self.best_model_, "n_components", None)
        self.covariance_type_ = getattr(self.best_model_, "covariance_type", None)

        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


class KernelAutoGMM(ClusterMixin):
    """
    AutoGMM + optional spectral front-end with:
      • global-median γ
      • MST-injection into both ASE & LSE graphs
      • PCA-MLE ≥2
      • SciPy fcluster for agglomerative inits
    """

    def __init__(
        self,
        kernel_embedding=False,
        n_neighbors=None,
        gamma=None,
        use_mle=True,
        **auto_kwargs,
    ):
        self.kernel_embedding = kernel_embedding
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.use_mle = use_mle
        self.auto_kwargs = auto_kwargs.copy()
        self._best_method = None
        self.best_score_ = np.inf
        self.best_model_ = None
        self._knn_graph = None
        self._rbf_graph = None
        self._dhat = None

    def _determine_neighbors(self, n):
        return int(np.sqrt(n))

    def _prepare_graphs_and_dhat(self, X):
        n, d = X.shape
        k = self.n_neighbors or self._determine_neighbors(n)

        # — 1) compute kNN connectivity A —
        A = kneighbors_graph(
            X,
            n_neighbors=k,
            mode="connectivity",
            include_self=True,
            n_jobs=self.auto_kwargs.get("n_jobs", 1),
        )
        A = 0.5 * (A + A.T)

        # — 2) compute kNN distances Dk —
        Dk = kneighbors_graph(
            X,
            n_neighbors=k,
            mode="distance",
            include_self=False,
            n_jobs=self.auto_kwargs.get("n_jobs", 1),
        )
        Dk = 0.5 * (Dk + Dk.T)

        # — 3) global-median heuristic for γ —
        full_d = pdist(X, metric="euclidean")
        σ = np.median(full_d)
        γ = self.gamma if self.gamma is not None else 1.0 / (2 * σ * σ)

        # — 4) RBF on kNN graph —
        W = Dk.copy()
        W.data = np.exp(-γ * (W.data**2))
        W = 0.5 * (W + W.T)

        # — 5) MST on full graph, inject into both A & W —
        Dfull = squareform(full_d)
        T = minimum_spanning_tree(Dfull)
        rows, cols = T.nonzero()

        # inject with RBF weighting (now γ small enough to keep >0)
        W = W.tolil()
        for i, j in zip(rows, cols):
            w = np.exp(-γ * (T[i, j] ** 2))
            # ensure strictly positive
            W[i, j] = max(w, np.finfo(float).tiny)
            W[j, i] = W[i, j]

        # and connect ASE graph (binary)
        A = A.tolil()
        for i, j in zip(rows, cols):
            A[i, j] = 1
            A[j, i] = 1

        self._knn_graph = A.tocsr()
        self._rbf_graph = W.tocsr()

        # — 6) PCA-MLE ≥2 guard —
        if self.use_mle:
            p = PCA(n_components="mle", svd_solver="full").fit(X).n_components_
            self._dhat = max(p, 2)
        else:
            self._dhat = max(d, 2)

    def _ase(self):
        vals, vecs = eigsh(self._knn_graph, k=self._dhat, which="LA")
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        vals = np.clip(vals, 0, None)
        return vecs * np.sqrt(vals)

    def _lse(self):
        se = SpectralEmbedding(
            n_components=self._dhat, affinity="precomputed", eigen_solver="arpack"
        )
        return se.fit_transform(self._rbf_graph)

    def fit(self, X, y=None):
        X = check_array(X)

        # — raw AutoGMM —
        base = AutoGMM(**self.auto_kwargs)
        base.fit(X, y)
        best_score, best_model, best_method = base.best_score_, base.best_model_, "none"

        # — optional ASE / LSE —
        if self.kernel_embedding:
            self._prepare_graphs_and_dhat(X)
            for method in ("ase", "lse"):
                try:
                    Xp = getattr(self, f"_{method}")()
                except Exception as e:
                    logging.warning(f"Embedding {method} failed: {e}")
                    continue
                m = AutoGMM(**self.auto_kwargs)
                m.fit(Xp, y)
                if m.best_score_ < best_score:
                    best_score, best_model, best_method = (
                        m.best_score_,
                        m.best_model_,
                        method,
                    )

        self.best_score_, self.best_model_, self._best_method = (
            best_score,
            best_model,
            best_method,
        )
        self.labels_ = self.best_model_.predict(X)
        self.n_components_ = getattr(self.best_model_, "n_components", None)
        self.covariance_type_ = getattr(self.best_model_, "covariance_type", None)

        return self

    def predict(self, X):
        if self.kernel_embedding and self._best_method in ("ase", "lse"):
            Xp = getattr(self, f"_{self._best_method}")()
        else:
            Xp = X
        return self.best_model_.predict(Xp)

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)
