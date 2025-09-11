"""Module for Slow Feature Analysis (SFA).

This is used as a starting point (initial vector) for starting the ForeCA vector search.

Usually ForeCA loadings are quite similar to SFA loadings, where SFA can be calculated in
closed form & quickly.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg as sla
import sklearn
from numpy.typing import NDArray
from sklearn.utils.validation import check_array


def _sfa_eigendecomposition(
    U: NDArray[np.floating],
    *,
    eps: float = 1e-10,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve the SFA generalized eigenproblem CΔ w = λ C w.

    Returns eigenvalues (ascending) and eigenvectors as columns (V ∈ R^{K×K}).
    The columns V satisfy V.T @ C @ V = I (C-orthonormal).

    Args:
      U: (n_samples, n_features) data matrix.
      eps: small ridge added to C and CΔ for numerical stability.

    Returns:
      evals: (K,) eigenvalues in ascending order (slowness scores).
      V: (K, K) eigenvectors as columns in the original feature space.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] < 3:
        raise ValueError("SFA requires U with shape (n_samples>=3, n_features).")

    # Center and differences
    Uc = U - U.mean(axis=0, keepdims=True)
    dU = Uc[1:] - Uc[:-1]

    # Covariances (unbiased)
    C = np.cov(Uc, rowvar=False) + eps * np.eye(Uc.shape[1])
    C_delta = np.cov(dU, rowvar=False) + eps * np.eye(Uc.shape[1])

    # Reduce to standard eigenproblem via whitening by C: C = L L^T
    L = sla.cholesky(C, lower=True, check_finite=True)
    Linv = sla.solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=True)
    A = Linv @ C_delta @ Linv.T  # symmetric PSD

    # Standard Hermitian eigendecomposition (ascending)
    evals, Q = sla.eigh(A, check_finite=True)  # Q orthonormal

    # Map back: V = L^{-T} Q (columns are eigenvectors of (CΔ, C))
    V = sla.solve_triangular(L.T, Q, lower=False, check_finite=True)
    return evals, V


class SFA(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Slow Feature Analysis (SFA).

    Finds directions that minimize Var(wᵀ ΔU) subject to Var(wᵀ U) = 1.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of slow features to keep. If None, keeps all.
    copy : bool, default=True
        Whether to copy input in validation.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    components_ : ndarray of shape (n_components, n_features)
        Rows are loading vectors (each row is wᵀ).
    slowness_ : ndarray of shape (n_components,)
        Generalized eigenvalues λ (smaller = slower).
    mean_ : ndarray of shape (n_features,)
        Per-feature mean used for centering.
    transformation_matrix_ : ndarray of shape (n_features, n_components)
        Column-stacked eigenvectors (same as components_.T).
    pinv_transformation_ : ndarray of shape (n_components, n_features)
        Moore–Penrose pseudo-inverse of transformation_matrix_.
    """

    def __init__(self, n_components: Optional[int] = None, *, copy: bool = True):
        self.n_components = n_components
        self.copy = copy

    def fit(self, X: NDArray[np.floating], y=None):
        X = check_array(X, copy=self.copy, dtype=np.float64)
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 samples for SFA")
        self.n_features_in_ = X.shape[1]

        # Center only (no standardization; SFA uses C in the constraint)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Full eigendecomposition
        evals, V = _sfa_eigendecomposition(Xc)  # evals ascending, V columns

        # Determine number of components
        n_total = V.shape[0]
        n_comp = self.n_components if self.n_components is not None else n_total
        if not (1 <= n_comp <= n_total):
            raise ValueError(
                f"n_components must be in [1, {n_total}] (got {self.n_components})"
            )

        # Keep the slowest components (smallest eigenvalues)
        Vk = V[:, :n_comp]  # (K, k)
        self.transformation_matrix_ = Vk  # columns are eigenvectors
        self.components_ = Vk.T  # rows are eigenvectors
        self.slowness_ = evals[:n_comp]  # ascending
        # Pseudo-inverse supports both reduced and full cases
        self.pinv_transformation_ = np.linalg.pinv(Vk)
        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if not hasattr(self, "transformation_matrix_"):
            raise ValueError("SFA must be fitted before transforming")
        X = check_array(X, copy=self.copy, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SFA was fitted with {self.n_features_in_} features"
            )
        Xc = X - self.mean_
        # slow_feats = Xc @ V  (n×K @ K×k → n×k)
        return Xc @ self.transformation_matrix_

    def inverse_transform(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        if not hasattr(self, "transformation_matrix_"):
            raise ValueError("SFA must be fitted before inverse transforming")
        Z = check_array(Z, copy=self.copy, dtype=np.float64)
        k = self.transformation_matrix_.shape[1]
        if Z.shape[1] != k:
            raise ValueError(
                f"Z has {Z.shape[1]} features, but SFA expects {k} features"
            )

        # Xc ≈ Z @ V^+ ; V^+ is Moore–Penrose pseudoinverse (exact if k=K)
        Xc = Z @ self.pinv_transformation_
        return Xc + self.mean_

    def get_feature_names_out(self, input_features=None):
        k = self.transformation_matrix_.shape[1]
        return np.array([f"sf_{i + 1}" for i in range(k)], dtype=object)
