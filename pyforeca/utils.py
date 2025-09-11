"""
Decorrelator Transformer: sklearn-compatible transformer for decorrelating data.
"""

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array


def cov2corr(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert covariance matrix to correlation matrix."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square 2D array")

    std = np.sqrt(np.diag(cov))
    if np.any(std <= 0):
        raise ValueError("covariance matrix must have positive diagonal entries")

    denom = np.outer(std, std)
    corr = cov / denom
    # clip numerical noise outside [-1,1]
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def partial_cov(X: pd.DataFrame) -> pd.DataFrame:
    """Computes partial covariance between X_i and X_j given other covariates."""
    cov = X.cov()
    part_cov = np.linalg.inv(cov)
    part_cov_df = pd.DataFrame(part_cov, index=cov.index, columns=cov.columns)
    return part_cov_df


def partial_corr(X: pd.DataFrame) -> pd.DataFrame:
    """Computes partial covariance between X_i and X_j given other covariates."""
    part_cov = partial_cov(X)
    part_corr = cov2corr(part_cov)
    part_corr_df = pd.DataFrame(
        part_corr, index=part_cov.index, columns=part_cov.columns
    )
    return part_corr_df


def _acf_1d(x: np.ndarray, nlags: int) -> np.ndarray:
    """ACF r_k, k=0..nlags (unbiased)."""
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    n = x.size
    # FFT-based autocovariance
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(fx * np.conj(fx), n=nfft).real[: nlags + 1]
    acov /= (n - np.arange(nlags + 1)).astype(float)  # unbiased
    return acov / acov[0]


def ljung_box(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Ljung–Box portmanteau test p-value for H0: white noise."""
    x = np.asarray(x, float).ravel()
    n = x.size
    m = lags if lags is not None else max(1, min(20, n // 4))
    r = _acf_1d(x, nlags=m)
    Q = n * (n + 2.0) * np.sum((r[1:] ** 2) / (n - np.arange(1, m + 1)))
    return Q, float(scipy.stats.chi2.sf(Q, df=m))


def assert_decorrelated(U: np.ndarray, tolerance=1e-8) -> dict[str, Any] | None:
    """
    Assert the input is decorrelated.

    Parameters
    ----------
    U : array-like of shape (n_samples, n_features), optional
        Decorrelated data to verify.

    tolerance : float, default=1e-8
        Numerical tolerance for identity matrix check.

    Returns
    -------
    If True, returns nothing; if False, returns dict
        Dictionary containing verification results:
        - 'is_decorrelated': bool, whether covariance is close to identity
        - 'max_off_diagonal': float, maximum off-diagonal covariance
        - 'covariance_matrix': ndarray, the covariance matrix
        - 'variances': ndarray, diagonal elements (should be close to 1)
    """
    # Compute covariance matrix
    cov_U = np.cov(U.T)

    # Check off-diagonal elements (should be close to zero)
    off_diagonal_mask = ~np.eye(cov_U.shape[0], dtype=bool)
    max_off_diagonal = np.max(np.abs(cov_U[off_diagonal_mask]))

    # Check diagonal elements (should be close to 1)
    variances = np.diag(cov_U)

    # Overall decorrelation check
    identity = np.eye(cov_U.shape[0])
    max_deviation = np.max(np.abs(cov_U - identity))
    is_decorrelated = bool(max_deviation < tolerance)

    if is_decorrelated:
        return None

    return {
        "is_decorrelated": is_decorrelated,
        "max_off_diagonal": max_off_diagonal,
        "max_deviation_from_identity": max_deviation,
        "covariance_matrix": cov_U,
        "variances": variances,
    }


class Decorrelator2(PCA):
    """
    Decorrelator Transformer using PCA to produce uncorrelated, unit variance components.

    This transformer converts input data X into decorrelated data U where:
    - All components are uncorrelated (covariance matrix is diagonal/identity)
    - All components have unit variance
    - The transformation preserves all information (no dimensionality reduction)

    This is particularly useful as a preprocessing step for algorithms that assume
    uncorrelated features or for ForeCA analysis.

    Parameters
    ----------
    standardize : bool, default=True
        Whether to standardize the input data before PCA transformation.

    copy : bool, default=True
        Whether to make a copy of the input data or modify in place.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,), optional
        Names of features seen during fit.

    scaler_ : StandardScaler or None
        The fitted StandardScaler if standardize=True, None otherwise.

    pca_ : PCA
        The fitted PCA transformer.

    component_stds_ : ndarray of shape (n_features_in_,)
        Standard deviations of PCA components before unit variance scaling.

    transformation_matrix_ : ndarray of shape (n_features_in_, n_features_in_)
        The transformation matrix W such that U = X_processed @ W.

    inverse_transformation_matrix_ : ndarray of shape (n_features_in_, n_features_in_)
        The inverse transformation matrix to convert back to processed space.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Generate correlated data
    >>> X, _ = make_classification(n_samples=100, n_features=4, n_redundant=2, random_state=42)
    >>>
    >>> # Fit and transform
    >>> decorrelator = Decorrelator(standardize=True)
    >>> U = decorrelator.fit_transform(X)
    >>>
    >>> # Verify decorrelation
    >>> cov_U = np.cov(U.T)
    >>> print("Covariance matrix (should be close to identity):")
    >>> print(np.round(cov_U, 3))
    >>>
    >>> # Transform back
    >>> X_reconstructed = decorrelator.inverse_transform(U)
    >>> reconstruction_error = np.mean((X - X_reconstructed)**2)
    >>> print(f"Reconstruction error: {reconstruction_error:.6f}")
    """

    def __init__(self):
        super(Decorrelator, self).__init__(whiten=True)


class Decorrelator(BaseEstimator, TransformerMixin):
    """
    Decorrelator Transformer using PCA to produce uncorrelated, unit variance components.

    This transformer converts input data X into decorrelated data U where:
    - All components are uncorrelated (covariance matrix is diagonal/identity)
    - All components have unit variance
    - The transformation preserves all information (no dimensionality reduction)

    This is particularly useful as a preprocessing step for algorithms that assume
    uncorrelated features or for ForeCA analysis.

    Parameters
    ----------
    standardize : bool, default=True
        Whether to standardize the input data before PCA transformation.

    copy : bool, default=True
        Whether to make a copy of the input data or modify in place.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,), optional
        Names of features seen during fit.

    scaler_ : StandardScaler or None
        The fitted StandardScaler if standardize=True, None otherwise.

    pca_ : PCA
        The fitted PCA transformer.

    component_stds_ : ndarray of shape (n_features_in_,)
        Standard deviations of PCA components before unit variance scaling.

    transformation_matrix_ : ndarray of shape (n_features_in_, n_features_in_)
        The transformation matrix W such that U = X_processed @ W.

    inverse_transformation_matrix_ : ndarray of shape (n_features_in_, n_features_in_)
        The inverse transformation matrix to convert back to processed space.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Generate correlated data
    >>> X, _ = make_classification(n_samples=100, n_features=4, n_redundant=2, random_state=42)
    >>>
    >>> # Fit and transform
    >>> decorrelator = Decorrelator(standardize=True)
    >>> U = decorrelator.fit_transform(X)
    >>>
    >>> # Verify decorrelation
    >>> cov_U = np.cov(U.T)
    >>> print("Covariance matrix (should be close to identity):")
    >>> print(np.round(cov_U, 3))
    >>>
    >>> # Transform back
    >>> X_reconstructed = decorrelator.inverse_transform(U)
    >>> reconstruction_error = np.mean((X - X_reconstructed)**2)
    >>> print(f"Reconstruction error: {reconstruction_error:.6f}")
    """

    def __init__(self, standardize: bool = True, copy: bool = True):
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        """
        Fit the Decorrelator transformer to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), optional
            Target values (ignored, present for API consistency).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Input validation
        X = check_array(X, copy=self.copy, dtype=np.float64)

        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for decorrelation")

        # Store input info
        self.n_features_in_ = X.shape[1]

        # Step 1: Fit standardization if requested
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_processed = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            # Center the data (PCA requires centered data)
            self._data_mean = np.mean(X, axis=0)
            X_processed = X - self._data_mean

        # Step 2: Fit PCA with all components
        self.pca_ = PCA(n_components=self.n_features_in_)
        U_raw = self.pca_.fit_transform(X_processed)

        # Step 3: Compute scaling to achieve unit variance
        self.component_stds_ = np.std(U_raw, axis=0, ddof=1)

        # Avoid division by zero for constant components
        self.component_stds_ = np.where(
            self.component_stds_ < 1e-12, 1.0, self.component_stds_
        )

        # Step 4: Compute transformation matrices
        # U = X_processed @ W
        V = self.pca_.components_.T  # PCA loadings: (n_features, n_features)
        S_inv = np.diag(1.0 / self.component_stds_)  # Scaling to unit variance

        self.transformation_matrix_ = V @ S_inv

        # Inverse transformation: X_processed = U @ W_inv
        S = np.diag(self.component_stds_)
        self.inverse_transformation_matrix_ = S @ V.T

        return self

    def transform(self, X):
        """
        Transform data to decorrelated space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        U : ndarray of shape (n_samples, n_features)
            Decorrelated data with uncorrelated, unit variance components.
        """
        # Check if fitted
        if not hasattr(self, "pca_"):
            raise ValueError("Decorrelator must be fitted before transforming")

        # Input validation
        X = check_array(X, copy=self.copy, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Decorrelator was fitted with {self.n_features_in_} features"
            )

        # Step 1: Apply same preprocessing as during fit
        if self.standardize:
            X_processed = self.scaler_.transform(X)
        else:
            X_processed = X - self._data_mean

        # Step 2: Apply the full transformation
        U = X_processed @ self.transformation_matrix_

        return U

    def inverse_transform(self, U):
        """
        Transform decorrelated data back to original space.

        Parameters
        ----------
        U : array-like of shape (n_samples, n_features)
            Decorrelated data to transform back.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Data transformed back to original space.
        """
        # Check if fitted
        if not hasattr(self, "pca_"):
            raise ValueError("Decorrelator must be fitted before inverse transforming")

        # Input validation
        U = check_array(U, copy=self.copy, dtype=np.float64)

        if U.shape[1] != self.n_features_in_:
            raise ValueError(
                f"U has {U.shape[1]} features, but Decorrelator expects {self.n_features_in_} features"
            )

        # Step 1: Transform back to processed space
        X_processed = U @ self.inverse_transformation_matrix_

        # Step 2: Apply inverse preprocessing
        if self.standardize:
            X_reconstructed = self.scaler_.inverse_transform(X_processed)
        else:
            X_reconstructed = X_processed + self._data_mean

        return X_reconstructed

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names.
        """
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(self.n_features_in_)]

        return np.array(
            [f"decorrelated_{name}" for name in input_features], dtype=object
        )
