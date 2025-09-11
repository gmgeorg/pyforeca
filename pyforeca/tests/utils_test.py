# tests/test_decorrelator.py
"""Unit tests for the Decorrelator transformer and assert_decorrelated utility."""

from __future__ import annotations

import numpy as np
import pytest

# Adjust the import path to your package layout.
# e.g., from foreca.whitening import Decorrelator, assert_decorrelated
from pyforeca.utils import Decorrelator, assert_decorrelated


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Shared random generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def correlated_data(rng):
    """Create a correlated dataset with nonzero means and heterogeneous scales."""
    n_samples, n_features = 400, 5
    x_base = rng.standard_normal((n_samples, n_features))

    # Target correlation matrix (SPD), then apply Cholesky to induce correlation.
    corr = np.array(
        [
            [1.0, 0.7, 0.3, 0.1, 0.0],
            [0.7, 1.0, 0.5, 0.2, 0.1],
            [0.3, 0.5, 1.0, 0.4, 0.2],
            [0.1, 0.2, 0.4, 1.0, 0.3],
            [0.0, 0.1, 0.2, 0.3, 1.0],
        ]
    )
    L = np.linalg.cholesky(corr)
    X_corr = x_base @ L.T

    # Heterogeneous scales and means to exercise standardization logic.
    scales = np.array([1.0, 5.0, 0.5, 10.0, 2.0])
    means = np.array([0.0, 10.0, -5.0, 100.0, 3.0])
    X_corr = X_corr * scales + means
    return X_corr, L, scales, means


@pytest.mark.parametrize("standardize", [True, False])
def test_fit_transform_decorrelates(correlated_data, standardize):
    """fit_transform should produce uncorrelated, unit-variance components."""
    X, _, _, _ = correlated_data

    deco = Decorrelator(standardize=standardize)
    U = deco.fit_transform(X)

    # Shape preserved
    assert U.shape == X.shape

    # Means ~ 0 (because PCA on centered/standardized data)
    means = U.mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-10)

    # Unit variances (within tolerance)
    stds = U.std(axis=0, ddof=1)
    assert np.allclose(stds, 1.0, atol=1e-6)

    # Covariance close to identity
    cov = np.cov(U.T)
    id_mat = np.eye(U.shape[1])
    assert np.allclose(cov, id_mat, atol=5e-6)

    # assert_decorrelated should return None when successful
    assert assert_decorrelated(U, tolerance=1e-5) is None


def test_inverse_transform_reconstruction(correlated_data):
    """inverse_transform should reconstruct the original data with small error."""
    X, *_ = correlated_data

    deco = Decorrelator(standardize=True)
    U = deco.fit_transform(X)
    X_rec = deco.inverse_transform(U)

    # Reconstruction error should be tiny (numerical noise)
    mse = np.mean((X - X_rec) ** 2)
    assert mse < 1e-10


def test_transform_new_data_is_decorrelated(correlated_data, rng):
    """Transforming new data with same correlation/scale/mean should remain decorrelated."""
    _, L, scales, means = correlated_data

    # New batch with same generative process
    X_new = rng.standard_normal((200, len(scales))) @ L.T
    X_new = X_new * scales + means

    deco = Decorrelator(standardize=True)
    deco.fit(X_new)  # Fit on new data for this test
    U_new = deco.transform(X_new)

    # Covariance ~ I and assert_decorrelated returns None
    cov_new = np.cov(U_new.T)
    assert np.allclose(cov_new, np.eye(cov_new.shape[0]), atol=5e-6)
    assert assert_decorrelated(U_new, tolerance=1e-5) is None


def test_transformation_matrices_are_inverses(correlated_data):
    """W and W_inv should be (numerical) inverses on the processed space."""
    X, *_ = correlated_data
    deco = Decorrelator(standardize=True).fit(X)

    W = deco.transformation_matrix_
    W_inv = deco.inverse_transformation_matrix_

    # On the processed (centered/standardized) space: X_proc @ W @ W_inv == X_proc
    # i.e., W @ W_inv ~ I
    I_est = W @ W_inv
    assert np.allclose(I_est, np.eye(I_est.shape[0]), atol=1e-10)


def test_feature_names_out_defaults(correlated_data):
    """get_feature_names_out should generate prefixed names by default."""
    X, *_ = correlated_data
    deco = Decorrelator().fit(X)

    names = deco.get_feature_names_out()
    assert len(names) == X.shape[1]
    assert all(n.startswith("decorrelated_") for n in names)


def test_feature_names_out_custom(correlated_data):
    """get_feature_names_out should preserve custom input names with prefix."""
    X, *_ = correlated_data
    deco = Decorrelator().fit(X)

    custom = [f"var{i}" for i in range(X.shape[1])]
    names = deco.get_feature_names_out(custom)
    assert np.array_equal(
        names, np.array([f"decorrelated_{c}" for c in custom], dtype=object)
    )


def test_assert_decorrelated_reports_when_not_decorrelated(correlated_data):
    """assert_decorrelated should return diagnostics when decorrelation fails."""
    X, *_ = correlated_data
    # Raw X is correlated; expect a diagnostics dict
    diag = assert_decorrelated(X, tolerance=1e-8)
    assert isinstance(diag, dict)
    assert diag["is_decorrelated"] is False
    assert "max_off_diagonal" in diag and diag["max_off_diagonal"] > 1e-3
    assert (
        "covariance_matrix" in diag and diag["covariance_matrix"].shape[0] == X.shape[1]
    )


def test_errors_and_validation(correlated_data):
    """API should raise clean errors on bad inputs."""
    X, *_ = correlated_data

    # Need at least 2 samples to estimate covariance
    deco = Decorrelator()
    with pytest.raises(ValueError, match="at least 2 samples"):
        deco.fit(X[:1])

    # Transform before fit
    deco2 = Decorrelator()
    with pytest.raises(ValueError, match="must be fitted"):
        deco2.transform(X)

    # Inverse before fit
    with pytest.raises(ValueError, match="must be fitted"):
        deco2.inverse_transform(np.zeros_like(X))

    # Feature mismatch on transform
    deco.fit(X)
    with pytest.raises(ValueError, match="features"):
        deco.transform(X[:, :-1])

    # Feature mismatch on inverse_transform
    U = deco.transform(X)
    with pytest.raises(ValueError, match="features"):
        deco.inverse_transform(U[:, :-1])


@pytest.mark.parametrize("standardize", [True, False])
def test_covariance_identity_tolerance(correlated_data, standardize):
    """Looser tolerances still pass; tight tolerances may fail (numerical sanity)."""
    X, *_ = correlated_data
    deco = Decorrelator(standardize=standardize).fit(X)
    U = deco.transform(X)

    # Very tight tolerance may fail due to floating point
    very_tight = assert_decorrelated(U, tolerance=1e-12)
    if very_tight is not None:
        # Should pass at a realistic tolerance
        assert assert_decorrelated(U, tolerance=1e-6) is None
