"""
Pytest unit tests for spectral quadratic form functions.

Run with: pytest test_spectral_quadratic.py -v
"""

import numpy as np
import pytest

# Adjust this import to your package layout.
# e.g. from foreca.algorithms import optimizer.initialize_weightvector
from pyforeca import optimizer


def _get_test_data():
    # Generate synthetic multivariate time series
    rng = np.random.RandomState(42)
    n_samples = 500
    t = np.linspace(0, 10, n_samples)

    # Create signals with different levels of predictability
    # Highly predictable: sine waves
    x1 = np.sin(2 * np.pi * 0.5 * t) + 0.1 * rng.randn(n_samples)
    x2 = np.cos(2 * np.pi * 0.3 * t) + 0.1 * rng.randn(n_samples)

    # Less predictable: AR process + noise
    x3 = np.random.randn(n_samples)
    for i in range(1, n_samples):
        x3[i] = 0.7 * x3[i - 1] + 0.3 * rng.randn()

    # Unpredictable: white noise
    x4 = rng.randn(n_samples)

    # Combine into multivariate series
    X = np.column_stack([x1, x2, x3, x4])
    return X


# Fixtures for test data
@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def simple_real_matrix():
    """Simple 2x2 real matrix for testing."""
    return np.array([[1, 2], [3, 4]], dtype=float)


@pytest.fixture
def simple_real_vector():
    """Simple real vector for testing."""
    return np.array([1, -1], dtype=float)


@pytest.fixture
def hermitian_matrix():
    """Simple Hermitian matrix for testing."""
    return np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]], dtype=complex)


@pytest.fixture
def complex_vector():
    """Simple complex vector for testing."""
    return np.array([1 + 1j, 1 - 1j], dtype=complex)


@pytest.fixture
def positive_definite_mvspectrum(random_seed):
    """Generate positive definite multivariate spectral density matrices."""
    n_freqs, n_features = 32, 3
    mvspec = np.zeros((n_freqs, n_features, n_features), dtype=complex)

    for freq_idx in range(n_freqs):
        # Generate random Hermitian matrix
        A = np.random.randn(n_features, n_features) + 1j * np.random.randn(
            n_features, n_features
        )
        A_hermitian = (A + np.conj(A.T)) / 2
        # Ensure positive definiteness
        mvspec[freq_idx] = A_hermitian + 2 * np.eye(n_features)

    return mvspec


def test_average_initialization_unit_norm_and_shape():
    d = 7
    w = optimizer.initialize_weightvector(
        n_series=d, init_method=optimizer.InitMethod.AVERAGE
    )
    assert w.shape == (d,)
    np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-12)
    # All entries equal
    np.testing.assert_allclose(w, w[0])
    # Each entry should be 1/sqrt(d)
    np.testing.assert_allclose(w[0], 1.0 / np.sqrt(d), atol=1e-12)


def test_random_initialization_deterministic_and_unit_norm():
    d = 10
    w1 = optimizer.initialize_weightvector(
        n_series=d, init_method=optimizer.InitMethod.NORMAL, seed=123
    )
    w2 = optimizer.initialize_weightvector(
        n_series=d, init_method=optimizer.InitMethod.NORMAL, seed=123
    )
    w3 = optimizer.initialize_weightvector(
        n_series=d, init_method=optimizer.InitMethod.NORMAL, seed=124
    )

    # Same seed -> identical vector
    np.testing.assert_allclose(w1, w2, atol=0)
    # Different seed -> likely different vector
    assert not np.allclose(w1, w3)

    # Unit norm and shape
    assert w1.shape == (d,)
    np.testing.assert_allclose(np.linalg.norm(w1), 1.0, atol=1e-12)


def test_pca_initialization_uses_vector_and_normalizes():
    x = _get_test_data()
    w = optimizer.initialize_weightvector(x, init_method=optimizer.InitMethod.PCA)

    # Unit norm and correct shape
    assert w.shape == (x.shape[1],)
    np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-12)


def test_invalid_method_raises():
    with pytest.raises(TypeError, match="init_method must be of type InitMethod."):
        optimizer.initialize_weightvector(n_series=10, init_method="nope")  # type: ignore[arg-type]


def test_dimension_validation():
    with pytest.raises(ValueError, match="'n_series' must be > 0. Got 0"):
        optimizer.initialize_weightvector(
            n_series=0, init_method=optimizer.InitMethod.UNIFORM
        )
    with pytest.raises(ValueError, match="'n_series' must be > 0. Got -3"):
        optimizer.initialize_weightvector(
            n_series=-3, init_method=optimizer.InitMethod.NORMAL
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
