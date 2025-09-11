from __future__ import annotations

import numpy as np
import pytest

from pyforeca.sfa import SFA, _sfa_eigendecomposition


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _ar1(
    rng: np.random.Generator, n: int, phi: float, sigma: float = 1.0
) -> np.ndarray:
    e = rng.normal(scale=sigma, size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test__sfa_eigendecomposition_shapes_and_orthogonality(rng):
    n, k = 1000, 4
    X = rng.standard_normal((n, k))
    evals, V = _sfa_eigendecomposition(X)
    assert evals.shape == (k,)
    assert V.shape == (k, k)
    # C-orthonormality: V.T C V ≈ I
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False)
    M = V.T @ C @ V
    assert np.allclose(M, np.eye(k), atol=1e-8)


def test_sfa_prefers_slow_signal_alignment(rng):
    """Two-channel test: AR(1) slow vs white noise — the first SFA component aligns with slow."""
    n = 5000
    slow = _ar1(rng, n, phi=0.95, sigma=1.0)
    fast = rng.standard_normal(n)  # essentially φ≈0
    X = np.column_stack([slow, fast])

    sfa = SFA(n_components=1).fit(X)
    Z = sfa.transform(X)[:, 0]  # first slow feature

    # Up to scale/sign, Z should be highly correlated with 'slow'
    corr = np.corrcoef(Z, slow)[0, 1]
    assert abs(corr) > 0.9

    # First loading should place larger weight on slow channel than fast
    w = sfa.components_[0]  # shape (2,)
    assert abs(w[0]) > abs(w[1])


def test_sfa_full_inverse_transform_roundtrip(rng):
    """With k=K, inverse_transform should reconstruct (up to numerical noise)."""
    n, k = 2000, 3
    # Make something correlated with nonzero means
    Z = rng.standard_normal((n, k))
    A = rng.normal(size=(k, k))
    X = Z @ A.T + np.array([1.0, -2.0, 5.0])

    sfa = SFA(n_components=k).fit(X)
    Z_slow = sfa.transform(X)
    X_rec = sfa.inverse_transform(Z_slow)

    mse = float(np.mean((X - X_rec) ** 2))
    assert mse < 1e-10


def test_sfa_shapes_and_attrs(rng):
    n, k = 512, 5
    X = rng.standard_normal((n, k)) + 10.0 * np.array([0, 1, -1, 2, 3])
    sfa = SFA(n_components=3).fit(X)

    # Attributes
    assert sfa.components_.shape == (3, k)
    assert sfa.transformation_matrix_.shape == (k, 3)
    assert sfa.slowness_.shape == (3,)

    # Transform shape
    Z = sfa.transform(X)
    assert Z.shape == (n, 3)

    # Feature names
    names = sfa.get_feature_names_out()
    assert list(names) == ["sf_1", "sf_2", "sf_3"]


def test_sfa_input_validation(rng):
    X = rng.standard_normal((2, 4))
    with pytest.raises(ValueError, match="at least 3 samples"):
        SFA().fit(X)

    X = rng.standard_normal((100, 4))
    with pytest.raises(ValueError, match="n_components"):
        SFA(n_components=0).fit(X)
    with pytest.raises(ValueError, match="n_components"):
        SFA(n_components=5).fit(X)

    sfa = SFA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="fitted before transforming"):
        SFA(n_components=2).transform(X)
    with pytest.raises(ValueError, match="features"):
        sfa.transform(rng.standard_normal((10, 3)))

    Z = sfa.transform(X)
    with pytest.raises(ValueError, match="fitted before inverse"):
        SFA(n_components=2).inverse_transform(Z)
    with pytest.raises(ValueError, match="expects"):
        sfa.inverse_transform(Z[:, :1])
