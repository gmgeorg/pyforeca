"""Module for testing foreca."""

import numpy as np
import pytest

from pyforeca import optimizer
from pyforeca.base import ForeCA
from pyforeca.datasets import simulations


def _ar1(
    n: int, phi: float, rng: np.random.Generator, sigma: float = 1.0
) -> np.ndarray:
    e = rng.normal(scale=sigma, size=n)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("method", ["welch", "periodogram"])
def test_fit_transform_shapes_and_attrs(method: str):
    rng = np.random.default_rng(0)
    n, d, k = 512, 3, 2
    X = rng.standard_normal((n, d))

    foreca = ForeCA(
        n_components=k,
        spectrum_method=method,
        nperseg=128,
        init_method=optimizer.InitMethod.NORMAL,
        max_iter=50,
        tol=1e-6,
    )
    foreca.fit(X)
    Y = foreca.transform(X)

    assert Y.shape == (n, k)
    assert foreca.components_whitened_.shape == (d, k)
    assert foreca.components_.shape == (d, k)
    assert foreca.omegas_.shape == (k,)
    assert np.all((foreca.omegas_ >= 0.0) & (foreca.omegas_ <= 1.0))
    # explained ratio sums to ~1
    assert np.isclose(foreca.explained_forecastability_ratio_.sum(), 1.0, atol=1e-6)


def test_scores_are_uncorrelated_after_transform():
    rng = np.random.default_rng(1)
    n, d, k = 1024, 4, 3
    X = rng.standard_normal((n, d))

    foreca = ForeCA(
        n_components=k,
        spectrum_method="welch",
        nperseg=128,
        init_method=optimizer.InitMethod.NORMAL,
        max_iter=50,
        tol=1e-6,
    ).fit(X)

    Y = foreca.transform(X)
    covY = np.cov(Y.T)
    # off-diagonals close to 0; diagonals close to 1
    offdiag = np.max(np.abs(covY - np.diag(np.diag(covY))))
    assert offdiag < 0.05
    np.testing.assert_allclose(np.diag(covY), 1.0, atol=0.1)


def test_ar1_forecastability_order_and_slowness():
    """On independent AR(1) channels, first component should be the most forecastable."""
    rng = np.random.default_rng(2)
    n = 4096
    X = np.column_stack(
        [
            _ar1(n, 0.2, rng),
            _ar1(n, 0.6, rng),
            _ar1(n, 0.9, rng),
        ]
    )

    foreca = ForeCA(
        n_components=2,
        spectrum_method="welch",
        nperseg=256,
        init_method=optimizer.InitMethod.NORMAL,
        max_iter=120,
        tol=1e-7,
    ).fit(X)

    # omegas sorted decreasing by class logic; assert strict ordering and a decent magnitude
    assert foreca.omegas_[0] >= foreca.omegas_[1]
    assert foreca.omegas_[0] > 0.25

    # The first score should have larger lag-1 autocorrelation than the second
    Y = foreca.transform(X)

    def acf1(y):
        return float(np.corrcoef(y[1:], y[:-1])[0, 1])

    assert acf1(Y[:, 0]) > acf1(Y[:, 1])


def test_invalid_n_components_raises():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 3))
    with pytest.raises(ValueError, match="n_components must be in"):
        ForeCA(n_components=5).fit(X)


def test_transform_before_fit_raises():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((100, 2))
    with pytest.raises(ValueError, match="must be fitted"):
        ForeCA(n_components=1).transform(X)


def test_readme_simulation():
    _, _, observed = simulations.gen_toy_data(1000)

    # Apply ForeCA
    foreca = ForeCA(n_components=3, spectrum_method="welch")
    # forecastable components found by ForeCA
    forecs = foreca.fit_transform(observed)

    assert forecs.index.equals(observed.index)
