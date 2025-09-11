# tests/test_mvspectrum_wcov.py
from __future__ import annotations

import numpy as np
import pytest

from pyforeca.spectral import (
    MVSpectrumEstimator,
)
from pyforeca.spectrum_types import MVSpectrum, UnivariateSpectrum
from pyforeca.utils import Decorrelator, assert_decorrelated


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture()
def correlated_data(rng):
    """Create zero-mean correlated Gaussian data with heterogeneous scales."""
    n_samples, n_features = 4096, 4
    Z = rng.standard_normal((n_samples, n_features))

    # Target correlation (SPD)
    corr = np.array(
        [
            [1.00, 0.70, 0.30, 0.10],
            [0.70, 1.00, 0.50, 0.20],
            [0.30, 0.50, 1.00, 0.40],
            [0.10, 0.20, 0.40, 1.00],
        ]
    )
    L = np.linalg.cholesky(corr)
    X = Z @ L.T

    # Add different scales and means (to exercise preprocessing)
    scales = np.array([1.0, 5.0, 0.5, 10.0])
    means = np.array([0.0, 10.0, -5.0, 100.0])
    X = X * scales + means
    return X


def test_wcov_is_identity_after_decorrelation(correlated_data):
    """After decorrelation, wcov() from multivariate Welch should be ~ I."""
    X = correlated_data

    # 1) Decorrelate (unit variance, uncorrelated components)
    deco = Decorrelator(standardize=True)
    U = deco.fit_transform(X)

    # Sanity: check decorrelated
    assert assert_decorrelated(U, tolerance=1e-5) is None

    # 2) Multivariate Welch spectrum (one-sided, SciPy)
    #    Use a reasonably sized segment for stable PSD estimates.
    mvspec_estimator = MVSpectrumEstimator(method="periodogram")
    mv: MVSpectrum = mvspec_estimator.fit(U).mvspectrum_

    # 3) Weighted covariance via spectral integral
    Sigma_hat = mv.wcov()

    # 4) Should be close to identity
    id_mat = np.eye(U.shape[1])
    # absolute tolerance chosen to allow small spectral estimation error
    (
        np.testing.assert_allclose(Sigma_hat, id_mat, atol=5e-2),
        (f"wcov deviates from I:\n{Sigma_hat}"),
    )


def test_welch_wcov_matches_empirical_covariance_before_decorrelation(correlated_data):
    """(Bonus) wcov should approximate the empirical covariance on raw X."""
    X = correlated_data
    # Center for fair comparison (Welch detrends segments by default)
    Xc = X - X.mean(axis=0, keepdims=True)

    mvspec_estimator = MVSpectrumEstimator(method="welch")
    mv_X = mvspec_estimator.fit(Xc).mvspectrum_
    Sigma_welch = mv_X.wcov()
    Sigma_emp = np.cov(Xc, rowvar=False)

    # PSD integration and sample covariance should roughly agree
    assert np.allclose(Sigma_welch, Sigma_emp, rtol=0.1), (
        f"Welch wcov and sample cov differ:\nwcov=\n{Sigma_welch}\nemp=\n{Sigma_emp}"
    )


def test_periodogram_wcov_matches_empirical_covariance_before_decorrelation(
    correlated_data,
):
    """(Bonus) wcov should approximate the empirical covariance on raw X."""
    X = correlated_data
    # Center for fair comparison (Welch detrends segments by default)
    Xc = X - X.mean(axis=0, keepdims=True)

    mvspec_estimator = MVSpectrumEstimator(method="periodogram")
    mv_X = mvspec_estimator.fit(Xc).mvspectrum_
    Sigma_welch = mv_X.wcov()
    Sigma_emp = np.cov(Xc, rowvar=False)

    # PSD integration and sample covariance should roughly agree
    assert np.allclose(Sigma_welch, Sigma_emp, atol=0.15), (
        f"Welch wcov and sample cov differ:\nwcov=\n{Sigma_welch}\nemp=\n{Sigma_emp}"
    )


@pytest.fixture()
def mvspec_welch(correlated_data) -> MVSpectrum:
    """Welch MV spectrum for realistic cross-spectra."""
    X = correlated_data
    mvspec_estimator = MVSpectrumEstimator(method="welch", nperseg=256)
    mv: MVSpectrum = mvspec_estimator.fit(X).mvspectrum_
    return mv


def test_linear_combo_matches_manual_quadratic_form(mvspec_welch, rng):
    """βᴴ F(ω) β computed by method equals manual einsum across all frequencies."""
    mv: MVSpectrum = mvspec_welch
    K = mv.n_series
    beta = rng.standard_normal(K)

    uy: UnivariateSpectrum = mv.linear_combo(beta)
    assert isinstance(uy, UnivariateSpectrum)
    assert uy.freqs.shape == (mv.n_freqs,)
    assert uy.psd.shape == (mv.n_freqs,)

    # Manual quadratic form (real part)
    Fy_manual = np.real(np.einsum("fij,j,i->f", mv.csd, beta, beta))
    assert np.allclose(uy.psd, Fy_manual, atol=1e-10)


def test_linear_combo_basis_vector_equals_diagonal(mvspec_welch):
    """For β = e_i, the result equals the i-th diagonal auto-spectrum."""
    mv: MVSpectrum = mvspec_welch
    diag = np.real(np.diagonal(mv.csd, axis1=1, axis2=2))  # (F, K)
    K = mv.n_series

    for i in range(K):
        e = np.zeros(K)
        e[i] = 1.0
        uy_i = mv.linear_combo(e)
        assert np.allclose(uy_i.psd, diag[:, i], atol=1e-12)


def test_linear_combo_nonnegativity_for_psd_spectra(mvspec_welch):
    """Welch spectra are PSD ⇒ quadratic form should be >= 0 up to tiny numeric wiggles."""
    mv: MVSpectrum = mvspec_welch
    K = mv.n_series
    for i in range(3):  # a few random β's
        rng = np.random.default_rng(100 + i)
        beta = rng.standard_normal(K)
        uy = mv.linear_combo(beta)
        assert uy.psd.min() > -1e-12  # allow tiny negative due to fp


def test_linear_combo_clips_negatives_when_requested(rng):
    """If a frequency's quadratic form is (slightly) negative, check_positive=True clips to 0."""
    F, K = 8, 3
    freqs = np.linspace(0.01, 0.49, F)

    # Build PSD csd for all freqs, then inject one slightly negative definite slice
    csd = np.zeros((F, K, K), dtype=np.complex128)
    for f in range(F):
        A = rng.standard_normal((K, K))
        csd[f] = A @ A.T  # real symmetric PSD

    # Make the first frequency slightly negative definite
    csd[0] = -1e-8 * np.eye(K)

    mv = MVSpectrum(freqs=freqs, csd=csd, normalized="none")
    beta = np.array([0.7, -0.2, 0.1])

    uy_clip = mv.linear_combo(beta, check_positive=True)
    # First element should be clipped to zero in the clipped version and be negative in raw
    # assert uy_raw[0] < 0
    assert uy_clip.psd[0] == 0.0


@pytest.fixture()
def uniform_grid_psd():
    """Build an MVSpectrum with constant PSD on a uniform grid so integrals are analytic."""
    F, K = 256, 3
    freqs = np.linspace(0.01, 0.5, F)  # uniform positive grid
    # Constant PSD across frequencies: csd[f] = A A^T (real PSD)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((K, K))
    Sigma0 = A @ A.T  # (K,K) SPD
    csd = np.repeat(Sigma0[None, :, :], F, axis=0).astype(np.complex128)
    mv = MVSpectrum(freqs=freqs, csd=csd, normalized="none")
    return mv, Sigma0


def test_wcov_no_weights_matches_sum_times_bandwidth(uniform_grid_psd):
    mv, Sigma0 = uniform_grid_psd
    # Integral over [f_min, f_max] of a constant PSD is length * Sigma0
    bandwidth = mv.freqs[-1] - mv.freqs[0]
    Sigma = mv.wcov()
    np.testing.assert_allclose(Sigma, 2 * bandwidth * Sigma0, atol=0.02)


def test_wcov_with_indicator_weights_limits_band(uniform_grid_psd):
    mv, Sigma0 = uniform_grid_psd
    f = mv.freqs
    F = f.size

    # Indicator on upper half of the band
    w = np.zeros(F)
    w[f >= (f[0] + f[-1]) / 2] = 1.0

    Sigma_full = mv.wcov()
    Sigma_half = mv.wcov(frequency_weights=w)

    # The ratio should be the ratio of covered bandwidths for constant PSD
    full_bw = f[-1] - f[0]
    half_bw = f[-1] - (f[0] + f[-1]) / 2
    expected = (half_bw / full_bw) * Sigma_full
    np.testing.assert_allclose(Sigma_half, expected, atol=0.01)


def test_wcov_trapz_and_sum_agree_on_uniform_grid(uniform_grid_psd):
    mv, _ = uniform_grid_psd
    Sigma_sum = mv.wcov(method="sum")
    Sigma_trapz = mv.wcov(method="trapz")
    np.testing.assert_allclose(Sigma_sum, Sigma_trapz, atol=0.02)


def test_wcov_weight_length_validation(uniform_grid_psd):
    mv, _ = uniform_grid_psd
    with pytest.raises(ValueError, match="frequency_weights length"):
        mv.wcov(frequency_weights=np.ones(mv.n_freqs + 1))


def _rand_psd_stack(F, K, rng):
    csd = np.zeros((F, K, K), dtype=np.complex128)
    for f in range(F):
        A = rng.standard_normal((K, K))
        csd[f] = A @ A.T  # real PSD
    return csd


def test_linear_combo_basis_matches_diag():
    rng = np.random.default_rng(0)
    F, K = 64, 4
    freqs = np.linspace(0.01, 0.5, F)
    csd = _rand_psd_stack(F, K, rng)
    mv = MVSpectrum(freqs=freqs, csd=csd)
    diag = np.real(np.diagonal(csd, axis1=1, axis2=2))  # (F,K)
    for i in range(K):
        e = np.zeros(K)
        e[i] = 1.0
        uy = mv.linear_combo(e)
        assert np.allclose(uy.psd, diag[:, i], atol=1e-12)


def test_wcov_one_sided_matches_sum_df():
    rng = np.random.default_rng(1)
    F, K = 128, 3
    freqs = np.linspace(0.01, 0.5, F)  # uniform
    csd = _rand_psd_stack(F, K, rng)
    mv = MVSpectrum(freqs=freqs, csd=csd)
    df = freqs[1] - freqs[0]
    Sigma_sum = 2 * np.real(csd.sum(axis=0) * df)
    Sigma_wcov = mv.wcov(method="sum")
    np.testing.assert_allclose(Sigma_wcov, Sigma_sum, atol=1e-12)


def test_project_shape_and_psd():
    rng = np.random.default_rng(2)
    F, K = 32, 5
    freqs = np.linspace(0.01, 0.5, F)
    csd = _rand_psd_stack(F, K, rng)
    mv = MVSpectrum(freqs=freqs, csd=csd)
    # random 2D subspace
    Q, _ = np.linalg.qr(rng.standard_normal((K, 2)))
    mv2 = mv.project(Q)
    assert mv2.csd.shape == (F, 2, 2)
    # PSD should be Hermitian & ~PSD (nonnegative diagonals)
    assert np.allclose(mv2.csd, np.conj(np.swapaxes(mv2.csd, 1, 2)), atol=1e-12)
    assert np.all(np.real(np.diagonal(mv2.csd, axis1=1, axis2=2)) >= -1e-10)
