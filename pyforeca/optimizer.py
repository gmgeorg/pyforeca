"""
Functions for computing quadratic forms on multivariate spectral density matrices.

These functions are equivalent to the R ForeCA functions for computing spectra
of linear combinations from multivariate spectral density matrices.
"""

import enum

import numpy as np

from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from pyforeca import linalg, sfa


class InitMethod(enum.Enum):
    """Enum for initialize method."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    CAUCHY = "cauchy"
    MAX = "max"
    AVERAGE = "average"
    PCA = "pca"
    PCA_LARGE = "pca_large"
    PCA_SMALL = "pca_small"
    SFA = "sfa"
    SFA_SLOW = "sfa_slow"
    SFA_FAST = "sfa_fast"


def _unit(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return x / ||x|| with guard for tiny norms."""
    x = np.asarray(x, dtype=float)
    nrm = linalg.l2_norm(x)
    if not np.isfinite(nrm) or nrm < 1e-12:
        raise ValueError("Initialization vector has near-zero norm.")

    x = x / nrm
    return x


def _spectral_entropy_per_channel(
    f_U: NDArray[np.floating | np.complexfloating],
) -> NDArray[np.floating]:
    """Discrete spectral entropy for each univariate channel from a (possibly multivariate) spectrum.

    Accepts shapes:
      - (n_freq, n_series) univariate spectra per series
      - (n_freq, n_series, n_series) cross-spectral matrices (uses diagonal)

    Returns
    -------
    H : (n_series,) array
        Shannon entropy of normalized discrete spectrum for each channel.
    """
    f = np.asarray(f_U)
    if f.ndim == 3:
        # Take diagonal power spectrum for each series
        # Ensure real nonnegative power
        f = np.real(np.einsum("fii->fi", f))
    elif f.ndim != 2:
        raise ValueError("f_U must be 2D (freq×series) or 3D (freq×series×series).")

    # Normalize each column to a probability mass over frequency bins
    power = np.clip(f, 0.0, np.inf)
    col_sums = power.sum(axis=0, keepdims=True)
    # Avoid division by zero: if a column is all zeros, make it uniform to keep H well-defined.
    col_sums = np.where(col_sums < 1e-16, 1.0, col_sums)
    p = power / col_sums

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    H = -(p * logp).sum(axis=0)
    return H


def _argmax_forecastable_channel(
    f_U: NDArray[np.floating | np.complexfloating],
) -> int:
    """Heuristic: pick the channel with lowest spectral entropy given `f_U`."""
    H = _spectral_entropy_per_channel(f_U)
    return int(np.argmin(H))


def initialize_weightvector(
    U: NDArray[np.floating] | None = None,
    f_U: NDArray[np.floating | np.complexfloating] | None = None,
    n_series: int | None = None,
    init_method: InitMethod = InitMethod.SFA,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Initialize a weight vector ``w0`` for ForeCA optimization.

    Parameters
    ----------
    U : ndarray of shape (n_samples, n_features), optional
        (Whitened) data matrix. Required by methods that look at data
        statistics: "max", "PCA", "PCA.large", "PCA.small", "SFA*", unless
        you supply an explicit ``pca_components`` for "PCA".
    f_U : ndarray, optional
        Spectrum of U. Either shape (n_freq, n_features) with univariate
        spectra per channel, or (n_freq, n_features, n_features) with
        cross-spectra (diagonals are used). If provided and ``method="max"``,
        the channel with **lowest spectral entropy** is selected.
    n_series : int, optional
        Number of features (dimension of w). If None, inferred from U or
        pca_components.
    init_method : see InitMethod.
    seed : int or None, default=None
        Random seed for reproducibility for random methods.
    pca_components : ndarray of shape (n_features,), optional
        If given and method in {"PCA", "PCA.large"}, this vector is used
        instead of fitting PCA on U.

    Returns
    -------
    w0 : ndarray of shape (n_features,)
        Unit-norm initialization vector.

    Raises
    ------
    ValueError
        If inputs are inconsistent for the chosen method.
    """
    if not isinstance(init_method, InitMethod):
        raise TypeError("init_method must be of type InitMethod.")

    rng = check_random_state(seed)

    if n_series is None:
        if U is not None:
            n_series = U.shape[1]
        elif f_U is not None:
            n_series = f_U.shape[1] if f_U.ndim == 2 else f_U.shape[1]
        else:
            raise ValueError(
                "n_series could not be inferred; provide U, f_U, or n_series."
            )

    assert isinstance(n_series, int)

    if n_series <= 0:
        raise ValueError(f"'n_series' must be > 0. Got {n_series}")

    # --- Random families ---
    if init_method == InitMethod.NORMAL:
        return _unit(rng.normal(size=n_series))
    if init_method == InitMethod.UNIFORM:
        return _unit(rng.uniform(low=-1.0, high=1.0, size=n_series))
    if init_method == InitMethod.CAUCHY:
        v = rng.standard_cauchy(size=n_series)
        # Clip extremes to avoid overflow; normalization handles the rest.
        v = np.clip(v, -1e3, 1e3)
        return _unit(v)

    # --- Channel-based heuristic ("max") ---
    if init_method == InitMethod.MAX:
        idx = _argmax_forecastable_channel(f_U)
        e = np.zeros(n_series)
        e[idx] = 1.0
        return e  # already unit

    if init_method == InitMethod.AVERAGE:
        e = np.ones(n_series)
        return _unit(e)

    # --- PCA families ---
    if init_method in [InitMethod.PCA, InitMethod.PCA_LARGE, InitMethod.PCA_SMALL]:
        if U is None:
            raise ValueError(
                "U is required for PCA-based initialization if pca_components is not provided."
            )

        # Center (PCA expects centered data). If U is whitened, this is cheap.
        Uc = U - U.mean(axis=0, keepdims=True)
        mod_pca = PCA(n_components=n_series, svd_solver="full")
        mod_pca.fit(Uc)

        if init_method in [InitMethod.PCA, InitMethod.PCA_LARGE]:
            w = mod_pca.components_[0]  # first loading (largest variance)
        else:  # "PCA.small"
            w = mod_pca.components_[-1]  # smallest variance
        return _unit(w)

    # --- SFA families ---
    if init_method in [InitMethod.SFA, InitMethod.SFA_SLOW, InitMethod.SFA_FAST]:
        if U is None:
            raise ValueError("U is required for SFA-based initialization.")

        Uc = U - U.mean(axis=0, keepdims=True)

        mod_sfa = sfa.SFA(n_components=n_series)
        mod_sfa.fit(Uc)

        if init_method in [InitMethod.SFA, InitMethod.SFA_SLOW]:
            w = mod_sfa.components_[0]  # first loading (slowest signal)
        else:  # "SFA_FAST"
            w = mod_sfa.components_[-1]  # last loading (fastest signal)
        return w

    raise ValueError(f"Unknown initialization method: {init_method!r}")


def em_one_basic(
    mvspec,  # MVSpectrum of whitened U
    U,  # (n_samples, n_features) (only for init methods that use data)
    init_method,  # your optimizer.InitMethod
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    epsilon: float = 1e-12,
):
    """EM step for one weight vector."""
    vectors: list[np.ndarray] = []
    psds = []
    sigmas = []

    # init w0
    w0 = initialize_weightvector(U=U, init_method=init_method)
    w0 = w0 / np.linalg.norm(w0)
    vectors.append(w0)

    for _ in range(max_iter):
        # spectrum of current projection
        uy = mvspec.linear_combo(vectors[-1], check_positive=False)
        p = np.maximum(uy.as_pmf().psd, float(epsilon))  # pmf over freqs

        # build surrogate Sbar via wcov with weights = -log p
        Sbar = mvspec.wcov(
            frequency_weights=-np.log(p)
        )  # scaling won’t affect eigenvector
        sigmas.append(Sbar)

        # smallest-eigenvector update
        w_new = linalg.eigenvector_symmetric(Sbar, min=True)
        vectors.append(w_new)

        # sign-invariant convergence: angle metric
        prev, curr = vectors[-2], vectors[-1]
        delta = 1.0 - abs(float(prev @ curr))
        if delta <= tol:
            break

        # keep PSD history (optional)
        psds.append(uy.as_pmf())

    # final stats
    uy_final = mvspec.linear_combo(vectors[-1], check_positive=True).as_pmf()
    H = uy_final.entropy(normalize_by_max=True)  # in [0,1]
    omega = 1.0 - H

    return {
        "w": vectors[-1],
        "vectors": vectors,
        "psds": psds + [uy_final],
        "Sbars": sigmas,
        "entropy_norm": float(H),
        "omega": float(omega),
    }
