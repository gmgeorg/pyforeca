"""Module for spectral estimation."""

import numpy as np
import pandas as pd
import sklearn

from numpy.typing import NDArray
from scipy import signal

from pyforeca.spectrum_types import MVSpectrum, UnivariateSpectrum

_EPS = 1e-9
_MIN_SERIES_LENGTH = 5


def univariate_psd(
    x: np.ndarray,
    method: str = "welch",
    nperseg: int | None = None,
) -> UnivariateSpectrum:
    """Estimate a one-sided univariate spectrum as a UnivariateSpectrum."""
    x = np.asarray(x, dtype=float).reshape(-1)

    if method == "welch":
        freqs, psd = signal.welch(x, nperseg=nperseg)
        psd /= 2.0
    elif method == "periodogram":
        freqs, psd = signal.periodogram(x)
    else:
        raise ValueError(f"method={method!r} not implemented")

    # Remove DC (0 Hz) to match ForeCA's positive-frequency convention for entropy
    nz_mask = freqs > 0
    freqs = freqs[nz_mask]
    psd = psd[nz_mask]

    return UnivariateSpectrum(freqs=freqs, psd=psd, normalized="none")


def univariate_spectral_entropy(
    x: np.ndarray,
    normalize_by_max: bool = False,
    method: str = "welch",
    nperseg: int | None = None,
):
    """
    Compute spectral entropy of a univariate time series.

    Parameters
    ----------
    x : array-like
        Input time series.
    method : str, default='welch'
        Spectral estimation method.
    nperseg : int, optional
        Segment length for Welch's method.

    Returns
    -------
    entropy : float
        Spectral entropy value. Lower values indicate higher forecastability.
    """
    x = np.asarray(x)
    if x.shape[0] < _MIN_SERIES_LENGTH:
        raise ValueError("Input must have at least length 5.")

    # If constant signal, then entropy is 0. (by construction/corner case of all mass at lambda = 0).
    if np.std(x) < _EPS:
        return 0.0

    upsd = univariate_psd(x, method=method, nperseg=nperseg)
    entr = upsd.entropy(normalize_by_max=normalize_by_max)
    return entr


def omega(x: np.ndarray, method: str = "welch", nperseg: int | None = None) -> float:
    """Computes forecastability mesaure Omega(x_t)."""
    upsd = univariate_psd(x, method=method, nperseg=nperseg)
    return 1.0 - upsd.entropy(normalize_by_max=True)


def _multivariate_periodogram(X: NDArray[np.floating]) -> tuple[np.ndarray, np.ndarray]:
    """Compute multivariate spectral density using periodogram."""
    n_samples, n_features = X.shape

    if n_samples < _MIN_SERIES_LENGTH:
        raise ValueError(f"Input must have at least length {_MIN_SERIES_LENGTH}.")

    # Get frequency grid
    freqs = np.fft.fftfreq(n_samples, 1.0)[: n_samples // 2 + 1]
    n_freqs = len(freqs)

    # FFT of each series
    X_fft = np.fft.fft(X, axis=0)
    X_fft = X_fft[:n_freqs]  # Take only positive frequencies

    # Compute cross-spectral density matrix
    csd_matrix = np.zeros((n_freqs, n_features, n_features), dtype=complex)

    for k in range(n_freqs):
        # Cross-spectral density matrix at frequency k
        fft_k = X_fft[k].reshape(-1, 1)
        csd_matrix[k] = np.outer(fft_k, np.conj(fft_k))

    # Normalize
    csd_matrix = csd_matrix / n_samples

    return freqs, csd_matrix


def _multivariate_welch(
    X: NDArray[np.floating], nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multivariate spectral density using Welch's method."""
    assert isinstance(X, np.ndarray)

    n_samples, n_features = X.shape

    if n_samples < _MIN_SERIES_LENGTH:
        raise ValueError(f"Input must have at least length {_MIN_SERIES_LENGTH}.")

    # Get frequency grid from first series
    freqs, _ = signal.welch(X[:, 0], nperseg=nperseg)
    n_freqs = len(freqs)

    # Initialize cross-spectral density matrix
    csd_matrix = np.zeros((n_freqs, n_features, n_features), dtype=complex)

    # Compute all pairwise cross-spectra
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                # Auto-spectrum (real)
                _, psd = signal.welch(X[:, i], nperseg=nperseg)
                csd_matrix[:, i, j] = psd
            else:
                # Cross-spectrum (complex)
                _, csd = signal.csd(X[:, i], X[:, j], nperseg=nperseg)
                csd_matrix[:, i, j] = csd
    # Welch returns corrected version by multiplying by 2; remove this multiplier here

    csd_matrix /= 2.0

    return freqs, csd_matrix


class MVSpectrumEstimator(sklearn.base.BaseEstimator):
    """
    sklearn-style estimator for multivariate spectral density estimation.

    Parameters
    ----------
    spectrum_method : {"welch", "periodogram"}, default="welch"
        Method for spectral density estimation.
    nperseg : int or None, default=None
        Segment length for Welch's method. If None, uses min(n_samples/10, 256).

    Attributes
    ----------
    mspectrum_ : MVSpectrum
        Estimated multivariate spectrum after fitting.
    """

    def __init__(self, method: str = "welch", nperseg: int | None = None):
        """Initializes the class."""
        self.method = method
        self.nperseg = nperseg
        self.mvspectrum_ = None

    def fit(self, X: NDArray[np.floating] | pd.DataFrame, y=None):
        """
        Estimate the multivariate spectrum from input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input multivariate time series (each column is a series).
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator with attribute `mspectrum_`.
        """
        del y

        self.nperseg = self.nperseg or min(X.shape[0] // 10, 256)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=["x" + str(i + 1) for i in range(X.shape[1])])

        X_names = X.columns

        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")

        if self.method == "welch":
            freqs, csd = _multivariate_welch(
                X.values,
                nperseg=self.nperseg,
            )
        elif self.method == "periodogram":
            freqs, csd = _multivariate_periodogram(X.values)
        else:
            raise ValueError(f"Unknown spectrum_method={self.method!r}")

        nz_mask = freqs > 0
        self.mvspectrum_ = MVSpectrum(
            freqs=freqs[nz_mask], csd=csd[nz_mask], normalized="none", names=X_names
        )
        return self


# def get_spectrum_from_mvspectrum(
#     mvspectrum_output: np.ndarray, which: int | list | np.ndarray | None = None
# ) -> np.ndarray:
#     """
#     Extract univariate spectra from multivariate spectral density matrix.

#     This extracts the diagonal elements (auto-spectra) from the multivariate
#     spectral density matrix, equivalent to the R function get_spectrum_from_mvspectrum.

#     Parameters
#     ----------
#     mvspectrum_output : np.ndarray of shape (n_freqs, n_features, n_features)
#         Multivariate spectral density matrix.

#     which : int, list, array, or None, default=None
#         Which series to extract. If None, returns all series.

#     Returns
#     -------
#     spectra : np.ndarray of shape (n_freqs,) or (n_freqs, n_selected)
#         Univariate spectra. Single column if which is int,
#         matrix if which is list/array or None.

#     Examples
#     --------
#     >>> # Extract first series spectrum
#     >>> spectrum_1 = get_spectrum_from_mvspectrum(mvspec, which=0)
#     >>>
#     >>> # Extract all spectra
#     >>> all_spectra = get_spectrum_from_mvspectrum(mvspec)
#     """
#     mvspectrum_output = np.asarray(mvspectrum_output)

#     if mvspectrum_output.ndim == 1:
#         # Already univariate
#         return mvspectrum_output

#     if mvspectrum_output.ndim != 3:
#         raise ValueError("mvspectrum_output must be 1D or 3D array")

#     n_freqs, n_features, _ = mvspectrum_output.shape

#     if which is None:
#         which = list(range(n_features))
#     elif isinstance(which, int):
#         which = [which]
#     else:
#         which = list(which)

#     # Validate indices
#     for idx in which:
#         if idx < 0 or idx >= n_features:
#             raise ValueError(f"Index {idx} out of range [0, {n_features - 1}]")

#     # Extract diagonal elements for each frequency
#     all_spectra = np.zeros((n_freqs, n_features), dtype=np.float64)

#     for freq_idx in range(n_freqs):
#         diag_elements = np.diag(mvspectrum_output[freq_idx])

#         # Check for imaginary diagonal elements (should be real)
#         max_imag_diag = np.max(np.abs(np.imag(diag_elements)))
#         if max_imag_diag > 1e-10:
#             warnings.warn(
#                 f"Multivariate spectrum has significant imaginary diagonal elements "
#                 f"(max={max_imag_diag:.2e}). Check spectrum estimates."
#             )

#         all_spectra[freq_idx] = np.real(diag_elements)

#     # Return selected columns
#     result = all_spectra[:, which]

#     # If single series requested, return 1D array
#     if len(which) == 1:
#         result = result.flatten()

#     return result
