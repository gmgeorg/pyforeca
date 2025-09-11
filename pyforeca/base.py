"""Base module for ForeCA: Forecastable Component Analysis in Python

A minimal implementation of Forecastable Component Analysis (ForeCA)
as described in Goerg (2013).

ForeCA finds linear combinations of multivariate time series that are
maximally forecastable, where forecastability is measured by spectral entropy.
"""

import numpy as np
import pandas as pd
import tqdm

from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from pyforeca import linalg, optimizer, spectral
from pyforeca.linalg import eigenvector_symmetric
from pyforeca.spectrum_types import MVSpectrum, UnivariateSpectrum
from pyforeca.utils import Decorrelator

_EPS = 1e-9


def _get_foreca_colnames(n_components: int) -> list[str]:
    """Gets foreca column names"""
    return [f"forec{i + 1}" for i in range(n_components)]


def _em_one_unconstrained(
    U_r: NDArray[np.floating],
    mv_r: MVSpectrum,
    *,
    init_method: optimizer.InitMethod,
    w0: NDArray[np.floating] | None = None,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, UnivariateSpectrum, float, bool, int]:
    """EM-like loop in reduced space to get a single ForeCA vector z (len=r)."""
    if w0 is None:
        z = optimizer.initialize_weightvector(
            U=U_r, f_U=mv_r.csd, init_method=init_method
        )
    else:
        z = np.asarray(w0, float).ravel()

    z /= linalg.l2_norm(z) + _EPS

    converged = False
    for _ in range(1, max_iter + 1):
        uy = mv_r.linear_combo(z, check_positive=False)
        p = np.maximum(uy.as_pmf().psd, float(_EPS))

        mv_r_norm = mv_r.normalize_whiten()
        Sbar = mv_r_norm.wcov(frequency_weights=-np.log(p))
        z_new = eigenvector_symmetric(Sbar, min=True)

        delta = linalg.angle_delta(z, z_new)
        z = z_new
        if delta <= tol:
            converged = True
            break

    uy_final = mv_r_norm.linear_combo(z, check_positive=True).as_pmf()
    H_norm = uy_final.entropy(normalize_by_max=True)
    omega = 1.0 - float(H_norm)
    return z, uy_final, omega, converged


def _em_one_unconstrained_multiple_tries(
    n_init: int,
    U_r: NDArray[np.floating],
    mv_r: MVSpectrum,
    *,
    init_method: optimizer.InitMethod,
    w0: NDArray[np.floating] | None = None,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, UnivariateSpectrum, float, bool, int]:
    """EM-like loop in reduced space to get a single ForeCA vector z (len=r)."""

    best_result = _em_one_unconstrained(
        U_r=U_r, mv_r=mv_r, init_method=init_method, tol=tol, max_iter=max_iter
    )
    best_omega = best_result[2]

    for i, im in enumerate(optimizer.InitMethod):
        current_result = _em_one_unconstrained(
            U_r=U_r, mv_r=mv_r, init_method=im, w0=w0, tol=tol, max_iter=max_iter
        )
        if current_result[2] > best_omega:
            best_omega = current_result[2]
            best_result = current_result

        if i > n_init - 1:
            break
    return best_result


class ForeCA(BaseEstimator, TransformerMixin):
    """
    Forecastable Component Analysis (ForeCA) via EM with null-space deflation.
    """

    def __init__(
        self,
        n_components: int | None = None,
        spectrum_method: str = "welch",
        nperseg: int | None = None,
        *,
        init_method: optimizer.InitMethod = optimizer.InitMethod.NORMAL,
        whiten: bool = False,
        max_iter: int = 100,
        tol: float = 1e-8,
        n_init: int = 10,
        random_state: int | None = None,
    ):
        """Initializes the class."""
        self.n_components = n_components
        self.spectrum_method = spectrum_method
        self.nperseg = nperseg
        self.whiten = whiten
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.decorrelator = Decorrelator()
        self.n_init = n_init
        self.random_state = random_state
        self.omegas_ = None
        self.omegas_X_ = None

    def _estimate_mvspectrum(self, U: NDArray[np.floating]) -> MVSpectrum:
        """Estimates the multivariate spectrum."""
        mvspectrum_estimator = spectral.MVSpectrumEstimator(
            method=self.spectrum_method, nperseg=self.nperseg
        )
        mvspectrum_estimator.fit(U)
        return mvspectrum_estimator.mvspectrum_

    def fit(self, X: NDArray[np.floating] | pd.DataFrame, y=None):
        """Fits ForeCA to 'X' time series."""
        is_df_input = isinstance(X, pd.DataFrame)
        if is_df_input:
            x_cols = X.columns
        else:
            x_cols = ["x" + str(i + 1) for i in range(X.shape[1])]

        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        _, n_series = X.shape

        k = self.n_components if self.n_components is not None else n_series
        self.n_components = k
        if not (1 <= k <= n_series):
            raise ValueError(
                f"n_components must be in [1,{n_series}] (got {self.n_components})"
            )

        # Whiten and spectrum
        self.decorrelator.fit(X)
        U = self.decorrelator.transform(X)  # cov ≈ I
        mv_full = self._estimate_mvspectrum(U)  # spectrum on whitened data

        mv_x_full = self._estimate_mvspectrum(X)
        omegas_x = pd.Series(
            [psd.omega() for psd in mv_x_full.get_univariate()],
            index=x_cols,
            name="omega",
        )
        self.omegas_X_ = omegas_x

        # Null-space deflation loop
        W = np.zeros((n_series, 0))  # whitened-space loadings (d × c)
        omegas, entrs = [], []

        pbar = tqdm.tqdm(range(k), total=k)
        for comp_id in pbar:
            pbar.set_description(f"Finding ForeC {comp_id + 1} ...")
            # Complement basis Q (d × r)
            Q = linalg.null_space_complement(W, n_series)
            r = Q.shape[1]
            if r == 0:
                break

            # If we're down to 1 dimension: solution is the only basis vector
            if r == 1:
                w = Q[:, 0]
                uy = mv_full.linear_combo(w, check_positive=True).as_pmf()
                Hn = uy.entropy(normalize_by_max=True)
                omega = 1.0 - float(Hn)
            else:
                # Reduce both data & spectrum, then run unconstrained EM in r-dim space
                U_r = U @ Q
                mv_r = mv_full.project(Q)
                (
                    z,
                    uy,
                    omega,
                    _,
                ) = _em_one_unconstrained_multiple_tries(
                    n_init=self.n_init,
                    U_r=U_r,
                    mv_r=mv_r,
                    init_method=self.init_method,
                    tol=self.tol,
                    max_iter=self.max_iter,
                )
                w = Q @ z  # lift back to full space

            # Append and re-orthonormalize W (stabilizes rank)
            W = np.column_stack([W, w])
            W, _ = np.linalg.qr(W, mode="reduced")

            omegas.append(omega)
            entrs.append(1.0 - omega)

        # Save whitened and processed-space loadings
        self.components_whitened_ = W  # (d, k_eff)
        Wd = self.decorrelator.transformation_matrix_  # U = X_proc @ Wd
        self.components_ = Wd @ W  # processed-space loadings

        self.omegas_ = np.array(omegas, float)
        self.spectral_entropy_ = np.array(entrs, float)
        denom = float(self.omegas_.sum()) if self.omegas_.size else 1.0
        self.explained_forecastability_ratio_ = (
            self.omegas_ / denom if denom > 0 else np.zeros_like(self.omegas_)
        )

        # Optionally reorder by decreasing Omega (to match R)
        order = np.argsort(-self.omegas_)
        if not np.all(order == np.arange(order.size)):
            self.components_whitened_ = self.components_whitened_[:, order]
            self.components_ = self.components_[:, order]
            self.omegas_ = self.omegas_[order]
            self.spectral_entropy_ = self.spectral_entropy_[order]
            self.explained_forecastability_ratio_ = (
                self.explained_forecastability_ratio_[order]
            )

        self.transformation_matrix_ = self.components_
        self.omegas_ = pd.Series(
            self.omegas_, index=_get_foreca_colnames(self.n_components), name="omega"
        )

        if is_df_input:
            self.components_ = pd.DataFrame(
                self.components_,
                index=x_cols,
                columns=_get_foreca_colnames(self.n_components),
            )
        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Transforms X time series to forecastable components (ForeCs)."""
        if not hasattr(self, "transformation_matrix_"):
            raise ValueError("ForeCA must be fitted before transforming data")
        U = self.decorrelator.transform(X)
        scores = U @ self.components_whitened_

        if isinstance(X, pd.DataFrame):
            scores = pd.DataFrame(
                scores, index=X.index, columns=_get_foreca_colnames(scores.shape[1])
            )

        return scores

    def fit_transform(self, X: NDArray[np.floating], y=None) -> NDArray[np.floating]:
        """Fits and transforms from X."""
        return self.fit(X).transform(X)
