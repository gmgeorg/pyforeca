"""Module for holding spectrum types (univariate & multivariate)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt  # local import
import numpy as np
import pandas as pd

from numpy.typing import NDArray

from pyforeca import entropy, linalg

ArrayLike = np.ndarray | Sequence[float]

_EPS = 1e-9


@dataclass(frozen=True)
class UnivariateSpectrum:
    """One-sided univariate spectrum on a frequency grid.

    Attributes:
      freqs: (F,) nonnegative frequencies. Typically excludes DC (0) for entropy.
      psd: (F,) real nonnegative power spectral density at `freqs`.
      normalized: flag describing mass scaling: 'none' or 'pmf'.
    """

    freqs: NDArray[np.floating]
    psd: NDArray[np.floating]
    name: str = ""
    normalized: str = "none"  # {'none', 'pmf'}

    def __post_init__(self):
        """Post initialization."""
        freqs = np.asarray(self.freqs, dtype=float)
        psd = np.asarray(self.psd, dtype=float)
        if freqs.ndim != 1 or psd.ndim != 1:
            raise ValueError("freqs and psd must be 1D arrays.")
        if freqs.shape[0] != psd.shape[0]:
            raise ValueError("freqs and psd must have the same length.")
        if np.any(psd < -1e-12):
            raise ValueError("psd must be nonnegative (within numerical tolerance).")
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "psd", np.clip(psd, 0.0, np.inf))
        object.__setattr__(self, "name", self.name)

    def as_pmf(self) -> "UnivariateSpectrum":
        """Return a copy with psd normalized to sum to .5 (for entropy)."""
        s = float(np.sum(self.psd))
        if s <= 0:
            raise ValueError("Cannot normalize: total power is nonpositive.")
        pmf = np.maximum(self.psd / s, _EPS)
        return UnivariateSpectrum(
            self.freqs, pmf / 2.0, normalized="pmf", name=self.name
        )

    def entropy(self, normalize_by_max: bool = False) -> float:
        """Discrete Shannon entropy over the one-sided grid."""
        p = self.as_pmf().psd
        p = np.concat([p, p])
        entr = entropy.discrete_entropy(p, normalize_by_max=normalize_by_max)
        return entr

    def omega(self) -> float:
        """Computes the forecastability measure Omega(x_t) for a univariate series."""
        norm_entr = self.entropy(True)
        return 1.0 - norm_entr

    def plot(
        self,
        ax=None,
        *,
        log_y: bool = True,
        log_x: bool = False,
        floor: float = 1e-12,
        **kwargs,
    ):
        """Quick PSD plot.

        Args:
            ax: Optional matplotlib Axes.
            log_y: If True (default), use logarithmic y-scale.
            log_x: If True, use logarithmic x-scale.
            floor: Minimum y value used to avoid log(0) when log_y=True.
            **kwargs: Passed to matplotlib plot call.
        """
        ax = ax or plt.gca()

        y = np.maximum(self.psd, float(floor)) if log_y else self.psd

        if log_x and log_y:
            ax.loglog(self.freqs, y, **kwargs)
        elif log_x:
            ax.semilogx(self.freqs, y, **kwargs)
        elif log_y:
            ax.semilogy(self.freqs, y, **kwargs)
        else:
            ax.plot(self.freqs, y, **kwargs)

        ax.set_xlabel("frequency")
        ax.set_ylabel("PSD")
        ax.set_title(self.name)
        ax.grid(True, which="both", ls=":")
        return ax


@dataclass(frozen=True)
class MVSpectrum:
    """One-sided multivariate spectrum f(ω) on a frequency grid.

    Attributes:
      freqs: (F,) frequencies (nonnegative one-sided grid).
      csd: (F, K, K) complex Hermitian cross-spectral density matrices.
      normalized: {'none', 'whitened'}; 'whitened' means ∑ f(ω) dω ≈ 0.5 * I_K.
    """

    freqs: NDArray[np.floating]
    csd: NDArray[np.complexfloating]
    names: list[str] | None = None
    normalized: str = "none"  # {'none', 'whitened'}
    normalized: Literal["none", "whitened"] = "none"
    normalizing_transform: NDArray[np.floating] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        freqs = np.asarray(self.freqs, dtype=float)
        csd = np.asarray(self.csd)
        if freqs.ndim != 1 or csd.ndim != 3:
            raise ValueError("freqs must be (F,), csd must be (F,K,K).")
        if freqs.shape[0] != csd.shape[0]:
            raise ValueError("freqs and csd first dimension (F) must match.")
        if csd.shape[1] != csd.shape[2]:
            raise ValueError("csd matrices must be square (KxK).")
        # Hermitianize softly
        csd = 0.5 * (csd + np.conj(np.swapaxes(csd, -1, -2)))

        names = self.names
        if self.names is None:
            names = [""] * csd.shape[1]

        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "csd", csd)
        object.__setattr__(self, "names", names)

    @property
    def n_freqs(self) -> int:
        """Number of frequencies."""
        return self.csd.shape[0]

    @property
    def n_series(self) -> int:
        """Number of series."""
        return self.csd.shape[1]

    def get_univariate(self, which: int | None = None) -> NDArray[np.floating]:
        """Return diagonal auto-spectra; if `which` is None returns (F,K) array."""
        diag = np.real(np.diagonal(self.csd, axis1=1, axis2=2))  # (F,K)
        if which is None:
            all_univariate = [self.get_univariate(i) for i in range(self.n_series)]
            return all_univariate

        idx = [which] if isinstance(which, int) else list(which)
        return UnivariateSpectrum(
            freqs=self.freqs, psd=diag[:, idx].squeeze(), name=self.names[which]
        )

    def omegas(self) -> pd.Series:
        """Get omegas of each diagonal."""
        psds = self.get_univariate(None)
        omegas = [p.omega() for p in psds]
        return pd.Series(omegas, index=self.names, name="omega")

    def linear_combo(
        self, beta: ArrayLike, check_positive: bool = True
    ) -> UnivariateSpectrum:
        """Spectrum of y_t = X_t β as βᴴ f(ω) β."""
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if beta.size != self.n_series:
            raise ValueError("beta length must equal number of series.")
        # Quadratic form for each frequency
        Fy = np.einsum("fij,j,i->f", self.csd, beta.conj(), beta)  # (F,)
        Fy = np.real(Fy)
        if check_positive:
            neg = Fy < -1e-10
            if np.any(neg):
                # zero clip and optionally renormalize (keep simple: clip only)
                Fy = np.maximum(Fy, 0.0)
        return UnivariateSpectrum(
            self.freqs, Fy, normalized="none", name="linear_combination"
        )

    def wcov(
        self,
        frequency_weights: np.ndarray | None = None,
        *,
        method: Literal["auto", "trapz", "sum"] = "auto",
    ) -> np.ndarray:
        """Weighted covariance via discrete integral.

        Σ ≈ C * ∫ ( w(ω) f(ω) ) dω over the one-sided frequency grid,
        where C = 1 for 'one-sided' spectra (SciPy Welch/periodogram) and
            C = 2 for 'positive-no-double' (raw rFFT positives without doubling).

        Args:
        frequency_weights: Optional (F,) array of nonnegative weights w(ω_k) applied
            pointwise to the cross-spectral matrices before integration.
        kernel_weights: Alias for `frequency_weights` (kept for R parity).
        method: 'auto' (use sum×Δf for uniform grids, else trapz), or force 'trapz'/'sum'.

        Returns:
        Sigma: (K,K) real symmetric weighted covariance estimate.
        """
        if frequency_weights is not None:
            w = np.asarray(frequency_weights, dtype=float).reshape(-1)
            if w.shape[0] != self.n_freqs:
                raise ValueError(
                    f"frequency_weights length {w.shape[0]} != n_freqs {self.n_freqs}"
                )
            if np.any(~np.isfinite(w)) or np.any(w < 0):
                raise ValueError("frequency_weights must be finite and nonnegative.")
            csd_use = self.csd * w[:, None, None]
        else:
            csd_use = self.csd

        # Choose integration method
        f = self.freqs
        diffs = np.diff(f)
        uniform = np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12)
        if method == "sum" or (method == "auto" and uniform):
            df = float(diffs[0]) if f.size > 1 else 1.0
            integ = csd_use.sum(axis=0) * df
        else:
            integ = np.trapezoid(csd_use, x=f, axis=0)

        C = 2.0
        Sigma = C * integ

        # Symmetrize and sanity-check
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma_imag = np.imag(Sigma)
        # linalg.l2_norm is your helper; if not available, replace with np.linalg.norm
        if linalg.l2_norm(Sigma_imag) > _EPS:
            raise ValueError("Non-zero imaginary parts of weighted covariance matrix")

        return np.real(Sigma)

    def normalize_whiten(self) -> "MVSpectrum":
        """Normalize & whiten spectrum  f̃(ω) = Bᴴ f(ω) B  with B = Σ^{-1/2}.

        After whitening, ∫ f̃(ω) dω ≈ 0.5 * I (one-sided), matching ForeCA conventions.
        """
        Sigma = self.wcov()
        # Eigendecomposition and inverse sqrt
        evals, evecs = np.linalg.eigh(Sigma)
        if np.any(evals <= 0):
            # Regularize tiny/negative
            evals = np.maximum(evals, 1e-12)
        invsqrt = (evecs / np.sqrt(evals)) @ evecs.T
        object.__setattr__(self, "normalizing_transform", invsqrt)
        # Apply B to each frequency: Bᴴ f B
        B = invsqrt
        f_wh = np.einsum("ij,fjk,kl->fil", B.T, self.csd, B)
        return MVSpectrum(
            freqs=self.freqs, csd=f_wh, normalized="whitened", names=self.names
        )

    def project(self, Q: np.ndarray) -> "MVSpectrum":
        """Project the multivariate spectrum with Y = Qᵀ U  ⇒  f_Y(ω) = Qᴴ f_U(ω) Q."""
        Q = np.asarray(Q)
        if Q.ndim != 2 or Q.shape[0] != self.n_series:
            raise ValueError(f"Q must be (d, r) with d={self.n_series}. Got {Q.shape}.")

        # tmp = f_U(ω) @ Q   → (F, d, r)
        tmp = np.tensordot(self.csd, Q, axes=([2], [0]))
        # csd_r = Qᴴ @ tmp   → (r, F, r)  then move axes to (F, r, r)
        csd_r = np.tensordot(Q.conj().T, tmp, axes=([1], [1]))
        csd_r = np.moveaxis(csd_r, 1, 0)

        # Hermitize to kill tiny asymmetries
        csd_r = 0.5 * (csd_r + np.conj(np.swapaxes(csd_r, 1, 2)))

        return MVSpectrum(
            freqs=self.freqs.copy(), csd=csd_r, normalized=False, names=self.names
        )

    def plot(
        self,
        which: int | Sequence[int] | None = None,
        ax=None,
        *,
        log_y: bool = True,
        log_x: bool = False,
        floor: float = 1e-12,
        legend: bool = True,
        **kwargs,
    ):
        """Plot diagonal auto-spectra; selects channels via `which`.

        Args:
            which: Series index/int or list; None plots all diagonals.
            ax: Optional matplotlib Axes.
            log_y: If True, use logarithmic y-scale.
            log_x: If True, use logarithmic x-scale.
            floor: Minimum y value used to avoid log(0) when log_y=True.
            legend: Whether to add a legend for multiple series.
            **kwargs: Passed to matplotlib plot calls.
        """
        ax = ax or plt.gca()

        diag = self.get_univariate(which=None if which is None else which)

        def _plot_line(freqs, y, label=None):
            y = np.maximum(y, float(floor)) if log_y else y
            if log_x and log_y:
                ax.loglog(freqs, y, label=label, **kwargs)
            elif log_x:
                ax.semilogx(freqs, y, label=label, **kwargs)
            elif log_y:
                ax.semilogy(freqs, y, label=label, **kwargs)
            else:
                ax.plot(freqs, y, label=label, **kwargs)

        if isinstance(diag, UnivariateSpectrum):
            _plot_line(diag.freqs, diag.psd, None)

        elif isinstance(diag, list):
            for k, sp in enumerate(diag):
                _plot_line(
                    sp.freqs, sp.psd, label=getattr(self, "names", [f"series {k}"])[k]
                )
            if legend:
                ax.legend()

        elif isinstance(diag, np.ndarray) and diag.ndim == 2:
            for k in range(diag.shape[1]):
                _plot_line(self.freqs, diag[:, k], label=f"series {k}")
            if legend:
                ax.legend()

        ax.set_xlabel("frequency")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", ls=":")
        return ax
