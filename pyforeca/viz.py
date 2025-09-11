"""Visualization of ForeCA results"""

import math

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyforeca import base, utils


def biplot(
    mod_foreca: base.ForeCA,
    X: np.ndarray | None = None,  # (n, d) original data used to fit
    *,
    comps: tuple[int, int] = (0, 1),
    obs_labels: Iterable[str] | None = None,
    var_labels: Iterable[str] | None = None,
    sample_obs: int | None = None,  # downsample points for large n
    point_alpha: float = 0.5,
    point_size: float = 12.0,
    arrow_color: str | None = None,  # None → default cycle
    arrow_width: float = 0.0025,
    arrow_headwidth: float = 8.0,
    arrow_headlength: float = 10.0,
    arrow_alpha: float = 0.9,
    text_kwargs: dict | None = None,
    equal_axes: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (7.5, 6.0),
):
    """Biplot for ForeCA (or PCA-like) results: points = scores, arrows = loadings.

    Args:
      scores: (n, k) matrix of component scores (e.g., Y = foreca.transform(X)).
      loadings: (d, k) loadings in the *processed/original* space (model.components_).
      comps: pair of component indices to plot (0-based).
      obs_labels: optional labels for observations (len n). If None, just dots.
      var_labels: optional labels for variables (len d). If None, use v1..vd.
      sample_obs: if set (e.g., 1000), randomly subsample that many observations.
      point_alpha/point_size: styling for score points.
      arrow_*: styling for loading arrows.
      text_kwargs: dict passed to text labels for variables.
      equal_axes: if True, sets equal aspect so directions are faithful.
      title: optional plot title.
      figsize: figure size.
    """
    scores = mod_foreca.transform(X)
    loadings = np.asarray(mod_foreca.components_)
    var_labels = X.columns
    obs_labels = X.index
    scores = np.asarray(scores, float)
    # loadings = np.asarray(loadings, float)
    i, j = comps
    if scores.ndim != 2 or loadings.ndim != 2:
        raise ValueError("scores and loadings must be 2D arrays.")
    if i >= scores.shape[1] or j >= scores.shape[1]:
        raise ValueError("Component index out of range for scores.")
    if i >= loadings.shape[1] or j >= loadings.shape[1]:
        raise ValueError("Component index out of range for loadings.")

    # Select 2D slices
    S = scores[:, [i, j]]
    L = loadings[:, [i, j]]

    # Optional downsampling for visibility
    n = S.shape[0]
    if sample_obs is not None and sample_obs < n:
        idx = np.random.default_rng(0).choice(n, size=sample_obs, replace=False)
        S_plot = S[idx]
        obs_labels_plot = (
            [obs_labels[k] for k in idx] if obs_labels is not None else None
        )
    else:
        S_plot = S
        obs_labels_plot = obs_labels

    del obs_labels_plot
    # Scale loadings to fit nicely inside the score cloud (R biplot-like)
    # Put arrow tips roughly within the 90% span of points
    span_scores = np.percentile(np.abs(S_plot), 99, axis=0)  # robust span
    span_scores[span_scores == 0] = 1.0
    span_load = np.max(np.abs(L), axis=0)
    span_load[span_load == 0] = 1.0
    scale = 0.9 * (span_scores / span_load)  # per-axis scaling
    L_scaled = L * scale  # (d,2)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(
        S_plot[:, 0], S_plot[:, 1], s=point_size, alpha=point_alpha, color="gray"
    )

    # Arrows for variables (loadings)
    ck = {"fontsize": 10, "ha": "left", "va": "center"}
    if text_kwargs:
        ck.update(text_kwargs)
    for r in range(L_scaled.shape[0]):
        x, y = L_scaled[r, 0], L_scaled[r, 1]
        ax.arrow(
            0.0,
            0.0,
            x,
            y,
            length_includes_head=True,
            width=arrow_width,
            head_width=arrow_headwidth * arrow_width,
            head_length=arrow_headlength * arrow_width,
            alpha=arrow_alpha,
            color=arrow_color,
        )
        label = var_labels[r] if var_labels is not None else f"v{r + 1}"
        ax.text(x * 1.03, y * 1.03, label, **ck)

    # Labels / aesthetics
    ax.set_xlabel(f"ForeC{i + 1}")
    ax.set_ylabel(f"ForeC{j + 1}")
    if equal_axes:
        ax.set_aspect("equal", adjustable="datalim")
    # Expand limits a bit to fit arrowheads/text
    xlim = np.array(ax.get_xlim(), float)
    ylim = np.array(ax.get_ylim(), float)
    ax.set_xlim(xlim * 1.1)
    ax.set_ylim(ylim * 1.1)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    return fig, ax


def _acf_1d(x: np.ndarray, nlags: int, unbiased: bool = True) -> np.ndarray:
    """Autocorrelation function r_k, k=0..nlags (FFT-based)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D.")
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 observations")
    x = x - np.nanmean(x)

    # zero-pad to next power of two for speed
    nfft = 1 << (int(np.ceil(np.log2(2 * n - 1))))
    fx = np.fft.rfft(x, n=nfft)
    acov_full = np.fft.irfft(fx * np.conjugate(fx), n=nfft).real[: nlags + 1]

    if unbiased:
        denom = (n - np.arange(nlags + 1)).astype(float)
    else:
        denom = float(n)
    acov = acov_full / denom
    return acov / acov[0]


def _pacf_durbin_levinson(r: np.ndarray, nlags: int) -> np.ndarray:
    """Partial autocorrelations φ_k at lags 0..nlags via Durbin–Levinson.

    Args:
      r: autocorrelation values r_k, k=0..nlags (with r_0 = 1).
    """
    if r.ndim != 1 or r.size < nlags + 1:
        raise ValueError("r must be 1-D with length >= nlags+1")

    phi = np.zeros((nlags + 1, nlags + 1), dtype=float)
    sig = np.zeros(nlags + 1, dtype=float)

    phi[1, 1] = r[1]
    sig[1] = 1 - r[1] ** 2

    for k in range(2, nlags + 1):
        num = r[k] - np.dot(phi[1:k, k - 1], r[1:k][::-1])
        den = sig[k - 1]
        den = den if den > 1e-14 else 1e-14
        phi[k, k] = num / den
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)

    pacf = np.zeros(nlags + 1, dtype=float)
    pacf[0] = 1.0
    for k in range(1, nlags + 1):
        pacf[k] = phi[k, k]
    return pacf


# ---------- plotting helpers ----------


def _conf_z(alpha: float) -> float:
    """≈ z_{1-α/2}. Uses 1.96 for α=0.05 without SciPy; else Wilson–Hilferty approx."""
    if abs(alpha - 0.05) < 1e-12:
        return 1.96
    # Wilson–Hilferty approximation for normal quantile
    # (good enough for plotting if SciPy is not available)
    p = 1 - alpha / 2
    t = np.sqrt(2) * erfinv_safe(2 * p - 1)
    return float(t)


def erfinv_safe(y: float) -> float:
    """Approximate inverse erf via a rational approximation (sufficient for plot bands)."""
    # Source: Winitzki (2008) approximation
    a = 0.147  # magic constant
    sgn = 1 if y >= 0 else -1
    ln = np.log(1 - y * y)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sgn * np.sqrt(np.sqrt(first * first - second) - first)


def plot_acf(
    x: np.ndarray,
    nlags: int = 40,
    *,
    alpha: float = 0.05,
    unbiased: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """Stem plot of the autocorrelation function with confidence bands."""
    if pd is not None and isinstance(x, (pd.Series | pd.DataFrame)):
        x = np.asarray(x).squeeze()
    r = _acf_1d(x, nlags=nlags, unbiased=unbiased)
    n = len(np.asarray(x))
    z = _conf_z(alpha)
    band = z / math.sqrt(n)

    ax = ax or plt.gca()
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(band, color="tab:blue", linestyle="--", linewidth=1)
    ax.axhline(-band, color="tab:blue", linestyle="--", linewidth=1)
    ax.vlines(np.arange(1, nlags + 1), [0], r[1:], colors="tab:orange")
    ax.set_xlim(-0.5, nlags + 0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("ACF")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_pacf(
    x: np.ndarray,
    nlags: int = 40,
    *,
    alpha: float = 0.05,
    unbiased: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """Stem plot of the partial autocorrelation function with confidence bands."""
    if pd is not None and isinstance(x, (pd.Series | pd.DataFrame)):
        x = np.asarray(x).squeeze()
    r = _acf_1d(x, nlags=nlags, unbiased=unbiased)
    pacf = _pacf_durbin_levinson(r, nlags=nlags)
    n = len(np.asarray(x))
    z = _conf_z(alpha)
    band = z / math.sqrt(n)

    ax = ax or plt.gca()
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(band, color="tab:blue", linestyle="--", linewidth=1)
    ax.axhline(-band, color="tab:blue", linestyle="--", linewidth=1)
    ax.vlines(np.arange(1, nlags + 1), [0], pacf[1:], colors="tab:green")
    ax.set_xlim(-0.5, nlags + 0.5)
    # ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("lag")
    ax.set_ylabel("PACF")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_foreca(
    mod_foreca: base.ForeCA,
    *,
    title: str = "Forecastability Ω comparison",
    add_white_noise_test: bool = False,
    X: np.ndarray | None = None,  # (n, d) original data used to fit
    input_names: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> tuple[plt.Figure, np.ndarray]:
    """Plots results of a trained ForeCA transformer.

    Forecastability (Ω) of ForeCs vs. original input series and (optionally)
    show white-noise test p-values (Ljung–Box) in a 2×2 grid.

    Requirements:
      - mod_foreca.omegas_        : array-like of shape (k,)
      - mod_foreca.omegas_X_      : array-like of shape (d,)  (your per-variable Ω)
    """
    if not hasattr(mod_foreca, "omegas_"):
        raise ValueError("ForeCA object has no omegas_. Did you call fit()?")

    if not hasattr(mod_foreca, "omegas_X_"):
        raise ValueError("ForeCA object has no omegas_X_. You must compute/store it.")

    # Convert to % for display
    omegas_fc = np.asarray(mod_foreca.omegas_, float) * 100.0
    omegas_in = np.asarray(mod_foreca.omegas_X_, float) * 100.0

    # Build labels
    comp_labels = (
        list(mod_foreca.omegas_.index)
        if hasattr(mod_foreca.omegas_, "index")
        else [f"ForeC{i + 1}" for i in range(len(omegas_fc))]
    )
    var_labels = (
        list(mod_foreca.omegas_X_.index)
        if hasattr(mod_foreca.omegas_X_, "index")
        else (
            list(input_names)
            if input_names is not None
            else [f"x{i + 1}" for i in range(len(omegas_in))]
        )
    )

    # Global reference lines
    omega_min = float(min(omegas_fc.min(), omegas_in.min()))
    omega_max = float(max(omegas_fc.max(), omegas_in.max()))

    # Choose layout
    if add_white_noise_test:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
        ax_fc, ax_in = axes[0, 0], axes[0, 1]
        ax_fc_p, ax_in_p = axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        ax_fc, ax_in = axes[0], axes[1]
        ax_fc_p = ax_in_p = None

    bar_color = "gray"

    # --- Top-left: ForeCs Ω ---
    sns.barplot(x=comp_labels, y=omegas_fc, color=bar_color, ax=ax_fc)
    ax_fc.set_title("Forecastability (ForeCs)")
    ax_fc.set_ylabel("Ω (%)")
    ax_fc.set_xlabel("")
    ax_fc.set_ylim(0, omega_max * 1.1)
    ax_fc.axhline(omega_min, color="tab:blue", linestyle="--", linewidth=1)
    ax_fc.axhline(omega_max, color="tab:red", linestyle="--", linewidth=1)
    ax_fc.tick_params(axis="x", rotation=90)
    ax_fc.grid(axis="y", alpha=0.4)

    # --- Top-right: Original Ω ---
    sns.barplot(x=var_labels, y=omegas_in, color=bar_color, ax=ax_in)
    ax_in.set_title("Forecastability (X)")
    ax_in.set_xlabel("")
    ax_in.axhline(omega_min, color="tab:blue", linestyle="--", linewidth=1)
    ax_in.axhline(omega_max, color="tab:red", linestyle="--", linewidth=1)
    ax_in.tick_params(axis="x", rotation=90)
    ax_in.grid(axis="y", alpha=0.4)

    # --- Bottom row: white-noise p-values (optional) ---
    if add_white_noise_test:
        if X is None:
            raise ValueError(
                "add_white_noise_test=True requires `scores` and `X_input` arrays."
            )

        scores = np.asarray(mod_foreca.transform(X), float)
        X = np.asarray(X, float)

        # ForeCs p-values
        p_fc = [utils.ljung_box(scores[:, i])[1] for i in range(scores.shape[1])]
        sns.barplot(x=comp_labels, y=p_fc, color=bar_color, ax=ax_fc_p)
        ax_fc_p.set_title("White-noise test p-value (ForeCs)")
        ax_fc_p.set_ylabel("p-value")
        ax_fc_p.set_xlabel("")
        ax_fc_p.axhline(alpha, color="tab:blue", linestyle="--", linewidth=1)
        ax_fc_p.set_ylim(0, max([0.05, max(p_fc) * 1.15]))
        ax_fc_p.tick_params(axis="x", rotation=60)
        ax_fc_p.grid(axis="y", alpha=0.4)

        # Original-series p-values
        p_in = [utils.ljung_box(X[:, j])[1] for j in range(X.shape[1])]
        sns.barplot(x=var_labels, y=p_in, color=bar_color, ax=ax_in_p)
        ax_in_p.set_title("White-noise test p-value (original)")
        ax_in_p.set_ylabel("p-value")
        ax_in_p.set_xlabel("")
        ax_in_p.axhline(alpha, color="tab:blue", linestyle="--", linewidth=1)
        ax_in_p.set_ylim(0, max([0.05, max(p_in) * 1.15]))
        ax_in_p.tick_params(axis="x", rotation=60)
        ax_in_p.grid(axis="y", alpha=0.4)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, axes


def plot_time_series(scores, *, title: str = "", time_index=None, n_cols: int = 1):
    """
    Faceted time series plot of time series.

    Args
    ----
    scores : array-like (n_samples, n_components) or DataFrame
        Time series
    title : str
        Title for the figure.
    time_index : array-like, optional
        Time axis (defaults to np.arange(n_samples)).
    n_cols : int, default=1
        Number of columns in facet grid (rows are computed automatically).
    """
    if isinstance(scores, pd.DataFrame):
        X = scores.values
        comp_names = scores.columns
        if time_index is None:
            time_index = scores.index
    else:
        X = np.asarray(scores)
        comp_names = [f"x{i + 1}" for i in range(X.shape[1])]
        if time_index is None:
            time_index = np.arange(X.shape[0])

    _, k = X.shape
    n_rows = math.ceil(k / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()

    for i in range(k):
        ax = axes[i]
        ax.plot(time_index, X[:, i], color="black", linewidth=0.8)
        ax.axhline(np.mean(X[:, i]), color="gray", linewidth=0.7, linestyle="--")
        ax.set_title(comp_names[i])
        ax.grid(True, linestyle=":", alpha=0.6)

    # Remove unused subplots
    for j in range(k, len(axes)):
        fig.delaxes(axes[j])

    axes[min(k, len(axes)) - 1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, axes
