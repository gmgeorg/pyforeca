"""Generating simulated data."""

import numpy as np
import pandas as pd


def gen_toy_data(
    n_samples: int, phi: float = 0.8
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Generates toy data for simulation / illustration purposes."""

    # -----------------------------
    # Step 1: simulate latent signals
    # -----------------------------
    rng = np.random.default_rng(42)
    t = np.arange(n_samples)

    sine_signal = np.sin(2 * np.pi * t / 50) + rng.normal(scale=0.3, size=n_samples)
    ar_signal = np.zeros(n_samples)
    for i in range(1, n_samples):
        ar_signal[i] = phi * ar_signal[i - 1] + rng.normal(scale=1.0)
    wn_signal = rng.normal(size=n_samples)

    S = np.column_stack([sine_signal, ar_signal, wn_signal])
    S = pd.DataFrame(S, columns=["sine_plus_noise", "ar1", "white_noise"])

    # -----------------------------
    # Step 2: mix the sources
    # -----------------------------
    # mixing matrix
    A = np.array(
        [
            [0.6, 0.4, 0.7],
            [0.2, 0.8, 0.3],
            [0.7, 0.3, 0.5],
        ]
    )
    X = S @ A.T
    X.columns = ["mixture1", "mixture2", "mixture3"]

    return S, A, X
