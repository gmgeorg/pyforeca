"""Entropy functions."""

import numpy as np

from scipy import stats

_EPS = 1e-6


def assert_pmf(probs: np.ndarray, is_half: bool) -> None:
    """Asserts that probs is a probability mass function."""
    assert (probs >= 0.0).all(), "All entries must be non-negative."

    probs_sum = probs.sum()
    if is_half:
        if np.abs(probs_sum - 0.5) > _EPS:
            raise ValueError(
                f"probs are not a (half) probability mass function. Got {probs_sum}"
            )

    else:
        if np.abs(probs_sum - 1.0) > _EPS:
            raise ValueError(
                f"probs are not a probability mass function. Got {probs_sum}"
            )


def discrete_entropy(probs: np.ndarray, normalize_by_max: bool) -> float:
    """Computes entropy on x; optionally normalize by maximal possible entropy."""
    assert_pmf(probs, False)
    entropy = stats.entropy(probs)
    if normalize_by_max:
        n_probs = len(probs)
        entropy /= np.log(n_probs)

    return float(entropy)
