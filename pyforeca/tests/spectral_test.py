"""
Pytest unit tests for ForeCA spectral entropy and multivariate spectral density functions.

Run with: pytest test_foreca_functions.py -v
"""

import numpy as np
import pytest

from pyforeca import spectral


# Fixtures for test data
@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def sample_size():
    """Standard sample size for tests."""
    return 2048


@pytest.fixture
def pure_sine_wave(sample_size):
    """Generate a pure sine wave (highly predictable)."""
    t = np.linspace(0, 10, sample_size)
    return np.sin(2 * np.pi * 0.1 * t)


@pytest.fixture
def noisy_sine_wave(sample_size, random_seed):
    """Generate a noisy sine wave (moderately predictable)."""
    t = np.linspace(0, 10, sample_size)
    return np.sin(2 * np.pi * 0.1 * t) + 0.2 * np.random.randn(sample_size)


@pytest.fixture
def white_noise(sample_size, random_seed):
    """Generate white noise (unpredictable)."""
    return np.random.randn(sample_size)


@pytest.fixture
def ar_process(sample_size, random_seed):
    """Generate AR(1) process (moderately predictable)."""
    x = np.zeros(sample_size)
    x[0] = np.random.randn()
    for i in range(1, sample_size):
        x[i] = 0.7 * x[i - 1] + 0.3 * np.random.randn()
    return x


@pytest.fixture
def multivariate_data(sample_size, random_seed):
    """Generate multivariate time series data."""
    t = np.linspace(0, 10, sample_size)
    x1 = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(sample_size)
    x2 = np.cos(2 * np.pi * 0.15 * t) + 0.1 * np.random.randn(sample_size)
    x3 = np.random.randn(sample_size)
    return np.column_stack([x1, x2, x3])


@pytest.fixture
def positive_definite_mvspectrum(random_seed):
    """Generate positive definite multivariate spectral density matrices."""
    n_freqs, n_features = 32, 3
    mvspec = np.zeros((n_freqs, n_features, n_features), dtype=complex)

    for freq_idx in range(n_freqs):
        # Generate random Hermitian matrix
        A = np.random.randn(n_features, n_features) + 1j * np.random.randn(
            n_features, n_features
        )
        A = np.abs(A)
        A_hermitian = (A + np.conj(A.T)) / 2
        # Ensure positive definiteness
        mvspec[freq_idx] = A_hermitian + 2 * np.eye(n_features)

    return mvspec


class TestUnivariateSpectralEntropy:
    """Test cases for spectral.univariate_spectral_entropy function."""

    def test_entropy_is_positive(self, pure_sine_wave, white_noise):
        """Test that spectral entropy is always positive."""
        entropy_sine = spectral.univariate_spectral_entropy(pure_sine_wave)
        entropy_noise = spectral.univariate_spectral_entropy(white_noise)

        assert entropy_sine > 0
        assert entropy_noise > 0
        assert entropy_sine < entropy_noise

    def test_entropy_ordering_predictability(
        self, pure_sine_wave, noisy_sine_wave, white_noise
    ):
        """Test that entropy increases with unpredictability."""
        entropy_pure = spectral.univariate_spectral_entropy(pure_sine_wave)
        entropy_noisy = spectral.univariate_spectral_entropy(noisy_sine_wave)
        entropy_noise = spectral.univariate_spectral_entropy(white_noise)

        # More predictable signals should have lower entropy
        assert entropy_pure < entropy_noisy
        assert entropy_noisy < entropy_noise

    def test_welch_vs_periodogram_methods(self, noisy_sine_wave):
        """Test both spectral estimation methods."""
        entropy_welch = spectral.univariate_spectral_entropy(
            noisy_sine_wave, method="welch"
        )
        entropy_periodogram = spectral.univariate_spectral_entropy(
            noisy_sine_wave, method="periodogram"
        )

        # Both should be positive and reasonably close
        assert entropy_welch > 0
        assert entropy_periodogram > 0
        # smoothed estimates give larger entropy (more uniform like, less spikey)
        assert entropy_welch > entropy_periodogram

    def test_nperseg_parameter(self, noisy_sine_wave):
        """Test nperseg parameter for Welch's method."""
        entropy_default = spectral.univariate_spectral_entropy(
            noisy_sine_wave, method="welch"
        )
        entropy_custom = spectral.univariate_spectral_entropy(
            noisy_sine_wave, method="welch", nperseg=128
        )

        assert entropy_default > 0
        assert entropy_custom > 0
        # Different segment lengths should give different but reasonable results
        assert abs(entropy_default - entropy_custom) < 2.0

    def test_constant_signal(self):
        """Test behavior with constant signal."""
        constant_signal = np.ones(100)
        # Constant signal should have very low entropy (highly predictable)
        entropy = spectral.univariate_spectral_entropy(constant_signal)
        assert (
            entropy >= 0
        )  # Should be close to 0 but numerical precision may make it slightly positive
        assert entropy < 0.1  # Should be very small

    def test_input_types(self, white_noise):
        """Test different input types are handled correctly."""
        # Test list input
        entropy_list = spectral.univariate_spectral_entropy(white_noise.tolist())
        entropy_array = spectral.univariate_spectral_entropy(white_noise)

        assert abs(entropy_list - entropy_array) < 1e-10

    def test_empty_input(self):
        """Test behavior with empty input."""
        with pytest.raises(ValueError):
            spectral.univariate_spectral_entropy(np.array([]))

    def test_single_value_input(self):
        """Test behavior with single value input."""
        with pytest.raises(ValueError):
            spectral.univariate_spectral_entropy(np.array([1.0]))

    def test_very_short_signal(self):
        """Test behavior with very short signals."""
        short_signal = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        entropy = spectral.univariate_spectral_entropy(
            short_signal, method="periodogram"
        )
        assert entropy > 0

    def test_invalid_method(self, white_noise):
        """Test behavior with invalid method parameter."""
        with pytest.raises(ValueError):
            spectral.univariate_spectral_entropy(white_noise, method="invalid_method")

    def test_reproducibility(self, sample_size):
        """Test that results are reproducible with same input."""
        np.random.seed(123)
        x1 = np.random.randn(sample_size)
        entropy1 = spectral.univariate_spectral_entropy(x1)

        np.random.seed(123)
        x2 = np.random.randn(sample_size)
        entropy2 = spectral.univariate_spectral_entropy(x2)

        assert abs(entropy1 - entropy2) < 1e-10


class TestMultivariateWelch:
    """Test cases for spectral._multivariate_welch function."""

    def test_output_shapes(self, multivariate_data):
        """Test that output shapes are correct."""
        _, n_features = multivariate_data.shape
        nperseg = 128

        freqs, csd = spectral._multivariate_welch(multivariate_data, nperseg)

        # Check frequency array shape (+1. for 0)
        expected_n_freqs = nperseg // 2 + 1
        assert len(freqs) == expected_n_freqs

        # Check CSD matrix shape
        assert csd.shape == (expected_n_freqs, n_features, n_features)

    def test_hermitian_property(self, multivariate_data):
        """Test that cross-spectral density matrix is Hermitian at each frequency."""
        freqs, csd_matrix = spectral._multivariate_welch(multivariate_data, nperseg=128)

        for k in range(len(freqs)):
            csd_k = csd_matrix[k]
            # Check Hermitian property: A = A^H (conjugate transpose)
            np.testing.assert_allclose(csd_k, np.conj(csd_k.T), rtol=1e-10)

    def test_diagonal_elements_real_positive(self, multivariate_data):
        """Test that diagonal elements (auto-spectra) are real and positive."""
        freqs, csd_matrix = spectral._multivariate_welch(multivariate_data, nperseg=128)

        for k in range(len(freqs)):
            for i in range(multivariate_data.shape[1]):
                diagonal_element = csd_matrix[k, i, i]
                # Should be real
                assert abs(diagonal_element.imag) < 1e-10
                # Should be positive (power spectral density)
                assert diagonal_element.real > 0

    def test_consistency_with_scipy_welch(self, multivariate_data):
        """Test that diagonal elements match scipy's welch function."""
        nperseg = 128
        freqs, csd_matrix = spectral._multivariate_welch(
            multivariate_data, nperseg=nperseg
        )

        # Compare each diagonal element with scipy's welch
        for i in range(multivariate_data.shape[1]):
            upsd = spectral.univariate_psd(
                multivariate_data[:, i], method="welch", nperseg=nperseg
            )

            # Extract diagonal elements (auto-spectrum)
            # Remove 0 frequency
            auto_spectrum = np.real(csd_matrix[1:, i, i])
            nz_mask = freqs > 0
            freqs = freqs[nz_mask]
            np.testing.assert_allclose(freqs, upsd.freqs, rtol=1e-9)
            np.testing.assert_allclose(auto_spectrum, upsd.psd, rtol=1e-9)

    def test_univariate_case(self, pure_sine_wave):
        """Test behavior with single time series (univariate case)."""
        X_univariate = pure_sine_wave.reshape(-1, 1)
        freqs, csd_matrix = spectral._multivariate_welch(X_univariate, nperseg=128)

        # Should have shape (n_freqs, 1, 1)
        assert csd_matrix.shape[1:] == (1, 1)

        # Single element should be real and positive
        for k in range(len(freqs)):
            assert abs(csd_matrix[k, 0, 0].imag) < 1e-10
            assert csd_matrix[k, 0, 0].real > 0

    def test_cross_spectrum_properties(self, sample_size):
        """Test properties of cross-spectrum between correlated signals."""
        # Create two correlated signals
        np.random.seed(42)
        t = np.linspace(0, 10, sample_size)
        x1 = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(sample_size)
        x2 = x1 + 0.1 * np.random.randn(sample_size)  # x2 is correlated with x1

        X = np.column_stack([x1, x2])
        _, csd_matrix = spectral._multivariate_welch(X, nperseg=128)

        # Cross-spectrum should be non-zero for correlated signals
        cross_spectrum_magnitude = np.abs(csd_matrix[:, 0, 1])
        assert np.mean(cross_spectrum_magnitude) > 1e-6

    def test_uncorrelated_signals(self, sample_size):
        """Test cross-spectrum for uncorrelated signals."""
        np.random.seed(42)
        x1 = np.random.randn(sample_size)
        x2 = np.random.randn(sample_size)  # Independent of x1

        X = np.column_stack([x1, x2])
        _, csd_matrix = spectral._multivariate_welch(X, nperseg=32)

        # Cross-spectrum should be close to zero for uncorrelated signals
        cross_spectrum_magnitude = np.abs(csd_matrix[:, 0, 1])
        # Use a reasonable threshold considering finite sample effects
        # SKIP for now.
        assert np.mean(cross_spectrum_magnitude) < 0.2

    def test_nperseg_parameter(self, multivariate_data):
        """Test different nperseg values."""
        freqs1, csd1 = spectral._multivariate_welch(multivariate_data, nperseg=64)
        freqs2, csd2 = spectral._multivariate_welch(multivariate_data, nperseg=128)

        # Different nperseg should give different frequency resolutions
        assert len(freqs1) != len(freqs2)
        assert csd1.shape[0] != csd2.shape[0]

        # But same number of features
        assert csd1.shape[1:] == csd2.shape[1:]

    def test_input_validation(self):
        """Test input validation."""
        # Test 1D input (should work but needs to be 2D)
        x_1d = np.random.randn(100)
        with pytest.raises(ValueError):
            spectral._multivariate_welch(x_1d, nperseg=32)

        # Test empty input
        with pytest.raises(ValueError):
            spectral._multivariate_welch(np.array([]).reshape(0, 1), nperseg=32)

    def test_large_nperseg(self, multivariate_data):
        """Test behavior when nperseg is larger than signal length."""
        n_samples = multivariate_data.shape[0]
        large_nperseg = n_samples + 100

        # Should handle gracefully (scipy will adjust nperseg internally)
        freqs, csd_matrix = spectral._multivariate_welch(
            multivariate_data, nperseg=large_nperseg
        )
        assert len(freqs) > 0
        assert csd_matrix.shape[1:] == (
            multivariate_data.shape[1],
            multivariate_data.shape[1],
        )


# Parametrized tests
@pytest.mark.parametrize("method", ["welch", "periodogram"])
def test_entropy_methods_positive(method, white_noise):
    """Parametrized test for different methods."""
    entropy = spectral.univariate_spectral_entropy(white_noise, method=method)
    assert entropy > 0


@pytest.mark.parametrize("nperseg", [32, 64, 128, 256])
def test_different_nperseg_values(nperseg, multivariate_data):
    """Parametrized test for different nperseg values."""
    if nperseg < multivariate_data.shape[0]:
        freqs, csd_matrix = spectral._multivariate_welch(
            multivariate_data, nperseg=nperseg
        )
        # remove 0 from output (welch includes 0 --> +1)
        expected_n_freqs = nperseg // 2 + 1
        assert len(freqs) == expected_n_freqs
        assert len(freqs) == csd_matrix.shape[0]


# Integration tests
class TestIntegration:
    """Integration tests combining both functions."""

    def test_entropy_of_transformed_data(self, multivariate_data):
        """Test spectral entropy of components from multivariate analysis."""
        # Get multivariate spectral analysis
        #  freqs, csd_matrix = spectral._multivariate_welch(multivariate_data, nperseg=128)

        # Extract first component (just the first original series)
        component = multivariate_data[:, 0]

        # Compute its spectral entropy
        entropy = spectral.univariate_spectral_entropy(component)

        # Should be positive and reasonable
        assert entropy > 0
        assert entropy < 10  # Reasonable upper bound

    def test_comparative_entropy_analysis(self, multivariate_data):
        """Test comparative analysis of multiple series."""
        entropies = []
        for i in range(multivariate_data.shape[1]):
            entropy = spectral.univariate_spectral_entropy(multivariate_data[:, i])
            entropies.append(entropy)

        # All entropies should be positive
        assert all(e > 0 for e in entropies)

        # Should have reasonable range
        assert max(entropies) - min(entropies) > 0


def test_omega_white_noise():
    rng = np.random.RandomState(42)
    x = rng.normal(size=10000)
    result = spectral.omega(x)
    assert result < 0.01
    assert result > 0


def test_omega_sinusoid():
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    rng = np.random.RandomState(42)

    x1 = np.sin(2 * np.pi * 0.5 * t)
    x1pn = x1 + 0.1 * rng.randn(n_samples)

    omega_x1 = spectral.omega(x1, method="periodogram")
    omega_x1pn = spectral.omega(x1pn, method="periodogram")

    assert omega_x1 > omega_x1pn
    assert omega_x1 > 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
