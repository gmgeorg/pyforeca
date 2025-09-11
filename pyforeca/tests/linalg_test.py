"""Tests for linalg"""

import numpy as np
import pytest

from pyforeca import linalg


# Fixtures for test data
@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def simple_real_matrix():
    """Simple 2x2 real matrix for testing."""
    return np.array([[1, 2], [3, 4]], dtype=float)


@pytest.fixture
def simple_real_vector():
    """Simple real vector for testing."""
    return np.array([1, -1], dtype=float)


@pytest.fixture
def hermitian_matrix():
    """Simple Hermitian matrix for testing."""
    return np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]], dtype=complex)


@pytest.fixture
def complex_vector():
    """Simple complex vector for testing."""
    return np.array([1 + 1j, 1 - 1j], dtype=complex)


def _rand_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """QR-based random orthogonal matrix with determinant +1."""
    M = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class TestFillHermitian:
    """Test cases for linalg.fill_hermitian function."""

    def test_basic_functionality(self):
        """Test basic Hermitian filling."""
        # Create upper triangular matrix with NaN in lower triangle
        upper = np.array(
            [
                [1 + 0j, 2 + 1j, 3 - 2j],
                [np.nan, 4 + 0j, 5 + 3j],
                [np.nan, np.nan, 6 + 0j],
            ],
            dtype=complex,
        )

        result = linalg.fill_hermitian(upper)

        # Check Hermitian property: A = conj(A.T)
        np.testing.assert_allclose(result, np.conj(result.T), rtol=1e-12)

        # Check specific values
        assert result[1, 0] == np.conj(result[0, 1])  # Should be 2-1j
        assert result[2, 0] == np.conj(result[0, 2])  # Should be 3+2j
        assert result[2, 1] == np.conj(result[1, 2])  # Should be 5-3j

    def test_real_matrix(self):
        """Test with real matrix (should become symmetric)."""
        upper = np.array(
            [[1, 2, 3], [np.nan, 4, 5], [np.nan, np.nan, 6]], dtype=complex
        )

        result = linalg.fill_hermitian(upper)

        # Should be symmetric for real matrices
        np.testing.assert_allclose(result, result.T, rtol=1e-12)

        # Check specific values
        assert result[1, 0] == 2
        assert result[2, 0] == 3
        assert result[2, 1] == 5

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""
        single = np.array([[5 + 0j]], dtype=complex)
        result = linalg.fill_hermitian(single)

        np.testing.assert_array_equal(result, single)

    def test_diagonal_validation(self):
        """Test validation that diagonal elements are real."""
        # Matrix with complex diagonal element
        bad_matrix = np.array([[1 + 1j, 2], [np.nan, 3]], dtype=complex)

        with pytest.raises(ValueError, match="Diagonal elements must be real"):
            linalg.fill_hermitian(bad_matrix)

    def test_non_square_matrix(self):
        """Test error with non-square matrix."""
        non_square = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)

        with pytest.raises(ValueError, match="Matrix must be square"):
            linalg.fill_hermitian(non_square)

    def test_preservation_of_upper_triangle(self):
        """Test that upper triangular elements are preserved."""
        original_upper = np.array([[1 + 0j, 2 + 1j], [np.nan, 3 + 0j]], dtype=complex)
        result = linalg.fill_hermitian(original_upper)

        # Upper triangular part should be unchanged
        assert result[0, 0] == 1 + 0j
        assert result[0, 1] == 2 + 1j
        assert result[1, 1] == 3 + 0j

        # Lower triangular part should be conjugate
        assert result[1, 0] == 2 - 1j


@pytest.mark.parametrize("matrix_size", [1, 2, 4, 8])
def test_quadratic_form_identity_parametrized(matrix_size):
    """Parametrized test for quadratic form with identity matrices."""
    identity_mat = np.eye(matrix_size)
    vec = np.random.randn(matrix_size)

    result = linalg.quadratic_form(identity_mat, vec)
    expected = np.dot(vec, vec)

    assert np.isclose(result, expected, rtol=1e-12)


def test_quadratic_form_examples():
    """Parametrized test for quadratic form with identity matrices."""
    # Real case
    mat_real = np.array([[1, 2], [3, 4]], dtype=float)
    vec_real = np.array([1, -1], dtype=float)
    result_real = linalg.quadratic_form(mat_real, vec_real)

    assert result_real == pytest.approx(0.0, 0.00001)

    # Complex case
    mat_complex = np.array([[1 + 0j, 2 + 1j], [2 - 1j, 3 + 0j]], dtype=complex)
    vec_complex = np.array([1 + 1j, 1 - 1j], dtype=complex)
    result_complex = linalg.quadratic_form(mat_complex, vec_complex)

    assert result_complex == pytest.approx(12.0, 0.0001)


class TestQuadraticForm:
    """Test cases for linalg.quadratic_form function."""

    def test_real_case(self, simple_real_matrix, simple_real_vector):
        """Test quadratic form with real matrix and vector."""
        result = linalg.quadratic_form(simple_real_matrix, simple_real_vector)
        # Manual calculation: [1, -1] @ [[1, 2], [3, 4]] @ [1, -1]^T = [1, -1] @ [-1, -1]^T = -2
        expected = 1 * (-1) + (-1) * (-1)  # = -1 + 1 = 0, wait let me recalculate
        # [1, -1] @ [[1, 2], [3, 4]] = [1-3, 2-4] = [-2, -2]
        # [-2, -2] @ [1, -1]^T = -2*1 + (-2)*(-1) = -2 + 2 = 0
        assert result == expected
        assert isinstance(result, (int, float, np.integer, np.floating))

    def test_complex_hermitian_case(self, hermitian_matrix, complex_vector):
        """Test quadratic form with Hermitian matrix and complex vector."""
        result = linalg.quadratic_form(hermitian_matrix, complex_vector)
        # Result should be real for Hermitian matrices
        assert np.isreal(result) or np.abs(np.imag(result)) < 1e-12

    def test_identity_matrix(self):
        """Test quadratic form with identity matrix."""
        identity_mat = np.eye(3)
        vec = np.array([1, 2, 3])
        result = linalg.quadratic_form(identity_mat, vec)
        expected = np.dot(vec, vec)  # Should equal ||vec||^2
        assert np.isclose(result, expected)

    def test_vector_dimension_mismatch(self, simple_real_matrix):
        """Test error when vector dimension doesn't match matrix."""
        wrong_vec = np.array([1, 2, 3])  # 3 elements for 2x2 matrix
        with pytest.raises(ValueError, match="vec length must match matrix dimension"):
            linalg.quadratic_form(simple_real_matrix, wrong_vec)

    def test_non_square_matrix(self):
        """Test error with non-square matrix."""
        non_square = np.array([[1, 2, 3], [4, 5, 6]])
        vec = np.array([1, 2])
        with pytest.raises(ValueError, match="mat must be a square matrix"):
            linalg.quadratic_form(non_square, vec)

    def test_positive_definite_property(self):
        """Test that positive definite matrices give positive quadratic forms."""
        # Create positive definite matrix A = B^T @ B
        B = np.random.randn(3, 4)
        A = B.T @ B + np.eye(4)  # Add identity to ensure strict positive definiteness

        for _ in range(10):  # Test with multiple random vectors
            vec = np.random.randn(4)
            if np.linalg.norm(vec) > 1e-10:  # Avoid zero vector
                result = linalg.quadratic_form(A, vec)
                assert result > 0, (
                    f"Positive definite matrix should give positive result, got {result}"
                )

    def test_input_types(self, simple_real_matrix):
        """Test different input types are handled correctly."""
        vec_list = [1, -1]
        vec_array = np.array([1, -1])

        result_list = linalg.quadratic_form(simple_real_matrix, vec_list)
        result_array = linalg.quadratic_form(simple_real_matrix, vec_array)

        assert np.isclose(result_list, result_array)


def test_spd_smallest_and_largest_match_ground_truth():
    """For A = Q diag(d) Q^T with ascending d, smallest/largest eigvecs are Q[:,0] and Q[:,-1] up to sign."""
    rng = np.random.default_rng(0)
    n = 6
    Q = _rand_orthogonal(n, rng)
    d = np.array([0.05, 0.3, 0.9, 2.0, 3.5, 7.0])  # ascending
    A = Q @ np.diag(d) @ Q.T

    v_min = linalg.eigenvector_symmetric(A, min=True)
    v_max = linalg.eigenvector_symmetric(A, min=False)

    # Align signs for comparison (eigenvectors are unique up to sign)
    if np.dot(v_min, Q[:, 0]) < 0:
        v_min = -v_min
    if np.dot(v_max, Q[:, -1]) < 0:
        v_max = -v_max

    assert np.allclose(v_min, Q[:, 0], atol=1e-10)
    assert np.allclose(v_max, Q[:, -1], atol=1e-10)

    # Unit norm
    assert np.isclose(np.linalg.norm(v_min), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(v_max), 1.0, atol=1e-12)


def test_degenerate_extreme_eigenvalue_rayleigh_quotient():
    """When the smallest (or largest) eigenvalue is repeated, Rayleigh quotient should equal that value."""
    # Smallest eigenvalue multiplicity 2
    A_small = np.diag([1.0, 1.0, 2.0, 5.0])
    v_min = linalg.eigenvector_symmetric(A_small, min=True)
    rq_min = float(v_min @ (A_small @ v_min))
    assert np.isclose(rq_min, 1.0, atol=1e-12)

    # Largest eigenvalue multiplicity 2
    A_large = np.diag([0.2, 3.5, 3.5, 2.0])
    v_max = linalg.eigenvector_symmetric(A_large, min=False)
    rq_max = float(v_max @ (A_large @ v_max))
    assert np.isclose(rq_max, 3.5, atol=1e-12)

    # Also verify Av ≈ λ v
    assert np.allclose(A_small @ v_min, rq_min * v_min, atol=1e-10)
    assert np.allclose(A_large @ v_max, rq_max * v_max, atol=1e-10)


def test_sign_canonicalization_largest_abs_component_nonnegative():
    """Ensure the returned vector’s largest-magnitude component is nonnegative."""
    rng = np.random.default_rng(123)
    A = rng.standard_normal((5, 5))
    A = 0.5 * (A + A.T)  # make symmetric

    v = linalg.eigenvector_symmetric(A, min=True)
    idx = int(np.argmax(np.abs(v)))
    assert v[idx] >= 0.0

    v2 = linalg.eigenvector_symmetric(A, min=False)
    idx2 = int(np.argmax(np.abs(v2)))
    assert v2[idx2] >= 0.0


def test_matches_numpy_eigh_columns_up_to_sign():
    """Compare against NumPy eigh output (columns 0 and -1) up to sign."""
    rng = np.random.default_rng(9)
    M = rng.standard_normal((7, 7))
    A = 0.5 * (M + M.T)

    w, V = np.linalg.eigh(A)  # ascending eigenvalues
    v_min = linalg.eigenvector_symmetric(A, min=True)
    v_max = linalg.eigenvector_symmetric(A, min=False)

    # Choose signs to maximize alignment
    if np.dot(v_min, V[:, 0]) < 0:
        V[:, 0] = -V[:, 0]
    if np.dot(v_max, V[:, -1]) < 0:
        V[:, -1] = -V[:, -1]

    assert np.allclose(v_min, V[:, 0], atol=1e-10)
    assert np.allclose(v_max, V[:, -1], atol=1e-10)


def test_1x1_matrix():
    v = linalg.eigenvector_symmetric(np.array([[2.5]]), min=True)
    assert v.shape == (1,)
    assert np.allclose(v, np.array([1.0]))


def test_input_validation_non_square_and_nan():
    with pytest.raises(ValueError, match="square"):
        linalg.eigenvector_symmetric(np.ones((3, 2)), min=True)

    A = np.eye(3)
    A[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        linalg.eigenvector_symmetric(A, min=False)
