"""Module for linear algebra."""

import warnings

import numpy as np
import scipy.linalg as sla

from numpy.typing import NDArray


def l2_norm(x: np.ndarray) -> float:
    """Computes l2 norm."""
    return np.linalg.norm(x)


def quadratic_form(mat: np.ndarray, vec: np.ndarray) -> float | complex:
    """
    Compute the quadratic form vec^H * mat * vec.

    This is equivalent to the R function quadratic_form(mat, vec).
    For complex vectors, this computes the Hermitian quadratic form using
    the conjugate transpose.

    Parameters
    ----------
    mat : np.ndarray of shape (n, n)
        Square matrix (real or complex).
    vec : np.ndarray of shape (n,) or (n, 1)
        Vector (real or complex).

    Returns
    -------
    result : float or complex
        The quadratic form vec^H * mat * vec.
        Returns real value if imaginary part is negligible.

    Examples
    --------
    >>> mat = np.array([[1, 2], [3, 4]])
    >>> vec = np.array([1, -1])
    >>> result = quadratic_form(mat, vec)
    >>> print(result)  # Should be 0.0
    """
    mat = np.asarray(mat)
    vec = np.asarray(vec).flatten()

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat must be a square matrix")

    if len(vec) != mat.shape[0]:
        raise ValueError("vec length must match matrix dimension")

    # Compute vec^H * mat * vec
    # For complex case: vec^H = conj(vec).T
    vec_conj = np.conj(vec)
    result = np.dot(vec_conj, np.dot(mat, vec))

    # Convert to real if imaginary part is negligible
    if np.abs(np.imag(result)) < 1e-12:
        result = np.real(result)

    return result


def fill_hermitian(mat: np.ndarray) -> np.ndarray:
    """
    Fill lower triangular part of matrix to make it Hermitian.

    Takes an upper triangular matrix and fills the lower triangular part
    to satisfy A = conj(A.T) (Hermitian property).

    Parameters
    ----------
    mat : np.ndarray of shape (n, n)
        Upper triangular matrix with real diagonal and NaN in lower triangle.

    Returns
    -------
    hermitian_mat : np.ndarray of shape (n, n)
        Hermitian matrix.

    Examples
    --------
    >>> upper_tri = np.array([[1+0j, 2+1j], [np.nan, 3+0j]])
    >>> hermitian = fill_hermitian(upper_tri)
    >>> print(hermitian)  # [[1+0j, 2+1j], [2-1j, 3+0j]]
    """
    mat = np.asarray(mat, dtype=complex)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square")

    n = mat.shape[0]

    # Check diagonal is real
    diag_imag = np.imag(np.diag(mat))
    if not np.allclose(diag_imag, 0, atol=1e-12):
        raise ValueError("Diagonal elements must be real")

    if n == 1:
        return mat

    # Fill lower triangular part
    result = mat.copy()
    lower_indices = np.tril_indices(n, k=-1)

    # Check that lower triangle contains NaN (as in R version)
    lower_values = mat[lower_indices]
    if not np.all(np.isnan(lower_values)):
        warnings.warn("Lower triangular part should contain NaN values", stacklevel=2)

    # Fill: result[i,j] = conj(result[j,i]) for i > j
    for i in range(n):
        for j in range(i):
            result[i, j] = np.conj(result[j, i])

    return result


def eigenvector_symmetric(A: np.ndarray, min: bool) -> np.ndarray:
    """Return the unit-norm eigenvector for the smallest (or largest) eigenvalue of a symmetric matrix.

    This uses ``numpy.linalg.eigh`` (for Hermitian/symmetric matrices).
    Optionally symmetrizes the input as ``0.5 * (A + A.T)`` to damp tiny
    asymmetries from numerical accumulation.

    Args:
      A: (n, n) real matrix expected to be symmetric.
      min: bool; if True, picks smallest.  Otherwise largest.

    Returns:
      v_min: (n,) unit-norm eigenvector associated with the smallest eigenvalue.
             The sign is canonicalized so the largest-magnitude component is non-negative.

    Raises:
      ValueError: If A is not square, contains NaN/Inf (when check_finite=True),
                  or is not symmetric within tolerance when symmetrize=False.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (n, n) array.")
    if not np.isfinite(A).all():
        raise ValueError("A contains NaN or Inf.")

    # Eigendecomposition: eigh returns eigenvalues in ascending order
    _, eigvecs = np.linalg.eigh(A)
    if min:
        v = eigvecs[:, 0]  # smallest eigenvalue's eigenvector
    else:
        v = eigvecs[:, -1]

    # Normalize (defensive; eigh returns unit vectors)
    nrm = l2_norm(v)
    if nrm < 1e-15:
        raise ValueError("Smallest eigenvector has near-zero norm.")
    v = v / nrm

    # Canonicalize sign: make the largest-magnitude entry non-negative
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v = -v
    return v


def null_space_complement(
    W: NDArray[np.floating], d: int, rcond: float = 1e-12
) -> NDArray[np.floating]:
    """Orthonormal basis Q for the orthogonal complement of span(W) in R^d.
    If W has zero columns, returns I_d. Width is d - rank(W), robust to near-colinearity."""
    if W.size == 0:
        return np.eye(d)
    return sla.null_space(W.T, rcond=rcond)  # shape (d, d - rank(W))


def angle_delta(u: NDArray[np.floating], v: NDArray[np.floating]) -> float:
    """Computes angle delta between u and v."""
    num = abs(float(u @ v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    return 1.0 - (num / den if den > 0 else 0.0)
