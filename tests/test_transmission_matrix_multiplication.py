import numpy as np
import numpy.testing as npt
import pytest
from pyfar import TransmissionMatrix


def _abcd_matrix_stack(abcd, shape_extra_dims=None,
                       frequencies=[100, 200, 300]):
    """Create T-Matrix with given extra-dimensions for testing purposes."""
    n_freq_bins = len(frequencies)
    if shape_extra_dims is None:
        shape = n_freq_bins
    else:
        shape = np.append(shape_extra_dims, n_freq_bins)

    A = np.ones(shape, dtype=type(abcd[0])) * abcd[0]
    B = np.ones(shape, dtype=type(abcd[1])) * abcd[1]
    C = np.ones(shape, dtype=type(abcd[2])) * abcd[2]
    D = np.ones(shape, dtype=type(abcd[3])) * abcd[3]
    return TransmissionMatrix.from_abcd(A, B, C, D, frequencies)

def _check_matrix_multiplication_result(
    abcd_mat1,
    abcd_mat2,
    abcd_target,
    shape_extra_dims=None,
    frequencies=[100, 200, 300],
):
    """Assertion check if matrix multiplication result matches given target."""
    tmat1 = _abcd_matrix_stack(abcd_mat1, shape_extra_dims, frequencies)
    tmat2 = _abcd_matrix_stack(abcd_mat2, shape_extra_dims, frequencies)
    tmat_out = tmat1 @ tmat2
    npt.assert_allclose(tmat_out.A.freq, abcd_target[0], atol=1e-15)
    npt.assert_allclose(tmat_out.B.freq, abcd_target[1], atol=1e-15)
    npt.assert_allclose(tmat_out.C.freq, abcd_target[2], atol=1e-15)
    npt.assert_allclose(tmat_out.D.freq, abcd_target[3], atol=1e-15)


@pytest.mark.parametrize("shape_extra_dims", [None, 4, [4, 5]])
def test_tmatrix_multiplication_identity(
    shape_extra_dims, frequencies=[100, 200, 300]):
    """Test result of T-Matrix multiplication with eye matrix."""
    rng = np.random.default_rng()
    abcd_identity = (1, 0, 0, 1)
    abcd_rng = rng.random(4)
    _check_matrix_multiplication_result(
        abcd_rng, abcd_identity, abcd_rng, shape_extra_dims, frequencies)

@pytest.mark.parametrize("shape_extra_dims", [None, 4, [4, 5]])
def test_tmatrix_multiplication_negative_identity(
    shape_extra_dims, frequencies=[100, 200, 300]):
    """Test result of T-Matrix multiplication with negated eye matrix."""
    rng = np.random.default_rng()
    abcd_identity = (-1, 0, 0, -1)
    abcd_rng = rng.random(4)
    _check_matrix_multiplication_result(
        abcd_rng, abcd_identity, -abcd_rng, shape_extra_dims, frequencies)

@pytest.mark.parametrize("shape_extra_dims", [None, 4, [4, 5]])
def test_tmatrix_multiplication_bottom_row_zero(
    shape_extra_dims, frequencies=[100, 200, 300]):
    """Test result of T-Matrix multiplication with bottom row zero matrix.

    Inputs:
        [ 2  1 ]    [ a  b ]
        [ 0  0 ]    [ c  d ]
    Expected Output:
        [2a+c  2b+d]
        [  0     0 ]
    """
    rng = np.random.default_rng()
    abcd_ApluB = (2, 1, 0, 0)
    abcd_rng = rng.random(4)
    abcd_target = (2 * abcd_rng[0] + abcd_rng[2],
                   2 * abcd_rng[1] + abcd_rng[3], 0, 0)
    _check_matrix_multiplication_result(
        abcd_ApluB, abcd_rng, abcd_target, shape_extra_dims, frequencies)

@pytest.mark.parametrize("shape_extra_dims", [None, 4, [4, 5]])
def test_tmatrix_multiplication_random(
    shape_extra_dims, frequencies=[100, 200, 300]):
    """
    Test T-Matrix multiplication with random matrices.

    Inputs:
        [ a1  b1 ]    [ a2  b2 ]
        [ c1  d1 ]    [ c2  d2 ]
    Expected Output:
        [a1*a2 + b1*c2   a1*b2 + b1*d2]
        [c1*a2 + d1*c2   c1*b2 + d1*d2]
    """
    rng = np.random.default_rng()
    abcd_rng1 = rng.random(4)
    abcd_rng2 = rng.random(4)
    a1, b1, c1, d1 = abcd_rng1[0], abcd_rng1[1], abcd_rng1[2], abcd_rng1[3]
    a2, b2, c2, d2 = abcd_rng2[0], abcd_rng2[1], abcd_rng2[2], abcd_rng2[3]
    abcd_target = (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
    )
    _check_matrix_multiplication_result(
        abcd_rng1, abcd_rng2, abcd_target, shape_extra_dims, frequencies)
