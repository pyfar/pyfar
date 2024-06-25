import numpy as np
import pytest
import re
from pyfar import TransmissionMatrix
from pyfar import FrequencyData

def _expect_data_with_wrong_abcd_dims(data: np.ndarray, frequencies):
    error_msg = re.escape("'data' must have a shape like [..., 2, 2, N], "
                          "e.g. [2, 2, 100].")
    with pytest.raises(ValueError, match=error_msg):
        TransmissionMatrix(data, frequencies)
    with pytest.raises(ValueError, match=error_msg):
        TransmissionMatrix(np.ndarray.tolist(data), frequencies)

def test_tmatrix_init():
    frequencies = [100, 200, 300]
    TransmissionMatrix(np.ones([2, 2, len(frequencies)]), frequencies)
    TransmissionMatrix(np.ones([4, 2, 2, len(frequencies)]), frequencies)
    _expect_data_with_wrong_abcd_dims(np.ones([2, len(frequencies)]), frequencies)
    _expect_data_with_wrong_abcd_dims(np.ones([3, 2, len(frequencies)]), frequencies)
    _expect_data_with_wrong_abcd_dims(np.ones([2, 5, len(frequencies)]), frequencies)
    _expect_data_with_wrong_abcd_dims(np.ones([7, 4, 2, len(frequencies)]), frequencies)
    _expect_data_with_wrong_abcd_dims(np.ones([7,8,4,2, len(frequencies)]), frequencies)


def _get_example_init_data():
    frequencies = [100, 200]
    Mat_list = [1, 1]
    Mat_np = np.array(Mat_list)
    Mat_pf = FrequencyData(Mat_np, frequencies)
    return frequencies, Mat_list, Mat_np, Mat_pf
def _expect_error_abcd_same_type(A, B, C, D):
    with pytest.raises(
        ValueError, match="A-,B-,C- and D-Matrices must be of the same type"
    ):
        TransmissionMatrix.from_abcd(A, B, C, D, 1000)

def test_tmatrix_from_abcd_input_types():
    frequencies, Mat_list, Mat_np, Mat_pf = _get_example_init_data()

    TransmissionMatrix.from_abcd(Mat_list, Mat_list, Mat_list, Mat_list, frequencies)
    TransmissionMatrix.from_abcd(Mat_np, Mat_np, Mat_np, Mat_np, frequencies)
    TransmissionMatrix.from_abcd(Mat_pf, Mat_pf, Mat_pf, Mat_pf)

    _expect_error_abcd_same_type(Mat_np, Mat_np, Mat_np, Mat_pf)
    _expect_error_abcd_same_type(Mat_np, Mat_np, Mat_pf, Mat_np)
    _expect_error_abcd_same_type(Mat_np, Mat_pf, Mat_np, Mat_np)
    _expect_error_abcd_same_type(Mat_pf, Mat_np, Mat_np, Mat_np)

def test_tmatrix_from_abcd_optional_frequencies():
    ___, Mat_list, __, Mat_pf = _get_example_init_data()
    TransmissionMatrix.from_abcd(Mat_pf, Mat_pf, Mat_pf, Mat_pf)
    with pytest.raises(ValueError, match="'frequencies' must be specified if "
                       "not using 'FrequencyData' objects as input"
    ):
        TransmissionMatrix.from_abcd(Mat_list, Mat_list, Mat_list, Mat_list)


def _example_abcd_data_3x2():
    """ABCD matrices with 2 frequency bins and one additional dimension of size 3"""
    frequencies = [100, 200]
    A = FrequencyData([[1, 1], [1, 1], [1, 1]], frequencies)
    B = FrequencyData([[2, 2], [2, 2], [2, 2]], frequencies)
    C = FrequencyData([[3, 3], [3, 3], [3, 3]], frequencies)
    D = FrequencyData([[4, 4], [4, 4], [4, 4]], frequencies)
    tmat = TransmissionMatrix.from_abcd(A, B, C, D)
    return tmat, A, B, C, D
def _example_abcd_data_3x3x1():
    """ABCD matrices with 1 frequency bin and two additional dimension of size 3"""
    A = FrequencyData(
        [[[1.1], [1.1], [1.1]], [[1.2], [1.2], [1.2]], [[1.3], [1.3], [1.3]]], 100
    )
    B = A + 1
    C = A + 2
    D = A + 3
    tmat = TransmissionMatrix.from_abcd(A, B, C, D)
    return tmat, A, B, C, D

def _compare_tmat_vs_abcd(tmat, A, B, C, D):
    if isinstance(A, FrequencyData):
        assert tmat.A == A
        assert tmat.B == B
        assert tmat.C == C
        assert tmat.D == D
    else:
        assert np.all(tmat.A.freq == A)
        assert np.all(tmat.B.freq == B)
        assert np.all(tmat.C.freq == C)
        assert np.all(tmat.D.freq == D)

def test_tmatrix_abcd_cshape():
    tmat, A, __, __, __ = _example_abcd_data_3x2()
    assert tmat.abcd_cshape == A.cshape
    tmat, A, __, __, __  = _example_abcd_data_3x3x1()
    assert tmat.abcd_cshape == A.cshape

def test_tmatrix_abcd_entries():
    tmat, A, B, C, D = _example_abcd_data_3x2()
    _compare_tmat_vs_abcd(tmat, A, B, C, D)

    tmat, A, B, C, D = _example_abcd_data_3x3x1()
    _compare_tmat_vs_abcd(tmat, A, B, C, D)



def test_create_abcd_matrix_broadcast_input_shape():
    data_cshape_0d = FrequencyData([1,2,3], [100,200,300])
    data_cshape_1d = FrequencyData([[1], [1], [3]], [100])
    data_cshape_2x3 = FrequencyData([[[1], [1], [3]],[[1], [1], [3]]], [100])
    error_msg = re.escape("'abcd_data' must have a cshape of (2,2).")
    with pytest.raises(ValueError, match = error_msg):
        TransmissionMatrix._create_abcd_matrix_broadcast(data_cshape_0d, (3,4))
    with pytest.raises(ValueError, match = error_msg):
        TransmissionMatrix._create_abcd_matrix_broadcast(data_cshape_1d, (3,4))
    with pytest.raises(ValueError, match = error_msg):
        TransmissionMatrix._create_abcd_matrix_broadcast(data_cshape_2x3, (3,4))
        TransmissionMatrix._create_abcd_matrix_broadcast(data_cshape_2x3, (3,4))

@pytest.mark.parametrize("abcd_cshape", [(), (4,), (4, 5)])
def test_create_abcd_matrix_broadcast(abcd_cshape):
    a,b,c,d = 1,2,3,4
    abcd_2d = np.array([[a,b],[c,d]])
    frequencies = [100,200,300]
    abcd_data = np.repeat(abcd_2d[:, :, np.newaxis], len(frequencies), axis = 2)
    tmat = TransmissionMatrix._create_abcd_matrix_broadcast(
        FrequencyData(abcd_data, frequencies), abcd_cshape)
    assert tmat.abcd_cshape == abcd_cshape
    _compare_tmat_vs_abcd(tmat, a, b, c, d)

@pytest.mark.parametrize("abcd_cshape", [(), (4,), (4, 5)])
def test_tmatrix_create_indentity(abcd_cshape):
    frequencies = [100,200,300]
    identity_tmat = TransmissionMatrix.create_identity(frequencies, abcd_cshape)
    assert identity_tmat.abcd_cshape == abcd_cshape
    _compare_tmat_vs_abcd(identity_tmat, 1, 0, 0, 1)

def test_tmatrix_create_series_impedance(abcd_cshape = (4,5)):
    with pytest.raises(ValueError, match = "Number of channels for "
                       "'impedance' must be 1."):
        TransmissionMatrix.create_series_impedance(
            FrequencyData([[1,2], [3,4]],[10,20]))

    frequencies = [100,200,300]
    Z = FrequencyData([10, 20, 30], frequencies)
    tmat = TransmissionMatrix.create_series_impedance(Z, abcd_cshape)
    assert tmat.abcd_cshape == abcd_cshape
    _compare_tmat_vs_abcd(tmat, 1, Z.freq, 0, 1)

def test_tmatrix_create_shunt_admittance(abcd_cshape = (4,5)):
    with pytest.raises(ValueError, match = "Number of channels for "
                       "'admittance' must be 1."):
        TransmissionMatrix.create_shunt_admittance(
            FrequencyData([[1,2], [3,4]],[10,20]))

    frequencies = [100,200,300]
    Y = FrequencyData([10, 20, 30], frequencies)
    tmat = TransmissionMatrix.create_shunt_admittance(Y, abcd_cshape)
    assert tmat.abcd_cshape == abcd_cshape
    _compare_tmat_vs_abcd(tmat, 1, 0, Y.freq, 1)


if __name__ == "__main__":
    # test_tmatrix_from_abcd_optional_frequencies()
    # test_tmatrix_from_abcd_input_types()
    # test_tmatrix_abcd_cshape()
    # test_tmatrix_abcd_entries()
    test_create_abcd_matrix_broadcast_input_shape()
    # test_create_abcd_matrix_broadcast((4,5))
    # test_tmatrix_create_indentity((4,5))
    # test_tmatrix_create_series_impedance((4,5))
    # test_tmatrix_create_shunt_admittance((4,5))
