import numpy as np
import numpy.testing as npt
import pytest
import re
import pyfar as pf
from pyfar import TransmissionMatrix
from pyfar import FrequencyData

@pytest.fixture(scope="module")
def frequencies():
    return [100, 200, 300]
@pytest.fixture(scope="module")
def A_list():
    """Test data for a matrix-entry (e.g. A) using a list type."""
    return [1, 2, 3]
@pytest.fixture(scope="module")
def A_np(A_list):
    """Test data for a matrix-entry (e.g. A) using an np.ndarray."""
    return np.array(A_list)
@pytest.fixture(scope="module")
def A_FreqDat(A_np, frequencies):
    """Test data for a matrix-entry (e.g. A) using a FrequencyData object."""
    return FrequencyData(A_np, frequencies)

def _expect_data_with_wrong_abcd_dims(data: np.ndarray, frequencies):
    error_msg = re.escape("'data' must have a shape like "
                          "(..., 2, 2, n_bins), e.g. (2, 2, 100).")
    with pytest.raises(ValueError, match=error_msg):
        TransmissionMatrix(data, frequencies)
    with pytest.raises(ValueError, match=error_msg):
        TransmissionMatrix.from_tmatrix(data, frequencies)
    with pytest.raises(ValueError, match=error_msg):
        TransmissionMatrix(np.ndarray.tolist(data), frequencies)
    with pytest.raises(ValueError, match=error_msg):
        TransmissionMatrix.from_tmatrix(np.ndarray.tolist(data), frequencies)

def test_tmatrix_init(frequencies):
    """Test constructor with data of valid and invalid shapes."""
    num_bins = len(frequencies)
    TransmissionMatrix(np.ones([2, 2, num_bins]), frequencies)
    TransmissionMatrix(np.ones([4, 2, 2, num_bins]), frequencies)
    TransmissionMatrix.from_tmatrix(np.ones([2, 2, num_bins]), frequencies)
    TransmissionMatrix.from_tmatrix(np.ones([4, 2, 2, num_bins]), frequencies)

    _expect_data_with_wrong_abcd_dims(
        np.ones([2, num_bins]), frequencies)
    _expect_data_with_wrong_abcd_dims(
        np.ones([3, 2, num_bins]), frequencies)
    _expect_data_with_wrong_abcd_dims(
        np.ones([2, 5, num_bins]), frequencies)
    _expect_data_with_wrong_abcd_dims(
        np.ones([7, 4, 2, num_bins]), frequencies)
    _expect_data_with_wrong_abcd_dims(
        np.ones([7, 8, 4, 2, num_bins]), frequencies)

def _expect_error_abcd_same_type(A, B, C, D):
    with pytest.raises(
        ValueError, match=
                    "If using FrequencyData objects, all matrix entries "
                    "A, B, C, D, must be FrequencyData objects."):
        TransmissionMatrix.from_abcd(A, B, C, D, 1000)

def test_tmatrix_from_abcd_input_types(frequencies, A_list, A_np, A_FreqDat):
    """Test 'from_abcd' with valid and invalid data types."""
    TransmissionMatrix.from_abcd(A_list, A_list,
                                 A_list, A_list, frequencies)
    TransmissionMatrix.from_abcd(A_np, A_np, A_list, A_np, frequencies)
    TransmissionMatrix.from_abcd(A_np, A_np, A_np, A_np, frequencies)
    TransmissionMatrix.from_abcd(A_FreqDat, A_FreqDat, A_FreqDat, A_FreqDat)

    _expect_error_abcd_same_type(A_np, A_np, A_np, A_FreqDat)
    _expect_error_abcd_same_type(A_np, A_np, A_FreqDat, A_np)
    _expect_error_abcd_same_type(A_np, A_FreqDat, A_np, A_np)
    _expect_error_abcd_same_type(A_FreqDat, A_np, A_np, A_np)

def test_tmatrix_from_abcd_optional_frequencies(A_list, A_FreqDat):
    """Test from_abcd throws error if handing in arrays but no frequencies."""
    TransmissionMatrix.from_abcd(A_FreqDat, A_FreqDat, A_FreqDat, A_FreqDat)
    with pytest.raises(ValueError, match="'frequencies' must be specified if "
                       "not using 'FrequencyData' objects as input"):
        TransmissionMatrix.from_abcd(A_list, A_list, A_list, A_list)


# -------------------------
# TESTS FOR HIGHER DIM DATA
# -------------------------
@pytest.fixture(scope="module")
def abcd_data_1x2():
    """ABCD matrices with 2 frequency bins and one additional
    dimension of size 3.
    """
    frequencies = [100, 200]
    A = FrequencyData([[1, 1]], frequencies)
    B = FrequencyData([[2, 2]], frequencies)
    C = FrequencyData([[3, 3]], frequencies)
    D = FrequencyData([[4, 4]], frequencies)
    tmat = TransmissionMatrix.from_abcd(A, B, C, D)
    return tmat, A, B, C, D
@pytest.fixture(scope="module")
def abcd_data_3x2():
    """ABCD matrices with 2 frequency bins and one additional
    dimension of size 3.
    """
    frequencies = [100, 200]
    A = FrequencyData([[1, 1], [1, 1], [1, 1]], frequencies)
    B = FrequencyData([[2, 2], [2, 2], [2, 2]], frequencies)
    C = FrequencyData([[3, 3], [3, 3], [3, 3]], frequencies)
    D = FrequencyData([[4, 4], [4, 4], [4, 4]], frequencies)
    tmat = TransmissionMatrix.from_abcd(A, B, C, D)
    return tmat, A, B, C, D
@pytest.fixture(scope="module")
def abcd_data_3x3x1():
    """ABCD matrices with 1 frequency bin and two additional
    dimensions of size 3.
    """
    A = FrequencyData(
        [[[1.1], [1.1], [1.1]], [[1.2], [1.2], [1.2]], [[1.3], [1.3], [1.3]]],
        100)
    B = A + 1
    C = A + 2
    D = A + 3
    tmat = TransmissionMatrix.from_abcd(A, B, C, D)
    return tmat, A, B, C, D

def test_tmatrix_abcd_cshape(abcd_data_1x2, abcd_data_3x2, abcd_data_3x3x1):
    """Test whether abcd_cshape matches cshape of A-property."""
    tmat, A, __, __, __ = abcd_data_1x2
    assert tmat.abcd_cshape == A.cshape
    tmat, A, __, __, __ = abcd_data_3x2
    assert tmat.abcd_cshape == A.cshape
    tmat, A, __, __, __  = abcd_data_3x3x1
    assert tmat.abcd_cshape == A.cshape

def _compare_tmat_vs_abcd(tmat, A, B, C, D):
    """Test whether ABCD entries of T-Matrix match given data sets."""
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

def test_tmatrix_abcd_entries(abcd_data_3x2, abcd_data_3x3x1):
    """Test whether ABCD entries of T-Matrix match ABCD data used for
    initialization.
    """
    tmat, A, B, C, D = abcd_data_3x2
    _compare_tmat_vs_abcd(tmat, A, B, C, D)

    tmat, A, B, C, D = abcd_data_3x3x1
    _compare_tmat_vs_abcd(tmat, A, B, C, D)


# ------------------------
# TESTS FOR CREATE METHODS
# ------------------------
def test_tmatrix_create_identity(frequencies):
    """Test whether creation of identity matrix with frequencies."""
    tmat_eye = TransmissionMatrix.create_identity(frequencies)
    assert isinstance(tmat_eye, TransmissionMatrix)
    assert tmat_eye.abcd_cshape == (1,)
    _compare_tmat_vs_abcd(tmat_eye, 1, 0, 0, 1)

@pytest.mark.parametrize("no_freqs", [None, ()])
def test_tmatrix_create_identity_scalar_input(no_freqs):
    """Test whether creation of identity matrix without frequencies."""
    eye = TransmissionMatrix.create_identity(no_freqs)
    assert isinstance(eye, np.ndarray)
    npt.assert_allclose(eye, np.eye(2), atol=1e-15)

@pytest.mark.parametrize("method_name",
                         ["create_series_impedance", "create_shunt_admittance",
                          "create_transformer", "create_gyrator"])
def test_tmatrix_create_methods_wrong_input(method_name):
    """Test create methods raise error on invalid input data."""
    if method_name == "create_series_impedance":
        input_name = "impedance"
    elif method_name == "create_shunt_admittance":
        input_name = "admittance"
    else:
        input_name = "transducer_constant"

    func = getattr(TransmissionMatrix, method_name)
    err_msg = "'" + input_name + "' "
    "must be a numerical scalar or FrequencyData object."
    with pytest.raises(ValueError, match=err_msg):
        func("wrong_input")
    with pytest.raises(ValueError, match=err_msg):
        func('wrong_input')
    with pytest.raises(ValueError, match=err_msg):
        func((1,2,3))
    with pytest.raises(ValueError, match=err_msg):
        func([1,2,3])


@pytest.mark.parametrize("abcd_cshape", [(1,), (4,5)])
def test_tmatrix_create_series_impedance(A_FreqDat, abcd_cshape):
    """Test `create_series_impedance` using FrequencyData input."""
    Z = pf.utils.broadcast_cshape(A_FreqDat, abcd_cshape)
    tmat = TransmissionMatrix.create_series_impedance(Z)
    assert isinstance(tmat, TransmissionMatrix)
    assert tmat.abcd_cshape == abcd_cshape
    _compare_tmat_vs_abcd(tmat, 1, Z.freq, 0, 1)

def test_tmatrix_create_series_impedance_scalar_input():
    """Test create series impedance using scalar input."""
    Z = 42
    tmat = TransmissionMatrix.create_series_impedance(Z)
    assert isinstance(tmat, np.ndarray)
    assert tmat.shape == (2,2)
    npt.assert_allclose(tmat, [[1, Z],[0, 1]], atol=1e-15)

@pytest.mark.parametrize("abcd_cshape", [(1,), (4,5)])
def test_tmatrix_create_shunt_admittance(A_FreqDat, abcd_cshape):
    """Test `create_shunt_admittance` using FrequencyData input."""
    Y = pf.utils.broadcast_cshape(A_FreqDat, abcd_cshape)
    tmat = TransmissionMatrix.create_shunt_admittance(Y)
    assert isinstance(tmat, TransmissionMatrix)
    assert tmat.abcd_cshape == abcd_cshape
    _compare_tmat_vs_abcd(tmat, 1, 0, Y.freq, 1)

def test_tmatrix_create_shunt_admittance_scalar_input():
    """Test `create_shunt_admittance` using scalar input."""
    Y = 42
    tmat = TransmissionMatrix.create_shunt_admittance(Y)
    assert isinstance(tmat, np.ndarray)
    assert tmat.shape == (2,2)
    npt.assert_allclose(tmat, [[1, 0],[Y, 1]], atol=1e-15)

@pytest.mark.parametrize("transducer_constant", [
    2.5, FrequencyData([2.5, 5, 10], [1, 2, 3])])
def test_tmatrix_create_transformer(transducer_constant, frequencies):
    """Test `create_transformer` for FrequencyData and scalar input."""
    tmat = TransmissionMatrix.create_transformer(transducer_constant)
    if isinstance(transducer_constant, FrequencyData):
        assert(isinstance(tmat, TransmissionMatrix))
        N = transducer_constant.freq
    else:
        assert(isinstance(tmat, np.ndarray))
        N = transducer_constant
        # Convert to T-Matrix object
        tmat = TransmissionMatrix.create_identity(frequencies) @ tmat

    Zl = 100
    Zin_expected = N*N * Zl
    Zin = tmat.input_impedance(Zl)
    npt.assert_allclose(Zin.freq, Zin_expected, atol = 1e-15)

@pytest.mark.parametrize("transducer_constant", [
    2.5, FrequencyData([2.5, 5, 10], [1, 2, 3])])
def test_tmatrix_create_gyrator(transducer_constant, frequencies):
    """Test `create_gyrator` for FrequencyData and scalar input."""
    tmat = TransmissionMatrix.create_gyrator(transducer_constant)
    if isinstance(transducer_constant, FrequencyData):
        assert(isinstance(tmat, TransmissionMatrix))
        N = transducer_constant.freq
    else:
        assert(isinstance(tmat, np.ndarray))
        N = transducer_constant
        # Convert to T-Matrix object
        tmat = TransmissionMatrix.create_identity(frequencies) @ tmat

    Zl = 100
    Zin_expected = N*N / Zl
    Zin = tmat.input_impedance(Zl)
    npt.assert_allclose(Zin.freq, Zin_expected, atol = 1e-15)


def test_tmatrix_slicing(frequencies):
    """Test whether slicing a T-Matrix object return T-Matrix or raises correct
    error for invalid keys.
    """
    eye_2x2 = TransmissionMatrix.create_identity(frequencies)
    eye_1x2x2 = pf.utils.broadcast_cshape(eye_2x2, (1, 2, 2))
    eye_3x2x2 = pf.utils.broadcast_cshape(eye_2x2, (3, 2, 2))
    eye_4x3x2x2 = pf.utils.broadcast_cshape(eye_2x2, (4, 3, 2, 2))

    npt.assert_allclose(eye_1x2x2[0].freq, eye_2x2.freq, atol = 1e-15)
    npt.assert_allclose(eye_3x2x2[0].freq, eye_2x2.freq, atol = 1e-15)
    npt.assert_allclose(eye_3x2x2[1].freq, eye_2x2.freq, atol = 1e-15)
    npt.assert_allclose(eye_3x2x2[2].freq, eye_2x2.freq, atol = 1e-15)
    npt.assert_allclose(eye_4x3x2x2[0,0].freq, eye_2x2.freq, atol = 1e-15)
    npt.assert_allclose(eye_4x3x2x2[1,2].freq, eye_2x2.freq, atol = 1e-15)

    error_msg = "Object is not indexable, since ABCD-entries " \
    "only have a single channel"
    with pytest.raises(IndexError, match=error_msg):
        eye_2x2[0]

    error_msg = "Indexed dimensions must not exceed the ABCD " \
    "channel dimension (abcd_cdim), which is "
    with pytest.raises(IndexError, match=re.escape(error_msg + "1")):
        eye_1x2x2[0,1]
    with pytest.raises(IndexError, match=re.escape(error_msg + "1")):
        eye_1x2x2[0,:]
    with pytest.raises(IndexError, match=re.escape(error_msg + "2")):
        eye_4x3x2x2[0,0,0]
    with pytest.raises(IndexError, match=re.escape(error_msg + "2")):
        eye_4x3x2x2[0,:,0]
    with pytest.raises(IndexError, match=re.escape(error_msg + "2")):
        eye_4x3x2x2[0,:,:]
