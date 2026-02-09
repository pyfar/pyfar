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

@pytest.mark.parametrize(("A","B","C","D"), [
    ([1., 2., 3.], [1.], [1.], [1.]),
    ([1., 2., 3.], [1.], [1.], [[1.]]),
    ([1., 2., 3.], [1.], [1.], [[1.], [2.]]),
    ([1., 2., 3.], [1.], [1.], 1.),
])
def test_tmatrix_from_abcd_broadcasting_working(frequencies, A, B, C, D):
    """Test from_abcd with valid broadcasting combinations."""
    tmat = TransmissionMatrix.from_abcd(A, B, C, D, frequencies)
    broadcast_BCD = [1., 1., 1.]

    assert isinstance(tmat, TransmissionMatrix)
    npt.assert_allclose(tmat.A.freq[0], A, atol=1e-15)
    npt.assert_allclose(tmat.B.freq[0], broadcast_BCD, atol=1e-15)
    npt.assert_allclose(tmat.C.freq[0], broadcast_BCD, atol=1e-15)
    npt.assert_allclose(tmat.D.freq[0], broadcast_BCD, atol=1e-15)


@pytest.mark.parametrize(("A","B","C","D"), [
    ([1, 2, 3], [1], [1], [1, 2]),
    ([1, 2, 3], [1], [1], [1, 2, 3, 4]),
])
def test_tmatrix_from_abcd_broadcasting_expect_error(frequencies, A, B, C, D):
    """Test from_abcd raises error for invalid broadcasting combinations."""
    with pytest.raises(
        ValueError,
        match="shape mismatch: objects cannot be broadcast to a single shape",
    ):
        TransmissionMatrix.from_abcd(A, B, C, D, frequencies)


@pytest.mark.parametrize(("A","B","C","D","frequency"), [
    ([1], [2], [3], [4], 1000),
    ([1], [2], [3], [4], [1000]),
    (1, 2, 3, 4, 1000),
    (1, 2, 3, 4, [1000]),
])
def test_tmatrix_from_abcd_single_frequency(A, B, C, D, frequency):
    """Test from_abcd with single frequency values."""
    tmat = TransmissionMatrix.from_abcd(A, B, C, D, frequency)
    assert isinstance(tmat, TransmissionMatrix)
    npt.assert_allclose(tmat.A.freq[0], A, atol=1e-15)
    npt.assert_allclose(tmat.B.freq[0], B, atol=1e-15)
    npt.assert_allclose(tmat.C.freq[0], C, atol=1e-15)
    npt.assert_allclose(tmat.D.freq[0], D, atol=1e-15)

    # Check all frequency vectors
    npt.assert_allclose(tmat.A.frequencies, frequency, atol=1e-15)
    npt.assert_allclose(tmat.B.frequencies, frequency, atol=1e-15)
    npt.assert_allclose(tmat.C.frequencies, frequency, atol=1e-15)
    npt.assert_allclose(tmat.D.frequencies, frequency, atol=1e-15)


def test_tmatrix_from_abcd_frequencies_length_mismatch(A_list):
    """Test from_abcd throws error if frequencies length does not match."""
    frequencies = [100, 200]
    with pytest.raises(
        ValueError,
        match="Number of frequency values does not match the number",
    ):
        TransmissionMatrix.from_abcd(
            A_list,
            A_list,
            A_list,
            A_list,
            frequencies,
        )

    frequencies = [100]
    with pytest.raises(
        ValueError,
        match="Number of frequency values does not match the number",
    ):
        TransmissionMatrix.from_abcd(
            A_list,
            A_list,
            A_list,
            A_list,
            frequencies,
        )

    frequencies = 100
    with pytest.raises(
        ValueError,
        match="Number of frequency values does not match the number",
    ):
        TransmissionMatrix.from_abcd(
            A_list,
            A_list,
            A_list,
            A_list,
            frequencies,
        )

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
@pytest.fixture(scope="module")
def abcd_data_complex():
    """ABCD matrices with 2 frequency bins and one additional
    dimension of size 3.
    """
    frequencies = [100, 200]
    A = FrequencyData([[1j, 1+1j]], frequencies)
    B = FrequencyData([[2j, 2+1j]], frequencies)
    C = FrequencyData([[3j, 3+1j]], frequencies)
    D = FrequencyData([[4j, 4+1j]], frequencies)
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
        assert np.allclose(tmat.A.freq, A, atol=1e-15)
        assert np.allclose(tmat.B.freq, B, atol=1e-15)
        assert np.allclose(tmat.C.freq, C, atol=1e-15)
        assert np.allclose(tmat.D.freq, D, atol=1e-15)

def test_tmatrix_abcd_entries(abcd_data_1x2, abcd_data_3x2, abcd_data_3x3x1,
        abcd_data_complex):
    """Test whether ABCD entries of T-Matrix match ABCD data used for
    initialization.
    """
    tmat, A, B, C, D = abcd_data_1x2
    _compare_tmat_vs_abcd(tmat, A, B, C, D)

    tmat, A, B, C, D = abcd_data_3x2
    _compare_tmat_vs_abcd(tmat, A, B, C, D)

    tmat, A, B, C, D = abcd_data_3x3x1
    _compare_tmat_vs_abcd(tmat, A, B, C, D)

    tmat, A, B, C, D = abcd_data_complex
    _compare_tmat_vs_abcd(tmat, A, B, C, D)


# ------------------------
# TESTS FOR CREATE METHODS
# ------------------------
def test_tmatrix_create_frequency_independent_abcd():
    """Test creation of frequency-independent ABCD matrix."""
    tmat = TransmissionMatrix.create_frequency_independent_abcd(1, 2, 3, 4)
    assert isinstance(tmat, np.ndarray)
    assert tmat.shape == (2, 2)
    npt.assert_allclose(tmat, [[1, 2], [3, 4]], atol=1e-15)

@pytest.mark.parametrize("wrong_input", ["input", [1,2], (1,2), { 'A':1 }])
def test_tmatrix_create_frequency_independent_abcd_wrong_input(wrong_input):
    """Test whether creation of frequency-independent ABCD matrix raises
    error on invalid input data.
    """
    error_msg = "A, B, C, and D must be scalars"
    with pytest.raises(TypeError, match=error_msg):
        TransmissionMatrix.create_frequency_independent_abcd(
            wrong_input, 2, 3, 4)
    with pytest.raises(TypeError, match=error_msg):
        TransmissionMatrix.create_frequency_independent_abcd(
            1, wrong_input, 3, 4)
    with pytest.raises(TypeError, match=error_msg):
        TransmissionMatrix.create_frequency_independent_abcd(
            1, 2, wrong_input, 4)
    with pytest.raises(TypeError, match=error_msg):
        TransmissionMatrix.create_frequency_independent_abcd(
            1, 2, 3, wrong_input)

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

def test_create_transmission_line_number():
    """Test `create_transmission_line` with impedance as number."""
    kl = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = 4+0.1j
    tmat = TransmissionMatrix.create_transmission_line(kl, Z)
    assert isinstance(tmat, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat,
        np.cos(kl.freq), 1j*Z*np.sin(kl.freq),
        1j/Z*np.sin(kl.freq), np.cos(kl.freq))

def test_create_transmission_line_frequency_data():
    """Test `create_transmission_line` with impedance as FrequencyData."""
    kl = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = FrequencyData([1+1j, 2+2j, 3+1j], [1, 2, 3])
    tmat = TransmissionMatrix.create_transmission_line(kl, Z)
    assert isinstance(tmat, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat,
        np.cos(kl.freq), 1j*Z.freq*np.sin(kl.freq),
        1j/Z.freq*np.sin(kl.freq), np.cos(kl.freq))

def test_create_transmission_line_broadcasting():
    """Test `create_transmission_line` broadcasting."""
    kl = FrequencyData(np.array([[1j, 2, 3j],[4j, 5, 6j]]) , [1, 2, 3])
    Z = FrequencyData([1+1j, 2+2j, 3+1j], [1, 2, 3])
    tmat = TransmissionMatrix.create_transmission_line(kl, Z)
    assert isinstance(tmat, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat,
        np.cos(kl.freq), 1j*Z.freq*np.sin(kl.freq),
        1j/Z.freq*np.sin(kl.freq), np.cos(kl.freq))

@pytest.mark.parametrize(
        "kl",
        [2j, np.array([1, 2, 3]), "term"])
def test_create_transmission_line_kl_error(kl):
    """Test `create_transmission_line` error for kl input."""
    Z = 4+0.1j
    with pytest.raises(
        ValueError, match="The input kl"):
        TransmissionMatrix.create_transmission_line(kl, Z)

@pytest.mark.parametrize("Z", [np.array([1, 2, 3]), "imp"])
def test_create_transmission_line_characteristic_impedance_errors(Z):
    """Test `create_transmission_line` errors for impedance parameter."""
    kl = FrequencyData([1j, 2, 3j], [1, 2, 3])
    with pytest.raises(
        ValueError, match="characteristic impedance must be"):
        TransmissionMatrix.create_transmission_line(kl, Z)

def test_create_transmission_line_frequency_matching():
    """Test `create_transmission_line` frequency matching."""
    kl = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = FrequencyData([1+1j, 2+2j, 3+1j], [1, 2, 4])
    with pytest.raises(
        ValueError, match="The frequencies do not match"):
        TransmissionMatrix.create_transmission_line(kl, Z)

@pytest.mark.parametrize(
    "area_narrow_end",
    [{"a": 1, "b": 2}, np.array([1, 2, 3]), "term", 0, -5, 2j],
)
def test_calculate_horn_geometry_parameters_S0_value_error(area_narrow_end):
    area_wide_end = 0.2
    horn_length = 0.35

    with pytest.raises(ValueError, match="The input area_narrow_end"):
        TransmissionMatrix.\
            _calculate_horn_geometry_parameters\
            (area_narrow_end, area_wide_end, horn_length)


@pytest.mark.parametrize(
    "area_wide_end",
    [{"a": 1, "b": 2}, np.array([1, 2, 3]), "term", 0, -5, 2j],
)
def test_calculate_horn_geometry_parameters_S1_value_error(area_wide_end):
    area_narrow_end = 0.2
    horn_length = 0.35

    with pytest.raises(ValueError, match="The input area_wide_end"):
        TransmissionMatrix.\
            _calculate_horn_geometry_parameters\
            (area_narrow_end, area_wide_end, horn_length)


@pytest.mark.parametrize(
    "horn_length",
    [{"a": 1, "b": 2}, np.array([1, 2, 3]), "term", 0, -5, 2j],
)
def test_calculate_horn_geometry_parameters_L_value_error(horn_length):
    area_narrow_end = 0.2
    area_wide_end = 0.3

    with pytest.raises(ValueError, match="The input horn_length"):
        TransmissionMatrix.\
            _calculate_horn_geometry_parameters\
            (area_narrow_end, area_wide_end, horn_length)


def test_calculate_horn_geometry_parameters_S0_larger_S1_value_error():
    area_narrow_end = 0.35
    area_wide_end = 0.3
    horn_length = 0.35

    with pytest.raises(
        ValueError,
        match="area_narrow_end must be strictly smaller than area_wide_end.",
    ):
        TransmissionMatrix.\
            _calculate_horn_geometry_parameters\
            (area_narrow_end, area_wide_end, horn_length)


def test_calculate_horn_geometry_parameters_S0_equal_S1_value_error():
    area_narrow_end = 0.35
    area_wide_end = 0.35
    horn_length = 0.2

    with pytest.raises(ValueError, match=\
        "For a conical horn area_narrow_end must be"):
        TransmissionMatrix.\
            _calculate_horn_geometry_parameters\
            (area_narrow_end, area_wide_end, horn_length)


def test_calculate_horn_geometry_calculations():
    a = 0.3
    b = 0.7
    Omega = 1.4

    area_narrow_end = Omega * a**2
    area_wide_end = Omega * b**2
    horn_length = b - a

    Omega_ret, a_ret, b_ret = (
        TransmissionMatrix._calculate_horn_geometry_parameters\
            (area_narrow_end, area_wide_end, horn_length)
    )

    assert np.isclose(Omega, Omega_ret, atol=1e-15)
    assert np.isclose(a, a_ret, atol=1e-15)
    assert np.isclose(b, b_ret, atol=1e-15)


def test_create_conical_horn_imp_number():
    """Test `create_conical_horn` with impedance as number."""
    wave_number = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = 4 + 0.1j

    a = 0.3
    b = 0.5
    Omega = 0.4

    area_narrow_end = Omega * a**2
    area_wide_end = Omega * b**2
    horn_length = b - a

    tmat_backward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        wave_number,
        Z,
        "backward",
    )
    tmat_forward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        wave_number,
        Z,
        "forward",
    )

    A = np.array([[1.02899125+0.j, 0.88607109+0.j, 1.26838249+0.j]])

    B = np.array([[-13.42240017 -0.33556j, -0.64903057+25.96122282j,
        -42.44357214 -1.0610893j ]])

    C = np.array([[-0.00328572+8.21430330e-05j, 0.00015905+6.36214741e-03j,
        -0.01037249+2.59312342e-04j]])

    D = np.array([[1.01471206+0.j, 0.94205494+0.j, 1.13571485+0.j]])

    inv_prefix = 1 / (A * D - B * C)

    assert isinstance(tmat_backward, TransmissionMatrix)
    _compare_tmat_vs_abcd(tmat_backward, A, B, C, D)

    assert isinstance(tmat_forward, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat_forward,
        inv_prefix * D,
        -1 * inv_prefix * B,
        -1 * inv_prefix * C,
        inv_prefix * A,
    )


def test_create_conical_horn_imp_frequency_data():
    """Test `create_conical_horn` with impedance as FrequencyData."""
    k = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = FrequencyData([1 + 1j, 2 + 2j, 3 + 1j], [1, 2, 3])

    a = 0.3
    b = 0.5
    Omega = 0.4

    area_narrow_end = Omega * a**2
    area_wide_end = Omega * b**2
    horn_length = b - a

    tmat_backward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
        Z,
        "backward",
    )
    tmat_forward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
        Z,
        "forward",
    )

    A = np.array([[1.02899125+0.j, 0.88607109+0.j, 1.26838249+0.j]])

    B = np.array([[ -3.35560004 -3.35560004j, -12.98061141+12.98061141j,
        -31.83267911-10.61089304j]])

    C = np.array([[-0.00657555+0.00657555j, 0.00636612+0.00636612j,
        -0.01245477+0.00415159j]])

    D = np.array([[1.01471206+0.j, 0.94205494+0.j, 1.13571485+0.j]])

    inv_prefix = 1 / (A * D - B * C)

    assert isinstance(tmat_backward, TransmissionMatrix)
    _compare_tmat_vs_abcd(tmat_backward, A, B, C, D)

    assert isinstance(tmat_forward, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat_forward,
        inv_prefix * D,
        -1 * inv_prefix * B,
        -1 * inv_prefix * C,
        inv_prefix * A,
    )

def test_create_conical_horn_k_frequency_data():
    """Test `create_conical_horn` with impedance as FrequencyData."""
    k = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = FrequencyData([1 + 1j, 2 + 2j, 3 + 1j], [1, 2, 3])

    a = 0.3
    b = 0.5
    Omega = 0.4

    area_narrow_end = Omega * a**2
    area_wide_end = Omega * b**2
    horn_length = b - a

    tmat_backward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
        Z,
        "backward",
    )
    tmat_forward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
        Z,
        "forward",
    )

    A = np.array([[1.02899125+0.j, 0.88607109+0.j, 1.26838249+0.j]])

    B = np.array([[ -3.35560004 -3.35560004j, -12.98061141+12.98061141j,
        -31.83267911-10.61089304j]])

    C = np.array([[-0.00657555+0.00657555j, 0.00636612+0.00636612j,
        -0.01245477+0.00415159j]])

    D = np.array([[1.01471206+0.j, 0.94205494+0.j, 1.13571485+0.j]])

    inv_prefix = 1 / (A * D - B * C)

    assert isinstance(tmat_backward, TransmissionMatrix)
    _compare_tmat_vs_abcd(tmat_backward, A, B, C, D)

    assert isinstance(tmat_forward, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat_forward,
        inv_prefix * D,
        -1 * inv_prefix * B,
        -1 * inv_prefix * C,
        inv_prefix * A,
    )


def test_create_conical_horn_broadcasting():
    """Test `create_conical_horn` broadcasting."""
    k = FrequencyData(np.array([[1j, 2, 3j], [4j, 5, 6j]]), [1, 2, 3])
    Z = FrequencyData([1 + 1j, 2 + 2j, 3 + 1j], [1, 2, 3])

    a = 0.3
    b = 0.5
    Omega = 0.4

    area_narrow_end = Omega * a**2
    area_wide_end = Omega * b**2
    horn_length = b - a

    tmat_backward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
        Z,
        "backward",
    )
    tmat_forward = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
        Z,
        "forward",
    )

    A = np.array([[1.02899125+0.j, 0.88607109+0.j, 1.26838249+0.j],
       [1.48896993+0.j, 0.33952319+0.j, 2.17916964+0.j]])

    B = np.array([[ -3.35560004 -3.35560004j, -12.98061141+12.98061141j,
        -31.83267911-10.61089304j],
       [-14.80176637-14.80176637j, -28.04903283+28.04903283j,
        -75.47306777-25.15768926j]])

    C = np.array([[-0.00657555+0.00657555j,  0.00636612+0.00636612j,
        -0.01245477+0.00415159j],
       [-0.0289162 +0.0289162j ,  0.01382674+0.01382674j,
        -0.02938139+0.0097938j ]])

    D = np.array([[1.01471206+0.j, 0.94205494+0.j, 1.13571485+0.j],
       [1.24651396+0.j, 0.66076978+0.j, 1.58954713+0.j]])

    inv_prefix = 1 / (A * D - B * C)

    assert isinstance(tmat_backward, TransmissionMatrix)
    _compare_tmat_vs_abcd(tmat_backward, A, B, C, D)

    assert isinstance(tmat_forward, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat_forward,
        inv_prefix * D,
        -1 * inv_prefix * B,
        -1 * inv_prefix * C,
        inv_prefix * A,
    )


def test_create_conical_horn_default_parameters():
    """Test `create_conical_horn` default parameters."""
    k = FrequencyData([1j, 2, 3j], [1, 2, 3])

    a = 0.3
    b = 0.5
    Omega = 0.4

    area_narrow_end = Omega * a**2
    area_wide_end = Omega * b**2
    horn_length = b - a

    tmat = TransmissionMatrix.create_conical_horn(
        area_narrow_end,
        area_wide_end,
        horn_length,
        k,
    )  # using default Z and default propagation_direction

    A = np.array([[1.02899125+0.j, 0.88607109+0.j, 1.26838249+0.j]])

    B = np.array([[-1386.57688918+0.j,0.+2681.87739328j,
        -4384.55682183+0.j]])

    C = np.array([[-3.18264574e-05+0.0e+00j, 0.0e+00+6.162562e-05j,
        -1.00471007e-04+0.0e+00j]])

    D = np.array([[1.01471206+0.j, 0.94205494+0.j, 1.13571485+0.j]])

    inv_prefix = 1 / (A * D - B * C)

    assert isinstance(tmat, TransmissionMatrix)
    _compare_tmat_vs_abcd(
        tmat,
        inv_prefix * D,
        -1 * inv_prefix * B,
        -1 * inv_prefix * C,
        inv_prefix * A,
    )


@pytest.mark.parametrize("k", [{"a": 1, "b": 3j}, np.array([1, 2, 3]), "term"])
def test_create_conical_horn_k_errors(k):
    """Test `create_conical_horn` error for k input."""
    Z = 4 + 0.1j
    area_narrow_end = 0.03
    area_wide_end = 0.04
    horn_length = 0.25

    with pytest.raises(TypeError, match=\
        "wave_number must be"):
        TransmissionMatrix.create_conical_horn\
            (area_narrow_end, area_wide_end, horn_length, k, Z, "backward")


@pytest.mark.parametrize("Z", [np.array([1, 2, 3]), "imp"])
def test_create_conical_horn_medium_impedance_errors(Z):
    """Test `create_conical_horn` errors for impedance parameter."""
    k = FrequencyData([1j, 2, 3j], [1, 2, 3])
    area_narrow_end = 0.03
    area_wide_end = 0.04
    horn_length = 0.25

    with pytest.raises(TypeError, match=\
        "The input medium_impedance"):
        TransmissionMatrix.create_conical_horn\
            (area_narrow_end, area_wide_end, horn_length, k, Z, "backward")


@pytest.mark.parametrize(
    "propagation_direction",
    [np.array([1, 2, 3]), 2, -5, 8j],
)
def test_create_conical_horn_propagation_direction_type_errors(
    propagation_direction,
):
    """Test `create_conical_horn` errors for propagation_direction."""
    k = FrequencyData(np.array([[1j, 2, 3j], [4j, 5, 6j]]), [1, 2, 3])
    Z = FrequencyData([1 + 1j, 2 + 2j, 3 + 1j], [1, 2, 3])

    area_narrow_end = 0.03
    area_wide_end = 0.04
    horn_length = 0.25

    with pytest.raises(TypeError, match="The input propagation_direction"):
        TransmissionMatrix.create_conical_horn(
            area_narrow_end,
            area_wide_end,
            horn_length,
            k,
            Z,
            propagation_direction,
        )


@pytest.mark.parametrize("propagation_direction", ["test", "", "forwards"])
def test_create_conical_horn_propagation_direction_value_errors(
    propagation_direction,
):
    """Test `create_conical_horn` errors for propagation_direction."""
    k = FrequencyData(np.array([[1j, 2, 3j], [4j, 5, 6j]]), [1, 2, 3])
    Z = FrequencyData([1 + 1j, 2 + 2j, 3 + 1j], [1, 2, 3])

    area_narrow_end = 0.03
    area_wide_end = 0.04
    horn_length = 0.25

    with pytest.raises(
        ValueError,
        match="The string propagation_direction must either",
    ):
        TransmissionMatrix.create_conical_horn(
            area_narrow_end,
            area_wide_end,
            horn_length,
            k,
            Z,
            propagation_direction,
        )


def test_create_conical_horn_frequency_matching():
    """Test `create_conical_horn` frequency matching."""
    k = FrequencyData([1j, 2, 3j], [1, 2, 3])
    Z = FrequencyData([1 + 1j, 2 + 2j, 3 + 1j], [1, 2, 4])

    area_narrow_end = 0.03
    area_wide_end = 0.04
    horn_length = 0.25

    with pytest.raises(ValueError, match="The frequencies of"):
        TransmissionMatrix.create_conical_horn\
            (area_narrow_end, area_wide_end, horn_length, k, Z, "backward")

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
