import numpy as np
import numpy.testing as npt
import pytest
import re
from pyfar import TransmissionMatrix
from pyfar import FrequencyData


@pytest.fixture(scope="module")
def frequencies():
    return [100,200,300]

#---------------
#| INPUT TESTS |
#---------------
@pytest.fixture(scope="module", params=
                ["input_impedance", "output_impedance",
                 "TF_voltage", "TF_current"])
def parameter_function(request):
    """Parametrized fixture with names of methods for derived parameters."""
    return request.param
@pytest.fixture(scope="module", params=
                ["abcd_cshape()", "abcd_cshape(4,)", "abcd_cshape(4,5)"])
def abcd_cshape(request):
    """Parametrized fixture for different abcd_cshapes."""
    if request.param == "abcd_cshape()":
        return ()
    elif request.param == "abcd_cshape(4,)":
        return (4,)
    elif request.param == "abcd_cshape(4,5)":
        return (4,5)
@pytest.fixture(scope="module")
def tmatrix_random_data(abcd_cshape, frequencies):
    """Fixture creating T-Matrix with rng data for different abcd_cshapes."""
    shape = (*abcd_cshape, 2,2, len(frequencies))
    rng = np.random.default_rng()
    data = rng.uniform(0.0001, 2000, shape)
    return TransmissionMatrix(data, frequencies)

@pytest.fixture(scope="module", params=
                ["scalar", "FrequencyData_vector",
                 "FrequencyData_abcd_cshape"])
def load_impedance_with_correct_format(request, frequencies, abcd_cshape):
    """Fixture creating load_impedance in different formats:
    1) scalar,
    2) array matching the frequency vector,
    3) FrequencyData object match abcd_cshape.
    """
    if request.param == "scalar":
        return 2
    elif request.param == "FrequencyData_vector":
        return FrequencyData(np.ones_like(frequencies)*2, frequencies)
    elif request.param == "FrequencyData_abcd_cshape":
        shape = (*abcd_cshape, len(frequencies))
        return FrequencyData(np.ones(shape), frequencies)

@pytest.mark.parametrize("impedance_type", ["input", "output"])
def test_valid_input_formats_impedance(impedance_type,
                                         load_impedance_with_correct_format,
                                         tmatrix_random_data):
    """Test whether input/output impedance runs without errors for Zl with
    valid input formats.
    """
    Zl = load_impedance_with_correct_format
    if impedance_type == "input":
        tmatrix_random_data.input_impedance(Zl)
    elif impedance_type == "output":
        tmatrix_random_data.output_impedance(Zl)

@pytest.fixture(scope="module")
def simple_tmat():
    return TransmissionMatrix(np.ones([2,2,1]), 100)
@pytest.mark.parametrize("quantity_indices",
                         [(0,0), (0,1), (1,0), (1,1), [0,0], np.array([1,0])])
def test_TF_valid_quantity_input(quantity_indices, simple_tmat):
    """Test whether TF method runs without errors for valid input types
    for indices.
    """
    simple_tmat.transfer_function(quantity_indices, np.inf)

@pytest.mark.parametrize("quantity_indices",
                         [1, "string", (1, "string"), [1,1,1]])
def test_TF_quantity_input_wrong_numel(quantity_indices, simple_tmat):
    """Test whether TF method raises error for invalid types or size
    of indices.
    """
    error_msg = re.escape("'quantity_indices' must be an array-like type "
                          "with two numeric elements.")
    with pytest.raises(ValueError, match=error_msg):
        simple_tmat.transfer_function(quantity_indices, np.inf)

@pytest.mark.parametrize("quantity_indices", [(-1,0), (0, 1.9)])
def test_TF_quantity_input_wrong_ints(quantity_indices, simple_tmat):
    """Test whether TF method raises error for invalid integer values
    for indices.
    """
    error_msg = re.escape("'quantity_indices' must contain two "
                          "integers between 0 and 1.")
    with pytest.raises(ValueError, match=error_msg):
        simple_tmat.transfer_function(quantity_indices, np.inf)

def test_TF_valid_load_input_format(load_impedance_with_correct_format,
                                            tmatrix_random_data):
    """Test whether TF method runs without errors for valid Zl input."""
    Zl = load_impedance_with_correct_format
    tmatrix_random_data.transfer_function((1,1), Zl)


#---------------
#| RESULT TESTS|
#---------------
@pytest.fixture(scope="module")
def impedance_random(frequencies) -> FrequencyData:
    """Fixture returning impedance as FrequencyData object with random data."""
    rng = np.random.default_rng()
    return FrequencyData(rng.random(len(frequencies)), frequencies)

@pytest.fixture(scope="module",
                params=["random_load", "inf_load", "zero_load", "mixed_load"])
def load_impedance(request, frequencies) -> FrequencyData:
    """Parametrized fixture for load impedance as FrequencyData object with
    different types of data:
    1) random, 2) infinite load, 3) zero load, 4) mix of all.
    """
    if request.param == "random_load":
        rng = np.random.default_rng()
        return FrequencyData(rng.random(len(frequencies)), frequencies)
    elif request.param == "inf_load":
        return FrequencyData(np.ones_like(frequencies)*np.inf, frequencies)
    elif request.param == "zero_load":
        return FrequencyData(np.zeros_like(frequencies), frequencies)
    elif request.param == "mixed_load":
        return FrequencyData([0, 2, np.inf], frequencies)

def _special_twoport_tmatrix(
        twoport_type, Zl: FrequencyData, Z: FrequencyData):
    """
    Returns a T-Matrix representing a twoport for special circuit types.

    1) By-pass system
    2) A series impedance != load_impedance
    3) A parallel impedance != load_impedance
    4) A series impedance == load_impedance
    5) A parallel impedance == load_impedance
    """
    if twoport_type == "bypass":
        return TransmissionMatrix.create_identity(Zl.frequencies)
    if twoport_type == "series_impedance":
        return TransmissionMatrix.create_series_impedance(Z)
    if twoport_type == "parallel_impedance":
        return TransmissionMatrix.create_shunt_admittance(1/Z)
    if twoport_type == "series_load_impedance":
        return TransmissionMatrix.create_series_impedance(Zl)
    if twoport_type == "parallel_load_impedance":
        return TransmissionMatrix.create_shunt_admittance(1/Zl)
    else:
        raise ValueError("Unexpected value for 'twoport_type'")

def _twoport_type_list():
    """A list including all valid twoport types used in the tests."""
    return ["bypass", "series_impedance", "parallel_impedance",
            "series_load_impedance", "parallel_load_impedance"]

def _expected_impedance(
        twoport_type, Zl: FrequencyData, Z: FrequencyData) -> FrequencyData:
    """Returns the expected input/output impedance for testes twoport types."""
    if twoport_type == "bypass":
        Zexpected =  Zl
    elif twoport_type == "series_impedance":
        Zexpected =  Zl + Z
    elif twoport_type == "parallel_impedance":
        Zexpected =  1/(1/Zl + 1/Z)
    elif twoport_type == "series_load_impedance":
        Zexpected =  2 * Zl
        Zexpected.freq[np.isinf(Zl.freq)] = np.nan #non-phyical case
    elif twoport_type == "parallel_load_impedance":
        Zexpected =  Zl / 2
        Zexpected.freq[Zl.freq == 0] = np.nan #non-phyical case
    else:
        raise ValueError("Unexpected value for 'twoport_type'")
    return Zexpected

def _expected_voltage_tf(
        twoport_type, Zl: FrequencyData, Z: FrequencyData) -> FrequencyData:
    """Returns the expected TF Uout/Uin for testes twoport types."""
    if twoport_type not in _twoport_type_list():
        raise ValueError("Unexpected value for 'twoport_type'")

    if twoport_type == "series_impedance":
        TF = Zl / (Zl + Z)
    else:
        TF = FrequencyData(np.ones_like(Zl.freq), Zl.frequencies)

    # Special TFs for zero of inf load
    TF.freq[Zl.freq == 0] = 0
    TF.freq[np.isinf(Zl.freq)] = 1

    if twoport_type == "series_load_impedance":
        TF = TF * (1/2)
        TF.freq[np.isinf(Zl.freq)] = np.nan
    return TF

def _expected_current_to_voltage_tf(
        twoport_type, Zl: FrequencyData, Z: FrequencyData) -> FrequencyData:
    """Returns the expected TF Iout/Uin for testes twoport types."""
    if twoport_type not in _twoport_type_list():
        raise ValueError("Unexpected value for 'twoport_type'")

    if twoport_type == "series_impedance":
        TF = 1 / (Zl + Z)
    elif twoport_type == "series_load_impedance":
        TF = 1 / (2 * Zl)
        TF.freq[Zl.freq == 0] = np.nan
    else:
        TF = 1 / Zl
        TF.freq[Zl.freq == 0] = np.nan

    return TF

def _expected_current_tf(
        twoport_type, Zl: FrequencyData, Z: FrequencyData) -> FrequencyData:
    """Returns the expected TF Iout/Iin for testes twoport types."""
    if twoport_type not in _twoport_type_list():
        raise ValueError("Unexpected value for 'twoport_type'")

    if twoport_type == "parallel_impedance":
        TF = Z / (Zl + Z)
    else:
        TF = FrequencyData(np.ones_like(Zl.freq), Zl.frequencies)

    # Special TFs for zero of inf load
    TF.freq[Zl.freq == 0] = 1
    TF.freq[np.isinf(Zl.freq)] = 0

    if twoport_type == "parallel_load_impedance":
        TF = TF * (1/2)
        TF.freq[Zl.freq == 0] = np.nan
    return TF

def _expected_voltage_to_current_tf(
        twoport_type, Zl: FrequencyData, Z: FrequencyData) -> FrequencyData:
    """Returns the expected TF Uout/Iin for testes twoport types."""
    if twoport_type not in _twoport_type_list():
        raise ValueError("Unexpected value for 'twoport_type'")

    idx_inf = np.isinf(Zl.freq)
    if twoport_type == "parallel_impedance":
        TF = Z * Zl / (Zl + Z)
        TF.freq[idx_inf] = Z.freq[idx_inf]
    elif twoport_type == "parallel_load_impedance":
        TF = Zl / 2
        TF.freq[Zl.freq == 0] = np.nan
        TF.freq[idx_inf] = np.nan
    else: #bypass, series_impedance + series_load_impedance
        TF = Zl.copy()
        TF.freq[idx_inf] = np.nan

    return TF

@pytest.mark.parametrize("impedance_type", ["input", "output"])
@pytest.mark.parametrize("twoport_type", _twoport_type_list())
def test_input_impedance(impedance_type : str, twoport_type : str,
                         load_impedance : FrequencyData, impedance_random):
    """Significantly parametrized test for result of input/output impedance
    method.
    """
    tmat = _special_twoport_tmatrix(
        twoport_type, load_impedance, impedance_random)
    if impedance_type == "input":
        Zres = tmat.input_impedance(load_impedance)
    else:
        Zres = tmat.output_impedance(load_impedance)
    Zexpected = _expected_impedance(
        twoport_type, load_impedance, impedance_random)

    idx_inf = Zexpected.freq == np.inf
    idx_default = np.logical_not(idx_inf)
    npt.assert_allclose(
        Zres.freq[idx_default], Zexpected.freq[idx_default], atol=1e-15)
    assert(np.all(np.abs(Zres.freq[idx_inf]) > 1e15))

@pytest.mark.parametrize("tf_type", ["voltage", "current", "voltage/current",
                                     "current/voltage"])
@pytest.mark.parametrize("twoport_type", _twoport_type_list())
def test_transfer_function(tf_type, twoport_type,
                           load_impedance, impedance_random):
    """Significantly parametrized test for result of TF method."""
    tmat = _special_twoport_tmatrix(
        twoport_type, load_impedance, impedance_random)

    if tf_type == "voltage":
        quantity_indices = (0,0)
        TF_expected = _expected_voltage_tf(twoport_type,
                                           load_impedance, impedance_random)
    elif tf_type == "current":
        quantity_indices = (1,1)
        TF_expected = _expected_current_tf(twoport_type,
                                           load_impedance, impedance_random)
    elif tf_type == "voltage/current":
        quantity_indices = (0,1)
        TF_expected = _expected_voltage_to_current_tf(
            twoport_type, load_impedance, impedance_random)
    elif tf_type == "current/voltage":
        quantity_indices = (1,0)
        TF_expected = _expected_current_to_voltage_tf(
            twoport_type, load_impedance, impedance_random)
    else:
        raise ValueError("Unexpected value for 'tf_type'.")

    TF = tmat.transfer_function(quantity_indices, load_impedance)
    if isinstance(TF_expected, FrequencyData):
        TF_expected = TF_expected.freq
    npt.assert_allclose(TF.freq, TF_expected, atol=1e-15)
