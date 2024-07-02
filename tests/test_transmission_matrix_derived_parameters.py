import numpy as np
import numpy.testing as npt
import pytest
from pyfar import TransmissionMatrix
from pyfar import FrequencyData


@pytest.fixture(scope="module")
def frequencies():
    return [100,200,300]

#---------------
#| INPUT TESTS |
#---------------
@pytest.fixture(scope="module", params=
                ["input_impedance", "output_impedance", "TF_voltage", "TF_current"])
def parameter_function(request):
    return request.param
@pytest.fixture(scope="module", params=
                ["abcd_cshape[]", "abcd_cshape[4]", "abcd_cshape[4,5]"])
def abcd_cshape(request):
    if request.param == "abcd_cshape[]":
        return []
    elif request.param == "abcd_cshape[4]":
        return [4]
    elif request.param == "abcd_cshape[4,5]":
        return [4,5]
@pytest.fixture(scope="module")
def tmatrix_random_data(abcd_cshape, frequencies):
    data = np.random.uniform(0.0001, 2000, abcd_cshape + [2,2, len(frequencies)])
    return TransmissionMatrix(data, frequencies)

@pytest.fixture(scope="module", params=
                ["scalar", "FrequencyData_vector", "FrequencyData_abcd_cshape"])
def load_impedance_with_correct_format(request, frequencies, abcd_cshape):
    if request.param == "scalar":
        return 2
    elif request.param == "FrequencyData_vector":
        return FrequencyData(np.ones_like(frequencies)*2, frequencies)
    elif request.param == "FrequencyData_abcd_cshape":
        shape = abcd_cshape + [len(frequencies)]
        return FrequencyData(np.ones(shape), frequencies)

@pytest.mark.parametrize("impedance_type", ["input", "output"])
def test_correct_input_formats_impedance(impedance_type,
                                         load_impedance_with_correct_format,
                                         tmatrix_random_data):
    Zl = load_impedance_with_correct_format
    if impedance_type == "input":
        tmatrix_random_data.input_impedance(Zl)
    elif impedance_type == "output":
        tmatrix_random_data.output_impedance(Zl)

def test_correct_input_formats_TF_quantity1(load_impedance_with_correct_format,
                                            tmatrix_random_data):
    Zl = load_impedance_with_correct_format
    tmatrix_random_data.transfer_function_quantity1(Zl)
def test_correct_input_formats_TF_quantity2(load_impedance_with_correct_format,
                                            tmatrix_random_data):
    Zl = load_impedance_with_correct_format
    tmatrix_random_data.transfer_function_quantity2(Zl)


#---------------
#| RESULT TESTS|
#---------------
@pytest.fixture(scope="module")
def impedance_random(frequencies) -> FrequencyData:
    rng = np.random.default_rng()
    return FrequencyData(rng.random(len(frequencies)), frequencies)

@pytest.fixture(scope="module",
                params=["random_load", "inf_load", "zero_load", "mixed_load"])
def load_impedance(request, frequencies):
    if request.param == "random_load":
        rng = np.random.default_rng()
        return FrequencyData(rng.random(len(frequencies)), frequencies)
    elif request.param == "inf_load":
        return FrequencyData(np.ones_like(frequencies)*np.inf, frequencies)
    elif request.param == "zero_load":
        return FrequencyData(np.zeros_like(frequencies), frequencies)
    elif request.param == "mixed_load":
        return FrequencyData([0, 2, np.inf], frequencies)

def _special_twoport_tmatrix(twoport_type, Zl : FrequencyData, Z : FrequencyData):
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
    return ["bypass", "series_impedance", "parallel_impedance",
            "series_load_impedance", "parallel_load_impedance"]

def _expected_impedance(twoport_type, Zl : FrequencyData, Z : FrequencyData) -> FrequencyData: #noqa 501
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

def _expected_voltage_tf(twoport_type, Zl : FrequencyData, Z : FrequencyData) -> FrequencyData: #noqa 501
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

def _expected_current_tf(twoport_type, Zl : FrequencyData, Z : FrequencyData) -> FrequencyData: #noqa 501
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

@pytest.mark.parametrize("impedance_type", ["input", "output"])
@pytest.mark.parametrize("twoport_type", _twoport_type_list())
def test_input_impedance(impedance_type : str, twoport_type : str,
                         load_impedance : FrequencyData, impedance_random):
    tmat = _special_twoport_tmatrix(twoport_type, load_impedance, impedance_random)
    if impedance_type == "input":
        Zres = tmat.input_impedance(load_impedance)
    else:
        Zres = tmat.output_impedance(load_impedance)
    Zexpected = _expected_impedance(twoport_type, load_impedance, impedance_random)

    idx_inf = Zexpected.freq == np.inf
    idx_default = np.logical_not(idx_inf)
    npt.assert_allclose(Zres.freq[idx_default], Zexpected.freq[idx_default], atol=1e-15)
    assert(np.all(np.abs(Zres.freq[idx_inf]) > 1e15))

@pytest.mark.parametrize("twoport_type", _twoport_type_list())
def test_transfer_function_quantity1(twoport_type, load_impedance,impedance_random):
    """Test for 'voltage' quantity transfer function"""
    tmat = _special_twoport_tmatrix(twoport_type, load_impedance, impedance_random)
    TF = tmat.transfer_function_quantity1(load_impedance)
    TF_expected = _expected_voltage_tf(twoport_type, load_impedance, impedance_random)
    if isinstance(TF_expected, FrequencyData):
        TF_expected = TF_expected.freq
    npt.assert_allclose(TF.freq, TF_expected, atol=1e-15)

@pytest.mark.parametrize("twoport_type", _twoport_type_list())
def test_transferfunction_quantity2(twoport_type, load_impedance,impedance_random):
    """Test for 'current' quantity transfer function"""
    tmat = _special_twoport_tmatrix(twoport_type, load_impedance, impedance_random)
    TF = tmat.transfer_function_quantity2(load_impedance)
    TF_expected = _expected_current_tf(twoport_type, load_impedance, impedance_random)
    if isinstance(TF_expected, FrequencyData):
        TF_expected = TF_expected.freq
    npt.assert_allclose(TF.freq, TF_expected, atol=1e-15)
