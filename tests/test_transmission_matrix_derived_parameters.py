import numpy as np
import numpy.testing as npt
import pytest
from pyfar import TransmissionMatrix
from pyfar import FrequencyData

@pytest.fixture(scope="module")
def frequencies():
    return [100,200,300]

@pytest.fixture(scope="module")
def impedance_random(frequencies) -> FrequencyData:
    rng = np.random.default_rng()
    return FrequencyData(rng.random(len(frequencies)), frequencies)

@pytest.fixture(scope="module",
                params=["random_load", "inf_load", "zero_load"])
def load_impedance(request, frequencies):
    if request.param == "random_load":
        rng = np.random.default_rng()
        return FrequencyData(rng.random(len(frequencies)), frequencies)
    elif request.param == "inf_load":
        return FrequencyData(np.ones_like(frequencies)*np.inf, frequencies)
    elif request.param == "zero_load":
        return FrequencyData(np.zeros_like(frequencies), frequencies)

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
    nonphysical = (twoport_type == "series_load_impedance") & np.any(np.isinf(Zl.freq))
    nonphysical |= (twoport_type == "parallel_load_impedance") & np.any(Zl.freq == 0)
    if nonphysical:
        return FrequencyData(np.ones_like(Zl.freq)*np.nan, Zl.frequencies)
    if twoport_type == "bypass":
        return Zl
    if twoport_type == "series_impedance":
        return Zl + Z
    if twoport_type == "parallel_impedance":
        return 1/(1/Zl + 1/Z)
    if twoport_type == "series_load_impedance":
        return 2 * Zl
    if twoport_type == "parallel_load_impedance":
        return Zl / 2
    else:
        raise ValueError("Unexpected value for 'twoport_type'")

def _expected_voltage_tf(twoport_type, Zl : FrequencyData, Z : FrequencyData) -> FrequencyData: #noqa 501
    if np.all(Zl.freq == 0):
        return 0
    if np.all(np.isinf(Zl.freq)):
        if twoport_type == "series_load_impedance": #non-physical
            return np.nan
        return 1

    if twoport_type == "bypass":
        return 1
    if twoport_type == "series_impedance":
        return Zl / (Zl + Z)
    if twoport_type == "parallel_impedance":
        return 1
    if twoport_type == "series_load_impedance":
        return 1/2
    if twoport_type == "parallel_load_impedance":
        return 1
    else:
        raise ValueError("Unexpected value for 'twoport_type'")
def _expected_current_tf(twoport_type, Zl : FrequencyData, Z : FrequencyData) -> FrequencyData: #noqa 501
    if np.all(Zl.freq == 0):
        if twoport_type == "parallel_load_impedance": #non-physical
            return np.nan
        return 1
    if np.all(np.isinf(Zl.freq)):
        return 0

    if twoport_type == "bypass":
        return 1
    if twoport_type == "series_impedance":
        return 1
    if twoport_type == "parallel_impedance":
        return Z / (Zl + Z)
    if twoport_type == "series_load_impedance":
        return 1
    if twoport_type == "parallel_load_impedance":
        return 1/2
    else:
        raise ValueError("Unexpected value for 'twoport_type'")

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
def test_transfer_function_quantity1(twoport_type : str, load_impedance,impedance_random):
    """Test for 'voltage' quantity transfer function"""
    tmat = _special_twoport_tmatrix(twoport_type, load_impedance, impedance_random)
    TF = tmat.transfer_function_quantity1(load_impedance)
    TF_expected = _expected_voltage_tf(twoport_type, load_impedance, impedance_random)
    if isinstance(TF_expected, FrequencyData):
        TF_expected = TF_expected.freq
    npt.assert_allclose(TF.freq, TF_expected, atol=1e-15)

@pytest.mark.parametrize("twoport_type", _twoport_type_list())
def test_transferfunction_quantity2(twoport_type : str, load_impedance,impedance_random):
    """Test for 'current' quantity transfer function"""
    tmat = _special_twoport_tmatrix(twoport_type, load_impedance, impedance_random)
    TF = tmat.transfer_function_quantity2(load_impedance)
    TF_expected = _expected_current_tf(twoport_type, load_impedance, impedance_random)
    if isinstance(TF_expected, FrequencyData):
        TF_expected = TF_expected.freq
    npt.assert_allclose(TF.freq, TF_expected, atol=1e-15)
