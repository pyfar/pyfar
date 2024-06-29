import numpy as np
import numpy.testing as npt
import pytest
from pyfar import TransmissionMatrix
from pyfar import FrequencyData

def _test_impedance(tmat, Zl, Ztarget, imp_type):
    if imp_type == "input":
        Zres = tmat.input_impedance(Zl)
    elif imp_type == "output":
        Zres = tmat.output_impedance(Zl)
    else:
        raise ValueError("Wrong impedance type, expected 'input' or 'output'.")

    idx_inf = Ztarget.freq == np.inf
    idx_default = np.logical_not(idx_inf)
    npt.assert_allclose(Zres.freq[idx_default], Ztarget.freq[idx_default], atol=1e-15)
    assert(np.all(np.abs(Zres.freq[idx_inf]) > 1e15))

def _rnd_impedance_of_same_shape(Zl : FrequencyData):
    frequencies = Zl.frequencies
    rng = np.random.default_rng()
    Z = FrequencyData(rng.random(len(frequencies)), frequencies)
    return Z

def _test_impedance_for_special_twoports(Zl : FrequencyData, imp_type : str,
                                         skip_series_Zl = False,
                                         skip_parallel_Zl = False):
    Z = _rnd_impedance_of_same_shape(Zl)

    # Identity system (bypass)
    tmat = TransmissionMatrix.create_identity(Zl.frequencies)
    _test_impedance(tmat, Zl, Zl, imp_type)

    # Series impedance (Z = random)
    tmat = TransmissionMatrix.create_series_impedance(Z)
    _test_impedance(tmat, Zl, Z+Zl, imp_type)

    # Parallel impedance (Z = random)
    tmat = TransmissionMatrix.create_shunt_admittance(1/Z)
    _test_impedance(tmat, Zl, 1/(1/Zl + 1/Z), imp_type)

    # Series impedance with Z = Zl
    if not skip_series_Zl:
        tmat = TransmissionMatrix.create_series_impedance(Zl)
        _test_impedance(tmat, Zl, 2.0*Zl, imp_type)

    # Parallel impedance with Z = Zl
    if not skip_parallel_Zl:
        tmat = TransmissionMatrix.create_shunt_admittance(1/Zl)
        _test_impedance(tmat, Zl, 0.5*Zl, imp_type)


@pytest.mark.parametrize("imp_type", ["input","output"])
def test_tmatrix_twoport_impedances(imp_type : str):
    frequencies = [100,200,300]
    rng = np.random.default_rng()
    Zl = FrequencyData(rng.random(len(frequencies)), frequencies)
    _test_impedance_for_special_twoports(Zl, imp_type)

@pytest.mark.parametrize("imp_type", ["input", "output"])
def test_tmatrix_twoport_impedance_Zl_inf(imp_type : str):
    Zl = FrequencyData(np.inf, 100)
    _test_impedance_for_special_twoports(Zl, imp_type, True) #Skip Zl + Zl

@pytest.mark.parametrize("imp_type", ["input", "output"])
def test_tmatrix_twoport_impedance_Zl_zero(imp_type : str):
    Zl = FrequencyData(0, 100)
    _test_impedance_for_special_twoports(Zl, imp_type, False, True) # Skip Zl || Zl



def _test_transfer_function(tmat, Zl, TF_type, TF_expected):
    if TF_type == "voltage":
        TF = tmat.transfer_function_quantity1(Zl)
    elif TF_type == "current":
        TF = tmat.transfer_function_quantity2(Zl)
    else:
        raise ValueError("Unexpected TF type.")
    if isinstance(TF_expected, FrequencyData):
        TF_expected = TF_expected.freq
    npt.assert_allclose(TF.freq, TF_expected, atol=1e-15)

def _test_tmatrix_transfer_function_special_twoports(
        Zl : FrequencyData, Z : FrequencyData, tf_type : str, TFs_expected,
        skip_series_Zl = False, skip_parallel_Zl = False):
    # Identity system (bypass)
    tmat = TransmissionMatrix.create_identity(Zl.frequencies)
    _test_transfer_function(tmat, Zl, tf_type, TFs_expected[0])

    # Series impedance (Z = random)
    tmat = TransmissionMatrix.create_series_impedance(Z)
    _test_transfer_function(tmat, Zl, tf_type, TFs_expected[1])

    # Parallel impedance (Z = random)
    tmat = TransmissionMatrix.create_shunt_admittance(1/Z)
    _test_transfer_function(tmat, Zl, tf_type, TFs_expected[2])

    # Series impedance with Z = Zl
    if not skip_series_Zl:
        tmat = TransmissionMatrix.create_series_impedance(Zl)
        _test_transfer_function(tmat, Zl, tf_type, TFs_expected[3])

    # Parallel impedance with Z = Zl
    if not skip_parallel_Zl:
        tmat = TransmissionMatrix.create_shunt_admittance(1/Zl)
        _test_transfer_function(tmat, Zl, tf_type, TFs_expected[4])

@pytest.mark.parametrize("tf_type", ["voltage", "current"])
def test_tmatrix_twoport_transfer_function(tf_type : str):
    frequencies = [100,200,300]
    rng = np.random.default_rng()
    Zl = FrequencyData(rng.random(len(frequencies)), frequencies)
    Z = _rnd_impedance_of_same_shape(Zl)
    if tf_type == "voltage":
        TFs_expected = (1, Zl / (Zl + Z), 1, 1/2, 1)
    elif tf_type == "current":
        TFs_expected = (1, 1, Z / (Zl + Z), 1, 1/2)
    else:
        raise ValueError("Unecpected value for tf_type.")

    _test_tmatrix_transfer_function_special_twoports(Zl, Z, tf_type, TFs_expected)

@pytest.mark.parametrize("tf_type", ["voltage", "current"])
def test_tmatrix_twoport_transfer_function_Zl_inf(tf_type : str):
    frequencies = [100,200,300]
    Zl = FrequencyData([np.inf, np.inf, np.inf], frequencies)
    Z = _rnd_impedance_of_same_shape(Zl)
    if tf_type == "voltage":
        TFs_expected = (1, 1, 1, np.nan, 1) #Series Zl + Zl-> inf/inf -> nan
    elif tf_type == "current":
        TFs_expected = (0, 0, 0, 0, 0)
    else:
        raise ValueError("Unecpected value for tf_type.")

    _test_tmatrix_transfer_function_special_twoports(Zl, Z, tf_type, TFs_expected)

@pytest.mark.parametrize("tf_type", ["voltage", "current"])
def test_tmatrix_twoport_transfer_function_Zl_zero(tf_type : str):
    frequencies = [100,200,300]
    Zl = FrequencyData([0, 0, 0], frequencies)
    Z = _rnd_impedance_of_same_shape(Zl)
    if tf_type == "voltage":
        TFs_expected = (0, 0, 0, 0, 0)
    elif tf_type == "current":
        TFs_expected = (1, 1, 1, 1, np.nan) #Parallel Zl || Zl-> 0/0 -> nan
    else:
        raise ValueError("Unecpected value for tf_type.")

    _test_tmatrix_transfer_function_special_twoports(Zl, Z, tf_type, TFs_expected)


if __name__ == "__main__":
    test_tmatrix_twoport_impedances("input")
    test_tmatrix_twoport_impedances("output")
    test_tmatrix_twoport_impedance_Zl_inf("input")
    test_tmatrix_twoport_impedance_Zl_inf("output")
    test_tmatrix_twoport_transfer_function("voltage")
    test_tmatrix_twoport_transfer_function("current")
