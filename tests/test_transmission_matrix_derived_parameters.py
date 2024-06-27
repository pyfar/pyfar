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



def test_tmatrix_transfer_functions():
    frequencies = [100,200,300]
    rng = np.random.default_rng()
    Zl = FrequencyData(rng.random(len(frequencies)), frequencies)

    # Identity system (bypass)
    tmat = TransmissionMatrix.create_identity(frequencies)
    TF_voltage = tmat.transfer_function_quantity1(Zl)
    TF_current = tmat.transfer_function_quantity2(Zl)
    npt.assert_allclose(TF_voltage.freq, 1, atol=1e-15)
    npt.assert_allclose(TF_current.freq, 1, atol=1e-15)

    # Series impedance with Z = Zl
    tmat = TransmissionMatrix.create_series_impedance(Zl)
    TF_voltage = tmat.transfer_function_quantity1(Zl)
    TF_current = tmat.transfer_function_quantity2(Zl)
    npt.assert_allclose(TF_voltage.freq, 1/2, atol=1e-15)
    npt.assert_allclose(TF_current.freq, 1, atol=1e-15)

    # Parallel impedance with Z = Zl
    tmat = TransmissionMatrix.create_shunt_admittance(1/Zl)
    TF_voltage = tmat.transfer_function_quantity1(Zl)
    TF_current = tmat.transfer_function_quantity2(Zl)
    npt.assert_allclose(TF_voltage.freq, 1, atol=1e-15)
    npt.assert_allclose(TF_current.freq, 1/2, atol=1e-15)


if __name__ == "__main__":
    test_tmatrix_twoport_impedances("input")
    test_tmatrix_twoport_impedances("output")
    test_tmatrix_twoport_impedance_Zl_inf("input")
    test_tmatrix_twoport_impedance_Zl_inf("output")
    test_tmatrix_transfer_functions()
