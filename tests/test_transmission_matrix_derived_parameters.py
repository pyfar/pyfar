import numpy as np
import numpy.testing as npt
import pytest
from pyfar import TransmissionMatrix
from pyfar import FrequencyData

def _get_impedance(tmat, Zl, type):
    if type == "input":
        return tmat.input_impedance(Zl)
    elif type == "output":
        return tmat.output_impedance(Zl)
    raise ValueError("Wrong impedance type, expected 'input' or 'output'.")

@pytest.mark.parametrize("type", ["input", "output"])
def test_tmatrix_twoport_impedances(type : str):
    frequencies = [100,200,300]
    rng = np.random.default_rng()
    Zl = FrequencyData(rng.random(len(frequencies)), frequencies)
    Z = FrequencyData(rng.random(len(frequencies)), frequencies)

    # Identity system (bypass)
    tmat = TransmissionMatrix.create_identity(frequencies)
    Zres = _get_impedance(tmat, Zl, type)
    npt.assert_allclose(Zres.freq, Zl.freq, atol=1e-15)

    # Series impedance with Z = Zl
    tmat = TransmissionMatrix.create_series_impedance(Zl)
    Zres = _get_impedance(tmat, Zl, type)
    npt.assert_allclose(Zres.freq, 2.0*Zl.freq, atol=1e-15)

    # Series impedance (Z = random)
    tmat = TransmissionMatrix.create_series_impedance(Z)
    Zres = _get_impedance(tmat, Zl, type)
    npt.assert_allclose(Zres.freq, Z.freq + Zl.freq, atol=1e-15)

    # Parallel impedance with Z = Zl
    tmat = TransmissionMatrix.create_shunt_admittance(1/Zl)
    Zres = _get_impedance(tmat, Zl, type)
    npt.assert_allclose(Zres.freq, 0.5*Zl.freq, atol=1e-15)

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
    test_tmatrix_transfer_functions()
