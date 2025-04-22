import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt


def test_saturation_vapor_pressure_scalar():
    temperature = 25
    expected = 3161.736
    result = pf.constants.saturation_vapor_pressure_magnus(temperature)
    npt.assert_allclose(result, expected, atol=0.001)


def test_saturation_vapor_pressure_array():
    temperature = np.array([0, 10, 20, 30])
    expected = np.array([610.94, 1226.0206, 2333.4406, 4236.6503])
    result = pf.constants.saturation_vapor_pressure_magnus(temperature)
    npt.assert_allclose(result, expected, atol=0.001)


def test_saturation_vapor_pressure_list():
    temperature = [0, 10, 20, 30]
    expected = [610.94, 1226.0206, 2333.4406, 4236.6503]
    result = pf.constants.saturation_vapor_pressure_magnus(temperature)
    npt.assert_allclose(result, expected, atol=0.001)


def test_saturation_vapor_pressure_out_of_range_low():
    with pytest.raises(
            ValueError,
            match="Temperature must be in the range of -45°C and 60°C."):
        pf.constants.saturation_vapor_pressure_magnus(-50)


def test_saturation_vapor_pressure_out_of_range_high():
    with pytest.raises(
            ValueError,
            match="Temperature must be in the range of -45°C and 60°C."):
        pf.constants.saturation_vapor_pressure_magnus(70)


def test_saturation_vapor_pressure_invalid_type():
    with pytest.raises(
            TypeError,
            match="temperature must be a number or array of numbers"):
        pf.constants.saturation_vapor_pressure_magnus("invalid")
