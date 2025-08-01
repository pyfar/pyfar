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


def test_density_of_air_standard_case():
    """Test density_of_air function reference air density."""
    result = pf.constants.density_of_air(
        pf.constants.reference_air_temperature_celsius, 0)
    expected = pf.constants.reference_air_density
    npt.assert_almost_equal(result, expected, decimal=3)


@pytest.mark.parametrize("temperature", [
    20,
    np.array([20, 20.0]),
    (20, 20.0),
    [20, 20.0],
])
@pytest.mark.parametrize("relative_humidity", [
    0,
    np.array([0, 0]),
    (0, 0),
    [0, 0],
])
@pytest.mark.parametrize("atmospheric_pressure", [
    101325,
    np.array([101325, 101325]),
    (101325, 101325),
    [101325, 101325],
])
def test_density_of_air_array(
    temperature, relative_humidity, atmospheric_pressure):
    """Test density_of_air function reference air density."""
    result = pf.constants.density_of_air(
        temperature, relative_humidity, atmospheric_pressure)
    expected = pf.constants.reference_air_density
    npt.assert_almost_equal(result, expected, decimal=3)


@pytest.mark.parametrize(
        ("temperature", "relative_humidity", "expected"), [
    (20, 0.5, 1.1991427805178105),  # typical conditions
    (0, 0.5, 1.2910965426610168),   # freezing point
    (40, 0.5, 1.1119555666903926),  # hot day
    (-10, 0.5, 1.3409709049215106), # below freezing
    (20, 0, 1.2043845867047402),    # dry air
    (20, 1, 1.193900974330881),     # fully saturated air
])
def test_density_of_air(
        temperature, relative_humidity, expected):
    """Test density_of_air function with various inputs."""
    result = pf.constants.density_of_air(
        temperature, relative_humidity)
    npt.assert_almost_equal(result, expected)


def test_density_of_air_invalid_temperature():
    """Test density_of_air with invalid temperature input."""
    with pytest.raises(
            TypeError,
            match="Temperature must be a number or array of numbers"):
        pf.constants.density_of_air("invalid", 0.5, 101325)


def test_density_of_air_invalid_relative_humidity():
    """Test density_of_air with invalid relative humidity input."""
    with pytest.raises(
            TypeError,
            match="Relative humidity must be a number or array of numbers"):
        pf.constants.density_of_air(20, "invalid", 101325)


def test_density_of_air_invalid_atmospheric_pressure():
    """Test density_of_air with invalid atmospheric pressure input."""
    with pytest.raises(
            TypeError,
            match="Atmospheric pressure must be a number or array of numbers"):
        pf.constants.density_of_air(20, 0.5, "invalid")


def test_density_of_air_relative_humidity_out_of_range():
    """Test density_of_air with relative humidity out of valid range."""
    with pytest.raises(
            ValueError, match="Relative humidity must be between 0 and 1."):
        pf.constants.density_of_air(20, -0.1, 101325)
    with pytest.raises(
            ValueError, match="Relative humidity must be between 0 and 1."):
        pf.constants.density_of_air(20, 1.1, 101325)


def test_density_of_air_negative_atmospheric_pressure():
    """Test density_of_air with negative atmospheric pressure."""
    with pytest.raises(
            ValueError,
            match="Atmospheric pressure must be larger than 0 Pa."):
        pf.constants.density_of_air(20, 0.5, -101325)
