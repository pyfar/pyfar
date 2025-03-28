import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt


def test_speed_of_sound_simple_scalar():
    temperature = 20
    expected = 343.2 * np.sqrt((temperature + 273.15) / 293.15)
    result = pf.constants.speed_of_sound_simple(temperature)
    npt.assert_array_equal(result, expected)

@pytest.mark.parametrize("temperature", [
        np.array([0, 10, 20, 30]),
        [0, 10, 20, 30],
])
def test_speed_of_sound_simple_array(temperature):
    expected = 343.2 * np.sqrt((np.array(temperature) + 273.15) / 293.15)
    result = pf.constants.speed_of_sound_simple(temperature)
    npt.assert_array_equal(result, expected)

def test_speed_of_sound_simple_below_range():
    with pytest.raises(
            ValueError, match="Temperature must be between"):
        pf.constants.speed_of_sound_simple(-21)

def test_speed_of_sound_simple_above_range():
    with pytest.raises(
            ValueError, match="Temperature must be between"):
        pf.constants.speed_of_sound_simple(51)

def test_speed_of_sound_simple_edge_cases():
    temperatures = [-20, 50]
    expected = 343.2 * np.sqrt((np.array(temperatures) + 273.15) / 293.15)
    result = pf.constants.speed_of_sound_simple(temperatures)
    npt.assert_array_equal(result, expected)


def test_speed_of_sound_ideal_gas_typical_values():
    temperature = 20.0  # Celsius
    relative_humidity = 0  # 0%
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity)
    expected = pf.constants.reference_speed_of_sound
    npt.assert_almost_equal(result, expected, decimal=2)


@pytest.mark.parametrize("temperature", [  # Celsius
    [20, 20.0],
    np.array([20, 20.0]),
    (20, 20.0),
    20,
])
@pytest.mark.parametrize("relative_humidity", [
    [.5, .5],
    np.array([.5, .5]),
    (.5, .5),
    .5,
])
@pytest.mark.parametrize("atmospheric_pressure", [  # Pa
    [101325, 101325],
    np.array([101325, 101325]),
    (101325, 101325),
    101325,
])
def test_speed_of_sound_ideal_gas_array_like(
        temperature, relative_humidity, atmospheric_pressure):
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    npt.assert_almost_equal(result, 343.8278399)


def test_speed_of_sound_ideal_gas_values_p_water():
    temperature = 20.0  # Celsius
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = pf.constants.reference_atmospheric_pressure  # Pa
    p_water = pf.constants.saturation_vapor_pressure_magnus(temperature)
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure, p_water)
    expected = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    npt.assert_allclose(result, expected)


def test_speed_of_sound_ideal_gas_edge_temperature():
    temperature = -20.0  # Celsius
    relative_humidity = 0.0  # 0%
    atmospheric_pressure = pf.constants.reference_atmospheric_pressure  # Pa
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    # calculated using the simplified ideal gas formula 'speed_of_sound_simple'
    expected = 318.927  # m/s
    npt.assert_allclose(result, expected, rtol=1e-4)


def test_speed_of_sound_ideal_gas_humidity():
    temperature = 20.0  # Celsius
    relative_humidity = 1.0  # 100%
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity)
    # calculated using the simplified ideal gas formula 'speed_of_sound_simple'
    expected = pf.constants.reference_speed_of_sound
    npt.assert_allclose(result, expected, rtol=1e-2)
    assert result > expected


def test_speed_of_sound_ideal_gas_edge_pressure():
    temperature = 20.0  # Celsius
    relative_humidity = 0.0  # 0%
    atmospheric_pressure = 90000  # Pa
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    expected = pf.constants.reference_speed_of_sound
    assert result < expected
    npt.assert_allclose(result, expected, rtol=1e-2)


def test_speed_of_sound_ideal_gas_invalid_temperature():
    temperature = -300.0  # Celsius, out of valid range
    relative_humidity = 0.0  # 0%
    atmospheric_pressure = pf.constants.reference_atmospheric_pressure  # Pa
    with pytest.raises(ValueError, match="Temperature must be"):
        pf.constants.speed_of_sound_ideal_gas(
            temperature, relative_humidity, atmospheric_pressure)


def test_speed_of_sound_ideal_gas_invalid_humidity():
    temperature = 20.0  # Celsius
    relative_humidity = 1.5  # 150%, out of valid range
    atmospheric_pressure = pf.constants.reference_atmospheric_pressure  # Pa
    with pytest.raises(ValueError, match="Relative humidity must be"):
        pf.constants.speed_of_sound_ideal_gas(
            temperature, relative_humidity, atmospheric_pressure)


def test_speed_of_sound_ideal_gas_invalid_pressure():
    temperature = 20.0  # Celsius
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = -50000  # Pa, out of valid range
    with pytest.raises(ValueError, match="Atmospheric pressure must be"):
        pf.constants.speed_of_sound_ideal_gas(
            temperature, relative_humidity, atmospheric_pressure)
