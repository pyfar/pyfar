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
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = 101325  # Pa
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    expected = np.sqrt(1.4 * 8.314 / 28.97 * (20.0 + 273.15))
    assert np.isclose(result, expected, rtol=1e-2)


def test_speed_of_sound_ideal_gas_edge_temperature():
    temperature = -20.0  # Celsius
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = 101325  # Pa
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    expected = np.sqrt(1.4 * 8.314 / 28.97 * (-20.0 + 273.15))
    assert np.isclose(result, expected, rtol=1e-2)


def test_speed_of_sound_ideal_gas_edge_humidity():
    temperature = 20.0  # Celsius
    relative_humidity = 1.0  # 100%
    atmospheric_pressure = 101325  # Pa
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    expected = np.sqrt(1.4 * 8.314 / 28.97 * (20.0 + 273.15))
    assert np.isclose(result, expected, rtol=1e-2)

def test_speed_of_sound_ideal_gas_edge_pressure():
    temperature = 20.0  # Celsius
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = 90000  # Pa
    result = pf.constants.speed_of_sound_ideal_gas(
        temperature, relative_humidity, atmospheric_pressure)
    expected = np.sqrt(1.4 * 8.314 / 28.97 * (20.0 + 273.15))
    assert np.isclose(result, expected, rtol=1e-2)

def test_speed_of_sound_ideal_gas_invalid_temperature():
    temperature = -30.0  # Celsius, out of valid range
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = 101325  # Pa
    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_ideal_gas(
            temperature, relative_humidity, atmospheric_pressure)

def test_speed_of_sound_ideal_gas_invalid_humidity():
    temperature = 20.0  # Celsius
    relative_humidity = 1.5  # 150%, out of valid range
    atmospheric_pressure = 101325  # Pa
    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_ideal_gas(
            temperature, relative_humidity, atmospheric_pressure)

def test_speed_of_sound_ideal_gas_invalid_pressure():
    temperature = 20.0  # Celsius
    relative_humidity = 0.5  # 50%
    atmospheric_pressure = -50000  # Pa, out of valid range
    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_ideal_gas(
            temperature, relative_humidity, atmospheric_pressure)