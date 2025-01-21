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


def test_speed_of_sound_cramer_typical_values():
    """Test speed_of_sound_cramer with typical values."""
    temperature = 20  # in degrees Celsius
    relative_humidity = .50  # 50%
    pressure = 101325  # in Pascals (standard atmospheric pressure)
    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, pressure, c02_ppm=200)
    expected = 343.2 * np.sqrt((temperature + 273.15) / 293.15)
    npt.assert_allclose(speed, expected)

def test_speed_of_sound_cramer_edge_cases():
    """Test speed_of_sound_cramer with edge cases."""
    # Low temperature
    temperature = -40
    relative_humidity = 0
    pressure = 101325
    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, pressure)
    assert speed == pytest.approx(317.35, rel=1e-2)

    # High temperature
    temperature = 50
    relative_humidity = 100
    pressure = 101325
    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, pressure)
    assert speed == pytest.approx(360.49, rel=1e-2)

    # Low pressure
    temperature = 20
    relative_humidity = 50
    pressure = 80000
    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, pressure)
    assert speed == pytest.approx(343.21, rel=1e-2)

    # High pressure
    temperature = 20
    relative_humidity = 50
    pressure = 120000
    speed = pf.constants.speed_of_sound_cramer(
        temperature, relative_humidity, pressure)
    assert speed == pytest.approx(343.21, rel=1e-2)

def test_speed_of_sound_cramer_invalid_inputs():
    """Test speed_of_sound_cramer with invalid inputs."""
    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_cramer("invalid", 50, 101325)

    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_cramer(20, "invalid", 101325)

    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_cramer(20, 50, "invalid")

    # Invalid relative humidity
    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_cramer(20, -10, 101325)

    # Invalid pressure
    with pytest.raises(ValueError):
        pf.constants.speed_of_sound_cramer(20, 50, -101325)
