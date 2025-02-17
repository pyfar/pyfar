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
