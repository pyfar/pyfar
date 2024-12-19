import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt


@pytest.mark.parametrize((
        "temperature", "frequency", "relative_humidity", "expected",
        "expected_accuracy"), [
    (10, 1000, .1, 2.16e1*1e-3, 10),
    (10, 100, .1, 5.85e-1*1e-3, 10),
    (10, 100, .005, 0.002094, 10),
])
def test_air_attenuation_iso(
        temperature, frequency, relative_humidity, expected,
        expected_accuracy):
    temperature = 10
    air_attenuation, accuracy = pf.constants.air_attenuation_iso(
        temperature, frequency, relative_humidity, calculate_accuracy=True)
    npt.assert_allclose(air_attenuation.freq, expected, atol=1e-3)
    npt.assert_allclose(accuracy.freq, expected_accuracy)


@pytest.mark.parametrize("temperature", [
        np.array([[10, 10, 10, 10]]).T,
        [10, 10],
        10,
])
@pytest.mark.parametrize("frequency", [
        np.array([1000, 20000, 40000]),
        [1000, 20000, 40000],
        1000,
])
@pytest.mark.parametrize("relative_humidity", [
        np.array([[.1, .1, .1, .1]]).T,
        [.1, .1],
        .1,
])
@pytest.mark.parametrize("atmospheric_pressure", [
        np.array([[101325, 101325, 101325, 101325]]).T,
        [101325, 101325],
        101325,
])
def test_air_attenuation_iso_array(
        temperature, frequency, relative_humidity, atmospheric_pressure):
    result, accuracy = pf.constants.air_attenuation_iso(
        temperature, frequency, relative_humidity, atmospheric_pressure, True)
    # test air attenuation
    expected = 2.16e1*1e-3 + np.zeros_like(result.freq[..., 0])
    npt.assert_allclose(result.freq[..., 0], expected, atol=1e-3)
    npt.assert_allclose(result.frequencies, frequency, atol=1e-3)
    # test accuracy
    expected_accuracy = 10 + np.zeros_like(accuracy.freq)
    npt.assert_allclose(accuracy.freq, expected_accuracy, atol=1e-3)
    npt.assert_allclose(accuracy.frequencies, frequency, atol=1e-3)


def test_air_attenuation_iso_inputs():
    temperature = 10
    frequency = 1000
    relative_humidity = .1
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation_iso(
            'test', frequency, relative_humidity, calculate_accuracy=True)
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation_iso(
            temperature, 'frequency', relative_humidity,
            calculate_accuracy=True)
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation_iso(
            temperature, frequency, 'relative_humidity',
            calculate_accuracy=True)
    with pytest.raises(TypeError, match='must be a bool'):
        pf.constants.air_attenuation_iso(
            temperature, frequency, relative_humidity, calculate_accuracy=5)


def test_saturation_vapor_pressure_scalar():
    temperature = 25
    expected = 31.61736
    result = pf.constants.saturation_vapor_pressure(temperature)
    npt.assert_allclose(result, expected, atol=0.001)


def test_saturation_vapor_pressure_array():
    temperature = np.array([0, 10, 20, 30])
    expected = np.array([6.1094, 12.260206, 23.334406, 42.366503])
    result = pf.constants.saturation_vapor_pressure(temperature)
    npt.assert_allclose(result, expected, atol=0.001)


def test_saturation_vapor_pressure_list():
    temperature = [0, 10, 20, 30]
    expected = [6.1094, 12.260206, 23.334406, 42.366503]
    result = pf.constants.saturation_vapor_pressure(temperature)
    npt.assert_allclose(result, expected, atol=0.001)


def test_saturation_vapor_pressure_out_of_range_low():
    with pytest.raises(
            ValueError,
            match="Temperature must be in the range of -45°C and 60°C."):
        pf.constants.saturation_vapor_pressure(-50)


def test_saturation_vapor_pressure_out_of_range_high():
    with pytest.raises(
            ValueError,
            match="Temperature must be in the range of -45°C and 60°C."):
        pf.constants.saturation_vapor_pressure(70)


def test_saturation_vapor_pressure_invalid_type():
    with pytest.raises(
            TypeError,
            match="temperature must be a number or array of numbers"):
        pf.constants.saturation_vapor_pressure("invalid")
