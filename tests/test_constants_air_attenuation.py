import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt
import re


@pytest.mark.parametrize((
        "temperature", "frequency", "relative_humidity", "expected",
        "expected_accuracy"), [
    (10, 1000, .1, 2.16e1*1e-3, 10),
    (10, 100, .1, 5.85e-1*1e-3, 10),
    (-20, 800, .3, 4.92e-3, 20),
    (35, 4000, .6, 2.58*10e-3, 10),
    (20, 5000, .1, 1.33e2*1e-3, 10),
    (20, 5000, .8, 3.06e1*1e-3, 10),
    (50, 1000, .7, 8.03e-3, 20),
])
def test_air_attenuation_after_table_iso(
        temperature, frequency, relative_humidity, expected,
        expected_accuracy):
    """
    Check alpha after value table in ISO 9613-1. The accuracy is calculated
    manually and not provided by the standard.
    """
    alpha, m, accuracy = pf.constants.air_attenuation(
        temperature, frequency, relative_humidity)
    npt.assert_allclose(alpha, expected, atol=1e-3)
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
def test_air_attenuation_array(
        temperature, frequency, relative_humidity, atmospheric_pressure):
    alpha, m, accuracy = pf.constants.air_attenuation(
        temperature, frequency, relative_humidity, atmospheric_pressure)
    # test air attenuation alpha
    expected = 2.16e1*1e-3 + np.zeros_like(alpha[..., 0])
    npt.assert_allclose(alpha[..., 0], expected, atol=1e-3)
    # test air attenuation factor m
    expected = 2.16e1*1e-3/4.34 + np.zeros_like(alpha[..., 0])
    npt.assert_allclose(m.freq[..., 0], expected, atol=1e-3)
    npt.assert_allclose(m.frequencies, frequency, atol=1e-3)
    # test accuracy
    expected_accuracy = 10 + np.zeros_like(accuracy.freq)
    npt.assert_allclose(accuracy.freq, expected_accuracy, atol=1e-3)
    npt.assert_allclose(accuracy.frequencies, frequency, atol=1e-3)


def test_air_attenuation_inputs():
    temperature = 10
    frequency = 1000
    relative_humidity = .1
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation(
            'test', frequency, relative_humidity)
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation(
            temperature, 'frequency', relative_humidity)
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation(
            temperature, frequency, 'relative_humidity')
    with pytest.raises(
            ValueError, match='frequencies must be one dimensional'):
        pf.constants.air_attenuation(
            temperature, [[1, 1]], relative_humidity)
    with pytest.raises(
            TypeError,
            match='atmospheric_pressure must be a number or array of numbers'):
        pf.constants.air_attenuation(
            temperature, frequency, relative_humidity, 'test')


def test_air_attenuation_broadcastable():
    frequency = 1000
    with pytest.raises(ValueError, match='same shape or be broadcastable'):
        pf.constants.air_attenuation(
            [10, 10], frequency, [.1, .1, .1])


def test_air_attenuation_limits():
    temperature = 10
    frequency = 1000
    relative_humidity = .1
    atmospheric_pressure = 101325
    match = 'Temperature must be greater than -73°C.'
    with pytest.raises(ValueError,match=match):
        pf.constants.air_attenuation(
            -100, frequency, relative_humidity, atmospheric_pressure)
    match = 'frequencies must be greater than 50 Hz'
    with pytest.raises(ValueError,match=match):
        pf.constants.air_attenuation(
            temperature, 20, relative_humidity, atmospheric_pressure)
    match = 'Relative humidity must be between 0 and 1.'
    with pytest.raises(ValueError, match=match):
        pf.constants.air_attenuation(
            temperature, frequency, -.1, atmospheric_pressure)
    with pytest.raises(ValueError, match=match):
        pf.constants.air_attenuation(
            temperature, frequency, 1.1, atmospheric_pressure)
    match = 'Atmospheric pressure must be less than 200 kPa.'
    with pytest.raises(ValueError, match=match):
        pf.constants.air_attenuation(
            temperature, frequency, relative_humidity, 200001)


@pytest.mark.parametrize((
        "concentration_water_vapour", "expected_accuracy"), [
    (0.05, 10),
    (5, 10),
    (1, 10),
    (10, 20),
    (0.005, 20),
    (0.006, 20),
    (0.0006, 50),
    (100, 20),
])
def test_accuracy_concentration_water_vapour(
        concentration_water_vapour, expected_accuracy):
    shape = (1,)
    temperature = 10
    atmospheric_pressure = 101325
    frequencies = 1e3
    accuracy = pf.constants.constants._air_attenuation_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape)
    npt.assert_array_equal(accuracy.freq, expected_accuracy)


@pytest.mark.parametrize((
        "temperature", "expected_accuracy"), [
    (-20, 10),
    (50, 10),
    (0, 10),
    (-100, -1),
    (-50, 50),
    (-73, 50),
    (100, 50),
])
def test_accuracy_temperature(
        temperature, expected_accuracy):
    shape = (1,)
    concentration_water_vapour = 1
    atmospheric_pressure = 101325
    frequencies = 1e3
    accuracy = pf.constants.constants._air_attenuation_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape)
    npt.assert_array_equal(accuracy.freq, expected_accuracy)


@pytest.mark.parametrize((
        "atmospheric_pressure", "expected_accuracy"), [
    (101325, 10),
    (200000-1, 10),
    (200000, -1),
    (300000, -1),
])
def test_accuracy_atmospheric_pressure(
        atmospheric_pressure, expected_accuracy):
    shape = (1,)
    temperature = 0
    concentration_water_vapour = 1
    frequencies = 1e3
    accuracy = pf.constants.constants._air_attenuation_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape)
    npt.assert_array_equal(accuracy.freq, expected_accuracy)



@pytest.mark.parametrize((
        "frequency_pressure_ratio", "expected_accuracy"), [
    (4e-4, 10),
    (10, 10),
    (1, 10),
    (3e-4, -1),
    (11, -1),
])
def test_accuracy_frequency_pressure_ratio(
        frequency_pressure_ratio, expected_accuracy):
    shape = (1,)
    temperature = 0
    concentration_water_vapour = 1
    atmospheric_pressure = 101325
    frequencies = frequency_pressure_ratio * atmospheric_pressure
    accuracy = pf.constants.constants._air_attenuation_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape)
    npt.assert_array_equal(accuracy.freq, expected_accuracy)


def test_accuracy_invalid_vapor():
    match = r"Concentration of water vapour must be between 0% and 100%."
    with pytest.raises(
            ValueError, match=re.escape(match)):
        pf.constants.constants._air_attenuation_accuracy(
            101, 20, 101325, 1000, (1,))
    with pytest.raises(
            ValueError, match=re.escape(match)):
        pf.constants.constants._air_attenuation_accuracy(
            -1, 20, 101325, 1000, (1,))


def test_accuracy_invalid_pressure():
    match = "Atmospheric pressure must be greater than 0 Pa."
    with pytest.raises(
            ValueError, match=re.escape(match)):
        pf.constants.constants._air_attenuation_accuracy(
            .05, 20, -101325, 1000, (1,))


def test_accuracy_invalid_temperature():
    match = "Temperature must be greater than -273.15°C."
    with pytest.raises(
            ValueError, match=re.escape(match)):
        pf.constants.constants._air_attenuation_accuracy(
            .05, -280, 101325, 1000, (1,))


def test_accuracy_invalid_frequency():
    match = "Frequencies must be positive."
    with pytest.raises(
            ValueError, match=re.escape(match)):
        pf.constants.constants._air_attenuation_accuracy(
            .05, 20, 101325, -1, (1,))

