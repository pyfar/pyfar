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
    alpha, m, accuracy = pf.constants.air_attenuation_iso(
        temperature, frequency, relative_humidity)
    npt.assert_allclose(alpha.freq, expected, atol=1e-3)
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
    alpha, m, accuracy = pf.constants.air_attenuation_iso(
        temperature, frequency, relative_humidity, atmospheric_pressure)
    # test air attenuation alpha
    expected = 2.16e1*1e-3 + np.zeros_like(alpha.freq[..., 0])
    npt.assert_allclose(alpha.freq[..., 0], expected, atol=1e-3)
    npt.assert_allclose(alpha.frequencies, frequency, atol=1e-3)
    # test air attenuation factor m
    expected = 2.16e1*1e-3/4.34 + np.zeros_like(alpha.freq[..., 0])
    npt.assert_allclose(m.freq[..., 0], expected, atol=1e-3)
    npt.assert_allclose(m.frequencies, frequency, atol=1e-3)
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
            'test', frequency, relative_humidity)
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation_iso(
            temperature, 'frequency', relative_humidity)
    with pytest.raises(TypeError, match='must be a number or'):
        pf.constants.air_attenuation_iso(
            temperature, frequency, 'relative_humidity')
    with pytest.raises(
            ValueError, match='frequencies must be one dimensional'):
        pf.constants.air_attenuation_iso(
            temperature, [[1, 1]], relative_humidity)
    with pytest.raises(
            TypeError,
            match='atmospheric_pressure must be a number or array of numbers'):
        pf.constants.air_attenuation_iso(
            temperature, frequency, relative_humidity, 'test')


def test_air_attenuation_iso_broadcastable():
    frequency = 1000
    with pytest.raises(ValueError, match='same shape or be broadcastable'):
        pf.constants.air_attenuation_iso(
            [10, 10], frequency, [.1, .1, .1])


def test_air_attenuation_iso_limits():
    temperature = 10
    frequency = 1000
    relative_humidity = .1
    atmospheric_pressure = 101325
    match = 'Temperature must be greater than -73°C.'
    with pytest.raises(ValueError,match=match):
        pf.constants.air_attenuation_iso(
            -100, frequency, relative_humidity, atmospheric_pressure)
    match = 'frequencies must be greater than 50 Hz'
    with pytest.raises(ValueError,match=match):
        pf.constants.air_attenuation_iso(
            temperature, 20, relative_humidity, atmospheric_pressure)
    match = 'Relative humidity must be between 0 and 1.'
    with pytest.raises(ValueError, match=match):
        pf.constants.air_attenuation_iso(
            temperature, frequency, -.1, atmospheric_pressure)
    with pytest.raises(ValueError, match=match):
        pf.constants.air_attenuation_iso(
            temperature, frequency, 1.1, atmospheric_pressure)
    match = 'Atmospheric pressure must less than 200 kPa.'
    with pytest.raises(ValueError, match=match):
        pf.constants.air_attenuation_iso(
            temperature, frequency, relative_humidity, 200001)
