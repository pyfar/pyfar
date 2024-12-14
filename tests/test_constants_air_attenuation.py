import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt


@pytest.mark.parametrize((
        "temperature", "frequency", "relative_humidity", "expected"), [
    (10, 1000, .1, 2.16e1*1e-3),
    (10, 100, .1, 5.85e-1*1e-3),
])
def test_air_attenuation_iso(
        temperature, frequency, relative_humidity, expected):
    temperature = 10
    result = pf.constants.air_attenuation_iso(
        temperature, frequency, relative_humidity)
    npt.assert_allclose(result, expected, atol=1e-4)
