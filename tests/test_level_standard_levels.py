"""
Test for the standard-conform level functions.

Note that the tests for the shared parameters of these functions are in
`test_level_common_parameters.py`, so this file ony contains tests
against known values or other tests that are specific to a single function.
"""

import pytest
import pyfar as pf
import numpy as np

# 1 Pa in dB SPL
ONE_PA = 20 * np.log10(1 / pf.constants.reference_sound_pressure)
SINE_PAPR = 10 * np.log10(2)  # peak-to-average power ratio of sine signals


def test_level_equivalent_continuous_level_known_value():
    s = pf.signals.sine(1000, 22050)
    levels = pf.level.equivalent_continuous_level(
        s, "Z", None, 2e-5)
    assert np.isclose(levels, ONE_PA - SINE_PAPR, atol=0.001)


@pytest.mark.parametrize(("duration", "level_increase"), [
    (None, 0),  # signal is one second long, which is the unit length
    (1, 0),     # unit length in the standard
    (10, 10),   # 10 s => 10x the energy => 10 dB increase
    (100, 20),  # 100 s => 100x the energy => 20 dB increase
])
def test_level_exposure_level_duration(duration, level_increase):
    s = pf.signals.sine(1000, 44100, sampling_rate=44100)
    levels = pf.level.exposure_level(s, "Z", duration)
    # 94 dB is 1 Pa, -3.01 dB is the crest factor of sine signals
    assert np.isclose(levels, ONE_PA - SINE_PAPR + level_increase, atol=0.001)


@pytest.mark.parametrize("duration", [-1, 0, np.int32(-1)])
def test_level_exposure_level_duration_value_errors(duration):
    s = pf.signals.sine(1000, 44100, sampling_rate=44100)
    with pytest.raises(ValueError, match="positive"):
        pf.level.exposure_level(s, "Z", duration)


@pytest.mark.parametrize("duration", ["1", np.array([1]), [1], complex(1, 0)])
def test_level_exposure_level_duration_type_errors(duration):
    s = pf.signals.sine(1000, 44100, sampling_rate=44100)
    with pytest.raises(TypeError, match="number"):
        pf.level.exposure_level(s, "Z", duration)
