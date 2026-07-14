"""
Test for the standard-conform level functions.

Note that the tests for the shared parameters of these functions are in
`test_level_common_parameters.py`, so this file ony contains tests
against known values or other tests that are specific to a single function.
"""

import pytest
import pyfar as pf
import numpy as np


def test_level_equivalent_continuous_level_known_value():
    s = pf.signals.sine(1000, 22050)
    levels = pf.level.equivalent_continuous_level(
        s, "Z", None, 2e-5)
    # 94 dB is 1 Pa, -3.01 dB is the crest factor of sine signals
    assert np.isclose(levels, 94 - 3.01, atol=0.1)


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
    assert np.isclose(levels, 94 - 3.01 + level_increase, atol=0.1)
