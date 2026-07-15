"""
Test for the standard-conform level functions.

Note that the tests for the shared parameters of these functions are in
`test_level_common_parameters.py`, so this file ony contains tests
against known values or other tests that are specific to a single function.
"""

import pyfar as pf
import numpy as np


def test_level_equivalent_continuous_level_known_value():
    s = pf.signals.sine(1000, 22050)
    levels = pf.level.equivalent_continuous_level(
        s, "Z", None, 2e-5)
    # 94 dB is 1 Pa, -3.01 dB is the crest factor of sine signals
    assert np.isclose(levels, 94 - 3.01, atol=0.1)


def test_level_sliding_equivalent_continuous_level_known_value():
    # phase prevents the first sample from being exactly zero, which would
    # cause a division by zero in the level calculation
    s = pf.signals.sine(1000, 44100, phase=0.01, sampling_rate=44100)
    levels = pf.level.sliding_equivalent_continuous_level(
        s, "Z", None, 1, False, False, 2e-5)
    # 94 dB is 1 Pa, -3.01 dB is the crest factor of sine signals
    # with 1 second window size, the value at 1 second should be
    # the same as the equivalent continuous level of the full signal
    assert np.isclose(levels[0][-1], 94 - 3.01, atol=0.1)


def test_level_sliding_equivalent_continuous_level_shape():
    s = pf.signals.impulse(1000, sampling_rate=48000)
    levels = pf.level.sliding_equivalent_continuous_level(s, "Z")
    assert levels.shape == s.time.shape
