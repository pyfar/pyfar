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


@pytest.mark.parametrize("oversampling", [None, 4, 8])
def test_level_peak_level_known_value(oversampling):
    delay = 5432
    s = pf.signals.impulse(10000, delay)
    levels, times = pf.level.peak_level(s, "Z", oversampling, 1)
    assert np.isclose(levels[0], 0, atol=0.01)   # peak is 1, so 0 dbFS
    assert np.isclose(times[0], delay / s.sampling_rate, atol=0.001)


def test_level_peak_level_intersample_peak():
    """Test that oversampling is actually applied by checking that the
    peak level of a signal with intersample peaks is higher than the
    peak level of the same signal without oversampling.
    """
    s = pf.Signal([0, 0, 0, 1, 0.99, 0, 0, 0], sampling_rate=100)

    level_no_over, time_no_over = pf.level.peak_level(s, "Z", None)
    level_with_over, time_with_over = pf.level.peak_level(s, "Z", 2)

    assert level_with_over > level_no_over
    # peak must be at the 4th sample (0-indexed)
    assert time_no_over == 0.03
    # after oversampling, the peak is at the 7th sample, i.e. between the
    # 3rd and 4th sample at the original sampling rate
    assert time_with_over == 0.035 # at 7th oversampled sample
