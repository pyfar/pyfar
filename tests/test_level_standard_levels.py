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


def test_level_time_weighted_level_replace_zeros_false():
    """Test that setting replace_zeros to False returns -inf
    and raises a warning from numpy.
    """
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        levels_no_replace = pf.level.time_weighted_level(
            s, "Z", "F", replace_zeros=False)
    assert np.all(levels_no_replace == -np.inf)


def test_level_time_weighted_level_replace_zeros_true():
    """Test that setting replace_zeros to True replaces zeros with the
    array type's epsilon to avoid -inf values and numpy warnings.
    """
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)
    levels_replace = pf.level.time_weighted_level(
        s, "Z", "F", replace_zeros=True)

    # since there are only zeros in the signal, all values must be epsilon
    expected_value = 10 * np.log10(np.finfo(s.time.dtype).eps / 2e-5**2)
    assert np.allclose(levels_replace, expected_value)


@pytest.mark.parametrize(("time_weighting", "err_type"), [
    (None, TypeError), ("X", ValueError)])
def test_level_time_weighted_level_time_weighting_error(
    time_weighting, err_type,
):
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(err_type, match="Time weighting"):
        pf.level.time_weighted_level(s, "Z", time_weighting)
