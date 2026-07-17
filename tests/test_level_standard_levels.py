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


def test_level_time_weighted_level_replace_zeros_false():
    """Test that setting replace_zeros to False returns -inf
    and raises a warning from numpy.
    """
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        levels_no_replace = pf.level.time_weighted_level(
            s, "Z", "F", replace_zeros=False)
    assert levels_no_replace[0][100] == -np.inf


def test_level_time_weighted_level_replace_zeros_true():
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)
    levels_replace = pf.level.time_weighted_level(
        s, "Z", "F", replace_zeros=True)

    # since there are only zeros in the signal, all values must be the same
    assert np.all(levels_replace[0] == levels_replace[0][0])
    assert np.isfinite(levels_replace[0][100])


@pytest.mark.parametrize(("time_weighting", "err_type"), [
    (None, TypeError), ("X", ValueError)])
def test_level_time_weighted_level_time_weighting_error(
    time_weighting, err_type,
):
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(err_type, match="Time weighting"):
        pf.level.time_weighted_level(s, "Z", time_weighting)
