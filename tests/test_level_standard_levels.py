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
