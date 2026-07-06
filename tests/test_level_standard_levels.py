import pyfar as pf
import numpy as np


# TODO thorough parameter testing can be added more effectively once
# the more there are more functions using the same helper functions.

def test_level_equivalent_continuous_level_reference_pressure():
    s = pf.signals.sine(1000, 22050)
    levels = pf.level.equivalent_continuous_level(
        s, "Z", reference_pressure=2e-5)
    # 94 dB is 1 Pa, -3.01 dB is the crest factor of sine singals
    assert np.isclose(levels, 94 - 3.01, atol=0.1)
