import numpy as np
import pytest
from numpy import testing as npt
import pyfar

import pyfar.constants


def test_center_frequencies_iec():
    nominal_octs = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    actual_octs = pyfar.constants.fractional_octave_frequencies_nominal(
        frequency_range=(12, 20e3))
    npt.assert_allclose(actual_octs, nominal_octs)

    nominal_thirds = [
        10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000]
    actual_thirds = pyfar.constants.fractional_octave_frequencies_nominal(
        num_fractions=3, frequency_range=(9, 20e3))
    npt.assert_allclose(actual_thirds, nominal_thirds)

    with pytest.raises(ValueError, match="lower and upper limit"):
        pyfar.constants.fractional_octave_frequencies_nominal(frequency_range=(1,))

    with pytest.raises(ValueError, match="lower and upper limit"):
        pyfar.constants.fractional_octave_frequencies_nominal(frequency_range=(3,
                                                                         4, 5))

    with pytest.raises(
            ValueError, match="second frequency needs to be higher"):
        pyfar.constants.fractional_octave_frequencies_nominal(
            frequency_range=(8e3, 1e3))

    actual_octs = pyfar.constants.fractional_octave_frequencies_nominal(
        num_fractions=1, frequency_range=(100, 4e3))
    nominal_octs_part = [125, 250, 500, 1000, 2000, 4000]
    npt.assert_allclose(actual_octs, nominal_octs_part)


def test_fractional_frequencies_exact():
    actual_exact, cutoff = pyfar.constants.fractional_octave_frequencies_exact(
        num_fractions=1, frequency_range=(4e3, 64e3))
    npt.assert_allclose(actual_exact, [3981.071706, 7943.282347, 15848.93192,
                                       31622.7766, 63095.73445])


def test_fract_oct_bands_non_iec():
    exact, cutoff = pyfar.constants.fractional_octave_frequencies_exact(1,
                                                                 (2e3, 20e3))
    expected = np.array([1995.262315, 3981.071706, 7943.282347, 15848.93192])

    np.testing.assert_allclose(exact, expected)

    frac = 5
    exact, f_cutoff = pyfar.constants.fractional_octave_frequencies_exact(
        frac, (2e3, 20e3))

    octave_ratio = 10**(3/10)
    np.testing.assert_allclose(
        f_cutoff[0], exact*octave_ratio**(-1/2/frac))
    np.testing.assert_allclose(
        f_cutoff[1], exact*octave_ratio**(1/2/frac))
