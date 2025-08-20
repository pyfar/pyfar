import numpy as np
import pytest
from numpy import testing as npt
import pyfar.constants as constants


def test_nominal_iec_octave_default():
    nominal_octs = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    actual_octs = constants.fractional_octave_frequencies_nominal(
        frequency_range=(16, 16e3))
    npt.assert_allclose(actual_octs, nominal_octs)

def test_nominal_iec_third_octave_default():
    nominal_thirds = [
        10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000]
    actual_thirds = constants.fractional_octave_frequencies_nominal(
        num_fractions=3, frequency_range=(10, 20e3))
    npt.assert_allclose(actual_thirds, nominal_thirds)

def test_nominal_iec_error():
    with pytest.raises(ValueError, match="lower and upper limit"):
        constants.fractional_octave_frequencies_nominal(frequency_range=(1,))

    with pytest.raises(ValueError, match="lower and upper limit"):
        constants.fractional_octave_frequencies_nominal(frequency_range=(
                                                                    3, 4, 5))

    with pytest.raises(
            ValueError, match="second frequency needs to be higher"):
        constants.fractional_octave_frequencies_nominal(
            frequency_range=(8e3, 1e3))

    with pytest.raises(
            ValueError, match="num_fractions must be 1 or 3"):
        constants.fractional_octave_frequencies_nominal(num_fractions=7)

    with pytest.warns(UserWarning, match="octave-band are defined only"):
        constants.fractional_octave_frequencies_nominal(num_fractions=1,
                                                    frequency_range=(12, 25e3))

    with pytest.warns(UserWarning, match="one-third-octave-band " \
                                            "are defined only"):
        constants.fractional_octave_frequencies_nominal(num_fractions=3,
                                                frequency_range=(7, 20e3))

def test_nominal_iec_ocatve_subset():
    actual_octs = constants.fractional_octave_frequencies_nominal(
        num_fractions=1, frequency_range=(100, 4e3))
    nominal_octs_part = [125, 250, 500, 1000, 2000, 4000]
    npt.assert_allclose(actual_octs, nominal_octs_part)

def test_exact_octave():
    actual_exact = constants.fractional_octave_frequencies_exact(
        num_fractions=1, frequency_range=(4e3, 64e3))[0]
    npt.assert_allclose(actual_exact, [3981.071706, 7943.282347, 15848.93192,
                                       31622.7766, 63095.73445])


def test_exact_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                                    1, (2e3, 20e3))[0]
    expected = np.array([1995.262315, 3981.071706, 7943.282347, 15848.93192])
    np.testing.assert_allclose(exact, expected)

def test_exact_twelfth_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                                    12, (2e3, 4e3))[0]
    expected = np.array([2053.525026, 2175.204034, 2304.092976, 2440.619068,
                2585.23484, 2738.419634, 2900.681199, 3072.557365, 3254.617835,
                3447.466066, 3651.741273, 3868.120546, 4097.321098])
    np.testing.assert_allclose(exact, expected)

def test_exact_fifth_octave_cutoff():
    frac = 5
    exact, lower, upper = constants.fractional_octave_frequencies_exact(
        frac, (2e3, 20e3))

    G = 10**(3/10)
    np.testing.assert_allclose(
        lower, exact*G**(-1/2/frac))
    np.testing.assert_allclose(
        upper, exact*G**(1/2/frac))

