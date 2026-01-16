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

    with pytest.warns(UserWarning,
                      match="one-third-octave-band are defined only"):
        constants.fractional_octave_frequencies_nominal(num_fractions=3,
                                                frequency_range=(7, 20e3))
def test_exact_frequency_range_type_error():
    with pytest.raises(TypeError, match="must be a tuple, list or np.ndarray"):
        constants.fractional_octave_frequencies_exact(frequency_range=2)
    with pytest.raises(TypeError, match="must contain only integer or float"):
        constants.fractional_octave_frequencies_exact(frequency_range=['1',
                                                                     '2'])

def test_exact_frequency_range_value_error():
    with pytest.raises(ValueError, match="upper frequency must be greater"):
        constants.fractional_octave_frequencies_exact(frequency_range=(60, 12))
    with pytest.raises(ValueError, match="must contain exactly two"):
        constants.fractional_octave_frequencies_exact(frequency_range=(1,))
    with pytest.raises(ValueError, match="must contain exactly two"):
        constants.fractional_octave_frequencies_exact(frequency_range=(
                                                                    3, 4, 5))
    with pytest.raises(ValueError, match="frequencies must be" \
                                    " positive numbers"):
        constants.fractional_octave_frequencies_exact(frequency_range=(-1, 0))

def test_exact_num_fractions_type_error():
    with pytest.raises(TypeError, match="must be an integer"):
        constants.fractional_octave_frequencies_exact(num_fractions=0.5)

def test_exact_num_fractions_value_error():
    with pytest.raises(ValueError, match="must be a positive number"):
        constants.fractional_octave_frequencies_exact(num_fractions=-1)


def test_nominal_octave_subset():
    actual_nominal = constants.fractional_octave_frequencies_nominal(
        num_fractions=1, frequency_range=(16, 16e3))
    expected = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    npt.assert_allclose(actual_nominal, expected)


def test_nominal_octave_subset_critical_limits():
    actual_nominal = constants.fractional_octave_frequencies_nominal(
        num_fractions=1, frequency_range=(22.3, 5623.4))
    expected = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000]
    npt.assert_allclose(actual_nominal, expected)


def test_nominal_third_octave_subset():
    actual_nominal = constants.fractional_octave_frequencies_nominal(
        num_fractions=3, frequency_range=(105, 310))
    expected = [100, 125, 160, 200, 250, 315]
    npt.assert_allclose(actual_nominal, expected)


def test_exact_octave():
    actual_exact = constants.fractional_octave_frequencies_exact(
        num_fractions=1, frequency_range=(4e3, 64e3))[0]
    npt.assert_allclose(actual_exact, [3981.071706, 7943.282347,
                                        15848.93192, 31622.7766, 63095.73445])


def test_exact_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                        1, (2e3, 20e3))[0]
    expected = np.array([1995.262315, 3981.071706, 7943.282347, 15848.93192])
    np.testing.assert_allclose(exact, expected)


def test_exact_octave_empty():
    exact = constants.fractional_octave_frequencies_exact(
                                        3, (51, 62))[0]
    expected = np.array([50.11872336, 63.09573445])
    npt.assert_allclose(exact, expected)


def test_exact_half_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                        2, (500, 1000))[0]
    expected = np.array([421.6965034, 595.66214435, 841.3951416, 1188.502227])
    np.testing.assert_allclose(exact, expected)


def test_exact_third_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                        3, (12, 62))[0]
    expected = np.array([12.58925412, 15.84893192, 19.95262315, 25.11886432,
                          31.6227766, 39.81071706, 50.11872336, 63.09573445])
    npt.assert_allclose(exact, expected)


def test_exact_fourth_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                        4, (22, 200))[0]
    expected = np.array([20.53525026, 24.40619068, 29.00681199, 34.474660066,
     40.97321098, 48.69675252, 57.87619883, 68.78599123, 81.75230379,
      97.16279516, 115.4781985, 137.2460961, 163.1172909, 193.8652636 ])
    np.testing.assert_allclose(exact, expected)


def test_exact_twelfth_octave_subset():
    exact = constants.fractional_octave_frequencies_exact(
                                        12, (2e3, 4e3))[0]
    expected = np.array([2053.525026, 2175.204034, 2304.092976, 2440.619068,
                2585.23484, 2738.419634, 2900.681199, 3072.557365, 3254.617835,
                3447.466066, 3651.741273, 3868.120546, 4097.321098])
    np.testing.assert_allclose(exact, expected)


@pytest.mark.parametrize("num_fractions",[2, 5])
def test_exact_octave_cutoff(num_fractions):
    exact, lower, upper = constants.fractional_octave_frequencies_exact(
                                        num_fractions, (2e3, 20e3))
    G = 10**(3/10)
    np.testing.assert_allclose(
        lower, exact*G**(-1/2/num_fractions))
    np.testing.assert_allclose(
        upper, exact*G**(1/2/num_fractions))


@pytest.mark.parametrize("num_fractions", [1, 2, 3])
@pytest.mark.parametrize("freq_range", [(500, 1000), (12, 63),
                                                        (1400, 2230)])
def test_exact_limits_within_bands(num_fractions, freq_range):
    exact, lower, upper = constants.fractional_octave_frequencies_exact(
        num_fractions, freq_range)
    # It should be lower[0] <= freq_range[0] <= upper[0]
    assert lower[0] - freq_range[0] <= 1e-10
    assert freq_range[0] - upper[0] <= 1e-10
    # It should be lower[-1] <= freq_range[1] <= upper[-1]
    assert lower[-1] - freq_range[1] <= 1e-10
    assert freq_range[1] - upper[-1] <= 1e-10


@pytest.mark.parametrize(("num_fractions", "freq_range"),
    [(1, (63, 15000)),
    (3, (13.9, 177.9)),
    (3, (31.5, 1250)),
    (3, (2238.8, 11220.2))])
def test_exact_nominal_correlation(num_fractions, freq_range):
    exact = constants.fractional_octave_frequencies_exact(
                                        num_fractions, freq_range)[0]
    actual_nominal = constants.fractional_octave_frequencies_nominal(
                                        num_fractions, freq_range)
    np.testing.assert_allclose(exact, actual_nominal, rtol=1e-01)
