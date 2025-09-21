import numpy as np
import pytest
from numpy import testing as npt
import pyfar

from pyfar.dsp import filter
from pyfar import FilterSOS, Signal


def test_center_frequencies_iec():
    nominal_octs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    actual_octs = filter.fractional_octave_frequencies(num_fractions=1)
    actual_octs_nom = actual_octs[0]
    npt.assert_allclose(actual_octs_nom, nominal_octs)

    nominal_thirds = [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
        12500, 16000, 20000]
    actual_thirds = filter.fractional_octave_frequencies(
        num_fractions=3)
    actual_thirds_nom = actual_thirds[0]
    npt.assert_allclose(actual_thirds_nom, nominal_thirds)

    with pytest.raises(ValueError, match="lower and upper limit"):
        filter.fractional_octave_frequencies(frequency_range=(1,))

    with pytest.raises(ValueError, match="lower and upper limit"):
        filter.fractional_octave_frequencies(frequency_range=(3, 4, 5))

    with pytest.raises(
            ValueError, match="second frequency needs to be higher"):
        filter.fractional_octave_frequencies(
            frequency_range=(8e3, 1e3))

    actual_octs = filter.fractional_octave_frequencies(
        num_fractions=1, frequency_range=(100, 4e3))
    actual_octs_nom = actual_octs[0]
    nominal_octs_part = [125, 250, 500, 1000, 2000, 4000]
    npt.assert_allclose(actual_octs_nom, nominal_octs_part)


def test_fractional_coeff_oct_filter_iec():
    sr = 48e3
    order = 2

    expected = np.array([
        [[1.99518917e-03,  3.99037834e-03,  1.99518917e-03,
          1.00000000e+00, -1.89455465e+00,  9.21866028e-01],
         [1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
          1.00000000e+00, -1.94204953e+00,  9.52106382e-01]],
        [[7.47518158e-03,  1.49503632e-02,  7.47518158e-03,
          1.00000000e+00, -1.74644971e+00,  8.50561709e-01],
         [1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
          1.00000000e+00, -1.86728468e+00,  9.06300424e-01]],
        [[2.65806645e-02,  5.31613291e-02,  2.65806645e-02,
          1.00000000e+00, -1.34871529e+00,  7.26916714e-01],
         [1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
          1.00000000e+00, -1.67171842e+00,  8.18664740e-01]]])

    actual = filter.fractional_octaves._coefficients_fractional_octave_bands(
        sr, 1, frequency_range=(1e3, 4e3), order=order)
    np.testing.assert_allclose(actual, expected)

    sr = 16e3
    order = 6

    with pytest.warns(UserWarning, match="Skipping bands"):
        with pytest.warns(UserWarning, match="The upper frequency limit"):
            actual = filter.fractional_octaves. \
                        _coefficients_fractional_octave_bands(
                            sr, 1, frequency_range=(5e3, 20e3), order=order)

    assert actual.shape == (1, order, 6)


def test_fractional_frequencies_non_iec():
    actual_nominal, actual_exact = filter.fractional_octave_frequencies(
        num_fractions=1, frequency_range=(4e3, 64e3))
    npt.assert_allclose(actual_exact, [4e3, 8e3, 16e3, 32e3, 64e3])
    assert actual_nominal is None


def test_fract_oct_filter_iec():
    # Test only Filter object related stuff here, testing of coefficients is
    # done in separate test.
    sr = 48e3
    n_samples = 2**10
    impulse = pyfar.signals.impulse(n_samples, sampling_rate=sr)

    f_obj = filter.fractional_octave_bands(
        None, 3, sampling_rate=sr)
    assert isinstance(f_obj, FilterSOS)

    sig = filter.fractional_octave_bands(impulse, 3)
    assert isinstance(sig, Signal)

    ir_actual = filter.fractional_octave_bands(
        impulse, 1, frequency_range=(1e3, 4e3))

    assert ir_actual.time.shape[0] == 3


def test_fractional_octave_bands_errors():
    """Test if all errors are raised as expected."""

    message = 'Either signal or sampling_rate must be None.'
    with pytest.raises(ValueError, match=message):
        filter.fractional_octave_bands(
            Signal(1, 1), 1, 44100, (1e3, 8e3))


def test_fract_oct_bands_non_iec():
    exact = filter.fractional_octaves.\
        _exact_center_frequencies_fractional_octaves(1, (2e3, 20e3))
    expected = np.array([2e3, 4e3, 8e3, 16e3])

    np.testing.assert_allclose(exact, expected)

    nominal, exact = filter.fractional_octave_frequencies(
        5, (2e3, 20e3), return_cutoff=False)
    assert nominal is None

    frac = 5
    nominal, exact, f_crit = filter.fractional_octave_frequencies(
        frac, (2e3, 20e3), return_cutoff=True)

    octave_ratio = 10**(3/10)
    np.testing.assert_allclose(
        f_crit[0], exact*octave_ratio**(-1/2/frac))
    np.testing.assert_allclose(
        f_crit[1], exact*octave_ratio**(1/2/frac))


def test_sum_bands_din():
    """Check if the error in the sum of the quared magnitude of all bands is
    less than 1 dB. DIN 61260 requires this criterion to be fulfilled between
    two bands.
    """
    n_samples = 2**17
    sampling_rate = 48e3
    impulse = pyfar.signals.impulse(n_samples, sampling_rate=sampling_rate)

    ideal = np.squeeze(np.abs(impulse.freq)**2)

    bp_imp = filter.fractional_octave_bands(
        impulse, num_fractions=3, order=14)

    sum_bands = np.sum(np.abs(bp_imp.freq)**2, axis=0)
    diff = ideal / sum_bands

    mask = (impulse.frequencies > 30) & (impulse.frequencies < 20e3)

    assert not np.any(diff[:, mask] > 10**(1/10))
    assert not np.any(diff[:, mask] < 10**(-1/10))


@pytest.mark.parametrize(
        ('filter_order', 'num_fractions', 'tolerance_class', 'tolerance_met'),
        [(1, 1, 1, False), (1, 3, 1, False),  # Failing class 1
         (1, 1, 2, False), (1, 3, 2, False),  # Failing class 2
         (5, 1, 1, True), (4, 1, 2, True),    # Minimum order that passes for
                                              # octave bands
         (6, 3, 1, True), (5, 3, 2, True),    # Minimum order that passes
                                              # one-third octave bands
         (14, 1, 1, True), (14, 3, 1, True),  # Default order
         (14, 1, 2, True), (14, 3, 2, True)])
def test_check_fractional_octave_band_filter_tolerance(
    filter_order, num_fractions, tolerance_class, tolerance_met):
    """Test checking the tolerance of fractional octave band filters."""

    sampling_rate = 44100
    frequency_range = (20, 20000)

    with pytest.warns(UserWarning, match='The upper frequency limit'):
        fractional_octave_bands = filter.fractional_octave_bands(
            None, num_fractions, sampling_rate, frequency_range, filter_order)

    assert filter.check_fractional_octave_band_filter_tolerance(
        fractional_octave_bands, num_fractions,
        frequency_range, tolerance_class) == tolerance_met


def test_errors_check_fractional_octave_band_filter_tolerance():
    """Check if errors are raised as expected."""

    fractional_octave_band_filter = filter.fractional_octave_bands(
            None, 1, 44100, (100, 1000))

    # wrong type for fractional octave band filters
    with pytest.raises(TypeError, match='FilterSOS object'):
        filter.check_fractional_octave_band_filter_tolerance(
            None, 1, (100, 1000), 1)

    # wrong value for num_fractions
    with pytest.raises(ValueError, match='must be 1 or 3'):
        filter.check_fractional_octave_band_filter_tolerance(
            fractional_octave_band_filter, 2, (100, 1000), 1)

    # wrong value for tolerance_class
    with pytest.raises(ValueError, match='must be 1 or 2'):
        filter.check_fractional_octave_band_filter_tolerance(
            fractional_octave_band_filter, 1, (100, 1000), 3)

    # frequency range not matching filters
    with pytest.raises(ValueError, match='frequency_range does not match'):
        filter.check_fractional_octave_band_filter_tolerance(
            fractional_octave_band_filter, 1, (100, 4000), 2)
