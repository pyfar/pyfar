import pytest
import numpy as np
import numpy.testing as npt
import os
import pyfar as pf
import pyfar.dsp.filter as pfilt
from pyfar.dsp.filter.audiofilter import _shelving_cascade_slope_parameters


@pytest.mark.parametrize("shelve_type,g_s_b", [
    ("low", (20., -10., 2.)), ("high", (20., 10., 2.))])
def test_shelving_cascade_slope_parameters(shelve_type, g_s_b):
    """
    Test if all parameter combinations from gain (g), slope (s), and
    bandwidth (b) yield the correct values. Two out of the three parameters
    must be given.
    """

    # test with missing third parameter
    g_s_b_test = _shelving_cascade_slope_parameters(
        g_s_b[0], g_s_b[1], None, shelve_type)
    npt.assert_equal(g_s_b_test, g_s_b)
    # test with missing second parameter
    g_s_b_test = _shelving_cascade_slope_parameters(
        g_s_b[0], None, g_s_b[2], shelve_type)
    npt.assert_equal(g_s_b_test, g_s_b)
    # test with missing first parameter
    g_s_b_test = _shelving_cascade_slope_parameters(
        None, g_s_b[1], g_s_b[2], shelve_type)
    npt.assert_equal(g_s_b_test, g_s_b)


@pytest.mark.parametrize("shelve_type,g_s_b,match", [
    ("low", (20., 0, None), "slope must be non-zero"),
    ("low", (20., 10, None), "gain and slope must have different signs"),
    ("high", (20., -10, None), "gain and slope must have the same signs"),
    ("low", (20., -10, 2), "Exactly two out of the parameters")])
def test_shelving_cascade_slope_parameters_assertion(
        shelve_type, g_s_b, match):
    """Test assertions for shelving_cascade_slope_parameters"""

    with pytest.raises(ValueError, match=match):
        _shelving_cascade_slope_parameters(
            g_s_b[0], g_s_b[1], g_s_b[2], shelve_type)


def test_shelve_cascade_errors():
    """Test all value errors."""

    signal = pf.signals.impulse(10)

    # signal and sampling rate are both None
    with pytest.raises(ValueError, match="Either signal or sampling_rate"):
        pfilt.low_shelve_cascade(None, 1e3, sampling_rate=None)
    # signal is of wrong type
    with pytest.raises(ValueError, match="signal must be a pyfar Signal"):
        pfilt.low_shelve_cascade(1, 1e3, sampling_rate=None)
    # frequency_type has wrong value
    with pytest.raises(ValueError, match="frequency_type is 'mid'"):
        pfilt.low_shelve_cascade(signal, 1e3, 'mid', 10, -5)
    # lower characteristic frequency is 0 Hz
    with pytest.raises(ValueError, match="The lower characteristic frequency"):
        pfilt.low_shelve_cascade(signal, 0, 'lower', 10, None, 2)
    # lower characteristic frequency exceeds Nyquist
    with pytest.raises(ValueError, match="The lower characteristic frequency"):
        pfilt.low_shelve_cascade(signal, 40e3, 'lower', 10, None, 2)
    # UPPER characteristic frequency exceeds Nyquist
    with pytest.raises(ValueError, match="The upper characteristic frequency"):
        pfilt.low_shelve_cascade(signal, 40e3, 'upper', 10, None, 2)


def test_shelve_cascade_warnings():
    """Test all warnings."""

    signal = pf.signals.impulse(10)

    # bandwidth is too small
    with pytest.warns(UserWarning, match="The bandwidth is 0.5 octaves"):
        pfilt.low_shelve_cascade(signal, 1e3, "upper", 10, None, 0.5)
    # upper frequency is too high
    with pytest.warns(UserWarning, match="The upper frequency exceeded"):
        pfilt.high_shelve_cascade(signal, 10e3, "lower", 10, None, 2)
    # N is too low
    with pytest.warns(UserWarning, match="N is 4 but should be at least 5"):
        pfilt.low_shelve_cascade(signal, 1e3, "upper", -60, None, 4, 4)


def test_low_shelve_cascade():

    x = pf.signals.impulse(2**10)

    # low shelve cascade (slope < 12.04 dB)
    y, N, ideal = pf.dsp.filter.low_shelve_cascade(
        x, 4e3, "upper", -20, None, 4)
    # test reference
    reference = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            "filter.shelve_cascade_low.csv"))
    # restricting rtol was not needed locally. It was added for tests to
    # pass on travis ci
    npt.assert_allclose(
        y.time, np.atleast_2d(reference), rtol=.01, atol=1e-10)
    # test N and ideal
    assert N == 4
    npt.assert_equal(ideal.freq.flatten(), [.1, .1, 1, 1])
    npt.assert_equal(ideal.frequencies, [0, 250, 4000, 22050])

    # low shelve cascade (slope > 12.04 dB)
    y, N, ideal = pf.dsp.filter.low_shelve_cascade(
        x, 4e3, "upper", -20, None, 1)
    # test N and ideal
    # (y is not tested because is is tested above and all parameters determing
    #  y are tested `intest_shelving_cascade_slope_parameters``)
    assert N == 2
    npt.assert_equal(ideal.freq.flatten(), [.1, .1, 1, 1])
    npt.assert_equal(ideal.frequencies, [0, 2000, 4000, 22050])


def test_high_shelve_cascade():

    x = pf.signals.impulse(2**10)

    # high shelve cascade (slope < 12.04 dB)
    y, N, ideal = pf.dsp.filter.high_shelve_cascade(
        x, 250, "lower", -20, None, 4)
    # test reference
    reference = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            "filter.shelve_cascade_high.csv"))
    # restricting rtol was not needed locally. It was added for tests to
    # pass on travis ci
    npt.assert_allclose(
        y.time, np.atleast_2d(reference), rtol=.01, atol=1e-10)
    # test N and ideal
    assert N == 4
    npt.assert_equal(ideal.freq.flatten(), [1, 1, .1, .1])
    npt.assert_equal(ideal.frequencies, [0, 250, 4000, 22050])

    # low shelve cascade (slope > 12.04 dB)
    y, N, ideal = pf.dsp.filter.high_shelve_cascade(
        x, 250, "lower", -20, None, 1)
    # test N and ideal
    # (y is not tested because is is tested above and all parameters determing
    #  y are tested `intest_shelving_cascade_slope_parameters``)
    assert N == 2
    npt.assert_equal(ideal.freq.flatten(), [1, 1, .1, .1])
    npt.assert_equal(ideal.frequencies, [0, 250, 500, 22050])

    # high shelve cascade (upper characteristic frequency exceeds Nyquist)
    y, N, ideal = pf.dsp.filter.high_shelve_cascade(
        x, 22050/2, "lower", -20, None, 2)
    # test reference
    reference = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            "filter.shelve_cascade_high_exceed_nyquist.csv"))
    # restricting rtol was not needed locally. It was added for tests to
    # pass on travis ci
    npt.assert_allclose(
        y.time, np.atleast_2d(reference), rtol=.01, atol=1e-10)
    # test N and ideal
    assert N == 1
    npt.assert_equal(ideal.freq.flatten(), [1, 1, 10**(-10/20)])
    npt.assert_equal(ideal.frequencies, [0, 11025, 22050])


def test_shelve_cascade_filter_return():

    y, *_ = pf.dsp.filter.high_shelve_cascade(
        None, 250, "lower", -20, None, 4, sampling_rate=44100)

    assert isinstance(y, pf.FilterSOS)
