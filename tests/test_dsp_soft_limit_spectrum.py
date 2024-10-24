from pyfar.dsp import soft_limit_spectrum
import pyfar as pf
import numpy as np
import numpy.testing as npt
import pytest


def test_assertions_signal():
    """Test assertion for passing wrong type of audio object."""

    with pytest.raises(TypeError, match='input signal must be a pyfar'):
        soft_limit_spectrum(pf.TimeData([1, 2, 3], [0, 1, 3]), 0, 0)


def test_assertion_direction():
    """Test assertion for passing invalid value for `direction` parameter."""

    with pytest.raises(ValueError, match="direction is 'both'"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, 0, direction='both')


def test_assertion_knee():
    """Test assertion for passing invalid value for `knee` parameter."""

    # wrong string
    with pytest.raises(ValueError, match="knee is 'tangens'"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, knee='tangens')

    # wrong number
    with pytest.raises(ValueError, match="knee is -1"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, knee=-1)

    # wrong type
    with pytest.raises(TypeError, match="knee must be"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, knee=(1, 1))


@pytest.mark.parametrize('data_in', [
    pf.Signal([1, 0, 0], 1), pf.FrequencyData([1, 1, 1], [0, 1, 3])])
def test_signal_and_frequency_data_input(data_in):
    """Test with all possible audio objects as input."""

    data_out = soft_limit_spectrum(data_in, limit=0, knee=0)
    assert type(data_out) is type(data_in)


@pytest.mark.parametrize('limit', [-10, 0, 10])
@pytest.mark.parametrize('direction', ['upper', 'lower'])
def test_limit_and_direction(limit, direction):
    """Test if all values are correctly limited."""

    data_in = pf.FrequencyData(10**(np.arange(-100, 101)/20), np.arange(201))
    data_out = soft_limit_spectrum(data_in, limit, knee=0, direction=direction)

    if direction == 'upper':
        assert np.all(20*np.log10(np.abs(data_out.freq)) <= limit + 1e-14)
    else:
        assert np.all(20*np.log10(np.abs(data_out.freq)) >= limit - 1e-14)


def test_frequency_dependend_limit():
    """Test frequency-dependent limit passed as audio object."""

    limit = np.atleast_2d([0, -6, -12])
    data_in = pf.FrequencyData(10**(np.array([-3, -3, -3])/20), [0, 1, 3])
    data_out = soft_limit_spectrum(data_in, limit, knee=0)

    limit_applied = 20*np.log10(np.abs(data_in.freq)) > limit

    # test limited values
    npt.assert_almost_equal(20*np.log10(np.abs(data_out.freq[limit_applied])),
                            limit[limit_applied])
    # test non-limited values
    npt.assert_equal(data_out.freq[~limit_applied],
                     data_in.freq[~limit_applied])


@pytest.mark.parametrize('limit', [-10, 0, 10])
@pytest.mark.parametrize('knee', [0, 5, 10])
def test_knee_width_in_db(limit, knee):
    """Test the knee width given in decibel."""

    data_in = pf.FrequencyData(10**(np.arange(-100, 101)/20), np.arange(201))
    data_out = soft_limit_spectrum(data_in, limit, knee)

    freq_in = 20*np.log10(np.abs(data_in.freq))
    freq_out = 20*np.log10(np.abs(data_out.freq))

    # categorize data
    lower_knee = freq_in <= (limit - knee / 2)
    upper_knee = freq_in >= (limit + knee / 2)
    within_knee = np.logical_not(np.logical_or(lower_knee, upper_knee))

    npt.assert_almost_equal(freq_in[lower_knee], freq_out[lower_knee], 14)
    assert np.all(freq_in[within_knee] > freq_out[within_knee])
    assert np.all(freq_out[upper_knee] <= limit + 1e-14)


@pytest.mark.parametrize('limit', [-10, 0, 10])
def test_arctan_knee(limit):
    """Test the arcus tangens knee."""

    data_in = pf.FrequencyData(10**(np.arange(-100, 101)/20), np.arange(201))
    data_out = soft_limit_spectrum(data_in, limit, 'arctan')

    # arcus tangens knee does at least a minimal gain reduction everywhere
    assert np.all(data_out.freq < data_in.freq)
    assert np.all(20*np.log10(np.abs(data_out.freq)) <= limit + 1e-14)


@pytest.mark.parametrize(('fft_norm', 'log_prefix'), [
    ('none', 20), ('unitary', 20), ('amplitude', 20), ('rms', 20),
    ('power', 10), ('psd', 10)])
def test_automatic_setting_of_log_prefix(fft_norm, log_prefix):
    """
    Test if the `log_prefix` is correctly set depending on the FFT
    normalization.
    """

    # results in linearized limits of
    # - 100 if log_prefix = 10
    # -  10 if log_prefix = 20
    limit = 20

    data_in = pf.Signal([1, 0, 0], 44100)
    data_in.fft_norm = fft_norm

    # directly set data.freq which is used inside soft_limit_spectrum
    # first value only limited if log_prefix = 20
    # second value always limited
    data_in.freq = [20, 200]

    data_out = soft_limit_spectrum(data_in, limit, knee=0)

    if log_prefix == 20:
        npt.assert_almost_equal(data_out.freq.flatten(), [10, 10])
    else:
        npt.assert_almost_equal(data_out.freq.flatten(), [20, 100])


def test_frequency_range():
    """Test limiting inside and outside user specified frequency range."""

    data_in = pf.FrequencyData([2, 2, 2, 2], [0, 1, 2, 3])
    data_out = soft_limit_spectrum(data_in, 0, knee=0, frequency_range=[1, 2])
    npt.assert_almost_equal(data_out.freq.flatten(), [2, 1, 1, 2])


def test_custom_log_prefix():
    """Test user specified `log_prefix`."""

    data_in = pf.FrequencyData([5, 15], [0, 1])
    # results in linearized limit of 10
    data_out = soft_limit_spectrum(data_in, 1, knee=0, log_prefix=1)

    npt.assert_almost_equal(data_out.freq.flatten(), [5, 10])
