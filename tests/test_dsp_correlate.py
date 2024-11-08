from pyfar.dsp import correlate
import pyfar as pf
import numpy as np
import numpy.testing as npt
import pytest


def test_error_signal_type():
    """Test if a TypeError is raised for non Signal input."""

    signal = pf.Signal(1, 1)

    with pytest.raises(TypeError, match="signal_1 and signal_2 must be"):
        correlate('not a signal', signal)

    with pytest.raises(TypeError, match="signal_1 and signal_2 must be"):
        correlate(signal, 'not a signal')

    with pytest.raises(TypeError, match="signal_1 and signal_2 must be"):
        correlate('not a signal', 'not a signal')


def test_error_different_sampling_rates():
    """Test if signals with different sampling rates raise an error."""

    signal_1 = pf.Signal(1, 1)
    signal_2 = pf.Signal(1, 2)

    with pytest.raises(ValueError, match="same sampling rate"):
        correlate(signal_1, signal_2)


def test_error_mode_string():
    """Test if a ValueError is raised for invalid mode values."""

    signal = pf.Signal(1, 1)

    with pytest.raises(ValueError, match="invalid_mode"):
        correlate(signal, signal, 'invalid_mode')


def test_error_cyclic_mode():
    """
    Test if ValueError for signals with differing length is raised in cyclic
    mode.
    """

    signal_1 = pf.Signal(1, 1)
    signal_2 = pf.Signal([1, 2], 1)

    with pytest.raises(ValueError, match="must be of the same length"):
        correlate(signal_1, signal_2, mode='cyclic')


def test_mutable_objects():
    """Test if input does not change."""

    signal_1 = pf.Signal([1, 0, 0], 1)
    signal_2 = pf.Signal([2, 0, 0], 1)

    correlate(signal_1, signal_2)

    assert signal_1 == pf.Signal([1, 0, 0], 1)
    assert signal_2 == pf.Signal([2, 0, 0], 1)


@pytest.mark.parametrize('length_1', [4, 5])
@pytest.mark.parametrize('delay_1', [0, 1, 2])
@pytest.mark.parametrize('length_2', [4, 5])
@pytest.mark.parametrize('delay_2', [0, 1, 2])
def test_full_mode_1d(length_1, delay_1, length_2, delay_2):
    """Test full mode for different combinations of 1D signals."""

    # compute correlation
    signal_1 = pf.signals.impulse(length_1, delay_1, sampling_rate=1)
    signal_2 = pf.signals.impulse(length_2, delay_2, sampling_rate=1)

    correlation = correlate(signal_1, signal_2)

    # test output format
    assert type(correlation) == pf.TimeData
    assert correlation.time.shape == (1, length_1 + length_2 - 1)
    assert correlation.times.shape == (length_1 + length_2 - 1, )

    # test lag that maximizes the correlation
    lag = (delay_1 - delay_2)

    # test correlation and lags
    npt.assert_almost_equal(
        correlation.time[0, correlation.times==lag], 1., 10)
    npt.assert_almost_equal(
        correlation.time[0, correlation.times!=lag], 0., 10)


@pytest.mark.parametrize('length', [4, 5])
@pytest.mark.parametrize('delay_1', [0, 1, 2])
@pytest.mark.parametrize('delay_2', [0, 1, 2])
def test_cyclic_mode_1d(length, delay_1, delay_2):
    """Test cyclic mode for different combinations of 1D signals."""

    # compute correlation
    signal_1 = pf.signals.impulse(length, delay_1, sampling_rate=1)
    signal_2 = pf.signals.impulse(length, delay_2, sampling_rate=1)

    correlation = correlate(signal_1, signal_2, mode='cyclic')

    # test output format
    assert type(correlation) == pf.TimeData
    assert correlation.time.shape == (1, length)
    assert correlation.times.shape == (length, )

    # test lag that maximizes the correlation
    # (lag must be adjusted to cyclic case)
    lag = delay_1 - delay_2
    if length % 2 and lag < -(length//2):
        lag = length + lag
    if not length % 2 and lag <= -(length//2):
        lag = length + lag

    # test correlation and lags
    npt.assert_almost_equal(
        correlation.time[0, correlation.times==lag], 1., 10)
    npt.assert_almost_equal(
        correlation.time[0, correlation.times!=lag], 0., 10)


def test_full_mode_nd():
    """Test broadcasting and n-dimensional signals in full mode."""

    # compute correlation
    delays = np.array([[0, 1], [2, 3]], dtype=int)
    signal_1 = pf.signals.impulse(5, 0, sampling_rate=1)
    signal_2 = pf.signals.impulse(5, delays, sampling_rate=1)

    correlation = correlate(signal_1, signal_2)

    # test output shapes
    assert type(correlation) == pf.TimeData
    assert correlation.time.shape == (2, 2, 9)
    assert correlation.times.shape == (9, )

    # test correlation and lags
    for dim_1 in range(2):
        for dim_2 in range(2):
            npt.assert_almost_equal(
                correlation.time[dim_1, dim_2,
                                 correlation.times==-delays[dim_1, dim_2]],
                1., 10)
            npt.assert_almost_equal(
                correlation.time[dim_1, dim_2,
                                 correlation.times!=-delays[dim_1, dim_2]],
                0., 10)


@pytest.mark.parametrize('normalize', [False, True])
def test_normalize(normalize):
    """Test the normalize parameter."""
    # sine with 4 samples period -> auto-correlation is 4 samples period sine
    signal = pf.signals.sine(5, 8, 1, 0, 20)
    correlation = correlate(signal, signal, 'cyclic', normalize)
    max_correlation = 1 if normalize else pf.dsp.energy(signal)

    npt.assert_almost_equal(correlation.time[0, 0::2], np.zeros(4), 10)
    npt.assert_almost_equal(correlation.time[0, 1::4],
                            -max_correlation * np.ones(2), 10)
    npt.assert_almost_equal(correlation.time[0, 3::4],
                            max_correlation * np.ones(2), 10)


def test_complex_signals():
    """Test with complex signals."""

    signal_1 = pf.Signal([0, 0, 1+0j], 1, is_complex=True)
    signal_2 = pf.Signal([0+1j, 0, 0], 1, is_complex=True)

    correlation = correlate(signal_1, signal_2)

    # test correlation values
    lag = 2
    npt.assert_almost_equal(
        correlation.time[0, correlation.times==lag], 0-1j, 10)
    npt.assert_almost_equal(
        correlation.time[0, correlation.times!=lag], 0+0j, 10)


@pytest.mark.parametrize('complex_first', [True, False])
def test_complex_and_real_signals(complex_first):
    """Test with real and complex-valued signals."""

    if complex_first:
        signal_1 = pf.Signal([0, 0, 1j], 1, is_complex=True)
        signal_2 = pf.Signal([1, 0, 0], 1, is_complex=True)
        expected = 1j
    else:
        signal_1 = pf.Signal([0, 0, 1], 1, is_complex=True)
        signal_2 = pf.Signal([1j, 0, 0], 1, is_complex=True)
        expected = -1j

    correlation = correlate(signal_1, signal_2)

    # test correlation values
    lag = 2
    npt.assert_almost_equal(
        correlation.time[0, correlation.times==lag], expected, 10)
    npt.assert_almost_equal(
        correlation.time[0, correlation.times!=lag], 0+0j, 10)
