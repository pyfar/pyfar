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


def test_error_complex_and_real_signals():
    """Test if a mixture of complex and real signals raises an error."""

    signal_1 = pf.Signal(1, 1)
    signal_2 = pf.Signal(1+1j, 1, is_complex=True)

    with pytest.raises(ValueError, match="Both signals must be complex or"):
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


@pytest.mark.parametrize('length_1', [4, 5])
@pytest.mark.parametrize('delay_1', [0, 1, 2])
@pytest.mark.parametrize('length_2', [4, 5])
@pytest.mark.parametrize('delay_2', [0, 1, 2])
def test_linear_mode_1d(length_1, delay_1, length_2, delay_2):
    """Test linear mode for different combinations of 1D signals."""

    # compute correlation
    signal_1 = pf.signals.impulse(length_1, delay_1)
    signal_2 = pf.signals.impulse(length_2, delay_2)

    correlation, lags, argmax = correlate(signal_1, signal_2)

    # test output shapes
    assert correlation.shape == (1, length_1 + length_2 - 1)
    assert lags.shape == (length_1 + length_2 - 1, )
    assert argmax.shape == (1, )

    # test lag that maximizes the correlation
    lag = delay_1 - delay_2
    assert argmax == lag

    # test correlation and lags
    npt.assert_almost_equal(correlation[0, lags==lag], 1., 10)
    npt.assert_almost_equal(correlation[0, lags!=lag], 0., 10)


@pytest.mark.parametrize('length', [4, 5])
@pytest.mark.parametrize('delay_1', [0, 1, 2])
@pytest.mark.parametrize('delay_2', [0, 1, 2])
def test_cyclic_mode_1d(length, delay_1, delay_2):
    """Test cyclic mode for different combinations of 1D signals."""

    # compute correlation
    signal_1 = pf.signals.impulse(length, delay_1)
    signal_2 = pf.signals.impulse(length, delay_2)

    correlation, lags, argmax = correlate(signal_1, signal_2, mode='cyclic')

    # test output shapes
    assert correlation.shape == (1, length)
    assert lags.shape == (length, )
    assert argmax.shape == (1, )

    # test lag that maximizes the correlation
    # (lag must be adjusted to cyclic case)
    lag = delay_1 - delay_2
    if length % 2 and lag < -(length//2):
        lag = length + lag
    if not length % 2 and lag <= -(length//2):
        lag = length + lag

    assert argmax == lag

    # test correlation and lags
    npt.assert_almost_equal(correlation[0, lags==lag], 1., 10)
    npt.assert_almost_equal(correlation[0, lags!=lag], 0., 10)


def test_linear_mode_nd():
    """Test broadcasting and n-dimensional signals in linear mode."""

    # compute correlation
    delays = np.array([[0, 1], [2, 3]], dtype=int)
    signal_1 = pf.signals.impulse(5, 0)
    signal_2 = pf.signals.impulse(5, delays)

    correlation, lags, argmax = correlate(signal_1, signal_2)

    # test output shapes
    assert correlation.shape == (2, 2, 9)
    assert lags.shape == (9, )
    assert argmax.shape == (2, 2)

    # test lag that maximizes the correlation
    npt.assert_equal(argmax, -delays)

    # test correlation and lags
    for dim_1 in range(2):
        for dim_2 in range(2):
            npt.assert_almost_equal(
                correlation[dim_1, dim_2, lags==-delays[dim_1, dim_2]], 1., 10)
            npt.assert_almost_equal(
                correlation[dim_1, dim_2, lags!=-delays[dim_1, dim_2]], 0., 10)


def test_complex_signals():
    """Test with complex signals"""

    signal_1 = pf.Signal([0, 0, 1+0j], 1, is_complex=True)
    signal_2 = pf.Signal([0+1j, 0, 0], 1, is_complex=True)

    correlation, lags, argmax = correlate(signal_1, signal_2)

    # test lag that maximizes the correlation
    lag = 2
    assert argmax == lag

    # test correlation and lags
    npt.assert_almost_equal(correlation[0, lags==lag], 0-1j, 10)
    npt.assert_almost_equal(correlation[0, lags!=lag], 0+0j, 10)
