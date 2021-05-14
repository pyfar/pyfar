import numpy as np
import numpy.testing as npt
import scipy.signal as sgn
import pytest
import pyfar

from pyfar import dsp


def test_phase_rad(sine_plus_impulse):
    """Test the function returning the phase of a signal in radians."""
    phase = dsp.phase(sine_plus_impulse, deg=False, unwrap=False)
    truth = np.angle(sine_plus_impulse.freq)
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_phase_deg(sine_plus_impulse):
    """Test the function returning the phase of a signal in degrees."""
    phase = dsp.phase(sine_plus_impulse, deg=True, unwrap=False)
    truth = np.degrees(np.angle(sine_plus_impulse.freq))
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_phase_unwrap(sine_plus_impulse):
    """Test the function returning the unwrapped phase of a signal."""
    phase = dsp.phase(sine_plus_impulse, deg=False, unwrap=True)
    truth = np.unwrap(np.angle(sine_plus_impulse.freq))
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_phase_deg_unwrap(sine_plus_impulse):
    """Test the function returning the unwrapped phase of a signal in deg."""
    phase = dsp.phase(sine_plus_impulse, deg=True, unwrap=True)
    truth = np.degrees(np.unwrap(np.angle(sine_plus_impulse.freq)))
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_group_delay_single_channel(impulse_group_delay):
    """Test the function returning the group delay of a signal,
    single channel."""
    signal = impulse_group_delay[0]

    with pytest.raises(ValueError, match="Invalid method"):
        dsp.group_delay(signal, method='invalid')

    with pytest.raises(ValueError, match="not supported"):
        dsp.group_delay(signal, method='fft', frequencies=[1, 2, 3])

    grp = dsp.group_delay(signal, method='scipy')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)

    grp = dsp.group_delay(signal, method='fft')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)

    grp = dsp.group_delay(
        signal, method='fft')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)


def test_group_delay_two_channel(impulse_group_delay_two_channel):
    """Test the function returning the group delay of a signal,
    two channels."""
    signal = impulse_group_delay_two_channel[0]
    grp = dsp.group_delay(signal, method='scipy')
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(grp, impulse_group_delay_two_channel[1], rtol=1e-10)

    grp = dsp.group_delay(signal, method='fft')
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(grp, impulse_group_delay_two_channel[1], rtol=1e-10)


def test_group_delay_two_by_two_channel(
        impulse_group_delay_two_by_two_channel):
    """Test the function returning the group delay of a signal,
    2-by-2 channels."""
    signal = impulse_group_delay_two_by_two_channel[0]
    grp = dsp.group_delay(signal)
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(
        grp, impulse_group_delay_two_by_two_channel[1], rtol=1e-10)


def test_group_delay_custom_frequencies(impulse_group_delay):
    """Test the function returning the group delay of a signal,
    called for specific frequencies."""
    signal = impulse_group_delay[0]
    # Single frequency, of type int
    frequency = 1000
    frequency_idx = np.abs(signal.frequencies-frequency).argmin()
    grp = dsp.group_delay(signal, frequency, method='scipy')
    assert grp.shape == ()
    npt.assert_allclose(grp, impulse_group_delay[1][0, frequency_idx])

    # Multiple frequencies
    frequency = np.array([1000, 2000])
    frequency_idx = np.abs(
        signal.frequencies-frequency[..., np.newaxis]).argmin(axis=-1)
    grp = dsp.group_delay(signal, frequency, method='scipy')
    assert grp.shape == (2,)
    npt.assert_allclose(grp, impulse_group_delay[1][0, frequency_idx])


def test_xfade(impulse):
    first = np.ones(5001)
    idx_1 = 500
    second = np.ones(5001)*2
    idx_2 = 1000

    res = dsp.dsp._cross_fade(first, second, [idx_1, idx_2])
    np.testing.assert_array_almost_equal(first[:idx_1], res[:idx_1])
    np.testing.assert_array_almost_equal(second[idx_2:], res[idx_2:])

    idx_1 = 501
    idx_2 = 1000
    res = dsp.dsp._cross_fade(first, second, [idx_1, idx_2])
    np.testing.assert_array_almost_equal(first[:idx_1], res[:idx_1])
    np.testing.assert_array_almost_equal(second[idx_2:], res[idx_2:])


def test_regu_inversion(impulse):

    with pytest.raises(
            ValueError, match='needs to be of type pyfar.Signal'):
        dsp.regularized_spectrum_inversion('error', (1, 2))

    with pytest.raises(
            ValueError, match='lower and upper limits'):
        dsp.regularized_spectrum_inversion(impulse, (2))

    res = dsp.regularized_spectrum_inversion(impulse * 2, [200, 10e3])

    ind = impulse.find_nearest_frequency([200, 10e3])
    npt.assert_allclose(
        res.freq[:, ind[0]:ind[1]],
        np.ones((1, ind[1]-ind[0]), dtype=complex)*0.5)

    npt.assert_allclose(res.freq[:, 0], [0.25])
    npt.assert_allclose(res.freq[:, -1], [0.25])


def test_time_window_default():
    """ Test time_window function with default values."""
    sig = pyfar.Signal(np.ones(10), 2)
    sig_win = dsp.time_window(sig, interval=(0, sig.n_samples-1))
    time_win = np.atleast_2d(sgn.windows.hann(10, sym=True))
    npt.assert_allclose(sig_win.time, time_win)


def test_time_window_input():
    """Test errors when calling with incorrect parameters."""
    sig = pyfar.Signal(np.ones(5), 2)
    with pytest.raises(TypeError, match='signal'):
        dsp.time_window([1., 2.], interval=(0, 4))
    with pytest.raises(ValueError, match='shape'):
        dsp.time_window(sig, interval=(0, 4), shape='top')
    with pytest.raises(TypeError, match='crop'):
        dsp.time_window(sig, interval=(0, 4), crop='t')
    with pytest.raises(ValueError, match='unit'):
        dsp.time_window(sig, interval=[0, 1], unit='kg')
    with pytest.raises(TypeError, match='interval'):
        dsp.time_window(sig, interval=1)
    with pytest.raises(ValueError, match='contain'):
        dsp.time_window(sig, interval=[1, 2, 3])
    with pytest.raises(ValueError, match='longer'):
        dsp.time_window(sig, interval=[1, 11])
    with pytest.raises(ValueError):
        dsp.time_window(sig, interval=['a', 'b'])


def test_time_window_interval_types():
    sig = pyfar.Signal(np.ones(10), 2)
    dsp.time_window(sig, interval=(1, 2))
    dsp.time_window(sig, interval=[1, 2])
    dsp.time_window(sig, interval=(1, 2, 3, 4))
    dsp.time_window(sig, interval=[1, 2, 3, 4])


def test_time_window_interval_order_error():
    """ Test errors for incorrect order of values in interval."""
    sig = pyfar.Signal(np.ones(10), 2)
    with pytest.raises(ValueError, match='ascending'):
        dsp.time_window(sig, interval=[2, 1])
    with pytest.raises(ValueError, match='ascending'):
        dsp.time_window(sig, interval=[1, 2, 3, 0])


def test_time_window_interval_unit_error():
    """ Test errors for incorrect boundaries in combinations with unit."""
    sig = pyfar.Signal(np.ones(10), 2)
    with pytest.raises(ValueError, match='than signal'):
        dsp.time_window(sig, interval=[0, 11], unit='samples')
    with pytest.raises(ValueError, match='than signal'):
        dsp.time_window(sig, interval=[0, 6], unit='s')
    with pytest.raises(ValueError, match='than signal'):
        dsp.time_window(sig, interval=[0, 6e3], unit='ms')


def test_time_window_crop_none():
    """ Test crop option 'none'."""
    sig = pyfar.Signal(np.ones(10), 2)
    sig_win = dsp.time_window(sig, interval=[1, 3], crop='none')
    assert sig_win.n_samples == 10


def test_time_window_crop_interval():
    """ Test truncation of windowed signal to interval."""
    sig = pyfar.Signal(np.ones(10), 2)
    sig_win = dsp.time_window(
        sig, interval=[1, 3], shape='symmetric', unit='samples',
        crop='window')
    assert sig_win.n_samples == 3
    sig_win = dsp.time_window(
        sig, interval=[0.5, 1.5], shape='symmetric', unit='s',
        crop='window')
    assert sig_win.n_samples == 3
    sig_win = dsp.time_window(
        sig, interval=[500, 1500], shape='symmetric', unit='ms',
        crop='window')
    assert sig_win.n_samples == 3
    sig_win = dsp.time_window(
        sig, interval=[1, 3], shape='left', crop='window')
    assert sig_win.n_samples == 9
    sig_win = dsp.time_window(
        sig, interval=[1, 3], shape='right', crop='window')
    assert sig_win.n_samples == 4


def test_time_window_crop_end():
    """ Test crop option 'end'."""
    sig = pyfar.Signal(np.ones(10), 2)
    sig_win = dsp.time_window(
        sig, interval=[1, 3], shape='symmetric', unit='samples',
        crop='end')
    assert sig_win.n_samples == 4
    sig_win = dsp.time_window(
        sig, interval=[0.5, 1.5], shape='symmetric', unit='s',
        crop='end')
    assert sig_win.n_samples == 4
    sig_win = dsp.time_window(
        sig, interval=[500, 1500], shape='symmetric', unit='ms',
        crop='end')
    assert sig_win.n_samples == 4
    sig_win = dsp.time_window(
        sig, interval=[1, 3], shape='left', crop='end')
    assert sig_win.n_samples == 10
    sig_win = dsp.time_window(
        sig, interval=[1, 3], shape='right', crop='end')
    assert sig_win.n_samples == 4


def test_time_window_symmetric():
    """ Test window option symmetric."""
    sig = pyfar.Signal(np.ones(10), 2)
    sig_win = dsp.time_window(
        sig, interval=[1, 5], window='hann', shape='symmetric',
        crop='window')
    time_win = np.atleast_2d(sgn.windows.hann(5, sym=True))
    npt.assert_allclose(sig_win.time, time_win)


def test_time_window_symmetric_zero():
    """ Test window option symmetric_zero."""
    sig = pyfar.Signal(np.ones(12), 2)
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 4], shape='symmetric_zero')
    time_win = np.array([[1, 1, 1, 0.75, 0.25, 0, 0, 0, 0.25, 0.75, 1, 1]])
    npt.assert_allclose(sig_win.time, time_win)


def test_time_window_left():
    """ Test window options left."""
    sig = pyfar.Signal(np.ones(7), 1)
    # Odd number of samples, crop='none'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 4], shape='left', crop='none')
    time_win = np.array([[0, 0, 0.25, 0.75, 1, 1, 1]])
    npt.assert_allclose(sig_win.time, time_win)
    # Even number of samples, crop='none'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 5], shape='left', crop='none')
    time_win = np.array([[0, 0, 1/6, 3/6, 5/6, 1, 1]])
    npt.assert_allclose(sig_win.time, time_win)
    # crop='end'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 5], shape='left', crop='end')
    time_win = np.array([[0, 0, 1/6, 3/6, 5/6, 1, 1]])
    npt.assert_allclose(sig_win.time, time_win)
    # crop='window'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 5], shape='left', crop='window')
    time_win = np.array([[1/6, 3/6, 5/6, 1, 1]])
    npt.assert_allclose(sig_win.time, time_win)


def test_time_window_right():
    """ Test window options right."""
    sig = pyfar.Signal(np.ones(7), 1)
    # Odd number of samples, crop='none'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 4], shape='right', crop='none')
    time_win = np.array([[1, 1, 1, 0.75, 0.25, 0, 0]])
    npt.assert_allclose(sig_win.time, time_win)
    # Even number of samples, crop='none'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 5], shape='right', crop='none')
    time_win = np.array([[1, 1, 1, 5/6, 3/6, 1/6, 0]])
    npt.assert_allclose(sig_win.time, time_win)
    # crop='end'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 5], shape='right', crop='end')
    time_win = np.array([[1, 1, 1, 5/6, 3/6, 1/6]])
    npt.assert_allclose(sig_win.time, time_win)
    # crop='window'
    sig_win = dsp.time_window(
        sig, window='triang', interval=[2, 5], shape='right', crop='window')
    time_win = np.array([[1, 1, 1, 5/6, 3/6, 1/6]])
    npt.assert_allclose(sig_win.time, time_win)


def test_time_window_interval_four_values():
    """ Test time_window with four values given in interval."""
    sig = pyfar.Signal(np.ones(9), 1)
    sig_win = dsp.time_window(
        sig, window='triang', interval=[1, 3, 6, 7], unit='samples',
        crop='none')
    time_win = np.array([[0, 0.25, 0.75, 1, 1, 1, 1, 0.5, 0]])
    npt.assert_allclose(sig_win.time, time_win)
    sig = pyfar.Signal(np.ones(10), 1)
    sig_win = dsp.time_window(
        sig, window='triang', interval=[1, 3, 6, 7], unit='samples',
        crop='none')
    time_win = np.array([[0, 0.25, 0.75, 1, 1, 1, 1, 0.5, 0, 0]])
    npt.assert_allclose(sig_win.time, time_win)


def test_time_window_multichannel():
    """ Test time_window of multichannel signal."""
    time = np.array(
        [[[1, 1, 1, 1], [2, 2, 2, 2]], [[3, 3, 3, 3], [4, 4, 4, 4]]])
    sig = pyfar.Signal(time, 1)
    sig_win = dsp.time_window(
        sig, window='triang', interval=[1, 2], shape='symmetric',
        crop='window')
    time_win = np.array(
        [[[0.5, 0.5], [1, 1]], [[1.5, 1.5], [2, 2]]])
    npt.assert_allclose(sig_win.time, time_win)


def test_kaiser_window_beta():
    """ Test function call."""
    A = 51
    beta = dsp.kaiser_window_beta(A)
    beta_true = 0.1102*(A-8.7)
    assert beta == beta_true
    A = 30
    beta = dsp.kaiser_window_beta(A)
    beta_true = 0.5842*(A-21)**0.4+0.07886*(A-21)
    assert beta == beta_true
    A = 10
    beta = dsp.kaiser_window_beta(A)
    beta_true = 0.0
    assert beta == beta_true
