import numpy as np
import numpy.testing as npt
import scipy.signal as sgn
import pytest
import pyfar

from pyfar.signals import impulse
from pyfar import dsp
import pyfar as pf


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
    npt.assert_allclose(
        grp, impulse_group_delay[1].flatten(), rtol=1e-10, atol=1e-10)

    grp = dsp.group_delay(signal, method='fft')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(
        grp, impulse_group_delay[1].flatten(), rtol=1e-10, atol=1e-10)

    grp = dsp.group_delay(
        signal, method='fft')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(
        grp, impulse_group_delay[1].flatten(), rtol=1e-10, atol=1e-10)


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
    npt.assert_allclose(
        grp, impulse_group_delay[1][0, frequency_idx], atol=1e-10)

    # Multiple frequencies
    frequency = np.array([1000, 2000])
    frequency_idx = np.abs(
        signal.frequencies-frequency[..., np.newaxis]).argmin(axis=-1)
    grp = dsp.group_delay(signal, frequency, method='scipy')
    assert grp.shape == (2,)
    npt.assert_allclose(
        grp, impulse_group_delay[1][0, frequency_idx], atol=1e-10)


def test_linear_phase():
    # test signal
    N = 64
    fs = 44100
    x = pf.signals.impulse(N, sampling_rate=fs)

    # test default parameters
    y = dsp.linear_phase(x, N/2)
    # test output
    assert isinstance(y, pf.Signal)
    npt.assert_allclose(dsp.group_delay(y), N / 2 * np.ones(y.n_bins))
    # test if input did not change
    npt.assert_allclose(x.time, pf.signals.impulse(N).time)

    # test group delay in seconds
    y = dsp.linear_phase(x, N / 2 / fs, unit="s")
    npt.assert_allclose(dsp.group_delay(y), N / 2 * np.ones(y.n_bins))

    # test assertion
    with pytest.raises(TypeError, match="signal must be a pyfar Signal"):
        dsp.linear_phase(1, 0)
    with pytest.raises(ValueError, match="unit is km"):
        dsp.linear_phase(x, N / 2 / fs, unit="km")


def test_linear_phase_multichannel():
    # test signal
    N = 64
    fs = 44100
    x = pf.signals.impulse(N, [0, 0], sampling_rate=fs)

    # test with scalar group delay
    y = dsp.linear_phase(x, N/2)
    npt.assert_allclose(dsp.group_delay(y[0]), N / 2 * np.ones(y.n_bins))
    npt.assert_allclose(dsp.group_delay(y[1]), N / 2 * np.ones(y.n_bins))

    # test with array like group delay
    y = dsp.linear_phase(x, [N/2, N/4])
    npt.assert_allclose(dsp.group_delay(y[0]), N / 2 * np.ones(y.n_bins))
    npt.assert_allclose(dsp.group_delay(y[1]), N / 4 * np.ones(y.n_bins))


def test_zero_phase():
    """Test zero phase generation."""
    # generate test signal and zero phase version
    signal = pf.Signal([0, 0, 0, 2], 44100)
    signal_zero = dsp.zero_phase(signal)
    # assert type and id
    assert isinstance(signal_zero, pf.Signal)
    assert id(signal) != id(signal_zero)
    # assert freq data
    assert np.any(np.abs(np.imag(signal.freq)) > 1e-15)
    assert np.all(np.abs(np.imag(signal_zero.freq)) == 0)
    # assert time data
    npt.assert_allclose(signal_zero.time, np.atleast_2d([2, 0, 0, 0]))


def test_zero_phase_assertion():
    """Test assertion when passing a TimeData object."""
    with pytest.raises(TypeError, match="Input data has to be of type"):
        dsp.zero_phase(pf.TimeData([1, 0, 0], [0, 1, 3]))


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


def test_regularized_spectrum_inversion(impulse):
    """Test regularized_spectrum_inversion"""
    res = dsp.regularized_spectrum_inversion(impulse * 2, [200, 10e3])

    ind = impulse.find_nearest_frequency([200, 10e3])
    npt.assert_allclose(
        res.freq[:, ind[0]:ind[1]],
        np.ones((1, ind[1]-ind[0]), dtype=complex)*0.5)

    npt.assert_allclose(res.freq[:, 0], [0.25])
    npt.assert_allclose(res.freq[:, -1], [0.25])


def test_regularized_spectrum_inversion_assertions(impulse):
    """Test regularized_spectrum_inversion errors"""
    with pytest.raises(
            ValueError, match='needs to be of type pyfar.Signal'):
        dsp.regularized_spectrum_inversion('error', (1, 2))

    with pytest.raises(
            ValueError, match='lower and upper limits'):
        dsp.regularized_spectrum_inversion(impulse, (2))

    with pytest.raises(
            TypeError, match="The normalized parameter"):
        dsp.regularized_spectrum_inversion(impulse, [200, 10e3], normalized=1)


def test_regularized_spectrum_inversion_normalized(impulse):
    """Test normalized parameter of regularized_spectrum_inversion"""
    impulse.fft_norm = 'amplitude'

    # normalized = True
    res = dsp.regularized_spectrum_inversion(
        impulse * 2, [200, 10e3], normalized=True)
    ind = impulse.find_nearest_frequency([200, 10e3])
    npt.assert_allclose(
        res.freq[:, ind[0]:ind[1]],
        np.ones((1, ind[1]-ind[0]), dtype=complex)*0.5*0.5*impulse.n_samples)
    # normalized = False
    res = dsp.regularized_spectrum_inversion(
        impulse * 2, [200, 10e3], normalized=False)
    ind = impulse.find_nearest_frequency([200, 10e3])
    npt.assert_allclose(
        res.freq[:, ind[0]:ind[1]],
        np.ones((1, ind[1]-ind[0]), dtype=complex)*0.5)


@pytest.mark.parametrize("shift_samples", [2, -2, 0])
@pytest.mark.parametrize("unit", ["samples", "s"])
def test_time_shift_cyclic(shift_samples, unit):
    """Test cyclic time shift using samples and seconds"""
    # generate test signal
    sampling_rate = 100
    delay = 2
    n_samples = 10
    test_signal = impulse(n_samples, delay=delay, sampling_rate=sampling_rate)

    # apply shift
    shift = shift_samples if unit == "samples" else shift_samples/sampling_rate
    shifted = dsp.time_shift(test_signal, shift, unit=unit)

    # compare to reference
    ref = impulse(
        n_samples, delay=delay+shift_samples, sampling_rate=sampling_rate)

    npt.assert_allclose(shifted.time, ref.time)


@pytest.mark.parametrize("shift", [2, -2, 0])
@pytest.mark.parametrize("pad_value", [0, np.nan])
def test_time_shift_linear(shift, pad_value):
    """Test linear time shift with different pad values"""
    # generate test signal
    sampling_rate = 100
    delay = 2
    n_samples = 10
    test_signal = impulse(n_samples, delay=delay, sampling_rate=sampling_rate)

    # apply shift
    shifted = dsp.time_shift(
        test_signal, shift, "linear", "samples", pad_value)

    # compare to reference
    ref = impulse(
        n_samples, delay=delay+shift, sampling_rate=sampling_rate)

    if pad_value != 0 and shift != 0:
        ref = pf.TimeData(ref.time, ref.times)
    if shift == 2:
        ref.time[0, :2] = pad_value
    elif shift == -2:
        ref.time[0, -2:] = pad_value

    npt.assert_allclose(shifted.time, ref.time)
    assert type(shifted) == type(ref)


@pytest.mark.parametrize("shift_samples", [(
    [1, 2, 3]), (np.array([1, 2, 3]))])
def test_time_shift_multi_dim(shift_samples):
    """Test with multi-channel signal and shift values as list and np.array"""
    delay = 2
    n_samples = 10

    # multi-dim signal with individual shifts
    test_signal = impulse(
        n_samples, delay=delay, amplitude=np.ones((2, 3)))
    shifted = dsp.time_shift(test_signal, shift_samples, unit='samples')
    ref = impulse(n_samples, delay=delay+np.array(shift_samples),
                  amplitude=np.ones((2, 3)))

    npt.assert_allclose(shifted.time, ref.time, atol=1e-16)


def test_time_shift_assertions():
    """Test assertions for shift_time"""

    # wrong mode
    with pytest.raises(ValueError, match="mode is 'cut'"):
        dsp.time_shift(impulse(10), 2, mode='cut')

    # wrong unit
    with pytest.raises(ValueError, match="unit is 'kg'"):
        dsp.time_shift(impulse(10), 2, unit='kg')

    # shift value exceeding signal length with both modes
    with pytest.raises(ValueError, match="Can not shift"):
        dsp.time_shift(impulse(10), 20, mode='linear')

    dsp.time_shift(impulse(10), 20, mode='cyclic')


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
        sig, window='triang', interval=[1, 3, 6, 7], crop='none')
    time_win = np.array([[0, 0.25, 0.75, 1, 1, 1, 1, 0.5, 0]])
    npt.assert_allclose(sig_win.time, time_win)
    sig = pyfar.Signal(np.ones(10), 1)
    sig_win = dsp.time_window(
        sig, window='triang', interval=[1, 3, 6, 7], crop='none')
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


@pytest.mark.parametrize("crop", ['none', 'window', 'end'])
def test_time_window_return_window(crop):
    """ Test return window parameter."""
    sig = pyfar.Signal(np.ones(10), 44100)
    sig_win, win = dsp.time_window(
        sig, interval=(4, 8), crop=crop, return_window=True)
    assert isinstance(win, pyfar.Signal)
    assert sig_win.sampling_rate == win.sampling_rate
    npt.assert_allclose(sig_win.time, win.time)
    desired_comment = (
        f"Time window with parameters interval=(4, 8),"
        f"window='hann', shape='symmetric', unit='samples', crop='{crop}'")
    assert win.comment == desired_comment


def test_time_window_return_window_error():
    """ Test return window with non bool parameter."""
    sig = pyfar.Signal(np.ones(10), 44100)
    with pytest.raises(TypeError, match="boolean"):
        dsp.time_window(sig, interval=(4, 8), return_window='a')


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


def test_minimum_phase_against_reference():
    # tests are separated since their reliability depends on the type of
    # filters. The homomorphic method works best for filters with odd numbers
    # of taps. Hilbert_2 approximates the ideal minimum phase
    input = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    output = [1, 0, 0, 0, 0]
    min_phase = pyfar.dsp.minimum_phase(
        pyfar.Signal(input, 44100))

    npt.assert_allclose(
        min_phase.time.flatten(), np.array(output, dtype=float),
        rtol=1e-10, atol=1e-10)


def test_minimum_phase_nfft():

    with pytest.raises(ValueError, match="n_fft is 5 but must be at least 6"):
        pf.dsp.minimum_phase(pf.Signal([0, 1, 0, 0, 0, 0], 44100), 5)


def test_minimum_phase_truncation():
    # test truncation parameter
    n_samples = 9
    imp_minphase = pyfar.dsp.minimum_phase(
        pyfar.signals.impulse(n_samples), truncate=False)

    assert imp_minphase.n_samples == n_samples


def test_minimum_phase_multidim():
    # test multidim (only shape is tested because output is tested above)
    n_samples = 8
    imp_linphase = pyfar.signals.impulse(
        n_samples+1, delay=int(n_samples/2), amplitude=np.ones((2, 3)))
    imp_minphase = pyfar.dsp.minimum_phase(
        imp_linphase, truncate=False)

    # assert imp_minphase.n_samples == imp_linphase.n_samples
    # assert imp_minphase.cshape == imp_linphase.cshape

    imp_zerophase = pyfar.signals.impulse(
        n_samples+1, amplitude=np.ones((2, 3)))

    npt.assert_allclose(imp_minphase.time, imp_zerophase.time, atol=1e-10)


def test_impulse_response_delay():
    """Test delay of an ideal impulse"""
    n_samples = 2**10
    snr = 60
    start_sample = np.array([24])

    ir = pf.signals.impulse(n_samples, delay=start_sample)
    noise = pf.signals.noise(n_samples, rms=10**(-snr/20), seed=1)

    start_sample_est = dsp.find_impulse_response_delay(ir)
    npt.assert_allclose(start_sample_est, start_sample, atol=1e-6)

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_delay(ir_awgn)
    npt.assert_allclose(start_sample_est, start_sample, atol=1e-2)


def test_impulse_response_delay_sinc():
    """Test delay of a band-limited sinc function shifted by 1/2 samples"""
    sr = 44100
    n_samples = 128
    samples = np.arange(n_samples)
    delay_samples = n_samples // 2 + 1/2

    sinc = np.sinc(samples - delay_samples)
    win = sgn.get_window('hann', n_samples, fftbins=False)

    ir = pf.Signal(sinc*win, sr)
    start_samples = pf.dsp.find_impulse_response_delay(ir)
    npt.assert_allclose(start_samples, delay_samples, atol=1e-3, rtol=1e-4)


def test_impulse_response_delay_multidim():
    """Ideal multi-dimensional Signal of ideal impulses"""
    n_samples = 2**10
    snr = 60

    start_sample = [[14, 12, 16], [24, 5, 43]]
    ir = pf.signals.impulse(n_samples, delay=start_sample)

    noise = pf.signals.noise(n_samples, rms=10**(-snr/20), seed=1)

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_delay(ir_awgn)

    npt.assert_allclose(start_sample_est, start_sample, atol=1e-2)


def test_impulse_response_start_insufficient_snr():
    n_samples = 2**9
    snr = 15

    ir = pf.signals.impulse(n_samples, 20)
    noise = pf.signals.noise(n_samples, rms=10**(-snr/20))
    ir_noise = ir + noise

    with pytest.warns(UserWarning, match='The SNR'):
        dsp.find_impulse_response_start(ir_noise)


def test_impulse_response_start():
    n_samples = 2**10
    ir = np.zeros(n_samples)
    snr = 60

    noise = pf.Signal(np.random.randn(n_samples) * 10**(-snr/20), 44100)

    start_sample = 24
    ir[start_sample] = 1

    ir = pf.Signal(ir, 44100)

    start_sample_est = dsp.find_impulse_response_start(ir)
    assert start_sample_est == start_sample - 1

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)
    assert start_sample_est == start_sample - 1


def test_impulse_response_theshold():
    n_samples = 2**10
    ir = np.zeros(n_samples)

    start_sample = 24
    ir[start_sample] = 1
    ir[start_sample-4:start_sample] = 10**(-5/10)

    ir = pf.Signal(ir, 44100)

    start_sample_est = dsp.find_impulse_response_start(ir, threshold=20)
    assert start_sample_est == start_sample - 4 - 1


def test_impulse_response_train():
    n_samples = 256
    # The start_sample is the last first below the threshold
    start_sample = 25
    delays = np.array([14, 22, 26, 30, 33])
    amplitudes = np.array([-40, -21, -6, 0, -9], dtype=float)

    ir = pf.signals.impulse(n_samples, delays, 10**(amplitudes/20))
    ir.time = np.sum(ir.time, axis=0)
    awgn = pf.signals.noise(n_samples, rms=10**(-60/20))
    ir += awgn

    start_sample_est = dsp.find_impulse_response_start(ir, threshold=20)

    assert start_sample_est == start_sample


def test_impulse_response_start_multidim():
    n_samples = 2**10
    n_channels = 3
    ir = np.zeros((n_channels, n_samples))

    snr = 60

    noise = pf.Signal(
        np.random.randn(n_channels, n_samples) * 10**(-snr/20), 44100)

    start_sample = [24, 5, 43]
    ir[[0, 1, 2], start_sample] = 1

    ir = pf.Signal(ir, 44100)

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)

    npt.assert_allclose(start_sample_est, np.array(start_sample) - 1)

    ir = np.zeros((2, n_channels, n_samples))
    noise = pf.Signal(
        np.random.randn(2, n_channels, n_samples) * 10**(-snr/20), 44100)

    start_sample_1 = [24, 5, 43]
    ir[0, [0, 1, 2], start_sample_1] = 1
    start_sample_2 = [14, 12, 16]
    ir[1, [0, 1, 2], start_sample_2] = 1

    ir = pf.Signal(ir, 44100)

    start_samples = np.vstack((start_sample_1, start_sample_2))

    ir_awgn = ir + noise
    start_sample_est = dsp.find_impulse_response_start(ir_awgn)

    npt.assert_allclose(start_sample_est, start_samples - 1)


def test_convolve_default():
    x = pf.Signal([1, 0.5, 0.25, 0], 44100)
    y = pf.Signal([1, -1, 0], 44100)

    res = dsp.convolve(x, y)
    desired = np.array([[1, -0.5, -0.25, -0.25, 0, 0]])
    np.testing.assert_allclose(res.time, desired, atol=1e-10)


def test_convolve_sampling_rate_error():
    x = pf.Signal([1, 0.5, 0.25, 0], 44100)
    y = pf.Signal([1, 0.5, 0.25, 0], 48000)

    with pytest.raises(ValueError, match="sampling rates"):
        dsp.convolve(x, y)


def test_convolve_fft_norm_error():
    x = pf.Signal([1, 0.5, 0.25, 0], 44100, fft_norm='unitary')
    y = pf.Signal([1, 0.5, 0.25, 0], 44100, fft_norm='amplitude')

    with pytest.raises(ValueError, match="fft_norm"):
        dsp.convolve(x, y)


@pytest.mark.parametrize("method", ['overlap_add', 'fft'])
@pytest.mark.parametrize("mode, desired", [
    ('full', np.array([[1, -0.5, 0.1, -0.35, -0.05, 0.01]])),
    ('cut', np.array([[1, -0.5, 0.1, -0.35]])),
    ('cyclic', np.array([[0.95, -0.49, 0.1, -0.35]]))])
def test_convolve_mode_and_method(method, mode, desired):
    x = pf.Signal([1, 0.5, 0.5, 0.1], 44100)
    y = pf.Signal([1, -1, 0.1], 44100)
    res = dsp.convolve(x, y, mode=mode, method=method)
    np.testing.assert_allclose(res.time, desired, atol=1e-10)


def test_convolve_mode_error():
    x = pf.Signal([1, 0.5, 0.25, 0], 44100)
    y = pf.Signal([1, -1, 0], 44100)

    with pytest.raises(ValueError, match='Invalid mode'):
        dsp.convolve(x, y, mode='invalid')


def test_convolve_method_error():
    x = pf.Signal([1, 0.5, 0.25, 0], 44100)
    y = pf.Signal([1, -1, 0], 44100)

    with pytest.raises(ValueError, match='Invalid method'):
        dsp.convolve(x, y, method='invalid')
