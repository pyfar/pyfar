import pytest
import numpy as np
import numpy.testing as npt
from unittest import mock
import copy
from pyfar import Signal
from pyfar import dsp


def test_phase_rad(sine_plus_impulse_mock):
    """Test the function returning the phase of a signal in radians."""
    phase = dsp.phase(sine_plus_impulse_mock, deg=False, unwrap=False)
    truth = np.angle(sine_plus_impulse_mock.freq)
    npt.assert_allclose(phase, truth, rtol=1e-7)


def test_phase_deg(sine_plus_impulse_mock):
    """Test the function returning the phase of a signal in degrees."""
    phase = dsp.phase(sine_plus_impulse_mock, deg=True, unwrap=False)
    truth = np.degrees(np.angle(sine_plus_impulse_mock.freq))
    npt.assert_allclose(phase, truth, rtol=1e-7)


def test_phase_unwrap(sine_plus_impulse_mock):
    """Test the function returning the unwrapped phase of a signal."""
    phase = dsp.phase(sine_plus_impulse_mock, deg=False, unwrap=True)
    truth = np.unwrap(np.angle(sine_plus_impulse_mock.freq))
    npt.assert_allclose(phase, truth, rtol=1e-7)


def test_phase_deg_unwrap(sine_plus_impulse_mock):
    """Test the function returning the unwrapped phase of a signal in deg."""
    phase = dsp.phase(sine_plus_impulse_mock, deg=True, unwrap=True)
    truth = np.degrees(np.unwrap(np.angle(sine_plus_impulse_mock.freq)))
    npt.assert_allclose(phase, truth, rtol=1e-7)


def test_group_delay_single_channel(impulse_mock):
    """Test the function returning the group delay of a signal."""
    # test single channel signal
    signal = impulse_mock[0]
    grp = dsp.group_delay(signal)
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_mock[3][0] * np.ones(signal.n_bins))


def test_group_delay_two_channel(impulse_mock):
    """Test the function returning the group delay of a signal."""
    # test two channel signal
    signal = impulse_mock[1]
    grp = dsp.group_delay(signal)
    assert grp.shape == (2, signal.n_bins)
    npt.assert_allclose(grp[0], impulse_mock[3][0] * np.ones(signal.n_bins))
    npt.assert_allclose(grp[1], impulse_mock[3][1] * np.ones(signal.n_bins))


def test_group_delay_two_by_two_channel(impulse_mock):
    """Test the function returning the group delay of a signal."""
    # test two by two channel signal
    signal = impulse_mock[2]
    grp = dsp.group_delay(signal)
    assert grp.shape == (2, 2, signal.n_bins)
    npt.assert_allclose(grp[0, 0], impulse_mock[3][0] * np.ones(signal.n_bins))
    npt.assert_allclose(grp[0, 1], impulse_mock[3][1] * np.ones(signal.n_bins))
    npt.assert_allclose(grp[1, 0], impulse_mock[3][2] * np.ones(signal.n_bins))
    npt.assert_allclose(grp[1, 1], impulse_mock[3][3] * np.ones(signal.n_bins))


def test_group_delay_custom_frequencies(impulse_mock):
    """Test the function returning the group delay of a signal."""
    # test single frequency
    signal = impulse_mock[0]
    grp = dsp.group_delay(signal, 1e3)
    assert grp.shape == ()
    npt.assert_allclose(grp, 1000)

    # test multiple frequencies
    signal = impulse_mock[0]
    grp = dsp.group_delay(signal, [1e3, 2e3])
    assert grp.shape == (2, )
    npt.assert_allclose(grp, np.array([1e3, 1e3]))


def test_normalization_time_max_max_value():
    """Test the function along time, max, max & value path."""
    signal = Signal([1, 2, 1], [1, 4, 1], 44100)
    truth = Signal([0.25, 0.5, 0.25], [0.25, 1, 0.25], 44100)
    answer = dsp.normalize(signal, normalize='time', normalize_to='max',
                           channel_handling='max')
    assert answer == truth


def test_normalization_magnitude_mean_min_freqrange():
    """Test the function along magnitude, mean, min & value path."""
    signal = Signal([1, 4, 1], [1, 10, 1], 44100, domain='freq')
    truth = Signal([5, 29, 5], [5, 50, 5], 44100, domain='freq')
    answer = dsp.normalize(signal, normalize='magnitude', normalize_to='mean',
                           channel_handling='min', value=10)
    assert answer == truth


def test_normalization_logmagnitude_rms():
    """Test the function along logmagnitude, rms,mean  & freq path."""
    signal = Signal([1, 100, 10], [10, 1000, 100], 44100, domain='freq')
    truth = Signal([0.33333333, 33.33333333, 3.33333333], [3.33333333,
                   333.33333333, 33.33333333], 44100, domain='freq')
    answer = dsp.normalize(signal, normalize='log_magnitude',
                           normalize_to='rms', channel_handling='mean',
                           freq_range=[0, 60])
    npt.assert_allclose(answer, truth, rtol=1e-7)


def test_average_time_phase():
    """Test the function in time domain and phase copy"""
    signal = Signal([1, 2+1j, 1], [1, 4, 1], 44100, domain='time')
    truth = Signal([1, 3+1j, 1], 44100, domain='time')
    answer = dsp.average(signal, average_mode='time', phase_copy=True)
    assert answer == truth


def test_average_log_magnitude_weights():
    """Test the function in freq domain and weights"""
    signal = Signal([1, 2, 1], [1, 4, 1], 44100, domain='freq')
    truth = Signal([3.16227766, 316.22776602, 31.6227766], 44100,
                   domain='freq')
    answer = dsp.average(signal, average_mode='log_magnitude', weights=None)
    assert answer == truth

# def test_wrap_to_2pi():
# def test_nextpow2():


@pytest.fixture
def impulse_mock():
    """ Generate impulse signals, in order to test independently of the Signal
    object.

    Returns
    -------
    signal_1 : Signal
        single channel dirac signal, delay 1000 samples
    signal_2 : Signal
        two channel dirac signal, delays 100 and 750 samples
    signal_3 : Signal
        two by two channel dirac signal, delays [[1000, 750], [500, 250]]

    """
    n_samples = 2000
    n_bins = int(n_samples / 2 + 1)
    sampling_rate = 4000
    times = np.arange(0, n_samples) / sampling_rate
    frequencies = np.arange(n_bins) * sampling_rate / n_bins

    group_delays = [1000, 750, 500, 250]

    # time signal:
    time = np.zeros((2, 2, n_samples))
    time[0, 0, group_delays[0]] = 1
    time[0, 1, group_delays[1]] = 1
    time[1, 0, group_delays[2]] = 1
    time[1, 1, group_delays[3]] = 1

    # create a mock object of Signal class to test the plot independently
    signal_1 = mock.Mock(spec_set=Signal(time, sampling_rate))
    signal_1.time = np.squeeze(time[0, 0])
    signal_1.cshape = (1, )
    signal_1.sampling_rate = sampling_rate
    signal_1.times = times
    signal_1.n_samples = n_samples
    signal_1.frequencies = frequencies
    signal_1.n_bins = n_bins
    signal_1.signal_type = 'energy'

    signal_2 = copy.deepcopy(signal_1)
    signal_2.time = np.squeeze(time[0])
    signal_2.cshape = (2, )

    signal_3 = copy.deepcopy(signal_1)
    signal_3.time = time
    signal_3.cshape = (2, 2)

    return signal_1, signal_2, signal_3, group_delays


@pytest.fixture
def sine_plus_impulse_mock():
    """ Generate a sine signal, superposed with an impulse at the beginning
        and sampling_rate = 4000 Hz. The Fourier transform is calculated
        analytically in order to test independently of the Signal object.

    Returns
    -------
    signal : Signal
        The sine signal

    """
    n_samples = 2000
    sampling_rate = 4000
    amplitude_impulse = 1
    idx_impulse = 0
    frequency = 200
    fullperiod = True

    norm = 1/np.sqrt(2)

    if fullperiod:
        # round to the nearest frequency resulting in a fully periodic sine
        # signal in the given time interval
        num_periods = np.floor(n_samples / sampling_rate * frequency)
        frequency = num_periods * sampling_rate / n_samples

    # time signal:
    times = np.arange(0, n_samples) / sampling_rate
    time = np.sin(2 * np.pi * times * frequency)
    time[idx_impulse] = amplitude_impulse

    # frequency vector:
    frequencies = np.arange(0, int(n_samples/2+1)) / (n_samples/sampling_rate)

    # Fourier coefficients of impulse:
    freq = np.ones(int(n_samples/2+1), dtype=np.complex_) * norm * 2/n_samples

    # superpose Fourier coefficient of sine wave:
    position = int(frequency / sampling_rate * n_samples)
    freq[position] += -1j * norm

    # create a mock object of Signal class to test the plot independently
    signal_object = mock.Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time
    signal_object.sampling_rate = sampling_rate
    signal_object.times = times
    signal_object.frequencies = frequencies
    signal_object.freq = freq
    signal_object.n_samples = n_samples
    signal_object.signal_type = 'power'

    return signal_object
