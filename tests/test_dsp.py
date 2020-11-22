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


def test_group_delay(impulse_mock):
    """Test the function returning the group delay of a signal."""
    # test single channel signal
    signal = impulse_mock[0]
    grp = dsp.group_delay(signal)
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, 1000 * np.ones(signal.n_bins))

    # test two channel signal
    signal = impulse_mock[1]
    grp = dsp.group_delay(signal)
    assert grp.shape == (2, signal.n_bins)
    npt.assert_allclose(grp[0], 1000 * np.ones(signal.n_bins))
    npt.assert_allclose(grp[1],  750 * np.ones(signal.n_bins))

    # test two by two channel signal
    signal = impulse_mock[2]
    grp = dsp.group_delay(signal)
    assert grp.shape == (2, 2, signal.n_bins)
    npt.assert_allclose(grp[0, 0], 1000 * np.ones(signal.n_bins))
    npt.assert_allclose(grp[0, 1],  750 * np.ones(signal.n_bins))
    npt.assert_allclose(grp[1, 0],  500 * np.ones(signal.n_bins))
    npt.assert_allclose(grp[1, 1],  250 * np.ones(signal.n_bins))

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

    # time signal:
    time = np.zeros((2, 2, n_samples))
    time[0, 0, 1000] = 1
    time[0, 1,  750] = 1
    time[1, 0,  500] = 1
    time[1, 1,  250] = 1

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

    return signal_1, signal_2, signal_3


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
    freq = np.ones(int(n_samples/2+1),dtype=np.complex_) * norm * 2/n_samples

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
