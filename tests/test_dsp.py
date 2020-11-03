import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from unittest import mock
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

def test_group_delay(sine_plus_impulse_mock):
    """Test the function returning the group delay of a signal."""
    group_delay = dsp.group_delay(sine_plus_impulse_mock)
    phase = np.unwrap(np.angle(sine_plus_impulse_mock.freq))
    bin_dist = (sine_plus_impulse_mock.sampling_rate /
                    sine_plus_impulse_mock.n_samples)
    truth = np.squeeze((- np.diff(phase,1,-1, prepend=0) /
                   (bin_dist * 2 * np.pi)))
    npt.assert_allclose(group_delay, truth, rtol=1e-7)

#def test_wrap_to_2pi():
#def test_nextpow2():

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
    amplitude_sine = 1
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
