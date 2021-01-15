import pytest
import numpy as np
import numpy.testing as npt
from unittest import mock
import copy
from pyfar import Signal
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
    grp = dsp.group_delay(signal)
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)


def test_group_delay_two_channel(impulse_group_delay_two_channel):
    """Test the function returning the group delay of a signal,
    two channels."""
    signal = impulse_group_delay_two_channel[0]
    grp = dsp.group_delay(signal)
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(grp, impulse_group_delay_two_channel[1], rtol=1e-10)


def test_group_delay_two_by_two_channel(impulse_group_delay_two_by_two_channel):
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
    # Single frequency
    frequency = 1000
    frequency_idx = np.abs(signal.frequencies-frequency).argmin()
    grp = dsp.group_delay(signal, frequency)
    assert grp.shape == ()
    npt.assert_allclose(grp, impulse_group_delay[1][0, frequency_idx])

    # Multiple frequencies
    frequency = np.array([1e3, 2e3])
    frequency_idx = np.abs(signal.frequencies-frequency[...,np.newaxis]).argmin(axis=-1)
    grp = dsp.group_delay(signal, frequency)
    assert grp.shape == (2,)
    npt.assert_allclose(grp, impulse_group_delay[1][0, frequency_idx])


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
