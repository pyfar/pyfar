import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
import numpy as np
from unittest import mock
import haiopy.plot as plot
from haiopy import Signal
from matplotlib.testing.compare import compare_images

baseline_path = 'tests/test_plot_data/baseline/'
output_path = 'tests/test_plot_data/output/'

def test_plot_time(sine_plus_impulse_mock):
    """Test the time plot."""
    filename = 'plot_time.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_time(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_time_dB(sine_plus_impulse_mock):
    """Test the time plot in Decibels."""
    filename = 'plot_time_dB.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_time_dB(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_freq(sine_plus_impulse_mock):
    """Test the magnitude plot."""
    filename = 'plot_freq.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_freq(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_phase(sine_plus_impulse_mock):
    """Test the phase plot."""
    filename = 'plot_phase.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_phase(sine_plus_impulse_mock, deg=False, unwrap=False)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_phase_deg(sine_plus_impulse_mock):
    """Test the phase plot."""
    filename = 'plot_phase_deg.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_phase(sine_plus_impulse_mock, deg=True, unwrap=False)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_phase_unwrap(sine_plus_impulse_mock):
    """Test the phase plot."""
    filename = 'plot_phase_unwrap.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_phase(sine_plus_impulse_mock, deg=False, unwrap=True)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_phase_unwrap_deg(sine_plus_impulse_mock):
    """Test the phase plot."""
    filename = 'plot_phase_unwrap_deg.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_phase(sine_plus_impulse_mock, deg=True, unwrap=True)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_group_delay(sine_plus_impulse_mock):
    """Test the group delay plot."""
    filename = 'plot_group_delay.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_group_delay(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_spectrogram(sine_plus_impulse_mock):
    """Test the spectrogram plot."""
    filename = 'plot_spectrogram.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_spectrogram(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_freq_phase(sine_plus_impulse_mock):
    """Test the combined magnitude/phase plot."""
    filename = 'plot_freq_phase.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_freq_phase(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_freq_group_delay(sine_plus_impulse_mock):
    """Test the combined magnitude/group delay plot."""
    filename = 'plot_freq_group_delay.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_freq_group_delay(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

def test_plot_all(sine_plus_impulse_mock):
    """Test the plot_all function."""
    filename = 'plot_all.png'
    baseline = baseline_path + filename
    output = output_path + filename
    plot.plot_all(sine_plus_impulse_mock)
    plt.savefig(output)

    assert compare_images(baseline, output, tol=0.01) == None

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
