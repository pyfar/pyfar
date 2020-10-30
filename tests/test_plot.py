import matplotlib
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import pytest
import numpy as np
from unittest import mock
import os
import haiopy.plot as plot
from haiopy import Signal
matplotlib.use('Agg')

baseline_path = os.path.join('tests', 'test_plot_data', 'baseline')
output_path = os.path.join('tests', 'test_plot_data', 'output')


def test_line_time(sine_plus_impulse_mock):
    """Test the time plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_time.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.time(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_time_dB(sine_plus_impulse_mock):
    """Test the time plot in Decibels."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_time_dB.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.time_dB(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_freq(sine_plus_impulse_mock):
    """Test the magnitude plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_freq.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.freq(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_phase(sine_plus_impulse_mock):
    """Test the phase plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_phase.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.phase(sine_plus_impulse_mock, deg=False, unwrap=False)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_phase_deg(sine_plus_impulse_mock):
    """Test the phase plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_phase_deg.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.phase(sine_plus_impulse_mock, deg=True, unwrap=False)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_phase_unwrap(sine_plus_impulse_mock):
    """Test the phase plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_phase_unwrap.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.phase(sine_plus_impulse_mock, deg=False, unwrap=True)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_phase_unwrap_deg(sine_plus_impulse_mock):
    """Test the phase plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_phase_unwrap_deg.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.phase(sine_plus_impulse_mock, deg=True, unwrap=True)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_group_delay(sine_plus_impulse_mock):
    """Test the group delay plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_group_delay.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.group_delay(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_spectrogram(sine_plus_impulse_mock):
    """Test the spectrogram plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_spectrogram.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.spectrogram(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_freq_phase(sine_plus_impulse_mock):
    """Test the combined magnitude/phase plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_freq_phase.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.freq_phase(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_freq_group_delay(sine_plus_impulse_mock):
    """Test the combined magnitude/group delay plot."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_freq_group_delay.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.freq_group_delay(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)


def test_line_all(sine_plus_impulse_mock):
    """Test the summary function."""
    matplotlib.testing.set_reproducibility_for_testing()
    filename = 'line_summary.png'
    baseline = os.path.join(baseline_path, filename)
    output = os.path.join(output_path, filename)
    plot.line.summary(sine_plus_impulse_mock)
    plt.savefig(output)

    compare_images(baseline, output, tol=10)

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
    signal_object.time = time[np.newaxis, :]
    signal_object.sampling_rate = sampling_rate
    signal_object.times = times
    signal_object.frequencies = frequencies
    signal_object.freq = freq
    signal_object.n_samples = n_samples
    signal_object.signal_type = 'power'

    return signal_object
