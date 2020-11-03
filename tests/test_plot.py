import matplotlib
import matplotlib.pyplot as plt
import matplotlib.testing as mpt
from matplotlib.testing.compare import compare_images
import pytest
import numpy as np
from unittest import mock
import os
import haiopy.plot as plot
from haiopy import Signal

# flag for creating new baseline plots (required if the plot look changed)
create_baseline = False

# path handling
base_path = os.path.join('tests', 'test_plot_data')
baseline_path = os.path.join(base_path, 'baseline')
output_path = os.path.join(base_path, 'output')

if not os.path.isdir(base_path):
    os.mkdir(base_path)
if not os.path.isdir(baseline_path):
    os.mkdir(baseline_path)
if not os.path.isdir(output_path):
    os.mkdir(output_path)


def test_line_plots(sine_plus_impulse_mock, create_baseline=create_baseline):

    # test all plots with default parameters
    function_list = [plot.line.time,
                     plot.line.time_dB,
                     plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay,
                     plot.line.spectrogram,
                     plot.line.freq_phase,
                     plot.line.freq_group_delay,
                     plot.line.summary]

    for function in function_list:
        print(f"Testing: {function.__name__}")
        # file names
        filename = 'line_' + function.__name__ + '.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)
        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        function(sine_plus_impulse_mock)

        # save baseline if it does not exist
        # make sure to visually check the baseline uppn creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # testing
        compare_images(baseline, output, tol=10)


def test_line_phase_options(sine_plus_impulse_mock):
    parameter_list = [['line_phase_deg.png', True, False],
                      ['line_phase_unwrap.png', False, True],
                      ['line_phase_deg_unwrap.png', True, True]]

    for param in parameter_list:
        print(f"Testing: {param[0]}")
        # file names
        filename = param[0]
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)
        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plot.line.phase(sine_plus_impulse_mock, deg=param[1], unwrap=param[2])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppn creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # testing
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
    freq = np.ones(int(n_samples/2+1), dtype=np.complex_) * norm * 2/n_samples

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
