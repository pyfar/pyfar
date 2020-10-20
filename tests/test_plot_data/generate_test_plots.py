import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import haiopy.plot as plot
from haiopy import Signal

def generate_test_plots(dir='tests/test_plot_data/baseline/'):
    """ Generate the reference plots used for testing the plot functions.

    Parameters
    -------
    dir : String
        Path to save the reference plots.
    """
    matplotlib.use('Agg')
    matplotlib.testing.set_reproducibility_for_testing()
    function_list = ['time',
                     'time_dB',
                     'freq',
                     'phase',
                     'group_delay',
                     'spectrogram',
                     'freq_phase',
                     'freq_group_delay',
                     'summary']

    for function_name in function_list:
        getattr(plot, function_name)(sine_plus_impulse())
        plt.savefig((dir + function_name + '.png'))


    # additional plots to check different options:
    plot.phase(sine_plus_impulse(), deg=True, unwrap=False)
    plt.savefig((dir + 'plot_phase_deg' + '.png'))
    plot.phase(sine_plus_impulse(), deg=False, unwrap=True)
    plt.savefig((dir + 'plot_phase_unwrap' + '.png'))
    plot.phase(sine_plus_impulse(), deg=True, unwrap=True)
    plt.savefig((dir + 'plot_phase_unwrap_deg' + '.png'))


def sine_plus_impulse():
    """ Generate a sine signal, superposed with an impulse at the beginning
        and sampling_rate = 4000 Hz.

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

    if fullperiod:
        # round to the nearest frequency resulting in a fully periodic sine
        # signal in the given time interval
        num_periods = np.floor(n_samples / sampling_rate * frequency)
        frequency = num_periods * sampling_rate / n_samples

    # time signal:
    times = np.arange(0, n_samples) / sampling_rate
    time = np.sin(2 * np.pi * times * frequency)
    time[idx_impulse] = amplitude_impulse

    # create Signal object:
    signal_object = Signal(time, sampling_rate, 'time', 'power')

    return signal_object
