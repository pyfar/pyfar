import matplotlib.pyplot as plt
import numpy as np

from haiopy import Signal


def plot_time_signal(signal, xlim=None, ylim=None):
    """Docstring.

    Parameters
    ----------
    signal : Signal object

    Returns
    -------
    """
    # if not isinstance(signal, Signal):
    #     raise TypeError("Expected Signal object, not {}".format(type(signal).__name__))
    time_data = signal.time[-1]
    times = signal.times[-1]

    fig, axes = plt.subplots()

    axes.set_title("Signal")
    axes.plot(times, time_data)
    axes.set_xlabel("Time [sec]")
    axes.set_ylabel("Amplitude")
    axes.grid(True)

    plt.show()


def plot_freq_signal(signal, xmin=20, xmax=20000, ymin=None, ymax=None):
    """Docstring.

    Parameters
    ----------
    signal : Signal object

    Returns
    -------
    """
    # if not isinstance(signal, Signal):
    #     raise TypeError("Expected Signal object, not {}".format(type(signal).__name__))
    time_data = signal.time[-1]
    samplingrate = signal.samplingrate

    fig, axes = plt.subplots()

    axes.set_title("Magnitude Spectrum")
    axes.magnitude_spectrum(time_data, Fs=samplingrate, scale='dB', color='C1')
    axes.set_xscale('log')
    axes.set_xlim(xmin, xmax)
    axes.grid(True)

    plt.show()
