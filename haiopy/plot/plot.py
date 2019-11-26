from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase, ToolToggleBase

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QLineEdit, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import numpy as np

from haiopy import Signal

from ._interaction import AxisModifierLinesLinYAxis, AxisModifierLinesLogYAxis
from ._interaction import AxisModifierDialog


def plot_time(signal, **kwargs):
    """Plot the time signal of a haiopy audio signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class
    **kwargs
        Arbitrary keyword arguments.
        Use 'xmin', 'xmax', 'ymin', 'ymax' to set axis limitations.

    Returns
    -------
    axes :  Axes object or array of Axes objects.

    See Also
    --------
    matplotlib.pyplot.plot() : Plot y versus x as lines and/or markers

    Examples
    --------
    """

    x_data = signal.times[0]
    y_data = signal.time.T

    fig, ax = plt.subplots()

    ax.plot(x_data, y_data)

    ax.set_title("Signal")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    modifier = AxisModifierLinesLinYAxis(ax, fig)
    modifier.connect()

    plt.show()

    return ax


def plot_freq(signal, **kwargs):
    """Plot the absolute values of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal object
        An adio signal object from the haiopy signal class
    **kwargs
        Arbitrary keyword arguments.
        Use 'xmin', 'xmax', 'ymin', 'ymax' to set axis limitations.

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    matplotlib.pyplot.magnitude_spectrum() : Plot the magnitudes of the corresponding frequencies.

    Examples
    --------
    """

    n_channels = signal.shape[0]
    time_data = signal.time
    samplingrate = signal.samplingrate

    fig, axes = plt.subplots()

    axes.set_title("Magnitude Spectrum")

    for i in range(n_channels):
        spectrum, freq, line = axes.magnitude_spectrum(time_data[i], Fs=samplingrate, scale='dB')

    axes.set_xscale('log')
    axes.grid(True)

    spectrum_db = 20 * np.log10(spectrum + np.finfo(float).tiny)
    ymax = np.max(spectrum_db)
    ymin = ymax - 90
    ymax = ymax + 10


    axes.set_ylim((ymin, ymax))

    ax = plt.gca()

    modifier = AxisModifierLinesLogYAxis(ax, fig)
    modifier.connect()

    plt.show()

    return axes

