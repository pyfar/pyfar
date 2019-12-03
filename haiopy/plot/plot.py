import matplotlib.pyplot as plt
import numpy as np

from ._interaction import (
    AxisModifierLinesLinYAxis,
    AxisModifierLinesLogYAxis)
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox)


def plot_time_dB(signal, log_prefix=20, log_reference=1, **kwargs):
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
    x_data = signal.times
    y_data = signal.time.T

    fig, axes = plt.subplots()

    data_dB = log_prefix*np.log10(np.abs(y_data)/log_reference)
    ymax = np.max(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    axes.plot(x_data, data_dB)

    axes.set_ylim((ymin, ymax))
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Amplitude [dB re {}]".format(log_reference))
    axes.grid(True)

    modifier = AxisModifierLinesLogYAxis(axes, fig)
    modifier.connect()

    plt.show()

    return axes


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
    x_data = signal.times
    y_data = signal.time.T

    fig, ax = plt.subplots()

    ax.plot(x_data, y_data)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    modifier = AxisModifierLinesLinYAxis(ax, fig)
    modifier.connect()

    plt.show()

    return ax


def plot_freq(signal, log_prefix=20, log_reference=1, **kwargs):
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
    matplotlib.pyplot.magnitude_spectrum() : Plot the magnitudes of the
        corresponding frequencies.

    Examples
    --------
    """
    time_data = signal.time
    sampling_rate = signal.sampling_rate

    fig, axes = plt.subplots()

    eps = np.finfo(float).tiny
    data_dB = log_prefix*np.log10(np.abs(signal.freq)/log_reference + eps)
    axes.semilogx(signal.frequencies, data_dB.T)


    axes.set_xlabel("Frequency [Hz]")
    axes.set_ylabel("Magnitude [dB]")

    axes.set_xscale('log')
    axes.grid(True, 'both')

    ymax = np.max(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    axes.set_ylim((ymin, ymax))
    axes.set_xlim((20, signal.sampling_rate/2))

    modifier = AxisModifierLinesLogYAxis(axes, fig)
    modifier.connect()
    axes.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    axes.xaxis.set_major_formatter(
        LogFormatterITAToolbox())

    plt.show()

    return axes
