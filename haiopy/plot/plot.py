import matplotlib.pyplot as plt
import numpy as np
from .dsp import *

from ._interaction import (
    AxisModifierLinesLinYAxis,
    AxisModifierLinesLogYAxis)
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)


def plot_time_dB(signal, log_prefix=20, log_reference=1, **kwargs):
    """Plot the time signal of a haiopy audio signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

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
    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    axes.plot(x_data, data_dB, **kwargs)

    axes.set_ylim((ymin, ymax))
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Amplitude [dB]")
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
        Keyword arguments that are piped to matplotlib.pyplot.plot

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

    ax.plot(x_data, y_data, **kwargs)

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
        Keyword arguments that are piped to matplotlib.pyplot.plot

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
    axes.semilogx(signal.frequencies, data_dB.T, **kwargs)


    axes.set_xlabel("Frequency [Hz]")
    axes.set_ylabel("Magnitude [dB]")

    axes.set_xscale('log')
    axes.grid(True, 'both')

    ymax = np.nanmax(data_dB)
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

    #plt.show()

    return axes

def plot_phase(signal, deg=False, unwrap=False, **kwargs):
    """Plot the phase of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    deg : Boolean
        Specifies, whether the phase is plotted in degrees or radians.
    unwrap : Boolean
        Specifies, whether the phase is unwrapped or not.
        If set to "360", the phase is wrapped to 2 pi.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    matplotlib.pyplot.phase_spectrum() : Plot the phase of the
        corresponding frequencies.

    Examples
    --------
    """
    time_data = signal.time
    sampling_rate = signal.sampling_rate

    fig, axes = plt.subplots()
    data = np.angle(signal.freq)

    # TODO: move to dsp.py
    ylabel_string = 'Phase '
    if(unwrap==True):
        data = np.unwrap(data)
        ylabel_string += '(unwrapped) '
    elif(unwrap=='360'):
        data = dsp.wrap_to_2pi(np.unwrap(data))
        ylabel_string += '(wrapped to 360) '

    if deg:
        data = dsp.rad_to_deg(data)
        ylabel_string += '[deg]'
    else:
        ylabel_string += '[rad]'
        axes.yaxis.set_major_locator(MultipleFractionLocator(np.pi, 2))
        axes.yaxis.set_minor_locator(MultipleFractionLocator(np.pi, 6))
        axes.yaxis.set_major_formatter(MultipleFractionFormatter(
            nominator=1, denominator=2, base=np.pi, base_str='\pi'))

    axes.semilogx(signal.frequencies, data.T, **kwargs)

    axes.set_xlabel("Frequency [Hz]")
    axes.set_ylabel(ylabel_string)

    axes.set_xscale('log')
    axes.grid(True, 'both')

    ymin = np.nanmin(data)
    ymax = np.nanmax(data)

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

def plot_freq_phase(signal, log_prefix=20, log_reference=1, deg=False, unwrap=False, **kwargs):
    """Plot the magnitude and phase of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    deg : Boolean
        Specifies, whether the phase is plotted in degrees or radians.
    unwrap : Boolean
        Specifies, whether the phase is unwrapped or not.
        If set to "360", the phase is wrapped to 2 pi.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    matplotlib.pyplot.phase_spectrum() : Plot the phase of the
        corresponding frequencies.

    Examples
    --------
    """
    # TODO:

    #fig, axes = plt.subplots(2,1,sharex=True)
    #plot_freq(signal, log_prefix, log_reference, axes[0], **kwargs)
    #plot_phase(signal, deg, unwrap, axes[0], **kwargs)
    #plt.show()
    return axes

def plot_groupdelay(signal, **kwargs): # TODO
    return

def plot_freq_groupdelay(signal, **kwargs): # TODO
    return

def plot_spectrogram(signal, **kwargs): # TODO
    return

def plot_all(signal, **kwargs):  # TODO
    return
