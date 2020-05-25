import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .. import dsp

from ._interaction import (
    AxisModifierLinesLinYAxis,
    AxisModifierLinesLogYAxis)
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)

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
    data = dsp.phase(signal, deg=deg, unwrap=unwrap)

    ylabel_string = 'Phase '
    if(unwrap==True):
        ylabel_string += '(unwrapped) '
    elif(unwrap=='360'):
        ylabel_string += '(wrapped to 360) '

    if deg:
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

def plot_group_delay(signal, **kwargs):
    """Plot the group delay of a given signal.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------

    Examples
    --------
    """

    fig, axes = plt.subplots()
    data = dsp.group_delay(signal)

    axes.semilogx(signal.frequencies, data, **kwargs)

    axes.set_xlabel("Frequency [Hz]")
    axes.set_ylabel("Group delay [sec]")

    axes.set_xscale('log')
    axes.grid(True, 'both')

    # TODO: Set y limits correctly.
    axes.set_xlim((20, signal.sampling_rate/2))

    modifier = AxisModifierLinesLogYAxis(axes, fig)
    modifier.connect()
    axes.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    axes.xaxis.set_major_formatter(
        LogFormatterITAToolbox())

    return axes

def plot_spectrogram(signal, log=False, nodb=False, window='hann',
                     window_length='auto', window_overlap_fct=0.5,
                     log_prefix=20, log_reference=1, cut=False,
                     clim=np.array([]), cmap=mpl.cm.get_cmap(name='magma')):
    """Plots the spectrogram for a given signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    window : String (Default: 'hann')
        Specifies the window type. See scipy.signal.get_window for details.
    window_length : integer
        Specifies the window length. If not set, it will be automatically
        calculated.
    window_overlap_fct : double
        Ratio of points to overlap between fft segments [0...1]
    log_prefix : integer
        Prefix for Decibel calculation.
    log_reference : double
        Prefix for Decibel calculation.
    log : Boolean
        Speciefies, whether the y axis is plotted logarithmically.
        Defaults to False.
    nodb : Boolean
        Speciefies, whether the spectrogram is plotted in decibels.
        Defaults to False.
    cut : Boolean
        Cut results to specified clim vector to avoid sparcles.
        Defaults to False.
    clim : np.array()
        Array of limits for the colorbar [lower, upper].
    cmap : matplotlib.colors.Colormap(name, N=256)
        Colormap for spectrogram. Defaults to matplotlibs 'magma' colormap.

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    scipy.signal.spectrogram() : Generate the spectrogram for a given signal.

    Examples
    --------
    """
    # Define figure and axes for plot:
    fig, axes = plt.subplots(1,2,gridspec_kw={"width_ratios":[1, 0.05]})
    # Generate necessarry data for plot:
    frequencies, times, spectrogram, clim = dsp.spectrogram(
            signal=signal, window=window, window_length=window_length,
            window_overlap_fct=window_overlap_fct, log_prefix=log_prefix,
            log_reference=log_reference, log=log, nodb=nodb, cut=cut, clim=clim)

    # Adjust axes:
    axes[0].pcolormesh(times, frequencies, spectrogram,
                   norm=None, cmap=cmap, shading='flat')
    axes[0].set_ylabel('Frequency [Hz]')
    axes[0].set_xlabel('Time [sec]')
    axes[0].set_ylim((20, signal.sampling_rate/2))
    if log:
        axes[0].set_yscale('symlog')
        axes[0].yaxis.set_major_locator(LogLocatorITAToolbox())
    axes[0].yaxis.set_major_formatter(LogFormatterITAToolbox())

    # Colorbar:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=None)
    sm.set_clim(vmin=clim[0], vmax=clim[1])
    cb = fig.colorbar(sm, cax=axes[1])
    cb.set_label('Modulus [dB]')

    return axes

def plot_freq_phase(signal, log_prefix=20, log_reference=1, deg=False,
                    unwrap=False, **kwargs):
    """Plot the magnitude and phase of the spectrum on the positive frequency
    axis.

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
    # TODO: Check possibilities for modular fuction call.
    # Is it possible to plot existing axes to a new figure?

    #fig, axes = plt.subplots(2,1,sharex=True)
    #plot_freq(signal, log_prefix, log_reference, axes[0], **kwargs)
    #plot_phase(signal, deg, unwrap, axes[1], **kwargs)
    #plt.show()
    return

def plot_freq_group_delay(signal, **kwargs): # TODO
    return

def plot_all(signal, **kwargs):  # TODO
    return
