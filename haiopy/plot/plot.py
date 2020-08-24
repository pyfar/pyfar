import matplotlib.pyplot as plt
plt.style.use(['default', 'ggplot', 'haiopy.mplstyle'])
import matplotlib as mpl
import numpy as np
from .. import dsp
from scipy import signal as sgn
from .. import Signal
from . import _plot as hplt

from ._interaction import Interaction
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)


def plot_time(signal, ax=None, **kwargs):
    """Plot the time signal of a haiopy audio signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() : Plot y versus x as lines and/or markers
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_time(signal, ax, **kwargs)

    plt.tight_layout()

    ia = Interaction('line_lin_Y', ax, signal)

    return ax

def plot_time_dB(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
    """Plot the time signal of a haiopy audio signal object in Decibels.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class
    log_prefix : integer
        Prefix for logarithmic representation of the signal.
    log_reference : integer
        Reference for logarithmic representation of the signal.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() : Plot y versus x as lines and/or markers
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_time_dB(signal, log_prefix, log_reference, ax, **kwargs)

    plt.tight_layout()
    ia = Interaction('line_log_Y', ax, signal)

    return ax

def plot_freq(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
    """Plot the absolute values of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal object
        An adio signal object from the haiopy signal class
    log_prefix : integer
        Prefix for logarithmic representation of the signal.
    log_reference : integer
        Reference for logarithmic representation of the signal.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.magnitude_spectrum() : Plot the magnitudes of the
        corresponding frequencies.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_freq(signal, log_prefix, log_reference, ax, **kwargs)

    plt.tight_layout()

    ia = Interaction('line_log_Y', ax, signal)

    return ax

def plot_phase(signal, deg=False, unwrap=False, ax=None, **kwargs):
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
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.phase_spectrum() : Plot the phase of the
        corresponding frequencies.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_phase(signal, deg, unwrap, ax, **kwargs)

    plt.tight_layout()

    ia = Interaction('line_log_Y', ax, signal)

    return ax

def plot_group_delay(signal, ax=None, **kwargs):
    """Plot the group delay of a given signal.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_group_delay(signal, ax, **kwargs)

    plt.tight_layout()

    modifier = AxisModifierLinesLogYAxis(ax, signal)
    modifier.connect()

    return ax


def plot_spectrogram(signal,
                     log=False,
                     nodb=False,
                     window='hann',
                     window_length='auto',
                     window_overlap_fct=0.5,
                     cmap=mpl.cm.get_cmap(name='magma'),
                     ax=None,
                     **kwargs):
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
    scale : String
        The scaling of the values in the spec. 'linear' is no scaling. 'dB'
        returns the values in dB scale. When mode is 'psd', this is dB power
        (10 * log10). Otherwise this is dB amplitude (20 * log10). 'default' is
        'dB' if mode is 'psd' or 'magnitude' and 'linear' otherwise. This must
        be 'linear' if mode is 'angle' or 'phase'.
    cut : Boolean // TODO
        Cut results to specified clim vector to avoid sparcles.
        Defaults to False.
    cmap : matplotlib.colors.Colormap(name, N=256)
        Colormap for spectrogram. Defaults to matplotlibs 'magma' colormap.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_spectrogram_cb(signal, log, nodb, window, window_length,
                                    window_overlap_fct, cmap, ax, **kwargs)

    plt.tight_layout()
    ia = Interaction('spectrogram', ax[0], signal)

    return ax

def plot_freq_phase(signal,
                    log_prefix=20,
                    log_reference=1,
                    deg=False,
                    unwrap=False,
                    ax=None,
                    **kwargs):
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
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_freq_phase(signal, log_prefix, log_reference, deg, unwrap,
                               **kwargs)
    plt.tight_layout()
    ia = Interaction('line_lin_Y', ax, signal)

    return ax

def plot_freq_group_delay(signal, log_prefix=20, log_reference=1, ax=None,
                          **kwargs):
    """Plot the magnitude spectrum and group delay on the positive frequency
    axis.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_freq_group_delay(signal, log_prefix, log_reference,
                                     **kwargs)

    plt.tight_layout()
    ia = Interaction('line_log_Y', ax, signal)

    return ax

def plot_all(signal, ax=None, **kwargs):
    """ TODO: Implement input parameters for this function.
    Plot the time domain, the time domain in dB, the magnitude spectrum,
    the frequency domain, the phase and group delay on shared x axis.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    Examples
    --------
    This example creates a Signal object containing a sine wave and plots it
    using haiopy.

            import numpy as np
            from haiopy import Signal
            from haiopy import plot

            amplitude = 1
            frequency = 440
            sampling_rate = 44100
            num_samples = 44100

            times = np.arange(0, num_samples) / sampling_rate
            sine = amplitude * np.sin(2 * np.pi * frequency * times)
            signal_object = Signal(sine, sampling_rate, 'time', 'power')

            plot.plot_all(signal_object)
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    ax = hplt._plot_all(signal, **kwargs)

    plt.tight_layout()

    return ax
