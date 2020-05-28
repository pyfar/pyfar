import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .. import dsp
from scipy import signal as sgn

from ._interaction import (
    AxisModifierLinesLinYAxis,
    AxisModifierLinesLogYAxis)
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)
#plt.style.use('ggplot')


def prepare_plot(ax=None):
    plt.style.use('ggplot')
    plt.style.use('haiopy.mplstyle')
    if ax is None:
        ax = plt.gca()
    fig = plt.gcf()

    return fig, ax

def plot_time(signal, ax=None, **kwargs):
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
    fig, ax = prepare_plot(ax)
    x_data = signal.times
    y_data = signal.time.T

    ax.plot(x_data, y_data, **kwargs)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    #ax.grid(True)
    ax.set_xlim((signal.times[0], signal.times[-1]))

    modifier = AxisModifierLinesLinYAxis(ax, fig)
    modifier.connect()

    return ax

def plot_time_dB(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
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
    fig, ax = prepare_plot(ax)

    x_data = signal.times
    y_data = signal.time.T

    data_dB = log_prefix*np.log10(np.abs(y_data)/log_reference)
    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    ax.plot(x_data, data_dB, **kwargs)

    ax.set_xlim((signal.times[0], signal.times[-1]))
    ax.set_ylim((ymin, ymax))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [dB]")
    ax.grid(True)

    modifier = AxisModifierLinesLogYAxis(ax, fig)
    modifier.connect()

    return ax

def plot_freq(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
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
    fig, ax = prepare_plot(ax)

    time_data = signal.time
    sampling_rate = signal.sampling_rate

    #fig, axes = plt.subplots()

    eps = np.finfo(float).tiny
    data_dB = log_prefix*np.log10(np.abs(signal.freq)/log_reference + eps)
    ax.semilogx(signal.frequencies, data_dB.T, **kwargs)


    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")

    ax.set_xscale('log')
    ax.grid(True, 'both')

    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((20, signal.sampling_rate/2))

    modifier = AxisModifierLinesLogYAxis(ax, fig)
    modifier.connect()
    ax.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(
        LogFormatterITAToolbox())
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
    fig, ax = prepare_plot(ax)

    time_data = signal.time
    sampling_rate = signal.sampling_rate

    phase_data = dsp.phase(signal, deg=deg, unwrap=unwrap)

    ylabel_string = 'Phase '
    if(unwrap==True):
        ylabel_string += '(unwrapped) '
    elif(unwrap=='360'):
        ylabel_string += '(wrapped to 360) '

    if deg:
        ylabel_string += '[deg]'
    else:
        ylabel_string += '[rad]'
        ax.yaxis.set_major_locator(MultipleFractionLocator(np.pi, 2))
        ax.yaxis.set_minor_locator(MultipleFractionLocator(np.pi, 6))
        ax.yaxis.set_major_formatter(MultipleFractionFormatter(
            nominator=1, denominator=2, base=np.pi, base_str='\pi'))

    ax.semilogx(signal.frequencies, phase_data.T, **kwargs)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(ylabel_string)

    ax.set_xscale('log')
    ax.grid(True, 'both')

    ymin = np.nanmin(phase_data)-0.001 # more elegant solution possible?
    ymax = np.nanmax(phase_data)

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((20, signal.sampling_rate/2))

    modifier = AxisModifierLinesLogYAxis(ax, fig)
    modifier.connect()
    ax.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(
        LogFormatterITAToolbox())

    return ax

def plot_group_delay(signal, ax=None, **kwargs):
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
    fig, ax = prepare_plot(ax)

    data = dsp.group_delay(signal)

    ax.semilogx(signal.frequencies, data, **kwargs)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Group delay [sec]")

    ax.set_xscale('log')
    ax.grid(True, 'both')

    # TODO: Set y limits correctly.
    ax.set_xlim((20, signal.sampling_rate/2))

    modifier = AxisModifierLinesLogYAxis(ax, fig)
    modifier.connect()
    ax.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(
        LogFormatterITAToolbox())

    return ax

def _plot_spectrogram(signal, log=False, scale='dB', window='hann',
                     window_length='auto', window_overlap_fct=0.5,
                     cmap=mpl.cm.get_cmap(name='magma'), ax=None, **kwargs):
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

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    scipy.signal.spectrogram() : Generate the spectrogram for a given signal.
    matplotlib.pyplot.specgram() : Plot the spectrogram for a given signal.

    Examples
    --------
    """
    fig, ax = prepare_plot(ax)

    # Define figure and axes for plot:
    #fig, axes = plt.subplots(1,2,gridspec_kw={"width_ratios":[1, 0.05]})

    if window_length == 'auto':
        window_length  = 2**dsp.nextpow2(signal.n_samples / 2000)
        if window_length < 1024: window_length = 1024
    window_overlap = int(window_length * window_overlap_fct)

    spectrum, freqs, t, im = ax.specgram(
            x=np.squeeze(signal.time),
            NFFT=window_length,
            Fs=signal.sampling_rate,
            window=sgn.get_window(window, window_length),
            noverlap=window_overlap,
            mode='magnitude',
            scale=scale,
            cmap=cmap,
            **kwargs)

    # Adjust axes:
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_xlim((signal.times[0], signal.times[-1]))
    ax.set_ylim((20, signal.sampling_rate/2))
    if log:
        ax.set_yscale('symlog')
        ax.yaxis.set_major_locator(LogLocatorITAToolbox())
    ax.yaxis.set_major_formatter(LogFormatterITAToolbox())
    ax.grid(ls='dotted')

    # Colorbar:
    #cb = plt.colorbar(mappable=im, cax=axes[1])
    #cb.set_label('Modulus [dB]')

    return ax

def plot_spectrogram(signal, log=False, scale='dB', window='hann',
                     window_length='auto', window_overlap_fct=0.5,
                     cmap=mpl.cm.get_cmap(name='magma'), ax=None, **kwargs):
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

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    scipy.signal.spectrogram() : Generate the spectrogram for a given signal.
    matplotlib.pyplot.specgram() : Plot the spectrogram for a given signal.

    Examples
    --------
    """
    plt.style.use('ggplot')
    plt.style.use('haiopy.mplstyle')

    # Define figure and axes for plot:
    fig, ax = plt.subplots(1,2,gridspec_kw={"width_ratios":[1, 0.05]})

    ax[0] = _plot_spectrogram(signal, log, scale, window,
                     window_length, window_overlap_fct,
                     cmap, ax[0], **kwargs)

    # Colorbar:
    cb = plt.colorbar(mappable=ax[0].get_images()[0], cax=ax[1])
    cb.set_label('Modulus [dB]')

    return ax

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


    Examples
    --------
    """
    fig, ax = plt.subplots(2,1,sharex=True)
    plot_freq(signal, log_prefix, log_reference, ax[0], **kwargs)
    plot_phase(signal, deg, unwrap, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    return ax

def plot_freq_group_delay(signal, log_prefix=20, log_reference=1, **kwargs):
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
    axes : Axes object or array of Axes objects.

    See Also
    --------

    Examples
    --------
    """
    fig, ax = plt.subplots(2,1,sharex=True)
    plot_freq(signal, log_prefix, log_reference, ax[0], **kwargs)
    plot_group_delay(signal, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    return ax

def plot_all(signal, **kwargs):
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
    axes : Axes object or array of Axes objects.

    See Also
    --------

    Examples
    --------
    """

    plt.style.use('ggplot')
    plt.style.use('haiopy.mplstyle')

    fig, ax = plt.subplots(3,2,sharex='col')
    fig.set_size_inches(10, 8)

    plot_time(signal, ax=ax[0,0], **kwargs)
    plot_time_dB(signal, ax=ax[1,0], **kwargs)
    _plot_spectrogram(signal, ax=ax[2,0], **kwargs)

    plot_freq(signal, ax=ax[0,1], **kwargs)
    plot_phase(signal, ax=ax[1,1], **kwargs)
    plot_group_delay(signal, ax=ax[2,1], **kwargs)

    ax[0,0].set_xlabel(None)
    ax[1,0].set_xlabel(None)
    ax[0,1].set_xlabel(None)
    ax[1,1].set_xlabel(None)

    plt.tight_layout()
    return ax
