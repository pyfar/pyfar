import matplotlib.pyplot as plt
# plt.style.use(['default', 'ggplot', 'haiopy.mplstyle'])
import matplotlib as mpl
from matplotlib.pyplot import colorbar
import numpy as np
from .. import dsp
from scipy import signal as sgn
from haiopy import Signal
import warnings
import os


from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)


def _prepare_plot(ax=None):
    """Activates the stylesheet and returns a figure to plot on.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be returned.

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes.
    """
    if ax is None:
        fig = plt.gcf()
        fig.clear()
        ax = fig.add_subplot()
        fig.set_size_inches(plt.rcParams.get('figure.figsize'))
    else:
        fig = ax.figure
        #fig.clear()
        #ax = fig.add_subplot()
        #fig.set_size_inches(plt.rcParams.get('figure.figsize'))
    return fig, ax


def _return_default_colors_rgb(**kwargs):
    """Replace color in kwargs with haiopy default color if possible."""

    # haiopy default colors
    colors = {'p': '#5F4690',  # purple
              'b': '#1471B9',  # blue
              't': '#4EBEBE',  # turqois
              'g': '#078554',  # green
              'l': '#72AF47',  # light green
              'y': '#ECAD20',  # yellow
              'o': '#E07D26',  # orange
              'r': '#D83C27'}  # red

    if 'c' in kwargs and isinstance(kwargs['c'], str):
        kwargs['c'] = colors[kwargs['c']] \
            if kwargs['c'] in colors else kwargs['c']
    if 'color' in kwargs and isinstance(kwargs['color'], str):
        kwargs['color'] = colors[kwargs['color']] \
            if kwargs['color'] in colors else kwargs['color']

    return kwargs


def _time(signal, ax=None, **kwargs):
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

    kwargs = _return_default_colors_rgb(**kwargs)

    fig, ax = _prepare_plot(ax)
    x_data = signal.times
    y_data = signal.time.T

    ax.plot(x_data, y_data, **kwargs)
    ax.set_xscale('linear')
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Amplitude")
    ax.set_xlim((signal.times[0], signal.times[-1]))

    plt.tight_layout()

    return ax

def _time_dB(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
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

    fig, ax = _prepare_plot(ax)

    kwargs = _return_default_colors_rgb(**kwargs)

    x_data = signal.times
    y_data = signal.time.T
    data_dB = log_prefix*np.log10(np.abs(
        y_data/np.amax(np.abs(y_data)))/log_reference)
    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    ax.plot(x_data, data_dB, **kwargs)

    ax.set_xlim((signal.times[0], signal.times[-1]))
    ax.set_ylim((ymin, ymax))
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Amplitude in dB")
    plt.tight_layout()

    return ax

def _freq(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
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

    fig, ax = _prepare_plot(ax)

    kwargs = _return_default_colors_rgb(**kwargs)

    eps = np.finfo(float).tiny
    data_dB = log_prefix*np.log10(np.abs(signal.freq)/log_reference + eps)
    ax.semilogx(signal.frequencies, data_dB.T, **kwargs)

    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel("Magnitude in dB")

    ax.set_xscale('log')
    ax.grid(True, 'both')

    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((20, signal.sampling_rate/2))

    ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())
    plt.tight_layout()

    return ax

def _phase(signal, deg=False, unwrap=False, ax=None, **kwargs):
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

    fig, ax = _prepare_plot(ax)

    kwargs = _return_default_colors_rgb(**kwargs)

    time_data = signal.time
    sampling_rate = signal.sampling_rate

    phase_data = dsp.phase(signal, deg=deg, unwrap=unwrap)

    # Construct the correct label string:
    ylabel_string = 'Phase '
    if(unwrap==True):
        ylabel_string += '(unwrapped) '
    elif(unwrap=='360'):
        ylabel_string += '(wrapped to 360) '

    if deg:
        ylabel_string += 'in degree'
        y_margin = 5
    else:
        ylabel_string += 'in radians'
        ax.yaxis.set_major_locator(MultipleFractionLocator(np.pi, 2))
        ax.yaxis.set_minor_locator(MultipleFractionLocator(np.pi, 6))
        ax.yaxis.set_major_formatter(MultipleFractionFormatter(
            nominator=1, denominator=2, base=np.pi, base_str='\pi'))
        y_margin = np.radians(5)

    ymin = np.nanmin(phase_data)-y_margin # more elegant solution possible?
    ymax = np.nanmax(phase_data)+y_margin

    ax.semilogx(signal.frequencies, phase_data.T, **kwargs)
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel(ylabel_string)
    ax.set_xscale('log')
    ax.grid(True, 'both')
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((20, signal.sampling_rate/2))
    ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())
    plt.tight_layout()

    return ax

def _group_delay(signal, ax=None, **kwargs):
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

    fig, ax = _prepare_plot(ax)

    kwargs = _return_default_colors_rgb(**kwargs)

    data = dsp.group_delay(signal)
    ax.semilogx(signal.frequencies, data.T, **kwargs)

    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel("Group delay in s")

    ax.set_xscale('log')
    ax.grid(True, 'both')

    # TODO: Set y limits correctly.
    ax.set_xlim((20, signal.sampling_rate/2))

    ax.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(
        LogFormatterITAToolbox())
    plt.tight_layout()

    return ax

def _spectrogram(signal,
                      log=False,
                      nodb=False,
                      window='hann',
                      window_length='auto',
                      window_overlap_fct=0.5,
                      cmap=mpl.cm.get_cmap(name='magma'),
                      ax=None,
                      cut=False,
                      **kwargs):
    """Plots the spectrogram for a given signal object without colorbar.

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

    See Also
    --------
    scipy.signal.spectrogram() : Generate the spectrogram for a given signal.
    matplotlib.pyplot.specgram() : Plot the spectrogram for a given signal.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    if signal.time.shape[0] > 1:
        warnings.warn(
            "You are trying to plot a spectrogram of "
            + str(signal.time.shape) + " signals.")
        signal.time = signal.time[0]

    fig, ax = _prepare_plot(ax)

    if window_length == 'auto':
        window_length  = 2**dsp.nextpow2(signal.n_samples / 2000)
        if window_length < 1024: window_length = 1024
    window_overlap = int(window_length * window_overlap_fct)


    frequencies, times, spectrogram, clim = dsp.spectrogram(signal,
                                                            window,
                                                            window_length,
                                                            window_overlap_fct,
                                                            20,
                                                            1,
                                                            log,
                                                            nodb,
                                                            cut)

    ax.pcolormesh(times, frequencies, spectrogram, cmap=cmap, shading='gouraud')
    #spectrum, freqs, t, im = ax.specgram(
    #        x=np.squeeze(signal.time),
    #        NFFT=window_length,
    #        Fs=signal.sampling_rate,
    #        window=sgn.get_window(window, window_length),
    #        noverlap=window_overlap,
    #        mode='magnitude',
    #        scale=scale,
    #        cmap=cmap,
    #        **kwargs)

    # Adjust axes:
    ax.set_ylabel('Frequency in Hz')
    ax.set_xlabel('Time in s')
    ax.set_xlim((signal.times[0], signal.times[-1]))
    ax.set_ylim((20, signal.sampling_rate/2))

    if log:
        ax.set_yscale('symlog')
        ax.yaxis.set_major_locator(LogLocatorITAToolbox())
    ax.yaxis.set_major_formatter(LogFormatterITAToolbox())
    ax.grid(ls='dotted')
    plt.tight_layout()

    return ax


def _spectrogram_cb(signal,
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

    # Define figure and axes for plot:
    fig, ax = _prepare_plot(ax)
    ax = ax.figure.subplots(1,2,gridspec_kw={"width_ratios":[1, 0.05]})
    fig.axes[0].remove()

    ax[0] = _spectrogram(signal, log, nodb, window,
                     window_length, window_overlap_fct,
                     cmap, ax[0], **kwargs)

    # Colorbar:
    for PCM in ax[0].get_children():
        if type(PCM) == mpl.collections.QuadMesh:
            break

    cb = plt.colorbar(PCM, cax=ax[1])
    cb.set_label('Modulus in dB')
    plt.tight_layout()

    return ax

def _freq_phase(signal,
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

    fig, ax = _prepare_plot(ax)
    kwargs = _return_default_colors_rgb(**kwargs)

    ax = fig.subplots(2,1,sharex=False)
    fig.axes[0].remove()
    _freq(signal, log_prefix, log_reference, ax[0], **kwargs)
    _phase(signal, deg, unwrap, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()
    plt.tight_layout()

    return ax

def _freq_group_delay(signal, log_prefix=20, log_reference=1, ax=None,
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

    fig, ax = _prepare_plot(ax)
    kwargs = _return_default_colors_rgb(**kwargs)

    ax = fig.subplots(2,1,sharex=False)
    fig.axes[0].remove()
    _freq(signal, log_prefix, log_reference, ax[0], **kwargs)
    _group_delay(signal, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()
    plt.tight_layout()

    return ax

def _plot_all(signal, ax=None, **kwargs):
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

    # Setup figure, axes and grid:
    fig, ax = _prepare_plot(ax)
    ax = fig.subplots(4,2, gridspec_kw={'height_ratios':[1,1,1,0.1]})
    fig.axes[0].remove()
    fig.set_size_inches(6, 6)

    kwargs = _return_default_colors_rgb(**kwargs)

    # Time domain plots:
    _time(signal, ax=ax[0,0], **kwargs)
    _time_dB(signal, ax=ax[1,0], **kwargs)
    _spectrogram(signal, ax=ax[2,0], **kwargs)

    # Frequency domain plots:
    _freq(signal, ax=ax[0,1], **kwargs)
    _phase(signal, ax=ax[1,1], **kwargs)
    _group_delay(signal, ax=ax[2,1], **kwargs)

    # Colorbar for spectrogram:
    for PCM in ax[2,0].get_children():
        if type(PCM) == mpl.collections.QuadMesh: break
    cb = plt.colorbar(PCM, cax=ax[3,0],orientation='horizontal')
    cb.set_label('Modulus [dB]')

    # Remove unnessecary labels and ticks:
    ax[0,0].set_xlabel(None)
    ax[1,0].set_xlabel(None)
    ax[0,1].set_xlabel(None)
    ax[1,1].set_xlabel(None)
    ax[3,1].axis('off')
    #ax[0,0].get_shared_x_axes().join(ax[0,0], ax[1,0], ax[2,0])
    ax[0,0].set_xticklabels([])
    ax[1,0].set_xticklabels([])
    #ax[0,1].get_shared_x_axes().join(ax[0,1], ax[1,1], ax[2,1])
    ax[0,1].set_xticklabels([])
    ax[1,1].set_xticklabels([])
    fig.align_ylabels()

    plt.tight_layout()

    return ax
