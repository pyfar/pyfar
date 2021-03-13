import matplotlib.pyplot as plt
import matplotlib as mpl
from pyfar.plot.utils import context
from .. import Signal, TimeData, FrequencyData
from . import _line
from . import _interaction as ia


def time(signal, dB=False, log_prefix=20, log_reference=1, unit=None, ax=None,
         style='light', **kwargs):
    """Plot the time data of a signal.

    Plots `prefix * log10(signal.time / log_reference)` if dB is True
    `signal.time` otherwise.

    Parameters
    ----------
    signal : Signal, TimeData
        pyfar Signal or TimeData object.
    dB : boolean
        Indicate if the data should be plotted in dB. The default is False.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is 1.
    unit : str, None
        Unit of the time axis. Can be 's', 'ms', 'mus', or 'samples'.
        The default is None, which sets the unit to 's' (seconds), 'ms'
        (milli seconds), or 'mus' (micro seconds) depending on the maximum.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._time(signal, dB, log_prefix, log_reference, unit,
                         ax, **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.time', dB_time=dB, log_prefix=log_prefix,
        log_reference=log_reference)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def freq(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
         ax=None, style='light', **kwargs):
    """
    Plot the logarithmic absolute spectrum on the positive frequency axis.

    Plots `prefix * log10(signal.freq / log_reference)` if dB is True and
    `signal.req` otherwise.

    Parameters
    ----------
    signal : Signal, FrequencyData
        pyfar Signal or FrequencyData object.
    dB : boolean
        Indicate if the data should be plotted in dB. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer, float
        Reference for calculating the logarithmic time data. The default is 1.
    xscale : str
        'linear' or 'log' to plot on a linear or logarithmic x-axis. The
        default is 'log'.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._freq(signal, dB, log_prefix, log_reference, xscale, ax,
                         **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.freq', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, xscale=xscale)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def phase(signal, deg=False, unwrap=False, xscale='log', ax=None,
          style='light', **kwargs):
    """Plot the phase of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal, FrequencyData
        pyfar Signal or FrequencyData object.
    deg : Boolean
        Flag to plot the phase in degrees. The default is False.
    unwrap : Boolean, str
        True to unwrap the phase or "360" to unwrap the phase to 2 pi. The
        default is False.
    xscale : str
        'linear' or 'log' to plot on a linear or logarithmic x-axis. The
        default is 'log'.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._phase(signal, deg, unwrap, xscale, ax, **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.phase', deg=deg, unwrap=unwrap, xscale=xscale)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def group_delay(signal, unit=None, xscale='log', ax=None, style='light',
                **kwargs):
    """Plot the group delay on the positive frequency axis.

    Parameters
    ----------
    signal : Signal
        pyfar Signal object.
    unit : str, None
        Unit of the group delay. Can be 's', 'ms', 'mus', or 'samples'.
        The default is None, which sets the unit to 's' (seconds), 'ms'
        (milli seconds), or 'mus' (micro seconds) depending on the maximum.
    xscale : str
        'linear' or 'log' to plot on a linear or logarithmic x-axis. The
        default is 'log'.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._group_delay(signal, unit, xscale, ax, **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.group_delay', unit=unit, xscale=xscale)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                yscale='linear', unit=None, window='hann', window_length=1024,
                window_overlap_fct=0.5, cmap=mpl.cm.get_cmap(name='magma'),
                ax=None, style='light'):
    """Plot the magnitude spectrum versus time.

    Parameters
    ----------
    signal : Signal
        pyfar Signal object.
    dB : Boolean
        Flag to plot the logarithmic magnitude specturm. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is 1.
    yscale : str
        'linear' or 'log' to plot on a linear or logarithmic y-axis. The
        default is 'linear'.
    unit : str, None
        Unit of the time axis. Can be 's', 'ms', 'mus', or 'samples'.
        The default is None, which sets the unit to 's' (seconds), 'ms'
        (milli seconds), or 'mus' (micro seconds) depending on the maximum.
    window : str
        Specifies the window (See scipy.signal.get_window). The default is
        'hann'.
    window_length : integer
        Specifies the window length in samples. The default ist 1024.
    window_overlap_fct : double
        Ratio of points to overlap between fft segments [0...1]. The default is
        0.5
    cmap : matplotlib.colors.Colormap(name, N=256)
        Colormap for spectrogram. Defaults to matplotlibs 'magma' colormap.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. If not given, the current figure will be used.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    with context(style):
        ax = _line._spectrogram_cb(
            signal, dB, log_prefix, log_reference, yscale, unit,
            window, window_length, window_overlap_fct,
            cmap, ax)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.spectrogram', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, yscale=yscale, unit=unit, window=window,
        window_length=window_length, window_overlap_fct=window_overlap_fct,
        cmap=cmap)
    interaction = ia.Interaction(signal, ax[0], style, plot_parameter)
    ax[0].interaction = interaction

    return ax


def time_freq(signal, dB_time=False, dB_freq=True, log_prefix=20,
              log_reference=1, xscale='log', unit=None,
              ax=None, style='light', **kwargs):
    """
    Plot the time signal and magnitude spectrum in a 2 by 1 subplot layout.

    Parameters
    ----------
    signal : Signal
        pyfar Signal object.
    dB_time : Boolean
        Flag to plot the logarithmic time signal. The default is False.
    dB_freq : Boolean
        Flag to plot the logarithmic magnitude specturm. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time/frequency data.
        The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time/frequency data.
        The default is 1.
    xscale : str
        'linear' or 'log' to plot on a linear or logarithmic x-axis for the
        frequency plot. The default is 'log'.
    unit : str
        Unit of the time axis. Can be 's', 'ms', 'mus', or 'samples'.
        The default is None, which sets the unit to 's' (seconds), 'ms'
        (milli seconds), or 'mus' (micro seconds) depending on the maximum.
    ax : matplotlib.pyplot.axes object
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._time_freq(signal, dB_time, dB_freq, log_prefix,
                              log_reference, xscale, unit, ax, **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.time', dB_time=dB_time, log_prefix=log_prefix,
        log_reference=log_reference)
    interaction = ia.Interaction(
        signal, ax[0], style, plot_parameter, **kwargs)
    ax[0].interaction = interaction

    return ax


def freq_phase(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
               deg=False, unwrap=False, ax=None, style='light', **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout.

    Parameters
    ----------
    signal : Signal, FrequencyData
        pyfar Signal or FrequencyData object.
    dB : Boolean
        Flag to plot the logarithmic magnitude specturm. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is 1.
    deg : Boolean
        Flag to plot the phase in degrees. The default is False.
    unwrap : Boolean, str
        True to unwrap the phase or "360" to unwrap the phase to 2 pi. The
        default is False.
    xscale : str
        'linear' or 'log' to plot on a linear or logarithmic x-axis. The
        default is 'log'.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._freq_phase(signal, dB, log_prefix, log_reference, xscale,
                               deg, unwrap, ax, **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.freq', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, xscale=xscale)
    interaction = ia.Interaction(
        signal, ax[0], style, plot_parameter, **kwargs)
    ax[0].interaction = interaction

    return ax


def freq_group_delay(signal, dB=True, log_prefix=20, log_reference=1,
                     unit=None, xscale='log', ax=None, style='light',
                     **kwargs):
    """Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.

    Parameters
    ----------
    signal : Signal, FrequencyData
        pyfar Signal or FrequencyData object.
    dB : Boolean
        Flag to plot the logarithmic magnitude specturm. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is 1.
    unit : str
        Unit of the group delay. Can be 's', 'ms', 'mus', or 'samples'.
        The default is None, which sets the unit to 's' (seconds), 'ms'
        (milli seconds), or 'mus' (micro seconds) depending on the maximum.
    xscale : str
        'linear' or 'log' to plot on a linear or logarithmic x-axis. The
        default is 'log'.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._freq_group_delay(signal, dB, log_prefix, log_reference,
                                     unit, xscale, ax, **kwargs)
    plt.tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'line.freq', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, xscale=xscale)
    interaction = ia.Interaction(
        signal, ax[0], style, plot_parameter, **kwargs)
    ax[0].interaction = interaction

    return ax


def custom_subplots(signal, plots, ax=None, style='light', **kwargs):
    """
    Generate subplot with a custom layout based on a list of plot function
    handles. The subplot layout is taken from the shape of the plot function
    handle list.

    Parameters
    ----------
    signal : Signal
        A pyfar Signal object
    plots : list, nested list
        list with function handles for plotting (e.g. pyfar.plot.line.time.
        See example below)
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        'light' or 'dark' to use the pyfar plot styles or style from
        matplotlib.pyplot.available. the default is 'light'
    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    ax : axes
        List of axes handles

    Examples
    --------
    >>> from pyfar import Signal
    >>> import pyfar.plot.line as ppl
    >>>
    >>> # generate a signal
    >>> s = Signal([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 1e3)
    >>>
    >>> # two by two plot of time and frequency signals
    >>> ppl.multi(s, [[ppl.time, ppl.time_dB], [ppl.freq, ppl.group_delay]])

    """

    with context(style):
        ax = _line._custom_subplots(signal, plots, ax, **kwargs)
    plt.tight_layout()

    return ax
