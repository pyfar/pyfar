import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.signal as sgn
from .. import dsp
from pyfar import Signal
import pyfar.fft as fft
import warnings
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)


def _prepare_plot(ax=None, subplots=None):
    """Activates the stylesheet and returns a figure to plot on.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None in which case the axes are
        obtained from the current figure. A new figure is created if it does
        not exist.
    subplots : tuple of length 2
        A tuple giving the desired subplot layout. E.g., (2, 1) creates a
        subplot layout with two rows and one column. The default is None in
        which case no layout will be enforced.

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        Axes or array/list of axes.
    """
    if ax is None:
        # get current figure or create new one
        fig = plt.gcf()
        # get current axes or create new one
        ax = fig.get_axes()
        if not len(ax):
            ax = plt.gca()
        elif len(ax) == 1:
            ax = ax[0]
    else:
        # obtain figure from axis
        # (ax objects can be inside an array or list)
        if isinstance(ax, np.ndarray):
            fig = ax.flatten()[0].figure
        elif isinstance(ax, list):
            fig = ax[0].figure
        else:
            fig = ax.figure

    # check for correct subplot layout
    if subplots is not None:

        # check type
        if len(subplots) > 2 or not isinstance(subplots, tuple):
            raise ValueError(
                "subplots must be a tuple with one or two elements.")

        # tuple to check against the shape of current axis
        # (convert (N, 1) and (1, N) to (N, ) - this is what Matplotlib does)
        ax_subplots = subplots
        if len(ax_subplots) == 2:
            ax_subplots = tuple(s for s in ax_subplots if s != 1)

        # check if current axis has the correct numner of subplots
        create_subplots = True
        if isinstance(ax, list):
            if len(ax) == np.prod(ax_subplots):
                create_subplots = False
        elif isinstance(ax, np.ndarray):
            if ax.shape == ax_subplots:
                create_subplots = False

        # create subplots
        if create_subplots:
            fig.clf()
            ax = fig.subplots(subplots[0], subplots[1], sharex=False)

    return fig, ax


def _set_axlim(ax, setter, low, high, limits):
    """
    Set axis limits depending on existing data.

    Sets the limits of an axis to `low` and `high` if there are no lines and
    collections asociated to the axis and to `min(limits[0], low)` and
    `max(limits[1], high)` otherwise.

    Parameters
    ----------
    ax : Matplotlib Axes Object
        axis for which the limits are to be set.
    setter : function
        function for setting the limits, .e.g., `ax.set_xlim`.
    low : number
        lower axis limit
    high : number
        upper axis limit
    limits : tuple of length 2
        current axis limits, e.g., `ax.get_xlim()`.
    """

    if not ax.lines and not ax.collections:
        # set desired limit if axis does not contain any lines or points
        setter((low, high))
    else:
        # check against current axes limits
        setter((min(limits[0], low), max(limits[1], high)))


def _return_default_colors_rgb(**kwargs):
    """Replace color in kwargs with pyfar default color if possible."""

    # pyfar default colors
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
    """Plot the time data of a signal."""

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    x_data = signal.times
    y_data = signal.time.T

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.set_xscale('linear')
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Amplitude")
    _set_axlim(ax, ax.set_xlim, signal.times[0], signal.times[-1],
               ax.get_xlim())

    # plot data
    ax.plot(x_data, y_data, **kwargs)
    plt.tight_layout()

    return ax


def _time_dB(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
    """Plot the time logairhmic data of a signal."""

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    x_data = signal.times
    y_data = signal.time.T
    # avoid any zero-values because they result in -inf in dB data
    eps = np.finfo(float).tiny
    data_dB = log_prefix * np.log10(np.abs(y_data) / log_reference + eps)
    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.set_xlabel("Time in s")
    ax.set_ylabel("Amplitude in dB")
    _set_axlim(ax, ax.set_xlim, signal.times[0], signal.times[-1],
               ax.get_xlim())
    _set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())

    # plot data
    ax.plot(x_data, data_dB, **kwargs)
    plt.tight_layout()

    return ax


def _freq(signal, log_prefix=20, log_reference=1, ax=None, **kwargs):
    """
    Plot the logarithmic absolute spectrum on the positive frequency axis.
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    eps = np.finfo(float).tiny
    data_dB = log_prefix*np.log10(np.abs(signal.freq)/log_reference + eps)
    ymax = np.nanmax(data_dB)
    ymin = ymax - 90
    ymax = ymax + 10

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel("Magnitude in dB")
    ax.set_xscale('log')
    ax.grid(True, 'both')
    _set_axlim(ax, ax.set_xlim, max(20, signal.frequencies[1]),
               signal.sampling_rate/2, ax.get_xlim())
    _set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    # plot data
    ax.semilogx(signal.frequencies, data_dB.T, **kwargs)
    plt.tight_layout()

    return ax


def _phase(signal, deg=False, unwrap=False, ax=None, **kwargs):
    """Plot the phase of the spectrum on the positive frequency axis."""

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # prepare figure
    _, ax = _prepare_plot(ax)

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    phase_data = dsp.phase(signal, deg=deg, unwrap=unwrap)
    # Construct the correct label string:
    ylabel_string = 'Phase '
    if unwrap == '360':
        ylabel_string += '(wrapped to 360) '
    elif unwrap:
        ylabel_string += '(unwrapped) '

    if deg:
        ylabel_string += 'in degree'
        y_margin = 5
    else:
        ylabel_string += 'in radians'
        ax.yaxis.set_major_locator(MultipleFractionLocator(np.pi, 2))
        ax.yaxis.set_minor_locator(MultipleFractionLocator(np.pi, 6))
        ax.yaxis.set_major_formatter(MultipleFractionFormatter(
            nominator=1, denominator=2, base=np.pi, base_str=r'\pi'))
        y_margin = np.radians(5)
    ymin = np.nanmin(phase_data) - y_margin  # more elegant solution possible?
    ymax = np.nanmax(phase_data) + y_margin

    # prepare figure
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel(ylabel_string)
    ax.set_xscale('log')
    ax.grid(True, 'both')
    _set_axlim(ax, ax.set_xlim, max(20, signal.frequencies[1]),
               signal.sampling_rate/2, ax.get_xlim())
    _set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    # plot data
    ax.semilogx(signal.frequencies, phase_data.T, **kwargs)
    plt.tight_layout()

    return ax


def _group_delay(signal, ax=None, **kwargs):
    """Plot the group delay on the positive frequency axis."""

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    data = dsp.group_delay(signal) / signal.sampling_rate

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel("Group delay in s")
    ax.set_xscale('log')
    ax.grid(True, 'both')
    _set_axlim(ax, ax.set_xlim, max(20, signal.frequencies[1]),
               signal.sampling_rate/2, ax.get_xlim())
    _set_axlim(ax, ax.set_ylim, .5 * np.nanmin(data), 1.5 * np.nanmax(data),
               ax.get_ylim())
    ax.xaxis.set_major_locator(
        LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(
        LogFormatterITAToolbox())

    # plot data
    ax.semilogx(signal.frequencies, data.T, **kwargs)
    plt.tight_layout()

    return ax


def _spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                 yscale='linear',
                 window='hann', window_length=1024, window_overlap_fct=0.5,
                 cmap=mpl.cm.get_cmap(name='magma'), ax=None):
    """Plot the magnitude spectrum versus time.

    See pyfar.line.spectogram for more information.
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    if window_length > signal.n_samples:
        raise ValueError("window_length exceeds signal length")

    if np.prod(signal.cshape) > 1:
        warnings.warn(("Using only the first channel of "
                       f"{np.prod(signal.cshape)}-channel signal."))

    # take only the first channel of time data
    time_data = signal.time[tuple(np.zeros(len(signal.cshape), dtype='int'))]

    _, ax = _prepare_plot(ax)

    # get spectrogram from scipy.signal
    window_overlap = int(window_length * window_overlap_fct)
    window = sgn.get_window(window, window_length)

    frequencies, times, spectrogram = sgn.spectrogram(
            x=time_data, fs=signal.sampling_rate, window=window,
            noverlap=window_overlap, mode='magnitude', scaling='spectrum')

    # remove normalization from scipy.signal.spectrogram
    spectrogram /= np.sqrt(1 / window.sum()**2)

    # apply normalization from signal
    spectrogram = fft.normalization(
        spectrogram, window_length, signal.sampling_rate,
        signal.signal_type, signal.fft_norm, window=window)

    # get in dB
    if dB:
        eps = np.finfo(float).tiny
        spectrogram = log_prefix*np.log10(
            np.abs(spectrogram) / log_reference + eps)

    # scipy.signal takes the center of the DFT blocks as time stamp we take the
    # beginning (looks nicer in plots, both conventions are used)
    times -= times[0]

    # plot the data
    ax.pcolormesh(times, frequencies, spectrogram, cmap=cmap,
                  shading='gouraud')

    # Adjust axes:
    ax.set_ylabel('Frequency in Hz')
    ax.set_xlabel('Time in s')
    ax.set_xlim((times[0], times[-1]))
    ax.set_ylim((max(20, signal.frequencies[1]), signal.sampling_rate/2))

    if yscale == 'log':
        ax.set_yscale('symlog')
        ax.yaxis.set_major_locator(LogLocatorITAToolbox())
    ax.yaxis.set_major_formatter(LogFormatterITAToolbox())
    ax.grid(ls='dotted', color='white')
    plt.tight_layout()

    return ax, spectrogram


def _spectrogram_cb(signal, dB=True, log_prefix=20, log_reference=1,
                    yscale='linear',
                    window='hann', window_length=1024, window_overlap_fct=0.5,
                    cmap=mpl.cm.get_cmap(name='magma'), ax=None):
    """Plot the magnitude spectrum versus time.

    See pyfar.line.spectogram for more information.
    """

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # Define figure and axes for plot:
    fig, ax = _prepare_plot(ax)
    # clear figure and axis - spectogram does not work with hold
    fig.clf()
    ax = plt.gca()
    # plot the data
    ax = ax.figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
    fig.axes[0].remove()

    ax[0], spectrogram = _spectrogram(
        signal, dB, log_prefix, log_reference, yscale,
        window, window_length, window_overlap_fct,
        cmap, ax[0])

    # Colorbar:
    for PCM in ax[0].get_children():
        if type(PCM) == mpl.collections.QuadMesh:
            break

    cb = plt.colorbar(PCM, cax=ax[1])
    cb.set_label('Modulus in dB')

    # color limits
    if dB:
        ymax = np.nanmax(spectrogram)
        ymin = ymax - 90
        ymax = ymax + 10
        PCM.set_clim(ymin, ymax)

    plt.tight_layout()

    return ax


def _freq_phase(signal, log_prefix=20, log_reference=1, deg=False,
                unwrap=False, ax=None, **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout."""

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    fig, ax = _prepare_plot(ax, (2, 1))
    kwargs = _return_default_colors_rgb(**kwargs)

    _freq(signal, log_prefix, log_reference, ax[0], **kwargs)
    _phase(signal, deg, unwrap, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()
    plt.tight_layout()

    return ax


def _freq_group_delay(signal, log_prefix=20, log_reference=1, ax=None,
                      **kwargs):
    """
    Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.
    """

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    fig, ax = _prepare_plot(ax, (2, 1))
    kwargs = _return_default_colors_rgb(**kwargs)

    _freq(signal, log_prefix, log_reference, ax[0], **kwargs)
    _group_delay(signal, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()
    plt.tight_layout()

    return ax


def _custom_subplots(signal, plots, ax, **kwargs):
    """
    Generate subplot with a custom layout based on a list of plot function
    handles. The subplot layout is taken from the shape of the plot function
    handle list.

    See pyfar.plot.line._custom_subplots for more information.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    plots = np.atleast_2d(np.asarray(plots))
    subplots = plots.shape
    fig, ax = _prepare_plot(ax, subplots)

    rows = subplots[0]
    cols = subplots[1]

    for row in range(rows):
        for col in range(cols):
            # current_axis
            # this is a bit tricky: if a new multi plot is created ax will can
            # be a nested list because it is created by fig.subplots() but ax
            # will be a flat list if multi plots into an existing plot, because
            # ax is obtained from ax = plt.gca()
            try:
                ca = ax[row][col] if rows > 1 and cols > 1 else \
                     ax[max(row, col)]
            except TypeError:
                ca = ax[row * cols + col]
            # plot
            plots[row][col](signal, ax=ca, **kwargs)
