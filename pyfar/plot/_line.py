import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pyfar import Signal, TimeData, FrequencyData
import pyfar.dsp as dsp
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


def _lower_frequency_limit(signal):
    """Return the lower frequency limit for plotting.

    pyfar frequency plots start at 20 Hz if data is availabe . If this is not
    the case, they start at the lowest available frequency.
    """
    if isinstance(signal, (Signal, FrequencyData)):
        # indices of non-zero frequencies
        idx = np.flatnonzero(signal.frequencies)
        if len(idx) == 0:
            raise ValueError(
                "Signals must have frequencies > 0 Hz for plotting.")
        # get the frequency limit
        lower_frequency_limit = max(20, signal.frequencies[idx[0]])
    else:
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')

    return lower_frequency_limit


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


def _check_time_unit(unit):
    """Check if a valid time unit is passed."""
    units = ['s', 'ms', 'mus', 'samples']
    if unit is not None and unit not in units:
        raise ValueError(
            f"Unit is {unit} but must be {', '.join(units)}, or None.")


def _check_axis_scale(scale, axis='x'):
    """Check if a valid axis scale is passed."""
    if scale not in ['linear', 'log']:
        raise ValueError(
            f"{axis}scale is {scale} but must be 'linear', or 'log'.")


def _get_quad_mesh_from_axis(ax):
    """get the QuadMesh from an axis, if there is one.

    Parameters
    ----------
    ax : Matplotlib axes object

    Returns
    -------
    cm : Matplotlib QuadMesh object
    """
    quad_mesh_found = False
    for qm in ax.get_children():
        if type(qm) == mpl.collections.QuadMesh:
            quad_mesh_found = True
            break

    if not quad_mesh_found:
        raise ValueError("ax does not have a quad mesh.")

    return qm


def _time(signal, dB=False, log_prefix=20, log_reference=1, unit=None,
          ax=None, **kwargs):
    """Plot the time data of a signal."""

    # check input
    if not isinstance(signal, (Signal, TimeData)):
        raise TypeError('Input data has to be of type: Signal or TimeData.')
    _check_time_unit(unit)

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    data = signal.time.T
    if dB:
        # avoid any zero-values because they result in -inf in dB data
        eps = np.finfo(float).eps
        data = log_prefix * np.log10(np.abs(data) / log_reference + eps)
        ymax = np.nanmax(data)
        ymin = ymax - 90
        ymax = ymax + 10

    # auto detect the time unit
    if unit is None:
        unit = _time_auto_unit(signal.times[..., -1])
    # set the unit
    if unit == 'samples':
        times = np.arange(signal.n_samples)
    else:
        factor, unit = _deal_time_units(unit)
        times = signal.times * factor

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.set_xlabel(f"Time in {unit}")
    if dB:
        ax.set_ylabel("Amplitude in dB")
        _set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    else:
        ax.set_ylabel("Amplitude")
    _set_axlim(ax, ax.set_xlim, times[0], times[-1],
               ax.get_xlim())

    # plot data
    ax.plot(times, data, **kwargs)
    plt.tight_layout()

    return ax


def _freq(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
          ax=None, **kwargs):
    """
    Plot the logarithmic absolute spectrum on the positive frequency axis.
    """

    # check input
    if not isinstance(signal, (Signal, FrequencyData)):
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')
    _check_axis_scale(xscale)

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    if dB:
        eps = np.finfo(float).eps
        data = log_prefix*np.log10(np.abs(signal.freq)/log_reference + eps)
        ymax = np.nanmax(data)
        ymin = ymax - 90
        ymax = ymax + 10
    else:
        data = np.abs(signal.freq)

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.grid(True, 'both')
    if dB:
        ax.set_ylabel("Magnitude in dB")
        _set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    else:
        ax.set_ylabel("Magnitude")
    ax.set_xlabel("Frequency in Hz")
    _set_axlim(ax, ax.set_xlim, _lower_frequency_limit(signal),
               signal.frequencies[-1], ax.get_xlim())

    # plot data
    if xscale == 'log':
        ax.semilogx(signal.frequencies, data.T, **kwargs)
    else:
        ax.plot(signal.frequencies, data.T, **kwargs)

    # set and format ticks
    if xscale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    plt.tight_layout()

    return ax


def _phase(signal, deg=False, unwrap=False, xscale='log', ax=None, **kwargs):
    """Plot the phase of the spectrum on the positive frequency axis."""

    # check input
    if not isinstance(signal, (Signal, FrequencyData)):
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')
    _check_axis_scale(xscale)

    # prepare figure
    _, ax = _prepare_plot(ax)

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    phase_data = dsp.phase(signal, deg=deg, unwrap=unwrap)
    # Construct the correct label string:
    ylabel_string = 'Phase '
    if unwrap == '360':
        ylabel_string += '(wrapped to 360) '
    elif unwrap is True:
        ylabel_string += '(unwrapped) '
    elif not isinstance(unwrap, bool):
        raise ValueError(f"unwrap is {unwrap} but must be True, False, or 360")

    if deg:
        ylabel_string += 'in degree'
        y_margin = 5
    else:
        ylabel_string += 'in radians'
        # nice tick formatting is not done for unwrap=True. In this case
        # it can create 1000 or more ticks.
        if not unwrap or unwrap == "360":
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
    ax.grid(True, 'both')
    _set_axlim(ax, ax.set_xlim, _lower_frequency_limit(signal),
               signal.frequencies[-1], ax.get_xlim())
    _set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())

    # plot data
    if xscale == 'log':
        ax.semilogx(signal.frequencies, phase_data.T, **kwargs)
    else:
        ax.plot(signal.frequencies, phase_data.T, **kwargs)
    plt.tight_layout()

    # set and format ticks
    if xscale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    return ax


def _group_delay(signal, unit=None, xscale='log', ax=None, **kwargs):
    """Plot the group delay on the positive frequency axis."""

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')
    _check_time_unit(unit)
    _check_axis_scale(xscale)

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    data = dsp.group_delay(signal)
    # auto detect the unit
    if unit is None:
        unit = _time_auto_unit(
            np.nanmax(np.abs(data) / signal.sampling_rate))
    # set the unit
    if unit != "samples":
        factor, unit = _deal_time_units(unit)
        data = data / signal.sampling_rate * factor

    # prepare figure
    _, ax = _prepare_plot(ax)
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel(f"Group delay in {unit}")
    ax.grid(True, 'both')
    _set_axlim(ax, ax.set_xlim, _lower_frequency_limit(signal),
               signal.frequencies[-1], ax.get_xlim())
    _set_axlim(ax, ax.set_ylim, .5 * np.nanmin(data), 1.5 * np.nanmax(data),
               ax.get_ylim())

    # plot data
    if xscale == 'log':
        ax.semilogx(signal.frequencies, data.T, **kwargs)
    else:
        ax.plot(signal.frequencies, data.T, **kwargs)
    plt.tight_layout()

    # set and format ticks
    if xscale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    return ax


def _time_auto_unit(time_max):
    """
    Automatically set the unit for time axis according to the absolute maximum
    of the input data. This is a separate function for ease of testing and for
    use across different plots.

    Parameters
    ----------

    time_max : float
        Absolute maximum of the time data in seconds
    """

    if time_max == 0:
        unit = 's'
    elif time_max < 1e-3:
        unit = 'mus'
    elif time_max < 1:
        unit = 'ms'
    else:
        unit = 's'

    return unit


def _deal_time_units(unit='s'):
    """Return scaling factor and string representation for unit multiplier
    modifications.

    Parameters
    ----------
    unit : 'str'
        The unit to be used

    Returns
    -------
    factor : float
        Factor the data is to be multiplied with, i.e. 1e-3 for milliseconds
    string : str
        String representation of the unit using LaTeX formatting.
    """
    if unit == 's':
        factor = 1
        string = 's'
    elif unit == 'ms':
        factor = 1 / 1e-3
        string = 'ms'
    elif unit == 'mus':
        factor = 1 / 1e-6
        string = r'$\mathrm{\mu s}$'
    elif unit == 'samples':
        factor = 1
        string = 'samples'
    else:
        factor = 1
        string = ''
    return factor, string


def _spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                 yscale='linear', unit=None,
                 window='hann', window_length=1024, window_overlap_fct=0.5,
                 cmap=mpl.cm.get_cmap(name='magma'), ax=None):
    """Plot the magnitude spectrum versus time.

    See pyfar.line.spectogram for more information.
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')
    _check_time_unit(unit)
    _check_axis_scale(yscale, 'y')

    if window_length > signal.n_samples:
        raise ValueError("window_length exceeds signal length")

    if np.prod(signal.cshape) > 1:
        warnings.warn(("Using only the first channel of "
                       f"{np.prod(signal.cshape)}-channel signal."))

    # take only the first channel of time data
    first_channel = tuple(np.zeros(len(signal.cshape), dtype='int'))

    # get spectrogram
    frequencies, times, spectrogram = dsp.spectrogram(
        signal[first_channel], window, window_length, window_overlap_fct)

    # get magnitude data in dB
    if dB:
        eps = np.finfo(float).eps
        spectrogram = log_prefix*np.log10(
            np.abs(spectrogram) / log_reference + eps)

    # auto detect the time unit
    if unit is None:
        unit = _time_auto_unit(times[..., -1])
    # set the unit
    if unit == 'samples':
        times *= signal.sampling_rate
    else:
        factor, unit = _deal_time_units(unit)
        times = times * factor

    # plot the data
    _, ax = _prepare_plot(ax)
    ax.pcolormesh(times, frequencies, spectrogram, cmap=cmap,
                  shading='gouraud')

    # Adjust axes:
    ax.set_ylabel('Frequency in Hz')
    ax.set_xlabel(f'Time in {unit}')
    ax.set_xlim((times[0], times[-1]))
    ax.set_ylim((max(20, frequencies[1]), signal.sampling_rate/2))

    # color limits
    if dB:
        for PCM in ax.get_children():
            if type(PCM) == mpl.collections.QuadMesh:
                break

        ymax = np.nanmax(spectrogram)
        ymin = ymax - 90
        ymax = ymax + 10
        PCM.set_clim(ymin, ymax)

    if yscale == 'log':
        ax.set_yscale('symlog')
        ax.yaxis.set_major_locator(LogLocatorITAToolbox())
    ax.yaxis.set_major_formatter(LogFormatterITAToolbox())
    ax.grid(ls='dotted', color='white')
    plt.tight_layout()

    return ax, spectrogram


def _spectrogram_cb(signal, dB=True, log_prefix=20, log_reference=1,
                    yscale='linear', unit=None,
                    window='hann', window_length=1024, window_overlap_fct=0.5,
                    cmap=mpl.cm.get_cmap(name='magma'), ax=None):
    """Plot the magnitude spectrum versus time.

    See pyfar.line.spectogram for more information.
    """

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')
    _check_time_unit(unit)
    _check_axis_scale(yscale, 'y')

    # Define figure and axes for plot:
    fig, ax = _prepare_plot(ax)
    # clear figure and axis - spectogram does not work with hold
    fig.clf()
    ax = plt.gca()
    # plot the data
    ax = ax.figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
    fig.axes[0].remove()

    ax[0], _ = _spectrogram(
        signal, dB, log_prefix, log_reference, yscale, unit,
        window, window_length, window_overlap_fct,
        cmap, ax[0])

    # Colorbar:
    qm = _get_quad_mesh_from_axis(ax[0])

    cb = plt.colorbar(qm, cax=ax[1])
    cb_label = 'Magnitude in dB' if dB else 'Magnitude'
    cb.set_label(cb_label)

    plt.tight_layout()

    return ax


def _time_freq(signal, dB_time=False, dB_freq=True, log_prefix=20,
               log_reference=1, xscale='log', unit=None, ax=None, **kwargs):
    """
    Plot the time signal and magnitude spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _prepare_plot(ax, (2, 1))
    kwargs = _return_default_colors_rgb(**kwargs)

    _time(signal, dB_time, log_prefix, log_reference, unit, ax[0], **kwargs)
    _freq(signal, dB_freq, log_prefix, log_reference, xscale, ax[1], **kwargs)
    fig.align_ylabels()
    plt.tight_layout()

    return ax


def _freq_phase(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
                deg=False, unwrap=False, ax=None, **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout."""

    fig, ax = _prepare_plot(ax, (2, 1))
    kwargs = _return_default_colors_rgb(**kwargs)

    _freq(signal, dB, log_prefix, log_reference, xscale, ax[0], **kwargs)
    _phase(signal, deg, unwrap, xscale, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()
    plt.tight_layout()

    return ax


def _freq_group_delay(signal, dB=True, log_prefix=20, log_reference=1,
                      unit=None, xscale='log', ax=None, **kwargs):
    """
    Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _prepare_plot(ax, (2, 1))
    kwargs = _return_default_colors_rgb(**kwargs)

    _freq(signal, dB, log_prefix, log_reference, xscale, ax[0], **kwargs)
    _group_delay(signal, unit, xscale, ax[1], **kwargs)
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
