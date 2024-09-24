import numpy as np
from pyfar import Signal, TimeData, FrequencyData
import pyfar.dsp as dsp
from . import _utils
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)
from matplotlib.ticker import NullFormatter


def _time(signal, dB=False, log_prefix=20, log_reference=1, unit="s",
          ax=None, mode='real', **kwargs):
    """Plot the time data of a signal."""

    # check input
    if not isinstance(signal, (Signal, TimeData)):
        raise TypeError('Input data has to be of type: Signal or TimeData.')
    _utils._check_time_unit(unit)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    if dB:
        # avoid any zero-values because they result in -inf in dB data
        data = dsp.decibel(signal, 'time', log_prefix, log_reference).T
        ymax = np.nanmax(data) + 10
        ymin = ymax - 100
    else:
        data = signal.time.T
    # get data defined in 'mode'
    data, y_label = _utils._assert_and_match_data_to_mode(data, signal, mode)

    # auto detect the time unit
    if unit in [None, "auto"]:
        unit = _utils._time_auto_unit(signal.times[..., -1])
    # set the unit
    if unit == 'samples':
        times = np.arange(signal.n_samples)
    else:
        factor, unit = _utils._deal_time_units(unit)
        times = signal.times * factor

    # prepare figure
    _, ax = _utils._prepare_plot(ax)
    ax.set_xlabel(f"Time in {unit}")
    if dB:
        ax.set_ylabel(y_label + ' in dB')
        _utils._set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    else:
        ax.set_ylabel(y_label)
    _utils._set_axlim(ax, ax.set_xlim, times[0], times[-1],
                      ax.get_xlim())

    # plot data
    ax.plot(times, data, **kwargs)

    return ax


def _freq(signal, dB=True, log_prefix=None, log_reference=1, freq_scale='log',
          ax=None, side='right', **kwargs):
    """
    Plot the logarithmic absolute spectrum on the positive frequency axis.
    """

    # check input
    if not isinstance(signal, (Signal, FrequencyData)):
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')
    _utils._check_axis_scale(freq_scale)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    if dB:
        data = dsp.decibel(signal, 'freq', log_prefix, log_reference)
        ymax = np.nanmax(data)
        ymin = ymax - 90
        ymax = ymax + 10
    else:
        data = np.abs(signal.freq)

    data, frequencies, xlabel = _utils._assert_and_match_data_to_side(
        data, signal, side)

    # prepare figure
    _, ax = _utils._prepare_plot(ax)
    ax.grid(True, 'both')
    if dB:
        ax.set_ylabel("Magnitude in dB")
        _utils._set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    else:
        ax.set_ylabel("Magnitude")
    ax.set_xlabel(xlabel)

    _utils._set_axlim(ax, ax.set_xlim, _utils._lower_frequency_limit(signal),
                      frequencies[-1], ax.get_xlim())

    # plot data
    if freq_scale == 'log':
        ax.semilogx(frequencies, data.T, **kwargs)
    else:
        ax.plot(frequencies, data.T, **kwargs)

    # set and format ticks
    if freq_scale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    return ax


def _phase(signal, deg=False, unwrap=False, freq_scale='log', ax=None,
           side='right', **kwargs):
    """Plot the phase of the spectrum on the positive frequency axis."""

    # check input
    if not isinstance(signal, (Signal, FrequencyData)):
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')
    _utils._check_axis_scale(freq_scale)

    # prepare figure
    _, ax = _utils._prepare_plot(ax)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    phase_data = dsp.phase(signal, deg=deg, unwrap=unwrap)

    phase_data, frequencies, xlabel = _utils._assert_and_match_data_to_side(
        phase_data, signal, side)

    # Construct the correct label string
    ylabel_string = _utils._phase_label(unwrap, deg)

    # y-axis formatting
    y_margin = 5 if deg else np.radians(5)
    ymin = np.nanmin(phase_data) - y_margin  # more elegant solution possible?
    ymax = np.nanmax(phase_data) + y_margin
    if unwrap == '360':
        ymin = -y_margin
        ymax = 360+y_margin if deg else 2*np.pi+y_margin

    if not deg:
        if ymax-ymin < np.pi:
            # set yrange to +/- pi/2
            ymin = np.pi/2+y_margin
            ymax = -np.pi/2-y_margin
            # major ticks at multiples of pi/4
            ax.yaxis.set_major_locator(MultipleFractionLocator(np.pi, 4))
            ax.yaxis.set_major_formatter(MultipleFractionFormatter(
                nominator=1, denominator=4, base=np.pi, base_str=r'\pi'))
            # minor ticks at multiples of pi/8
            ax.yaxis.set_minor_locator(MultipleFractionLocator(np.pi, 8))
        elif unwrap is True:
            if ymax-ymin < 20*np.pi:
                # ticks at at multiples of 2 pi / pi
                # if yrange is smaller than 20 pi
                nmaj = 2
                nmin = 1
            else:
                # major ticks at multiples of x*20*pi
                # with x determined to maximum 10 intervals
                nmaj = np.ceil((ymax-ymin)/np.pi/9/20)*20
                # minor ticks at half of major ticks,
                # but rounded to multiples of 10*pi
                nmin = np.round(nmaj/2/10)*10
            ax.yaxis.set_major_locator(
                MultipleFractionLocator(nmaj, 1, np.pi))
            ax.yaxis.set_major_formatter(
                MultipleFractionFormatter(nmaj, 1, np.pi, r'\pi'))
            ax.yaxis.set_minor_locator(
                MultipleFractionLocator(nmin/2, 1, np.pi))
        else:
            # unwrap is False or "360"
            # major ticks at multiples of pi/2
            ax.yaxis.set_major_locator(MultipleFractionLocator(np.pi, 2))
            ax.yaxis.set_major_formatter(MultipleFractionFormatter(
                nominator=1, denominator=2, base=np.pi, base_str=r'\pi'))
            # minor ticks at multiples of pi/4
            ax.yaxis.set_minor_locator(MultipleFractionLocator(np.pi, 4))

    # prepare figure
    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel_string)
    ax.grid(True, 'both')
    _utils._set_axlim(ax, ax.set_xlim, _utils._lower_frequency_limit(signal),
                      frequencies[-1], ax.get_xlim())
    _utils._set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())

    # plot data
    if freq_scale == 'log':
        ax.semilogx(frequencies, phase_data.T, **kwargs)
    else:
        ax.plot(frequencies, phase_data.T, **kwargs)

    # set and format ticks
    if freq_scale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    return ax


def _group_delay(signal, unit="s", freq_scale='log', ax=None, side='right',
                 **kwargs):
    """Plot the group delay on the positive frequency axis."""

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')
    _utils._check_time_unit(unit)
    _utils._check_axis_scale(freq_scale)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    data = dsp.group_delay(signal)
    # auto detect the unit
    if unit in [None, "auto"]:
        unit = _utils._time_auto_unit(
            np.nanmax(np.abs(data) / signal.sampling_rate))
    # set the unit
    if unit != "samples":
        factor, unit = _utils._deal_time_units(unit)
        data = data / signal.sampling_rate * factor

    data, frequencies, xlabel = _utils._assert_and_match_data_to_side(
        data, signal, side)

    # prepare figure
    _, ax = _utils._prepare_plot(ax)
    ax.set_xlabel(xlabel)

    ax.set_ylabel(f"Group delay in {unit}")
    ax.grid(True, 'both')
    _utils._set_axlim(ax, ax.set_xlim, _utils._lower_frequency_limit(signal),
                      frequencies[-1], ax.get_xlim())
    _utils._set_axlim(ax, ax.set_ylim, .5 * np.nanmin(data),
                      1.5 * np.nanmax(data), ax.get_ylim())

    # plot data
    if freq_scale == 'log':
        ax.semilogx(frequencies, data.T, **kwargs)
    else:
        ax.plot(frequencies, data.T, **kwargs)

    # set and format ticks
    if freq_scale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    return ax


def _time_freq(signal, dB_time=False, dB_freq=True, log_prefix_time=20,
               log_prefix_freq=None, log_reference=1, freq_scale='log',
               unit="s", ax=None, side='right', mode='real',
               **kwargs):
    """
    Plot the time signal and magnitude spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _utils._prepare_plot(ax, (2, 1))
    kwargs = _utils._return_default_colors_rgb(**kwargs)

    _time(signal, dB_time, log_prefix_time, log_reference, unit, ax[0],
          mode, **kwargs)
    _freq(signal, dB_freq, log_prefix_freq, log_reference, freq_scale, ax[1],
          side, **kwargs)
    fig.align_ylabels()

    return ax


def _freq_phase(signal, dB=True, log_prefix=None, log_reference=1,
                freq_scale='log', deg=False, unwrap=False, ax=None,
                side='right', **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout."""

    fig, ax = _utils._prepare_plot(ax, (2, 1))
    kwargs = _utils._return_default_colors_rgb(**kwargs)

    _freq(signal, dB, log_prefix, log_reference, freq_scale, ax[0], side,
          **kwargs)
    _phase(signal, deg, unwrap, freq_scale, ax[1], side, **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()

    return ax


def _freq_group_delay(signal, dB=True, log_prefix=None, log_reference=1,
                      unit="s", freq_scale='log', ax=None, side='right',
                      **kwargs):
    """
    Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _utils._prepare_plot(ax, (2, 1))
    kwargs = _utils._return_default_colors_rgb(**kwargs)

    _freq(signal, dB, log_prefix, log_reference, freq_scale, ax[0], side,
          **kwargs)
    _group_delay(signal, unit, freq_scale, ax[1], side, **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()

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
    fig, ax = _utils._prepare_plot(ax, subplots)

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

    return ax
