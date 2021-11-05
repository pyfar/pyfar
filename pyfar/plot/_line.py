import matplotlib as mpl
import numpy as np
from pyfar import Signal, TimeData, FrequencyData
import pyfar.dsp as dsp
from . import _utils
import warnings
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox,
    MultipleFractionLocator,
    MultipleFractionFormatter)


def _time(signal, dB=False, log_prefix=20, log_reference=1, unit=None,
          ax=None, **kwargs):
    """Plot the time data of a signal."""

    # check input
    if not isinstance(signal, (Signal, TimeData)):
        raise TypeError('Input data has to be of type: Signal or TimeData.')
    _utils._check_time_unit(unit)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    data = signal.time.T
    if dB:
        # avoid any zero-values because they result in -inf in dB data
        eps = np.finfo(float).eps
        data = log_prefix * np.log10(np.abs(data) / log_reference + eps)
        ymax = np.nanmax(data) + 10
        ymin = ymax - 100

    # auto detect the time unit
    if unit is None:
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
        ax.set_ylabel("Amplitude in dB")
        _utils._set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    else:
        ax.set_ylabel("Amplitude")
    _utils._set_axlim(ax, ax.set_xlim, times[0], times[-1],
                      ax.get_xlim())

    # plot data
    ax.plot(times, data, **kwargs)

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
    _utils._check_axis_scale(xscale)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    if dB:
        eps = np.finfo(float).eps
        data = log_prefix*np.log10(np.abs(signal.freq)/log_reference + eps)
        ymax = np.nanmax(data)
        ymin = ymax - 90
        ymax = ymax + 10
    else:
        data = np.abs(signal.freq)

    # prepare figure
    _, ax = _utils._prepare_plot(ax)
    ax.grid(True, 'both')
    if dB:
        ax.set_ylabel("Magnitude in dB")
        _utils._set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())
    else:
        ax.set_ylabel("Magnitude")
    ax.set_xlabel("Frequency in Hz")
    _utils._set_axlim(ax, ax.set_xlim, _utils._lower_frequency_limit(signal),
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

    return ax


def _phase(signal, deg=False, unwrap=False, xscale='log', ax=None, **kwargs):
    """Plot the phase of the spectrum on the positive frequency axis."""

    # check input
    if not isinstance(signal, (Signal, FrequencyData)):
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')
    _utils._check_axis_scale(xscale)

    # prepare figure
    _, ax = _utils._prepare_plot(ax)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
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
    _utils._set_axlim(ax, ax.set_xlim, _utils._lower_frequency_limit(signal),
                      signal.frequencies[-1], ax.get_xlim())
    _utils._set_axlim(ax, ax.set_ylim, ymin, ymax, ax.get_ylim())

    # plot data
    if xscale == 'log':
        ax.semilogx(signal.frequencies, phase_data.T, **kwargs)
    else:
        ax.plot(signal.frequencies, phase_data.T, **kwargs)

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
    _utils._check_time_unit(unit)
    _utils._check_axis_scale(xscale)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    data = dsp.group_delay(signal)
    # auto detect the unit
    if unit is None:
        unit = _utils._time_auto_unit(
            np.nanmax(np.abs(data) / signal.sampling_rate))
    # set the unit
    if unit != "samples":
        factor, unit = _utils._deal_time_units(unit)
        data = data / signal.sampling_rate * factor

    # prepare figure
    _, ax = _utils._prepare_plot(ax)
    ax.set_xlabel("Frequency in Hz")
    ax.set_ylabel(f"Group delay in {unit}")
    ax.grid(True, 'both')
    _utils._set_axlim(ax, ax.set_xlim, _utils._lower_frequency_limit(signal),
                      signal.frequencies[-1], ax.get_xlim())
    _utils._set_axlim(ax, ax.set_ylim, .5 * np.nanmin(data),
                      1.5 * np.nanmax(data), ax.get_ylim())

    # plot data
    if xscale == 'log':
        ax.semilogx(signal.frequencies, data.T, **kwargs)
    else:
        ax.plot(signal.frequencies, data.T, **kwargs)

    # set and format ticks
    if xscale == 'log':
        ax.xaxis.set_major_locator(LogLocatorITAToolbox())
    ax.xaxis.set_major_formatter(LogFormatterITAToolbox())

    return ax


def _spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                 yscale='linear', unit=None,
                 window='hann', window_length=1024, window_overlap_fct=0.5,
                 cmap=mpl.cm.get_cmap(name='magma'), colorbar=True, ax=None):
    """Plot the magnitude spectrum versus time.

    See pyfar.line.spectogram for more information.

    Note: this function always returns only the axis of the actual plot
    together with the quadmesh and colorbar. It does not return an array of
    axes containing also the axis of the colorbar as the public function does.
    This makes  handling interactions easier. The axis of the colorbar is added
    in pyfar.line.spectrogram.
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')
    if not colorbar and isinstance(ax, (tuple, list, np.ndarray)):
        raise ValueError('A list of axes can not be used if colorbar is False')
    _utils._check_time_unit(unit)
    _utils._check_axis_scale(yscale, 'y')

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
        unit = _utils._time_auto_unit(times[..., -1])
    # set the unit
    if unit == 'samples':
        times *= signal.sampling_rate
    else:
        factor, unit = _utils._deal_time_units(unit)
        times = times * factor

    # prepare the figure and axis for plotting the data and colorbar
    fig, ax = _utils._prepare_plot(ax)
    if not isinstance(ax, (np.ndarray, list)):
        ax = [ax, None]

    # plot the data
    ax[0].pcolormesh(times, frequencies, spectrogram, cmap=cmap,
                     shading='gouraud')

    # Adjust axes:
    ax[0].set_ylabel('Frequency in Hz')
    ax[0].set_xlabel(f'Time in {unit}')
    ax[0].set_xlim((times[0], times[-1]))
    ax[0].set_ylim((max(20, frequencies[1]), signal.sampling_rate/2))

    # color limits
    qm = _utils._get_quad_mesh_from_axis(ax[0])

    if dB:
        ymax = np.nanmax(spectrogram)
        ymin = ymax - 90
        ymax = ymax + 10
        qm.set_clim(ymin, ymax)

    # scales and ticks
    if yscale == 'log':
        ax[0].set_yscale('symlog')
        ax[0].yaxis.set_major_locator(LogLocatorITAToolbox())
    ax[0].yaxis.set_major_formatter(LogFormatterITAToolbox())
    ax[0].grid(ls='dotted', color='white')

    # colorbar
    if colorbar:
        if ax[1] is None:
            cb = fig.colorbar(qm, ax=ax[0])
        else:
            cb = fig.colorbar(qm, cax=ax[1])
        cb.set_label('Magnitude in dB' if dB else 'Magnitude')
    else:
        cb = None

    return ax[0], qm, cb


def _time_freq(signal, dB_time=False, dB_freq=True, log_prefix=20,
               log_reference=1, xscale='log', unit=None, ax=None, **kwargs):
    """
    Plot the time signal and magnitude spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _utils._prepare_plot(ax, (2, 1))
    kwargs = _utils._return_default_colors_rgb(**kwargs)

    _time(signal, dB_time, log_prefix, log_reference, unit, ax[0], **kwargs)
    _freq(signal, dB_freq, log_prefix, log_reference, xscale, ax[1], **kwargs)
    fig.align_ylabels()

    return ax


def _freq_phase(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
                deg=False, unwrap=False, ax=None, **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout."""

    fig, ax = _utils._prepare_plot(ax, (2, 1))
    kwargs = _utils._return_default_colors_rgb(**kwargs)

    _freq(signal, dB, log_prefix, log_reference, xscale, ax[0], **kwargs)
    _phase(signal, deg, unwrap, xscale, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()

    return ax


def _freq_group_delay(signal, dB=True, log_prefix=20, log_reference=1,
                      unit=None, xscale='log', ax=None, **kwargs):
    """
    Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _utils._prepare_plot(ax, (2, 1))
    kwargs = _utils._return_default_colors_rgb(**kwargs)

    _freq(signal, dB, log_prefix, log_reference, xscale, ax[0], **kwargs)
    _group_delay(signal, unit, xscale, ax[1], **kwargs)
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
