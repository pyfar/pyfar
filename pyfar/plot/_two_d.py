import matplotlib as mpl
import numpy as np
from pyfar import Signal, TimeData, FrequencyData
import pyfar.dsp as dsp
from . import _utils
import warnings
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox)


def _time2d(signal, dB, log_prefix, log_reference, unit, points,
            orientation, cmap, colorbar, ax, **kwargs):

    # check input and prepare the figure and axis
    fig, ax, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, TimeData), colorbar, ax, **kwargs)
    _utils._check_time_unit(unit)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    data = signal.time.T if orientation == "vertical" else signal.time
    if dB:
        if log_prefix is None:
            log_prefix = _utils._log_prefix(signal)
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

    # setup axis label and data
    if orientation == "vertical":
        ax[0].set_xlabel("Points")
        ax[0].set_ylabel(f"Time in {unit}")
        _utils._set_axlim(ax[0], ax[0].set_ylim, times[0], times[-1],
                          ax[0].get_ylim())
    else:
        ax[0].set_ylabel("Points")
        ax[0].set_xlabel(f"Time in {unit}")
        _utils._set_axlim(ax[0], ax[0].set_xlim, times[0], times[-1],
                          ax[0].get_xlim())

    if points is None:
        points = np.arange(signal.time.shape[0])

    # plot data
    points_x = points if orientation == "vertical" else times
    points_y = times if orientation == "vertical" else points
    qm = ax[0].pcolormesh(points_x, points_y, data, cmap=cmap, **kwargs)

    # color limits and colorbar
    if dB:
        qm.set_clim(ymin, ymax)

    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              "Amplitude in dB" if dB else "Amplitude")

    return ax[0], qm, cb


def _freq2d(signal, dB, log_prefix, log_reference, xscale, points, orientation,
            cmap, colorbar, ax, **kwargs):

    # check input and prepare the figure and axis
    fig, ax, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, FrequencyData), colorbar, ax, **kwargs)
    _utils._check_axis_scale(xscale)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    data = signal.freq.T if orientation == "vertical" else signal.freq
    if dB:
        if log_prefix is None:
            log_prefix = _utils._log_prefix(signal)
        eps = np.finfo(float).eps
        data = log_prefix*np.log10(np.abs(data)/log_reference + eps)
        ymax = np.nanmax(data)
        ymin = ymax - 90
        ymax = ymax + 10
    else:
        data = np.abs(data)

    # setup axis label and data
    labels = np.array(["Points", "Frequency in Hz"])
    if orientation == "horizontal":
        labels = np.roll(labels, 1)
    axlims = (ax[0].set_ylim, ax[0].get_ylim()) if orientation == "vertical" \
        else (ax[0].set_xlim, ax[0].get_xlim())

    ax[0].set_xlabel(labels[0])
    ax[0].set_ylabel(labels[1])
    _utils._set_axlim(
        ax[0], axlims[0], _utils._lower_frequency_limit(signal),
        signal.frequencies[-1], axlims[1])

    if points is None:
        points = np.arange(signal.time.shape[0])

    # plot data
    points_x = points if orientation == "vertical" else signal.frequencies
    points_y = signal.frequencies if orientation == "vertical" else points
    qm = ax[0].pcolormesh(points_x, points_y, data, cmap=cmap,
                          shading='gouraud')

    if orientation == "vertical":
        if xscale == "log":
            ax[0].yaxis.set_major_locator(LogLocatorITAToolbox())
        ax[0].set_yscale(xscale)
        ax[0].yaxis.set_major_formatter(LogFormatterITAToolbox())
    else:
        if xscale == "log":
            ax[0].xaxis.set_major_locator(LogLocatorITAToolbox())
        ax[0].set_xscale(xscale)
        ax[0].xaxis.set_major_formatter(LogFormatterITAToolbox())

    # color limits and colorbar
    if dB:
        qm.set_clim(ymin, ymax)

    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              "Magnitude in dB" if dB else "Magnitude")

    return ax[0], qm, cb


def _spectrogram(signal, dB=True, log_prefix=None, log_reference=1,
                 yscale='linear', unit=None,
                 window='hann', window_length=1024, window_overlap_fct=0.5,
                 cmap=mpl.cm.get_cmap(name='magma'), colorbar=True, ax=None,
                 **kwargs):
    """Plot the magnitude spectrum versus time.

    See pyfar.line.spectogram for more information.

    Note: this function always returns only the axis of the actual plot
    together with the quadmesh and colorbar. It does not return an array of
    axes containing also the axis of the colorbar as the public function does.
    This makes  handling interactions easier. The axis of the colorbar is added
    in pyfar.line.spectrogram.
    """

    # check input
    # check input and prepare the figure and axis
    fig, ax, kwargs = _utils._prepare_2d_plot(
        signal, (Signal), colorbar, ax, **kwargs)
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
        if log_prefix is None:
            log_prefix = _utils._log_prefix(signal)
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

    # plot the data
    qm = ax[0].pcolormesh(times, frequencies, spectrogram, cmap=cmap, **kwargs)

    # Adjust axes:
    ax[0].set_ylabel('Frequency in Hz')
    ax[0].set_xlabel(f'Time in {unit}')
    ax[0].set_xlim((times[0], times[-1]))
    ax[0].set_ylim((max(20, frequencies[1]), signal.sampling_rate/2))

    # color limits
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
    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              'Magnitude in dB' if dB else 'Magnitude')

    return ax[0], qm, cb
