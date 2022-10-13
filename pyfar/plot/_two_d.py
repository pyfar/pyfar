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


def _time_2d(signal, dB, log_prefix, log_reference, unit, indices,
             orientation, method, colorbar, ax, **kwargs):

    # check input and prepare the figure, axis, and common parameters
    fig, ax, indices, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, TimeData), 2, indices, method, ax, colorbar, **kwargs)
    _utils._check_time_unit(unit)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    if dB:
        data = dsp.decibel(signal, 'time', log_prefix, log_reference)
        ymax = np.nanmax(data) + 10
        ymin = ymax - 100
    else:
        data = signal.time
    data = data.T if orientation == "vertical" else data
    # auto detect the time unit
    if unit in [None, "auto"]:
        unit = _utils._time_auto_unit(signal.times[..., -1])
    # set the unit
    if unit == 'samples':
        times = np.arange(signal.n_samples)
    else:
        factor, unit = _utils._deal_time_units(unit)
        times = signal.times * factor

    # setup axis label and data
    axis = [ax[0].yaxis, ax[0].xaxis]
    ax_lim = [ax[0].set_ylim, ax[0].set_xlim]
    if orientation == "horizontal":
        axis = np.roll(axis, 1)
        ax_lim = np.roll(ax_lim, 1)

    axis[1].set_label_text("Indices")
    axis[0].set_label_text(f"Time in {unit}")
    ax_lim[0](times[0], times[-1])

    # color limits
    if dB and "vmin" not in kwargs:
        kwargs["vmin"] = ymin
    if dB and "vmax" not in kwargs:
        kwargs["vmax"] = ymax

    # plot data
    indices_x = indices if orientation == "vertical" else times
    indices_y = times if orientation == "vertical" else indices
    qm = _plot_2d(indices_x, indices_y, data, method, ax[0], **kwargs)

    # colorbar
    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              "Amplitude in dB" if dB else "Amplitude")

    return ax[0], qm, cb


def _freq_2d(signal, dB, log_prefix, log_reference, freq_scale, indices,
             orientation, method, colorbar, ax, **kwargs):

    # check input and prepare the figure, axis, and common parameters
    fig, ax, indices, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, FrequencyData), 2, indices, method, ax, colorbar,
        **kwargs)
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
    data = data.T if orientation == "vertical" else data
    # setup axis label and data
    axis = [ax[0].yaxis, ax[0].xaxis]
    ax_lim = [ax[0].set_ylim, ax[0].set_xlim]
    ax_scale = [ax[0].set_yscale, ax[0].set_xscale]
    if orientation == "horizontal":
        axis = np.roll(axis, 1)
        ax_lim = np.roll(ax_lim, 1)
        ax_scale = np.roll(ax_scale, 1)

    axis[1].set_label_text("Indices")
    axis[0].set_label_text("Frequency in Hz")
    ax_lim[0](_utils._lower_frequency_limit(signal), signal.frequencies[-1])

    ax_scale[0](freq_scale)
    if freq_scale == "log":
        axis[0].set_major_locator(LogLocatorITAToolbox())
    axis[0].set_major_formatter(LogFormatterITAToolbox())

    # color limits
    if dB and "vmin" not in kwargs:
        kwargs["vmin"] = ymin
    if dB and "vmax" not in kwargs:
        kwargs["vmax"] = ymax

    # plot data
    indices_x = indices if orientation == "vertical" else signal.frequencies
    indices_y = signal.frequencies if orientation == "vertical" else indices
    qm = _plot_2d(indices_x, indices_y, data, method, ax[0], **kwargs)

    # colorbar
    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              "Magnitude in dB" if dB else "Magnitude")

    return ax[0], qm, cb


def _phase_2d(signal, deg, unwrap, freq_scale, indices, orientation, method,
              colorbar, ax, **kwargs):

    # check input and prepare the figure, axis, and common parameters
    fig, ax, indices, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, FrequencyData), 2, indices, method, ax, colorbar,
        **kwargs)
    _utils._check_axis_scale(freq_scale)

    # prepare input
    data = dsp.phase(signal, deg=deg, unwrap=unwrap)
    data = data.T if orientation == "vertical" else data

    # setup axis label and data
    axis = [ax[0].yaxis, ax[0].xaxis]
    ax_lim = [ax[0].set_ylim, ax[0].set_xlim]
    ax_scale = [ax[0].set_yscale, ax[0].set_xscale]
    if orientation == "horizontal":
        axis = np.roll(axis, 1)
        ax_lim = np.roll(ax_lim, 1)
        ax_scale = np.roll(ax_scale, 1)

    axis[1].set_label_text("Indices")
    axis[0].set_label_text("Frequency in Hz")
    ax_lim[0](_utils._lower_frequency_limit(signal), signal.frequencies[-1])

    # color limits
    phase_margin = 5 if deg else np.radians(5)
    if "vmin" not in kwargs:
        kwargs["vmin"] = np.nanmin(data) - phase_margin
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.nanmax(data) + phase_margin

    # plot data
    indices_x = indices if orientation == "vertical" else signal.frequencies
    indices_y = signal.frequencies if orientation == "vertical" else indices
    qm = _plot_2d(indices_x, indices_y, data, method, ax[0], **kwargs)

    ax_scale[0](freq_scale)
    if freq_scale == "log":
        axis[0].set_major_locator(LogLocatorITAToolbox())
    axis[0].set_major_formatter(LogFormatterITAToolbox())

    # colorbar
    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              _utils._phase_label(unwrap, deg))

    if colorbar and not deg and (not unwrap or unwrap == "360"):
        # nice tick formatting is not done for unwrap=True. In this case
        # it can create 1000 or more ticks.
        cb.locator = MultipleFractionLocator(np.pi, 2)
        cb.formatter = MultipleFractionFormatter(
            nominator=1, denominator=2, base=np.pi, base_str=r'\pi')
        cb.update_ticks()

    return ax[0], qm, cb


def _group_delay_2d(signal, unit, freq_scale, indices, orientation, method,
                    colorbar, ax, **kwargs):

    # check input and prepare the figure, axis, and common parameters
    fig, ax, indices, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, ), 2, indices, method, ax, colorbar, **kwargs)
    _utils._check_axis_scale(freq_scale)

    # prepare input
    kwargs = _utils._return_default_colors_rgb(**kwargs)
    data = dsp.group_delay(signal)
    data = np.reshape(data, signal.freq.shape)
    data = data.T if orientation == "vertical" else data
    # auto detect the unit
    if unit is None:
        unit = _utils._time_auto_unit(
            np.nanmax(np.abs(data) / signal.sampling_rate))
    # set the unit
    if unit != "samples":
        factor, unit = _utils._deal_time_units(unit)
        data = data / signal.sampling_rate * factor

    # setup axis label and data
    axis = [ax[0].yaxis, ax[0].xaxis]
    ax_lim = [ax[0].set_ylim, ax[0].set_xlim]
    ax_scale = [ax[0].set_yscale, ax[0].set_xscale]
    if orientation == "horizontal":
        axis = np.roll(axis, 1)
        ax_lim = np.roll(ax_lim, 1)
        ax_scale = np.roll(ax_scale, 1)

    axis[1].set_label_text("Indices")
    axis[0].set_label_text("Frequency in Hz")
    ax_lim[0](_utils._lower_frequency_limit(signal), signal.frequencies[-1])

    ax_scale[0](freq_scale)
    if freq_scale == "log":
        axis[0].set_major_locator(LogLocatorITAToolbox())
    axis[0].set_major_formatter(LogFormatterITAToolbox())

    # color limits
    if "vmin" not in kwargs:
        kwargs["vmin"] = .5 * np.nanmin(data)
    if "vmax" not in kwargs:
        kwargs["vmax"] = 1.5 * np.nanmax(data)

    # plot data
    indices_x = indices if orientation == "vertical" else signal.frequencies
    indices_y = signal.frequencies if orientation == "vertical" else indices
    qm = _plot_2d(indices_x, indices_y, data, method, ax[0], **kwargs)

    # colorbar
    cb = _utils._add_colorbar(colorbar, fig, ax, qm, f"Group delay in {unit}")

    return ax[0], qm, cb


def _time_freq_2d(signal, dB_time, dB_freq, log_prefix_time, log_prefix_freq,
                  log_reference, freq_scale, unit, indices, orientation,
                  method, colorbar, ax, **kwargs):
    """
    Plot the time signal and magnitude spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _utils._prepare_plot(ax, (2, 1))

    _, qm_0, cb_0 = _time_2d(
        signal, dB_time, log_prefix_time, log_reference, unit, indices,
        orientation, method, colorbar, ax[0], **kwargs)
    _, qm_1, cb_1 = _freq_2d(
        signal, dB_freq, log_prefix_freq, log_reference, freq_scale, indices,
        orientation, method, colorbar, ax[1], **kwargs)
    fig.align_ylabels()

    return ax, [qm_0, qm_1], [cb_0, cb_1]


def _freq_phase_2d(signal, dB, log_prefix, log_reference, freq_scale, deg,
                   unwrap, indices, orientation, method, colorbar, ax,
                   **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout."""

    fig, ax = _utils._prepare_plot(ax, (2, 1))

    _, qm_0, cb_0 = _freq_2d(signal, dB, log_prefix, log_reference, freq_scale,
                             indices, orientation, method, colorbar,
                             ax[0], **kwargs)
    _, qm_1, cb_1 = _phase_2d(signal, deg, unwrap, freq_scale, indices,
                              orientation, method, colorbar, ax[1],
                              **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()

    return ax, [qm_0, qm_1], [cb_0, cb_1]


def _freq_group_delay_2d(
        signal, dB, log_prefix, log_reference, unit, freq_scale, indices,
        orientation, method, colorbar, ax, **kwargs):
    """
    Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.
    """

    fig, ax = _utils._prepare_plot(ax, (2, 1))

    _, qm_0, cb_0 = _freq_2d(signal, dB, log_prefix, log_reference, freq_scale,
                             indices, orientation, method, colorbar,
                             ax[0], **kwargs)
    _, qm_1, cb_1 = _group_delay_2d(
        signal, unit, freq_scale, indices, orientation, method,
        colorbar, ax[1], **kwargs)
    ax[0].set_xlabel(None)
    fig.align_ylabels()

    return ax, [qm_0, qm_1], [cb_0, cb_1]


def _spectrogram(signal, dB=True, log_prefix=None, log_reference=1,
                 freq_scale='linear', unit="s", window='hann',
                 window_length=1024, window_overlap_fct=0.5,
                 colorbar=True, ax=None, **kwargs):
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
    fig, ax, _, kwargs = _utils._prepare_2d_plot(
        signal, (Signal, ), 1, None, 'pcolormesh', ax, colorbar, **kwargs)
    _utils._check_time_unit(unit)
    _utils._check_axis_scale(freq_scale, 'y')

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
    if unit in [None, "auto"]:
        unit = _utils._time_auto_unit(times[..., -1])
    # set the unit
    if unit == 'samples':
        times *= signal.sampling_rate
    else:
        factor, unit = _utils._deal_time_units(unit)
        times = times * factor

    # plot the data
    qm = ax[0].pcolormesh(times, frequencies, spectrogram, **kwargs)

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
    if freq_scale == 'log':
        ax[0].set_yscale('symlog')
        ax[0].yaxis.set_major_locator(LogLocatorITAToolbox())
    ax[0].yaxis.set_major_formatter(LogFormatterITAToolbox())
    ax[0].grid(ls='dotted', color='white')

    # colorbar
    cb = _utils._add_colorbar(colorbar, fig, ax, qm,
                              'Magnitude in dB' if dB else 'Magnitude')

    return ax[0], qm, cb


def _plot_2d(x, y, data, method, ax, **kwargs):
    # Choose method and plot
    if method == 'contourf':
        # adjust levels according to vmin and vmax
        if "vmin" in kwargs and "vmax" in kwargs:
            if "levels" not in kwargs:
                # Default: 11 levels
                kwargs["levels"] = np.linspace(
                    kwargs["vmin"], kwargs["vmax"], 11)
            elif isinstance(kwargs["levels"], int):
                kwargs["levels"] = np.linspace(
                    kwargs["vmin"], kwargs["vmax"], kwargs["levels"])
        qm = ax.contourf(x, y, data, **kwargs)
    else:
        qm = ax.pcolormesh(x, y, data, **kwargs)
    return qm
