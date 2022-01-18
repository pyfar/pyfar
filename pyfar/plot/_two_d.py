import matplotlib as mpl
import numpy as np
from pyfar import Signal
import pyfar.dsp as dsp
from . import _utils
import warnings
from .ticker import (
    LogFormatterITAToolbox,
    LogLocatorITAToolbox)


def _spectrogram(signal, dB=True, log_prefix=None, log_reference=1,
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
