import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfar import Signal, TimeData
from pyfar.plot._line import (
    _check_time_unit, _return_default_colors_rgb, _time_auto_unit,
    _deal_time_units, _prepare_plot, _set_axlim, _get_quad_mesh_from_axis)


def _time2d(signal, dB, log_prefix, log_reference, unit, points,
            orientation, cmap, ax, **kwargs):

    # check input
    if not isinstance(signal, (Signal, TimeData)):
        raise TypeError('Input data has to be of type: Signal or TimeData.')
    _check_time_unit(unit)

    # prepare input
    kwargs = _return_default_colors_rgb(**kwargs)
    data = signal.time.T if orientation == "vertical" else signal.time
    if dB:
        # avoid any zero-values because they result in -inf in dB data
        eps = np.finfo(float).eps
        data = log_prefix * np.log10(np.abs(data) / log_reference + eps)
        ymax = np.nanmax(data) + 10
        ymin = ymax - 100

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
    fig, ax = _prepare_plot(ax)
    # clear figure and axis - spectogram does not work with hold
    fig.clf()
    ax = plt.gca()
    # plot the data
    ax = ax.figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
    fig.axes[0].remove()

    if orientation == "vertical":
        ax[0].set_xlabel("Points")
        ax[0].set_ylabel(f"Time in {unit}")
        _set_axlim(ax[0], ax[0].set_ylim, times[0], times[-1],
               ax[0].get_ylim())
    else:
        ax[0].set_ylabel("Points")
        ax[0].set_xlabel(f"Time in {unit}")
        _set_axlim(ax[0], ax[0].set_xlim, times[0], times[-1],
               ax[0].get_xlim())

    if points is None:
        points = range(signal.time.shape[0])

    # plot data
    points_x = points if orientation == "vertical" else times
    points_y = times if orientation == "vertical" else points
    ax[0].pcolormesh(points_x, points_y, data, cmap=cmap,
                  shading='gouraud')

    # color limits
    if dB:
        for PCM in ax[0].get_children():
            if type(PCM) == mpl.collections.QuadMesh:
                break
        PCM.set_clim(ymin, ymax)

    # Colorbar:
    qm = _get_quad_mesh_from_axis(ax[0])

    cb = plt.colorbar(qm, cax=ax[1])
    cb_label = "Amplitude in dB" if dB else "Amplitude"
    cb.set_label(cb_label)

    return ax
