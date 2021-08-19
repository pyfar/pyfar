import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfar.plot.utils import context
from . import _line_2d
# from . import _interaction as ia


def time2d(signal, dB=False, log_prefix=20, log_reference=1, unit=None,
           points=None, sort_points="ascending", orientation="vertical",
           cmap=mpl.cm.get_cmap(name='magma'), ax=None, style='light',
           **kwargs):
    """
    Surface plot of the time signal.

    Plots ``signal.time`` and passes keyword arguments (`kwargs`) to
    ``matplotlib.pyplot.pcolormesh()``.

    Parameters
    ----------

    Parameters
    ----------
    signal : Signal, TimeData
        The input data to be plotted.
    dB : bool
        Indicate if the data should be plotted in dB in which case
        ``log_prefix * np.log10(signal.time / log_reference)`` is used. The
        default is ``False``.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is
        ``20``.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is
        ``1``.
    unit : str, None
        Unit of the time axis. Can be ``s``, ``ms``, ``mus``, or ``samples``.
        The default is ``None``, which sets the unit to ``s`` (seconds), ``ms``
        (milli seconds), or ``mus`` (micro seconds) depending on the data.
    points: array like, optional
        Points at which the channels of `signal` were sampled. `points` must
        have as many entries as `signal` has channels. Examples for points
        might be azimuth angles, if `signal` holds data on the horizontal plane
        or x-values, if `signal` was sampled along a line in Cartesian
        coordinates. `points` is used to for labeling the axis. The default is
        ``'None'`` which labels the channels in `signal` from 0 to N.
    sort_points: string, optional
        ``'ascending'``
            sort the channels in `signal` so that `points` are ascending.
        ``'descending'``
            sort the channels in `signal` so that `points` are descending.
        ``'none'``
            do not sort the channels in `signal`

        The default is ``'ascending'``
    orientation: string, optional
        ``'vertical'``
            The channels of `signal` will be plotted as as vertical lines.
        ``'horizontal'``
            The channels of `signal` will be plotted as horizontal lines.

        The default is ``'vertical'``
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.


    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Examples
    --------

    Plot a multichannel signal with information about the points where it was
    samples

    .. plot::
        >>> import pyfar as pf
        >>> import numpy as np
        >>> # generate the signal
        >>> angles = np.arange(0, 180)
        >>> delays = np.round(100 * np.sin(angles / 180 * np.pi)).astype(int)
        >>> signal = pf.signals.impulse(128, delays)
        >>> # plot the signal
        >>> pf.plot.time2d(signal, points=angles)
    """

    with context(style):
        ax = _line_2d._time2d(
            signal.flatten(), dB, log_prefix, log_reference, unit,
            points, sort_points, orientation,
            cmap, ax, **kwargs)
    plt.tight_layout()

    # plot_parameter = ia.PlotParameter(
    #     'time', dB_time=dB, log_prefix=log_prefix,
    #     log_reference=log_reference)
    # interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    # ax.interaction = interaction
