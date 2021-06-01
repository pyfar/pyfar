def time2d():
    """
    Surface plot of the time signal.

    Parameters
    ----------
    .
    . (all parameters from plot.time)
    .
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


    Returns
    -------
    .
    . same as plot.line
    .

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
    pass
