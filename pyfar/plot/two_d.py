import matplotlib as mpl
from pyfar.plot.utils import context
from .. import Signal
from . import (_two_d, _utils)
from . import _interaction as ia


def time2d(signal, dB=False, log_prefix=None, log_reference=1, unit=None,
           points=None, orientation="vertical",
           cmap=mpl.cm.get_cmap(name='magma'), colorbar=True, ax=None,
           style='light', **kwargs):
    """
    2D plot of multi-channel time signals with color coded amplitude.

    Plots ``signal.time`` and passes keyword arguments (`kwargs`) to
    ``matplotlib.pyplot.pcolormesh()``.

    Parameters
    ----------
    signal : Signal, TimeData
        The input data to be plotted. `signal.cshape` must be `(m, )` with
        $m>0$.
    dB : bool
        Indicate if the data should be plotted in dB in which case
        ``log_prefix * np.log10(signal.time / log_reference)`` is used. The
        default is ``False``.
    log_prefix : integer, float
        Prefix for calculating the logarithmic frequency data. The default is
        ``None``, so ``10`` is chosen if ``signal.fft_norm`` is ``'power'`` or
        ``'psd'`` and ``20`` otherwise.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is
        ``1``.
    unit : str, None
        Unit of the time axis. Can be ``s``, ``ms``, ``mus``, or ``samples``.
        The default is ``None``, which sets the unit to ``s`` (seconds), ``ms``
        (milli seconds), or ``mus`` (micro seconds) depending on the data.
    points: array like, optional
        Points at which the channels of `signal` were sampled (e.g. azimuth
        angles or x values). `points` must be monotonously increasing/
        decreasing and have as many entries as `signal` has channels. The
        default is ``'None'`` which labels the channels in `signal` from
        0 to N.
    orientation: string, optional
        ``'vertical'``
            The channels of `signal` will be plotted as as vertical lines.
        ``'horizontal'``
            The channels of `signal` will be plotted as horizontal lines.

        The default is ``'vertical'``
    colorbar : bool, optional
        Control the colorbar. The default is ``True``, which adds a colorbar
        to the plot. ``False`` omits the colorbar.
    ax : matplotlib.pyplot.axes
        Axes to plot on.

        ``None``
            Use the current axis, or create a new axis (and figure) if there is
            none.
        ``ax``
            If a single axis is passed, this is used for plotting. If
            `colorbar` is ``True`` the space for the colorbar is taken from
            this axis.
        ``[ax, ax]``
            If a list or array of two axes is passed, the first is used to plot
            the data and the second to plot the colorbar. In this case
            `colorbar` must be ``True``

        The default is ``None``.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to
        ``matplotlib.pyplot.pcolormesh()``.


    Returns
    -------
    ax : matplotlib.pyplot.axes
        If `colorbar` is ``True`` an array of two axes is returned. The first
        is the axis on which the data is plotted, the second is the axis of the
        colorbar. If `colorbar` is ``False``, only the axis on which the data
        is plotted is returned
    quad_mesh : QuadMesh
        The Matplotlib quad mesh collection. This can be used to manipulate the
        way the data is displayed, e.g., by limiting the range of the colormap
        by ``quad_mesh.set_clim()``. It can also be used to generate a colorbar
        by ``cb = fig.colorbar(qm, ...)``.
    colorbar : Colorbar
        The Matplotlib colorbar object if `colorbar` is ``True`` and ``None``
        otherwise. This can be used to control the appearance of the colorbar,
        e.g., the label can be set by ``colorbar.set_label()``.

    Examples
    --------

    Plot a multichannel signal with information about the points where it was
    samples

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> # generate the signal
        >>> angles = np.arange(0, 180) / 180 * np.pi
        >>> delays = np.round(100 * np.sin(angles)).astype(int)
        >>> amplitudes = .5 + .5 * np.abs(np.cos(angles))
        >>> signal = pf.signals.impulse(128, delays, amplitudes)
        >>> # plot the signal
        >>> pf.plot.time2d(signal, points=angles)
    """

    with context(style):
        ax, qm, cb = _two_d._time2d(
            signal, dB, log_prefix, log_reference, unit,
            points, orientation, cmap, colorbar, ax, **kwargs)
    _utils._tight_layout()

    plot_parameter = ia.PlotParameter(
        'time2d', dB_time=dB, log_prefix_time=log_prefix,
        log_reference=log_reference, unit=unit, points=points,
        orientation=orientation, cmap=cmap, colorbar=colorbar)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    if colorbar:
        ax = [ax, cb.ax]

    return ax, qm, cb


def freq2d(signal, dB=True, log_prefix=None, log_reference=1, xscale='log',
           points=None, orientation="vertical",
           cmap=mpl.cm.get_cmap(name='magma'), colorbar=True, ax=None,
           style='light', **kwargs):
    """
    Plot the magnitude spectrum.

    Plots ``abs(signal.freq)`` and passes keyword arguments (`kwargs`) to
    ``matplotlib.pyplot.plot()``.

    Parameters
    ----------
    signal : Signal, FrequencyData
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    dB : bool
        Indicate if the data should be plotted in dB in which case
        ``log_prefix * np.log10(abs(signal.freq) / log_reference)`` is used.
        The default is ``True``.
    log_prefix : integer, float
        Prefix for calculating the logarithmic frequency data. The default is
        ``None``, so ``10`` is chosen if ``signal.fft_norm`` is ``'power'`` or
        ``'psd'`` and ``20`` otherwise.
    log_reference : integer, float
        Reference for calculating the logarithmic frequency data. The default
        is ``1``.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    points: array like, optional
        Points at which the channels of `signal` were sampled (e.g. azimuth
        angles or x values). `points` must be monotonously increasing/
        decreasing and have as many entries as `signal` has channels. The
        default is ``'None'`` which labels the channels in `signal` from
        0 to N.
    orientation: string, optional
        ``'vertical'``
            The channels of `signal` will be plotted as as vertical lines.
        ``'horizontal'``
            The channels of `signal` will be plotted as horizontal lines.

        The default is ``'vertical'``
    colorbar : bool, optional
        Control the colorbar. The default is ``True``, which adds a colorbar
        to the plot. ``False`` omits the colorbar.
    ax : matplotlib.pyplot.axes
        Axes to plot on.

        ``None``
            Use the current axis, or create a new axis (and figure) if there is
            none.
        ``ax``
            If a single axis is passed, this is used for plotting. If
            `colorbar` is ``True`` the space for the colorbar is taken from
            this axis.
        ``[ax, ax]``
            If a list or array of two axes is passed, the first is used to plot
            the data and the second to plot the colorbar. In this case
            `colorbar` must be ``True``

        The default is ``None``.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to
        ``matplotlib.pyplot.pcolormesh()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        If `colorbar` is ``True`` an array of two axes is returned. The first
        is the axis on which the data is plotted, the second is the axis of the
        colorbar. If `colorbar` is ``False``, only the axis on which the data
        is plotted is returned
    quad_mesh : QuadMesh
        The Matplotlib quad mesh collection. This can be used to manipulate the
        way the data is displayed, e.g., by limiting the range of the colormap
        by ``quad_mesh.set_clim()``. It can also be used to generate a colorbar
        by ``cb = fig.colorbar(qm, ...)``.
    colorbar : Colorbar
        The Matplotlib colorbar object if `colorbar` is ``True`` and ``None``
        otherwise. This can be used to control the appearance of the colorbar,
        e.g., the label can be set by ``colorbar.set_label()``.

    Example
    -------

    .. plot::

        >>> import pyfar as pf
        >>> sine = pf.signals.sine(100, 4410)
        >>> pf.plot.freq(sine)
    """

    with context(style):
        ax, qm, cb = _two_d._freq2d(
            signal, dB, log_prefix, log_reference, xscale, points,
            orientation, cmap, colorbar, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'freq2d', dB_freq=dB, log_prefix_freq=log_prefix,
        log_reference=log_reference, xscale=xscale, points=points,
        orientation=orientation, cmap=cmap, colorbar=colorbar)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    if colorbar:
        ax = [ax, cb.ax]

    return ax, qm, cb


def spectrogram(signal, dB=True, log_prefix=None, log_reference=1,
                yscale='linear', unit=None, window='hann', window_length=1024,
                window_overlap_fct=0.5, cmap=mpl.cm.get_cmap(name='magma'),
                colorbar=True, ax=None, style='light', **kwargs):
    """Plot blocks of the magnitude spectrum versus time.

    Parameters
    ----------
    signal : Signal
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    dB : bool
        Indicate if the data should be plotted in dB in which case
        ``log_prefix * np.log10(abs(signal.freq) / log_reference)`` is used.
        The default is ``True``.
    log_prefix : integer, float
        Prefix for calculating the logarithmic frequency data. The default is
        ``None``, so ``10`` is chosen if ``signal.fft_norm`` is ``'power'`` or
        ``'psd'`` and ``20`` otherwise.
    log_reference : integer
        Reference for calculating the logarithmic frequency data. The default
        is ``1``.
    yscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``linear``.
    unit : str, None
        Unit of the time axis. Can be ``s``, ``ms``, ``mus``, or ``samples``.
        The default is ``None``, which sets the unit to ``s`` (seconds), ``ms``
        (milli seconds), or ``mus`` (micro seconds) depending on the data.
    window : str
        Specifies the window that is applied to each block of the time data
        before applying the Fourier transform. The default is ``hann``. See
        ``scipy.signal.get_window`` for a list of possible windows.
    window_length : integer
        Specifies the window/block length in samples. The default is ``1024``.
    window_overlap_fct : double
        Ratio of points to overlap between blocks [0...1]. The default is
        ``0.5``, which would result in 512 samples overlap for a window length
        of 1024 samples.
    cmap : matplotlib.colors.Colormap(name, N=256)
        Colormap for spectrogram. Defaults to matplotlibs ``magma`` colormap.
    colorbar : bool, optional
        Control the colorbar. The default is ``True``, which adds a colorbar
        to the plot. ``False`` omits the colorbar.
    ax : matplotlib.pyplot.axes
        Axes to plot on.

        ``None``
            Use the current axis, or create a new axis (and figure) if there is
            none.
        ``ax``
            If a single axis is passed, this is used for plotting. If
            `colorbar` is ``True`` the space for the colorbar is taken from
            this axis.
        ``[ax, ax]``
            If a list or array of two axes is passed, the first is used to plot
            the data and the second to plot the colorbar. In this case
            `colorbar` must be ``True``

        The default is ``None``.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to
        ``matplotlib.pyplot.pcolormesh()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        If `colorbar` is ``True`` an array of two axes is returned. The first
        is the axis on which the data is plotted, the second is the axis of the
        colorbar. If `colorbar` is ``False``, only the axis on which the data
        is plotted is returned
    quad_mesh : QuadMesh
        The Matplotlib quad mesh collection. This can be used to manipulate the
        way the data is displayed, e.g., by limiting the range of the colormap
        by ``quad_mesh.set_clim()``. It can also be used to generate a colorbar
        by ``cb = fig.colorbar(qm, ...)``.
    colorbar : Colorbar
        The Matplotlib colorbar object if `colorbar` is ``True`` and ``None``
        otherwise. This can be used to control the appearance of the colorbar,
        e.g., the label can be set by ``colorbar.set_label()``.

    Example
    -------

    .. plot::

        >>> import pyfar as pf
        >>> sweep = pf.signals.linear_sweep_time(2**14, [0, 22050])
        >>> pf.plot.spectrogram(sweep)
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    with context(style):
        ax, qm, cb = _two_d._spectrogram(
            signal.flatten(), dB, log_prefix, log_reference, yscale, unit,
            window, window_length, window_overlap_fct,
            cmap, colorbar, ax)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'spectrogram', dB_freq=dB, log_prefix_freq=log_prefix,
        log_reference=log_reference, yscale=yscale, unit=unit, window=window,
        window_length=window_length, window_overlap_fct=window_overlap_fct,
        cmap=cmap)
    interaction = ia.Interaction(signal, ax, style, plot_parameter)
    ax.interaction = interaction

    if colorbar:
        ax = [ax, cb.ax]

    return ax, qm, cb
