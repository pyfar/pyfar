import matplotlib as mpl
from pyfar.plot.utils import context
from .. import Signal
from . import (_line, _utils)
from . import _interaction as ia


def time(signal, dB=False, log_prefix=20, log_reference=1, unit=None, ax=None,
         style='light', **kwargs):
    """Plot the time signal.

    Plots ``signal.time`` and passes keyword arguments (`kwargs`) to
    ``matplotlib.pyplot.plot()``.

    Parameters
    ----------
    signal : Signal, TimeData
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
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
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Examples
    --------

    .. plot::

        >>> import pyfar as pf
        >>> sine = pf.signals.sine(100, 4410)
        >>> pf.plot.time(sine)

    """

    with context(style):
        ax = _line._time(signal.flatten(), dB, log_prefix, log_reference, unit,
                         ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'time', dB_time=dB, log_prefix=log_prefix,
        log_reference=log_reference)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def freq(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
         ax=None, style='light', **kwargs):
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
        ``20``.
    log_reference : integer, float
        Reference for calculating the logarithmic frequency data. The default
        is ``1``.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Example
    -------

    .. plot::

        >>> import pyfar as pf
        >>> sine = pf.signals.sine(100, 4410)
        >>> pf.plot.freq(sine)
    """

    with context(style):
        ax = _line._freq(signal.flatten(), dB, log_prefix, log_reference,
                         xscale, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'freq', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, xscale=xscale)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def phase(signal, deg=False, unwrap=False, xscale='log', ax=None,
          style='light', **kwargs):
    """Plot the phase of the spectrum.

    Plots ``angle(signal.freq)`` and passes keyword arguments (`kwargs`) to
    ``matplotlib.pyplot.plot()``.

    Parameters
    ----------
    signal : Signal, FrequencyData
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    deg : bool
        Plot the phase in degrees. The default is ``False``, which plots the
        phase in radians.
    unwrap : bool, str
        True to unwrap the phase or "360" to unwrap the phase to 2 pi. The
        default is ``False``, which plots the wrapped phase.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Example
    -------

    .. plot::

        >>> import pyfar as pf
        >>> impulse = pf.signals.impulse(100, 10)
        >>> pf.plot.phase(impulse, unwrap=True)
    """

    with context(style):
        ax = _line._phase(signal.flatten(), deg, unwrap, xscale, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'phase', deg=deg, unwrap=unwrap, xscale=xscale)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def group_delay(signal, unit=None, xscale='log', ax=None, style='light',
                **kwargs):
    """Plot the group delay.

    Passes keyword arguments (`kwargs`) to ``matplotlib.pyplot.plot()``.

    Parameters
    ----------
    signal : Signal
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    unit : str, None
        Unit of the group delay. Can be ``s``, ``ms``, ``mus``, or ``samples``.
        The default is ``None``, which sets the unit to ``s`` (seconds), ``ms``
        (milli seconds), or ``mus`` (micro seconds) depending on the data.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Examples
    --------

    .. plot::

        >>> import pyfar as pf
        >>> impulse = pf.signals.impulse(100, 10)
        >>> pf.plot.group_delay(impulse, unit='samples')
    """

    with context(style):
        ax = _line._group_delay(signal.flatten(), unit, xscale, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'group_delay', unit=unit, xscale=xscale)
    interaction = ia.Interaction(signal, ax, style, plot_parameter, **kwargs)
    ax.interaction = interaction

    return ax


def spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                yscale='linear', unit=None, window='hann', window_length=1024,
                window_overlap_fct=0.5, cmap=mpl.cm.get_cmap(name='magma'),
                colorbar=True, ax=None, style='light'):
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
        ``20``.
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
        ax, qm, cb = _line._spectrogram(
            signal.flatten(), dB, log_prefix, log_reference, yscale, unit,
            window, window_length, window_overlap_fct,
            cmap, colorbar, ax)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'spectrogram', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, yscale=yscale, unit=unit, window=window,
        window_length=window_length, window_overlap_fct=window_overlap_fct,
        cmap=cmap)
    interaction = ia.Interaction(signal, ax, style, plot_parameter)
    ax.interaction = interaction

    if colorbar:
        ax = [ax, cb.ax]

    return ax, qm, cb


def time_freq(signal, dB_time=False, dB_freq=True, log_prefix=20,
              log_reference=1, xscale='log', unit=None,
              ax=None, style='light', **kwargs):
    """
    Plot the time signal and magnitude spectrum in a 2 by 1 subplot layout.


    Parameters
    ----------
    signal : Signal
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    dB_time : bool
        Indicate if the data should be plotted in dB in which case
        ``log_prefix * np.log10(signal.time / log_reference)`` is used. The
        default is ``False``.
    dB_freq : bool
        Indicate if the data should be plotted in dB in which case
        ``log_prefix * np.log10(abs(signal.freq) / log_reference)`` is used.
        The default is ``True``.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time/frequency data.
        The default is ``20``.
    log_reference : integer
        Reference for calculating the logarithmic time/frequency data.
        The default is ``1``.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    unit : str
        Unit of the time axis. Can be ``s``, ``ms``, ``mus``, or ``samples``.
        The default is ``None``, which sets the unit to ``s`` (seconds), ``ms``
        (milli seconds), or ``mus`` (micro seconds) depending on the data.
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Examples
    --------

    .. plot::

        >>> import pyfar as pf
        >>> sine = pf.signals.sine(100, 4410)
        >>> pf.plot.time_freq(sine)
    """

    with context(style):
        ax = _line._time_freq(signal.flatten(), dB_time, dB_freq, log_prefix,
                              log_reference, xscale, unit, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'time', dB_time=dB_time, log_prefix=log_prefix,
        log_reference=log_reference)
    interaction = ia.Interaction(
        signal, ax[0], style, plot_parameter, **kwargs)
    ax[0].interaction = interaction

    return ax


def freq_phase(signal, dB=True, log_prefix=20, log_reference=1, xscale='log',
               deg=False, unwrap=False, ax=None, style='light', **kwargs):
    """Plot the magnitude and phase spectrum in a 2 by 1 subplot layout.

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
        ``20``.
    log_reference : integer
        Reference for calculating the logarithmic frequency data. The default
        is ``1``.
    deg : bool
        Flag to plot the phase in degrees. The default is ``False``.
    unwrap : bool, str
        True to unwrap the phase or "360" to unwrap the phase to 2 pi. The
        default is ``False``.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current figure
        ore creates a new one if no figure exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are forwarded to matplotlib.pyplot.plot

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    See Also
    --------
    matplotlib.pyplot.plot() for possible **kwargs.
    """

    with context(style):
        ax = _line._freq_phase(signal.flatten(), dB, log_prefix, log_reference,
                               xscale, deg, unwrap, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'freq', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, xscale=xscale)
    interaction = ia.Interaction(
        signal, ax[0], style, plot_parameter, **kwargs)
    ax[0].interaction = interaction

    return ax


def freq_group_delay(signal, dB=True, log_prefix=20, log_reference=1,
                     unit=None, xscale='log', ax=None, style='light',
                     **kwargs):
    """Plot the magnitude and group delay spectrum in a 2 by 1 subplot layout.

    Passes keyword arguments (`kwargs`) to ``matplotlib.pyplot.plot()``.

    Parameters
    ----------
    signal : Signal, FrequencyData
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    dB : bool
        Flag to plot the logarithmic magnitude spectrum. The default is
        ``True``.
    log_prefix : integer, float
        Prefix for calculating the logarithmic frequency data. The default is
        ``20``.
    log_reference : integer
        Reference for calculating the logarithmic frequency data. The default
        is ``1``.
    unit : str
        Unit of the group delay. Can be ``s``, ``ms``, ``mus``, or ``samples``.
        The default is ``None``, which sets the unit to ``s`` (seconds), ``ms``
        (milli seconds), or ``mus`` (micro seconds) depending on the data.
    xscale : str
        ``linear`` or ``log`` to plot on a linear or logarithmic frequency
        axis. The default is ``log``.
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        Axes or array of axes containing the plot.

    Examples
    --------

    .. plot::

        >>> import pyfar as pf
        >>> impulse = pf.signals.impulse(100, 10)
        >>> pf.plot.freq_group_delay(impulse, unit='samples')
    """

    with context(style):
        ax = _line._freq_group_delay(
            signal.flatten(), dB, log_prefix, log_reference,
            unit, xscale, ax, **kwargs)
    _utils._tight_layout()

    # manage interaction
    plot_parameter = ia.PlotParameter(
        'freq', dB_freq=dB, log_prefix=log_prefix,
        log_reference=log_reference, xscale=xscale)
    interaction = ia.Interaction(
        signal, ax[0], style, plot_parameter, **kwargs)
    ax[0].interaction = interaction

    return ax


def custom_subplots(signal, plots, ax=None, style='light', **kwargs):
    """
    Plot multiple pyfar plots with a custom layout and default parameters.

    The plots are passed as a list of :py:mod:`pyfar.plot` function handles.
    The subplot layout is taken from the shape of that list
    (see example below).

    Parameters
    ----------
    signal : Signal
        The input data to be plotted. Multidimensional data are flattened for
        plotting, e.g, a signal of ``signal.cshape = (2, 2)`` would be plotted
        in the order ``(0, 0)``, ``(0, 1)``, ``(1, 0)``, ``(1, 1)``.
    plots : list, nested list
        Function handles for plotting.
    ax : matplotlib.pyplot.axes
        Axes to plot on. The default is ``None``, which uses the current axis
        or creates a new figure if none exists.
    style : str
        ``light`` or ``dark`` to use the pyfar plot styles or a plot style from
        ``matplotlib.style.available``. The default is ``light``.
    **kwargs
        Keyword arguments that are passed to ``matplotlib.pyplot.plot()``.

    Returns
    -------
    ax : matplotlib.pyplot.axes
        List of axes handles

    Examples
    --------

    Generate a two by two subplot layout

    .. plot::

        >>> import pyfar as pf
        >>> impulse = pf.signals.impulse(100, 10)
        >>> plots = [[pf.plot.time, pf.plot.phase],
        ...          [pf.plot.freq, pf.plot.group_delay]]
        >>> pf.plot.custom_subplots(impulse, plots)

    """

    with context(style):
        ax = _line._custom_subplots(signal.flatten(), plots, ax, **kwargs)
    _utils._tight_layout()

    return ax
