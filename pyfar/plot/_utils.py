"""All private utility functions of the plot module should go here."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.tight_layout import get_subplotspec_list
from pyfar import (Signal, FrequencyData)


def _tight_layout(fig=plt.gcf()):
    """
    Apply Matplotlibs tight_layout only when it is likely to work.

    Tight layout messes up the Figure for irregular subplot layouts. The
    if-case to check if tight layout is applied was taken directly from
    Matplotlib. However, Matplotlib only raises a warning but still applies
    the tight layout.

    Parameters
    ----------
    fig : Matplotlib Figure
    """
    subplotspec_list = get_subplotspec_list(fig.get_axes())
    if None not in subplotspec_list:
        plt.tight_layout()


def _prepare_plot(ax=None, subplots=None):
    """Activates the stylesheet and returns a figure to plot on.

    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
        Axes to plot on. The default is None in which case the axes are
        obtained from the current figure. A new figure is created if it does
        not exist.
    subplots : tuple of length 2
        A tuple giving the desired subplot layout. E.g., (2, 1) creates a
        subplot layout with two rows and one column. The default is None in
        which case no layout will be enforced.

    Returns
    -------
    ax : matplotlib.pyplot.axes object
        The current axes if `subplots` is ``None`` all axes from the
        current figure as a single axis or array/list of axes otherwise.
    """
    if ax is None:
        # get current figure or create new one
        fig = plt.gcf()
        # get the current axis of all axes
        if subplots is None:
            ax = fig.gca()
        else:
            ax = fig.get_axes()
    else:
        # obtain figure from axis
        # (ax objects can be inside an array or list)
        if isinstance(ax, np.ndarray):
            fig = ax.flatten()[0].figure
        elif isinstance(ax, list):
            fig = ax[0].figure
        else:
            fig = ax.figure

    # check for correct subplot layout
    if subplots is not None:

        # check type
        if len(subplots) > 2 or not isinstance(subplots, tuple):
            raise ValueError(
                "subplots must be a tuple with one or two elements.")

        # tuple to check against the shape of current axis
        # (convert (N, 1) and (1, N) to (N, ) - this is what Matplotlib does)
        ax_subplots = subplots
        if len(ax_subplots) == 2:
            ax_subplots = tuple(s for s in ax_subplots if s != 1)

        # check if current axis has the correct numner of subplots
        create_subplots = True
        if isinstance(ax, list):
            if len(ax) == np.prod(ax_subplots):
                create_subplots = False
        elif isinstance(ax, np.ndarray):
            if ax.shape == ax_subplots:
                create_subplots = False

        # create subplots
        if create_subplots:
            fig.clf()
            ax = fig.subplots(subplots[0], subplots[1], sharex=False)

    return fig, ax


def _set_axlim(ax, setter, low, high, limits):
    """
    Set axis limits depending on existing data.

    Sets the limits of an axis to `low` and `high` if there are no lines and
    collections asociated to the axis and to `min(limits[0], low)` and
    `max(limits[1], high)` otherwise.

    Parameters
    ----------
    ax : Matplotlib Axes Object
        axis for which the limits are to be set.
    setter : function
        function for setting the limits, .e.g., `ax.set_xlim`.
    low : number
        lower axis limit
    high : number
        upper axis limit
    limits : tuple of length 2
        current axis limits, e.g., `ax.get_xlim()`.
    """

    if not ax.lines and not ax.collections:
        # set desired limit if axis does not contain any lines or points
        setter((low, high))
    else:
        # check against current axes limits
        setter((min(limits[0], low), max(limits[1], high)))


def _lower_frequency_limit(signal):
    """Return the lower frequency limit for plotting.

    pyfar frequency plots start at 20 Hz if data is availabe . If this is not
    the case, they start at the lowest available frequency.
    """
    if isinstance(signal, (Signal, FrequencyData)):
        # indices of non-zero frequencies
        idx = np.flatnonzero(signal.frequencies)
        if len(idx) == 0:
            raise ValueError(
                "Signals must have frequencies > 0 Hz for plotting.")
        # get the frequency limit
        lower_frequency_limit = max(20, signal.frequencies[idx[0]])
    else:
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')

    return lower_frequency_limit


def _return_default_colors_rgb(**kwargs):
    """Replace color in kwargs with pyfar default color if possible."""

    # pyfar default colors
    colors = {'p': '#5F4690',  # purple
              'b': '#1471B9',  # blue
              't': '#4EBEBE',  # turqois
              'g': '#078554',  # green
              'l': '#72AF47',  # light green
              'y': '#ECAD20',  # yellow
              'o': '#E07D26',  # orange
              'r': '#D83C27'}  # red

    if 'c' in kwargs and isinstance(kwargs['c'], str):
        kwargs['c'] = colors[kwargs['c']] \
            if kwargs['c'] in colors else kwargs['c']
    if 'color' in kwargs and isinstance(kwargs['color'], str):
        kwargs['color'] = colors[kwargs['color']] \
            if kwargs['color'] in colors else kwargs['color']

    return kwargs


def _default_color_dict():
    """pyfar default colors in the order matching the plotstyles"""

    colors = {'b': '#1471B9',  # blue
              'r': '#D83C27',  # red
              'y': '#ECAD20',  # yellow
              'p': '#5F4690',  # purple
              'g': '#078554',  # green
              't': '#4EBEBE',  # turqois
              'o': '#E07D26',  # orange
              'l': '#72AF47'}  # light green

    return colors


def _check_time_unit(unit):
    """Check if a valid time unit is passed."""
    units = ['s', 'ms', 'mus', 'samples']
    if unit is not None and unit not in units:
        raise ValueError(
            f"Unit is {unit} but must be {', '.join(units)}, or None.")


def _check_axis_scale(scale, axis='x'):
    """Check if a valid axis scale is passed."""
    if scale not in ['linear', 'log']:
        raise ValueError(
            f"{axis}scale is {scale} but must be 'linear', or 'log'.")


def _get_quad_mesh_from_axis(ax):
    """get the QuadMesh from an axis, if there is one.

    Parameters
    ----------
    ax : Matplotlib axes object

    Returns
    -------
    cm : Matplotlib QuadMesh object
    """
    quad_mesh_found = False
    for qm in ax.get_children():
        if type(qm) == mpl.collections.QuadMesh:
            quad_mesh_found = True
            break

    if not quad_mesh_found:
        raise ValueError("ax does not have a quad mesh.")

    return qm


def _time_auto_unit(time_max):
    """
    Automatically set the unit for time axis according to the absolute maximum
    of the input data. This is a separate function for ease of testing and for
    use across different plots.

    Parameters
    ----------

    time_max : float
        Absolute maximum of the time data in seconds
    """

    if time_max == 0:
        unit = 's'
    elif time_max < 1e-3:
        unit = 'mus'
    elif time_max < 1:
        unit = 'ms'
    else:
        unit = 's'

    return unit


def _deal_time_units(unit='s'):
    """Return scaling factor and string representation for unit multiplier
    modifications.

    Parameters
    ----------
    unit : 'str'
        The unit to be used

    Returns
    -------
    factor : float
        Factor the data is to be multiplied with, i.e. 1e-3 for milliseconds
    string : str
        String representation of the unit using LaTeX formatting.
    """
    if unit == 's':
        factor = 1
        string = 's'
    elif unit == 'ms':
        factor = 1 / 1e-3
        string = 'ms'
    elif unit == 'mus':
        factor = 1 / 1e-6
        string = r'$\mathrm{\mu}$s'
    elif unit == 'samples':
        factor = 1
        string = 'samples'
    else:
        factor = 1
        string = ''
    return factor, string


def _log_prefix(signal):
    """Return prefix for dB calculation in frequency domain depending on
    fft_norm.

    For the FFT normalizations ``'psd'`` and ``'power'`` the prefix is 10,
    for the other normalizations it is 20.

    Parameters
    ----------
    fft_norm : str
        FFT normalization
    """
    if isinstance(signal, Signal) and signal.fft_norm in ('power', 'psd'):
        log_prefix = 10
    else:
        log_prefix = 20
    return log_prefix
