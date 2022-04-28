import matplotlib.style as mpl_style
import os
import json
import contextlib
from . import _utils
from pyfar.plot._interaction import PlotParameter


def plotstyle(style='light'):
    """
    Get the fullpath of the pyfar plotstyles ``light`` or ``dark``.

    The plotstyles are defined by mplstyle files, which is Matplotlibs format
    to define styles. By default, pyfar uses the ``light`` plotstyle.

    Parameters
    ----------
    style : str
        ``light``, or ``dark``

    Returns
    -------
    style : str
        Full path to the pyfar plotstyle.

    See also
    --------
    pyfar.plot.use
    pyfar.plot.context

    """

    if style in ['light', 'dark']:
        style = os.path.join(
            os.path.dirname(__file__), 'plotstyles', f'{style}.mplstyle')

    return style


@contextlib.contextmanager
def context(style='light', after_reset=False):
    """Context manager for using plot styles temporarily.

    This context manager supports the two pyfar styles ``light`` and ``dark``.
    It is a wrapper for ``matplotlib.pyplot.style.context()``.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see                          |
        |      | ``matplotlib.style.available``.                             |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | ``matplotlib.rcParams``.                                    |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    after_reset : bool
        If ``True``, apply style after resetting settings to their defaults;
        otherwise, apply style on top of the current settings.

    See also
    --------
    pyfar.plot.plotstyle

    Examples
    --------

    Generate customizable subplots with the default pyfar plot style

    >>> import pyfar as pf
    >>> import matplotlib.pyplot as plt
    >>> with pf.plot.context():
    >>>     fig, ax = plt.subplots(2, 1)
    >>>     pf.plot.time(pf.Signal([0, 1, 0, -1], 44100), ax=ax[0])
    """

    # get pyfar plotstyle if desired
    style = plotstyle(style)

    # apply plot style
    with mpl_style.context(style):
        yield


def use(style="light"):
    """
    Use plot style settings from a style specification.

    The style name of ``default`` is reserved for reverting back to
    the default style settings. This is a wrapper for ``matplotlib.style.use``
    that supports the pyfar plot styles ``light`` and ``dark``.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see                          |
        |      | ``matplotlib.style.available``.                             |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | ``matplotlib.rcParams``.                                    |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    See also
    --------
    pyfar.plot.plotstyle

    Notes
    -----
    This updates the `rcParams` with the settings from the style. `rcParams`
    not defined in the style are kept.

    Examples
    --------

    Permanently use the pyfar default plot style

    >>> import pyfar as pf
    >>> import matplotlib.pyplot as plt
    >>> pf.plot.utils.use()
    >>> fig, ax = plt.subplots(2, 1)
    >>> pf.plot.time(pf.Signal([0, 1, 0, -1], 44100), ax=ax[0])

    """

    # get pyfar plotstyle if desired
    style = plotstyle(style)
    # use plot style
    mpl_style.use(style)


def color(color):
    """Return pyfar default color as HEX string.

    Parameters
    ----------
    color : int, str
        The colors can be specified by their index, their full name,
        or the first letter. Available colors are:

        +---+---------+-------------+
        | 1 | ``'b'`` | blue        |
        +---+---------+-------------+
        | 2 | ``'r'`` | red         |
        +---+---------+-------------+
        | 3 | ``'y'`` | yellow      |
        +---+---------+-------------+
        | 4 | ``'p'`` | purple      |
        +---+---------+-------------+
        | 5 | ``'g'`` | green       |
        +---+---------+-------------+
        | 6 | ``'t'`` | turquois    |
        +---+---------+-------------+
        | 7 | ``'o'`` | orange      |
        +---+---------+-------------+
        | 8 | ``'l'`` | light green |
        +---+---------+-------------+

    Returns
    -------
    color_hex : str
        pyfar default color as HEX string
    """
    color_dict = _utils._default_color_dict()
    colors = list(color_dict.keys())
    if isinstance(color, str):
        if color[0] not in colors:
            raise ValueError((f"color is '{color}' but must be one of the "
                              f"following {', '.join(colors)}"))
        else:
            # all colors differ by their first letter
            color_hex = color_dict[color[0]]
    elif isinstance(color, int):
        color_hex = list(color_dict.values())[color % len(colors)]
    else:
        raise ValueError("color is has to be of type str or int.")
    return color_hex


def shortcuts(show=True):
    """Show and return keyboard shortcuts for interactive figures.

    Note that shortcuts are only available if using an interactive backend in
    Matplotlib, e.g., by ``%matplotlib qt``.

    Parameters
    ----------
    show : bool, optional
        Print the keyboard shortcuts to the default console. The default is
        ``True``.

    Returns
    -------
    short_cuts : dict
        Dictionary that contains all the shortcuts.
    """  # noqa: W605 (to ignore \*)
    # Note: The end of the docstring can be generated by calling shortcuts()

    # load short cuts from json file
    sc = os.path.join(os.path.dirname(__file__), 'shortcuts', 'shortcuts.json')
    with open(sc, "r") as read_file:
        short_cuts = json.load(read_file)

    # print list of short cuts
    if show:
        # get list of plots that allow toggling axes and colormaps
        x_toggle = []
        y_toggle = []
        cm_toggle = []
        for plot in short_cuts["plots"]:
            params = PlotParameter(plot)
            if params.x_type is not None:
                if len(params.x_type) > 1:
                    x_toggle.append(plot)
            if params.y_type is not None:
                if len(params.y_type) > 1:
                    y_toggle.append(plot)
            if params.cm_type is not None:
                if len(params.cm_type) > 1:
                    cm_toggle.append(plot)

        # print information
        print("Use these shortcuts to show different plots")
        print("-------------------------------------------")
        plt = short_cuts["plots"]
        for p in plt:
            if "key_verbose" in plt[p]:
                key = plt[p]["key_verbose"]
            else:
                key = plt[p]["key"]
            print(f'{", ".join(key)}: {p}')
        print(" ")
        print(("Note that not all plots are available for TimeData and "
               "FrequencyData objects as detailed in the documentation of "
               "plots.\n\n"))

        print("Use these shortcuts to control the plot")
        print("---------------------------------------")
        ctr = short_cuts["controls"]
        for action in ctr:
            if "key_verbose" in ctr[action]:
                key = ctr[action]["key_verbose"]
            else:
                key = ctr[action]["key"]
            print(f'{", ".join(key)}: {ctr[action]["info"]}')
        print(" ")

        print("Notes on plot controls")
        print("----------------------")
        print("Moving and zooming the x and y axes is supported by all plots.")
        print(" ")
        print(("Moving and zooming the colormap is only supported by plots "
               "that have a colormap."))
        print(" ")
        print(f"Toggling the x-axis is supported by: {', '.join(x_toggle)}")
        print(" ")
        print(f"Toggling the y-axis is supported by: {', '.join(y_toggle)}")
        print(" ")
        print(f"Toggling the colormap is supported by: {', '.join(cm_toggle)}")
        print(" ")
        print(("Toggling between line and 2D plots is not supported by:"
               " spectrogram"))
        print(" ")

    return short_cuts
