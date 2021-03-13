import matplotlib.style as mpl_style
import os
import json
import contextlib
import pyfar.plot._line as _line
from pyfar.plot._interaction import PlotParameter


def plotstyle(style='light'):
    """
    Get fullpath of pyfar plotstyle 'light' or 'dark'.

    Can be used to plot the pyfar plotstyles 'light' and 'dark' saved as
    mplstyle-file inside the pyfar package.

    Parameters
    ----------
    style : str
        'light', or 'dark'

    Returns
    -------
    style : str
        Full path to the pyfar plotstyle. Input parameter style otherwise.

    """

    if style in ['light', 'dark']:
        style = os.path.join(
            os.path.dirname(__file__), 'plotstyles', f'{style}.mplstyle')

    return style


@contextlib.contextmanager
def context(style='light', after_reset=False):
    """Context manager for using plot styles temporarily.

    This context manager supports the two pyfar styles 'light' and 'dark'. It
    is a wrapper for `matplotlib.pyplot.style.context()`.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `style.available`.       |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    after_reset : bool
        If True, apply style after resetting settings to their defaults;
        otherwise, apply style on top of the current settings.

    Examples
    --------
    >>> import pyfar
    >>> import matplotlib.pyplot as plt
    >>>
    >>> with pyfar.plot.utils.context():
    >>>     fig, ax = plt.subplots(2, 1)
    >>>     pyfar.plot.time(pyfar.Signal([0, 1, 0, -1], 44100), ax=ax[0])
    """

    # get pyfar plotstyle if desired
    style = plotstyle(style)

    # apply plot style
    with mpl_style.context(style):
        yield


def use(style="light"):
    """
    Use plot style settings from a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings. This is a wrapper for `matplotlib.style.use`
    that supports the pyfar plot styles 'light' and 'dark'.

    .. note::

       This updates the `.rcParams` with the settings from the style.
       `.rcParams` not defined in the style are kept.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `style.available`.       |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    Examples
    --------
    >>> import pyfar
    >>> import matplotlib.pyplot as plt
    >>>
    >>> pyfar.plot.utils.use()
    >>> fig, ax = plt.subplots(2, 1)
    >>> pyfar.plot.time(pyfar.Signal([0, 1, 0, -1], 44100), ax=ax[0])

    """

    # get pyfar plotstyle if desired
    style = plotstyle(style)
    # use plot style
    mpl_style.use(style)


def color(color: str) -> str:
    """Return pyfar default color as HEX string.

    Parameters
    ----------
    color : str
        'p' - purple
        'b' - blue
        't' - turqois
        'g' - green
        'l' - light green
        'y' - yellow
        'o' - orange
        'r' - red

    Returns
    -------
    color : str
        pyfar default color as HEX string

    """
    colors = ['p', 'b', 't', 'g', 'l', 'y', 'o', 'r']
    if color not in colors:
        raise ValueError((f"color is '{color}' but must be one of the "
                          f"following {', '.join(colors)}"))

    kwargs = {'c': color}
    kwargs = _line._return_default_colors_rgb(**kwargs)

    color = kwargs['c']
    return color


def shortcuts(show=True):
    """Show and return keyboard shortcuts for interactive figures.

    Note that shortcuts are only available if using an interactive backend in
    Matplotlib, e.g., by typing `%matplotlib qt`. Shortcuts can be customized
    by edition 'shortcuts/shortcuts.json'. See below for the default shortcuts.

    Parameters
    ----------
    show : bool, optional
        print the keyboard shortcuts to the default console. The default is
        True.

    Returns
    -------
    short_cuts : dict
        dictionary that contains all the shortcuts

    Use these shortcuts to show different plots
    -------------------------------------------
    1: line.time
    2: line.freq
    3: line.phase
    4: line.group_delay
    5: line.spectrogram
    6: line.time_freq
    7: line.freq_phase
    8: line.freq_group_delay

    Use these shortcuts to control the plot
    ---------------------------------------
    left: move x-axis view to the left
    right: move x-axis view to the right
    up: move y-axis view upwards
    down: y-axis view downwards
    +: move colormap range up
    -: move colormap range down
    shift+right: zoom in x-axis
    shift+left: zoom out x-axis
    shift+up: zoom out y-axis
    shift+down: zoom in y-axis
    *: zoom colormap range in
    _: zoom colormap range out
    x: toggle between linear and logarithmic x-axis
    y: toggle between linear and logarithmic y-axis
    c: toggle between linear and logarithmic color data
    a: toggle between plotting all channels and plottinng single channels
    .: show next channel
    ,: show previous channel

    Notes on plot controls
    ----------------------
    Moving and zooming the x and y axes is supported by all plots.

    Moving and zooming the colormap is only supported by plots that have a
    colormap.

    Toggling the x-axis is supported by: line.time, line.freq, line.phase,
    line.group_delay, line.spectrogram, line.time_freq, line.freq_phase,
    line.freq_group_delay

    Toggling the y-axis is supported by: line.time, line.freq, line.phase,
    line.group_delay, line.spectrogram, line.time_freq, line.freq_phase,
    line.freq_group_delay

    Toggling the colormap is supported by: line.spectrogram

    """
    # Note: The end of the docstring can be generated by calling shortcuts()

    # load short cuts from json file
    sc = os.path.join(os.path.dirname(__file__), 'shortcuts', 'shortcuts.json')
    with open(sc, "r") as read_file:
        short_cuts = json.load(read_file)

    # print list of short cuts
    if show:
        # get list of plots that allow toogling axes and colormaps
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

    return short_cuts
