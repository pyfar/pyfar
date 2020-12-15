import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pyfar.plot._line as _line


def plotstyle(style='light'):
    """
    Get name and fullpath of plotstyle.

    Can be used to plot the pyfar plotstyles 'light' and 'dark' saved as
    mplstyle-file inside the pyfar package.

    Parameters
    ----------
    style : str
        'light', 'dark', or stlye from matplotlib.pyplot.available. Raises a
        ValueError if style is not known.

    Returns
    -------
    style : str
        Full path to the pyfar plotstyle or name of the matplotlib plotstyle

    """

    if style is None:
        # get the currently used plotstyle
        style = mpl.matplotlib_fname()
    elif style in ['light', 'dark']:
        # use pyfar style
        style = os.path.join(
            os.path.dirname(__file__), 'plotstyles', f'{style}.mplstyle')
    elif style not in plt.style.available:
        # error if style not found
        ValueError((f"plotstyle '{style}' not available. Valid styles are "
                    "None, 'light', 'dark' and styles from "
                    "matplotlib.pyplot.available"))

    return style


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
    Matplotlib, e.g., by typing `%matplotlib qt`.

    Parameters
    ----------
    show : bool, optional
        print the keyboard shortcuts to the default console. The default is
        True.

    Returns
    -------
    short_cuts : dict
        dictionary that contains all the shortcuts

    """
    short_cuts = {
        # not yet implemented as intended
        "plots": {
            "line.time": {
                "key": "1",
                "info": "line.time"},
            "line.freq": {
                "key": "2",
                "info": "line.freq"},
            "line.phase": {
                "key": "3",
                "info": "line.phase"},
            "line.group_delay": {
                "key": "4",
                "info": "line.group_delay"},
            "line.spectrogram": {
                "key": "5",
                "info": "line.spectrogram"},
            "line.time_freq": {
                "key": "6",
                "info": "line.time_freq"},
            "line.freq_phase": {
                "key": "7",
                "info": "line.freq_phase"},
            "line.freq_group_delay": {
                "key": "8",
                "info": "line.freq_group_delay"}
        },
        "controls": {
            "move_left": {
                "key": "left",
                "info": "move x-axis view to the left"},
            "move_right": {
                "key": "right",
                "info": "move x-axis view to the right"},
            "move_up": {
                "key": "up",
                "info": "move y-axis view upwards"},
            "move_down": {
                "key": "down",
                "info": "y-axis view downwards"},
            "move_cm_up": {
                "key": "+",
                "info": "move colormap range up"},
            "move_cm_down": {
                "key": "-",
                "info": "move colormap range down"},
            "zoom_x_in": {
                "key": "shift+right",
                "info": "zoom in x-axis"},
            "zoom_x_out": {
                "key": "shift+left",
                "info": "zoom out x-axis"},
            "zoom_y_in": {
                "key": "shift+up",
                "info": "zoom out y-axis"},
            "zoom_y_out": {
                "key": "shift+down",
                "info": "zoom in y-axis"},
            "zoom_cm_in": {
                "key": "*",
                "info": "zoom colormap range in"},
            "zoom_cm_out": {
                "key": "_",
                "info": "zoom colormap range out"},
            "toggle_x": {
                "key": "X",
                "info": "toggle between linear and logarithmic x-axis"},
            "toggle_y": {
                "key": "Y",
                "info": "toggle between linear and logarithmic y-axis"},
            "toggle_cm": {
                "key": "C",
                "info": "toggle between linear and logarithmic color data"},
            "toogle_all": {
                "key": "a",
                "info": ("toggle between plotting all channels and plottinng "
                         "single channels")},
            "next": {
                "key": ".",
                "info": "show next channel"},
            "prev": {
                "key": ",",
                "info": "show previous channel"}
        }
    }

    if show:
        print("Use these shortcuts to show different plots:")
        # not yet implemented as intended
        for plot in short_cuts["plots"]:
            print(f'{plot}: {short_cuts["plots"][plot]}')
        print(" ")
        print("Use these shortcuts to control the plot:")
        ctr = short_cuts["controls"]
        for action in ctr:
            print(f'{ctr[action]["key"]}: {ctr[action]["info"]}')

    return short_cuts
