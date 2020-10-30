import matplotlib as mpl
import matplotlib.pyplot as plt
import os


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
        # use haiopy style
        style = os.path.join(
            os.path.dirname(__file__), 'plotstyles', f'{style}.mplstyle')
    elif style not in plt.style.available:
        # error if style not found
        ValueError((f"plotstyle '{style}' not available. Valid styles are "
                    "None, 'light', 'dark' and styles from "
                    "matplotlib.pyplot.available"))

    return style


# def shortcuts(show=True):
#     """Show and return keyboard shortcuts for interactive figures.

#     Note that shortcuts are only available if using an interactive backend in
#     Matplotlib, e.g., by typing `%matplotlib qt`.

#     Parameters
#     ----------
#     show : bool, optional
#         print the keyboard shortcuts to the default console. The default is
#         True.

#     Returns
#     -------
#     short_cuts : dict
#         dictionary that contains all the shortcuts

#     """
#     short_cuts = {
#         "plots": {
#             "ctr+1": "time signal",
#             "ctr+2": "magnitude response",
#             "ctr+3": "phase response",
#             "ctr+4": "group delay",
#             "ctr+5": "spectrogram",
#             "ctr+6": "time, magnitude, phase, and group delay"
#         },
#         "controls": {
#             "up": "move y-axis view upwards",
#             "down": "move y-axis view downwards",
#             "left": "move x-axis view to the left",
#             "right": "move x-axis view to the right",
#             "shift+up": "zoom out y-axis",
#             "shift+down": "zoom in y-axis",
#             "shift+right": "zoom in x-axis",
#             "shift+left": "zoom out x-axis",
#             "shift+y": "toogle logarithmic y-axis",
#             "shift+x": "toogle logarithmic x-axis",
#             "c": "toogle legend with channel numbers",
#             "a": "toogle show all channels",
#             ".": "show next channel",
#             ",": "show previous chanel"
#         }
#     }

#     if show:
#         print("Use these shortcuts to show different plots:")
#         for plot in short_cuts["plots"]:
#             print(f'{plot}: {short_cuts["plots"][plot]}')
#         print(" ")
#         print("Use these shortcuts to control the plot look:")
#         for ctr in short_cuts["controls"]:
#             print(f'{ctr}: {short_cuts["controls"][ctr]}')
