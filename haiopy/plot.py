from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import numpy as np

from haiopy import Signal

plt.rcParams['toolbar'] = 'toolmanager'


def plot_time(signal, **kwargs):
    """Plot the time signal of a haiopy audio signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    axes :  Axes object or array of Axes objects.
    
    See Also
    --------
    matplotlib.pyplot.plot() : Plot y versus x as lines and/or markers
    
    Examples
    --------
    """
    # if not isinstance(signal, Signal):
    #     raise TypeError("Expected Signal object, not {}".format(type(signal).__name__))
    # n_channels = signal.shape[0]
    x_data = signal.times
    y_data = signal.time.T

    def on_m_click(event):
        if event.key == 'm':
            print("event triggered")

    fig, axes = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_m_click)

    axes.plot(x_data, y_data)

    axes.set_title("Signal")
    axes.set_xlabel("Time [sec]")
    axes.set_ylabel("Amplitude")
    axes.grid(True)

    fig.canvas.manager.toolmanager.add_tool(
        'ChannelCycle', CycleChannels, axes=axes)
    fig.canvas.manager.toolmanager.add_tool(
        'ChannelToggle', ToggleChannels, axes=axes)
    fig.canvas.manager.toolmanager.add_tool(
        'DarkMode', ToggleDarkMode, axes=axes)

    plt.show()

    return axes


def plot_freq(signal, xmin=20, xmax=20000, ymin=None, ymax=None):
    """Plot the absolute values of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal object
        An adio signal object from the haiopy signal class
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    axes : Axes object or array of Axes objects.
    
    See Also
    --------
    matplotlib.pyplot.magnitude_spectrum() : Plot the magnitudes of the corresponding frequencies.
    
    Examples
    --------
    """
    # if not isinstance(signal, Signal):
    #     raise TypeError("Expected Signal object, not {}".format(type(signal).__name__))
    n_channels = signal.shape[0]
    time_data = signal.time
    samplingrate = signal.samplingrate

    fig, axes = plt.subplots()

    axes.set_title("Magnitude Spectrum")
    for i in range(n_channels):
        axes.magnitude_spectrum(time_data[i], Fs=samplingrate, scale='dB')
    axes.set_xscale('log')
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(-90, 10)
    axes.grid(True)

    fig.canvas.manager.toolmanager.add_tool(
        'ChannelCycle', CycleChannels, axes=axes)
    fig.canvas.manager.toolmanager.add_tool(
        'ChannelToggle', ToggleChannels, axes=axes)
    fig.canvas.manager.toolmanager.add_tool(
        'DarkMode', ToggleDarkMode, axes=axes)

    plt.show()

    return axes


class CycleChannels(ToolBase):
    """Cycle throgh the channels.
    
    This class adds custom functionalities to the matplotlib UI
    
    Reference
    ---------
    `Backend Managers <https://matplotlib.org/api/backend_managers_api.html>`_
    """
    default_keymap = '*'
    description = "Cycle through the channels"

    def __init__(self, *args, axes, **kwargs):
        self.axes = axes
        self.line_cycle = cycle(self.axes.lines)
        self.current_line = next(self.line_cycle)

    def trigger(self, *args, **kwargs):
        for i in range(len(self.axes.lines)):
            self.axes.lines[i].set_visible(False)
        self.current_line.set_visible(True)
        self.current_line = next(self.line_cycle)
        self.figure.canvas.draw()


class ToggleChannels(ToolToggleBase):
    """Toggles between all channels and single channel.
    
    This class adds custom functionalities to the matplotlib UI
    
    Reference
    ---------
    `Backend Managers <https://matplotlib.org/api/backend_managers_api.html>`_
    """
    default_keymap = 'a'
    description = "Show all/single channel"
    default_toggle = True

    def __init__(self, *args, axes, **kwargs):
        self.axes = axes
        self.current_line = self.axes.lines[0]
        super().__init__(*args, **kwargs)

    def enable(self, *args):
        self.set_allchannels_visibility(True)

    def disable(self, *args):
        self.set_allchannels_visibility(False)

    def set_allchannels_visibility(self, state):
        is_visible = []
        for vis_iter in range(len(self.axes.lines)):
            is_visible.append(self.axes.lines[vis_iter].get_visible())
        if not all(is_visible):
            self.current_line = self.axes.lines[is_visible.index(True)]

        if state is False:
            for i in range(len(self.axes.lines)):
                self.axes.lines[i].set_visible(True)
        else:
            for i in range(len(self.axes.lines)):
                self.axes.lines[i].set_visible(False)
            self.current_line.set_visible(True)
        self.figure.canvas.draw()


class ToggleDarkMode(ToolToggleBase):
    """Toggles the dark mode of the plot.
    
    This class adds custom functionalities to the matplotlib UI
    
    Reference
    ---------
    `Backend Managers <https://matplotlib.org/api/backend_managers_api.html>`_
    """
    default_keymap = 'b'
    description = "Change to black/white background"
    default_toggle = False

    def __init__(self, *args, axes, **kwargs):
        self.axes = axes
        super().__init__(*args, **kwargs)

    def enable(self, *args):
        self.set_darkmode(True)

    def disable(self, *args):
        self.set_darkmode(False)

    def set_darkmode(self, state):
        if state is True:
            self.figure.patch.set_facecolor('k')
            self.axes.set_facecolor('k')
            self.axes.spines['bottom'].set_color('w')
            self.axes.spines['top'].set_color('w')
            self.axes.spines['right'].set_color('w')
            self.axes.spines['left'].set_color('w')
            self.axes.tick_params(axis='x', colors='w')
            self.axes.tick_params(axis='y', colors='w')
            self.axes.yaxis.label.set_color('w')
            self.axes.xaxis.label.set_color('w')
            self.axes.title.set_color('w')
        else:
            self.figure.patch.set_facecolor('w')
            self.axes.set_facecolor('w')
            self.axes.spines['bottom'].set_color('k')
            self.axes.spines['top'].set_color('k')
            self.axes.spines['right'].set_color('k')
            self.axes.spines['left'].set_color('k')
            self.axes.tick_params(axis='x', colors='k')
            self.axes.tick_params(axis='y', colors='k')
            self.axes.yaxis.label.set_color('k')
            self.axes.xaxis.label.set_color('k')
            self.axes.title.set_color('k')
        self.figure.canvas.draw()
