import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolBase, ToolToggleBase

from haiopy import Signal


def plot_time_signal(signal, xlim=None, ylim=None):
    """Docstring.

    Parameters
    ----------
    signal : Signal object

    Returns
    -------
    """
    # if not isinstance(signal, Signal):
    #     raise TypeError("Expected Signal object, not {}".format(type(signal).__name__))
    # n_channels = signal.shape[0]
    x_data = signal.times
    y_data = signal.time.T
    
    def on_m_click(event):
        if event.key == 'm':
            print ("event triggered")

    fig, axes = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_m_click)
    
    axes.plot(x_data, y_data)
        
    axes.set_title("Signal")
    axes.set_xlabel("Time [sec]")
    axes.set_ylabel("Amplitude")
    axes.grid(True)

    fig.canvas.manager.toolmanager.add_tool('ChannelToggle', ToggleChannels, axes=axes, active_line=0)
    plt.show()

    return 

def plot_freq_signal(signal, xmin=20, xmax=20000, ymin=None, ymax=None):
    """Docstring.

    Parameters
    ----------
    signal : Signal object

    Returns
    -------
    """
    # if not isinstance(signal, Signal):
    #     raise TypeError("Expected Signal object, not {}".format(type(signal).__name__))
    time_data = signal.time[-1]
    samplingrate = signal.samplingrate

    fig, axes = plt.subplots()

    axes.set_title("Magnitude Spectrum")
    axes.magnitude_spectrum(time_data, Fs=samplingrate, scale='dB', color='C1')
    axes.set_xscale('log')
    axes.set_xlim(xmin, xmax)
    axes.grid(True)

    plt.show()


class ToggleChannels(ToolToggleBase):
    """Toggles between all channels and single channel"""
    default_keymap = 'z'
    description = "Show all/single channel"
    default_toggle = True
    
    def __init__(self, *args, axes, active_line, **kwargs):
        self.axes = axes
        self.active_line = active_line
        super().__init__(*args, **kwargs)
        
    def enable(self, *args):
        self.set_allchannels_visibility(True)

    def disable(self, *args):
        self.set_allchannels_visibility(False)
    
    def set_allchannels_visibility(self, state):
        if state == False:
            for i in range(len(self.axes.lines)):
                self.axes.lines[i].set_visible(True)         
        else:
            for i in range(len(self.axes.lines)):
                self.axes.lines[i].set_visible(False)
            self.axes.lines[self.active_line].set_visible(True)
        self.figure.canvas.draw()                
    