from itertools import cycle
import matplotlib.pyplot as plt

from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.ticker import FixedLocator, FixedFormatter

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QLineEdit, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator

import numpy as np

from haiopy import Signal

plt.rcParams['toolbar'] = 'toolmanager'


class FractionalOctaveFormatter(FixedFormatter):
    def __init__(self, n_fractions=1):
        if n_fractions == 1:
            ticks = ['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
        elif n_fractions == 3:
            ticks = ['12.5', '16', '20',
                     '25', '31.5', '40',
                     '50', '63', '80',
                     '100', '125', '160',
                     '200', '250', '315',
                     '400', '500', '630',
                     '800', '1k', '1.25k',
                     '1.6k', '2k', '2.5k',
                     '3.15k', '4k', '5k',
                     '6.3k', '8k', '10k',
                     '12.5k', '16k', '20k']
        else:
            raise ValueError("Unsupported number of fractions.")
        super().__init__(ticks)


class FractionalOctaveLocator(FixedLocator):
    def __init__(self, n_fractions=1):
        if n_fractions == 1:
            ticks = [16, 31.5, 63, 125, 250, 500, 1e3, 2e3, 4e3, 8e3, 16e3]
        elif n_fractions == 3:
            ticks = [12.5, 16, 20,
                     25, 31.5, 40,
                     50, 63, 80,
                     100, 125, 160,
                     200, 250, 315,
                     400, 500, 630,
                     800, 1e3, 1250,
                     1600, 2e3, 2500,
                     3150, 4e3, 5e3,
                     6300, 8e3, 10e3,
                     12.5e3, 16e3, 20e3]
        else:
            raise ValueError("Unsupported number of fractions.")
        super().__init__(ticks)


def plot_time(signal, **kwargs):
    """Plot the time signal of a haiopy audio signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class
    **kwargs
        Arbitrary keyword arguments.
        Use 'xmin', 'xmax', 'ymin', 'ymax' to set axis limitations.

    Returns
    -------
    axes :  Axes object or array of Axes objects.

    See Also
    --------
    matplotlib.pyplot.plot() : Plot y versus x as lines and/or markers

    Examples
    --------
    """

    x_data = signal.times[0]
    y_data = signal.time.T

    fig, axes = plt.subplots()

    axes.plot(x_data, y_data)

    axes.set_title("Signal")
    axes.set_xlabel("Time [sec]")
    axes.set_ylabel("Amplitude")
    axes.grid(True)

    if "xmin" in kwargs:
        if isinstance(kwargs.get("xmin"), (int, float)):
            axes.set_xlim(left=kwargs.get("xmin"))
        else:
            raise TypeError("Expected int/double")
    if "xmax" in kwargs:
        if isinstance(kwargs.get("xmax"), (int, float)):
            axes.set_xlim(right=kwargs.get("xmax"))
        else:
            raise TypeError("Expected int/double")
    if "ymin" in kwargs:
        if isinstance(kwargs.get("ymin"), (int, float)):
            axes.set_ylim(bottom=kwargs.get("ymin"))
        else:
            raise TypeError("Expected int/double")
    if "ymax" in kwargs:
        if isinstance(kwargs.get("ymax"), (int, float)):
            axes.set_ylim(top=kwargs.get("ymax"))
        else:
            raise TypeError("Expected int/double")

    if 'Qt' in plt.get_backend():
        fig.canvas.manager.toolmanager.add_tool(
            'ChannelCycle', CycleChannels, axes=axes)
        fig.canvas.manager.toolmanager.add_tool(
            'ChannelToggle', ToggleChannels, axes=axes)
        fig.canvas.manager.toolmanager.add_tool(
            'DarkMode', ToggleDarkMode, axes=axes)
        fig.canvas.manager.toolmanager.add_tool(
                'AxisUpdate', UpdateAxis, axes=axes)

    plt.show()

    return axes


def plot_freq(signal, **kwargs):
    """Plot the absolute values of the spectrum on the positive frequency axis.

    Parameters
    ----------
    signal : Signal object
        An adio signal object from the haiopy signal class
    **kwargs
        Arbitrary keyword arguments.
        Use 'xmin', 'xmax', 'ymin', 'ymax' to set axis limitations.

    Returns
    -------
    axes : Axes object or array of Axes objects.

    See Also
    --------
    matplotlib.pyplot.magnitude_spectrum() : Plot the magnitudes of the corresponding frequencies.

    Examples
    --------
    """

    n_channels = signal.shape[0]
    time_data = signal.time
    samplingrate = signal.samplingrate

    fig, axes = plt.subplots()

    axes.set_title("Magnitude Spectrum")

    for i in range(n_channels):
        spectrum, freq, line = axes.magnitude_spectrum(time_data[i], Fs=samplingrate, scale='dB')

    axes.set_xscale('log')
    axes.grid(True)

    spectrum_db = 20 * np.log10(spectrum + np.finfo(float).tiny)
    ymax = np.max(spectrum_db)
    ymin = ymax - 90
    ymax = ymax + 10

    if "xmin" in kwargs:
        if isinstance(kwargs.get("xmin"), (int, float)):
            axes.set_xlim(left=kwargs.get("xmin"))
        else:
            raise TypeError("Expected int/double")
    else:
        axes.set_xlim(left=20)
    if "xmax" in kwargs:
        if isinstance(kwargs.get("xmax"), (int, float)):
            axes.set_xlim(right=kwargs.get("xmax"))
        else:
            raise TypeError("Expected int/double")
    else:
        axes.set_xlim(right=20000)
    if "ymin" in kwargs:
        if isinstance(kwargs.get("ymin"), (int, float)):
            axes.set_ylim(bottom=kwargs.get("ymin"))
        else:
            raise TypeError("Expected int/double")
    else:
        axes.set_ylim(bottom=ymin)
    if "ymax" in kwargs:
        if isinstance(kwargs.get("ymax"), (int, float)):
            axes.set_ylim(top=kwargs.get("ymax"))
        else:
            raise TypeError("Expected int/double")
    else:
        axes.set_ylim(top=ymax)

    if 'Qt' in plt.get_backend():
        fig.canvas.manager.toolmanager.add_tool(
            'ChannelCycle', CycleChannels, axes=axes)
        fig.canvas.manager.toolmanager.add_tool(
            'ChannelToggle', ToggleChannels, axes=axes)
        fig.canvas.manager.toolmanager.add_tool(
            'DarkMode', ToggleDarkMode, axes=axes)
        fig.canvas.manager.toolmanager.add_tool(
                'AxisUpdate', UpdateAxis, axes=axes)

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


class UpdateAxis(ToolBase):
    """Open the AxisDialog to update the axis limits.

    This class adds custom functionalities to the matplotlib UI

    Reference
    ---------
    `Backend Managers <https://matplotlib.org/api/backend_managers_api.html>`_
    """
    default_keymap = 'd'
    description = "Update Axis GUI"

    def __init__(self, *args, axes, **kwargs):
        self.axes = axes
        self.line_cycle = cycle(self.axes.lines)
        self.current_line = next(self.line_cycle)

    def trigger(self, *args, **kwargs):
        xlim_update, ylim_update, result = AxisDialog.update_axis(axes=self.axes)

        self.axes.set_xlim(xlim_update)
        self.axes.set_ylim(ylim_update)
        self.figure.canvas.draw()


class AxisDialog(QDialog):
    """Qt GUI to update the axix limits"""

    def __init__(self, parent, axes):
        super().__init__(parent)

        # Get axes info
        self.axes = axes
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()

        self.setWindowTitle('Update Axis Limits')
        self.grid = QGridLayout(self)

        # Create widgets
        self.label_x = QLabel("x-axis")
        self.label_y = QLabel("y-axis")
        self.label_start = QLabel("Start")
        self.label_stop = QLabel("Stop")
        # self.label_delta = QLabel("Delta")
        self.edit_xmin = QLineEdit("{:0.5f}".format(self.xlim[0]))
        self.edit_xmax = QLineEdit("{:0.5f}".format(self.xlim[1]))
        # self.edit_xdelta = QLineEdit()
        self.edit_ymin = QLineEdit("{:0.5f}".format(self.ylim[0]))
        self.edit_ymax = QLineEdit("{:0.5f}".format(self.ylim[1]))
        # self.edit_ydelta = QLineEdit()

        self.edit_xmin.setValidator(QDoubleValidator())
        self.edit_xmax.setValidator(QDoubleValidator())
        self.edit_ymin.setValidator(QDoubleValidator())
        self.edit_ymax.setValidator(QDoubleValidator())

        # OK and Cancel buttons
        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        # Add widgets to grid
        self.grid.addWidget(self.label_start, 0, 1, 1, 1)
        self.grid.addWidget(self.label_stop, 0, 2, 1, 1)
        # self.grid.addWidget(self.label_delta, 0, 3, 1, 1)
        self.grid.addWidget(self.label_x, 1, 0, 1, 1)
        self.grid.addWidget(self.label_y, 2, 0, 1, 1)
        self.grid.addWidget(self.edit_xmin, 1, 1, 1, 1)
        self.grid.addWidget(self.edit_xmax, 1, 2, 1, 1)
        # self.grid.addWidget(self.edit_xdelta, 1, 3, 1, 1)
        self.grid.addWidget(self.edit_ymin, 2, 1, 1, 1)
        self.grid.addWidget(self.edit_ymax, 2, 2, 1, 1)
        # self.grid.addWidget(self.edit_ydelta, 2, 3, 1, 1)
        self.grid.addWidget(self.buttonbox, 3, 1, 1, 2)

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def update_axis(axes, parent=None):
        dialog = AxisDialog(parent, axes)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            xlim = (float(dialog.edit_xmin.text()),
                    float(dialog.edit_xmax.text()))
            ylim = (float(dialog.edit_ymin.text()),
                    float(dialog.edit_ymax.text()))
        else:
            xlim = dialog.axes.get_xlim()
            ylim = dialog.axes.get_ylim()

        return (xlim, ylim, result == QDialog.Accepted)
