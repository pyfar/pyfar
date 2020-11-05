import warnings
import matplotlib as mpl
import numpy as np
from . import _line
from . import line
import matplotlib.pyplot as plt
from pyfar import Signal


class Cycle(object):
    """ Cycle class implementation inspired by itertools.cycle. Supports
    circular iterations into two directions by using next and previous.
    """
    def __init__(self, data):
        """
        Parameters
        ----------
        data : array like
            The data to be iterated over.
        """
        self.data = data
        self.n_elements = len(self.data)
        self.index = 0

    def __next__(self):
        index = (self.index + 1) % self.n_elements
        self.index = index
        return self.data[self.index]

    def next(self):
        return self.__next__()

    def previous(self):
        index = self.index - 1
        if index < 0:
            index = self.n_elements + index
        self.index = index
        return self.data[self.index]

    def current(self):
        return self.data[self.index]


class Interaction(object):
    def __init__(self, plot_type, axes, signal, style, **kwargs):
        self._plot_type = plot_type
        self._axes = np.asarray(axes)
        self._signal = signal
        self._style = style
        self._kwargs = kwargs
        self._figure = axes.figure
        if plot_type == 'LineXLin':
            self.axis_modifier = AxisModifierLinesXLin(axes, signal)
            self.axis_modifier.connect()
            self.connect()
        elif plot_type == 'LineXLog':
            self.axis_modifier = AxisModifierLinesXLog(axes, signal)
            self.axis_modifier.connect()
            self.connect()
        elif plot_type == 'spectrogram':
            self.axis_modifier = AxisModifierSpectrogram(axes, signal)
            self.axis_modifier.connect()
            self.connect()

    @property
    def plot_type(self):
        return self._plot_type

    @property
    def axes(self):
        return self._axes

    @property
    def signal(self):
        return self._signal

    @property
    def style(self):
        return self._style

    @property
    def kwargs(self):
        return self._kwargs

    @axes.setter
    def axes(self, ax):
        self._axes = ax

    @property
    def figure(self):
        return self._figure

    def clear_axes(self):
        self._figure.clear()
        self.axes = self._figure.add_subplot()
        self._figure.set_size_inches(plt.rcParams.get('figure.figsize'),
                                     forward=True)

    def toggle_plot(self, event):
        if event.key not in ['ctrl+1', 'ctrl+2', 'ctrl+3', 'ctrl+4', 'ctrl+5',
                             'ctrl+6', 'ctrl+7', 'ctrl+8', 'ctrl+9']:
            return
        with plt.style.context(line.plotstyle(self.style)):
            if event.key in ['ctrl+1']:  # plot time domain
                self.clear_axes()
                _line._time(self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
                self.change_modifier('line_lin_Y')
            if event.key in ['ctrl+2']:  # plot magnitude
                self.clear_axes()
                _line._freq(self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
                self.change_modifier('line_log_Y')
            if event.key in ['ctrl+3']:  # plot phase
                self.clear_axes()
                _line._phase(self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
                self.change_modifier('line_lin_Y')
            if event.key in ['ctrl+4']:  # plot time domain in decibels
                self.clear_axes()
                _line._time_dB(self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
                self.change_modifier('line_log_Y')
            if event.key in ['ctrl+5']:  # plot group delay
                self.clear_axes()
                _line._group_delay(self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
                self.change_modifier('line_lin_Y')
            if event.key in ['ctrl+6']:  # plot spectrogram
                self.clear_axes()
                self._plot_signal = Signal(
                    self.signal.time[0], self.signal.sampling_rate,
                    'time', self.signal.signal_type)
                self.axes = _line._spectrogram_cb(
                    self._plot_signal, ax=self.axes)
                self.figure.canvas.draw()
                self.change_modifier('spectrogram')
            if event.key in ['ctrl+7']:  # plot magnitude and phase
                self.clear_axes()
                self.axes = _line._freq_phase(
                    self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
            if event.key in ['ctrl+8']:  # plot magnitude and group delay
                self.clear_axes()
                self.axes = _line._freq_group_delay(
                    self.signal, ax=self.axes, **self.kwargs)
                self.figure.canvas.draw()
            if event.key in ['ctrl+9']:  # plot all
                if self.signal.time.shape[0] > 1:
                    warnings.warn(
                        "For multiple dimensions not implemented yet.")
                else:
                    self.clear_axes()
                    self.figure.set_size_inches(6, 6, forward=True)
                    self.axes = _line._signal(
                        self.signal, ax=self.axes, **self.kwargs)
                    self.figure.canvas.draw()

    def change_modifier(self, plot_type):
        self.axis_modifier.disconnect()
        if plot_type == 'line_lin_Y':
            self.axis_modifier = AxisModifierLinesXLin(
                self.axes, self.signal)
            self.axis_modifier.connect()
        elif plot_type == 'line_log_Y':
            self.axis_modifier = AxisModifierLinesXLog(
                self.axes, self.signal)
            self.axis_modifier.connect()
        elif plot_type == 'spectrogram':
            self.axis_modifier = AxisModifierSpectrogram(
                self.axes[0], self.signal)
            self.axis_modifier.connect()

    def connect(self):
        self.figure.AxisModifier = self
        self._toggle_plot = self.figure.canvas.mpl_connect(
            'key_press_event', self.toggle_plot)

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(
            'key_press_event', self._toggle_plot)


class AxisModifier(object):
    def __init__(self, axes):
        self._axes = axes
        self._figure = axes.figure

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, ax):
        self._axes = ax

    @property
    def figure(self):
        return self._figure

    def clear_axes(self):
        self._figure.clear()
        self.axes = self._figure.add_subplot()
        self._figure.set_size_inches(plt.rcParams.get('figure.figsize'),
                                     forward=True)


class AxisModifierSpectrogram(AxisModifier):
    """ Keybindings for cycling and toggling visible channels. Uses the event
    API provided by matplotlib. Works for the jupyter-matplotlib and qt
    backends.
    """
    def __init__(self, axes, signal):
        super(AxisModifierSpectrogram, self).__init__(axes)
        self.index = 0
        self._signal = signal
        self._plot_signal = 0
        self.axes = axes

    @property
    def signal(self):
        return self._signal

    def cycle_lines(self, event):
        if event.key in [',', '.']:
            if event.key in ['.']:
                self.index = (self.index + 1) % self.signal.time.shape[0]
            elif event.key in [',']:
                self.index = self.index - 1
                if self.index < 0:
                    self.index = self.signal.time.shape[0] + self.index

            self.clear_axes()
            self._plot_signal = Signal(
                self.signal.time[self.index], self.signal.sampling_rate,
                'time', self.signal.signal_type)
            self.axes = _line._plot_spectrogram_cb(
                self._plot_signal, ax=self.axes)
            self.figure.canvas.draw()

    def move_y_axis(self, event):
        if event.key in ['up', 'down']:
            if isinstance(self.axes, (np.ndarray)):
                for ax in self.axes.ravel():
                    lims = np.asarray(ax.get_ylim())
                    dyn_range = (np.diff(lims))
                    shift = 0.1 * dyn_range
                    if event.key in ['up']:
                        ax.set_ylim(lims + shift)
                    elif event.key in ['down']:
                        ax.set_ylim(lims - shift)
            else:
                lims = np.asarray(self.axes.get_ylim())
                dyn_range = (np.diff(lims))
                shift = 0.1 * dyn_range
                if event.key in ['up']:
                    self.axes.set_ylim(lims + shift)
                elif event.key in ['down']:
                    self.axes.set_ylim(lims - shift)
            self.figure.canvas.draw()

    def zoom_y_axis(self, event):
        if event.key in ['shift+up', 'shift+down']:
            if isinstance(self.axes, (np.ndarray)):
                for ax in self.axes.ravel():
                    lims = np.asarray(ax.get_ylim())
                    dyn_range = (np.diff(lims))
                    zoom = 0.1 * dyn_range

                    if event.key in ['shift+up']:
                        lims[0] = lims[0] + zoom
                    elif event.key in ['shift+down']:
                        lims[0] = lims[0] - zoom

                    ax.set_ylim(lims)
            self.figure.canvas.draw()

    def connect(self):
        self._move_y_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.move_y_axis)
        self._zoom_y_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.zoom_y_axis)
        self._cycle_lines = self.figure.canvas.mpl_connect(
            'key_press_event', self.cycle_lines)

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self._move_y_axis)
        self.figure.canvas.mpl_disconnect(self._zoom_y_axis)
        self.figure.canvas.mpl_disconnect(self._cycle_lines)


class AxisModifierLines(AxisModifier):
    """ Keybindings for cycling and toggling visible channels. Uses the event
    API provided by matplotlib. Works for the jupyter-matplotlib and qt
    backends.
    """
    def __init__(self, axes, signal):
        super(AxisModifierLines, self).__init__(axes)
        self.cycle = Cycle(self.axes.lines)
        self.current_line = self.cycle.current()
        self.all_visible = True
        self._signal = signal

    @property
    def signal(self):
        return self._signal

    def cycle_lines(self, event):
        if event.key in [',', '.']:
            if self.all_visible:
                for i in range(len(self.axes.lines)):
                    self.axes.lines[i].set_visible(False)
            else:
                self.current_line.set_visible(False)
            if event.key in ['.']:
                self.current_line = next(self.cycle)
            elif event.key in [',']:
                self.current_line = self.cycle.previous()
            self.current_line.set_visible(True)
            self.all_visible = False
            self.figure.canvas.draw()

    def toggle_all_lines(self, event):
        if event.key in ['a']:
            if self.all_visible:
                for i in range(len(self.axes.lines)):
                    self.axes.lines[i].set_visible(False)
                self.current_line.set_visible(True)
                self.all_visible = False
            else:
                for i in range(len(self.axes.lines)):
                    self.axes.lines[i].set_visible(True)
                self.all_visible = True
            self.figure.canvas.draw()

    def move_x_axis(self, event):
        raise NotImplementedError("Realized in child classes")

    def zoom_x_axis(self, event):
        raise NotImplementedError("Realized in child classes")

    def move_y_axis(self, event):
        if event.key in ['up', 'down']:
            if isinstance(self.axes, (np.ndarray)):
                for ax in self.axes.ravel():
                    lims = np.asarray(ax.get_ylim())
                    dyn_range = (np.diff(lims))

                    shift = 0.1 * dyn_range
                    if event.key in ['up']:
                        ax.set_ylim(lims + shift)
                    elif event.key in ['down']:
                        ax.set_ylim(lims - shift)
            else:
                lims = np.asarray(self.axes.get_ylim())
                dyn_range = (np.diff(lims))

                shift = 0.1 * dyn_range
                if event.key in ['up']:
                    self.axes.set_ylim(lims + shift)
                elif event.key in ['down']:
                    self.axes.set_ylim(lims - shift)
            self.figure.canvas.draw()

    def zoom_y_axis(self, event):
        if event.key in ['shift+up', 'shift+down']:
            lims = np.asarray(self.axes.get_ylim())
            dyn_range = (np.diff(lims))

            zoom = 0.05 * dyn_range
            if event.key in ['shift+up']:
                pass
            elif event.key in ['shift+down']:
                zoom = -zoom

            lims[0] = lims[0] + zoom
            lims[1] = lims[1] - zoom

            self.axes.set_ylim(lims)
            self.figure.canvas.draw()

    def connect(self):
        self._cycle_lines = self.figure.canvas.mpl_connect(
            'key_press_event', self.cycle_lines)
        self._toogle_lines = self.figure.canvas.mpl_connect(
            'key_press_event', self.toggle_all_lines)
        self._move_x_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.move_x_axis)
        self._move_y_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.move_y_axis)
        self._zoom_x_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.zoom_x_axis)
        self._zoom_y_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.zoom_y_axis)

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self._cycle_lines)
        self.figure.canvas.mpl_disconnect(self._toogle_lines)
        self.figure.canvas.mpl_disconnect(self._move_x_axis)
        self.figure.canvas.mpl_disconnect(self._move_y_axis)
        self.figure.canvas.mpl_disconnect(self._zoom_x_axis)
        self.figure.canvas.mpl_disconnect(self._zoom_y_axis)


class AxisModifierLinesXLog(AxisModifierLines):
    def __init__(self, axes, signal):
        super(AxisModifierLinesXLog, self).__init__(axes, signal)

    def move_x_axis(self, event):
        if event.key in ['left', 'right']:
            if isinstance(self.axes, (np.ndarray)):
                for ax in self.axes.ravel():
                    lims = np.asarray(ax.get_xlim())
                    exp = np.floor(np.log10(lims))

                    if event.key in ['left']:
                        lims_new = np.zeros(2)

                        # handling log10 boundaries (e.g., 1, 10, 100, ..)
                        for n in range(len(lims)):

                            shift = 10**exp[n]
                            lim_new = lims[n] - shift
                            exp_new = np.floor(np.log10(lim_new))

                            if exp[n] == np.log10(lims[n]):
                                # lim is at a boundary - decrease step size
                                exp[n] -= 1
                                shift = 10**exp[n]
                                lims_new[n] = lims[n] - shift
                            elif exp_new < exp[n]:
                                # lim passed boundary - avoid this
                                lims_new[n] = 10**exp[n]
                            else:
                                # simple case
                                lims_new[n] = lim_new

                        ax.set_xlim(lims_new)

                    elif event.key in ['right']:
                        shift = 10**exp
                        ax.set_xlim(lims + shift)
            else:
                lims = np.asarray(self.axes.get_xlim())
                exp = np.floor(np.log10(lims))

                if event.key in ['left']:
                    lims_new = np.zeros(2)

                    # handling log10 boundaries (e.g., 1, 10, 100, ..)
                    for n in range(len(lims)):

                        shift = 10**exp[n]
                        lim_new = lims[n] - shift
                        exp_new = np.floor(np.log10(lim_new))

                        if exp[n] == np.log10(lims[n]):
                            # lim is at a boundary - decrease step size
                            exp[n] -= 1
                            shift = 10**exp[n]
                            lims_new[n] = lims[n] - shift
                        elif exp_new < exp[n]:
                            # lim passed boundary - avoid this
                            lims_new[n] = 10**exp[n]
                        else:
                            # simple case
                            lims_new[n] = lim_new

                    self.axes.set_xlim(lims_new)

                elif event.key in ['right']:
                    shift = 10**exp
                    self.axes.set_xlim(lims + shift)

            self.figure.canvas.draw()

    def zoom_x_axis(self, event):
        if event.key in ['shift+left', 'shift+right']:
            lims = np.asarray(self.axes.get_xlim())
            exp = np.floor(np.log10(lims))
            shifts = np.array([0, 0])

            zoom_out = True if event.key in ['shift+left'] else False
            if zoom_out:
                n = 0
                shifts[1] = 10**exp[1]
            else:
                n = 1
                shifts[0] = 10**exp[0]

            # handling log10 boundaries (e.g., 1, 10, 100, ..)
            shift = 10**exp[n]
            lim_new = lims[n] - shift
            exp_new = np.floor(np.log10(lim_new))

            if exp[n] == np.log10(lims[n]):
                # lim is at a boundary - decrease step size
                exp[n] -= 1
                shift = 10**exp[n]
                shifts[n] = -shift
            elif exp_new < exp[n]:
                # lim passed boundary - avoid this
                shifts[n] = 10**exp[n] - lims[n]
            else:
                # simple case
                shifts[n] = -shift

            if zoom_out:
                lims += shifts
            else:
                # avoid flipping the axis
                lims_new = lims + shifts
                while lims_new[0] > lims_new[1]:
                    shifts /= 2
                    lims_new = lims + shifts

                lims = lims_new

            self.axes.set_xlim(lims)
            self.figure.canvas.draw()


class AxisModifierLinesXLin(AxisModifierLines):
    def __init__(self, axes, signal):
        super(AxisModifierLinesXLin, self).__init__(axes, signal)

    def move_x_axis(self, event):
        if event.key in ['left', 'right']:
            if isinstance(self.axes, (np.ndarray)):
                for ax in self.axes.ravel():
                    lims = np.asarray(ax.get_xlim())
                    dyn_range = (np.diff(lims))

                    shift = 0.1 * dyn_range
                    if event.key in ['left']:
                        ax.set_xlim(lims - shift)
                    elif event.key in ['right']:
                        ax.set_xlim(lims + shift)
            else:
                lims = np.asarray(self.axes.get_xlim())
                dyn_range = (np.diff(lims))

                shift = 0.1 * dyn_range
                if event.key in ['left']:
                    self.axes.set_xlim(lims - shift)
                elif event.key in ['right']:
                    self.axes.set_xlim(lims + shift)
            self.figure.canvas.draw()

    def zoom_x_axis(self, event):
        if event.key in ['shift+left', 'shift+right']:
            lims = np.asarray(self.axes.get_xlim())
            dyn_range = (np.diff(lims))

            zoom = 0.1 * dyn_range
            if event.key in ['shift+right']:
                # zoom in
                pass
            elif event.key in ['shift+left']:
                # zoom out
                zoom = -zoom

            lims[0] = lims[0] + zoom
            lims[1] = lims[1] - zoom

            self.axes.set_xlim(lims)
            self.figure.canvas.draw()


class AxisModifierDialog(AxisModifier):
    def __init__(self, axes):
        super(AxisModifierDialog, self).__init__(axes)

    def open(self, event):
        if 'Qt' in mpl.get_backend():
            from .gui import AxisDialog
            if event.key in ['d']:
                xlim_update, ylim_update, result = \
                    AxisDialog.update_axis(axes=self.axes)
                self.axes.set_xlim(xlim_update)
                self.axes.set_ylim(ylim_update)
                self.figure.canvas.draw()
        else:
            warnings.warn("Only implemented for matplotlib's Qt backends")

    def connect(self):
        super().connect()
        self._open = self.figure.canvas.mpl_connect(
            'key_press_event', self.open)
