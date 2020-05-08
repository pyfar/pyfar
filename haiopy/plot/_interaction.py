import warnings

import matplotlib as mpl
import numpy as np


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


class AxisModifier(object):
    def __init__(self, axes, figure):
        self._axes = axes
        self._figure = figure

    @property
    def axes(self):
        return self._axes

    @property
    def figure(self):
        return self._figure

    def connect(self):
        self.figure.AxisModifier = self


class AxisModifierLines(AxisModifier):
    """ Keybindings for cycling and toggling visible channels. Uses the event
    API provided by matplotlib. Works for the jupyter-matplotlib and qt
    backends.
    """
    def __init__(self, axes, figure):
        super(AxisModifierLines, self).__init__(axes, figure)
        self.cycle = Cycle(self.axes.lines)
        self.current_line = self.cycle.current()
        self.all_visible = True

    def cycle_lines(self, event):
        if event.key in ['*', ']', '[', '/', '7']:
            if self.all_visible:
                for i in range(len(self.axes.lines)):
                    self.axes.lines[i].set_visible(False)
            else:
                self.current_line.set_visible(False)
            if event.key in ['*', ']']:
                self.current_line = next(self.cycle)
            elif event.key in ['[', '/', '7']:
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

    def move_y_axis(self, event):
        if event.key in ['up', 'down']:
            lims = np.asarray(self.axes.get_ylim())
            dyn_range = (np.diff(lims))
            shift = np.int(np.round(0.1 * dyn_range))
            if event.key in ['up']:
                self.axes.set_ylim(lims + shift)
            elif event.key in ['down']:
                self.axes.set_ylim(lims - shift)
            self.figure.canvas.draw()

    def zoom_y_axis(self, event):
        raise NotImplementedError("Use child classes to specify if the zoom \
            is symmetric or asymmetric.")

    def connect(self):
        super().connect()
        self._cycle_lines = self.figure.canvas.mpl_connect(
            'key_press_event', self.cycle_lines)
        self._toogle_lines = self.figure.canvas.mpl_connect(
            'key_press_event', self.toggle_all_lines)
        self._move_y_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.move_y_axis)
        self._zoom_y_axis = self.figure.canvas.mpl_connect(
            'key_press_event', self.zoom_y_axis)

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(
            'key_press_event', self._cycle_lines)
        self.figure.canvas.mpl_disconnect(
            'key_press_event', self._toogle_lines)
        self.figure.canvas.mpl_disconnect(
            'key_press_event', self._move_y_axis)
        self.figure.canvas.mpl_disconnect(
            'key_press_event', self._zoom_y_axis)


class AxisModifierLinesLogYAxis(AxisModifierLines):
    def __init__(self, axes, figure):
        super(AxisModifierLinesLogYAxis, self).__init__(axes, figure)

    def zoom_y_axis(self, event):
        if event.key in ['shift+up', 'shift+down']:
            lims = np.asarray(self.axes.get_ylim())
            dyn_range = (np.diff(lims))
            zoom = np.int(np.round(0.1 * dyn_range))

            if event.key in ['shift+up']:
                lims[0] = lims[0] + zoom
            elif event.key in ['shift+down']:
                lims[0] = lims[0] - zoom

            self.axes.set_ylim(lims)
            self.figure.canvas.draw()


class AxisModifierLinesLinYAxis(AxisModifierLines):
    def __init__(self, axes, figure):
        super(AxisModifierLinesLinYAxis, self).__init__(axes, figure)

    def zoom_y_axis(self, event):
        if event.key in ['shift+up', 'shift+down']:
            lims = np.asarray(self.axes.get_ylim())
            dyn_range = (np.diff(lims))
            zoom = np.int(np.round(0.1 * dyn_range))

            if event.key in ['shift+up']:
                pass
            elif event.key in ['shift+down']:
                zoom = -zoom

            lims[0] = lims[0] + zoom
            lims[1] = lims[1] - zoom

            self.axes.set_ylim(lims)
            self.figure.canvas.draw()


class AxisModifierDialog(AxisModifier):
    def __init__(self, axes, figure):
        super(AxisModifierDialog, self).__init__(axes, figure)

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
