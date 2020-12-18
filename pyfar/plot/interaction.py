# TODO:
# - implement toggle all and cycler
# - use new Interaction class for all plots in .line module
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfar import Signal
from pyfar.plot import utils
from pyfar.plot import _line


class Cycle(object):
    """ Cycle class implementation inspired by itertools.cycle. Supports
    circular iterations into two directions by using next and previous.
    """
    def __init__(self, data, index=0):
        """
        Parameters
        ----------
        data : array like, Signal
            The data to be iterated over.
        """
        self.data = data
        self.index = index

    def next(self):
        self.increase_index()
        return self.data[self.index]

    def previous(self):
        self.decrease_index()
        return self.data[self.index]

    def current(self):
        return self.data[self.index]

    def increase_index(self):
        self.index = (self.index + 1) % self.n_elements

    def decrease_index(self):
        index = self.index - 1
        if index < 0:
            index = self.n_elements + index
        self.index = index

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        # set the number of elements
        if isinstance(data, Signal):
            self._n_elements = data.cshape[0]
        else:
            self.n_elements = len(data)


class EventEmu(object):
    def __init__(self, key):
        """
        Helper class to emulate events. This makes it possible to call
        functions in Interaction without user input and can be helpfull for
        debugging.

        Parameters
        ----------
        key : str
            name of key to emulate, e.g., 'left' for left arrow.
        """
        self.key = key


class PlotParameter(object):
    """Store and change plot parameter.

    This class stores all unique input parameters that controll the plot look.
    The plot parameters are changed upon request from Interaction. The signal
    and axes are stored in Interaction.

    """
    def __init__(self, plot,
                 dB_time=False, dB_freq=True,              # dB properties
                 log_prefix=20, log_reference=10,          # same for time/freq
                 xscale='log', yscale='linear',            # axis scaling
                 deg=False, unwrap=False,                  # phase properties
                 unit=None,                                # group delay unit
                 window='hann', window_length=1014,        # spectrogram
                 window_overlap_fct=.5,
                 cmap=mpl.cm.get_cmap(name='magma')):      # colormap

        # store input
        self.dB_time = dB_time
        self.dB_freq = dB_freq
        self.log_prefix = log_prefix
        self.log_reference = log_reference
        self.xscale = xscale
        self.yscale = yscale
        self.deg = deg
        self.unwrap = unwrap
        self.unit = unit
        self.window = window
        self.window_length = window_length
        self.window_overlap_fct = window_overlap_fct
        self.cmap = cmap

        # set axis types based on `plot`
        self.update_axis_type(plot)

    def update_axis_type(self, plot):
        """Set axis types based on plot.

        A plot can have two or three axis. All plots have x- and y-axis. Some
        plots also have a colormap (cm, which we see as an axis here). Each
        axis can have one or multiple display options, e.g., a log or linear
        frequency axis. These options are set by this function using four
        variables:

        self._*_type : None, list
            The axis type as defined in `Interaction.apply_move_and_zoom`
        self._*_param : str, only if `len(self._*_type) > 1`
            The name of the plot parameter that changes if `self._*_type`
            changes
        self._*_values : str, only if `len(self._*_type) > 1`
            The values of `self._*_param`. Must have the same length as
            `self._*_type`
        self._*_id : int, only if `self._*_type is not None`
            An id that sets the current state of `self._*_type`

        * can be 'x', 'y', or 'cm'.

        In addtition `self._cycler_type` is set:

        self._cycler_type : 'line', 'signal', None
            determines in which way `Cycle` cycles through the channels.
            'line'   - cycling is done by toggling line visibility
                       (e.g. for pyfar.plot.line.time)
            'signal' - cycling is done by re-calling the plot function with a
                       signal slice (e.g. for pyfar.plot.line.spectrogram)
            None     - cycling is not possible

        Parameters
        ----------
        plot : str
            Defines the plot by module.plot_function, e.g., 'line.freq'

        """

        # set the axis, color map, and cycle, parameter for each plot
        if plot == 'line.time':
            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            # y-axis
            self._y_type = ['other', 'dB']
            self._y_param = 'dB_time'
            self._y_values = [False, True]
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        elif plot == 'line.freq':
            # x-axis
            self._x_type = ['freq', 'other']
            self._x_param = 'xscale'
            self._x_values = ['log', 'linear']
            self._x_id = self._x_values.index(getattr(self, self._x_param))
            # y-axis
            self._y_type = ['dB', 'other']
            self._y_param = 'dB_freq'
            self._y_values = [True, False]
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        elif plot == 'line.phase':
            # x-axis
            self._x_type = ['freq', 'other']
            self._x_param = 'xscale'
            self._x_values = ['log', 'linear']
            self._x_id = self._x_values.index(getattr(self, self._x_param))
            # y-axis
            self._y_type = ['other', 'other', 'other']
            self._y_id = 0
            self._y_param = 'unwrap'
            self._y_values = [True, False, "360"]
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        elif plot == 'line.group_delay':
            # x-axis
            self._x_type = ['freq', 'other']
            self._x_param = 'xscale'
            self._x_values = ['log', 'linear']
            self._x_id = self._x_values.index(getattr(self, self._x_param))
            # y-axis
            self._y_type = ['other', 'other', 'other', 'other', 'other']
            self._y_param = 'unit'
            self._y_values = [None, 's', 'ms', 'mus', 'samples']
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        elif plot == 'line.spectrogram':
            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            # y-axis
            self._y_type = ['freq', 'other']
            self._y_param = 'yscale'
            self._y_values = ['log', 'linear']
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = ['dB', 'other']
            self._cm_param = 'dB_freq'
            self._cm_values = [True, False]
            self._cm_id = self._cm_values.index(getattr(self, self._cm_param))
            # cycler type
            self._cycler_type = 'signal'

        elif plot == 'line.time_freq':
            # same as time
            # (currently interaction uses only the axis of the top plot)

            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            # y-axis
            self._y_type = ['other', 'dB']
            self._y_param = 'dB_time'
            self._y_values = [False, True]
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        elif plot in ['line.freq_phase', 'line.freq_group_delay']:
            # same as freq
            # (currently interaction uses only the axis of the top plot)

            # x-axis
            self._x_type = ['freq', 'other']
            self._x_param = 'xscale'
            self._x_values = ['log', 'linear']
            self._x_id = self._x_values.index(getattr(self, self._x_param))
            # y-axis
            self._y_type = ['dB', 'other']
            self._y_param = 'dB_freq'
            self._y_values = [True, False]
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        else:
            raise ValueError(f"{plot} not known.")

        self.plot = plot

    def toggle_x(self):
        """Toogle the x-axis type."""
        changed = False
        if self.x_type is not None:
            if len(self._x_type) > 1:
                self._x_id = (self._x_id + 1) % len(self._x_type)
                setattr(self, self._x_param, self._x_values[self._x_id])
                changed = True

        return changed

    def toggle_y(self):
        """Toogle the y-axis type."""
        changed = False
        if self.y_type is not None:
            if len(self._y_type) > 1:
                self._y_id = (self._y_id + 1) % len(self._y_type)
                setattr(self, self._y_param, self._y_values[self._y_id])
                changed = True

        return changed

    def toggle_cm(self):
        """Toogle the color map type."""
        changed = False
        if self.cm_type is not None:
            if len(self._cm_type) > 1:
                self._cm_id = (self._cm_id + 1) % len(self._cm_type)
                setattr(self, self._cm_param, self._cm_values[self._cm_id])
                changed = True

        return changed

    @property
    def x_type(self):
        """Return x-axis type."""
        x_type = self._x_type[self._x_id] if self._x_type is not None else None
        return x_type

    @property
    def y_type(self):
        """Return y-axis type."""
        y_type = self._y_type[self._y_id] if self._y_type is not None else None
        return y_type

    @property
    def cm_type(self):
        """Return color map type."""
        cm_type = self._cm_type[self._cm_id] if self._cm_type is not None \
            else None
        return cm_type


class Interaction(object):
    """Change the plot and plot parameters based on keyboard shortcuts.

    Actions:
    Toggle between plots; move or zoom axis or color map; toogle axis types;
    cycle channels.
    """
    def __init__(self, signal, axes, style, plot_parameter, **kwargs):
        """
        Change the plot and plot parameters based on keyboard shortcuts.

        Parameters
        ----------
        signal : Signal
            audio data
        axes : Matplotlib axes
            axes handle
        style : plot style
            E.g. 'light'
        plot_parameter : PlotParameter
            An object of the PlotParameter class
        """

        # save input arguments
        self.signal = signal
        self.ax = axes
        self.figure = axes.figure
        self.style = style
        self.params = plot_parameter
        self.kwargs = kwargs

        # store last key event (done in self.select_action)
        self.event = None

        # initialize cycler
        self.init_cycler()

        # get keyboard shortcuts
        self.keys = utils.shortcuts(False)
        # get control shortcuts (we don't need the 'info' field here)
        self.ctr = self.keys["controls"]
        for ctr in self.ctr:
            self.ctr[ctr] = self.ctr[ctr]["key"]
        # get plot shortcuts (we don't need the 'info' field here)
        self.plot = self.keys["plots"]
        for plot in self.plot:
            self.plot[plot] = self.plot[plot]["key"]

        # connect to Matplotlib
        self.connect()

    def select_action(self, event):
        """
        Select what to do based on the keyboard event

        Parameters
        ----------
        event : mpl_connect event
            class that contains the action, e.g., the pressed key as a string
        """

        ctr = self.ctr
        self.event = event

        # toggle plot
        if event.key in [self.plot[plot] for plot in self.plot]:
            self.toggle_plot(event)

        # x-axis move/zoom
        elif event.key in [ctr["move_left"], ctr["move_right"],
                           ctr["zoom_x_in"], ctr["zoom_x_out"]]:
            self.move_and_zoom(event, 'x')

        # y-axis move/zoom
        elif event.key in [ctr["move_up"], ctr["move_down"],
                           ctr["zoom_y_in"], ctr["zoom_y_out"]]:
            self.move_and_zoom(event, 'y')

        # color map move/zoom
        elif event.key in [ctr["move_cm_up"], ctr["move_cm_down"],
                           ctr["zoom_cm_in"], ctr["zoom_cm_out"]]:
            self.move_and_zoom(event, 'cm')

        # x-axis toggle
        elif event.key == ctr["toggle_x"]:
            changed = self.params.toggle_x()
            if changed:
                self.toggle_plot(EventEmu(self.plot[self.params.plot]))

        # y-axis toggle
        elif event.key == ctr["toggle_y"]:
            changed = self.params.toggle_y()
            if changed:
                self.toggle_plot(EventEmu(self.plot[self.params.plot]))

        # color map toggle
        elif event.key == ctr["toggle_cm"]:
            changed = self.params.toggle_cm()
            if changed:
                self.toggle_plot(EventEmu(self.plot[self.params.plot]))

        # toggle line visibility
        elif event.key == ctr["toggle_all"]:
            if self.plot != 'line.spectrogram':
                self.toggle_all_lines()

        # cycle channels
        elif event.key in [ctr["next"], ctr["prev"]]:
            self.cycle_lines(event)

    def toggle_plot(self, event):
        """Toggle between plot types."""

        plot = self.plot
        prm = self.params

        with plt.style.context(utils.plotstyle(self.style)):
            self.figure.clear()
            self.ax = None

            if event.key in plot['line.time']:
                self.ax = _line._time(
                    self.signal, prm.dB_time, prm.log_prefix,
                    prm.log_reference, self.ax, **self.kwargs)
                self.params.update_axis_type('line.time')

            elif event.key in plot['line.freq']:
                self.ax = _line._freq(
                    self.signal, prm.dB_freq, prm.log_prefix,
                    prm.log_reference, prm.xscale, self.ax, **self.kwargs)
                self.params.update_axis_type('line.freq')

            elif event.key in plot['line.phase']:
                self.ax = _line._phase(
                    self.signal, prm.deg, prm.unwrap, prm.xscale,
                    self.ax, **self.kwargs)
                self.params.update_axis_type('line.phase')

            elif event.key in plot['line.group_delay']:
                self.ax = _line._group_delay(
                    self.signal, prm.unit, prm.xscale, self.ax, **self.kwargs)
                self.params.update_axis_type('line.group_delay')

            elif event.key in plot['line.spectrogram']:
                ax = _line._spectrogram_cb(
                    self.signal, prm.dB_freq, prm.log_prefix,
                    prm.log_reference, prm.yscale, prm.window,
                    prm.window_length, prm.window_overlap_fct, prm.cmap,
                    self.ax, **self.kwargs)
                self.ax = ax[0]
                self.params.update_axis_type('line.spectrogram')

            elif event.key in plot['line.time_freq']:
                ax = _line._time_freq(
                    self.signal, prm.dB_time, prm.dB_freq, prm.log_prefix,
                    prm.log_reference, prm.xscale, self.ax, **self.kwargs)
                self.ax = ax[0]
                self.params.update_axis_type('line.time')

            elif event.key in plot['line.freq_phase']:
                ax = _line._freq_phase(
                    self.signal, prm.dB_freq, prm.log_prefix,
                    prm.log_reference, prm.xscale, prm.deg, prm.unwrap,
                    self.ax, **self.kwargs)
                self.ax = ax[0]
                self.params.update_axis_type('line.freq')

            elif event.key in plot['line.freq_group_delay']:
                ax = _line._freq_group_delay(
                    self.signal, prm.dB_freq, prm.log_prefix,
                    prm.log_reference, prm.unit, prm.xscale,
                    self.ax, **self.kwargs)
                self.ax = ax[0]
                self.params.update_axis_type('line.freq')

            self.update_cycler()
            if not self.all_visible:
                self.cycle_lines(EventEmu('redraw'))
            else:
                self.figure.canvas.draw()

    def move_and_zoom(self, event, axis):
        """
        Get parameters for moving/zoom and call apply_move_and_zoom().
        See apply_move_and_zoom for more parameter description.
        """

        ctr = self.ctr
        getter = None

        # move/zoom x-axis
        if axis == "x":

            if self.params.x_type is None:
                return

            getter = self.ax.get_xlim
            setter = self.ax.set_xlim
            axis_type = self.params.x_type
            if event.key == ctr["move_left"] or event.key == ctr["move_right"]:
                operation = "move"
            else:
                operation = "zoom"
            if event.key == ctr["move_right"] or event.key == ctr["zoom_x_in"]:
                direction = "increase"
            else:
                direction = "decrease"

        # move/zoom y-axis
        elif axis == "y":

            if self.params.y_type is None:
                return

            getter = self.ax.get_ylim
            setter = self.ax.set_ylim
            axis_type = self.params.y_type
            if event.key == ctr["move_up"] or event.key == ctr["move_down"]:
                operation = "move"
            else:
                operation = "zoom"
            if event.key == ctr["move_up"] or event.key == ctr["zoom_y_in"]:
                direction = "increase"
            else:
                direction = "decrease"

        # move/zoom colorbar
        elif axis == "cm":

            if self.params.cm_type is None:
                return

            for cm in self.ax.get_children():
                if type(cm) == mpl.collections.QuadMesh:
                    break

            getter = cm.get_clim
            setter = cm.set_clim
            axis_type = self.params.cm_type
            if event.key == ctr["move_cm_up"] or \
                    event.key == ctr["move_cm_down"]:
                operation = "move"
            else:
                operation = "zoom"
            if event.key == ctr["move_cm_up"] or \
                    event.key == ctr["zoom_cm_in"]:
                direction = "increase"
            else:
                direction = "decrease"

        # move or zoom
        if getter is not None:
            self.apply_move_and_zoom(getter, setter, axis_type,
                                     operation, direction)

    def apply_move_and_zoom(self, getter_function, setter_function,
                            axis_type, operation, direction, amount=.1):
        """
        Move or zoom axes or colormap by a specified amount.

        Parameters
        ----------
        getter_function : callable
            Function handle, e.g., `ax.get_xlim`
        setter_function : callable
            Function handle, e.g., `ax.set_xlim`
        axis_type : 'freq', 'dB', 'other'
            String that sets constraints on how axis/colormaps are moved and
            zoomed
            'freq' : zoom and move is applied according to the ratios of the
                     lower to upper axis limit, i.e., the change is smaller
                     on the lower limit.
            'dB' : Only the lower axis limit is changed when zooming.
            'other' : move and zoom without constraints
        operation : 'move', 'zoom'
            'move' to shift the section of the axis or colormap to the
            left/right (if setter_function is an x-axis)or up/down (if setter
            function is a y-axis or colorbar). 'zoom' to zoom in our out.
        direction : 'increase', 'decrease'
            'increase' to move up/right or zoom in. 'decrease' to move
            down/left or zoom out.
        amount : number
            amount to move or zoom in percent. E.g., `amount=.1` will move/zoom
            10 percent of the current axis/colormap range. The default is 0.1

        """

        # shift 10 percent of the current axis range
        lims = np.asarray(getter_function())
        dyn_range = np.diff(lims)
        shift = amount * dyn_range

        # distribute shift to the lower and upper bound of frequency axes
        if axis_type == 'freq':
            shift = np.array([lims[0] / lims[1] * shift,
                             (1 - lims[0] / lims[1]) * shift]).flatten()
        else:
            shift = np.tile(shift, 2)

        if operation == 'move':
            # reverse the sign
            if direction == 'decrease':
                shift *= -1

        elif operation == 'zoom':
            # reverse one sign for zooming in/out
            if direction == 'decrease':
                shift[0] *= -1
            else:
                shift[1] *= -1

            # dB axes only zoom at the lower end
            if axis_type == 'dB':
                shift = np.array([2 * shift[0], 0])
        else:
            raise ValueError(
                f"operation must be 'move' or 'zoom' but is {operation}")

        # get new limits
        lims_new = lims + shift

        # apply limits
        setter_function(lims_new[0], lims_new[1])

        self.figure.canvas.draw()

    def toggle_all_lines(self):
        if self.all_visible:
            for i in range(len(self.ax.lines)):
                self.ax.lines[i].set_visible(False)
            self.current_line.set_visible(True)
            self.all_visible = False
        else:
            for i in range(len(self.ax.lines)):
                self.ax.lines[i].set_visible(True)
            self.all_visible = True
        self.figure.canvas.draw()

    def init_cycler(self, index=0, all_visible=True):
        if self.params._cycler_type == 'line':
            self.cycler = Cycle(self.ax.lines, index)
        elif self.params._cycler_type == 'signal':
            self.cycler = Cycle(self.signal, index)
        else:
            self.cycler = None
        if self.cycler is not None:
            self.cycler_data = self.cycler.current()
            self.all_visible = all_visible

    def cycle_lines(self, event):
        if self.all_visible or event.key == 'redraw':
            for i in range(len(self.ax.lines)):
                self.ax.lines[i].set_visible(False)
        else:
            self.current_line.set_visible(False)
        if event.key == self.ctr["next"]:
            self.current_line = self.cycler.next()
        elif event.key == self.ctr["prev"]:
            self.current_line = self.cycler.previous()
        self.current_line.set_visible(True)
        self.all_visible = False
        self.figure.canvas.draw()

    def update_cycler(self):
        self.cycler.data = self.ax.lines
        self.current_line = self.cycler.current()

    def connect(self):
        """Connect to Matplotlib figure."""
        self.figure.AxisModifier = self
        self.mpl_id = self.figure.canvas.mpl_connect(
            'key_press_event', self.select_action)

    def disconnect(self):
        """Disconnect from Matplotlib figure."""
        self.figure.canvas.mpl_disconnect(self.mpl_id)
