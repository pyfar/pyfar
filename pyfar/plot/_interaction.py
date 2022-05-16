"""Private class for managing interactive plots through keyboard shortcuts.

It is possible to interact with pyfar plots, i.e., to zoom an axis, through
keyboard shortcuts. Keyboard shortcuts are managed and documented by
`pyfar.plot.utils.shortcuts`

How this works
--------------

The interaction is managed by three classes:
1. `Cycle` stores the currently active channel and cycles channels. This is
   only used internally.
2. `PlotParameter` stores parameters for all plots. An object of this class
    must be created inside each plot to enable interaction.
3. `Interaction` is the main class that handles the interaction. An object of
   this class must also be created inside each plot to enable interaction. The
   object is added to the axes mostly for debugging purposes. Therefore this
   is not documented in the docstring of the plot functions.

What to do when changing plot functions?
----------------------------------------

If you change or add parameters of a plot function you have to do at least two
changes:
1. Make sure that the `PlotParameter` object created in that plot function is
   up to date.
2. Make sure that the call of that plot function in `Interaction.toggle_plot()`
   is up to data.

How to debug interaction
------------------------

Debugging is a bit tricky because you do not see errors in the terminal. It can
be done this way

>>> # %% plot something
>>> import pyfar as pf
>>> import numpy as np
>>> from pyfar.plot._interaction import EventEmu as EventEmu
>>> %matplotlib qt
>>>
>>> sig = pf.Signal(np.random.normal(0, 1, 2**16), 48e3)
>>> ax = pf.plot.time(sig)
>>>
>>> # %% debug this cell
>>> # EventEmu can be used to simulate pressing a key
>>> ax[0].interaction.select_action(EventEmu('Y'))

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyfar.plot import utils
from pyfar.plot import _line
from pyfar.plot import _two_d
from pyfar.plot import _utils


class Cycle(object):
    """ Cycle class implementation inspired by itertools.cycle. Supports
    circular iterations into two directions.
    """
    def __init__(self, cshape, index=0):
        """
        Parameters
        ----------
        cshape : tupel, array like, int
            cshape of the signal
        index : int, optional
            index of the current channel. The default is 0
        """
        self._n_channels = np.prod(cshape)
        self._index = index

        self._channels = []
        for index in np.ndindex(cshape):
            self._channels.append(index)

    def increase_index(self):
        self._index = (self.index + 1) % self.n_channels

    def decrease_index(self):
        index = self.index - 1
        if index < 0:
            index = self.n_channels + index
        self._index = index

    @property
    def index(self):
        return int(self._index)

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def current_channel(self):
        channel = self._channels[self.index]
        if len(channel) == 1:
            channel = channel[0]
        return channel


class EventEmu(object):
    def __init__(self, key):
        """
        Helper class to emulate events. This makes it possible to call
        functions in Interaction without user input and can be helpfull for
        debugging.

        Parameters
        ----------
        key : str, list
            name of key or list of keys to emulate a key event (e.g., 'left'
            for left arrow). If a list is passed, it is assumed that the keys
            all trigger the same event and only `key[0]` ist used
        """
        if isinstance(key, str):
            self.key = key
        elif isinstance(key, list):
            self.key = key[0]


class PlotParameter(object):
    """Store and change plot parameter.

    This class stores all unique input parameters  and axis types that control
    the plot look. The plot parameters are changed upon request from
    Interaction. The signal and axes are stored in Interaction. See
    `self.update` for more information.

    """
    def __init__(self, plot,
                 dB_time=False, dB_freq=True,              # dB properties
                 log_prefix_time=20, log_prefix_freq=20,
                 log_reference=1,                          # same for time/freq
                 xscale='log', yscale='linear',            # axis scaling
                 deg=False, unwrap=False,                  # phase properties
                 unit_time='s',                            # time axis unit
                 unit_gd='s',                              # group delay unit
                 window='hann', window_length=1014,        # spectrogram
                 window_overlap_fct=.5,
                 colorbar=True,
                 orientation='vertical', indices=None,     # 2D plots
                 method='pcolormesh'):

        # set plot type
        self._plot_type = ['line', '2d']
        if plot == "spectrogram" or plot.endswith("_2d"):
            self._plot_type = np.roll(self._plot_type, -1)

        # store input
        self.dB_time = dB_time
        self.dB_freq = dB_freq
        self.log_prefix_time = log_prefix_time
        self.log_prefix_freq = log_prefix_freq
        self.log_reference = log_reference
        self.xscale = xscale
        self.yscale = yscale
        self.deg = deg
        self.unwrap = unwrap
        self.unit_time = unit_time
        self.unit_gd = unit_gd
        self.window = window
        self.window_length = window_length
        self.window_overlap_fct = window_overlap_fct
        self.colorbar = colorbar
        self.orientation = orientation
        self.indices = indices
        self.method = method

        # set axis types based on `plot`
        self.update(plot)

    def update(self, plot):
        """Set plot type and axis types based on plot.

        A plot can have two or three axis. All plots have x- and y-axis. Some
        plots also have a colormap (cm, which we see as an axis here). Each
        axis can have one or multiple display options, e.g., a log or linear
        frequency axis. These options are set by this function using four
        variables:

        self._*_type : None, list
            The axis type as defined in `get_new_axis_limits`
        self._*_param : str, only if `len(self._*_type) > 1`
            The name of the plot parameter that changes if `self._*_type`
            changes
        self._*_values : str, only if `len(self._*_type) > 1`
            The values of `self._*_param`. Must have the same length as
            `self._*_type`
        self._*_id : int, only if `self._*_type is not None`
            An id that sets the current state of `self._*_type`

        * can be 'x', 'y', or 'cm'.

        Additional parameters that are set:

        self._cycler_type : 'line', 'signal', None
            determines in which way `Cycle` cycles through the channels.
            'line'   - cycling is done by toggling line visibility
                       (e.g. for pyfar.plot.line.time)
            'signal' - cycling is done by re-calling the plot function with a
                       signal slice (e.g. for pyfar.plot.line.spectrogram)
            None     - cycling is not possible
        self._plot : str
            name of the plot

        Parameters
        ----------
        plot : str
            Defines the plot by module.plot_function, e.g., 'freq'

        """

        # strip "2d" from plot name
        self._plot = plot[:-3] if plot.endswith("_2d") else plot
        self.plot = plot

        # set the axis, color map, and cycle, parameter for each plot
        if plot == 'time':
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

        elif plot == 'time_2d':
            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            self._x_param = None
            self._x_values = None
            # y-axis
            self._y_type = ['other']
            self._y_id = 0
            self._y_param = None
            self._y_values = None
            # color map
            self._cm_type = ['other', 'dB']
            self._cm_param = 'dB_time'
            self._cm_values = [False, True]
            self._cm_id = self._cm_values.index(getattr(self, self._cm_param))
            # cycler type
            self._cycler_type = None

        elif plot == 'freq':
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

        elif plot == 'freq_2d':
            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            self._x_param = None
            self._x_values = None
            # y-axis
            self._y_type = ['freq', 'other']
            self._y_param = 'xscale'
            self._y_values = ['log', 'linear']
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = ['other', 'dB']
            self._cm_param = 'dB_freq'
            self._cm_values = [False, True]
            self._cm_id = self._cm_values.index(getattr(self, self._cm_param))
            # cycler type
            self._cycler_type = None

        elif plot == 'phase':
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

        elif plot == 'phase_2d':
            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            self._x_param = None
            self._x_values = None
            # y-axis
            self._y_type = ['freq', 'other']
            self._y_param = 'xscale'
            self._y_values = ['log', 'linear']
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = ['other', 'other', 'other']
            self._cm_id = 0
            self._cm_param = 'unwrap'
            self._cm_values = [True, False, "360"]
            self._cm_id = self._cm_values.index(getattr(self, self._cm_param))
            # cycler type
            self._cycler_type = None

        elif plot == 'group_delay':
            # x-axis
            self._x_type = ['freq', 'other']
            self._x_param = 'xscale'
            self._x_values = ['log', 'linear']
            self._x_id = self._x_values.index(getattr(self, self._x_param))
            # y-axis
            self._y_type = ['other', 'other', 'other', 'other', 'other']
            self._y_param = 'unit_gd'
            self._y_values = ['auto', 's', 'ms', 'mus', 'samples']
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = None
            self._cm_id = None
            # cycler type
            self._cycler_type = 'line'

        elif plot == 'group_delay_2d':
            # x-axis
            self._x_type = ['other']
            self._x_id = 0
            self._x_param = None
            self._x_values = None
            # y-axis
            self._y_type = ['freq', 'other']
            self._y_param = 'xscale'
            self._y_values = ['log', 'linear']
            self._y_id = self._y_values.index(getattr(self, self._y_param))
            # color map
            self._cm_type = ['other', 'other', 'other', 'other', 'other']
            self._cm_param = 'unit_gd'
            self._cm_values = ['auto', 's', 'ms', 'mus', 'samples']
            self._cm_id = self._cm_values.index(getattr(self, self._cm_param))
            # cycler type
            self._cycler_type = None

        elif plot == 'spectrogram':
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

        elif plot == 'time_freq':
            # same as time
            # (currently interaction uses only the axis of the top plot)
            self.update("time")
            self._plot = "time_freq"
            return

        elif plot == 'time_freq_2d':
            # same as time_2d
            # (currently interaction uses only the axis of the top plot)
            self.update("time_2d")
            self._plot = "time_freq"
            return

        elif plot == 'freq_phase':
            # same as freq
            # (currently interaction uses only the axis of the top plot)
            self.update("freq")
            self._plot = "freq_phase"
            return

        elif plot == 'freq_phase_2d':
            # same as freq_2d
            # (currently interaction uses only the axis of the top plot)
            self.update("freq_2d")
            self._plot = "freq_phase"
            return

        elif plot == 'freq_group_delay':
            # same as freq
            # (currently interaction uses only the axis of the top plot)
            self.update("freq")
            self._plot = "freq_group_delay"
            return

        elif plot == 'freq_group_delay_2d':
            # same as freq_2d
            # (currently interaction uses only the axis of the top plot)
            self.update("freq_2d")
            self._plot = "freq_group_delay"
            return

        else:
            raise ValueError(f"{plot} not known.")

        # toggle plot parameter (switch x and y axis)
        if self.orientation == "horizontal" and self.plot_type == "2d":
            for attr in ["type", "id", "param", "values"]:
                tmp = getattr(self, f"_x_{attr}")
                setattr(self, f"_x_{attr}", getattr(self, f"_y_{attr}"))
                setattr(self, f"_y_{attr}", tmp)

    def toggle_x(self):
        """Toggle the x-axis type.

        For example toggle between lin and log frequency axis."""
        changed = False
        if self.x_type is not None:
            if len(self._x_type) > 1:
                self._x_id = (self._x_id + 1) % len(self._x_type)
                setattr(self, self._x_param, self._x_values[self._x_id])
                changed = True

        return changed

    def toggle_y(self):
        """Toggle the y-axis type.

        For example toggle between showing lin and log time signals."""
        changed = False
        if self.y_type is not None:
            if len(self._y_type) > 1:
                self._y_id = (self._y_id + 1) % len(self._y_type)
                setattr(self, self._y_param, self._y_values[self._y_id])
                changed = True

        return changed

    def toggle_colormap(self):
        """Toggle the color map type.

        For example toggle between showing lin and log magnitude."""
        changed = False
        if self.cm_type is not None:
            if len(self._cm_type) > 1:
                self._cm_id = (self._cm_id + 1) % len(self._cm_type)
                setattr(self, self._cm_param, self._cm_values[self._cm_id])
                changed = True

        return changed

    def toggle_orientation(self):
        """Toggle the orientation of 2D plots"""
        self.orientation = "horizontal" if self.orientation == "vertical" \
            else "vertical"

    def cycle_plot_types(self):
        """Cycle the plot types"""
        self._plot_type = np.roll(self._plot_type, -1)

    @property
    def plot_type(self):
        """Return current the plot type"""
        return self._plot_type[0]

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
    Toggle between plots; move or zoom axis or color map; toggle axis types;
    cycle channels.
    """
    def __init__(self, signal, axes, colorbars, style, plot_parameter,
                 **kwargs):
        """
        Change the plot and plot parameters based on keyboard shortcuts.

        Parameters
        ----------
        signal : Signal
            audio data
        axes : Matplotlib axes, array like
            axes objects of all axes in the plot holding data
        colorbars : Matplotlib colorbar, array like
            all colorbar objects in the plot
        style : plot style
            E.g. 'light'
        plot_parameter : PlotParameter
            An object of the PlotParameter class

        **kwargs are passed to the plot functions
        """

        # save input arguments
        self.cshape = signal.cshape
        self.signal = signal.flatten()
        self.ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
        self.all_axes = axes
        self.all_bars = colorbars
        self.figure = self.ax.figure
        self.style = style
        self.params = plot_parameter
        if self.params.plot_type == "line":
            self.kwargs_line = kwargs
            self.kwargs_2d = {}
        if self.params.plot_type == "2d":
            self.kwargs_line = {}
            self.kwargs_2d = kwargs

        # store last key event (done in self.select_action)
        self.event = None

        # initialize cycler
        self.cycler = Cycle(self.cshape)

        # initialize visibility
        self.all_visible = True

        # set initial value for text object showing the channel number
        self.txt = None

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
        toggle_plot = False
        for plot in self.plot:
            if event.key in self.plot[plot]:
                toggle_plot = True
                break

        if toggle_plot:

            self.toggle_plot(event)

        # toggle plot type
        elif event.key in ctr["cycle_plot_types"]:

            # no toggling for spectrogram
            if self.params._plot == "spectrogram":
                return
            # no toggling if signal has less than 2 channels
            if np.prod(self.signal.cshape) < 2 \
                    and self.params.plot_type == "line":
                return

            # cycle the plot type
            self.params.cycle_plot_types()
            # emulate key event and toggle plot
            event_emu = EventEmu(self.plot[self.params._plot][0])
            self.toggle_plot(event_emu)

        # toggle orientation
        elif event.key in ctr["toggle_orientation"] \
                and self.params.plot_type == "2d":

            # toggle the orientation
            self.params.toggle_orientation()
            # emulate key event and toggle plot
            event_emu = EventEmu(self.plot[self.params._plot][0])
            self.toggle_plot(event_emu)

        # x-axis move/zoom
        elif event.key in ctr["move_left"] + ctr["move_right"] + \
                ctr["zoom_x_in"] + ctr["zoom_x_out"]:
            self.move_and_zoom(event, 'x')

        # y-axis move/zoom
        elif event.key in ctr["move_up"] + ctr["move_down"] + \
                ctr["zoom_y_in"] + ctr["zoom_y_out"]:
            self.move_and_zoom(event, 'y')

        # color map move/zoom
        elif event.key in ctr["move_cm_up"] + ctr["move_cm_down"] + \
                ctr["zoom_cm_in"] + ctr["zoom_cm_out"]:
            self.move_and_zoom(event, 'cm')

        # x-axis toggle
        elif event.key in ctr["toggle_x"]:
            changed = self.params.toggle_x()
            if changed:
                self.toggle_plot(EventEmu(self.plot[self.params._plot]))

        # y-axis toggle
        elif event.key in ctr["toggle_y"]:
            changed = self.params.toggle_y()
            if changed:
                self.toggle_plot(EventEmu(self.plot[self.params._plot]))

        # color map toggle
        elif event.key in ctr["toggle_cm"]:
            changed = self.params.toggle_colormap()
            if changed:
                self.toggle_plot(EventEmu(self.plot[self.params._plot]))

        # toggle line visibility
        elif event.key in ctr["toggle_all"]:
            if self.params._cycler_type == 'line' \
                    and self.cycler.n_channels > 1:
                self.toggle_all_lines()

        # cycle channels
        elif event.key in ctr["next"] + ctr["prev"]:
            if self.cycler.n_channels > 1:
                self.cycle(event)

        # clear/write channel info
        if not self.all_visible or self.params._cycler_type == 'signal':
            self.write_current_channel_text()
        else:
            self.delete_current_channel_text()

    def toggle_plot(self, event):
        """Toggle between plot types."""

        plot = self.plot
        prm = self.params

        # cases that are not allowed
        # spectogram plot if signal has less samples than the window length
        if event.key in plot['spectrogram'] \
                and self.signal.n_samples < prm.window_length:
            return

        # prepare for toggling
        with plt.style.context(utils.plotstyle(self.style)):
            self.figure.clear()
            # This saves the axis used for interaction
            self.ax = None
            # This saves all axes and colorbars
            self.all_axes = None
            self.all_bars = None

            # Toggle:
            # 1. select plot
            # 2. select plot_type
            # 3. update self.params (PlotParameter instance)
            # 4. plot and update all current axes and colorbars (None by
            #    default) and self.ax (axes used for interaction)
            if event.key in plot['time']:
                if self.params.plot_type == "line":
                    self.params.update('time')
                    self.all_axes = self.ax = _line._time(
                        self.signal, prm.dB_time, prm.log_prefix_time,
                        prm.log_reference, prm.unit_time, self.ax,
                        **self.kwargs_line)
                elif self.params.plot_type == "2d":
                    self.params.update('time_2d')
                    self.all_axes, _, self.all_bars = _two_d._time_2d(
                        self.signal, prm.dB_time, prm.log_prefix_time,
                        prm.log_reference, prm.unit_time, prm.indices,
                        prm.orientation, prm.method, prm.colorbar,
                        self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes

            elif event.key in plot['freq']:
                if self.params.plot_type == "line":
                    self.params.update('freq')
                    self.all_axes = self.ax = _line._freq(
                        self.signal, prm.dB_freq, prm.log_prefix_freq,
                        prm.log_reference, prm.xscale, self.ax,
                        **self.kwargs_line)
                elif self.params.plot_type == "2d":
                    self.params.update('freq_2d')
                    self.all_axes, _, self.all_bars = _two_d._freq_2d(
                        self.signal, prm.dB_freq, prm.log_prefix_freq,
                        prm.log_reference, prm.xscale, prm.indices,
                        prm.orientation, prm.method, prm.colorbar,
                        self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes

            elif event.key in plot['phase']:
                if self.params.plot_type == "line":
                    self.params.update('phase')
                    self.all_axes = self.ax = _line._phase(
                        self.signal, prm.deg, prm.unwrap, prm.xscale,
                        self.ax, **self.kwargs_line)
                if self.params.plot_type == "2d":
                    self.params.update('phase_2d')
                    self.all_axes, _, self.all_bars = _two_d._phase_2d(
                        self.signal, prm.deg, prm.unwrap, prm.xscale,
                        prm.indices, prm.orientation, prm.method,
                        prm.colorbar, self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes

            elif event.key in plot['group_delay']:
                if self.params.plot_type == "line":
                    self.params.update('group_delay')
                    self.all_axes = self.ax = _line._group_delay(
                        self.signal, prm.unit_gd, prm.xscale, self.ax,
                        **self.kwargs_line)
                if self.params.plot_type == "2d":
                    self.params.update('group_delay_2d')
                    self.all_axes, _, self.all_bars = _two_d._group_delay_2d(
                        self.signal, prm.unit_gd, prm.xscale, prm.indices,
                        prm.orientation, prm.method, prm.colorbar,
                        self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes

            elif event.key in plot['spectrogram']:
                self.params.update('spectrogram')
                self.all_axes, _, self.all_bars = _two_d._spectrogram(
                    self.signal[self.cycler.index], prm.dB_freq,
                    prm.log_prefix_freq, prm.log_reference, prm.yscale,
                    prm.unit_time, prm.window, prm.window_length,
                    prm.window_overlap_fct, prm.colorbar, self.ax,
                    **self.kwargs_2d)
                self.ax = self.all_axes

            elif event.key in plot['time_freq']:
                if self.params.plot_type == "line":
                    self.params.update('time_freq')
                    self.all_axes = _line._time_freq(
                        self.signal, prm.dB_time, prm.dB_freq,
                        prm.log_prefix_time, prm.log_prefix_freq,
                        prm.log_reference, prm.xscale, prm.unit_time, self.ax,
                        **self.kwargs_line)
                    self.ax = self.all_axes[0]
                elif self.params.plot_type == "2d":
                    self.params.update('time_freq_2d')
                    self.all_axes, _, self.all_bars = _two_d._time_freq_2d(
                        self.signal, prm.dB_time, prm.dB_freq,
                        prm.log_prefix_time, prm.log_prefix_freq,
                        prm.log_reference, prm.xscale, prm.unit_time,
                        prm.indices, prm.orientation, prm.method,
                        prm.colorbar, self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes[0]

            elif event.key in plot['freq_phase']:
                if self.params.plot_type == "line":
                    self.params.update('freq_phase')
                    self.all_axes = _line._freq_phase(
                        self.signal, prm.dB_freq, prm.log_prefix_freq,
                        prm.log_reference, prm.xscale, prm.deg, prm.unwrap,
                        self.ax, **self.kwargs_line)
                    self.ax = self.all_axes[0]
                elif self.params.plot_type == "2d":
                    self.params.update('freq_phase_2d')
                    self.all_axes, _, self.all_bars = _two_d._freq_phase_2d(
                        self.signal, prm.dB_freq, prm.log_prefix_freq,
                        prm.log_reference, prm.xscale, prm.deg, prm.unwrap,
                        prm.indices, prm.orientation, prm.method,
                        prm.colorbar, self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes[0]

            elif event.key in plot['freq_group_delay']:
                if self.params.plot_type == "line":
                    self.params.update('freq_group_delay')
                    self.all_axes = _line._freq_group_delay(
                        self.signal, prm.dB_freq, prm.log_prefix_freq,
                        prm.log_reference, prm.unit_gd, prm.xscale,
                        self.ax, **self.kwargs_line)
                    self.ax = self.all_axes[0]
                if self.params.plot_type == "2d":
                    self.params.update('freq_group_delay_2d')
                    self.all_axes, _, self.all_bars = \
                        _two_d._freq_group_delay_2d(
                            self.signal, prm.dB_freq, prm.log_prefix_freq,
                            prm.log_reference, prm.unit_gd, prm.xscale,
                            prm.indices, prm.orientation, prm.method,
                            prm.colorbar, self.ax, **self.kwargs_2d)
                    self.ax = self.all_axes[0]

            # update figure
            if self.params._cycler_type == 'line':
                if not self.all_visible:
                    self.cycle(EventEmu('redraw'))
                else:
                    self.draw_canvas()
            else:
                self.draw_canvas()

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
            if event.key in ctr["move_left"] + ctr["move_right"]:
                operation = "move"
            else:
                operation = "zoom"
            if event.key in ctr["move_right"] + ctr["zoom_x_in"]:
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
            if event.key in ctr["move_up"] + ctr["move_down"]:
                operation = "move"
            else:
                operation = "zoom"
            if event.key in ctr["move_up"] + ctr["zoom_y_in"]:
                direction = "increase"
            else:
                direction = "decrease"

        # move/zoom colorbar
        elif axis == "cm":

            if self.params.cm_type is None:
                return

            qm = _utils._get_quad_mesh_from_axis(self.ax)

            getter = qm.get_clim
            setter = qm.set_clim
            axis_type = self.params.cm_type
            if event.key in ctr["move_cm_up"] + ctr["move_cm_down"]:
                operation = "move"
            else:
                operation = "zoom"
            if event.key in ctr["move_cm_up"] + ctr["zoom_cm_in"]:
                direction = "increase"
            else:
                direction = "decrease"

        if getter is not None:
            # get the new axis limits
            current_limits = np.asarray(getter())
            new_limits = get_new_axis_limits(
                current_limits, axis_type, operation, direction)

            # apply the new axis limits
            setter(new_limits[0], new_limits[1])
            self.draw_canvas()

    def toggle_all_lines(self):
        if self.all_visible:
            for i in range(len(self.ax.lines)):
                self.ax.lines[i].set_visible(False)
            self.ax.lines[self.cycler.index].set_visible(True)
            self.all_visible = False
        else:
            for i in range(len(self.ax.lines)):
                self.ax.lines[i].set_visible(True)
            self.all_visible = True

        self.draw_canvas()

    def cycle(self, event):
        if self.params._cycler_type == 'line':
            self.cycle_lines(event)
        elif self.params._cycler_type == 'signal':
            self.cycle_signals(event)

    def cycle_lines(self, event):
        # set visible lines invisible
        if self.all_visible or event.key == 'redraw':
            for i in range(len(self.ax.lines)):
                self.ax.lines[i].set_visible(False)
        else:
            self.ax.lines[self.cycler.index].set_visible(False)

        # cycle
        if event.key in self.ctr["next"]:
            self.cycler.increase_index()
        elif event.key in self.ctr["prev"]:
            self.cycler.decrease_index()

        # set current line visible
        self.ax.lines[self.cycler.index].set_visible(True)
        self.all_visible = False
        self.draw_canvas()

    def cycle_signals(self, event):
        # cycle index
        if event.key in self.ctr["next"]:
            self.cycler.increase_index()
        elif event.key in self.ctr["prev"]:
            self.cycler.decrease_index()

        # re-plot
        self.all_visible = False
        self.toggle_plot(EventEmu(self.plot['spectrogram']))

    def write_current_channel_text(self):

        # clear old text
        self.delete_current_channel_text()

        # position for new text (relative to axis)
        x_pos = .98
        y_pos = .02

        # write new text
        with plt.style.context(utils.plotstyle(self.style)):
            bbox = dict(boxstyle="round", fc=mpl.rcParams["axes.facecolor"],
                        ec=mpl.rcParams["axes.facecolor"], alpha=.5)

            self.txt = self.ax.text(
                x_pos, y_pos, f'Ch. {self.cycler.current_channel}',
                horizontalalignment='right', verticalalignment='baseline',
                bbox=bbox, transform=self.ax.transAxes)

            self.draw_canvas()

    def delete_current_channel_text(self):
        if self.txt is not None:
            self.txt.remove()
            self.txt = None
            self.draw_canvas()

    def draw_canvas(self):
        with plt.style.context(utils.plotstyle(self.style)):
            self.figure.canvas.draw()

    def connect(self):
        """Connect to Matplotlib figure."""
        self.figure.AxisModifier = self
        self.mpl_id = self.figure.canvas.mpl_connect(
            'key_press_event', self.select_action)

    def disconnect(self):
        """Disconnect from Matplotlib figure."""
        self.figure.canvas.mpl_disconnect(self.mpl_id)


def get_new_axis_limits(limits, axis_type, operation, direction, amount=.1):
    """
    Get new limits for plot axis.

    Parameters
    ----------
    limits : array like
        array like of length two with the current lower and upper axis limits.
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

    # get the amount to be shifted
    dyn_range = np.diff(np.array(limits))
    shift = amount * dyn_range

    # distribute shift to the lower and upper bound of frequency axes
    if axis_type == 'freq':
        shift = np.array([limits[0] / limits[1] * shift,
                         (1 - limits[0] / limits[1]) * shift]).flatten()
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
    new_limits = limits + shift

    return new_limits
