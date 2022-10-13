# %%
"""Test the interaction module from pyfar.plot.

Assertions are not tested because this is a private module. Test of the plots
is done using properties of the plots to avoid time intensive saving, reading,
and comparing of images.

Note that it is not tested if the properties for
pyfar.plot._interaction.PlotParameter are correctly set in pyfar.plot.function.
If the parameters are incorrect, interaction will behave incorrect.

*******************************************************************************
NOTE: These tests might fail in case tests that are conducted before use
      plotting without closing the created figures. Make sure that you always
      use matplotlib.pyplot.close("all") after creating tests with plots.
*******************************************************************************
"""
import numpy.testing as npt
import pytest
from inspect import getmembers, isfunction
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyfar as pf
import pyfar.plot._interaction as ia
from pyfar.plot._utils import _get_quad_mesh_from_axis
from pyfar.testing.plot_utils import create_figure

# use non showing backend for speed
mpl.use("Agg")
plt.close("all")

# get plot controls for universal testing
sc_plot = pf.plot.shortcuts(show=False)["plots"]
sc_ctr = pf.plot.shortcuts(show=False)["controls"]

# Interaction is tested by comparing axis and colobar labels. These properties
# must be defined here for each plot. Some test might fail if new plot
# functions are added to pyfar with out added their properties below.
# NOTE: Some labels depend on the lengths of the signals. The labels were
#       generated for a signal with 1024 samples (minimum length for
#       spectrogram) and a maximum group delay of 1000 samples @ 44.1 kHz
plots = {
    # line plots
    'time': {
        'shortcut': sc_plot["time"]["key"][0],
        'xlabel': ['Time in s'],
        'ylabel': ['Amplitude'],
        'cblabel': [None]},
    'freq': {
        'shortcut': sc_plot["freq"]["key"][0],
        'xlabel': ['Frequency in Hz'],
        'ylabel': ['Magnitude in dB'],
        'cblabel': [None]},
    'phase': {
        'shortcut': sc_plot["phase"]["key"][0],
        'xlabel': ['Frequency in Hz'],
        'ylabel': ['Phase in radians'],
        'cblabel': [None]},
    'group_delay': {
        'shortcut': sc_plot["group_delay"]["key"][0],
        'xlabel': ['Frequency in Hz'],
        'ylabel': ['Group delay in s'],
        'cblabel': [None]},
    'time_freq': {
        'shortcut': sc_plot["time_freq"]["key"][0],
        'xlabel': ['Time in s', 'Frequency in Hz'],
        'ylabel': ['Amplitude', 'Magnitude in dB'],
        'cblabel': [None]},
    'freq_phase': {
        'shortcut': sc_plot["freq_phase"]["key"][0],
        'xlabel':  ['Frequency in Hz', 'Frequency in Hz'],
        'ylabel': ['Magnitude in dB', 'Phase in radians'],
        'cblabel': [None]},
    'freq_group_delay': {
        'shortcut': sc_plot["freq_group_delay"]["key"][0],
        'xlabel': ['Frequency in Hz', 'Frequency in Hz'],
        'ylabel': ['Magnitude in dB', 'Group delay in s'],
        'cblabel': [None]},
    # 2D plots
    'time_2d': {
        'shortcut': sc_plot["time"]["key"][0],
        'xlabel': ['Indices'],
        'ylabel': ['Time in s'],
        'cblabel': ['Amplitude']},
    'freq_2d': {
        'shortcut': sc_plot["freq"]["key"][0],
        'xlabel': ['Indices'],
        'ylabel': ['Frequency in Hz'],
        'cblabel': ['Magnitude in dB']},
    'phase_2d': {
        'shortcut': sc_plot["phase"]["key"][0],
        'xlabel': ['Indices'],
        'ylabel': ['Frequency in Hz'],
        'cblabel': ['Phase in radians']},
    'group_delay_2d': {
        'shortcut': sc_plot["group_delay"]["key"][0],
        'xlabel': ['Indices'],
        'ylabel': ['Frequency in Hz'],
        'cblabel': ['Group delay in s']},
    'spectrogram': {
        'shortcut': sc_plot["spectrogram"]["key"][0],
        'xlabel': ['Time in s'],
        'ylabel': ['Frequency in Hz'],
        'cblabel': ['Magnitude in dB']},
    'time_freq_2d': {
        'shortcut': sc_plot["time_freq"]["key"][0],
        'xlabel': ['Indices', 'Indices'],
        'ylabel': ['Time in s', 'Frequency in Hz'],
        'cblabel': ['Amplitude', 'Magnitude in dB']},
    'freq_phase_2d': {
        'shortcut': sc_plot["freq_phase"]["key"][0],
        'xlabel': ['Indices', 'Indices'],
        'ylabel': ['Frequency in Hz', 'Frequency in Hz'],
        'cblabel': ['Magnitude in dB', 'Phase in radians']},
    'freq_group_delay_2d': {
        'shortcut': sc_plot["freq_group_delay"]["key"][0],
        'xlabel': ['Indices', 'Indices'],
        'ylabel': ['Frequency in Hz', 'Frequency in Hz'],
        'cblabel': ['Magnitude in dB', 'Group delay in s']}
    }


def test_event_emu():
    """Test the EventEmu Class."""
    event = ia.EventEmu('a')
    assert isinstance(event, ia.EventEmu)
    assert event.key == 'a'

    event = ia.EventEmu(['a'])
    assert isinstance(event, ia.EventEmu)
    assert event.key == 'a'


def test_interaction_attached():
    """Test if interaction object is attached to each plot function.

    This test will fail if a new plot function is added to the pyfar.plot that
    does not have an interaction. This is intended behavior.
    """

    # dummy signal (needs to as longe as the default spectrogram block size
    # with at least two channels)
    signal = pf.signals.impulse(1024, [0, 0])

    # Use create figure to specify the plot backend
    create_figure()

    # loop functions
    for function in getmembers(pf.plot.line, isfunction) + \
            getmembers(pf.plot.two_d, isfunction):
        # exclude functions that do not support interaction
        if function[0] in ["context", "custom_subplots"]:
            continue

        ax = function[1](signal)
        # axis is first return parameter if function returns multiple
        ax = ax[0] if isinstance(ax, (tuple)) else ax
        # interaction axis is first axis if functions returns multiple
        ax = ax[0] if isinstance(ax, (np.ndarray, list)) else ax
        # assert and close figure
        assert isinstance(ax.interaction, ia.Interaction)
        plt.close()


@pytest.mark.parametrize("plot_type,initial_function,function_list", [
    ("line", pf.plot.time, getmembers(pf.plot.line, isfunction)),
    ("2d", pf.plot.time_2d, getmembers(pf.plot.two_d, isfunction))
])
def test_toggle_plots(plot_type, initial_function, function_list):
    """Test toggling plots by checking labels after toggling."""

    # dummy signal (needs to as longe as the default spectrogram block size
    # with at least two channels)
    signal = pf.signals.impulse(1024, [0, 1000])

    # Use create figure to specify the plot backend
    create_figure()

    # initital plot
    ax = initial_function(signal)
    # get the axis containing the interaction instance
    if plot_type == "2d":
        ax = ax[0]
    # get the interaction
    interaction = ax[0].interaction if isinstance(ax, (list, np.ndarray)) \
        else ax.interaction

    for function in function_list:

        # exclude functions that do not support interaction
        if function[0] in ["context", "custom_subplots"]:
            continue

        print(f"testing: {function[0]}")

        # get current labels
        xlabel = plots[function[0]]["xlabel"]
        ylabel = plots[function[0]]["ylabel"]
        cblabel = plots[function[0]]["cblabel"]

        # toggle the interaction
        interaction.select_action(ia.EventEmu(plots[function[0]]["shortcut"]))

        # current axes and colorbars as array
        ca = np.atleast_1d(interaction.all_axes)
        cb = np.atleast_1d(interaction.all_bars)

        for idx in range(len(xlabel)):
            # test x- and y-label
            if idx == 0 and len(xlabel) > 1:
                assert ca[idx].get_xlabel() in (xlabel[idx], "")
            else:
                assert ca[idx].get_xlabel() == xlabel[idx]

            assert ca[idx].get_ylabel() == ylabel[idx]

            # test colorbar
            if plot_type == "2d":
                assert cb[idx].ax.get_ylabel() == cblabel[idx]

    plt.close("all")


@pytest.mark.parametrize("plot,signal,shortcut", [
    # togling spectrogram with short signal raises Value error if not caught
    (pf.plot.time, pf.Signal([1, 0, 0], 44100),
     sc_plot["spectrogram"]["key"][0]),
    # cycling plot style with 1 channel signal raises Value error if not caught
    (pf.plot.time, pf.Signal([1, 0, 0], 44100),
     sc_ctr["cycle_plot_types"]["key"][0]),
    ])
def test_interaction_not_allowed(plot, signal, shortcut):
    """
    Test interactions that are not allowed are caught.
    Can be extended if required in the future.
    """
    # Use create figure to specify the plot backend
    create_figure()

    # plot signal
    ax = plot(signal)
    # toggle the interaction would raise a ValueError if not caught
    ax.interaction.select_action(ia.EventEmu(shortcut))
    plt.close("all")


@pytest.mark.parametrize(
    "ax_type,operation,direction,limits,new_limits",
    [("freq", "move", "increase", [20, 20e3], [20 + 1.998, 20e3 + 1996.002]),
     ("freq", "move", "decrease", [20, 20e3], [20 - 1.998, 20e3 - 1996.002]),
     ("freq", "zoom", "increase", [20, 20e3], [20 + 1.998, 20e3 - 1996.002]),
     ("freq", "zoom", "decrease", [20, 20e3], [20 - 1.998, 20e3 + 1996.002]),
     ("dB", "move", "increase", [-20, 0], [-18, 2]),
     ("dB", "move", "decrease", [-20, 0], [-22, -2]),
     ("dB", "zoom", "increase", [-20, 0], [-16, 0]),
     ("dB", "zoom", "decrease", [-20, 0], [-24, 0]),
     ("other", "move", "increase", [0, 10], [1, 11]),
     ("other", "move", "decrease", [0, 10], [-1, 9]),
     ("other", "zoom", "increase", [0, 10], [1, 9]),
     ("other", "zoom", "decrease", [0, 10], [-1, 11])
     ])
def test_get_new_axis_limits(ax_type, operation, direction,
                             limits, new_limits):
    """Test calculation of new axis limits for all parameter combinations."""
    # Use create figure to specify the plot backend
    create_figure()

    test_limits = ia.get_new_axis_limits(
        limits, ax_type, operation, direction)

    npt.assert_allclose(test_limits, np.array(new_limits))


def test_move_and_zoom_linear():
    """Test moving and zooming linear x-axis, y-axis and colormap.

    This round trip test is only done for linear axis to save time. If the new
    axes limits are calculated correctly for all cases is tested separately.

    This is only done for one example plot. Other plots will work correctly
    if the pyfar.plot._interaction.PlotParameter are set correctly in
    pyfar.plot.function.
    """
    # Use create figure to specify the plot backend
    create_figure()

    # initialize the plot
    signal = pf.signals.impulse(1024)

    for axes in ['x', 'y', 'cm']:
        if axes == 'x':
            ax = pf.plot.time(signal, unit="samples")
            getter = ax.get_xlim

            move = [sc_ctr["move_right"]["key"][0],
                    sc_ctr["move_left"]["key"][0]]
            zoom = [sc_ctr["zoom_x_in"]["key"][0],
                    sc_ctr["zoom_x_out"]["key"][0]]
        if axes == 'y':
            ax = pf.plot.time(signal, unit="samples")
            getter = ax.get_ylim

            move = [sc_ctr["move_up"]["key"][0],
                    sc_ctr["move_down"]["key"][0]]
            zoom = [sc_ctr["zoom_y_in"]["key"][0],
                    sc_ctr["zoom_y_out"]["key"][0]]
        if axes == 'cm':
            ax, *_ = pf.plot.spectrogram(signal, dB=False)
            ax = ax[0]
            for cm in ax.get_children():
                if type(cm) == mpl.collections.QuadMesh:
                    break
            getter = cm.get_clim

            move = [sc_ctr["move_cm_up"]["key"][0],
                    sc_ctr["move_cm_down"]["key"][0]]
            zoom = [sc_ctr["zoom_cm_in"]["key"][0],
                    sc_ctr["zoom_cm_out"]["key"][0]]

        # starting limits and range for changing them
        lim = getter()
        shift = (lim[1] - lim[0]) * .1

        # move left and right
        ax.interaction.select_action(ia.EventEmu(move[0]))
        npt.assert_allclose(getter(), (lim[0] + shift, lim[1] + shift))
        ax.interaction.select_action(ia.EventEmu(move[1]))
        npt.assert_allclose(getter(), lim)

        # zoom in and out
        ax.interaction.select_action(ia.EventEmu(zoom[0]))
        npt.assert_allclose(getter(), (lim[0] + shift, lim[1] - shift))
        lim = getter()
        shift = (lim[1] - lim[0]) / 10
        ax.interaction.select_action(ia.EventEmu(zoom[1]))
        npt.assert_allclose(
            getter(), (lim[0] - shift, lim[1] + shift))

        plt.close()


def test_toggle_x_axis():
    """Test toggling the x-axis from logarithmic to linear and back.

    This is only tested for one example. Other plots will work correctly
    if the pyfar.plot._interaction.PlotParameter are set correctly in
    pyfar.plot.function.
    """
    # Use create figure to specify the plot backend
    create_figure()

    # init the plot
    ax = pf.plot.freq(pf.Signal([1, 2, 3, 4, 5], 44100))
    assert ax.get_xscale() == "log"
    # toggle x-axis
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_x"]["key"][0]))
    assert plt.gca().get_xscale() == "linear"
    # toggle x-axis
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_x"]["key"][0]))
    assert plt.gca().get_xscale() == "log"

    plt.close("all")


def test_toggle_y_axis():
    """Test toggling the y-axis from dB to linear and back.

    This is only tested for one example. Other plots will work correctly
    if the pyfar.plot._interaction.PlotParameter are set correctly in
    pyfar.plot.function.
    """
    # Use create figure to specify the plot backend
    create_figure()

    # init the plot
    ax = pf.plot.freq(pf.Signal([1, 2, 3, 4, 5], 44100))
    assert ax.get_ylabel() == "Magnitude in dB"
    # toggle x-axis
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_y"]["key"][0]))
    assert plt.gca().get_ylabel() == "Magnitude"
    # toggle x-axis
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_y"]["key"][0]))
    assert plt.gca().get_ylabel() == "Magnitude in dB"

    plt.close("all")


def test_toggle_colormap():
    """Test toggling the colormap from dB to linear and back.

    This is only tested for one example. Other plots will work correctly
    if the pyfar.plot._interaction.PlotParameter are set correctly in
    pyfar.plot.function.
    """
    # Use create figure to specify the plot backend
    create_figure()

    # init the plot
    ax, *_ = pf.plot.spectrogram(pf.signals.impulse(1024))
    assert ax[1].get_ylabel() == "Magnitude in dB"
    # toggle x-axis
    ax[0].interaction.select_action(ia.EventEmu(sc_ctr["toggle_cm"]["key"][0]))
    assert plt.gcf().get_axes()[1].get_ylabel() == "Magnitude"
    # toggle x-axis
    ax[0].interaction.select_action(ia.EventEmu(sc_ctr["toggle_cm"]["key"][0]))
    assert plt.gcf().get_axes()[1].get_ylabel() == "Magnitude in dB"

    plt.close("all")


def test_toggle_orientation_2d_plots():
    """
    Test if the orientation is toggled by comparing the axis labels before
    and after toggling.
    """
    # Use create figure to specify the plot backend
    create_figure()


    signal = pf.signals.impulse(1024, [0, 1000])
    key = sc_ctr["toggle_orientation"]["key"][0]

    for function in getmembers(pf.plot.two_d, isfunction):
        # exclude functions that do not support interaction
        if function[0] in ["context", "spectrogram"]:
            continue

        # plot
        print(f"testing: {function[0]}")
        ax, *_ = function[1](signal)
        # get the interaction
        interaction = ax[0].interaction if isinstance(ax, (list, np.ndarray)) \
            else ax.interaction

        # get current labels
        xlabel = plots[function[0]]["xlabel"]
        ylabel = plots[function[0]]["ylabel"]

        for nn in range(2):

            if nn:
                # toggle orientation
                print("toggling interaction")
                ax[0].interaction.select_action(ia.EventEmu(key))
                tmp = xlabel
                xlabel = ylabel
                ylabel = tmp

            # current axes as array
            ca = np.atleast_1d(interaction.all_axes)

            for mm in range(len(xlabel)):

                print(f"mm={mm}")

                # test x- and y-label
                if mm == 0 and len(xlabel) > 1:
                    assert ca[mm].get_xlabel() in (xlabel[mm], "")
                else:
                    assert ca[mm].get_xlabel() == xlabel[mm]

                assert ca[mm].get_ylabel() == ylabel[mm]

        plt.close()


def test_cycle_and_toggle_lines_1d_signal():
    """Test toggling and cycling channels of a one-dimensional Signal."""
    # Use create figure to specify the plot backend
    create_figure()

    # init and check start conditions
    signal = pf.Signal([[1, 0], [2, 0]], 44100)
    ax = pf.plot.time(signal)
    assert ax.lines[0].get_visible() is True
    assert ax.lines[1].get_visible() is True
    assert ax.interaction.txt is None
    # toggle all
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_all"]["key"][0]))
    assert ax.lines[0].get_visible() is True
    assert ax.lines[1].get_visible() is False
    assert ax.interaction.txt.get_text() == "Ch. 0"
    # next
    ax.interaction.select_action(ia.EventEmu(sc_ctr["next"]["key"][0]))
    assert ax.lines[0].get_visible() is False
    assert ax.lines[1].get_visible() is True
    assert ax.interaction.txt.get_text() == "Ch. 1"
    # previous
    ax.interaction.select_action(ia.EventEmu(sc_ctr["prev"]["key"][0]))
    assert ax.lines[0].get_visible() is True
    assert ax.lines[1].get_visible() is False
    assert ax.interaction.txt.get_text() == "Ch. 0"
    # toggle all
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_all"]["key"][0]))
    assert ax.lines[0].get_visible() is True
    assert ax.lines[1].get_visible() is True
    assert ax.interaction.txt is None

    plt.close("all")


def test_cycle_and_toggle_lines_2d_signal():
    """Test toggling and cycling channels of a two-dimensional Signal."""
    # Use create figure to specify the plot backend
    create_figure()

    # init and check start conditions
    signal = pf.signals.impulse(10, [[1, 2], [3, 4]])
    ax = pf.plot.time(signal)
    for line in [0, 1, 2, 3]:
        assert ax.lines[line].get_visible() is True
    assert ax.interaction.txt is None
    # toggle all
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_all"]["key"][0]))
    assert ax.lines[0].get_visible() is True
    for line in [1, 2, 3]:
        assert ax.lines[line].get_visible() is False
    assert ax.interaction.txt.get_text() == "Ch. (0, 0)"
    # next
    ax.interaction.select_action(ia.EventEmu(sc_ctr["next"]["key"][0]))
    for line in [0, 2, 3]:
        assert ax.lines[line].get_visible() is False
    assert ax.lines[1].get_visible() is True
    assert ax.interaction.txt.get_text() == "Ch. (0, 1)"
    # previous
    ax.interaction.select_action(ia.EventEmu(sc_ctr["prev"]["key"][0]))
    assert ax.lines[0].get_visible() is True
    for line in [1, 2, 3]:
        assert ax.lines[line].get_visible() is False
    assert ax.interaction.txt.get_text() == "Ch. (0, 0)"
    # toggle all
    ax.interaction.select_action(ia.EventEmu(sc_ctr["toggle_all"]["key"][0]))
    for line in [0, 1, 2, 3]:
        assert ax.lines[line].get_visible() is True
    assert ax.interaction.txt is None

    plt.close("all")


def test_cycle_and_toggle_signals():
    """Test toggling and cycling Signal slices."""
    # Use create figure to specify the plot backend
    create_figure()

    # init and check start conditions
    signal = pf.signals.impulse(1024, amplitude=[1, 2])
    ax, *_ = pf.plot.spectrogram(signal)

    assert ax[0].interaction.txt is None
    # use the clim because the image data is identical
    clim = _get_quad_mesh_from_axis(ax[0]).get_clim()
    npt.assert_allclose(clim, (-96, 4), atol=.5)

    # next
    ax[0].interaction.select_action(ia.EventEmu(sc_ctr["next"]["key"][0]))
    assert ax[0].interaction.txt.get_text() == "Ch. 1"
    clim = _get_quad_mesh_from_axis(plt.gcf().get_axes()[0]).get_clim()
    npt.assert_allclose(clim, (-90, 10), atol=.5)

    # previous
    # next
    ax[0].interaction.select_action(ia.EventEmu(sc_ctr["prev"]["key"][0]))
    assert ax[0].interaction.txt.get_text() == "Ch. 0"
    clim = _get_quad_mesh_from_axis(plt.gcf().get_axes()[0]).get_clim()
    npt.assert_allclose(clim, (-96, 4), atol=.5)

    plt.close("all")


def test_cycle_plot_styles():
    """Test cycle the plot styles between line and 2d plots."""
    # Use create figure to specify the plot backend
    create_figure()

    # dummy signal (needs to as longe as the default spectrogram block size
    # with at least two channels)
    signal = pf.signals.impulse(1024, [0, 1000])
    # shortcut for toggling the orientation
    key = sc_ctr["cycle_plot_types"]["key"][0]

    for function in getmembers(pf.plot.line, isfunction):

        # exclude functions that do not support interaction
        if function[0] in ["context", "custom_subplots"]:
            continue

        # plot and toggle type
        ax = function[1](signal)
        ax = ax[0] if isinstance(ax, (np.ndarray, list)) else ax
        interaction = ax.interaction
        interaction.select_action(ia.EventEmu(key))

        # get labels for testing
        xlabel = plots[function[0] + "_2d"]["xlabel"]
        ylabel = plots[function[0] + "_2d"]["ylabel"]
        cblabel = plots[function[0] + "_2d"]["cblabel"]

        # get current axis interaction instance
        ax = ax[0] if isinstance(ax, (np.ndarray, list)) else ax
        ca = np.atleast_1d(interaction.all_axes)
        cb = np.atleast_1d(interaction.all_bars)

        for idx in range(len(xlabel)):
            # test x- and y-label
            if idx == 0 and len(xlabel) > 1:
                assert ca[idx].get_xlabel() in (xlabel[idx], "")
            else:
                assert ca[idx].get_xlabel() == xlabel[idx]

            assert ca[idx].get_ylabel() == ylabel[idx]

            # test colorbar
            assert cb[idx].ax.get_ylabel() == cblabel[idx]

        plt.close("all")
