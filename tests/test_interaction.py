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
from pyfar.plot._line import _get_quad_mesh_from_axis

# use non showing backend for speed
mpl.use("Agg")
plt.close("all")

# get plot controls for universal testing
sc_plot = pf.plot.shortcuts(show=False)["plots"]
sc_ctr = pf.plot.shortcuts(show=False)["controls"]


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

    # dummy signal (needs to as longe as the default spectrogram block size)
    signal = pf.signals.impulse(1024)

    # loop functions
    for function in getmembers(pf.plot.line, isfunction):
        # exclude functions that do not support interaction
        if function[0] in ["context", "custom_subplots"]:
            continue

        ax = function[1](signal)
        # get axis of first subplot, if we have subplots
        ax = ax[0] if isinstance(ax, (np.ndarray, list)) else ax
        # assert and close figure
        assert isinstance(ax.interaction, ia.Interaction)
        plt.close()


def test_toggle_plots():
    """Test toggling plots by checking x- and y-label after toggling.

    This test will fail if a new plot function is added that does not have
    an interaction or if the new plot function is not added to the plots
    dictionary. This is intended behavior.
    """

    plots = {
        'time': {
            'shortcut': sc_plot["time"]["key"][0],
            'xlabel': ['Time in ms'],
            'ylabel': ['Amplitude']},
        'freq': {
            'shortcut': sc_plot["freq"]["key"][0],
            'xlabel': ['Frequency in Hz'],
            'ylabel': ['Magnitude in dB']},
        'phase': {
            'shortcut': sc_plot["phase"]["key"][0],
            'xlabel': ['Frequency in Hz'],
            'ylabel': ['Phase in radians']},
        'group_delay': {
            'shortcut': sc_plot["group_delay"]["key"][0],
            'xlabel': ['Frequency in Hz'],
            'ylabel': ['Group delay in s']},
        'spectrogram': {
            'shortcut': sc_plot["spectrogram"]["key"][0],
            'xlabel': ['Time in s'],
            'ylabel': ['Frequency in Hz']},
        'time_freq': {
            'shortcut': sc_plot["time_freq"]["key"][0],
            'xlabel': ['Time in ms', 'Frequency in Hz'],
            'ylabel': ['Amplitude', 'Magnitude in dB']},
        'freq_phase': {
            'shortcut': sc_plot["freq_phase"]["key"][0],
            'xlabel':  ['', 'Frequency in Hz'],
            'ylabel': ['Magnitude in dB', 'Phase in radians']},
        'freq_group_delay': {
            'shortcut': sc_plot["freq_group_delay"]["key"][0],
            'xlabel': ['', 'Frequency in Hz'],
            'ylabel': ['Magnitude in dB', 'Group delay in s']}
    }

    # dummy signal (needs to as longe as the default spectrogram block size)
    signal = pf.signals.impulse(1024)
    # initialize the plot
    ax = pf.plot.time(signal)

    for function in getmembers(pf.plot.line, isfunction):

        # exclude functions that do not support interaction
        if function[0] in ["context", "custom_subplots"]:
            continue

        # get current short cut and target values
        shortcut = plots[function[0]]["shortcut"]
        xlabel = plots[function[0]]["xlabel"]
        ylabel = plots[function[0]]["ylabel"]

        # toggle the interaction
        ax.interaction.select_action(ia.EventEmu(shortcut))

        # current axis or array/list of axes
        ca = plt.gcf().get_axes()
        print(f"testing: {function[0]} with axes {ca}")

        # test x- and y-label
        for idx in range(len(xlabel)):
            assert ca[idx].get_xlabel() == xlabel[idx]
            assert ca[idx].get_ylabel() == ylabel[idx]

    plt.close("all")


@pytest.mark.parametrize("plot,signal,toggle", [
    # togling spectrogram with short signal would raise Value error
    # if not caught
    (pf.plot.time, pf.Signal([1, 0, 0], 44100), "spectrogram")
    ])
def test_toggle_plot_not_allowed(plot, signal, toggle):
    """Test if toggles that are not allowed are caught.

    Can be extended if required in the future.
    """

    # toggling spectrogram, if signal is too short
    ax = plot(signal)
    shortcut = sc_plot[toggle]["key"][0]
    # toggle the interaction
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
            ax = pf.plot.spectrogram(signal, dB=False)
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

    # init the plot
    ax = pf.plot.spectrogram(pf.signals.impulse(1024))
    assert ax[1].get_ylabel() == "Magnitude in dB"
    # toggle x-axis
    ax[0].interaction.select_action(ia.EventEmu(sc_ctr["toggle_cm"]["key"][0]))
    assert plt.gcf().get_axes()[1].get_ylabel() == "Magnitude"
    # toggle x-axis
    ax[0].interaction.select_action(ia.EventEmu(sc_ctr["toggle_cm"]["key"][0]))
    assert plt.gcf().get_axes()[1].get_ylabel() == "Magnitude in dB"

    plt.close("all")


def test_cycle_and_toggle_lines():
    """Test toggling and cycling channels of a Signal."""

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


def test_cycle_and_toggle_signals():
    """Test toggling and cycling Signal slices."""

    # init and check start conditions
    signal = pf.signals.impulse(1024, amplitude=[1, 2])
    ax = pf.plot.spectrogram(signal)

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
