"""Test the interaction module from pyfar.plot.

Assertions are not tested because this is a private module. Test of the plots
is done using properties of the plots to avoid time intensive saving, reading,
and comparing of images.

Note that it is not tested if the properties for
pyfar.plot._interaction.PlotParameter are correctly set in pyfar.plot.function.
If the parameters are incorrect, interaction will behave incorrect.
"""
import numpy.testing as npt
from inspect import getmembers, isfunction
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyfar as pf
import pyfar.plot._interaction as ia

# use non showing backend for speed
mpl.use("Agg")

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

    This test will fail if a new plot function is added that does not have
    an interaction. This is intended behavior.
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
    # initialze the plot
    ax = pf.plot.time(signal)

    for function in getmembers(pf.plot.line, isfunction):

        # exclude functions that do not support interaction
        if function[0] in ["context", "custom_subplots"]:
            continue

        # get current short cut and target values
        shortcut = plots[function[0]]["shortcut"]
        xlabel = plots[function[0]]["xlabel"]
        ylabel = plots[function[0]]["ylabel"]

        # toogle the interaction
        ax.interaction.select_action(ia.EventEmu(shortcut))

        # current axis or array/list of axes
        ca = plt.gcf().get_axes()
        print(f"testing: {function[0]} with axes {ca}")

        # test x- and y-label
        for idx in range(len(xlabel)):
            assert ca[idx].get_xlabel() == xlabel[idx]
            assert ca[idx].get_ylabel() == ylabel[idx]

    plt.close("all")


def test_move_and_zoom_linear():
    """Test moving and zooming linear x-axis, y-axis and colormap.

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
        rng = (lim[1] - lim[0]) / 10

        # move left and right
        ax.interaction.select_action(ia.EventEmu(move[0]))
        npt.assert_allclose(getter(), (lim[0] + rng, lim[1] + rng))
        ax.interaction.select_action(ia.EventEmu(move[1]))
        npt.assert_allclose(getter(), lim)

        # zoom in and out
        ax.interaction.select_action(ia.EventEmu(zoom[0]))
        npt.assert_allclose(getter(), (lim[0] + rng, lim[1] - rng))
        lim = getter()
        rng = (lim[1] - lim[0]) / 10
        ax.interaction.select_action(ia.EventEmu(zoom[1]))
        npt.assert_allclose(
            getter(), (lim[0] - rng, lim[1] + rng))

        plt.close()
