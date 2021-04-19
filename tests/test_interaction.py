"""Test the interaction module from pyfar.plot.

Assertions are not tested because this is a private module. Test of the plots
is done using properties of the plots to avoid time intensive saving, reading,
and comparing of images.

Note that it is not tested if the properties for
pyfar.plot._interaction.PlotParameter are correctly set in pyfar.plot.function.
If the parameters are incorrect, interaction will behave incorrect.
"""
from inspect import getmembers, isfunction
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyfar as pf
import pyfar.plot._interaction as ia

mpl.use("Agg")


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

    # dummy signal (needs to as longe as than the default spectrogram block
    # size)
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
            'shortcut': '1',
            'xlabel': ['Time in ms'],
            'ylabel': ['Amplitude']},
        'freq': {
            'shortcut': '2',
            'xlabel': ['Frequency in Hz'],
            'ylabel': ['Magnitude in dB']},
        'phase': {
            'shortcut': '3',
            'xlabel': ['Frequency in Hz'],
            'ylabel': ['Phase in radians']},
        'group_delay': {
            'shortcut': '4',
            'xlabel': ['Frequency in Hz'],
            'ylabel': ['Group delay in s']},
        'spectrogram': {
            'shortcut': '5',
            'xlabel': ['Time in s'],
            'ylabel': ['Frequency in Hz']},
        'time_freq': {
            'shortcut': '6',
            'xlabel': ['Time in ms', 'Frequency in Hz'],
            'ylabel': ['Amplitude', 'Magnitude in dB']},
        'freq_phase': {
            'shortcut': '7',
            'xlabel':  ['', 'Frequency in Hz'],
            'ylabel': ['Magnitude in dB', 'Phase in radians']},
        'freq_group_delay': {
            'shortcut': '8',
            'xlabel': ['', 'Frequency in Hz'],
            'ylabel': ['Magnitude in dB', 'Group delay in s']}
    }

    signal = pf.signals.impulse(1024)

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
