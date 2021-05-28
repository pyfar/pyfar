# -*- coding: utf-8 -*-

"""
Plot pyfar audio objects in the time and frequency domain for quickly
inspecting audio data and generating scientific plots.

Keyboard shortcuts are available for to ease the inspection
(See :py:func:`pyfar.plot.shortcuts`).

:py:mod:`pyfar.plot` is based on `Matplotlib <https://matplotlib.org>`_ and
all plot functions return Matplotlib axis objects for a flexible customization
of plots. In addition most plot functions pass keyword arguments to Matplotlib.

This is an example for customizing the line color and axis limits:

.. plot::

    >>> import pyfar as pf
    >>> noise = pf.signals.noise(2**14)
    >>> ax = pf.plot.freq(noise, color=(.3, .3, .3))
    >>> ax.set_ylim(-60, -20)
"""

from .line import (
    freq,
    time,
    phase,
    group_delay,
    spectrogram,
    time_freq,
    freq_phase,
    freq_group_delay,
    custom_subplots
)
from .spatial import (
    scatter,
    quiver
)

from .utils import (
    plotstyle,
    context,
    use,
    color,
    shortcuts
)

__all__ = [
    'freq',
    'time',
    'phase',
    'group_delay',
    'spectrogram',
    'time_freq',
    'freq_phase',
    'freq_group_delay',
    'custom_subplots',
    'scatter',
    'quiver',
    'plotstyle',
    'context',
    'use',
    'color',
    'shortcuts'
]
