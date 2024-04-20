# -*- coding: utf-8 -*-

"""
The following documents the pyfar plot functions. Make sure to have a look at
the `plotting <https://pyfar-gallery.readthedocs.io/en/latest/gallery/
interactive/pyfar_plotting.html>`_  and `interactive plotting
<https://pyfar-gallery.readthedocs.io/en/latest/gallery/interactive/
pyfar_interactive_plots.html>`_ examples. The latter make use of the pyfar
:py:func:`~pyfar.plot.shortcuts` to quickly explore acoustic signals.
"""

from .line import (
    freq,
    time,
    phase,
    group_delay,
    time_freq,
    freq_phase,
    freq_group_delay,
    custom_subplots
)

from .two_d import (
    time_2d,
    freq_2d,
    phase_2d,
    group_delay_2d,
    time_freq_2d,
    freq_phase_2d,
    freq_group_delay_2d,
    spectrogram
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
    'freq_2d',
    'time',
    'time_2d',
    'phase',
    'phase_2d',
    'group_delay',
    'group_delay_2d',
    'spectrogram',
    'time_freq',
    'time_freq_2d',
    'freq_phase',
    'freq_phase_2d',
    'freq_group_delay',
    'freq_group_delay_2d',
    'custom_subplots',
    'scatter',
    'quiver',
    'plotstyle',
    'context',
    'use',
    'color',
    'shortcuts'
]
