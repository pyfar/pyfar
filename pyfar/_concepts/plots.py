
"""
Plot contains many :py:mod:`plot functions <pyfar.plot>` that can be used to
visualize pyfar :py:mod:`audio objects <pyfar._concepts.audio_classes>` in the
time and frequency domain for inspecting data and generating scientific plots.

The plots are based on `Matplotlib <https://matplotlib.org>`_ and
all plot functions return Matplotlib axis objects for a flexible customization
of plots. In addition most plot functions pass keyword arguments (kwargs) to
Matplotlib.

This is an example for customizing the line color using a keyword argument and
the axis limits using the Matplotlib axis object:

.. plot::

    >>> import pyfar as pf
    >>> noise = pf.signals.noise(2**14)
    >>> ax = pf.plot.freq(noise, color=(.3, .3, .3))
    >>> ax.set_ylim(-60, -20)

Interactive plots
-----------------

It is often helpful to quickly navigate through the channels of multi-channel
data or zoom into the plot around a specific frequency or amplitude. This can
be done with the pyfar :py:func:`keyboard shortcuts <pyfar.plot.shortcuts>` and
requires an interactive `backend
<https://matplotlib.org/stable/users/explain/backends.html#what-is-a-backend>`_
like *QtAgg*.
"""
