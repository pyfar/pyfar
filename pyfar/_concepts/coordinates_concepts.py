"""
The core of this module is the
:py:func:`~pyfar.classes.coordinates.Coordinates` class. It can convert
between coordinate conventions and rotate, query and plot coordinates points.
Functions for converting coordinates not stored in a
:py:func:`~pyfar.classes.coordinates.Coordinates`
object are available for convenience. However, it is strongly recommended to
use the :py:func:`~pyfar.classes.coordinates.Coordinates` class for all
conversions.

Coordinate systems are defined by their `domain` (e.g. ``'spherical'``),
`convention` (e.g. ``'top_elev'``), and `unit` (e.g. ``'deg'``). A complete
list and description of supported coordinate systems is given in the image
below

|coordinate_systems|

and can be obtained by

>>> coords = Coordinates()  # get an empty instance of the class
>>> coords.systems()        # list all systems

A plethora of sampling schemes to generate coordinate objects is contained in
:py:mod:`~pyfar.samplings`.

.. |coordinate_systems| image:: resources/coordinate_systems.png
   :width: 100%
   :alt: Alternative text

See :py:class:`~pyfar.classes.coordinates` for a complete documentation.
"""
