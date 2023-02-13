"""
The following introduces the concept of the
:py:class:`~pyfar.classes.coordinates` class and the coorindate systems that
are available in pyfar.

Coordinates Class
-----------------

Different coordinate systems are frequently used in acoustics research and
handling sampling points and different systems can be cumbersome. The
:py:func:`Coordinates class <pyfar.classes.coordinates.Coordinates>` was
designed with this in mind. It can convert between coordinate systems and
rotate, query and plot coordinates points. Functions for converting coordinates
not stored in a :py:func:`~pyfar.classes.coordinates.Coordinates` object are
available for convenience. However, it is strongly recommended to
use the :py:func:`~pyfar.classes.coordinates.Coordinates` class for all
conversions.

Coordinate Systems
------------------

Coordinate systems are defined by their `domain` (e.g. ``'spherical'``),
`convention` (e.g. ``'top_elev'``), and `unit` (e.g. ``'deg'``). The available
coordinate systems are shown in the image below

|coordinate_systems|

The unit for length is always meter, while the unit for angles are in radians.

Samplings
---------

A plethora of sampling schemes to generate coordinate objects is contained in
:py:mod:`~pyfar.samplings`.

.. |coordinate_systems| image:: resources/coordinate_systems.png
   :width: 100%
   :alt: Alternative text
"""
