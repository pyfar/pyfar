r"""
The following introduces the concept of the
:py:class:`~pyfar.classes.coordinates` class and the coordinate systems that
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

Coordinate systems are defined by their domain (cartesian, spherical,
cylindrical), and, if multiple definitions of the domain exist, also by
their convention. The available coordinate systems are shown
in the image below

|coordinate_systems|

The unit for length is always meter, while the unit for angles is radians.
For more details see the table below.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Coordinate
     - Descriptions
   * - :py:func:`x`, :py:func:`y`, :py:func:`z`
     - x, y, z coordinate of a right handed Cartesian coordinate system in
       meter (:math:`-\infty` < x,y,z < :math:`\infty`).
   * - :py:func:`azimuth`
     - Counter clock-wise angle in the x-y plane of the right handed Cartesian
       coordinate system in radians. :math:`0` radians are defined in positive
       x-direction, :math:`\pi/2` radians in positive y-direction and so on
       (:math:`-\infty` < azimuth < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`elevation`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians elevation are defined in positive
       z-direction, :math:`\pi/2` radians in positive x-direction, and
       :math:`\pi` in negative z-direction
       (:math:`\pi/2` < elevation < :math:`\pi/2`). The elevation is a
       variation of the elevation angle.
   * - :py:func:`colatitude`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians colatitude are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction, and
       :math:`-\pi/2` in negative z-direction
       (:math:`0` < colatitude < :math:`\pi`). The colatitude is a
       variation of the colatitude.
   * - :py:func:`lateral`
     - Counter clock-wise angle in the x-y plane of the right handed Cartesian
       coordinate system in radians. :math:`0` radians are defined in positive
       x-direction, :math:`\pi/2` radians in positive y-direction and
       :math:`-\pi/2` in negative y-direction
       (:math:`-\pi/2` < lateral < :math:`\pi/2`).
   * - :py:func:`polar`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians elevation are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction,
       :math:`\pi` in negative x-direction and so on
       (:math:`-\infty` < polar < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`frontal`
     - Angle in the y-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians elevation are defined in positive
       y-direction, :math:`\pi/2` radians in positive z-direction,
       :math:`\pi` in negative y-direction and so on
       (:math:`-\infty` < frontal < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`upper`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians elevation are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction, and
       :math:`\pi` in negative x-direction
       (:math:`0` < upper < :math:`\pi`).
   * - :py:func:`radius`
     - Distance to the origin of the right handed Cartesian coordinate system
       in meters (:math:`0` < radius < :math:`\infty`).
   * - :py:func:`rho`
     - Distance perpendicular to the the z-axis of the right handed Cartesian
       coordinate system (:math:`0` < rho < :math:`\infty`).


Samplings
---------

A plethora of sampling schemes to generate coordinate objects is contained in
:py:mod:`~pyfar.samplings`.

.. |coordinate_systems| image:: resources/coordinate_systems.png
   :width: 100%
   :alt: Alternative text
"""
