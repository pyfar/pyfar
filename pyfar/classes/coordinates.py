r"""
The following introduces the
:py:func:`Coordinates class <pyfar.classes.coordinates.Coordinates>`
and the coordinate systems that are available in pyfar. Available sampling
schemes are listed at :py:mod:`spharpy.samplings <spharpy.samplings>`.
:ref:`Examples <gallery:/gallery/interactive/pyfar_coordinates.ipynb>` for
working with Coordinates objects are part of the pyfar gallery.

Different coordinate systems are frequently used in acoustics research and
handling sampling points and different systems can be cumbersome. The
Coordinates class was designed with this in mind. It stores coordinates in
cartesian coordinates internally and can convert to all coordinate systems
listed below. Additionally, the the class can  query and plot coordinates
points. Functions for converting coordinates not stored in a Coordinates object
are available for convenience. However, it is strongly recommended to use the
Coordinates class for all conversions.

.. _coordinate_systems:

Coordinate Systems
------------------

Each coordinate system has a unique name, e.g., `spherical_elevation`, and is
defined by three coordinates, in this case `azimuth`, `elevation`, and
`radius`. The available coordinate systems are shown in the image below.

|coordinate_systems|

.. _coordinates:

Coordinates
-----------

The unit for length for the coordinates is always meter, while the unit for
angles is radians. Each coordinate is unique, but can appear in multiple
coordinate systems, e.g., the `azimuth` angle is contained in two coordinate
systems (`spherical_colatitude` and `spherical_elevation`). The table below
lists all coordinates.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Coordinate
     - Descriptions
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.x`,
       :py:func:`~pyfar.classes.coordinates.Coordinates.y`,
       :py:func:`~pyfar.classes.coordinates.Coordinates.z`
     - x, y, z coordinate of a right handed Cartesian coordinate system in
       meter (:math:`-\infty` < x,y,z < :math:`\infty`).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.azimuth`
     - Counter clock-wise angle in the x-y plane of the right handed Cartesian
       coordinate system in radians. :math:`0` radians are defined in positive
       x-direction, :math:`\pi/2` radians in positive y-direction and so on
       (:math:`-\infty` < azimuth < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.colatitude`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians colatitude are defined in positive
       z-direction, :math:`\pi/2` radians in positive x-direction, and
       :math:`\pi` in negative z-direction
       (:math:`0\leq` colatitude :math:`\leq\pi`). The colatitude is a
       variation of the elevation angle.
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.elevation`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians elevation are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction, and
       :math:`-\pi/2` in negative z-direction
       (:math:`-\pi/2\leq` elevation :math:`\leq\pi/2`). The elevation is a
       variation of the colatitude.
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.lateral`
     - Counter clock-wise angle in the x-y plane of the right handed Cartesian
       coordinate system in radians. :math:`0` radians are defined in positive
       x-direction, :math:`\pi/2` radians in positive y-direction and
       :math:`-\pi/2` in negative y-direction
       (:math:`-\pi/2\leq` lateral :math:`\leq\pi/2`).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.polar`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians polar angle are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction,
       :math:`\pi` in negative x-direction and so on
       (:math:`-\infty` < polar < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.frontal`
     - Angle in the y-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians frontal angle are defined in positive
       y-direction, :math:`\pi/2` radians in positive z-direction,
       :math:`\pi` in negative y-direction and so on
       (:math:`-\infty` < frontal < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.upper`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians upper angle are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction, and
       :math:`\pi` in negative x-direction
       (:math:`0\leq` upper :math:`\leq\pi`).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.radius`
     - Distance to the origin of the right handed Cartesian coordinate system
       in meters (:math:`0` < radius < :math:`\infty`).
   * - :py:func:`~pyfar.classes.coordinates.Coordinates.rho`
     - Radial distance to the the z-axis of the right handed Cartesian
       coordinate system (:math:`0` < rho < :math:`\infty`).

.. |coordinate_systems| image:: resources/coordinate_systems.png
   :width: 100%
   :alt: pyfar coordinate systems
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as sp_rot
import re
from copy import deepcopy
import warnings
from pyfar.classes.warnings import PyfarDeprecationWarning

import pyfar as pf


class Coordinates():
    """
    This function will be changed in pyfar 0.8.0 and will just be able to
    get cartesian coordinates. If you want to initialize in an other
    domain use :py:func:`from_spherical_colatitude`,
    :py:func:`from_spherical_elevation`, :py:func:`from_spherical_front`,
    :py:func:`from_spherical_side`, or :py:func:`from_cylindrical`
    instead. For conversions from or into degree
    use :py:func:`deg2rad` and :py:func:`rad2deg`.

    Create :py:func:`Coordinates` object with or without coordinate points.
    The points that enter the Coordinates object are defined by the `name`
    (`domain`, `convention`, and `unit`. The `unit` will be deprecated in pyfar
    v0.8.0 in favor of fixed default units, see :ref:`coordinate_systems` and
    :ref:`coordinates`)

    +--------------------+----------+------------+----------+----------+
    | domain, convention | points_1 | points_2   | points_3 | unit     |
    +====================+==========+============+==========+==========+
    | cart, right        | x        | y          | z        | met      |
    +--------------------+----------+------------+----------+----------+
    | sph, top_colat     | azimuth  | colatitude | radius   | rad, deg |
    +--------------------+----------+------------+----------+----------+
    | sph, top_elev      | azimuth  | elevation  | radius   | rad, deg |
    +--------------------+----------+------------+----------+----------+
    | sph, side          | lateral  | polar      | radius   | rad, deg |
    +--------------------+----------+------------+----------+----------+
    | sph, front         | phi      | theta      | radius   | rad, deg |
    +--------------------+----------+------------+----------+----------+
    | cyl, top           | azimuth  | z          | radius_z | rad, deg |
    +--------------------+----------+------------+----------+----------+

    Parameters
    ----------
    points_1 : array like, number
        Points for the first coordinate.
        ``'points_1'``, ``'points_2'``, and ``'points_3'`` will be renamed
        to ``'x'``, ``'y'`` and ``'z'`` in pyfar 0.8.0.
    points_2 : array like, number
        Points for the second coordinate.
        ``'points_1'``, ``'points_2'``, and ``'points_3'`` will be renamed
        to ``'x'``, ``'y'`` and ``'z'`` in pyfar 0.8.0.
    points_3 : array like, number
        Points for the third coordinate.
        ``'points_1'``, ``'points_2'``, and ``'points_3'`` will be renamed
        to ``'x'``, ``'y'`` and ``'z'`` in pyfar 0.8.0.
    domain : string
        ``'domain'``, ``'unit'`` and ``'convention'`` initialization
        parameters will be deprecated in pyfar 0.8.0 in favor of
        ``from_*``.  Different units are no longer supported. The unit is
        meter for distances and radians for angles.
        domain of the coordinate system

        ``'cart'``
            Cartesian
        ``'sph'``
            Spherical
        ``'cyl'``
            Cylindrical

        The default is ``'cart'``.
    convention: string
        ``'domain'``, ``'unit'`` and ``'convention'`` initialization
        parameters will be deprecated in pyfar 0.8.0 in favor of
        ``from_*``.  Different units are no longer supported.
        Default angle unit is radiant.

        Coordinate convention (see above)
        The default is ``'right'`` if domain is ``'cart'``,
        ``'top_colat'`` if domain is ``'sph'``, and ``'top'`` if domain is
        ``'cyl'``.
    unit: string
        ``'domain'``, ``'unit'`` and ``'convention'`` initialization
        parameters will be deprecated in pyfar 0.8.0 in favor of
        ``from_*``. Different units are no longer supported. Default
        angle unit is radiant.
        The ``'deg'`` parameter will be deprecated in pyfar 0.8.0 in favor
        of the :py:func:`deg2rad` and :py:func:`rad2deg`.

        Unit of the coordinate system. By default the first available unit
        is used, which is meters (``'met'``) for ``domain = 'cart'`` and
        radians (``'rad'``) in all other cases (See above).
    weights: array like, number, optional
        Weighting factors for coordinate points. The `shape` of the array
        must match the `shape` of the individual coordinate arrays.
        The default is ``None``.
    sh_order : int, optional
        This property will be deprecated in pyfar 0.8.0 in favor of
        :py:class:`spharpy.samplings.SamplingSphere`

        Maximum spherical harmonic order of the sampling grid.
        The default is ``None``.
    comment : str, optional
        Comment about the stored coordinate points. The default is
        ``""``, which initializes an empty string.
    """
    _x: np.array = np.empty
    _y: np.array = np.empty
    _z: np.array = np.empty
    _weights: np.array = None
    _sh_order: int = None
    _comment: str = None
    _system: dict = None

    def __init__(
            self, points_1: np.array = np.asarray([]),
            points_2: np.array = np.asarray([]),
            points_3: np.array = np.asarray([]),
            domain: str = 'cart', convention: str = None, unit: str = None,
            weights: np.array = None, sh_order=None,
            comment: str = "") -> None:

        # init empty object
        super(Coordinates, self).__init__()

        # test Deprecation warning
        if domain != 'cart' or convention is not None or unit is not None:
            warnings.warn((
                "This function will be changed in pyfar 0.8.0 to "
                "init(x, y, z)."),
                    PyfarDeprecationWarning)

        # set the coordinate system
        system = self._make_system(domain, convention, unit)
        self._system = system

        # set coordinates according to system
        if domain == 'cart':
            self._set_points(points_1, points_2, points_3)
        elif domain == 'sph':
            self._set_sph(
                points_1, points_2, points_3,
                system['convention'], system['unit'])
        elif domain == 'cyl':
            self._set_cyl(
                points_1, points_2, points_3,
                system['convention'], system['unit'])
        else:
            raise ValueError(
                f"Domain for {domain} is not implemented.")

        # save meta data
        self._set_weights(weights)
        self.sh_order = sh_order
        self._comment = comment

        if sh_order is not None:
            warnings.warn((
                "This function will be deprecated in pyfar 0.8.0 in favor "
                "of spharpy.samplings.SamplingSphere."),
                    PyfarDeprecationWarning)

    @classmethod
    def from_cartesian(
            cls, x, y, z, weights: np.array = None, comment: str = ""):
        r"""
        Create a Coordinates class object from a set of points in the
        right-handed cartesian coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.

        Parameters
        ----------
        x : ndarray, number
            X coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < x < \infty).
        y : ndarray, number
            Y coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < y < \infty).
        z : ndarray, number
            Z coordinate of a right handed Cartesian coordinate system in
            meters (-\infty < z < \infty).
        weights: array like, number, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.

        Examples
        --------

        Create a coordinates object

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates.from_cartesian(0, 0, 1)

        Or the using init

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates(0, 0, 1)
        """
        return cls(x, y, z, weights=weights, comment=comment)

    @classmethod
    def from_spherical_elevation(
            cls, azimuth, elevation, radius, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.

        Parameters
        ----------
        azimuth : ndarray, double
            Angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        elevation : ndarray, double
            Angle in radiant with respect to horizontal plane (x-z-plane).
            Used for spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        weights: array like, float, None, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.

        Examples
        --------

        Create a coordinates object

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates.from_spherical_elevation(0, 0, 1)
        """

        x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
        return cls(x, y, z, weights=weights, comment=comment)

    @classmethod
    def from_spherical_colatitude(
            cls, azimuth, colatitude, radius, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.

        Parameters
        ----------
        azimuth : ndarray, double
            Angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        colatitude : ndarray, double
            Angle in radiant with respect to polar axis (z-axis). Used for
            spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        weights: array like, number, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.

        Examples
        --------

        Create a coordinates object

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates.from_spherical_colatitude(0, 0, 1)
        """

        x, y, z = sph2cart(azimuth, colatitude, radius)
        return cls(x, y, z, weights=weights, comment=comment)

    @classmethod
    def from_spherical_side(
            cls, lateral, polar, radius, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.

        Parameters
        ----------
        lateral : ndarray, double
            Angle in radiant with respect to horizontal plane (x-y-plane).
            Used for spherical coordinate systems.
        polar : ndarray, double
            Angle in radiant of rotation from the x-z-plane facing towards
            positive x direction. Used for spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        weights: array like, number, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.

        Examples
        --------

        Create a coordinates object

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates.from_spherical_side(0, 0, 1)
        """

        x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
        return cls(x, y, z, weights=weights, comment=comment)

    @classmethod
    def from_spherical_front(
            cls, frontal, upper, radius, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.

        Parameters
        ----------
        frontal : ndarray, double
            Angle in radiant of rotation from the y-z-plane facing towards
            positive y direction. Used for spherical coordinate systems.
        upper : ndarray, double
            Angle in radiant with respect to polar axis (x-axis). Used for
            spherical coordinate systems.
        radius : ndarray, double
            Distance to origin for each point. Used for spherical coordinate
            systems.
        weights: array like, number, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.

        Examples
        --------

        Create a coordinates object

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates.from_spherical_front(0, 0, 1)
        """

        y, z, x = sph2cart(frontal, upper, radius)
        return cls(x, y, z, weights=weights, comment=comment)

    @classmethod
    def from_cylindrical(
            cls, azimuth, z, rho, weights: np.array = None,
            comment: str = ""):
        """Create a Coordinates class object from a set of points in the
        cylindrical coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.

        Parameters
        ----------
        azimuth : ndarray, double
            Angle in radiant of rotation from the x-y-plane facing towards
            positive x direction. Used for spherical and cylindrical coordinate
            systems.
        z : ndarray, double
            The z coordinate
        rho : ndarray, double
            Distance to origin for each point in the x-y-plane. Used for
            cylindrical coordinate systems.
        weights: array like, number, optional
            Weighting factors for coordinate points. The `shape` of the array
            must match the `shape` of the individual coordinate arrays.
            The default is ``None``.
        comment : str, optional
            Comment about the stored coordinate points. The default is
            ``""``, which initializes an empty string.

        Examples
        --------

        Create a coordinates object

        >>> import pyfar as pf
        >>> coordinates = pf.Coordinates.from_cylindrical(0, 0, 1)
        """

        x, y, z = cyl2cart(azimuth, z, rho)
        return cls(x, y, z, weights=weights, comment=comment)

    def set_cart(self, x, y, z, convention='right', unit='met'):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of :py:func:`cartesian`, :py:func:`x`, :py:func:`y` or :py:func:`z`.
        Enter coordinate points in cartesian coordinate systems.

        The points that enter the Coordinates object are defined by the
        `domain`, `convention`, and `unit`

        +--------------------+----------+------------+----------+----------+
        | domain, convention | points_1 | points_2   | points_3 | unit     |
        +====================+==========+============+==========+==========+
        | cart, right        | x        | y          | z        | met      |
        +--------------------+----------+------------+----------+----------+

        For more information run

        >>> coords = Coordinates()
        >>> coords.systems()

        Parameters
        ----------
        x, y, z: array like, float
            Points for the first, second, and third coordinate
        convention : string, optional
            Convention in which the coordinate points are stored. The default
            is ``'right'``.
        unit : string, optional
            Unit in which the coordinate points are stored. The default is
            ``'met'`` for meters.
        """

        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of .cart, .x, .y or .z."),
                PyfarDeprecationWarning)

        # set the coordinate system
        self._system = self._make_system('cart', convention, unit)

        # save coordinates to self
        self._set_cart(x, y, z)

    def _set_cart(self, x, y, z, convention='right', unit='met'):
        if convention != 'right':
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                (f"Conversion for {convention} is not implemented."))

        # make array
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))

        # squeeze
        if len(x.shape) == 2 and (x.shape[0] == 1 or x.shape[1] == 1):
            x = x.flatten()
        if len(y.shape) == 2 and (y.shape[0] == 1 or y.shape[1] == 1):
            y = y.flatten()
        if len(z.shape) == 2 and (z.shape[0] == 1 or z.shape[1] == 1):
            z = z.flatten()

        # save coordinates to self
        self._set_points(x, y, z)

    def get_cart(self, convention='right', unit='met', convert=False):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of :py:func:`cartesian`
        Get coordinate points in cartesian coordinate systems.

        The points that are returned are defined by the `domain`, `convention`,
        and `unit`:

        +--------------------+----------+------------+----------+----------+
        | domain, convention | p[...,1] | p[...,1]   | p[...,1] | units    |
        +====================+==========+============+==========+==========+
        | cart, right        | x        | y          | z        | met      |
        +--------------------+----------+------------+----------+----------+

        For more information run

        >>> coords = Coordinates()
        >>> coords.systems()

        Parameters
        ----------
        convention : string, optional
            Convention in which the coordinate points are stored. The default
            is ``'right'``.
        unit : string, optional
            Unit in which the coordinate points are stored. The default is
           ``'met'``.
        convert : boolean, optional
            If True, the internal representation of the samplings points will
            be converted to the queried coordinate system. The default is
            ``False``, i.e., the internal presentation remains as it is.

        Returns
        -------
        points : numpy array
            Coordinate points. ``points[...,0]`` holds the points for the first
            coordinate, ``points[...,1]`` the points for the second, and
            ``points[...,2]`` the points for the third coordinate.
        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of .cartesian"),
                PyfarDeprecationWarning)

        self._system = self._make_system('cart', convention, unit)
        return self.cartesian

    def set_sph(
            self, angles_1, angles_2, radius,
            convention='top_colat', unit='rad'):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the ``spherical_*`` properties. For conversions from or into degree
        use :py:func:`deg2rad` and :py:func:`rad2deg`.
        Enter coordinate points in spherical coordinate systems.

        The points that enter the Coordinates object are defined by the
        `domain`, `convention`, and `unit`

        +--------------------+----------+------------+----------+----------+
        | domain, convention | points_1 | points_2   | points_3 | unit     |
        +====================+==========+============+==========+==========+
        | sph, top_colat     | azimuth  | colatitude | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+
        | sph, top_elev      | azimuth  | elevation  | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+
        | sph, side          | lateral  | polar      | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+
        | sph, front         | phi      | theta      | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+

        For more information run

        >>> coords = Coordinates()
        >>> coords.systems()

        Parameters
        ----------
        points_i: array like, number
            Points for the first, second, and third coordinate
        convention : string, optional
            Convention in which the coordinate points are stored. The default
            is ``'top_colat'``.
        unit : string, optional
            Unit in which the coordinate points are stored. The default is
            ``'rad'``.
            The ``'deg'`` parameter will be deprecated in pyfar 0.8.0 in favor
            of the :py:func:`deg2rad` and :py:func:`rad2deg`.
        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of the spherical_... properties"),
                PyfarDeprecationWarning)

        # make array
        angles_1 = np.atleast_1d(np.asarray(angles_1, dtype=np.float64))
        angles_2 = np.atleast_1d(np.asarray(angles_2, dtype=np.float64))
        radius = np.atleast_1d(np.asarray(radius, dtype=np.float64))

        self._set_sph(angles_1, angles_2, radius, convention, unit)

    def _set_sph(
            self, angles_1, angles_2, radius,
            convention='top_colat', unit='rad'):

        # Convert to array
        angles_1 = np.asarray(angles_1)
        angles_2 = np.asarray(angles_2)
        radius = np.asarray(radius)

        # squeeze
        if len(angles_1.shape) == 2 and \
                (angles_1.shape[0] == 1 or angles_1.shape[1] == 1):
            angles_1 = angles_1.flatten()
        if len(angles_2.shape) == 2 and \
                (angles_2.shape[0] == 1 or angles_2.shape[1] == 1):
            angles_2 = angles_2.flatten()
        if len(radius.shape) == 2 and \
                (radius.shape[0] == 1 or radius.shape[1] == 1):
            radius = radius.flatten()

        # convert to radians
        if unit == 'deg':
            warnings.warn((
                "'deg' parameter will be deprecated in pyfar 0.8.0 in favor "
                "of the pyfar.deg2rad and pyfar.rad2deg"),
                    PyfarDeprecationWarning)
            angles_1 = angles_1 / 180 * np.pi
            angles_2 = angles_2 / 180 * np.pi

        # convert to cartesian ...
        # ... from spherical coordinate systems
        if convention == 'top_colat':
            x, y, z = sph2cart(angles_1, angles_2, radius)

        elif convention == 'top_elev':
            x, y, z = sph2cart(angles_1, np.pi / 2 - angles_2, radius)

        elif convention == 'side':
            x, z, y = sph2cart(angles_2, np.pi / 2 - angles_1, radius)

        elif convention == 'front':
            y, z, x = sph2cart(angles_1, angles_2, radius)

        else:
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                (f"Conversion for {convention} is not implemented."))

        # set the coordinate system
        self._system = self._make_system('sph', convention, unit)

        # save coordinates to self
        self._set_points(x, y, z)

    def get_sph(self, convention='top_colat', unit='rad', convert=False):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the `spherical_...` properties. For conversions from or into degree
        use :py:func:`deg2rad` and :py:func:`rad2deg`.
        Get coordinate points in spherical coordinate systems.

        The points that are returned are defined by the `domain`,
        `convention`, and `unit`:

        +--------------------+----------+------------+----------+----------+
        | domain, convention | p[...,1] | p[...,1]   | p[...,1] | units    |
        +====================+==========+============+==========+==========+
        | sph, top_colat     | azimuth  | colatitude | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+
        | sph, top_elev      | azimuth  | elevation  | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+
        | sph, side          | lateral  | polar      | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+
        | sph, front         | phi      | theta      | radius   | rad, deg |
        +--------------------+----------+------------+----------+----------+

        For more information run

        >>> coords = Coordinates()
        >>> coords.systems()

        Parameters
        ----------
        convention : string, optional
            Convention in which the coordinate points are stored. The default
            is ``'top_colat'``.
        unit : string, optional
            Unit in which the coordinate points are stored. The default is
            ``'rad'``.
            The ``'deg'`` parameter will be deprecated in pyfar 0.8.0 in favor
            of the :py:func:`deg2rad` and :py:func:`rad2deg`.
        convert : boolean, optional
            If True, the internal representation of the samplings points will
            be converted to the queried coordinate system. The default is
            ``False``, i.e., the internal presentation remains as it is.

        Returns
        -------
        points : numpy array
            Coordinate points. ``points[...,0]`` holds the points for the first
            coordinate, ``points[...,1]`` the points for the second, and
            ``points[...,2]`` the points for the third coordinate.
        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of the `spherical_*` properties."),
                PyfarDeprecationWarning)

        if convention == 'top_colat':
            points = self.spherical_colatitude
            self._system = self._make_system('sph', 'top_colat', 'rad')
        elif convention == 'top_elev':
            points = self.spherical_elevation
            self._system = self._make_system('sph', 'top_elev', 'rad')
        elif convention == 'front':
            points = self.spherical_front
            self._system = self._make_system('sph', 'front', 'rad')
        elif convention == 'side':
            points = self.spherical_side
            self._system = self._make_system('sph', 'side', 'rad')
        else:
            raise ValueError(
                f"Conversion for {convention} is not implemented.")

        conversion_factor = 1 if unit == 'rad' else 180 / np.pi
        points[..., 0] = points[..., 0] * conversion_factor
        points[..., 1] = points[..., 1] * conversion_factor
        return points

    def _get_sph(self, convention='top_colat', unit='rad', convert=False):
        # check if object is empty
        self._check_empty()

        x = self._x
        y = self._y
        z = self._z

        # convert to spherical...
        # ... top polar systems
        if convention[0:3] == 'top':
            angles_1, angles_2, radius = cart2sph(x, y, z)
            if convention == 'top_elev':
                angles_2 = np.pi / 2 - angles_2

        # ... side polar system
        # (idea for simple conversions from Robert Baumgartner and SOFA_API)
        elif convention == 'side':
            angles_2, angles_1, radius = cart2sph(x, z, -y)
            # range angles
            angles_1 = angles_1 - np.pi / 2
            angles_2 = np.mod(angles_2 + np.pi / 2, 2 * np.pi) - np.pi / 2
        # ... front polar system
        elif convention == 'front':
            angles_1, angles_2, radius = cart2sph(y, z, x)

        else:
            raise ValueError(
                f"Conversion for {convention} is not implemented.")

        # convert to degrees
        if unit == 'deg':
            warnings.warn((
                "'deg' parameter will be deprecated in pyfar 0.8.0 in favor "
                "of the pyfar.deg2rad and pyfar.rad2deg"),
                    PyfarDeprecationWarning)
            angles_1 = angles_1 / np.pi * 180
            angles_2 = angles_2 / np.pi * 180
        elif not unit == 'rad':
            raise ValueError(
                f"{unit} is not implemented.")

        # return points
        return angles_1, angles_2, radius

    def set_cyl(self, azimuth, z, radius_z, convention='top', unit='rad'):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the :py:func:`cylindrical` property. For conversions from or
        into degree use :py:func:`deg2rad` and :py:func:`rad2deg`.
        Enter coordinate points in cylindrical coordinate systems.

        The points that enter the Coordinates object are defined by the
        `domain`, `convention`, and `unit`

        +--------------------+----------+------------+----------+----------+
        | domain, convention | points_1 | points_2   | points_3 | unit     |
        +====================+==========+============+==========+==========+
        | cyl, top           | azimuth  | z          | radius_z | rad, deg |
        +--------------------+----------+------------+----------+----------+

        For more information run

        >>> coords = Coordinates()
        >>> coords.systems()

        Parameters
        ----------
        points_i: array like, number
            Points for the first, second, and third coordinate
        convention : string, optional
            Convention in which the coordinate points are stored. The default
            is ``'top'``.
        unit : string, optional
            Unit in which the coordinate points are stored. The default is
            ``'rad'``.
        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of the cylindrical property."),
                PyfarDeprecationWarning)
        self._set_cyl(azimuth, z, radius_z, convention)

    def _set_cyl(self, azimuth, z, rho, convention='top', unit='rad'):

        # Convert to array
        azimuth = np.asarray(azimuth)
        z = np.asarray(z)
        rho = np.asarray(rho)

        # squeeze
        if len(azimuth.shape) == 2 and \
                (azimuth.shape[0] == 1 or azimuth.shape[1] == 1):
            azimuth = azimuth.flatten()
        if len(z.shape) == 2 and \
                (z.shape[0] == 1 or z.shape[1] == 1):
            z = z.flatten()
        if len(rho.shape) == 2 and \
                (rho.shape[0] == 1 or rho.shape[1] == 1):
            rho = rho.flatten()

        # convert to radians
        if unit == 'deg':
            warnings.warn((
                "'deg' parameter will be deprecated in pyfar 0.8.0 in favor "
                "of the pyfar.deg2rad and pyfar.rad2deg"),
                    PyfarDeprecationWarning)
            azimuth = azimuth / 180 * np.pi
        elif not unit == 'rad':
            raise ValueError(
                f"{unit} is not implemented.")

        # ... from cylindrical coordinate systems
        if convention == 'top':
            x, y, z = cyl2cart(azimuth, z, rho)
        else:
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                (f"Conversion for {convention} is not implemented."))

        # set the coordinate system
        self._system = self._make_system('cyl', convention, unit)

        # save coordinates to self
        self._set_points(x, y, z)

    def get_cyl(self, convention='top', unit='rad', convert=False):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the `cylindrical` property. For conversions from or into degree
        use :py:func:`deg2rad` and :py:func:`rad2deg`.
        Get coordinate points in cylindrical coordinate system.

        The points that are returned are defined by the `domain`, `convention`,
        and `unit`:

        +--------------------+----------+------------+----------+----------+
        | domain, convention | p[...,1] | p[...,1]   | p[...,1] | units    |
        +====================+==========+============+==========+==========+
        | cyl, top           | azimuth  | z          | radius_z | rad, deg |
        +--------------------+----------+------------+----------+----------+

        For more information run

        >>> coords = Coordinates()
        >>> coords.systems()

        Parameters
        ----------
        convention : string, optional
            Convention in which the coordinate points are stored. The default
            is ``'right'``.
        unit : string, optional
            Unit in which the coordinate points are stored. The default is
            ``'rad'``.
            The ``'deg'`` parameter will be deprecated in pyfar 0.8.0 in favor
            of the :py:func:`deg2rad` and :py:func:`rad2deg`.

        convert : boolean, optional
            If True, the internal representation of the samplings points will
            be converted to the queried coordinate system. The default is
            False, i.e., the internal presentation remains as it is.

        Returns
        -------
        points : numpy array
            Coordinate points. ``points[...,0]`` holds the points for the first
            coordinate, ``points[...,1]`` the points for the second, and
            ``points[...,2]`` the points for the third coordinate.
        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of the cylindrical property."),
                PyfarDeprecationWarning)
        points = self.cylindrical

        conversion_factor = 1 if unit == 'rad' else 180 / np.pi
        points[..., 0] = points[..., 0] * conversion_factor
        return points

    def _get_cyl(self, convention='top', unit='rad'):
        """internal function to convert cart to cyl coordinates"""

        # check if object is empty
        self._check_empty()

        # convert to cylindrical ...
        # ... top systems
        if convention == 'top':
            azimuth, z, rho = cart2cyl(self.x, self.y, self.z)
        else:
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                f"Conversion for {convention} is not implemented.")

        # convert to degrees
        if unit == 'deg':
            warnings.warn((
                "'deg' parameter will be deprecated in pyfar 0.8.0 in favor "
                "of the pyfar.deg2rad and pyfar.rad2deg"),
                    PyfarDeprecationWarning)
            azimuth = azimuth / np.pi * 180
        elif unit != 'rad':
            raise ValueError(
                f"unit for {unit} is not implemented.")

        # return points and convert internal state if desired
        return azimuth, z, rho

    @property
    def weights(self):
        """Get sampling weights."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set sampling weights."""
        self._set_weights(value)

    @property
    def sh_order(self):
        """This function will be deprecated in pyfar 0.8.0 in favor
            of :py:class:`spharpy.samplings.SamplingSphere`.
            Get the maximum spherical harmonic order."""
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of spharpy.samplings.SamplingSphere."),
                PyfarDeprecationWarning)

        return self._sh_order

    @sh_order.setter
    def sh_order(self, value):
        """This function will be deprecated in pyfar 0.8.0 in favor
            of :py:class:`spharpy.samplings.SamplingSphere`.
            Set the maximum spherical harmonic order."""
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of spharpy.samplings.SamplingSphere."),
                PyfarDeprecationWarning)

        self._sh_order = int(value) if value is not None else None

    @property
    def comment(self):
        """Get comment."""
        return self._comment

    @comment.setter
    def comment(self, value):
        """Set comment."""
        if not isinstance(value, str):
            raise TypeError("comment has to be of type string.")
        else:
            self._comment = value

    @property
    def cshape(self):
        """
        Return channel shape.

        The channel shape gives the shape of the coordinate points excluding
        the last dimension, which is always 3.
        """
        if self._x.size:
            return self._x.shape
        else:
            return (0,)

    @property
    def cdim(self):
        """
        Return channel dimension.

        The channel dimension gives the number of dimensions of the coordinate
        points excluding the last dimension.
        """
        if self._x.size:
            return self._x.ndim
        else:
            return 0

    @property
    def csize(self):
        """
        Return channel size.

        The channel size gives the number of points stored in the coordinates
        object.
        """
        return self._x.size

    @property
    def cartesian(self):
        """
        Returns :py:func:`x`, :py:func:`y`, :py:func:`z`.
        Right handed cartesian coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information."""
        return np.atleast_2d(np.moveaxis(
            np.array([self.x, self.y, self.z]), 0, -1))

    @cartesian.setter
    def cartesian(self, value):
        self._set_points(value[..., 0], value[..., 1], value[..., 2])

    @property
    def spherical_elevation(self):
        """
        Spherical coordinates according to the top pole elevation coordinate
        system. :py:func:`azimuth`, :py:func:`elevation`,
        :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information."""
        azimuth, elevation, radius = cart2sph(self.x, self.y, self.z)
        elevation = np.pi / 2 - elevation
        return np.atleast_2d(np.moveaxis(
            np.array([azimuth, elevation, radius]), 0, -1))

    @spherical_elevation.setter
    def spherical_elevation(self, value):
        value[..., 1] = _check_array_limits(
            value[..., 1], -np.pi/2, np.pi/2, 'elevation angle')
        x, y, z = sph2cart(
            value[..., 0], np.pi / 2 - value[..., 1], value[..., 2])
        self._set_points(x, y, z)

    @property
    def spherical_colatitude(self):
        """
        Spherical coordinates according to the top pole colatitude coordinate
        system.
        Returns :py:func:`azimuth`, :py:func:`colatitude`,
        :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information."""
        azimuth, colatitude, radius = cart2sph(self.x, self.y, self.z)
        return np.atleast_2d(np.moveaxis(
            np.array([azimuth, colatitude, radius]), 0, -1))

    @spherical_colatitude.setter
    def spherical_colatitude(self, value):
        value[..., 1] = _check_array_limits(
            value[..., 1], 0, np.pi, 'colatitude angle')
        x, y, z = sph2cart(value[..., 0], value[..., 1], value[..., 2])
        self._set_points(x, y, z)

    @property
    def spherical_side(self):
        """
        Spherical coordinates according to the side pole coordinate system.
        Returns :py:func:`lateral`, :py:func:`polar`, :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information."""
        polar, lateral, radius = cart2sph(self.x, self.z, -self.y)
        lateral = lateral - np.pi / 2
        polar = np.mod(polar + np.pi / 2, 2 * np.pi) - np.pi / 2
        return np.atleast_2d(np.moveaxis(
            np.array([lateral, polar, radius]), 0, -1))

    @spherical_side.setter
    def spherical_side(self, value):
        value[..., 0] = _check_array_limits(
            value[..., 0], -np.pi/2, np.pi/2, 'polar angle')
        x, z, y = sph2cart(
            value[..., 1], np.pi / 2 - value[..., 0], value[..., 2])
        self._set_points(x, y, z)

    @property
    def spherical_front(self):
        """
        Spherical coordinates according to the frontal pole coordinate system.
        Returns :py:func:`frontal`, :py:func:`upper`, :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information."""

        frontal, upper, radius = cart2sph(self.y, self.z, self.x)
        return np.atleast_2d(np.moveaxis(
            np.array([frontal, upper, radius]), 0, -1))

    @spherical_front.setter
    def spherical_front(self, value):
        value[..., 1] = _check_array_limits(
            value[..., 1], 0, np.pi, 'frontal angle')
        y, z, x = sph2cart(value[..., 0], value[..., 1], value[..., 2])
        self._set_points(x, y, z)

    @property
    def cylindrical(self):
        """
        Cylindrical coordinates.
        Returns :py:func:`azimuth`, :py:func:`z`, :py:func:`rho`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information."""
        azimuth, z, rho = cart2cyl(self.x, self.y, self.z)
        return np.atleast_2d(np.moveaxis(
            np.array([azimuth, z, rho]), 0, -1))

    @cylindrical.setter
    def cylindrical(self, value):
        x, y, z = cyl2cart(value[..., 0], value[..., 1], value[..., 2])
        self._set_points(x, y, z)

    @property
    def x(self):
        r"""
        X coordinate of a right handed Cartesian coordinate system in meters
        (:math:`-\infty` < x < :math:`\infty`)."""
        self._check_empty()
        return self._x

    @x.setter
    def x(self, value):
        self._set_points(value, self.y, self.z)

    @property
    def y(self):
        r"""
        Y coordinate of a right handed Cartesian coordinate system in meters
        (:math:`-\infty` < y < :math:`\infty`)."""
        self._check_empty()
        return self._y

    @y.setter
    def y(self, value):
        self._set_points(self.x, value, self.z)

    @property
    def z(self):
        r"""
        Z coordinate of a right handed Cartesian coordinate system in meters
        (:math:`-\infty` < z < :math:`\infty`)."""
        self._check_empty()
        return self._z

    @z.setter
    def z(self, value):
        self._set_points(self.x, self.y, value)

    @property
    def rho(self):
        r"""
        Radial distance to the the z-axis of the right handed Cartesian
        coordinate system (:math:`0` < rho < :math:`\infty`)."""
        return self.cylindrical[..., 2]

    @rho.setter
    def rho(self, rho):
        cylindrical = self.cylindrical
        cylindrical[..., 2] = rho
        self.cylindrical = cylindrical

    @property
    def radius(self):
        r"""
        Distance to the origin of the right handed Cartesian coordinate system
        in meters (:math:`0` < radius < :math:`\infty`)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @radius.setter
    def radius(self, radius):
        spherical_colatitude = self.spherical_colatitude
        spherical_colatitude[..., 2] = radius
        self.spherical_colatitude = spherical_colatitude

    @property
    def azimuth(self):
        r"""
        Counter clock-wise angle in the x-y plane of the right handed Cartesian
        coordinate system in radians. :math:`0` radians are defined in positive
        x-direction, :math:`\pi/2` radians in positive y-direction and so on
        (:math:`-\infty` < azimuth < :math:`\infty`, :math:`2\pi`-cyclic)."""
        return self.spherical_colatitude[..., 0]

    @azimuth.setter
    def azimuth(self, azimuth):
        spherical_colatitude = self.spherical_colatitude
        spherical_colatitude[..., 0] = azimuth
        self.spherical_colatitude = spherical_colatitude

    @property
    def elevation(self):
        r"""
        Angle in the x-z plane of the right handed Cartesian coordinate system
        in radians. :math:`0` radians elevation are defined in positive
        x-direction, :math:`\pi/2` radians in positive z-direction, and
        :math:`-\pi/2` in negative z-direction
        (:math:`-\pi/2\leq` elevation :math:`\leq\pi/2`). The elevation is a
        variation of the colatitude."""
        return self.spherical_elevation[..., 1]

    @elevation.setter
    def elevation(self, elevation):
        spherical_elevation = self.spherical_elevation
        spherical_elevation[..., 1] = elevation
        self.spherical_elevation = spherical_elevation

    @property
    def colatitude(self):
        r"""
        Angle in the x-z plane of the right handed Cartesian coordinate system
        in radians. :math:`0` radians colatitude are defined in positive
        z-direction, :math:`\pi/2` radians in positive x-direction, and
        :math:`\pi` in negative z-direction
        (:math:`0\leq` colatitude :math:`\leq\pi`). The colatitude is a
        variation of the elevation angle."""
        return self.spherical_colatitude[..., 1]

    @colatitude.setter
    def colatitude(self, colatitude):
        spherical_colatitude = self.spherical_colatitude
        spherical_colatitude[..., 1] = colatitude
        self.spherical_colatitude = spherical_colatitude

    @property
    def frontal(self):
        r"""
        Angle in the y-z plane of the right handed Cartesian coordinate system
        in radians. :math:`0` radians frontal angle are defined in positive
        y-direction, :math:`\pi/2` radians in positive z-direction,
        :math:`\pi` in negative y-direction and so on
        (:math:`-\infty` < frontal < :math:`\infty`, :math:`2\pi`-cyclic)."""
        return self.spherical_front[..., 0]

    @frontal.setter
    def frontal(self, frontal):
        spherical_front = self.spherical_front
        spherical_front[..., 0] = frontal
        self.spherical_front = spherical_front

    @property
    def upper(self):
        r"""
        Angle in the x-z plane of the right handed Cartesian coordinate system
        in radians. :math:`0` radians upper angle are defined in positive
        x-direction, :math:`\pi/2` radians in positive z-direction, and
        :math:`\pi` in negative x-direction
        (:math:`0\leq` upper :math:`\leq\pi`)."""
        return self.spherical_front[..., 1]

    @upper.setter
    def upper(self, upper):
        spherical_front = self.spherical_front
        spherical_front[..., 1] = upper
        self.spherical_front = spherical_front

    @property
    def lateral(self):
        r"""
        Counter clock-wise angle in the x-y plane of the right handed Cartesian
        coordinate system in radians. :math:`0` radians are defined in positive
        x-direction, :math:`\pi/2` radians in positive y-direction and
        :math:`-\pi/2` in negative y-direction
        (:math:`-\pi/2\leq` lateral :math:`\leq\pi/2`)."""
        return self.spherical_side[..., 0]

    @lateral.setter
    def lateral(self, lateral):
        spherical_side = self.spherical_side
        spherical_side[..., 0] = lateral
        self.spherical_side = spherical_side

    @property
    def polar(self):
        r"""
        Angle in the x-z plane of the right handed Cartesian coordinate system
        in radians. :math:`0` radians polar angle are defined in positive
        x-direction, :math:`\pi/2` radians in positive z-direction,
        :math:`\pi` in negative x-direction and so on
        (:math:`-\infty` < polar < :math:`\infty`, :math:`2\pi`-cyclic)."""
        return self.spherical_side[..., 1]

    @polar.setter
    def polar(self, polar):
        spherical_side = self.spherical_side
        spherical_side[..., 1] = polar
        self.spherical_side = spherical_side

    def systems(self, show='all', brief=False):
        """
        This function will be deprecated in pyfar 0.8.0, check the
        documentation instead.
        Print coordinate systems and their description on the console.

        .. note::
           All coordinate systems are described with respect to the right
           handed cartesian system (``domain='cart'``, ``convention='right'``).
           Distances are always specified in meters, while angles can be
           radians or degrees (``unit='rad'`` or ``unit='deg'``).


        Parameters
        ----------
        show: string, optional
            ``'current'`` to list the current coordinate system or ``'all'``
            to list all coordinate systems. The default is ``'all'``.
        brief : boolean, optional
            Will only list the domains, conventions and units if True. The
            default is ``False``.

        Returns
        -------
        Prints to console.
        """

        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0."),
                PyfarDeprecationWarning)

        if show == 'current':
            domain = self._system['domain']
            convention = self._system['convention']
            unit = self._system['unit']
        elif show == 'all':
            domain = convention = unit = 'all'
        else:
            raise ValueError("show must be 'current' or 'all'.")

        # get coordinate systems
        systems = self._systems()

        # print information
        domains = list(systems) if domain == 'all' else [domain]

        if brief:
            print('domain, convention, unit')
            print('- - - - - - - - - - - - -')
            for dd in domains:
                conventions = list(systems[dd]) if convention == 'all' \
                    else [convention]
                for cc in conventions:
                    # current coordinates
                    coords = systems[dd][cc]['coordinates']
                    # current units
                    if unit != 'all':
                        units = [units for units in systems[dd][cc]['units']
                                 if unit == units[0][0:3]]
                    else:
                        units = systems[dd][cc]['units']
                    # key for unit
                    unit_key = [u[0][0:3] for u in units]
                    print(f"{dd}, {cc}, [{', '.join(unit_key)}]")
        else:
            for dd in domains:
                conventions = \
                    list(systems[dd]) if convention == 'all' else [convention]
                for cc in conventions:
                    # current coordinates
                    coords = systems[dd][cc]['coordinates']
                    # current units
                    if unit != 'all':
                        units = [units for units in systems[dd][cc]['units']
                                 if unit == units[0][0:3]]
                    else:
                        units = systems[dd][cc]['units']
                    # key for unit
                    unit_key = [u[0][0:3] for u in units]
                    print("- - - - - - - - - - - - - - - - - "
                          "- - - - - - - - - - - - - - - - -")
                    print(f"domain: {dd}, convention: {cc}, unit: "
                          f"[{', '.join(unit_key)}]\n")
                    print(systems[dd][cc]['description_short'] + '\n')
                    print("Coordinates:")
                    for nn, coord in enumerate(coords):
                        cur_units = [u[nn] for u in units]
                        print(
                            f"points_{nn + 1}: {coord} ",
                            f"[{', '.join(cur_units)}]")
                    print('\n' + systems[dd][cc]['description'] + '\n\n')

    def show(self, mask=None, **kwargs):
        """
        Show a scatter plot of the coordinate points.

        Parameters
        ----------
        mask : boolean numpy array, None, optional
            Mask or indexes to highlight. Highlight points in red if
            ``mask==True``.
            The default is ``None``, which plots all points in the same color.
        kwargs : optional
            Keyword arguments are passed to
            :py:func:`matplotlib.pyplot.scatter`.
            If a mask is provided and the key `c` is contained in kwargs, it
            will be overwritten.

        Returns
        -------
        ax : :py:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`
            The axis used for the plot.

        """
        if mask is None:
            ax = pf.plot.scatter(self, **kwargs)
        else:
            mask = np.asarray(mask)
            colors = np.full(self.cshape, pf.plot.color('b'))
            colors[mask] = pf.plot.color('r')
            ax = pf.plot.scatter(self, c=colors.flatten(), **kwargs)

        ax.set_box_aspect([
            np.ptp(self.x),
            np.ptp(self.y),
            np.ptp(self.z)])
        ax.set_aspect('equal')

        return ax

    def find_nearest(self, find, k=1, distance_measure='euclidean',
                     radius_tol=None):
        """
        Find the k nearest coordinates points.

        Parameters
        ----------
        find : pf.Coordinates
            Coordinates to which the nearest neighbors are searched.
        k : int, optional
            Number of points to return. k must be > 0. The default is ``1``.
        distance_measure : string, optional
            ``'euclidean'``
                distance is determined by the euclidean distance.
                This is default.
            ``'spherical_radians'``
                distance is determined by the great-circle distance
                expressed in radians.
            ``'spherical_meter'``
                distance is determined by the great-circle distance
                expressed in meters.
        radius_tol : float, None
            For all spherical distance measures, the coordinates must be on
            a sphere, so the radius must be constant. This parameter defines
            the maximum allowed difference within the radii. Note that
            increasing the tolerance decreases the accuracy of the search.
            The default ``None`` uses a tolerance of two times the decimal
            resolution, which is determined from the data type of the
            coordinate points using ``numpy.finfo``.

        Returns
        -------
        index : tuple of arrays
            Indices of the neighbors. Arrays of shape ``(k, find.cshape)``
            if k>1 else ``(find.cshape, )``.
        distance : numpy array of floats
            Distance between the points, after the given ``distance_measure``.
            It's of shape (k, find.cshape).

        Notes
        -----
        This is a wrapper for :py:class:`scipy.spatial.cKDTree`.

        Examples
        --------

        Find frontal point from a spherical coordinate system

        .. plot::

            >>> import pyfar as pf
            >>> coords = pf.samplings.sph_lebedev(sh_order=10)
            >>> to_find = pf.Coordinates(1, 0, 0)
            >>> index, distance = coords.find_nearest(to_find)
            >>> coords.show(index)
            >>> distance
            0.0

        Find multidimensional points in multidimensional coordinates with k=1

        >>> import pyfar as pf
        >>> import numpy as np
        >>> coords = pf.Coordinates(np.arange(9).reshape((3, 3)), 0, 1)
        >>> to_find = pf.Coordinates(
        >>>     np.array([[0, 1], [2, 3]]), 0, 1)
        >>> i, d = coords.find_nearest(to_find)
        >>> coords[i] == find
        True
        >>> i
        (array([[0, 0],
                [0, 1]], dtype=int64),
         array([[0, 1],
                [2, 0]], dtype=int64))
        >>> d
        array([[0., 0.],
               [0., 0.]])

        Find multidimensional points in multidimensional coordinates with k=3

        >>> import pyfar as pf
        >>> import numpy as np
        >>> coords = pf.Coordinates(np.arange(9).reshape((3, 3)), 0, 1)
        >>> find = pf.Coordinates(
        >>>     np.array([[0, 1], [2, 3]]), 0, 1)
        >>> i, d = coords.find_nearest(find, 3)
        >>> # the k-th dimension is at the end
        >>> i[0].shape
        (3, 2, 2)
        >>> # now just access the k=0 dimension
        >>> coords[i][0].cartesian
        array([[[0., 0., 1.],
                [1., 0., 1.]],
               [[2., 0., 1.],
                [3., 0., 1.]]])
        """

        # check the input
        if radius_tol is None:
            radius_tol = 2 * np.finfo(self.x.dtype).resolution
        if not isinstance(radius_tol, float) or radius_tol < 0:
            raise ValueError("radius_tol must be a non negative number.")
        if not isinstance(k, int) or k <= 0 or k > self.csize:
            raise ValueError("k must be an integer > 0 and <= self.csize.")
        if not isinstance(find, Coordinates):
            raise ValueError("find must be an pf.Coordinates object.")
        allowed_measures = [
                'euclidean', 'spherical_radians', 'spherical_meter']
        if distance_measure not in allowed_measures:
            raise ValueError(
                f"distance_measure needs to be in {allowed_measures} and "
                f"it is {distance_measure}")

        # get target point in cartesian coordinates
        points = find.cartesian

        # get KDTree
        kdtree = self._make_kdtree()

        # query nearest neighbors
        points = points.flatten() if find.csize == 1 else points

        # nearest points
        distance, index = kdtree.query(points, k=k)

        if distance_measure in ['spherical_radians', 'spherical_meter']:
            # determine validate radius
            radius = np.concatenate((self.radius, find.radius))
            delta_radius = np.max(radius) - np.min(radius)
            if delta_radius > radius_tol:
                raise ValueError(
                    f"find_nearest_sph only works if all points have the same \
                    radius. Differences are larger than {radius_tol}")
            radius = np.max(radius)

            # convert cartesian coordinates to length on the great circle using
            # the Haversine formula
            distance = 2 * np.arcsin(distance / (2 * radius))

            if distance_measure == 'spherical_meter':
                # convert angle in radiant to distance on the sphere
                # distance = 2*radius*pi*distance/(2*pi) = radius*distance
                distance *= radius

        if self.cdim == 1:
            if k > 1:
                index_multi = np.moveaxis(index, -1, 0)
                index = np.empty((k), dtype=tuple)
                for kk in range(k):
                    index[kk] = tuple([index_multi[kk]], )
            else:
                index = tuple([index], )
        else:
            index_array = np.arange(self.csize).reshape(self.cshape)
            index_multi = []
            for dim in range(self.cdim):
                index_multi.append([])
                for i in index.flatten():
                    index_multi[dim].append(np.where(i == index_array)[dim][0])
                index_multi[dim] = np.asarray(
                    index_multi[dim]).reshape(index.shape)
            if k > 1:
                index_multi = np.moveaxis(index_multi, -1, 0)
                index = np.empty((k), dtype=tuple)
                for kk in range(k):
                    index[kk] = tuple(index_multi[kk])
            else:
                index = tuple(index_multi)

        if k > 1:
            distance = np.moveaxis(distance, -1, 0)

        return index, distance

    def find_within(
            self, find, distance=0., distance_measure='euclidean',
            atol=None, return_sorted=True, radius_tol=None):
        """
        Find coordinates within a certain distance to the query points.

        Parameters
        ----------
        find : pf.Coordinates
            Coordinates to which the nearest neighbors are searched.
        distance : number, optional
            Maximum allowed distance to the given points ``find``.
            Distance must be >= 0. For just exact matches use ``0``.
            The default is ``0``.
        distance_measure : string, optional
            ``'euclidean'``
                distance is determined by the euclidean distance.
                This is default.
            ``'spherical_radians'``
                distance is determined by the great-circle distance
                expressed in radians.
            ``'spherical_meter'``
                distance is determined by the great-circle distance
                expressed in meters.
        atol : float, None
            Absolute tolerance for distance. The default ``None`` uses a
            tolerance of two times the decimal resolution, which is
            determined from the data type of the coordinate points
            using :py:class:`numpy.finfo`.
        return_sorted : bool, optional
            Sorts returned indices if True and does not sort them if False.
            The default is True.
        radius_tol : float, None
            For all spherical distance measures, the coordinates must be on
            a sphere, so the radius must be constant. This parameter defines
            the maximum allowed difference within the radii. Note that
            increasing the tolerance decreases the accuracy of the search,
            i.e., points that are within the search distance might not be
            found or points outside the search distance may be returned.
            The default ``None`` uses a tolerance of two times the decimal
            resolution, which is determined from the data type of the
            coordinate points using :py:class:`numpy.finfo`.

        Returns
        -------
        index : tuple of array
            Indices of the containing coordinates. Arrays of shape
            (find.cshape).

        Notes
        -----
        This is a wrapper for :py:class:`scipy.spatial.cKDTree`.
        Compared to previous implementations, it supports self.ndim>1 as well.

        Examples
        --------

        Find all point with 1m distance from the frontal point

        .. plot::

            >>> import pyfar as pf
            >>> coords = pf.samplings.sph_lebedev(sh_order=10)
            >>> find = pf.Coordinates(1, 0, 0)
            >>> index = coords.find_within(find, 1)
            >>> coords.show(index)

        Find all point with 1m distance from two points

        .. plot::

            >>> import pyfar as pf
            >>> coords = pf.Coordinates(np.arange(6), 0, 0)
            >>> find = pf.Coordinates([2, 3], 0, 0)
            >>> index = coords.find_within(find, 1)
            >>> coords.show(index[0])
        """

        # check the input
        if radius_tol is None:
            radius_tol = 2 * np.finfo(self.x.dtype).resolution
        if atol is None:
            atol = 2 * np.finfo(self.x.dtype).resolution
        if float(distance) < 0:
            raise ValueError("distance must be a non negative number.")
        if not isinstance(atol, float) or atol < 0:
            raise ValueError("atol must be a non negative number.")
        if not isinstance(radius_tol, float) or radius_tol < 0:
            raise ValueError("radius_tol must be a non negative number.")
        if not isinstance(find, Coordinates):
            raise ValueError("coords must be an pf.Coordinates object.")
        if not isinstance(return_sorted, bool):
            raise ValueError("return_sorted must be a bool.")
        allowed_measures = [
            'euclidean', 'spherical_radians', 'spherical_meter']
        if distance_measure not in allowed_measures:
            raise ValueError(
                f"distance_measure needs to be in {allowed_measures} and "
                f"it is {distance_measure}")

        # get target point in cartesian coordinates
        points = find.cartesian

        # get KDTree
        kdtree = self._make_kdtree()

        # query nearest neighbors
        points = points.flatten() if find.csize == 1 else points

        # nearest points
        if distance_measure == 'euclidean':
            index = kdtree.query_ball_point(
                points, distance + atol, return_sorted=return_sorted)
        if distance_measure in ['spherical_radians', 'spherical_meter']:
            # determine validate radius
            radius = self.radius
            delta_radius = np.max(radius) - np.min(radius)
            if delta_radius > radius_tol:
                raise ValueError(
                    "find_nearest_sph only works if all points have the same "
                    f"radius. Differences are larger than {radius_tol}")
            radius = np.max(radius)

            if distance_measure == 'spherical_radians':
                # convert angle in radiant to distance on the sphere
                # d = 2r*pi*d/(2*pi) = r*d
                distance = radius * distance

            # convert length on the great circle to euclidean distance
            distance = 2 * radius * np.sin(distance / (2 * radius))

            index = kdtree.query_ball_point(
                points, distance + atol, return_sorted=return_sorted)

        if self.cdim == 1:
            if find.csize > 1:
                for i in range(len(index)):
                    index[i] = tuple([index[i]], )
            else:
                index = tuple([index], )

        else:
            index_array = np.arange(self.csize).reshape(self.cshape)
            index_new = np.empty((find.csize), dtype=tuple)
            for i in range(find.csize):
                index_multi = []
                if find.csize > 1:
                    for j in index[i]:
                        index_multi.append(np.where(j == index_array))
                else:
                    for j in index:
                        index_multi.append(np.where(j == index_array))

                index_multi = np.moveaxis(np.squeeze(
                    np.asarray(index_multi)), -1, 0)
                if find.csize > 1:
                    index_new[i] = tuple(index_multi)
                else:
                    index_new = tuple(index_multi)

            index = index_new

        return index

    def find_nearest_k(self, points_1, points_2, points_3, k=1,
                       domain='cart', convention='right', unit='met',
                       show=False):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the ``find_nearest`` method.

        Find the k nearest coordinates points.

        Parameters
        ----------
        points_i : array like, number
            First, second and third coordinate of the points to which the
            nearest neighbors are searched.
        k : int, optional
            Number of points to return. k must be > 0. The default is ``1``.
        domain : string, optional
            Domain of the points. The default is ``'cart'``.
        convention: string, optional
            Convention of points. The default is ``'right'``.
        unit : string, optional
            Unit of the points. The default is ``'met'`` for meters.
        show : bool, optional
            Show a plot of the coordinate points. The default is ``False``.

        Returns
        -------
        index : numpy array of ints
            The locations of the neighbors in the getter methods (e.g.,
            ``self.cartesian``). Dimension according to `distance` (see below).
            Missing neighbors are indicated with ``csize``. Also see Notes
            below.
        mask : boolean numpy array
            Mask that contains ``True`` at the positions of the selected points
            and ``False`` otherwise. Mask is of shape ``cshape``.

        Notes
        -----
        :py:class:`scipy.spatial.cKDTree` is used for the search, which
        requires an (N, 3) array. The coordinate points in self are thus
        reshaped to (`csize`, 3) before they are passed to ``cKDTree``.
        The index that  is returned refers to the reshaped coordinate points.
        To access the points for example use

        >>> points_reshaped = self.cartesian.reshape((self.csize, 3))
        >>> points_reshaped[index]

        Examples
        --------

        Find the nearest point in a line

        .. plot::

            >>> import pyfar as pf
            >>> coords = pf.Coordinates(np.arange(-5, 5), 0, 0)
            >>> result = coords.find_nearest_k(0, 0, 0, show=True)
        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of find_nearest method."),
                PyfarDeprecationWarning)

        # check the input
        assert isinstance(k, int) and k > 0 and k <= self.csize, \
            "k must be an integer > 0 and <= self.csize."

        # get the points
        _, index, mask = self._find_nearest(
            points_1, points_2, points_3,
            domain, convention, unit, show, k, 'k')

        return index, mask

    def find_nearest_cart(self, points_1, points_2, points_3, distance,
                          domain='cart', convention='right', unit='met',
                          show=False, atol=1e-15):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the ``find_within`` method.
        Find coordinates within a certain distance in meters to query points.

        Parameters
        ----------
        points_i : array like, number
            First, second and third coordinate of the points to which the
            nearest neighbors are searched.
        distance : number
            Euclidean distance in meters in which the nearest points are
            searched. Must be >= 0.
        domain : string, optional
            Domain of the points. The default is ``'cart'``.
        convention: string, optional
            Convention of points. The default is ``'right'``.
        unit : string, optional
            Unit of the points. The default is ``'met'`` for meters.
        show : bool, optional
            Show a plot of the coordinate points. The default is ``False``.
        atol : float, optional
            A tolerance that is added to `distance`. The default is`` 1e-15``.

        Returns
        -------
        index : numpy array of ints
            The locations of the neighbors in the getter methods (e.g.,
            ``cartesian``). Dimension as in :py:func:`~find_nearest_k`.
            Missing neighbors are indicated with ``csize``. Also see Notes
            below.
        mask : boolean numpy array
            Mask that contains ``True`` at the positions of the selected points
            and ``False`` otherwise. Mask is of shape ``cshape``.

        Notes
        -----
        :py:class:`scipy.spatial.cKDTree` is used for the search, which
        requires an
        (N, 3) array. The coordinate points in self are thus reshaped to
        (`csize`, 3) before they are passed to ``cKDTree``. The index that
        is returned refers to the reshaped coordinate points. To access the
        points for example use

        >>> points_reshaped = self.cartesian.reshape((self.csize, 3))
        >>> points_reshaped[index]

        Examples
        --------

        Find frontal points within a distance of 0.5 meters

        .. plot::

            >>> import pyfar as pf
            >>> coords = pf.Coordinates(np.arange(-5, 5), 0, 0)
            >>> result = coords.find_nearest_cart(2, 0, 0, 0.5, show=True)

        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of find_within method."),
                PyfarDeprecationWarning)

        # check the input
        assert distance >= 0, "distance must be >= 0"

        # get the points
        distance, index, mask = self._find_nearest(
            points_1, points_2, points_3,
            domain, convention, unit, show, distance, 'cart', atol)

        return index, mask

    def find_nearest_sph(self, points_1, points_2, points_3, distance,
                         domain='sph', convention='top_colat', unit='rad',
                         show=False, atol=1e-15):
        """
        This function will be deprecated in pyfar 0.8.0 in favor
        of the ``find_within`` method.
        Find coordinates within certain angular distance to the query points.

        Parameters
        ----------
        points_i : array like, number
            First, second and third coordinate of the points to which the
            nearest neighbors are searched.
        distance : number
            Great circle distance in degrees in which the nearest points are
            searched. Must be >= 0 and <= 180.
        domain : string, optional
            Domain of the input points. The default is ``'sph'``.
        convention: string, optional
            Convention of the input points. The default is ``'top_colat'``.
        unit: string, optional
            Unit of the input points. The default is ``'rad'``.
        show : bool, optional
            Show a plot of the coordinate points. The default is ``False``.
        atol : float, optional
            A tolerance that is added to `distance`. The default is ``1e-15``.

        Returns
        -------
        index : numpy array of ints
            The locations of the neighbors in the getter methods (e.g.,
            ``cartesian``). Dimension as in :py:func:`~find_nearest_k`.
            Missing neighbors are indicated with ``csize``. Also see Notes
            below.
        mask : boolean numpy array
            Mask that contains ``True`` at the positions of the selected points
            and ``False`` otherwise. Mask is of shape ``cshape``.

        Notes
        -----
        :py:class:`scipy.spatial.cKDTree` is used for the search, which
        requires an
        (N, 3) array. The coordinate points in self are thus reshaped to
        (`csize`, 3) before they are passed to ``cKDTree``. The index that
        is returned refers to the reshaped coordinate points. To access the
        points for example use

        ``points_reshaped = points.cartesian.reshape((points.csize, 3))``
        ``points_reshaped[index]``

        Examples
        --------

        Find top points within a distance of 45 degrees

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> coords = pf.Coordinates.from_spherical_elevation(
            >>>     0, np.arange(-90, 91, 10)*np.pi/180, 1)
            >>> result = coords.find_nearest_sph(0, np.pi/2, 1, 45, show=True)

        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0 in favor "
            "of find_within method."),
                PyfarDeprecationWarning)

        # check the input
        assert distance >= 0 and distance <= 180, \
            "distance must be >= 0 and <= 180."

        # get radius and check for equality
        radius = self.radius
        delta_radius = np.max(radius) - np.min(radius)
        if delta_radius > 1e-15:
            raise ValueError(
                "find_nearest_sph only works if all points have the same \
                radius. Differences are larger than 1e-15")

        # get the points
        distance, index, mask = self._find_nearest(
            points_1, points_2, points_3,
            domain, convention, unit, show, distance, 'sph', atol,
            np.max(radius))

        return index, mask

    def find_slice(self, coordinate: str, unit: str, value, tol=0,
                   show=False, atol=1e-15):
        """
        This function will be deprecated in pyfar 0.8.0. Use properties and
        slicing instead, e.g. ``coords = coords[coords.azimuth>=np.pi]``.

        Find a slice of the coordinates points.

        Parameters
        ----------
        coordinate : str
            Coordinate for slicing.
        unit : str
            Unit in which the value is passed
        value : number
            Value of the coordinate around which the points are sliced.
        tol : number, optional
            Tolerance for slicing. Points are sliced within the range
            ``[value-tol, value+tol]``. The default is ``0``.
        show : bool, optional
            Show a plot of the coordinate points. The default is ``False``.
        atol : number, optional
            A tolerance that is added to `tol`. The default is ``1e-15``.

        Returns
        -------
        index : tuple of numpy arrays
            The indices of the selected points as a tuple of arrays. The length
            of the tuple matches :py:func:`~cdim`. The length of each array
            matches the number of selected points.
        mask : boolean numpy array
            Mask that contains True at the positions of the selected points and
            False otherwise. Mask is of shape self.cshape.

        Notes
        -----
        `value` must be inside the range of the coordinate (see ``.systems``).
        However, `value` +/- `tol` may exceed the range.

        Examples
        --------

        Find horizontal slice of spherical coordinate system within a ring of
        +/- 10 degrees

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> coords = pf.Coordinates.from_spherical_elevation(
            >>>     np.arange(-30, 30, 5)*np.pi/180, 0, 1)
            >>> result = coords.find_slice('azimuth', 'deg', 0, 5, show=True)

        """
        warnings.warn((
            "This function will be deprecated in pyfar 0.8.0. Use properties"
            " and slicing instead."),
                PyfarDeprecationWarning)

        # check if the coordinate and unit exist
        domain, convention, index = self._exist_coordinate(coordinate, unit)

        # get type and range of coordinate
        c_info = self._systems()[domain][convention][coordinate]

        # convert input to radians
        value = value / 180 * np.pi if unit == 'deg' else value
        tol = tol / 180 * np.pi if unit == 'deg' else tol

        # check if  value is within the range of coordinate
        if c_info[0] in ["bound", "cyclic"]:
            assert c_info[1][0] <= value <= c_info[1][1], \
                f"'value' is {value} but must be in the range {c_info[1]}."

        # get the search range
        rng = [value - tol, value + tol]

        # wrap range if coordinate is cyclic
        if c_info[0] == 'cyclic':
            low = c_info[1][0]
            upp = c_info[1][1]
            if rng[0] < c_info[1][0] - atol:
                rng[0] = (rng[0] - low) % (upp - low) + low
            if rng[1] > c_info[1][1] + atol:
                rng[1] = (rng[1] - low) % (upp - low) + low

        # get the coordinates
        coords = eval(f"self.{coordinate}")

        # get the mask
        if rng[0] <= rng[1]:
            mask = (coords >= rng[0] - atol) & (coords <= rng[1] + atol)
        else:
            mask = (coords >= rng[0] - atol) | (coords <= rng[1] + atol)

        # plot all and returned points
        if show:
            self.show(mask)

        index = np.where(mask)

        return index, mask

    def rotate(self, rotation: str, value=None, degrees=True, inverse=False):
        """
        Rotate points stored in the object around the origin of coordinates.

        This is a wrapper for :py:class:`scipy.spatial.transform.Rotation`
        (see this class for more detailed information).

        Parameters
        ----------
        rotation : str
            ``'quat'``
                Rotation given by quaternions.
            ``'matrix'``
                Rotation given by matrixes.
            ``'rotvec'``
                Rotation using rotation vectors.
            ``'xyz'``
                Rotation using euler angles. Up to three letters. E.g., ``'x'``
                will rotate about the x-axis only, while ``'xz'`` will rotate
                about the x-axis and then about the z-axis. Use lower letters
                for extrinsic rotations (rotations about the axes of the
                original coordinate system xyz, which remains motionless) and
                upper letters for intrinsic rotations (rotations about the axes
                of the rotating coordinate system XYZ, solidary with the moving
                body, which changes its orientation after each elemental
                rotation).
        value : number, array like
            Amount of rotation in the format specified by `rotation` (see
            above).
        degrees : bool, optional
            Pass angles in degrees if using ``'rotvec'`` or euler angles
            (``'xyz'``). The default is ``True``. Use False to pass angles in
            radians.
        inverse : bool, optional
            Apply inverse rotation. The default is ``False``.

        Notes
        -----
        Points are converted to the cartesian right handed coordinate system
        for the rotation.

        Examples
        --------

        Get a coordinates object

        >>> import pyfar as pf
        >>> coords = pf.Coordinates(np.arange(-5, 5), 0, 0)

        Rotate 45 degrees about the y-axis using

        1. quaternions

        >>> coordinates.rotate('quat', [0 , 0.38268343, 0 , 0.92387953])

        2. a rotation matrix

        >>> coordinates.rotate('matrix',
        ...    [[ 0.70710678,  0 ,  0.70710678],
        ...     [ 0         ,  1 ,  0.        ],
        ...     [-0.70710678,  0 ,  0.70710678]])

        3. a rotation vector

        >>> coordinates.rotate('rotvec', [0, 45, 0])

        4. euler angles

        >>> coordinates.rotate('XYZ', [0, 45, 0])

        To see the result of the rotation use

        >>> coordinates.show()

        """

        # initialize rotation object
        if rotation == 'quat':
            rot = sp_rot.from_quat(value)
        elif rotation == 'matrix':
            rot = sp_rot.from_matrix(value)
        elif rotation == 'rotvec':
            if degrees:
                value = np.asarray(value) / 180 * np.pi
            rot = sp_rot.from_rotvec(value)
        elif not bool(re.search('[^x-z]', rotation.lower())):
            # only check if string contains xyz, everything else is checked in
            # from_euler()
            rot = sp_rot.from_euler(rotation, value, degrees)
        else:
            raise ValueError("rotation must be 'quat', 'matrix', 'rotvec', "
                             "or from ['x', 'y', 'z'] or ['X', 'Y', 'Z'] but "
                             f"is '{rotation}'")

        # current shape
        shape = self.cshape

        # apply rotation
        points = rot.apply(self.cartesian.reshape((self.csize, 3)), inverse)

        # set points
        self._set_points(
            points[:, 0].reshape(shape),
            points[:, 1].reshape(shape),
            points[:, 2].reshape(shape))

    def copy(self):
        """Return a deep copy of the Coordinates object."""
        return deepcopy(self)

    def _encode(self):
        """Return dictionary for the encoding."""
        return self.copy().__dict__

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective ``_encode`` counterpart."""
        obj = cls()
        obj.__dict__.update(obj_dict)
        return obj

    @staticmethod
    def _systems():
        """
        Get class internal information about all coordinate systems.

        Returns
        -------
        _systems : nested dictionary
            List all available coordinate systems.
            Key 0  - domain, e.g., 'cart'
            Key 1  - convention, e.g., 'right'
            Key 2a - 'short_description': string
            Key 2b - 'coordinates': ['coordinate_1',
                                     'coordinate_2',
                                     'coordinate_3']
            Key 2c - 'units': [['unit_1.1','unit_2.1','unit_3.1'],
                                            ...
                               ['unit_1.N','unit_2.N','unit_3.N']]
            Key 2d - 'description'
            Key 2e - 'front': positive x (for debugging, meters and radians)
            Key 2f - 'left' : positive y (for debugging, meters and radians)
            Key 2g - 'back' : negative x (for debugging, meters and radians)
            Key 2h - 'right': negative y (for debugging, meters and radians)
            Key 2i - 'up'   : positive z (for debugging, meters and radians)
            Key 2j - 'down' : negative z (for debugging, meters and radians)
            Key 2k,l,m - coordinate_1,2,3 : [type, [lower_lim, upper_lim]]
                         type can be 'unbound', 'bound', or 'cyclic'
        """

        # define coordinate systems
        _systems = {
            "cart": {
                "right": {
                    "description_short":
                        "Right handed cartesian coordinate system.",
                    "coordinates":
                        ["x", "y", "z"],
                    "units":
                        [["meters", "meters", "meters"]],
                    "description":
                        "Right handed cartesian coordinate system with x,y, "
                        "and z in meters.",
                    "positive_x": [1, 0, 0],
                    "positive_y": [0, 1, 0],
                    "negative_x": [-1, 0, 0],
                    "negative_y": [0, -1, 0],
                    "positive_z": [0, 0, 1],
                    "negative_z": [0, 0, -1],
                    "x": ["unbound", [-np.inf, np.inf]],
                    "y": ["unbound", [-np.inf, np.inf]],
                    "z": ["unbound", [-np.inf, np.inf]]}
            },
            "sph": {
                "top_colat": {
                    "description_short":
                        "Spherical coordinate system with North and South "
                        "Pole.",
                    "coordinates":
                        ["azimuth", "colatitude", "radius"],
                    "units":
                        [["radians", "radians", "meters"],
                         ["degrees", "degrees", "meters"]],
                    "description":
                        "The azimuth denotes the counter clockwise angle in "
                        "the x/y-plane with 0 pointing in positive x-"
                        "direction and pi/2 in positive y-direction. The "
                        "colatitude denotes the angle downwards from the z-"
                        "axis with 0 pointing in positive z-direction and pi "
                        "in negative z-direction. The azimuth and colatitude "
                        "can be in radians or degrees, the radius is always "
                        "in meters.",
                    "positive_x": [0, np.pi / 2, 1],
                    "positive_y": [np.pi / 2, np.pi / 2, 1],
                    "negative_x": [np.pi, np.pi / 2, 1],
                    "negative_y": [3 * np.pi / 2, np.pi / 2, 1],
                    "positive_z": [0, 0, 1],
                    "negative_z": [0, np.pi, 1],
                    "azimuth": ["cyclic", [0, 2 * np.pi]],
                    "colatitude": ["bound", [0, np.pi]],
                    "radius": ["bound", [0, np.inf]]},
                "top_elev": {
                    "description_short":
                        "Spherical coordinate system with North and South "
                        "Pole. Conform with AES69-2015: AES standard for file "
                        "exchange - Spatial acoustic data file format (SOFA).",
                    "coordinates":
                        ["azimuth", "elevation", "radius"],
                    "units":
                        [["radians", "radians", "meters"],
                         ["degrees", "degrees", "meters"]],
                    "description":
                        "The azimuth denotes the counter clockwise angle in "
                        "the x/y-plane with 0 pointing in positive x-"
                        "direction and pi/2 in positive y-direction. The "
                        "elevation denotes the angle upwards and downwards "
                        "from the x/y-plane with pi/2 pointing at positive "
                        "z-direction and -pi/2 pointing in negative z-"
                        "direction. The azimuth and elevation can be in "
                        "radians or degrees, the radius is always in meters.",
                    "positive_x": [0, 0, 1],
                    "positive_y": [np.pi / 2, 0, 1],
                    "negative_x": [np.pi, 0, 1],
                    "negative_y": [3 * np.pi / 2, 0, 1],
                    "positive_z": [0, np.pi / 2, 1],
                    "negative_z": [0, -np.pi / 2, 1],
                    "azimuth": ["cyclic", [0, 2 * np.pi]],
                    "elevation": ["bound", [-np.pi / 2, np.pi / 2]],
                    "radius": ["bound", [0, np.inf]]},
                "side": {
                    "description_short":
                        "Spherical coordinate system with poles on the "
                        "y-axis.",
                    "coordinates":
                        ["lateral", "polar", "radius"],
                    "units":
                        [["radians", "radians", "meters"],
                         ["degrees", "degrees", "meters"]],
                    "description":
                        "The lateral angle denotes the angle in the x/y-plane "
                        "with pi/2 pointing in positive y-direction and -pi/2 "
                        "in negative y-direction. The polar angle denotes the "
                        "angle in the x/z-plane with -pi/2 pointing in "
                        "negative z-direction, 0 in positive x-direction, "
                        "pi/2 in positive z-direction, pi in negative x-"
                        "direction. The polar and lateral angle can be in "
                        "radians and degree, the radius is always in meters.",
                    "positive_x": [0, 0, 1],
                    "positive_y": [np.pi / 2, 0, 1],
                    "negative_x": [0, np.pi, 1],
                    "negative_y": [-np.pi / 2, 0, 1],
                    "positive_z": [0, np.pi / 2, 1],
                    "negative_z": [0, -np.pi / 2, 1],
                    "lateral": ["bound", [-np.pi / 2, np.pi / 2]],
                    "polar": ["cyclic", [-np.pi / 2, np.pi * 3 / 2]],
                    "radius": ["bound", [0, np.inf]]},
                "front": {
                    "description_short":
                        "Spherical coordinate system with poles on the x-axis."
                        " Conform with AES56-2008 (r2019): AES standard on "
                        "acoustics - Sound source modeling.",
                    "coordinates":
                        ["phi", "theta", "radius"],
                    "units":
                        [["radians", "radians", "meters"],
                         ["degrees", "degrees", "meters"]],
                    "description":
                        "Phi denotes the angle in the y/z-plane with 0 "
                        "pointing in positive y-direction, pi/2 in positive "
                        "z-direction, pi in negative y-direction, and 3*pi/2 "
                        "in negative z-direction. Theta denotes the angle "
                        "measured from the x-axis with 0 pointing in positive "
                        "x-direction and pi in negative x-direction. Phi and "
                        "theta can be in radians and degrees, the radius is "
                        "always in meters.",
                    "positive_x": [0, 0, 1],
                    "positive_y": [0, np.pi / 2, 1],
                    "negative_x": [0, np.pi, 1],
                    "negative_y": [np.pi, np.pi / 2, 1],
                    "positive_z": [np.pi / 2, np.pi / 2, 1],
                    "negative_z": [3 * np.pi / 2, np.pi / 2, 1],
                    "phi": ["cyclic", [0, 2 * np.pi]],
                    "theta": ["bound", [0, np.pi]],
                    "radius": ["bound", [0, np.inf]]}
            },
            "cyl": {
                "top": {
                    "description_short":
                        "Cylindrical coordinate system along the z-axis.",
                    "coordinates":
                        ["azimuth", "z", "radius_z"],
                    "units":
                        [["radians", "meters", "meters"],
                         ["degrees", "meters", "meters"]],
                    "description":
                        "The azimuth denotes the counter clockwise angle in "
                        "the x/y-plane with 0 pointing in positive x-"
                        "direction and pi/2 in positive y-direction. The "
                        "height is given by z, and radius_z denotes the "
                        "radius measured orthogonal to the z-axis.",
                    "positive_x": [0, 0, 1],
                    "positive_y": [np.pi / 2, 0, 1],
                    "negative_x": [np.pi, 0, 1],
                    "negative_y": [3 * np.pi / 2, 0, 1],
                    "positive_z": [0, 1, 0],
                    "negative_z": [0, -1, 0],
                    "azimuth": ["cyclic", [0, 2 * np.pi]],
                    "z": ["unbound", [-np.inf, np.inf]],
                    "radius_z": ["bound", [0, np.inf]]}
            }
        }

        return _systems

    def _exist_system(self, domain=None, convention=None, unit=None):
        """
        Check if a coordinate system exists and throw an error if it does not.

        The coordinate systems are defined in self._systems.

        Parameters
        ----------
        domain : string
            Specify the domain of the coordinate system, e.g., 'cart'.
        convention : string
            The convention of the coordinate system, e.g., 'top_colat'
        units: string
            The unit of the coordinate system (rad, deg, or met for radians,
            degrees, or meters)
        """

        if domain is None:
            raise ValueError('The domain must be specified')

        # get available coordinate systems
        systems = self._systems()

        # check if domain exists
        assert domain in systems or domain is None, \
            f"{domain} does not exist. Domain must be one of the following: "\
            f"{', '.join(list(systems))}."

        # check if convention exists in domain
        if convention is not None:
            assert convention in systems[domain] or convention is None, \
                f"{convention} does not exist in {domain}. Convention must "\
                f"be one of the following: {', '.join(list(systems[domain]))}."

        # check if units exist
        if unit is not None:
            # get first convention in domain
            # (units are the same within a domain)
            if convention is None:
                convention = list(systems[domain])[0]

            cur_units = [u[0][0:3] for u in
                         systems[domain][convention]['units']]
            assert unit in cur_units, \
                f"{unit} does not exist in {domain} convention "\
                f"Unit must be one of the following: {', '.join(cur_units)}."

    def _exist_coordinate(self, coordinate, unit):
        """
        Check if coordinate and unit exist.

        Returns domain and convention, and the index of coordinate if
        coordinate and unit exists and raises a value error otherwise.
        """
        # get all systems
        systems = self._systems()

        # find coordinate and unit in systems
        for domain in systems:
            for convention in systems[domain]:
                if coordinate in systems[domain][convention]['coordinates']:
                    # get position of coordinate
                    index = systems[domain][convention]['coordinates'].\
                        index(coordinate)
                    # get possible units
                    units = [u[index][0:3] for u in
                             systems[domain][convention]['units']]
                    # return or raise ValueError
                    if unit in units:
                        return domain, convention, index

        raise ValueError(
            (f"'{coordinate}' in '{unit}' does not exist. See "
             "self.systems() for a list of possible "
             "coordinates and units"))

    def _make_system(self, domain=None, convention=None, unit=None):
        """
        Make and return class internal information about coordinate system.
        """

        # check if coordinate system exists
        self._exist_system(domain, convention, unit)

        # get the new system
        system = self._systems()
        if convention is None:
            convention = list(system[domain])[0]
        system = system[domain][convention]

        # get the units
        if unit is not None:
            units = [units for units in system['units']
                     if unit == units[0][0:3]]
            units = units[0]
        else:
            units = system['units'][0]
            unit = units[0][0:3]

        # add class internal keys
        system['domain'] = domain
        system['convention'] = convention
        system['unit'] = unit
        system['units'] = units

        return system

    def _set_points(self, x, y, z):
        """
        Check points and convert to matrix.

        Parameters
        ----------
        convert : boolean, optional
            Set self._points if convert = True. Return points as
            matrix otherwise. The default is False.
        system: dict, optional
            The coordinate system against which the range of the points are
            checked as returned from self._make_system. If system = None
            self._system is used.

        Set self._points, which is an atleast_2d numpy array of shape
        [L,M,...,N, 3].
        """
        # cast to numpy array
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))

        # shapes of non scalar entries
        shapes = [p.shape for p in [x, y, z] if p.ndim != 1 or p.shape[0] > 1]

        # repeat scalar entries if non-scalars exists
        if len(shapes):
            if x.size == 1:
                x = np.tile(x, shapes[0])
            if y.size == 1:
                y = np.tile(y, shapes[0])
            if z.size == 1:
                z = np.tile(z, shapes[0])

        # check for equal shape
        assert (x.shape == y.shape) and (x.shape == z.shape), \
            "x, y, and z must be scalar or of the \
            same shape."

        # set values
        self._x = x
        self._y = y
        self._z = z

    def _set_weights(self, weights):
        """
        Check and set sampling weights.

        Set self._weights, which is an atleast_1d numpy array of shape
        [L,M,...,N].
        """

        # check input
        if weights is None:
            self._weights = weights
            return

        # cast to np.array
        weights = np.asarray(weights, dtype=np.float64)

        # reshape according to self._points
        assert weights.size == self.csize, \
            "weights must have same size as self.csize"
        weights = weights.reshape(self.cshape)

        # set class variable
        self._weights = weights

    def _find_nearest(self, points_1, points_2, points_3,
                      domain, convention, unit, show,
                      value, measure, atol=1e-15, radius=None):

        # get KDTree
        kdtree = self._make_kdtree()

        # get target point in cartesian coordinates
        coords = Coordinates(points_1, points_2, points_3,
                             domain, convention, unit)
        points = coords.cartesian

        # query nearest neighbors
        points = points.flatten() if coords.csize == 1 else points

        # get the points depending on measure and value
        if measure == 'k':
            # nearest points
            distance, index = kdtree.query(points, k=value)
        elif measure == 'cart':
            # points within euclidean distance
            index = kdtree.query_ball_point(points, value + atol)
            distance = None
        elif measure == 'sph':
            # get radius and check for equality
            radius = self.radius
            delta_radius = np.max(radius) - np.min(radius)
            if delta_radius > 1e-15:
                raise ValueError(
                    "find_nearest_sph only works if all points have the same \
                    radius. Differences are larger than 1e-15")
            radius = np.max(radius)

            # convert great circle to euclidean distance
            x, y, z = sph2cart([0, value / 180 * np.pi],
                               [np.pi / 2, np.pi / 2],
                               [radius, radius])
            value = np.sqrt((x[0] - x[1])**2
                            + (y[0] - y[1])**2
                            + (z[0] - z[1])**2)
            # points within great circle distance
            index = kdtree.query_ball_point(points, value + atol)
            distance = None

        # mask for scatter plot
        mask = np.full((self.csize), False)
        mask[index] = True
        mask = mask.reshape(self.cshape)

        # plot all and returned points
        if show:
            self.show(mask)

        return distance, index, mask

    def _make_kdtree(self):
        """Make a numpy KDTree for fast search of nearest points."""

        xyz = self.cartesian
        kdtree = cKDTree(xyz.reshape((self.csize, 3)))

        return kdtree

    def __getitem__(self, index):
        """Return copied slice of Coordinates object at index."""

        new = self.copy()
        # slice points
        new._x = np.atleast_1d(new._x[index])
        new._y = np.atleast_1d(new._y[index])
        new._z = np.atleast_1d(new._z[index])
        # slice weights
        if new._weights is not None:
            new._weights = new._weights[index]

        return new

    def __array__(self):
        """Instances of Coordinates behave like `numpy.ndarray`, array_like."""
        # copy to avoid changing the coordinate system of the original object
        return self.copy().cartesian

    def __repr__(self):
        """Get info about Coordinates object."""
        # object type
        if self.cshape != (0,):
            obj = f"{self.cdim}D Coordinates object with {self.csize} points "\
                  f"of cshape {self.cshape}"
        else:
            obj = "Empty Coordinates object"

        # join information
        _repr = obj + "\n"

        # check for sampling weights
        if self._weights is None:
            _repr += "\nDoes not contain sampling weights"
        else:
            _repr += "\nContains sampling weights"

        # check for sh_order
        if self._sh_order is not None:
            _repr += f"\nSpherical harmonic order: {self._sh_order}"

        # check for comment
        if self._comment != "":
            _repr += f"\nComment: {self._comment}"

        return _repr

    def __eq__(self, other):
        """Check for equality of two objects."""
        # return not deepdiff.DeepDiff(self, other)
        if self.cshape != other.cshape:
            return False
        eq_x = self._x == other._x
        eq_y = self._y == other._y
        eq_z = self._z == other._z
        eq_weights = self._weights == other._weights
        eq_sh_order = self._sh_order == other._sh_order
        eq_comment = self._comment == other._comment
        eq_system = self._system == other._system
        if self._x.shape == ():
            return eq_x & eq_y & eq_z & eq_weights & eq_comment \
                & eq_sh_order & eq_system
        return (eq_x & eq_y & eq_z).all() & eq_weights & eq_comment \
            & eq_sh_order & eq_system

    def _check_empty(self):
        """check if object is empty"""
        if self.cshape == (0,):
            raise ValueError('Object is empty.')


def cart2sph(x, y, z):
    """
    Transforms from Cartesian to spherical coordinates.

    Spherical coordinates follow the common convention in Physics/Mathematics.
    The `colatitude` is measured downwards from the z-axis and is 0 at the
    North Pole and pi at the South Pole. The `azimuth` is 0 at positive
    x-direction and pi/2 at positive y-direction (counter clockwise rotation).

    Cartesian coordinates follow the right hand rule.

    .. math::

        azimuth &= \\arctan(\\frac{y}{x}),

        colatitude &= \\arccos(\\frac{z}{r}),

        radius &= \\sqrt{x^2 + y^2 + z^2}

    .. math::

        0 < azimuth < 2 \\pi,

        0 < colatitude < \\pi

    Parameters
    ----------
    x : numpy array, number
        X values
    y : numpy array, number
        Y values
    z : numpy array, number
        Z values

    Returns
    -------
    azimuth : numpy array, number
        Azimuth values
    colatitude : numpy array, number
        Colatitude values
    radius : numpy array, number
        Radii

    Notes
    -----
    To ensure proper handling of the azimuth angle, the
    :py:data:`numpy.arctan2` implementation from numpy is used.
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    z_div_r = np.divide(
        z, radius, out=np.zeros_like(radius, dtype=float), where=radius != 0)
    colatitude = np.arccos(z_div_r)
    azimuth = np.mod(np.arctan2(y, x), 2 * np.pi)

    return azimuth, colatitude, radius


def sph2cart(azimuth, colatitude, radius):
    """
    Transforms from spherical to Cartesian coordinates.

    Spherical coordinates follow the common convention in Physics/Mathematics.
    The `colatitude` is measured downwards from the z-axis and is 0 at the
    North Pole and pi at the South Pole. The `azimuth` is 0 at positive
    x-direction and pi/2 at positive y-direction (counter clockwise rotation).

    Cartesian coordinates follow the right hand rule.

    .. math::

        x &= radius \\cdot \\sin(colatitude) \\cdot \\cos(azimuth),

        y &= radius \\cdot \\sin(colatitude) \\cdot \\sin(azimuth),

        z &= radius \\cdot \\cos(colatitude)

    .. math::

        0 < azimuth < 2 \\pi

        0 < colatitude < \\pi


    Parameters
    ----------
    azimuth : numpy array, number
        Azimuth values
    colatitude : numpy array, number
        Colatitude values
    radius : numpy array, number
        Radii

    Returns
    -------
    x : numpy array, number
        X values
    y : numpy array, number
        Y values
    z : numpy array, number
        Z vales
    """
    azimuth = np.atleast_1d(azimuth)
    colatitude = np.atleast_1d(colatitude)
    radius = np.atleast_1d(radius)

    r_sin_cola = radius * np.sin(colatitude)
    x = r_sin_cola * np.cos(azimuth)
    y = r_sin_cola * np.sin(azimuth)
    z = radius * np.cos(colatitude)

    x[np.abs(x) < np.finfo(x.dtype).eps] = 0
    y[np.abs(y) < np.finfo(y.dtype).eps] = 0
    z[np.abs(z) < np.finfo(x.dtype).eps] = 0

    return x, y, z


def cart2cyl(x, y, z):
    """
    Transforms from Cartesian to cylindrical coordinates.

    Cylindrical coordinates follow the convention that the `azimuth` is 0 at
    positive x-direction and pi/2 at positive y-direction (counter clockwise
    rotation). The `height` is identical to the z-coordinate and the `radius`
    is measured orthogonal from the z-axis.

    Cartesian coordinates follow the right hand rule.

    .. math::

        azimuth &= \\arctan(\\frac{y}{x}),

        height &= z,

        radius &= \\sqrt{x^2 + y^2},

    .. math::

        0 < azimuth < 2 \\pi

    Parameters
    ----------
    x : numpy array, number
        X values
    y : numpy array, number
        Y values
    z : numpy array, number
        Z values

    Returns
    -------
    azimuth : numpy array, number
        Azimuth values
    height : numpy array, number
        Height values
    radius : numpy array, number
        Radii

    Notes
    -----
    To ensure proper handling of the azimuth angle, the
    :py:data:`numpy.arctan2` implementation from numpy is used.
    """
    azimuth = np.mod(np.arctan2(y, x), 2 * np.pi)
    if isinstance(z, np.ndarray):
        height = z.copy()
    else:
        height = z
    radius = np.sqrt(x**2 + y**2)

    return azimuth, height, radius


def cyl2cart(azimuth, height, radius):
    """
    Transforms from cylindrical to Cartesian coordinates.

    Cylindrical coordinates follow the convention that the `azimuth` is 0 at
    positive x-direction and pi/2 at positive y-direction (counter clockwise
    rotation). The `height` is identical to the z-coordinate and the `radius`
    is measured orthogonal from the z-axis.

    Cartesian coordinates follow the right hand rule.

    .. math::

        x &= radius \\cdot \\cos(azimuth),

        y &= radius \\cdot \\sin(azimuth),

        z &= height

    .. math::

        0 < azimuth < 2 \\pi

    Parameters
    ----------
    azimuth : numpy array, number
        Azimuth values
    height : numpy array, number
        Height values
    radius : numpy array, number
        Radii

    Returns
    -------
    x : numpy array, number
        X values
    y : numpy array, number
        Y values
    z : numpy array, number
        Z values

    Notes
    -----
    To ensure proper handling of the azimuth angle, the
    :py:data:`numpy.arctan2` implementation from numpy is used.
    """
    azimuth = np.atleast_1d(azimuth)
    height = np.atleast_1d(height)
    radius = np.atleast_1d(radius)

    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)
    if isinstance(height, np.ndarray):
        z = height.copy()
    else:
        z = height

    x[np.abs(x) < np.finfo(x.dtype).eps] = 0
    y[np.abs(y) < np.finfo(y.dtype).eps] = 0
    z[np.abs(z) < np.finfo(x.dtype).eps] = 0

    return x, y, z


def rad2deg(coordinates, domain='spherical'):
    """
    Convert a copy of coordinates in radians to degree

    Parameters
    ----------
    coordinates : array like
        N-dimensional array of shape `(..., 3)`.
    domain : str, optional
        Specifies what data are contained in `coordinates`

        ``'spherical'``
            Spherical coordinates with angles contained in
            ``coordinates[..., 0:2]`` and radii in ``coordinates[..., 2]``.
            The radii are ignored during the conversion.
        ``'cylindrical'``
            Cylindrical coordinates with angles contained in
            ``coordinates[..., 0]``, heights contained in
            ``coordinates[..., 1]``, and radii in ``coordinates[..., 2]``.
            The heights and radii are ignored during the conversion.


    Returns
    -------
    coordinates : numpy array
        The converted coordinates of the same shape as the input data.
    """
    return _convert_angles(coordinates, domain, 180/np.pi)


def deg2rad(coordinates, domain='spherical'):
    """
    Convert a copy of coordinates in degree to radians

    Parameters
    ----------
    coordinates : array like
        N-dimensional array of shape `(..., 3)`.
    domain : str, optional
        Specifies what data are contained in `coordinates`

        ``'spherical'``
            Spherical coordinates with angles contained in
            ``coordinates[..., 0:2]`` and radii in ``coordinates[..., 2]``.
            The radii are ignored during the conversion.
        ``'cylindrical'``
            Cylindrical coordinates with angles contained in
            ``coordinates[..., 0]``, heights contained in
            ``coordinates[..., 1]``, and radii in ``coordinates[..., 2]``.
            The heights and radii are ignored during the conversion.


    Returns
    -------
    coordinates : numpy array
        The converted coordinates of the same shape as the input data.
    """
    return _convert_angles(coordinates, domain, np.pi/180)


def _convert_angles(coordinates, domain, factor):
    """Private function called by rad2deg and deg2rad"""

    # check coordinates
    coordinates = np.atleast_2d(coordinates).astype(float)
    if coordinates.shape[-1] != 3:
        raise ValueError(('coordinates must be of shape (..., 3) but are of '
                          f'shape {coordinates.shape}'))

    # check domain and create mask
    if domain == 'spherical':
        mask = [True, True, False]
    elif domain == 'cylindrical':
        mask = [True, False, False]
    else:
        raise ValueError(("domain must be  'spherical' or 'cylindrical' but "
                          f"is {domain}"))

    # convert data
    converted = coordinates.copy()
    converted[..., mask] = converted[..., mask] * factor

    return converted


def _check_array_limits(values, lower_limit, upper_limit, variable_name=None):
    """
    Values will be clipped to its range if deviations are below 2 eps
    for 32 bit float numbers otherwise Error is raised.

    Notes
    -----
    This is mostly used for the colatitude angle.

    Parameters
    ----------
    values : np.ndarray
        Input array angle
    lower_limit : float
        Lower limit for angle definition
    upper_limit : float
        Upper limit for angle definition
    variable_name : string
        Name of variable, just relevant for error message. 'value' by default.

    Returns
    -------
    values : np.ndarray
        Clipped input values
    """
    if variable_name is None:
        variable_name = 'value'
    if any(values < lower_limit):
        mask = values < lower_limit
        eps = np.finfo(float).eps
        if any(values[mask]+2*eps < lower_limit):
            raise ValueError(
                f'one or more {variable_name} are below '
                f'{lower_limit} including 2 eps')
        values[mask] = lower_limit
    if any(values > upper_limit):
        mask = values > upper_limit
        eps = np.finfo(float).eps
        if any(values[mask] + 2*eps > upper_limit):
            raise ValueError(
                f'one or more {variable_name} are above '
                f'{upper_limit} including 2 eps')
        values[mask] = upper_limit
    return values
