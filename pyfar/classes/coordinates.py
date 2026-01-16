r"""
The following introduces the
:py:func:`Coordinates class <pyfar.Coordinates>`
and the coordinate systems that are available in pyfar. Available sampling
schemes are listed at :py:mod:`spharpy.samplings <spharpy.samplings>`.
:ref:`Examples <gallery:/gallery/interactive/pyfar_coordinates.ipynb>` for
working with Coordinates objects are part of the pyfar gallery.

Different coordinate systems are frequently used in acoustics research and
handling sampling points and different systems can be cumbersome. The
Coordinates class was designed with this in mind. It stores coordinates in
cartesian coordinates internally and can convert to all coordinate systems
listed below. Additionally, the class can query and plot coordinates
points. Addition and subtraction are supported with numbers and Coordinates
objects, while multiplication and division are supported with numbers only.
All arithmetic operations are performed element-wise on Cartesian coordinates
using the appropriate operator.
Functions for converting coordinates not stored in a Coordinates object
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

.. note::
    All coordinates are returned as copies of the internal data. This means
    that for example ``coordinates.x[0] = 0`` does not change
    ``coordinates.x``. This can be done using

    .. code-block:: python

       new_x = coordinates.x
       new_x[0] = 0
       coordinates.x = new_x

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Coordinate
     - Descriptions
   * - :py:func:`~pyfar.Coordinates.x`,
       :py:func:`~pyfar.Coordinates.y`,
       :py:func:`~pyfar.Coordinates.z`
     - x, y, z coordinate of a right handed Cartesian coordinate system in
       meter (:math:`-\infty` < x,y,z < :math:`\infty`).
   * - :py:func:`~pyfar.Coordinates.azimuth`
     - Counter clock-wise angle in the x-y plane of the right handed Cartesian
       coordinate system in radians. :math:`0` radians are defined in positive
       x-direction, :math:`\pi/2` radians in positive y-direction and so on
       (:math:`-\infty` < azimuth < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`~pyfar.Coordinates.colatitude`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians colatitude are defined in positive
       z-direction, :math:`\pi/2` radians in positive x-direction, and
       :math:`\pi` in negative z-direction
       (:math:`0\leq` colatitude :math:`\leq\pi`). The colatitude is a
       variation of the elevation angle.
   * - :py:func:`~pyfar.Coordinates.elevation`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians elevation are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction, and
       :math:`-\pi/2` in negative z-direction
       (:math:`-\pi/2\leq` elevation :math:`\leq\pi/2`). The elevation is a
       variation of the colatitude.
   * - :py:func:`~pyfar.Coordinates.lateral`
     - Counter clock-wise angle in the x-y plane of the right handed Cartesian
       coordinate system in radians. :math:`0` radians are defined in positive
       x-direction, :math:`\pi/2` radians in positive y-direction and
       :math:`-\pi/2` in negative y-direction
       (:math:`-\pi/2\leq` lateral :math:`\leq\pi/2`).
   * - :py:func:`~pyfar.Coordinates.polar`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians polar angle are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction,
       :math:`\pi` in negative x-direction and so on
       (:math:`-\infty` < polar < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`~pyfar.Coordinates.frontal`
     - Angle in the y-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians frontal angle are defined in positive
       y-direction, :math:`\pi/2` radians in positive z-direction,
       :math:`\pi` in negative y-direction and so on
       (:math:`-\infty` < frontal < :math:`\infty`, :math:`2\pi`-cyclic).
   * - :py:func:`~pyfar.Coordinates.upper`
     - Angle in the x-z plane of the right handed Cartesian coordinate system
       in radians. :math:`0` radians upper angle are defined in positive
       x-direction, :math:`\pi/2` radians in positive z-direction, and
       :math:`\pi` in negative x-direction
       (:math:`0\leq` upper :math:`\leq\pi`).
   * - :py:func:`~pyfar.Coordinates.radius`
     - Distance to the origin of the right handed Cartesian coordinate system
       in meters (:math:`0` < radius < :math:`\infty`).
   * - :py:func:`~pyfar.Coordinates.rho`
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

import pyfar as pf


class Coordinates():
    r"""
    Create a Coordinates class object from a set of points in the
    right-handed cartesian coordinate system. See
    see :ref:`coordinate_systems` and :ref:`coordinates` for
    more information.

    If you want to initialize in another
    domain use :py:func:`from_spherical_colatitude`,
    :py:func:`from_spherical_elevation`, :py:func:`from_spherical_front`,
    :py:func:`from_spherical_side`, or :py:func:`from_cylindrical`
    instead. For conversions from or into degree
    use :py:func:`deg2rad` and :py:func:`rad2deg`.


    Parameters
    ----------
    x : ndarray, number
        X coordinate of a right handed Cartesian coordinate system in
        meters (:math:`-\infty` < x < :math:`\infty`).
    y : ndarray, number
        Y coordinate of a right handed Cartesian coordinate system in
        meters (:math:`-\infty` < y < :math:`\infty`).
    z : ndarray, number
        Z coordinate of a right handed Cartesian coordinate system in
        meters (:math:`-\infty` < z < :math:`\infty`).
    weights: array like, number, optional
        Weighting factors for coordinate points. The `shape` of the array
        must match the `shape` of the individual coordinate arrays.
        The default is ``None``.
    comment : str, optional
        Comment about the stored coordinate points. The default is
        ``""``, which initializes an empty string.
    """

    _data: np.array = np.empty
    """Internal storage for coordinates and optional weights.

    Stored as a numpy array with shape (..., 3) for [x, y, z] or (..., 4)
    for [x, y, z, weights]. The last axis contains the components.
    """

    _comment: str = None
    """Comment about the stored coordinate object."""

    def __init__(
            self, x: np.array = np.asarray([]),
            y: np.array = np.asarray([]),
            z: np.array = np.asarray([]),
            weights: np.array = None,
            comment: str = "") -> None:

        # init empty object
        super(Coordinates, self).__init__()

        # init cartesian coordinates and weights
        self._set_points_weights(x, y, z, weights)

        # save meta data
        self._comment = comment

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
            meters (:math:`-\infty` < x < :math:`\infty`).
        y : ndarray, number
            Y coordinate of a right handed Cartesian coordinate system in
            meters ():math:`-\infty` < y < :math:`\infty`).
        z : ndarray, number
            Z coordinate of a right handed Cartesian coordinate system in
            meters (:math:`-\infty` < z < :math:`\infty`).
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

    @property
    def weights(self):
        """Get sampling weights."""
        if self._data.shape[-1] == 3:
            return None
        return self._data[..., 3].copy()

    @weights.setter
    def weights(self, value):
        """Set sampling weights."""
        self._set_points_weights(self.x, self.y, self.z, value)

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
        if self.csize:
            return self._data[..., 0].shape
        else:
            return (0,)

    @property
    def cdim(self):
        """
        Return channel dimension.

        The channel dimension gives the number of dimensions of the coordinate
        points excluding the last dimension.
        """
        if self._data[..., 0].size:
            return self._data[..., 0].ndim
        else:
            return 0

    @property
    def csize(self):
        """
        Return channel size.

        The channel size gives the number of points stored in the coordinates
        object.
        """
        return self._data[..., 0].size

    @property
    def cartesian(self):
        """
        Returns :py:func:`x`, :py:func:`y`, :py:func:`z`.
        Right handed cartesian coordinate system. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.
        """
        return self._data[..., :3].copy()

    @cartesian.setter
    def cartesian(self, value):
        self._set_points_weights(
            value[..., 0], value[..., 1], value[..., 2], self.weights)

    @property
    def spherical_elevation(self):
        """
        Spherical coordinates according to the top pole elevation coordinate
        system. :py:func:`azimuth`, :py:func:`elevation`,
        :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.
        """
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
        self._set_points_weights(x, y, z, self.weights)

    @property
    def spherical_colatitude(self):
        """
        Spherical coordinates according to the top pole colatitude coordinate
        system.
        Returns :py:func:`azimuth`, :py:func:`colatitude`,
        :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.
        """
        azimuth, colatitude, radius = cart2sph(self.x, self.y, self.z)
        return np.atleast_2d(np.moveaxis(
            np.array([azimuth, colatitude, radius]), 0, -1))

    @spherical_colatitude.setter
    def spherical_colatitude(self, value):
        value[..., 1] = _check_array_limits(
            value[..., 1], 0, np.pi, 'colatitude angle')
        x, y, z = sph2cart(value[..., 0], value[..., 1], value[..., 2])
        self._set_points_weights(x, y, z, self.weights)

    @property
    def spherical_side(self):
        """
        Spherical coordinates according to the side pole coordinate system.
        Returns :py:func:`lateral`, :py:func:`polar`, :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.
        """
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
        self._set_points_weights(x, y, z, self.weights)

    @property
    def spherical_front(self):
        """
        Spherical coordinates according to the frontal pole coordinate system.
        Returns :py:func:`frontal`, :py:func:`upper`, :py:func:`radius`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.
        """

        frontal, upper, radius = cart2sph(self.y, self.z, self.x)
        return np.atleast_2d(np.moveaxis(
            np.array([frontal, upper, radius]), 0, -1))

    @spherical_front.setter
    def spherical_front(self, value):
        value[..., 1] = _check_array_limits(
            value[..., 1], 0, np.pi, 'frontal angle')
        y, z, x = sph2cart(value[..., 0], value[..., 1], value[..., 2])
        self._set_points_weights(x, y, z, self.weights)

    @property
    def cylindrical(self):
        """
        Cylindrical coordinates.
        Returns :py:func:`azimuth`, :py:func:`z`, :py:func:`rho`. See
        see :ref:`coordinate_systems` and :ref:`coordinates` for
        more information.
        """
        azimuth, z, rho = cart2cyl(self.x, self.y, self.z)
        return np.atleast_2d(np.moveaxis(
            np.array([azimuth, z, rho]), 0, -1))

    @cylindrical.setter
    def cylindrical(self, value):
        x, y, z = cyl2cart(value[..., 0], value[..., 1], value[..., 2])
        self._set_points_weights(x, y, z, self.weights)

    @property
    def x(self):
        r"""
        X coordinate of a right handed Cartesian coordinate system in meters
        (:math:`-\infty` < x < :math:`\infty`).
        """
        self._check_empty()
        return self._data[..., 0].copy()

    @x.setter
    def x(self, value):
        self._set_points_weights(value, self.y, self.z, self.weights)

    @property
    def y(self):
        r"""
        Y coordinate of a right handed Cartesian coordinate system in meters
        (:math:`-\infty` < y < :math:`\infty`).
        """
        self._check_empty()
        return self._data[..., 1].copy()

    @y.setter
    def y(self, value):
        self._set_points_weights(self.x, value, self.z, self.weights)

    @property
    def z(self):
        r"""
        Z coordinate of a right handed Cartesian coordinate system in meters
        (:math:`-\infty` < z < :math:`\infty`).
        """
        self._check_empty()
        return self._data[..., 2].copy()

    @z.setter
    def z(self, value):
        self._set_points_weights(self.x, self.y, value, self.weights)

    @property
    def rho(self):
        r"""
        Radial distance to the the z-axis of the right handed Cartesian
        coordinate system (:math:`0` < rho < :math:`\infty`).
        """
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
        in meters (:math:`0` < radius < :math:`\infty`).
        """
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
        (:math:`-\infty` < azimuth < :math:`\infty`, :math:`2\pi`-cyclic).
        """
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
        variation of the colatitude.
        """
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
        variation of the elevation angle.
        """
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
        (:math:`-\infty` < frontal < :math:`\infty`, :math:`2\pi`-cyclic).
        """
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
        (:math:`0\leq` upper :math:`\leq\pi`).
        """
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
        (:math:`-\pi/2\leq` lateral :math:`\leq\pi/2`).
        """
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
        (:math:`-\infty` < polar < :math:`\infty`, :math:`2\pi`-cyclic).
        """
        return self.spherical_side[..., 1]

    @polar.setter
    def polar(self, polar):
        spherical_side = self.spherical_side
        spherical_side[..., 1] = polar
        self.spherical_side = spherical_side

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
            >>> import numpy as np
            >>> coords = pf.Coordinates.from_spherical_elevation(
            >>>     np.arange(0, 360, 10)*np.pi/180, 0, 1)
            >>> to_find = pf.Coordinates(1, 0, 0)
            >>> index, distance = coords.find_nearest(to_find)
            >>> ax = coords.show(index)
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
            radius = np.concatenate(
                (self.radius.flatten(), find.radius.flatten()))
            delta_radius = np.max(radius) - np.min(radius)
            if delta_radius > radius_tol:
                raise ValueError(
                    f"find_nearest only works if all points have the same \
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
                    index[kk] = (index_multi[kk], )
            else:
                index = (index, )
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
        Find all point with 0.2 m distance from the frontal point

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>> coords = pf.Coordinates.from_spherical_elevation(
            >>>     np.arange(0, 360, 5)*np.pi/180, 0, 1)
            >>> find = pf.Coordinates(.2, 0, 0)
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
            radius = np.concatenate(
                (self.radius.flatten(), find.radius.flatten()))
            delta_radius = np.max(radius) - np.min(radius)
            if delta_radius > radius_tol:
                raise ValueError(
                    "find_within only works if all points have the same "
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
                    index[i] = (index[i], )
            else:
                index = (index, )

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
            # only check if string contains xyz, everything else is checked
            # in from_euler()
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
        self._set_points_weights(
            points[:, 0].reshape(shape),
            points[:, 1].reshape(shape),
            points[:, 2].reshape(shape),
            self.weights,
        )

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

    def _set_points_weights(self, x, y, z, weights):
        """
        Convert all points into at least 1d numpy arrays and broadcast them
        to the same shape by calling ``_check_points``, then check the weights
        by calling ``_check_weights`` and reshape them to the cshape if needed.
        Finally, ``_data`` is set to ``x``, ``y``, ``z`` and optional
        ``weights``.

        Parameters
        ----------
        x : array like, number
            First coordinate of the points in cartesian.
        y : array like, number
            Second coordinate of the points in cartesian.
        z : array like, number
            Third coordinate of the points in cartesian.
        weights : array like, number, None
            the weights for each point, should be broadcastable to
            ``x``, ``y`` and ``z``.
        """
        # check input
        x, y, z = self._check_points(x, y, z)
        weights = self._check_weights(weights, x.shape)

        # set values
        if weights is None:
            self._data = np.stack(
                [x, y, z],
                axis=-1)
        else:
            self._data = np.stack(
                [x, y, z, weights],
                axis=-1)


    def _check_points(self, x, y, z):
        """
        Convert all coordinates into at least 1d float64 arrays and
        broadcast the shape of all three coordinates to the same shape.
        The returned arrays are explicitly set to be writeable, to make sure
        that the class does not become read-only.

        Parameters
        ----------
        x : array like, number
            First coordinate of the points in cartesian.
        y : array like, number
            Second coordinate of the points in cartesian.
        z : array like, number
            Third coordinate of the points in cartesian.

        Returns
        -------
        x : np.ndarray[float64]
            broadcasted first coordinate of the points in cartesian.
        y : np.ndarray[float64]
            broadcasted second coordinate of the points in cartesian.
        z : np.ndarray[float64]
            broadcasted third coordinate of the points in cartesian.
        """
        # cast to numpy array
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))

        # determine shape
        shapes = np.broadcast_shapes(x.shape, y.shape, z.shape)

        # broadcast to same shape
        x = np.broadcast_to(x, shapes)
        y = np.broadcast_to(y, shapes)
        z = np.broadcast_to(z, shapes)

        # set writeable, to make sure that the class does not become read-only
        x.setflags(write=True)
        y.setflags(write=True)
        z.setflags(write=True)

        return x, y, z

    def _check_weights(self, weights, cshape=None):
        """
        Convert weights into float64 numpy array and check versus the csize.
        It will be reshaped to the cshape if the csize matches.

        Parameters
        ----------
        weights : array like, number
            the weights for each point, should be of size of self.csize.
        cshape : tuple, None
            The shape to which the weights should be reshaped. If None,
            self.cshape is used.

        Returns
        -------
        weights : np.ndarray[float64], None
            The weights reshaped to the cshape of the coordinates if not None.
            Otherwise None.
        """
        # if None no further checks are needed
        if weights is None:
            return weights

        # cast to np.array
        weights = np.asarray(weights, dtype=np.float64)

        # reshape to cshape
        cshape = self.cshape if cshape is None else cshape
        try:
            weights = np.broadcast_to(weights, cshape).copy()
        except ValueError as e:
            raise ValueError(
                f"weights cannot be broadcasted cshape {cshape}") from e

        return weights

    def _make_kdtree(self):
        """Make a numpy KDTree for fast search of nearest points."""

        xyz = self.cartesian
        kdtree = cKDTree(xyz.reshape((self.csize, 3)))

        return kdtree

    def __getitem__(self, index):
        """Return copied slice of Coordinates object at index."""

        new = self.copy()
        # slice data
        new._data = np.atleast_2d(new._data[index])

        return new

    def __array__(self, copy=True, dtype=None):
        """Instances of Coordinates behave like `numpy.ndarray`, array_like."""
        # copy to avoid changing the coordinate system of the original object
        return np.array(self.cartesian, copy=copy, dtype=dtype)

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
        if self.weights is None:
            _repr += "\nDoes not contain sampling weights"
        else:
            _repr += "\nContains sampling weights"

        # check for comment
        if self._comment != "":
            _repr += f"\nComment: {self._comment}"

        return _repr

    def __eq__(self, other):
        """Check for equality of two objects."""
        if self.cshape != other.cshape:
            return False
        eq_data = self._data == other._data
        eq_comment = self._comment == other._comment
        return eq_data.all() & eq_comment

    def __add__(self, other):
        """Add two numbers/Coordinates objects."""
        return _arithmetics(self, other, 'add')

    def __radd__(self, other):
        """Add two numbers/Coordinates objects."""
        return _arithmetics(other, self, 'add')

    def __sub__(self, other):
        """Subtract two numbers/Coordinates objects."""
        return _arithmetics(self, other, 'sub')

    def __rsub__(self, other):
        """Subtract two numbers/Coordinates objects."""
        return _arithmetics(other, self, 'sub')

    def __mul__(self, other):
        """Multiply Coordinates object with number."""
        return _arithmetics(self, other, 'mul')

    def __rmul__(self, other):
        """Multiply number with Coordinates object."""
        return _arithmetics(other, self, 'mul')

    def __div__(self, other):
        """Divide Coordinates object with number."""
        return _arithmetics(self, other, 'div')

    def __truediv__(self, other):
        """Divide Coordinates object with number."""
        return _arithmetics(self, other, 'div')

    def __rtruediv__(self, other):
        """Divide number with Coordinates object."""
        return _arithmetics(other, self, 'div')

    def __rdiv__(self, other):
        """Divide number with Coordinates object."""
        return _arithmetics(other, self, 'div')

    def _check_empty(self):
        """Check if object is empty."""
        if self.cshape == (0,):
            raise ValueError('Object is empty.')


def dot(a, b):
    r"""Dot product of two Coordinates objects.

    .. math::
        \vec{a} \cdot \vec{b}
        = a_x \cdot b_x + a_y \cdot b_y + a_z \cdot b_z

    Parameters
    ----------
    a : pf.Coordinates
        first argument, must be broadcastable with b
    b : pf.Coordinates
        second argument, much be broadcastable with a

    Returns
    -------
    result : np.ndarray
        array with the dot product of the two objects

    Examples
    --------
    >>> import pyfar as pf
    >>> a = pf.Coordinates(1, 0, 0)
    >>> b = pf.Coordinates(1, 0, 0)
    >>> pf.dot(a, b)
    array([1.])
    """

    if not isinstance(a, Coordinates) or not isinstance(b, Coordinates):
        raise TypeError(
            "Dot product is only possible with Coordinates objects.")

    return a.x * b.x + a.y * b.y + a.z * b.z


def cross(a, b):
    r"""Cross product of two Coordinates objects.

    .. math::
        \vec{a} \times \vec{b}
        = (a_y \cdot b_z - a_z \cdot b_y) \cdot \hat{x}
        + (a_z \cdot b_x - a_x \cdot b_z) \cdot \hat{y}
        + (a_x \cdot b_y - a_y \cdot b_x) \cdot \hat{z}

    Parameters
    ----------
    a : pf.Coordinates
        first argument, must be broadcastable with b
    b : pf.Coordinates
        second argument, much be broadcastable with a

    Returns
    -------
    result : pf.Coordinates
        new Coordinates object with the cross product of the two objects

    Examples
    --------
    >>> import pyfar as pf
    >>> a = pf.Coordinates(1, 0, 0)
    >>> b = pf.Coordinates(0, 1, 0)
    >>> result = pf.cross(a, b)
    >>> result.cartesian
    array([0., 0., 1.])
    """

    if not isinstance(a, Coordinates) or not isinstance(b, Coordinates):
        raise TypeError(
            "Dot product is only possible with Coordinates objects.")

    new = Coordinates()
    new.cartesian = np.zeros(np.broadcast_shapes(
        a.cartesian.shape, b.cartesian.shape))

    # apply cross product
    new.x = a.y * b.z - a.z * b.y
    new.y = a.z * b.x - a.x * b.z
    new.z = a.x * b.y - a.y * b.x

    return new

def _arithmetics(first, second, operation):
    """Add or Subtract two Coordinates objects, numbers or arrays.

    Parameters
    ----------
    first : Coordinates, number, array
        first operand
    second : Coordinates, number, array
        second operand
    operation : 'add', 'sub', 'mul', 'div'
        whether to add or subtract the two objects

    Returns
    -------
    new : Coordinates
        result of the operation

    """
    # convert data
    data = []
    num_objects = 0
    for obj in [first, second]:
        if isinstance(obj, Coordinates):
            data.append(obj.cartesian)
            num_objects += 1
        elif isinstance(obj, (int, float)):
            data.append(np.array(obj))
        else:
            if operation == 'add':
                op = 'Addition'
            elif operation == 'sub':
                op = 'Subtraction'
            elif operation == 'mul':
                op = 'Multiplication'
            elif operation == 'div':
                op = 'Division'
            raise TypeError(
                f"{op} is only possible with Coordinates or number.")

    if operation in ['mul', 'div'] and num_objects > 1:
        raise TypeError(
            "Multiplication and division are only possible with one "
            "Coordinates object.")

    # broadcast shapes
    shape = np.broadcast_shapes(data[0].shape, data[1].shape)
    new = pf.Coordinates()
    new.cartesian = np.zeros(shape)

    # perform operation
    if operation == 'add':
        new.cartesian = data[0] + data[1]
    elif operation == 'sub':
        new.cartesian = data[0] - data[1]
    elif operation == 'mul':
        new.cartesian = data[0] * data[1]
    elif operation == 'div':
        new.cartesian = data[0] / data[1]
    return new


def cart2sph(x, y, z):
    r"""
    Transforms from Cartesian to spherical coordinates.

    Spherical coordinates follow the common convention in Physics/Mathematics.
    The `colatitude` is measured downwards from the z-axis and is 0 at the
    North Pole and pi at the South Pole. The `azimuth` is 0 at positive
    x-direction and pi/2 at positive y-direction (counter clockwise rotation).

    Cartesian coordinates follow the right hand rule.

    .. math::

        azimuth &= \arctan(\frac{y}{x}),

        colatitude &= \arccos(\frac{z}{r}),

        radius &= \sqrt{x^2 + y^2 + z^2}

    .. math::

        0 < azimuth < 2 \pi,

        0 < colatitude < \pi

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
    r"""
    Transforms from spherical to Cartesian coordinates.

    Spherical coordinates follow the common convention in Physics/Mathematics.
    The `colatitude` is measured downwards from the z-axis and is 0 at the
    North Pole and pi at the South Pole. The `azimuth` is 0 at positive
    x-direction and pi/2 at positive y-direction (counter clockwise rotation).

    Cartesian coordinates follow the right hand rule.

    .. math::

        x &= radius \cdot \sin(colatitude) \cdot \cos(azimuth),

        y &= radius \cdot \sin(colatitude) \cdot \sin(azimuth),

        z &= radius \cdot \cos(colatitude)

    .. math::

        0 < azimuth < 2 \pi

        0 < colatitude < \pi


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
    r"""
    Transforms from Cartesian to cylindrical coordinates.

    Cylindrical coordinates follow the convention that the `azimuth` is 0 at
    positive x-direction and pi/2 at positive y-direction (counter clockwise
    rotation). The `height` is identical to the z-coordinate and the `radius`
    is measured orthogonal from the z-axis.

    Cartesian coordinates follow the right hand rule.

    .. math::

        azimuth &= \arctan(\frac{y}{x}),

        height &= z,

        radius &= \sqrt{x^2 + y^2},

    .. math::

        0 < azimuth < 2 \pi

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
    r"""
    Transforms from cylindrical to Cartesian coordinates.

    Cylindrical coordinates follow the convention that the `azimuth` is 0 at
    positive x-direction and pi/2 at positive y-direction (counter clockwise
    rotation). The `height` is identical to the z-coordinate and the `radius`
    is measured orthogonal from the z-axis.

    Cartesian coordinates follow the right hand rule.

    .. math::

        x &= radius \cdot \cos(azimuth),

        y &= radius \cdot \sin(azimuth),

        z &= height

    .. math::

        0 < azimuth < 2 \pi

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
    Convert a copy of coordinates in radians to degree.

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
    Convert a copy of coordinates in degree to radians.

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
    """Private function called by rad2deg and deg2rad."""

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
        variable_name = 'values'
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
