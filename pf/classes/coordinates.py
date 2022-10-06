"""
The following documents the pyfar coordinates class and functions for
coordinate conversion. More background information is given in
:py:mod:`coordinates concepts <pyfar._concepts.coordinates>`.
Available sampling schemes are listed at :py:mod:`~pyfar.samplings`.
"""
import numpy as np
# from scipy.spatial import cKDTree
# from scipy.spatial.transform import Rotation as sp_rot
import re
from copy import deepcopy
from scipy.spatial.transform import Rotation as sp_rot

# import pyfar as pf


class Coordinates():
    """
    Container class for storing, converting, rotating, querying, and plotting
    3D coordinate systems.
    """
    _x: np.array = np.empty
    _y: np.array = np.empty
    _z: np.array = np.empty
    _weights: np.array = None
    _comment: str = ""

    def __init__(
            self, points_1: np.array = np.asarray([]),
            points_2: np.array = np.asarray([]),
            points_3: np.array = np.asarray([]),
            domain: str = 'cart', convention: str ='right', unit: str = 'met',
            weights: np.array = None, comment: str = "") -> None:
        """
        Create :py:func:`Coordinates` object with or without coordinate points.
        The points that enter the Coordinates object are defined by the
        `domain`, `convention`, and `unit` as illustrated in the
        :py:mod:`coordinates concepts <pyfar._concepts.coordinates>`:
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
        For more information run
        >>> coords = Coordinates()
        >>> coords.systems()
        Parameters
        ----------
        points_1 : array like, number
            points for the first coordinate
        points_2 : array like, number
            points for the second coordinate
        points_3 : array like, number
            points for the third coordinate
        domain : string
            domain of the coordinate system
            ``'cart'``
                Cartesian
            ``'sph'``
                Spherical
            ``'cyl'``
                Cylindrical
            The default is ``'cart'``.
        convention: string
             coordinate convention (see above)
             The default is ``'right'`` if domain is ``'cart'``,
             ``'top_colat'`` if domain is ``'sph'``, and ``'top'`` if domain is
             ``'cyl'``.
        unit: string
             unit of the coordinate system. By default the first available unit
             is used, which is meters (``'met'``) for ``domain = 'cart'`` and
             radians (``'rad'``) in all other cases (See above).
        weights: array like, number, optional
            sampling weights for the coordinate points. Must have same `size`
            as the points points, i.e., if `points` have five entries, the
            `weights` must also have five entries. The default is ``None``.
        comment : str, optional
            comment about the stored coordinate points. The default is
            ``None``.
        """

        if domain == 'cart':
            self.set_cart(points_1, points_2, points_3)
        elif domain == 'sph':
            self.set_sph(points_1, points_2, points_3, convention, unit)
        elif domain == 'cyl':
            self.set_cyl(points_1, points_2, points_3, convention, unit)
        else:
            raise ValueError(
                f"Domain for {domain} is not implemented.")

        self._set_weights(weights)
        self._comment = comment

    def __eq__(self, other):
        """Check for equality of two objects."""
        # return not deepdiff.DeepDiff(self, other)
        if self.cshape != other.cshape:
            return False
        eq_x = self._x == other._x
        eq_y = self._y == other._y
        eq_z = self._z == other._z
        eq_weights = self._weights == other._weights
        eq_comment = self._comment == other._comment
        if self._x.shape == ():
            return eq_x & eq_y & eq_z & eq_weights & eq_comment
        return all(eq_x & eq_y & eq_z) & eq_weights & eq_comment

    def copy(self):
        """Return a deep copy of the Coordinates object."""
        return deepcopy(self)

    @property
    def csize(self):
        """
        Return channel size.

        The channel size gives the number of points stored in the coordinates
        object.
        """
        return self._x.size

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
    def weights(self):
        """Get sampling weights."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set sampling weights."""
        self._set_weights(value)

    @property
    def comment(self):
        """Get comment."""
        return self._comment

    @comment.setter
    def comment(self, value):
        """Set comment."""
        self._comment = value

    @property
    def xyz(self):
        """The x-axis coordinates for each point in a right handed cartesian
        coordinate system."""
        return np.moveaxis(np.array([self.x, self.y, self.z]), 0, -1)

    @xyz.setter
    def xyz(self, value):
        self._set_points(value[..., 0], value[..., 1], value[..., 2])

    @property
    def x(self):
        """The x-axis coordinates for each point in a right handed cartesian
        coordinate system."""
        return self._x

    @x.setter
    def x(self, value):
        self._set_points(value, self.y, self.z)

    @property
    def y(self):
        """The y-axis coordinates for each point in a right handed cartesian
        coordinate system."""
        return self._y

    @y.setter
    def y(self, value):
        self._set_points(self.x, value, self.z)

    @property
    def z(self):
        """The z-axis coordinates for each point in a right handed cartesian
        coordinate system."""
        return self._z

    @z.setter
    def z(self, value):
        self._set_points(self.x, self.y, value)

    @property
    def radius_z(self):
        """The z-axis coordinates for each point in a right handed cartesian
        coordinate system."""
        azimuth, z, radius_z = self.get_cyl()
        return radius_z

    @radius_z.setter
    def radius_z(self, radius_z):
        azimuth, z, _ = self.get_cyl()
        self.set_cyl(azimuth, z, radius_z)

    @property
    def radius(self):
        """The radius for each point."""
        azimuth, elevation, radius = self.get_sph(convention='top_elev')
        return radius

    @radius.setter
    def radius(self, radius):
        azimuth, elevation, _ = self.get_sph(convention='top_elev')
        self.set_sph(azimuth, elevation, radius, convention='top_elev')

    @property
    def azimuth(self):
        """The azimuth angle for each point."""
        azimuth, _, _ = self.get_sph(convention='top_elev')
        return azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        _, elevation, radius = self.get_sph(convention='top_elev')
        self.set_sph(azimuth, elevation, radius, convention='top_elev')

    @property
    def elevation(self):
        """The elevation angle for each point"""
        _, elevation, _ = self.get_sph(convention='top_elev')
        return elevation

    @elevation.setter
    def elevation(self, elevation):
        azimuth, _, radius = self.get_sph(convention='top_elev')
        self.set_sph(azimuth, elevation, radius, convention='top_elev')

    @property
    def colatitude(self):
        """The colatitude angle for each point"""
        azimuth, colatitude, radius = self.get_sph(convention='top_colat')
        return colatitude

    @colatitude.setter
    def colatitude(self, colatitude):
        azimuth, _, radius = self.get_sph(convention='top_colat')
        self.set_sph(azimuth, colatitude, radius, convention='top_colat')

    @property
    def phi(self):
        """The phi angle for each point."""
        phi, theta, radius = self.get_sph(convention='front')
        return phi

    @phi.setter
    def phi(self, phi):
        _, theta, radius = self.get_sph(convention='front')
        self.set_sph(phi, theta, radius, convention='front')

    @property
    def theta(self):
        """The theta angle for each point"""
        phi, theta, radius = self.get_sph(convention='front')
        return theta

    @theta.setter
    def theta(self, theta):
        phi, _, radius = self.get_sph(convention='front')
        self.set_sph(phi, theta, radius, convention='front')

    @property
    def lateral(self):
        """The lateral angle for each point."""
        lateral, polar, radius = self.get_sph(convention='side')
        return lateral

    @lateral.setter
    def lateral(self, lateral):
        _, polar, radius = self.get_sph(convention='side')
        self.set_sph(lateral, polar, radius, convention='side')

    @property
    def polar(self):
        """The polar angle for each point"""
        lateral, polar, radius = self.get_sph(convention='side')
        return polar

    @polar.setter
    def polar(self, polar):
        lateral, _, radius = self.get_sph(convention='side')
        self.set_sph(lateral, polar, radius, convention='side')

    def _set_points(self, x, y, z):
        """
        Check points and convert to matrix.

        Parameters
        ----------
        convert : boolean, optional
            Set self._points if convert = True. Return points as
            matrix otherwise. The fefault is False.
        system: dict, optional
            The coordinate system against which the range of the points are
            checked as returned from self._make_system. If system = None
            self._system is used.

        Set self._points, which is an atleast_2d numpy array of shape
        [L,M,...,N, 3].
        """

        # cast to numpy array
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        # shapes of non scalar entries
        shapes = [p.shape for p in [x, y, z] if p.shape != ()]

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
        assert weights.size == self.csize,\
            "weights must have same size as self.csize"
        weights = weights.reshape(self.cshape)

        # set class variable
        self._weights = weights

    def _encode(self):
        """Return dictionary for the encoding."""
        return self.copy().__dict__

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective ``_encode`` counterpart."""
        obj = cls()
        obj.__dict__.update(obj_dict)
        return obj

    def set_cart(self, x, y, z, convention='right', unit='met'):
        """
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
        x, y, z: array like, number
            points for the first, second, and third coordinate
        convention : string, optional
            convention in which the coordinate points are stored. The default
            is ``'right'``.
        unit : string, optional
            unit in which the coordinate points are stored. The default is
            ``'met'`` for meters.
        """

        if convention != 'right':
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                (f"Conversion for {convention} is not implemented."))

        # save coordinates to self
        self._set_points(x, y, z)

    def get_cart(self, convention='right', unit='met', convert=False):
        """
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
            convention in which the coordinate points are stored. The default
            is ``'right'``.
        unit : string, optional
            unit in which the coordinate points are stored. The default is
           ``'met'``.
        convert : boolean, optional
            if True, the internal representation of the samplings points will
            be converted to the queried coordinate system. The default is
            ``False``, i.e., the internal presentation remains as it is.

        Returns
        -------
        points : numpy array
            coordinate points. ``points[...,0]`` holds the points for the first
            coordinate, ``points[...,1]`` the points for the second, and
            ``points[...,2]`` the points for the third coordinate.
        """

        # check if object is empty
        if self.cshape == (0,):
            raise ValueError('Object is empty.')

        return self._x, self._y, self._z

    def set_sph(self, angles_1, angles_2, radius,
                convention='top_colat', unit='rad'):
        """
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
            points for the first, second, and third coordinate
        convention : string, optional
            convention in which the coordinate points are stored. The default
            is ``'top_colat'``.
        unit : string, optional
            unit in which the coordinate points are stored. The default is
            ``'rad'``.
        """

        # convert to radians
        if unit == 'deg':
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

        # save coordinates to self
        self._set_points(x, y, z)

    def get_sph(self, convention='top_colat', unit='rad', convert=False):
        """
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
            convention in which the coordinate points are stored. The default
            is ``'top_colat'``.
        unit : string, optional
            unit in which the coordinate points are stored. The default is
            ``'rad'``.
        convert : boolean, optional
            if True, the internal representation of the samplings points will
            be converted to the queried coordinate system. The default is
            ``False``, i.e., the internal presentation remains as it is.

        Returns
        -------
        points : numpy array
            coordinate points. ``points[...,0]`` holds the points for the first
            coordinate, ``points[...,1]`` the points for the second, and
            ``points[...,2]`` the points for the third coordinate.
        """

        # check if object is empty
        if self.cshape == (0,):
            raise ValueError('Object is empty.')

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
        # (idea for simple converions from Robert Baumgartner and SOFA_API)
        elif convention == 'side':
            angles_2, angles_1, radius = cart2sph(x, z, -y)
            # range angles
            angles_1 = angles_1 - np.pi / 2
            angles_2 = np.mod(angles_2 + np.pi / 2, 2 * np.pi) - np.pi / 2
        # ... front polar system
        elif convention == 'front':
            angles_1, angles_2, radius = cart2sph(y, z, x)

        else:
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                f"Conversion for {convention} is not implemented.")

        # convert to degrees
        if unit == 'deg':
            angles_1 = angles_1 / np.pi * 180
            angles_2 = angles_2 / np.pi * 180
        elif not unit == 'rad':
            raise ValueError(
                f"unit for {unit} is not implemented.")

        # return points
        return angles_1, angles_2, radius

    def set_cyl(self, azimuth, z, radius_z, convention='top', unit='rad'):
        """
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
            points for the first, second, and third coordinate
        convention : string, optional
            convention in which the coordinate points are stored. The default
            is ``'top'``.
        unit : string, optional
            unit in which the coordinate points are stored. The default is
            ``'rad'``.
        """

        # convert to radians
        if unit == 'deg':
            azimuth = azimuth / 180 * np.pi
        elif not unit == 'rad':
            raise ValueError(
                f"unit for {unit} is not implemented.")

        # ... from cylindrical coordinate systems
        if convention == 'top':
            x, y, z = cyl2cart(azimuth, z, radius_z)
        else:
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                (f"Conversion for {convention} is not implemented."))

        # save coordinates to self
        self._set_points(x, y, z)

    def get_cyl(self, convention='top', unit='rad', convert=False):
        """
        Get coordinate points in cylindircal coordinate system.

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
            convention in which the coordinate points are stored. The default
            is ``'right'``.
        unit : string, optional
            unit in which the coordinate points are stored. The default is
            ``'met'``.
        convert : boolean, optional
            if True, the internal representation of the samplings points will
            be converted to the queried coordinate system. The default is
            False, i.e., the internal presentation remains as it is.

        Returns
        -------
        points : numpy array
            coordinate points. ``points[...,0]`` holds the points for the first
            coordinate, ``points[...,1]`` the points for the second, and
            ``points[...,2]`` the points for the third coordinate.
        """

        # check if object is empty
        if self.cshape == (0,):
            raise ValueError('Object is empty.')

        # convert to cylindrical ...
        # ... top systems
        if convention == 'top':
            azimuth, z, radius_z = cart2cyl(self.x, self.y, self.z)
        else:
            # Can not be tested. Will only be raised if a coordinate system
            # is not fully implemented.
            raise ValueError(
                f"Conversion for {convention} is not implemented.")

        # convert to degrees
        if unit == 'deg':
            azimuth = azimuth / np.pi * 180
        elif not unit == 'rad':
            raise ValueError(
                f"unit for {unit} is not implemented.")

        # return points and convert internal state if desired
        return azimuth, z, radius_z

    def rotate(self, rotation: str, value=None, degrees=True, inverse=False):
        """
        Rotate points stored in the object around the origin of coordinates.

        This is a wrapper for ``scipy.spatial.transform.Rotation`` (see this
        class for more detailed information).

        Parameters
        ----------
        rotation : str
            ``'quat'``
                rotation given by quaternions.
            ``'matrix'``
                rotation given by matrixes.
            ``'rotvec'``
                rotation using rotation vectors.
            ``'xyz'``
                rotation using euler angles. Up to three letters. E.g., ``'x'``
                will rotate about the x-axis only, while ``'xz'`` will rotate
                about the x-axis and then about the z-axis. Use lower letters
                for extrinsic rotations (rotations about the axes of the
                original coordinate system xyz, which remains motionless) and
                upper letters for intrinsic rotations (rotations about the axes
                of the rotating coordinate system XYZ, solidary with the moving
                body, which changes its orientation after each elemental
                rotation).
        value : number, array like
            amount of rotation in the format specified by `rotation` (see
            above).
        degrees : bool, optional
            pass angles in degrees if using ``'rotvec'`` or euler angles
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
        >>> coordinates = pf.samplings.sph_gaussian(sh_order=3)

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
        xyz = np.moveaxis(np.array([self.x.flatten(), self.y.flatten(), self.z.flatten()]), 0, -1)
        points = rot.apply(xyz, inverse)

        # set points
        self._set_points(
            points[:, 0].reshape(shape),
            points[:, 1].reshape(shape),
            points[:, 2].reshape(shape))


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
        x values
    y : numpy array, number
        y values
    z : numpy array, number
        z values

    Returns
    -------
    azimuth : numpy array, number
        azimuth values
    height : numpy array, number
        height values
    radius : numpy array, number
        radii

    Notes
    -----
    To ensure proper handling of the azimuth angle, the ``arctan2``
    implementation from numpy is used.
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
        azimuth values
    height : numpy array, number
        height values
    radius : numpy array, number
        radii

    Returns
    -------
    x : numpy array, number
        x values
    y : numpy array, number
        y values
    z : numpy array, number
        z values

    Notes
    -----
    To ensure proper handling of the azimuth angle, the ``arctan2``
    implementation from numpy is used.
    """
    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)
    if isinstance(height, np.ndarray):
        z = height.copy()
    else:
        z = height

    return x, y, z


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
        x values
    y : numpy array, number
        y values
    z : numpy array, number
        z values

    Returns
    -------
    azimuth : numpy array, number
        azimuth values
    colatitude : numpy array, number
        colatitude values
    radius : numpy array, number
        radii

    Notes
    -----
    To ensure proper handling of the azimuth angle, the ``arctan2``
    implementation from numpy is used.
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    z_div_r = np.where(radius != 0, z / radius, 0)
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
        azimuth values
    colatitude : numpy array, number
        colatitude values
    radius : numpy array, number
        radii

    Returns
    -------
    x : numpy array, number
        x values
    y : numpy array, number
        y values
    z : numpy array, number
        z vales
    """
    r_sin_cola = radius * np.sin(colatitude)
    x = r_sin_cola * np.cos(azimuth)
    y = r_sin_cola * np.sin(azimuth)
    z = radius * np.cos(colatitude)

    return x, y, z
