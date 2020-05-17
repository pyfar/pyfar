import numpy as np
from scipy.spatial import cKDTree


class Coordinates(object):
    """Container class for coordinates in a three-dimensional space, allowing
    for compact representation and convenient conversion into spherical as well
    as geospatial coordinate systems.
    The constructor as well as the internal representation are only
    available in Cartesian coordinates. To create a Coordinates object from
    a set of points in spherical coordinates, please use the
    Coordinates.from_spherical() method.

    Attributes
    ----------
    x : ndarray, double
        x-coordinate
    y : ndarray, double
        y-coordinate
    z : ndarray, double
        z-coordinate

    """

    # structure ----------------------
    #
    # * cordinate systems are defined in a nested dictionary. The dictionary
    #   is stored in a pseudo private mudole class def _coordinate_systems.
    #   The dictionary holds the following
    #   - domain: e.g., 'spherical'
    #     - convention: e.g., 'top pole 1' (unique within domain)
    #       - coordinates: e.g., ['azimuth', 'colatitude', 'radius']
    #       - units: e.g., [['radians', 'radians', 'meters']
    #                       ['degrees', 'degrees', 'meters']]
    #                units holds all possible units. Only the first entry is
    #                used for setting the units. x,y,z, and radius are always
    #                in meters
    #       - description:
    #
    # * @classmethod coordinate_systems(domain='all', convention='all') can be
    #    used to list all available coordinate systems or a specific subset
    #
    # * The Coordinates(object) has properties that specify the current
    #   coordinate system:
    #     @property domain,
    #     @property convention,
    #     @property coordinates,
    #     @property unit
    #
    # * The coordinate system is set upon construction
    #     __init__(self, coordinate1=None, coordinate2=None, coordinate3=None,
    #              domain='cartesian', convention='right handed', units='meters')
    #
    # * A dictionary search is used to check if domain, coordinates, and units
    #   are valid
    #
    # * The data are
    #     - stored in sefl._points
    #     - returned by @porpoerty points
    #     - and set by @points.setter points
    #
    # * A specific coordinate can for example be obtained by
    #    @property radius. This will throw an error if radius is not found in
    #                      @property coordinates
    #
    # * The coordinate system is changed by
    #    @classmethod convert(domain=None, convention=None, units=None)
    #      - this calls two converters
    #        1: _domain2cart(self, 'convention', 'units')
    #      - 2: _cart2domain(self, 'convention', 'units')
    #
    #
    # * The number of points are stored
    #     @property num_points
    #
    # * __repr__ will list the the current coordinate system and the number of
    #            points
    #
    # - explicit getter, e.g., c.get_sph('top_colat', 'deg')
    # - unique coordinate names, e.g., radius_cyl
    # - no explicit conversion
    # - coordinate_dictionary
    # - private property to store current domain, convention, units

    def __init__(self, x=None, y=None, z=None):
        """Init coordinates container

        Attributes
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """

        super(Coordinates, self).__init__()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if not np.shape(x) == np.shape(y) == np.shape(z):
            raise ValueError("Input arrays need to have same dimensions.")

        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def list_coordinate_systems(self, domain='all',convention='all'):

        # get implemented coordinate systems
        _coordinate_systems = self.get_coordinate_systems()

        # check what domains to list
        if domain=='all':
            domain=list(_coordinate_systems)[:]

        #
        for d in domain:
            if convention == 'all':
                print(d + ':')
            else:
                print(d + ', ' + convention)


    @property
    def x(self):
        """The x-axis coordinates for each point.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value, dtype=np.float64)

    @property
    def y(self):
        """The y-axis coordinate for each point."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.asarray(value, dtype=np.float64)

    @property
    def z(self):
        """The z-axis coordinate for each point."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = np.asarray(value, dtype=np.float64)

    @property
    def radius(self):
        """The radius for each point."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @radius.setter
    def radius(self, radius):
        x, y, z = _sph2cart(
            np.asarray(radius, dtype=np.float64),
            self.elevation,
            self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @property
    def azimuth(self):
        """The azimuth angle for each point."""
        return np.mod(np.arctan2(self.y, self.x), 2*np.pi)

    @azimuth.setter
    def azimuth(self, azimuth):
        x, y, z = _sph2cart(
            self.radius,
            self.elevation,
            np.asarray(azimuth, dtype=np.float64))
        self._x = x
        self._y = y
        self._z = z

    @property
    def elevation(self):
        """The elevation angle for each point"""
        rad = self.radius
        return np.arccos(self.z/rad)

    @elevation.setter
    def elevation(self, elevation):
        x, y, z = _sph2cart(
            self.radius,
            np.asarray(elevation, dtype=np.float64),
            self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_cartesian(cls, x, y, z):
        """Create a Coordinates class object from a set of points in the
        Cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """
        return Coordinates(x, y, z)

    @classmethod
    def from_spherical(cls, radius, elevation, azimuth):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        radius : ndarray, double
            The radius for each point
        elevation : ndarray, double
            The elevation angle in radians
        azimuth : ndarray, double
            The azimuth angle in radians
        """
        radius = np.asarray(radius, dtype=np.double)
        elevation = np.asarray(elevation, dtype=np.double)
        azimuth = np.asarray(azimuth, dtype=np.double)
        x, y, z = _sph2cart(radius, elevation, azimuth)
        return Coordinates(x, y, z)

    @classmethod
    def from_array(cls, values, coordinate_system='cartesian'):
        """Create a Coordinates class object from a set of points given as
        numpy array

        Parameters
        ----------
        values : double, ndarray
            Array with shape Nx3 where N is the number of points.
        coordinate_system : string
            Coordinate convention of the given values.
            Can be Cartesian or spherical coordinates.
        """
        coords = Coordinates()
        if coordinate_system == 'cartesian':
            coords.cartesian = values
        elif coordinate_system == 'spherical':
            coords.spherical = values
        else:
            return ValueError("This coordinate system is not supported.")

        return coords

    @property
    def latitude(self):
        """The latitude angle as used in geospatial coordinates."""
        return np.pi/2 - self.elevation

    @property
    def longitude(self):
        """The longitude angle as used in geospatial coordinates."""
        return np.arctan2(self.y, self.x)

    @property
    def cartesian(self):
        """Cartesian coordinates of all points."""
        return np.vstack((self.x, self.y, self.z))

    @cartesian.setter
    def cartesian(self, value):
        """Cartesian coordinates of all points."""
        self.x = value[0, :]
        self.y = value[1, :]
        self.z = value[2, :]

    @property
    def spherical(self):
        """Spherical coordinates of all points."""
        return np.vstack((self.radius, self.elevation, self.azimuth))

    @spherical.setter
    def spherical(self, value):
        """Cartesian coordinates of all points."""
        x, y, z = _sph2cart(value[0, :], value[1, :], value[2, :])
        self.cartesian = np.vstack((x, y, z))

    @property
    def n_points(self):
        """Return number of points stored in the object"""
        return self.x.size

    def find_nearest_point(self, point):
        """Find the closest Coordinate point to a given Point.
        The search for the nearest point is performed using the scipy
        cKDTree implementation.

        Parameters
        ----------
        point : Coordinates
            Point to find nearest neighboring Coordinate

        Returns
        -------
        distance : ndarray, double
            Distance between the point and it's closest neighbor
        index : int
            Index of the closest point.

        """
        kdtree = cKDTree(self.cartesian.T)
        distance, index = kdtree.query(point.cartesian.T)

        return distance, index

    def __repr__(self):
        """repr for Coordinate class

        """
        if self.n_points == 1:
            repr_string = "Coordinates of 1 point"
        else:
            repr_string = "Coordinates of {} points".format(self.n_points)
        return repr_string

    def __getitem__(self, index):
        """Return Coordinates at index
        """
        return Coordinates(self._x[index], self._y[index], self._z[index])

    def __setitem__(self, index, item):
        """Set Coordinates at index
        """
        self.x[index] = item.x
        self.y[index] = item.y
        self.z[index] = item.z

    def __len__(self):
        """Length of the object which is the number of points stored.
        """
        return self.n_points

def _coordinate_systems():
    """
    Get module internal information about available coordinate systems.

    Returns
    -------
    _systems : NESTED DICT
        List all available coordinate systems.
        Key 0  - domain, e.g., 'cart'
        Key 1  - convention, e.g., 'right'
        Key 2a - 'short_description'
        Key 2b - 'coordinates'
        Key 2c - 'units'
        Key 2d - 'description'

    _coordinates: NESTED DICT
        Resolve coordinate systems in which a coordinate ocurrs and the units
        that a coordinate can have.
        Key 0  - coordinate
        Key 1a - domain
        Key 1b - convention
        Key 1c - units

    """

    # define coordinate systems
    _systems = {
        "cart":
            {
            "right":{
                "description_short":
                    "Right handed cartesian coordinate system.",
                "coordinates":
                    ["x", "y", "z"],
                "units":
                    ["meters", "meters", "meters"],
                "description":
                    "Right handed cartesian coordinate system with x,y, and."\
                    "z in meters."}
            },
        "sph":
            {
            "top_colat":{
                "description_short":
                    "Spherical coordinate system with North and South Pole.",
                "coordinates":
                    ["azimuth", "colatitude", "radius"],
                "units":
                    [["radians", "radians", "meter"],
                     ["degrees", "degrees", "meters"]],
                "description":
                    "The azimuth denotes the counter clockwise angle in the "\
                    "x/y-plane with 0 pointing in positive x-direction and "\
                    " pi/2 in positive y-direction. The colatitude denotes "\
                    "the angle downwards from the z-axis with 0 pointing in "\
                    "positve z-direction and pi in negative z-direction. The "\
                    "azimuth and colatitude can be in radians or degrees, "\
                    "the radius is always in meters."},
            "top_elev":{
                "description_short":
                    "Spherical coordinate system with North and South Pole.",
                "coordinates":
                    ["azimuth", "elevation", "radius"],
                "units":
                    [["radians", "radians", "meter"],
                     ["degrees", "degrees", "meters"]],
                "description":
                    "The azimuth and radius are identical to 'sph', "\
                    "'top_colat'. The elevation denotes the angle upwards "\
                    "and downwards from the x/y-plane with pi/2 pointing at "\
                    "positive z-direction and -pi/2 pointing in negative "\
                    "z-direction. The azimuth and elevation can be in "\
                    "radians or degrees, the radius is always in meters."},
            "side":{
                "description_short":
                    "Spherical coordinate system with poles on the y-axis.",
                "coordinates":
                    ["lateral", "polar", "radius"],
                "units":
                    [["radians", "radians", "meter"],
                     ["degrees", "degrees", "meters"]],
                "description":
                    "The lateral angle denotes the angle in the x/y-plane "\
                    "with pi/2 pointing in positive y-direction and -pi/2 in "\
                    "negative y-direction. The polar angle denotes the angle "\
                    "in the x/z-plane with -pi/2 pointing in negative "\
                    "z-direction, 0 in positive x-direction, pi/2 in "\
                    "positive z-direction, pi in negative x-direction. The "\
                    "polar and lateral angle can be in radians and degree, "\
                    "the radius is always in meters."},
            "front":{
                "description_short":
                    "Spherical coordinate system with poles on the x-axis.",
                "coordinates":
                    ["phi", "theta", "radius"],
                "units":
                    [["radians", "radians", "meter"],
                     ["degrees", "degrees", "meters"]],
                "description":
                    "Phi denotes the angle measured from the x-axis with 0 "\
                    "pointing in positve x-direction and pi in negative x-"\
                    "direction. Theta denotes the angle in the y/z-plane "\
                    "with 0 pointing in positive z-direction, pi/2 in "\
                    "positive y-direction, pi in negative z-direction, and "\
                    "3*pi/2 in negative y-direction. Phi and theta can be "\
                    "in radians and degrees, the radius is always in meters."}
            },
        "cyl":
            {
            "top":{
                "description_short":
                    "Cylindrical coordinate system along the z-axis.",
                "coordinates":
                    ["azimuth", "z", "radius_z"],
                "units":
                    [["radians", "meters", "meter"],
                     ["degrees", "meters", "meters"]],
                "description":
                    "The azimuth denotes the counter clockwise angle in the "\
                    "x/y-plane with 0 pointing in positive x-direction and "\
                    " pi/2 in positive y-direction. The heigt is given by "\
                    "z, and radius_z denotes the radius measured orthogonal "\
                    "to the z-axis."}
            }
        }

    # resolve membership of coordinates
    _coordinates = {}

    # loop across domains and conventions
    for domain in _systems:
        for convention in _systems[domain]:
            # loop across coordinates
            for cc, coord in enumerate(_systems[domain][convention]['coordinates']):
                # units of the current coordinate
                all_units = _systems[domain][convention]['units']
                if np.asarray(all_units).ndim == 1:
                    cur_units = [all_units[cc]]
                else:
                    cur_units = [u[cc] for u in _systems[domain][convention]['units']]
                # add coordinate to _coordinates
                if not coord in _coordinates:
                    _coordinates[coord]= {}
                    _coordinates[coord]['domain']     = [domain]
                    _coordinates[coord]['convention'] = [convention]
                    _coordinates[coord]['units']      = [cur_units]
                else:
                    _coordinates[coord]['domain'].append(domain)
                    _coordinates[coord]['convention'].append(convention)
                    _coordinates[coord]['units'].append(cur_units)

    return _systems, _coordinates


def _sph2cart(r, theta, phi):
    """Transforms from spherical to Cartesian coordinates.
    Spherical coordinates follow the common convention in Physics/Mathematics
    Theta denotes the elevation angle with theta = 0 at the north pole and
    theta = pi at the south pole.
    Phi is the azimuth angle counting from phi = 0 at the x-axis in positive
    direction (counter clockwise rotation).

    .. math::

        x = r \\sin(\\theta) \\cos(\\phi),

        y = r \\sin(\\theta) \\sin(\\phi),

        z = r \\cos(\\theta)

    Parameters
    ----------
    r : ndarray, number
    theta : ndarray, number
    phi : ndarray, number

    Returns
    -------
    x : ndarray, number
    y : ndarray, number
    z : ndarray, number

    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z


def _cart2sph(x, y, z):
    """
    Transforms from Cartesian to spherical coordinates.
    Spherical coordinates follow the common convention in Physics/Mathematics
    Theta denotes the elevation angle with theta = 0 at the north pole and
    theta = pi at the south pole.
    Phi is the azimuth angle counting from phi = 0 at the x-axis in positive
    direction (counter clockwise rotation).

    .. math::

        r = \\sqrt{x^2 + y^2 + z^2},

        \\theta = \\arccos(\\frac{z}{r}),

        \\phi = \\arctan(\\frac{y}{x})

        0 < \\theta < \\pi,

        0 < \\phi < 2 \\pi


    Notes
    -----
    To ensure proper handling of the radiant for the azimuth angle, the arctan2
    implementatition from numpy is used here.

    Parameters
    ----------
    x : ndarray, number
    y : ndarray, number
    z : ndarray, number

    Returns
    -------
    r : ndarray, number
    theta : ndarray, number
    phi : ndarray, number

    """
    rad = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/rad)
    phi = np.mod(np.arctan2(y, x), 2*np.pi)
    return rad, theta, phi


def _cart2latlon(x, y, z):
    """Transforms from Cartesian coordinates to Geocentric coordinates

    .. math::

        h = \\sqrt{x^2 + y^2 + z^2},

        \\theta = \\pi/2 - \\arccos(\\frac{z}{r}),

        \\phi = \\arctan(\\frac{y}{x})

        -\\pi/2 < \\theta < \\pi/2,

        -\\pi < \\phi < \\pi

    where :math:`h` is the heigth, :math:`\\theta` is the latitude angle
    and :math:`\\phi` is the longitude angle

    Parameters
    ----------
    x : ndarray, number
        x-axis coordinates
    y : ndarray, number
        y-axis coordinates
    z : ndarray, number
        z-axis coordinates

    Returns
    -------
    height : ndarray, number
        The radius is rendered as height information
    latitude : ndarray, number
        Geocentric latitude angle
    longitude : ndarray, number
        Geocentric longitude angle

    """
    height = np.sqrt(x**2 + y**2 + z**2)
    latitude = np.pi/2 - np.arccos(z/height)
    longitude = np.arctan2(y, x)
    return height, latitude, longitude


def _latlon2cart(height, latitude, longitude):
    """Transforms from Geocentric coordinates to Cartesian coordinates

    .. math::

        x = h \\cos(\\theta) \\cos(\\phi),

        y = h \\cos(\\theta) \\sin(\\phi),

        z = h \\sin(\\theta)

        -\\pi/2 < \\theta < \\pi/2,

        -\\pi < \\phi < \\pi

    where :math:`h` is the heigth, :math:`\\theta` is the latitude angle
    and :math:`\\phi` is the longitude angle

    Parameters
    ----------
    height : ndarray, number
        The radius is rendered as height information
    latitude : ndarray, number
        Geocentric latitude angle
    longitude : ndarray, number
        Geocentric longitude angle

    Returns
    -------
    x : ndarray, number
        x-axis coordinates
    y : ndarray, number
        y-axis coordinates
    z : ndarray, number
        z-axis coordinates

    """

    x = height * np.cos(latitude) * np.cos(longitude)
    y = height * np.cos(latitude) * np.sin(longitude)
    z = height * np.sin(latitude)

    return x, y, z



class Coordinates_copy(object):
    """Container class for coordinates in a three-dimensional space, allowing
    for compact representation and convenient conversion into spherical as well
    as geospatial coordinate systems.
    The constructor as well as the internal representation are only
    available in Cartesian coordinates. To create a Coordinates object from
    a set of points in spherical coordinates, please use the
    Coordinates.from_spherical() method.

    Attributes
    ----------
    x : ndarray, double
        x-coordinate
    y : ndarray, double
        y-coordinate
    z : ndarray, double
        z-coordinate

    """
    def __init__(self, x=None, y=None, z=None):
        """Init coordinates container

        Attributes
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """

        super(Coordinates, self).__init__()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if not np.shape(x) == np.shape(y) == np.shape(z):
            raise ValueError("Input arrays need to have same dimensions.")

        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        """The x-axis coordinates for each point.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value, dtype=np.float64)

    @property
    def y(self):
        """The y-axis coordinate for each point."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.asarray(value, dtype=np.float64)

    @property
    def z(self):
        """The z-axis coordinate for each point."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = np.asarray(value, dtype=np.float64)

    @property
    def radius(self):
        """The radius for each point."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @radius.setter
    def radius(self, radius):
        x, y, z = _sph2cart(
            np.asarray(radius, dtype=np.float64),
            self.elevation,
            self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @property
    def azimuth(self):
        """The azimuth angle for each point."""
        return np.mod(np.arctan2(self.y, self.x), 2*np.pi)

    @azimuth.setter
    def azimuth(self, azimuth):
        x, y, z = _sph2cart(
            self.radius,
            self.elevation,
            np.asarray(azimuth, dtype=np.float64))
        self._x = x
        self._y = y
        self._z = z

    @property
    def elevation(self):
        """The elevation angle for each point"""
        rad = self.radius
        return np.arccos(self.z/rad)

    @elevation.setter
    def elevation(self, elevation):
        x, y, z = _sph2cart(
            self.radius,
            np.asarray(elevation, dtype=np.float64),
            self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_cartesian(cls, x, y, z):
        """Create a Coordinates class object from a set of points in the
        Cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """
        return Coordinates(x, y, z)

    @classmethod
    def from_spherical(cls, radius, elevation, azimuth):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        radius : ndarray, double
            The radius for each point
        elevation : ndarray, double
            The elevation angle in radians
        azimuth : ndarray, double
            The azimuth angle in radians
        """
        radius = np.asarray(radius, dtype=np.double)
        elevation = np.asarray(elevation, dtype=np.double)
        azimuth = np.asarray(azimuth, dtype=np.double)
        x, y, z = _sph2cart(radius, elevation, azimuth)
        return Coordinates(x, y, z)

    @classmethod
    def from_array(cls, values, coordinate_system='cartesian'):
        """Create a Coordinates class object from a set of points given as
        numpy array

        Parameters
        ----------
        values : double, ndarray
            Array with shape Nx3 where N is the number of points.
        coordinate_system : string
            Coordinate convention of the given values.
            Can be Cartesian or spherical coordinates.
        """
        coords = Coordinates()
        if coordinate_system == 'cartesian':
            coords.cartesian = values
        elif coordinate_system == 'spherical':
            coords.spherical = values
        else:
            return ValueError("This coordinate system is not supported.")

        return coords

    @property
    def latitude(self):
        """The latitude angle as used in geospatial coordinates."""
        return np.pi/2 - self.elevation

    @property
    def longitude(self):
        """The longitude angle as used in geospatial coordinates."""
        return np.arctan2(self.y, self.x)

    @property
    def cartesian(self):
        """Cartesian coordinates of all points."""
        return np.vstack((self.x, self.y, self.z))

    @cartesian.setter
    def cartesian(self, value):
        """Cartesian coordinates of all points."""
        self.x = value[0, :]
        self.y = value[1, :]
        self.z = value[2, :]

    @property
    def spherical(self):
        """Spherical coordinates of all points."""
        return np.vstack((self.radius, self.elevation, self.azimuth))

    @spherical.setter
    def spherical(self, value):
        """Cartesian coordinates of all points."""
        x, y, z = _sph2cart(value[0, :], value[1, :], value[2, :])
        self.cartesian = np.vstack((x, y, z))

    @property
    def n_points(self):
        """Return number of points stored in the object"""
        return self.x.size

    def find_nearest_point(self, point):
        """Find the closest Coordinate point to a given Point.
        The search for the nearest point is performed using the scipy
        cKDTree implementation.

        Parameters
        ----------
        point : Coordinates
            Point to find nearest neighboring Coordinate

        Returns
        -------
        distance : ndarray, double
            Distance between the point and it's closest neighbor
        index : int
            Index of the closest point.

        """
        kdtree = cKDTree(self.cartesian.T)
        distance, index = kdtree.query(point.cartesian.T)

        return distance, index

    def __repr__(self):
        """repr for Coordinate class

        """
        if self.n_points == 1:
            repr_string = "Coordinates of 1 point"
        else:
            repr_string = "Coordinates of {} points".format(self.n_points)
        return repr_string

    def __getitem__(self, index):
        """Return Coordinates at index
        """
        return Coordinates(self._x[index], self._y[index], self._z[index])

    def __setitem__(self, index, item):
        """Set Coordinates at index
        """
        self.x[index] = item.x
        self.y[index] = item.y
        self.z[index] = item.z

    def __len__(self):
        """Length of the object which is the number of points stored.
        """
        return self.n_points