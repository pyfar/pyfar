import numpy as np
from scipy.spatial import cKDTree


class Coordinates(object):
    """
    Container class for coordinates in a three-dimensional space, allowing
    for compact representation and convenient conversion from and to cartesian,
    sphercial, and cylindrical coordinate systems.

    To obtain a list of all available coordinate systems, use

    >>> coords = Coordinates()          # get an instance of the class
    >>> coords.list_systems()           # list all systems

    A coordinate system is defined by it's 'domain', 'convention', and 'unit'
    as given in the list obtained above.

    To enter coordinates into the class, for example use

    >>> coords = Coordinates([0, 1], [1, 0], [1, 1])

    wich will use the default cartesian right handed coordinate system in
    meters.
    """

    # structure ----------------------
    #
    # * cordinate systems are defined in a nested dictionary. The dictionary
    #   is stored in a pseudo private mudole class def _systems.
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
    # * list_systems(domain='all', convention='all') can be
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
    #     - stored in sefl._coords
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

    def __init__(self, points_1=None, points_2=None, points_3=None,
                  domain='cart', convention='right', unit=None, comment=None):
        """
        Init coordinates container.

        Attributes
        ----------
        points_1 : scalar or array like
            points for the first coordinate
        points_2 : scalar or array like
            points for the second coordinate
        points_3 : scalar or array like
            points for the third coordinate
        domain : string
            domain of the coordinate system (see self.list_systems)
        convention: string
             coordinate convention (see self.list_systems)
        unit: string
             unit of the coordinate system. By default the first available unit
             is used (see self.list_systems)
        comment : str
            Any comment about the stored coordinate points.
        """

        # init emtpy object
        super(Coordinates, self).__init__()

        # set the coordinate system
        self._system = self._make_system(domain, convention, unit)

        # save coordinates to self
        self._set_points(points_1, points_2, points_3)

        # save comment
        self._comment = comment


    def get_cart(self, convention='right', unit='met'):
        """
        Get coordinate points in cartesian coordinate system.

        Parameters
        ----------
        convention : string, optional
            convention in which the coordinate points are returned. The default
            is 'right'.
        unit : string, optional
            unit in which the coordinate points are returned. The default is
            'met'.

        Returns
        -------
        points np.array
            array that holds the coordinate points. points[...,0] hold the
            points for the first coordinate, points[...,1] the points for the
            second, and points[...,2] the points for the third coordinate.

        Note
        ----
        The current and all availanle coordinate systems can be seen with

        >>> c = Coordinates()
        >>> c.system
        >>> c.list_systems()

        """

        # check for points
        if self.num_points == 0:
            raise Exception('Object is empty or contains invalid points')

        # make the new system
        new_system = self._make_system('cart', convention, unit)

        # return if system has not changed
        if self._system == new_system:
            return self._points

        # convert to radians
        pts = self._points
        for nn, unit in enumerate(self._system['units']):
            if unit == 'degrees':
                pts[...,nn] = pts[...,nn] / 180*np.pi

        # convert to cartesian ...
        # ... from spherical coordinate systems
        if self._system['domain'] == 'sph':
            if self._system['convention'] == 'top_colat':
                x, y, z = sph2cart(pts[...,0], pts[...,1], pts[...,2])

            elif self._system['convention'] == 'top_elev':
                x, y, z = sph2cart(pts[...,0], np.pi/2-pts[...,1], pts[...,2])

            elif self._system['convention'] == 'side':
                x, z, y = sph2cart(pts[...,1], np.pi/2-pts[...,0], pts[...,2])

            elif self._system['convention'] == 'front':
                z, y, x = sph2cart(pts[...,0], pts[...,1], pts[...,2])

            else:
                raise Exception("Conversion for {} is not implemented.".\
                             format(self._system['convention']))

        # ... from cylindrical coordinate systems
        elif self._system['domain'] == 'cyl':
            if self._system['convention'] == 'top':
                x, y, z = cyl2cart(pts[...,0], pts[...,1], pts[...,2])
            else:
                raise Exception("Conversion for {} is not implemented.".\
                             format(self._system['convention']))
        else:
            raise Exception("Conversion for {} is not implemented.".\
                             format(convention))

        # set the new system
        self._system = new_system

        # set points and return
        self._points = self._set_points(x, y, z)
        return self._points


    def get_sph(self, convention='top_colat', unit='rad'):
        """
        Get coordinate points in spherical coordinate system.

        Parameters
        ----------
        convention : string, optional
            convention in which the coordinate points are returned. The default
            is 'top_colat'.
        unit : string, optional
            unit in which the coordinate points are returned. The default is
            'rad'.

        Returns
        -------
        points np.array
            array that holds the coordinate points. points[...,0] hold the
            points for the first coordinate, points[...,1] the points for the
            second, and points[...,2] the points for the third coordinate.

        Note
        ----
        The current and all availanle coordinate systems can be seen with

        >>> c = Coordinates()
        >>> c.system
        >>> c.list_systems()

        """

        # check for points
        if self.num_points == 0:
            raise Exception('Object is empty or contains invalid points')

        # make the new system
        new_system = self._make_system('sph', convention, unit)

        # return if system has not changed
        if new_system == self._system:
            return self._points

        # get cartesian system first
        if not(self._system['domain']=='cart' and self._system['convention']=='right'):
            pts = self.get_cart('right', 'met')
        else:
            pts = self._points

        # convert to spherical...
        # ... top polar systems
        if convention[0:3] == 'top':
            pts_1, pts_2, pts_3 = cart2sph(pts[...,0], pts[...,1], pts[...,2])
            if convention == 'top_elev':
                pts_2 = np.pi/2 - pts_2

        # ... side polar system
        # (ideal for simple converions from Robert Baumgartner and SOFA_API)
        elif convention == 'side':
            pts_2, pts_1, pts_3 = cart2sph(pts[...,0], pts[...,2], -pts[...,1])

            # range angles
            pts_1 = pts_1 - np.pi/2
            pts_2 = np.mod(pts_2 + np.pi/2, 2*np.pi) - np.pi/2

        # ... front polar system
        elif convention == 'front':
            pts_1, pts_2, pts_3 = cart2sph(pts[...,2], pts[...,1], pts[...,0])

        else:
            raise Exception("Conversion for {} is not implemented.".\
                             format(convention))

        # convert to degrees
        if new_system['unit'] == 'deg':
            pts_1 = pts_1 / np.pi*180
            pts_2 = pts_2 / np.pi*180

        # set the new system
        self._system = new_system

        # stack and return
        self._points = self._set_points(pts_1, pts_2, pts_3)
        return self._points


    def get_cyl(self, convention='top', unit='rad'):
        """
        Get coordinate points in cylindrical coordinate system.

        Parameters
        ----------
        convention : string, optional
            convention in which the coordinate points are returned. The default
            is 'top'.
        unit : string, optional
            unit in which the coordinate points are returned. The default is
            'rad'.

        Returns
        -------
        points np.array
            array that holds the coordinate points. points[...,0] hold the
            points for the first coordinate, points[...,1] the points for the
            second, and points[...,2] the points for the third coordinate.

        Note
        ----
        The current and all availanle coordinate systems can be seen with

        >>> c = Coordinates()
        >>> c.system
        >>> c.list_systems()

        """

        # check for points
        if self.num_points == 0:
            raise Exception('Object is empty or contains invalid points')

        # make the new system
        new_system = self._make_system('cyl', convention, unit)

        # return if system has not changed
        if new_system == self._system:
            return self._points

        # convert to cartesian system first
        if not(self._system['domain']=='cart' and self._system['convention']=='right'):
            pts = self.get_cart('right', 'met')
        else:
            pts = self._points

        # convert to cylindrical ...
        # ... top systems
        if convention == 'top':
            pts_1, pts_2, pts_3 = cart2cyl(pts[...,0], pts[...,1], pts[...,2])

        else:
            raise Exception("Conversion for {} is not implemented.".\
                             format(convention))


        # convert to degrees
        if self._system['unit'] == 'deg':
            pts_1 = pts_1 / np.pi*180

        # set the new system
        self._system = new_system

        # stack and return
        self._points = self._set_points(pts_1, pts_2, pts_3)
        return self._points


    @property
    def comment(self):
        """Comment for the data stored in the object."""
        print('getter')
        return self._comment

    @comment.setter
    def comment(self, value):
        print('setter')
        self._comment = value

    @property
    def num_points(self):
        """Return number of coordinate points stored in the object."""
        if np.isnan(self._points).any():
            return 0

        return self._points.shape[0]

    @property
    def coordinates(self):
        """Return current coordinate names and units as sting."""
        coords = ["{} in {}".format(c, u) for c, u in \
                  zip(self._system['coordinates'], self._system['units'])]
        return '; '.join(coords)

    @property
    def system(self):
        """
        Print information about current coordinate system.

        Returns
        -------
        None.

        """
        self.list_systems(self._system['domain'], self._system['convention'],
                          self._system['unit'])
        return None


    def list_systems(self, domain=None, convention=None, unit=None, brief=False):
        """
        List available coordinate systems on the console.

        Systems are specified by their 'domain', 'convention' and 'unit'. If
        multiple units are available for a convention, the unit listed first
        is the default.

        .. note::
           All coordinate systems are described with respect to the right
           handed cartesian system (domain='cart', convention='right').
           Distances are always specified in meters, while angles can be
           radians or degrees (unit='rad' or 'deg').


        Parameters
        ----------
        domain : string, None, optional
            string to get information about a sepcific system, None to get
            information about all systems. The default is None.
        convention : string, None, optional
            string to get information about a sepcific convention, None to get
            information about all conventions. The default is None.
        unit : string, None, optional
            string to get information about a sepcific unit, None to get
            information about all units. The default is None.
        brief . boolean
            Will only list the domains, conventions and units if True. The
            default is False.

        Returns
        -------
        Prints to console.

        Examples
        --------
        List information for all coordinate systems

        >>> self.list_systems()

        List information for a specific coordinate system, e.g.,

        >>> self.list_systems('sph', 'top_elev', 'deg')

        """

        # check user input
        self._exist_system(domain, convention, unit)

        # get coordinate systems
        systems = self._systems()

        # print information
        domains = list(systems) if domain == None else [domain]

        if brief:
            print('domain, convention, unit')
            print('- - - - - - - - - - - - -')
            for dd in domains:
                conventions = list(systems[dd]) if convention == None else [convention]
                for cc in conventions:
                    # current coordinates
                    coords = systems[dd][cc]['coordinates']
                    # current units
                    if unit != None:
                        units = [units for units in systems[dd][cc]['units'] \
                            if unit == units[0][0:3]]
                    else:
                        units = systems[dd][cc]['units']
                    # key for unit
                    unit_key = [u[0][0:3] for u in units]
                    print("{}, {}, [{}]"\
                          .format(dd, cc, ', '.join(unit_key)))
        else:
            for dd in domains:
                conventions = list(systems[dd]) if convention == None else [convention]
                for cc in conventions:
                    # current coordinates
                    coords = systems[dd][cc]['coordinates']
                    # current units
                    if unit != None:
                        units = [units for units in systems[dd][cc]['units'] \
                            if unit == units[0][0:3]]
                    else:
                        units = systems[dd][cc]['units']
                    # key for unit
                    unit_key = [u[0][0:3] for u in units]
                    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                    print("domain: {}, convention: {}, unit: [{}]\n"\
                          .format(dd, cc, ', '.join(unit_key)))
                    print(systems[dd][cc]['description_short'] + '\n')
                    print("Coordinates:")
                    for nn, coord in enumerate(coords):
                        cur_units = [u[nn] for u in units]
                        print("{}: {} [{}]".format(nn+1, coord, ', '.join(cur_units)))
                    print('\n' + systems[dd][cc]['description'] + '\n\n')


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
            Key 2b - 'coordinates': ['coordinate_1','coordinate_2','coordinate_3']
            Key 2c - 'units': [['unit_1.1','unit_2.1','unit_3.1'],
                                            ...
                               ['unit_1.N','unit_2.N','unit_3.N']]
            Key 2d - 'description'

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
                        [["meters", "meters", "meters"]],
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
                        [["radians", "radians", "meters"],
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
                        [["radians", "radians", "meters"],
                         ["degrees", "degrees", "meters"]],
                    "description":
                        "The azimuth denotes the counter clockwise angle in the "\
                        "x/y-plane with 0 pointing in positive x-direction and "\
                        " pi/2 in positive y-direction. The elevation denotes "\
                        "the angle upwards and downwards from the x/y-plane with "\
                        " pi/2 pointing at positive z-direction and -pi/2 "\
                        "pointing in negative z-direction. The azimuth and "\
                        "elevation can be in radians or degrees, the radius is "\
                        " always in meters."},
                "side":{
                    "description_short":
                        "Spherical coordinate system with poles on the y-axis.",
                    "coordinates":
                        ["lateral", "polar", "radius"],
                    "units":
                        [["radians", "radians", "meters"],
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
                        [["radians", "radians", "meters"],
                         ["degrees", "degrees", "meters"]],
                    "description":
                        "Phi denotes the angle in the y/z-plane with 0 "\
                        "pointing in positive z-direction, pi/2 in positive "
                        "y-direction, pi in negative z-direction, and 3*pi/2 "\
                        "in negative y-direction. Theta denotes the angle "\
                        "measured from the x-axis with 0 pointing in positve "\
                        "x-direction and pi in negative x-direction. Phi and "\
                        "theta can be in radians and degrees, the radius is "\
                        "always in meters."}
                },
            "cyl":
                {
                "top":{
                    "description_short":
                        "Cylindrical coordinate system along the z-axis.",
                    "coordinates":
                        ["azimuth", "z", "radius_z"],
                    "units":
                        [["radians", "meters", "meters"],
                         ["degrees", "meters", "meters"]],
                    "description":
                        "The azimuth denotes the counter clockwise angle in the "\
                        "x/y-plane with 0 pointing in positive x-direction and "\
                        " pi/2 in positive y-direction. The heigt is given by "\
                        "z, and radius_z denotes the radius measured orthogonal "\
                        "to the z-axis."}
                }
            }

        return _systems


    def _coordinates(self):
        """
        Get unique list of all coordinate names and their properties.

        Returns
        -------
        coords: nested dictionary
            Resolve coordinate systems in which a coordinate ocurrs and the
            units that a coordinate can have.
            Key 0  - coordinate
            Key 1a - domain
            Key 1b - convention
            Key 1c - units

        """

        # get coordinate systems
        systems = self._systems()

        # resolve membership of coordinates
        coords = {}

        # loop across domains and conventions
        for domain in systems:
            for convention in systems[domain]:
                # loop across coordinates
                for cc, coord in enumerate(systems[domain][convention]['coordinates']):
                    # units of the current coordinate
                    cur_units = [u[cc] for u in systems[domain][convention]['units']]
                    # add coordinate to coords
                    if not coord in coords:
                        coords[coord]= {}
                        coords[coord]['domain']     = [domain]
                        coords[coord]['convention'] = [convention]
                        coords[coord]['units']      = [cur_units]
                    else:
                        coords[coord]['domain'].append(domain)
                        coords[coord]['convention'].append(convention)
                        coords[coord]['units'].append(cur_units)

        return coords


    def _exist_system(self, domain=None, convention=None, unit=None):
        """
        Check if a coordinate system exists and throw an error if it does not.

        The coordinate systems are defined in self._systems.

        Parameters
        ----------
        domain : string
            Sepcify the domain of the coordinate system, e.g., 'cart'.
        convention : string
            The convention of the coordinate system, e.g., 'top_colat'
        units: string
            The unit of the coordinate system (rad, deg, or met for radians,
            degrees, or meters)
        """

        if domain == None and convention != None:
            raise ValueError('convention must be None if domain is None')

        if convention == None and unit != None:
            raise ValueError('units must be None if convention is None')

        # get available coordinate systems
        systems = self._systems()

        # check if domain exists
        assert domain in systems or domain == None, \
            "{} does not exist. Domain must be one of the follwing: {}.".\
                format(domain, ', '.join(list(systems)))

        #check if convention exisits in domain
        if convention != None:
            assert convention in systems[domain] or convention == None,\
                "{} does not exist in {}. Convention must be one of the following: {}.".\
                    format(convention, domain, ', '.join(list(systems[domain])))

        # check if units exist
        if unit != None:
            cur_units = [u[0][0:3] for u in systems[domain][convention]['units']]
            assert unit in cur_units, "{} does not exist in {} ({}). Units must "\
                "be one of the following: {}.".format(unit, domain, convention,
                                                      ', '.join(cur_units))


    def _make_system(self, domain=None, convention=None, unit=None):
        """
        Make and return class internal information about coordinate system.
        """

        # check if coordinate system exists
        self._exist_system(domain, convention, unit)

        # get the new system
        system = self._systems()
        system = system[domain][convention]

        # get the units
        if unit != None:
            units = [units for units in system['units']if unit == units[0][0:3]]
            units = units[0]
        else:
            units = system['units'][0]
            unit  = units[0][0:3]

        # add class internal keys
        system['domain']     = domain
        system['convention'] = convention
        system['unit']       = unit
        system['units']      = units

        return system


    def _set_points(self, points_1, points_2, points_3):
        """
        Check the format of points to be added to Coordinates().

        Retruns
        -------
        points : array
            size [3 x self._num_points]
        """

        # cast to numpy array
        pts_1 = np.asarray(points_1, dtype=np.float64)
        pts_2 = np.asarray(points_2, dtype=np.float64)
        pts_3 = np.asarray(points_3, dtype=np.float64)

        # check dimensions
        for cc, coord in enumerate([pts_1, pts_2, pts_3]):
            assert coord.ndim <= 2, "points_{}.ndim={} but must be <= 2."\
                .format(cc+1, coord.ndim)
            if coord.ndim == 2:
                assert coord.shape[0] == 1 or coord.shape[1] == 1,\
                    "points_{} has shape {} but should have shape ({},), "\
                    "({},1), or (1,{}).".format(cc+1, coord.shape, \
                    max(coord.shape), max(coord.shape), max(coord.shape))

        # flatten input
        pts_1 = np.transpose(np.atleast_2d(pts_1.flatten()))
        pts_2 = np.transpose(np.atleast_2d(pts_2.flatten()))
        pts_3 = np.transpose(np.atleast_2d(pts_3.flatten()))

        # check for scalar entries
        N_max = max([pts_1.shape[0], pts_2.shape[0], pts_3.shape[0]])
        if pts_1.shape[0] == 1:
            pts_1 = np.tile(pts_1, [N_max, 1])
        if pts_2.shape[0] == 1:
            pts_2 = np.tile(pts_2, [N_max, 1])
        if pts_3.shape[0] == 1:
            pts_3 = np.tile(pts_3, [N_max, 1])

        # check for equal length
        assert np.shape(pts_1) == np.shape(pts_2) == np.shape(pts_3),\
            "Input must be of equal length."

        # stack points
        self._points = np.hstack((pts_1, pts_2, pts_3))


def cart2sph(x, y, z):
    """
    Transforms from Cartesian to spherical coordinates.

    Spherical coordinates follow the common convention in Physics/Mathematics.
    The colatitude is measured downwards from the z-axis and is 0 at the North
    Pole and pi at the South Pole. The azimuth is 0 at positive x-direction
    and pi/2 at positive y-direction (counter clockwise rotation).

    Cartesian coordinates follow the right hand rule.

    .. math::

        azimuth &= \\arctan(\\frac{y}{x}),

        colatitude &= \\arccos(\\frac{z}{r}),

        radius &= \\sqrt{x^2 + y^2 + z^2}

    .. math::

        0 < azimuth < 2 \\pi,

        0 < colatitude < \\pi


    Notes
    -----
    To ensure proper handling of the azimuth angle, the arctan2 implementation
    from numpy is used.

    Parameters
    ----------
    x : ndarray, number

    y : ndarray, number

    z : ndarray, number

    Returns
    -------
    azimuth : ndarray, number

    colatitude : ndarray, number

    radius : ndarray, number
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    colatitude = np.arccos(z/radius)
    azimuth = np.mod(np.arctan2(y, x), 2*np.pi)
    return azimuth, colatitude, radius


def sph2cart(azimuth, colatitude, radius):
    """
    Transforms from spherical to Cartesian coordinates.

    Spherical coordinates follow the common convention in Physics/Mathematics.
    The colatitude is measured downwards from the z-axis and is 0 at the North
    Pole and pi at the South Pole. The azimuth is 0 at positive x-direction
    and pi/2 at positive y-direction (counter clockwise rotation).

    Cartesian coordinates follow the right hand rule.

    .. math::

        x &= radius * \\sin(colatitude) * \\cos(azimuth),

        y &= radius * \\sin(colatitude) * \\sin(azimuth),

        z &= radius * \\cos(colatitude)

    .. math::

        0 < azimuth < 2 \\pi

        0 < colatitude < \\pi


    Parameters
    ----------
    azimuth : ndarray, number

    colatitude : ndarray, number

    radius : ndarray, number

    Returns
    -------
    x : ndarray, number

    y : ndarray, number

    z : ndarray, number
    """
    r_sin_cola = radius * np.sin(colatitude)
    x = r_sin_cola * np.cos(azimuth)
    y = r_sin_cola * np.sin(azimuth)
    z = radius * np.cos(colatitude)

    return x, y, z


def cart2cyl(x, y, z):
    """
    Transforms from Cartesian to cylindrical coordinates.

    Cylindrical coordinates follow the convention that the azimuth is 0 at
    positive x-direction and pi/2 at positive y-direction (counter clockwise
    rotation). The height is identical to the z-coordinate and the radius is
    measured orthogonal from the z-axis.

    Cartesian coordinates follow the right hand rule.

    .. math::

        azimuth &= \\arctan(\\frac{y}{x}),

        height &= z,

        radius &= \\sqrt{x^2 + y^2},

    .. math::

        0 < azimuth < 2 \\pi


    Notes
    -----
    To ensure proper handling of the azimuth angle, the arctan2 implementation
    from numpy is used.

    Parameters
    ----------
    x : ndarray, number

    y : ndarray, number

    z : ndarray, number

    Returns
    -------
    azimuth : ndarray, number

    height : ndarray, number

    radius : ndarray, number
    """

    azimuth = np.mod(np.arctan2(y, x), 2*np.pi)
    try:
        height = z.copy()
    except:
        height = z
    radius  = np.sqrt(x**2 + y**2)

    return azimuth, height, radius


def cyl2cart(azimuth, height, radius):
    """
    Transforms from cylindrical to Cartesian coordinates.

    Cylindrical coordinates follow the convention that the azimuth is 0 at
    positive x-direction and pi/2 at positive y-direction (counter clockwise
    rotation). The height is identical to the z-coordinate and the radius is
    measured orthogonal from the z-axis.

    Cartesian coordinates follow the right hand rule.

    .. math::

        x &= radius * \\cos(azimuth),

        y &= radius * \\sin(azimuth),

        z &= height

    .. math::

        0 < azimuth < 2 \\pi


    Notes
    -----
    To ensure proper handling of the azimuth angle, the arctan2 implementation
    from numpy is used.

    Parameters
    ----------
    azimuth : ndarray, number

    height : ndarray, number

    radius : ndarray, number

    Returns
    -------
    x : ndarray, number

    y : ndarray, number

    z : ndarray, number
    """

    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)
    try:
        z = height.copy()
    except:
        z = height

    return x, y, z

# def _cart2latlon(x, y, z):
#     """Transforms from Cartesian coordinates to Geocentric coordinates

#     .. math::

#         h = \\sqrt{x^2 + y^2 + z^2},

#         \\theta = \\pi/2 - \\arccos(\\frac{z}{r}),

#         \\phi = \\arctan(\\frac{y}{x})

#         -\\pi/2 < \\theta < \\pi/2,

#         -\\pi < \\phi < \\pi

#     where :math:`h` is the heigth, :math:`\\theta` is the latitude angle
#     and :math:`\\phi` is the longitude angle

#     Parameters
#     ----------
#     x : ndarray, number
#         x-axis coordinates
#     y : ndarray, number
#         y-axis coordinates
#     z : ndarray, number
#         z-axis coordinates

#     Returns
#     -------
#     height : ndarray, number
#         The radius is rendered as height information
#     latitude : ndarray, number
#         Geocentric latitude angle
#     longitude : ndarray, number
#         Geocentric longitude angle

#     """
#     height = np.sqrt(x**2 + y**2 + z**2)
#     latitude = np.pi/2 - np.arccos(z/height)
#     longitude = np.arctan2(y, x)
#     return height, latitude, longitude


# def _latlon2cart(height, latitude, longitude):
#     """Transforms from Geocentric coordinates to Cartesian coordinates

#     .. math::

#         x = h \\cos(\\theta) \\cos(\\phi),

#         y = h \\cos(\\theta) \\sin(\\phi),

#         z = h \\sin(\\theta)

#         -\\pi/2 < \\theta < \\pi/2,

#         -\\pi < \\phi < \\pi

#     where :math:`h` is the heigth, :math:`\\theta` is the latitude angle
#     and :math:`\\phi` is the longitude angle

#     Parameters
#     ----------
#     height : ndarray, number
#         The radius is rendered as height information
#     latitude : ndarray, number
#         Geocentric latitude angle
#     longitude : ndarray, number
#         Geocentric longitude angle

#     Returns
#     -------
#     x : ndarray, number
#         x-axis coordinates
#     y : ndarray, number
#         y-axis coordinates
#     z : ndarray, number
#         z-axis coordinates

#     """

#     x = height * np.cos(latitude) * np.cos(longitude)
#     y = height * np.cos(latitude) * np.sin(longitude)
#     z = height * np.sin(latitude)

#     return x, y, z
