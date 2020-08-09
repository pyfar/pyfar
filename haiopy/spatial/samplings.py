"""
Collection of sampling schemes for the sphere
"""
# import os
import numpy as np
import urllib3
from haiopy.coordinates import Coordinates
from haiopy.spatial import _lebedev


def cart_equidistant_cube(n_points):
    """Create a cuboid sampling with equidistant spacings in x, y, and z.
    The cube will have dimensions 1 x 1 x 1

    Parameters
    ----------
    n_points : int, tuple
        Number of points in the sampling. If a single value is given, the
        number of sampling positions will be the same in every axis. If a
        tuple is given, the number of points will be set as (n_x, n_y, n_z)

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    """
    if np.size(n_points) == 1:
        n_x = n_points
        n_y = n_points
        n_z = n_points
    elif np.size(n_points) == 3:
        n_x = n_points[0]
        n_y = n_points[1]
        n_z = n_points[2]
    else:
        raise ValueError("The number of points needs to be either an integer \
                or a tuple with 3 elements.")

    x = np.linspace(-1, 1, n_x)
    y = np.linspace(-1, 1, n_y)
    z = np.linspace(-1, 1, n_z)

    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    sampling = Coordinates(
        x_grid.flatten(),
        y_grid.flatten(),
        z_grid.flatten(),
        domain='cart',
        comment='equidistant cuboid sampling grid')

    return sampling


def sph_dodecahedron(radius=1.):
    """Generate a sampling based on the center points of the twelve
    dodecahedron faces.

    Parameters
    ----------
    radius : number, optional
        Radius of the sampling grid

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    """

    dihedral = 2 * np.arcsin(np.cos(np.pi / 3) / np.sin(np.pi / 5))
    R = np.tan(np.pi / 3) * np.tan(dihedral / 2)
    rho = np.cos(np.pi / 5) / np.sin(np.pi / 10)

    theta1 = np.arccos((np.cos(np.pi / 5)
                       / np.sin(np.pi / 5))
                       / np.tan(np.pi / 3))

    a2 = 2 * np.arccos(rho / R)

    theta2 = theta1 + a2
    theta3 = np.pi - theta2
    theta4 = np.pi - theta1

    phi1 = 0
    phi2 = 2 * np.pi / 3
    phi3 = 4 * np.pi / 3

    theta = np.concatenate((
        np.tile(theta1, 3),
        np.tile(theta2, 3),
        np.tile(theta3, 3),
        np.tile(theta4, 3)))
    phi = np.tile(np.array([
        phi1,
        phi2,
        phi3,
        phi1 + np.pi / 3,
        phi2 + np.pi / 3,
        phi3 + np.pi / 3]), 2)
    rad = radius * np.ones(np.size(theta))

    sampling = Coordinates(
        phi, theta, rad, domain='sph', convention='top_colat',
        comment='dodecahedral sampling grid')
    return sampling


def sph_icosahedron(radius=1.):
    """Generate a sampling based on the center points of the twenty \
            icosahedron faces.

    Parameters
    ----------
    radius : number, optional
        Radius of the sampling grid

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    """
    gamma_R_r = np.arccos(np.cos(np.pi / 3) / np.sin(np.pi / 5))
    gamma_R_rho = np.arccos(1 / (np.tan(np.pi / 5) * np.tan(np.pi / 3)))

    theta = np.tile(np.array([np.pi - gamma_R_rho,
                              np.pi - gamma_R_rho - 2 * gamma_R_r,
                              2 * gamma_R_r + gamma_R_rho,
                              gamma_R_rho]), 5)
    theta = np.sort(theta)
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / 5)
    phi = np.concatenate((np.tile(phi, 2), np.tile(phi + np.pi / 5, 2)))

    rad = radius * np.ones(20)
    sampling = Coordinates(phi, theta, rad,
                           domain='sph', convention='top_colat',
                           comment='icosahedral spherical sampling grid')
    return sampling


def sph_equiangular(n_points=None, n_sh=None, radius=1.):
    """Generate an equiangular sampling of the sphere [1]_, Chapter 3.2.

    Parameters
    ----------
    n_points : int, tuple of two ints, optional
        number of sampling points in azimuth and elevation
    n_sh : int, optional
        maximum applicable spherical harmonics order. If this is provided,
        'n_points' is set to 2 * n_sh + 1
    radius : number, optional
        radius of the sampling grid

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    References
    ----------
    .. [1] B. Rafaely, Fundamentals of spherical array processing, 1st ed.
           Berlin, Heidelberg, Germany: Springer, 2015.

    """
    if (n_points is None) and (n_sh is None):
        raise ValueError("Either the n_points or n_sh needs to be specified.")

    # get number of points from required spherical harmonics order
    # ([1], equation 3.4)
    if n_sh is not None:
        n_points = 2 * (int(n_sh) + 1)

    # get the angles
    n_points = np.asarray(n_points)
    if n_points.size == 2:
        n_phi = n_points[0]
        n_theta = n_points[1]
    else:
        n_phi = n_points
        n_theta = n_points

    theta_angles = np.arange(np.pi / (n_theta * 2), np.pi, np.pi / n_theta)
    phi_angles = np.arange(0, 2 * np.pi, 2 * np.pi / n_phi)

    # construct the sampling grid
    theta, phi = np.meshgrid(theta_angles, phi_angles)
    rad = radius * np.ones(theta.size)

    # compute maximum applicable spherical harmonics order
    if n_sh is None:
        n_max = int(np.min([n_phi / 2 - 1, n_theta / 2 - 1]))
    else:
        n_max = int(n_sh)

    # compute sampling weights ([1], equation 3.11)
    tmp = 2 * np.arange(0, n_max + 1) + 1
    w = np.zeros_like(theta_angles)
    for nn, tt in enumerate(theta_angles):
        w[nn] = 2 * np.pi / (n_max + 1)**2 * np.sin(tt) \
            * np.sum(1 / tmp * np.sin(tmp * tt))

    # repeat and normalize sampling weights
    w = np.tile(w, n_phi)
    w = w / np.sum(w)

    # make Coordinates object
    sampling = Coordinates(phi.reshape(-1), theta.reshape(-1), rad,
                           domain='sph', convention='top_colat',
                           comment='equiangular spherical sampling grid',
                           weights=w, sh_order=n_max)

    return sampling


def sph_gaussian(n_points=None, n_sh=None, radius=1.):
    """Generate sampling of the sphere based on the Gaussian quadrature [1]_.

    Parameters
    ----------
    n_points : int, tuple of two ints, optional
        number of sampling points in azimuth and elevation
    n_sh : int, optional
        maximum applicable spherical harmonics order. If this is provided,
        'n_points' is set to [2 * (n_sh + 1), n_sh + 1]
    radius : number, optional
        radius of the sampling grid in meters

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    References
    ----------
    .. [1] B. Rafaely, Fundamentals of spherical array processing, 1st ed.
           Berlin, Heidelberg, Germany: Springer, 2015.

    """
    if (n_points is None) and (n_sh is None):
        raise ValueError("Either the n_points or n_sh needs to be specified.")

    # get number of points from required spherical harmonics order
    # ([1], equation 3.4)
    if n_sh is not None:
        n_points = [2 * (int(n_sh) + 1), int(n_sh) + 1]

    # get the number of points in both dimensions
    n_points = np.asarray(n_points)
    if n_points.size == 2:
        n_phi = n_points[0]
        n_theta = n_points[1]
    else:
        n_phi = n_points
        n_theta = n_points

    # compute the maximum applicable spherical harmonics order
    if n_sh is None:
        n_max = int(np.min([n_phi / 2 - 1, n_theta - 1]))
    else:
        n_max = int(n_sh)

    # construct the sampling grid
    legendre, weights = np.polynomial.legendre.leggauss(int(n_theta))
    theta_angles = np.arccos(legendre)

    phi_angles = np.arange(0, 2 * np.pi, 2 * np.pi / n_phi)
    theta, phi = np.meshgrid(theta_angles, phi_angles)

    rad = radius * np.ones(theta.size)

    # compute the sampling weights
    weights = np.tile(weights, n_phi)
    weights = weights / np.sum(weights)

    # make Coordinates object
    sampling = Coordinates(phi.reshape(-1), theta.reshape(-1), rad,
                           domain='sph', convention='top_colat',
                           comment='gaussian spherical sampling grid',
                           weights=weights, sh_order=n_max)

    return sampling


def sph_extremal(n_points=None, n_sh=None, radius=1.):
    """Gives the points of a Hyperinterpolation sampling grid
    after Sloan and Womersley [1]_.

    Parameters
    ----------
    n_points : int, optional
        number of sampling points in the grid. Related to the spherical
        harmonics order by n_points = (n_sh + 1)**2
    n_sh : int, optional
        maximum applicable spherical harmonics order. Related to the number of
        points by n_sh = np.sqrt(n_points) - 1
    radius : number, optional
        radius of the sampling grid in meters

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    Notes
    -----
    This implementation uses precalculated sets of points which are downloaded
    from Womersley's homepage [2]_.

    References
    ----------
    .. [1]  I. H. Sloan and R. S. Womersley, “Extremal Systems of Points and
            Numerical Integration on the Sphere,” Advances in Computational
            Mathematics, vol. 21, no. 1/2, pp. 107–125, 2004.
    .. [2]  http://web.maths.unsw.edu.au/~rsw/Sphere/Extremal/New/index.html

    """

    if (n_points is None) and (n_sh is None):
        raise ValueError("Either the n_points or n_sh needs to be specified.")

    # get number of points or spherical harmonics order
    if n_sh is not None:
        n_points = (n_sh + 1)**2
    else:
        n_sh = np.sqrt(n_points) - 1

    # load the data
    filename = "md%03d.%05d" % (n_sh, n_points)
    url = "https://web.maths.unsw.edu.au/~rsw/Sphere/Extremal/New/"
    fileurl = url + filename

    http = urllib3.PoolManager(cert_reqs=False)
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    # format data
    file_data = np.fromstring(
        file_data,
        dtype='double',
        sep=' ').reshape((int(n_points), 4))

    # generate Coordinates object
    sampling = Coordinates(file_data[:, 0] * radius,
                           file_data[:, 1] * radius,
                           file_data[:, 2] * radius,
                           sh_order=n_sh, weights=file_data[:, 3],
                           comment='extremal spherical sampling grid')

    return sampling


def sph_t_design(degree=None, n_sh=None, criterion='const_energy', radius=1.):
    r"""Return spherical t-design sampling grid [1]_.

    For a spherical harmonic order :math:`n_{sh}`, a t-Design of degree
    :math:`t=2n_{sh}` for constant energy or :math:`t=2n_{sh}+1` additionally
    ensuring a constant angular spread of energy is required [2]_. For a given
    degree t

    .. math::

        L = \lceil \frac{(t+1)^2}{2} \rceil+1,

    points will be generated, except for t = 3, 5, 7, 9, 11, 13, and 15.
    T-designs allow for an inverse spherical harmonic transform matrix
    calculated as :math:`D = \frac{4\pi}{L} \mathbf{Y}^\mathrm{H}` with
    :math:`\mathbf{Y}^\mathrm{H}` being the hermitian transpose of the
    spherical harmonics matrix.

    Parameters
    ----------
    degree : int, optional
        T-design degree
    criterion : 'const_energy', 'const_angular_spread'
        Design criterion ensuring only a constant energy or additionally
        constant angular spread of energy

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    Notes
    -----
    This function downloads a pre-calculated set of points from
    Rob Womersley's homepage [3]_ .

    References
    ----------

    .. [1]  C. An, X. Chen, I. H. Sloan, and R. S. Womersley, “Well Conditioned
            Spherical Designs for Integration and Interpolation on the
            Two-Sphere,” SIAM Journal on Numerical Analysis, vol. 48, no. 6,
            pp. 2135–2157, Jan. 2010.
    .. [2]  F. Zotter, M. Frank, and A. Sontacchi, “The Virtual T-Design
            Ambisonics-Rig Using VBAP,” in Proceedings on the Congress on
            Sound and Vibration, 2010.
    .. [3]  http://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/sf.html

    """

    # check input
    if (degree is None) and (n_sh is None):
        raise ValueError("Either the degree or n_sh needs to be specified.")

    if criterion not in ['const_energy', 'const_angular_spread']:
        raise ValueError("Invalid design criterion. Must be 'const_energy' \
                         or 'const_angular_spread'.")

    # get the degree
    if degree is None:
        if criterion == 'const_energy':
            degree = 2 * n_sh
        else:
            degree = 2 * n_sh + 1

    # get the number of points
    n_points = np.int(np.ceil((degree + 1)**2 / 2) + 1)
    n_points_exceptions = {3: 8, 5: 18, 7: 32, 9: 50, 11: 72, 13: 98, 15: 128}
    if degree in n_points_exceptions:
        n_points = n_points_exceptions[degree]

    # load the data
    filename = "sf%03d.%05d" % (degree, n_points)
    url = "http://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/SF29-Nov-2012/"
    fileurl = url + filename

    http = urllib3.PoolManager(
        cert_reqs=False)
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    elif http_data.status == 404:
        raise FileNotFoundError("File was not found. Check if the design you \
                are trying to calculate is a valid t-design.")
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    # format the data
    points = np.fromstring(
        file_data,
        dtype=np.double,
        sep=' ').reshape((n_points, 3))

    # generate Coordinates object
    sampling = Coordinates(points[..., 0] * radius,
                           points[..., 1] * radius,
                           points[..., 2] * radius,
                           sh_order=n_sh,
                           comment='spherical T-design sampling grid')

    return sampling


def sph_great_circle(elevation=np.linspace(-90, 90, 19), gcd=10, radius=1,
                     azimuth_res=1, match=360):
    r"""
    Spherical sampling grid according to the great circle distance criterion.

    Sampling grid where neighboring points of the same elevation have approx.
    the same great circle distance across elevations [1]_.

    Parameters
    ----------
    elevation : array like, optional
        Contains the elevation from wich the sampling grid is generated, with
        :math:`-90^\circ\leq elevation \leq 90^\circ` (:math:`90^\circ`:
        North Pole, :math:`-90^\circ`: South Pole). The default is
        np.linspace(-90,90,19).
    gcd : number, optional
        Desired great circle distance (GCD). Note that the actual GCD of the
        sampling grid is equal or larger then the desired GCD and that the GCD
        may vary across elevatoins. The default is 10.
    radius : number, optional
        Radius of the sampling grid in meters. The default is 1.
    azimuth_res : number, optional
        Minimum resolution of the azimuth angle in degree. The default is 1.
    match : number, optional
        Forces azimuth entries to appear with a period of match degrees. E.g.,
        if match=90, the grid will have azimuth angles at 0, 90, 180, and 270
        degrees (and possibly inbetween). The default is 360.

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    References
    ----------
    .. [1]  B. P. Bovbjerg, F. Christensen, P. Minnaar, and X. Chen, “Measuring
            the head-related transfer functions of an artificial head with a
            high directional resolution,” Los Angeles, USA, Sep. 2000.

    """

    # check input
    assert not 1 % azimuth_res, "1/azimuth_res must be an integer."
    assert not 360 % match, "360/match must be an integer."
    assert not match % azimuth_res, "match/azimuth_res must be an integer."

    elevation = np.atleast_1d(np.asarray(elevation))

    # calculate delta azimuth to meet the desired great circle distance.
    # (according to Bovbjerg et al. 2000: Measuring the head related transfer
    # functions of an artificial head with a high directional azimuth_res)
    d_az = 2 * np.arcsin(np.clip(
        np.sin(gcd / 360 * np.pi) / np.cos(elevation / 180 * np.pi), -1, 1))
    d_az = d_az / np.pi * 180
    # correct values at the poles
    d_az[np.abs(elevation) == 90] = 360
    # next smallest value in desired angular azimuth_res
    d_az = d_az // azimuth_res * azimuth_res

    # adjust phi to make sure that: match // d_az == 0
    for nn in range(d_az.size):
        if abs(elevation[nn]) != 90:
            while match % d_az[nn]:
                d_az[nn] = d_az[nn] - azimuth_res

    # construct the full sampling grid
    azim = np.empty(0)
    elev = np.empty(0)
    for nn in range(elevation.size):
        azim = np.append(azim, np.arange(0, 360, d_az[nn]))
        elev = np.append(elev, np.full(int(360 / d_az[nn]), elevation[nn]))

    # make Coordinates object
    sampling = Coordinates(azim, elev, radius, 'sph', 'top_elev', 'deg',
                           comment='spherical great circle sampling grid')

    return sampling


def sph_lebedev(n_points=None, n_sh=None, radius=1.):
    """
    Return Lebedev spherical sampling grid [1]_.

    Parameters
    ----------
    n_points : int, optional
        number of sampling points in the grid. Related to the spherical
        harmonics order by n_points = (n_sh + 1)**2
    n_sh : int, optional
        maximum applicable spherical harmonics order. Related to the number of
        points by n_sh = np.sqrt(n_points) - 1
    radius : number, optional
        radius of the sampling grid in meters

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    Notes
    -----
    This implementation is based on Matlab Code written by Rob Parrish [2]_.

    References
    ----------
    .. [1] V.I. Lebedev, and D.N. Laikov
           "A quadrature formula for the sphere of the 131st
           algebraic order of accuracy"
           Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    .. [2] https://de.mathworks.com/matlabcentral/fileexchange/27097-\
        getlebedevsphere

    """

    # possible degrees
    degrees = np.array([6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230,
                        266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730,
                        2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294,
                        5810], dtype=int)

    # corresponding spherical harmonics orders
    orders = np.array((np.floor(np.sqrt(degrees / 1.3) - 1)), dtype=int)

    # list possible sh orders and degrees
    if n_points is None and n_sh is None:
        for o, d in zip(orders, degrees):
            print(f"SH order {o}, number of points {d}\n")

        return 0

    # check input
    if n_points is not None and n_sh is not None:
        raise ValueError("Either n_points or n_sh must be None.")

    # check if the order is available
    if n_sh is not None:
        if n_sh not in orders:
            str_orders = [f"{o}" for o in orders]
            raise ValueError("Invalid spherical harmonics order 'n_sh'. \
                             Valid orders are: {}.".format(
                             ', '.join(str_orders)))

        n_points = int(degrees[orders == n_sh])

    # check if n_points is available
    if n_points not in degrees:
        str_degrees = [f"{d}" for d in degrees]
        raise ValueError("Invalid number of points n_points. Valid degrees \
                         are: {}.".format(', '.join(str_degrees)))

    # calculate sh_order
    n_sh = int(orders[degrees == n_points])

    leb = _lebedev._lebedevSphere(n_points)

    # generate Coordinates object
    sampling = Coordinates(leb["x"] * radius,
                           leb["y"] * radius,
                           leb["z"] * radius,
                           sh_order=n_sh, weights=leb["w"],
                           comment='spherical Lebedev sampling grid')

    return sampling
