"""
Collection of sampling schemes for the sphere
"""
# import os
import numpy as np
import urllib3
from haiopy.coordinates import Coordinates


def cube_equidistant(n_points):
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
        domain='cart')

    return sampling


def dodecahedron():
    """Generate a sampling based on the center points of the twelve
    dodecahedron faces.

    Returns
    -------
    rad : ndarray
        Radius of the sampling points
    theta : ndarray
        Elevation angle in the range [0, pi]
    phi : ndarray
        Azimuth angle in the range [0, 2 pi]
    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points
    """

    dihedral = 2*np.arcsin(np.cos(np.pi/3)/np.sin(np.pi/5))
    R = np.tan(np.pi/3)*np.tan(dihedral/2)
    rho = np.cos(np.pi/5)/np.sin(np.pi/10)

    theta1 = np.arccos((np.cos(np.pi/5)/np.sin(np.pi/5))/np.tan(np.pi/3))

    a2 = 2*np.arccos(rho/R)

    theta2 = theta1+a2
    theta3 = np.pi - theta2
    theta4 = np.pi - theta1

    phi1 = 0
    phi2 = 2*np.pi/3
    phi3 = 4*np.pi/3

    theta = np.concatenate((
        np.tile(theta1, 3),
        np.tile(theta2, 3),
        np.tile(theta3, 3),
        np.tile(theta4, 3)))
    phi = np.tile(np.array([
        phi1,
        phi2,
        phi3,
        phi1 + np.pi/3,
        phi2 + np.pi/3,
        phi3 + np.pi/3]), 2)
    rad = np.ones(np.size(theta))

    sampling = Coordinates(
        phi, theta, rad, domain='sph', convention='top_colat')
    return sampling


def icosahedron():
    """Generate a sampling based on the center points of the twenty \
            icosahedron faces.

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points
    """
    gamma_R_r = np.arccos(np.cos(np.pi/3) / np.sin(np.pi/5))
    gamma_R_rho = np.arccos(1/(np.tan(np.pi/5) * np.tan(np.pi/3)))

    theta = np.tile(np.array([np.pi - gamma_R_rho,
                              np.pi - gamma_R_rho - 2*gamma_R_r,
                              2*gamma_R_r + gamma_R_rho,
                              gamma_R_rho]), 5)
    theta = np.sort(theta)
    phi = np.arange(0, 2*np.pi, 2*np.pi/5)
    phi = np.concatenate((np.tile(phi, 2), np.tile(phi + np.pi/5, 2)))

    rad = np.ones(20)
    sampling = Coordinates(
        phi, theta, rad, domain='sph', convention='top_colat')
    return sampling


def sphere_equiangular(n_points=None, angles=None):
    """Generate an equiangular sampling of the sphere.

    Paramters
    ---------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

    """
    if (n_points is None) and (angles is None):
        raise ValueError("Either the number of points or the angular distance \
            needs to be specified.")

    if n_points is not None:
        n_points = np.asarray(n_points)
        if n_points.size == 2:
            n_phi = n_points[0]
            n_theta = n_points[1]
        else:
            n_phi = n_points
            n_theta = n_points

        theta_angles = np.arange(
            np.pi/(n_theta*2), np.pi, np.pi/n_theta)
        phi_angles = np.arange(
            0, 2*np.pi, 2*np.pi/n_phi)

    elif angles is not None:
        angles = np.asarray(angles)
        if angles.size == 2:
            alpha_phi = angles[0]
            alpha_theta = angles[1]
        else:
            alpha_phi = angles
            alpha_theta = angles

        theta_angles = np.arange(
            np.pi/(n_theta*2), np.pi, alpha_theta)
        phi_angles = np.arange(
            0, 2*np.pi, alpha_phi)

    theta, phi = np.meshgrid(theta_angles, phi_angles)
    rad = np.ones(theta.size)

    sampling = Coordinates(
        phi.reshape(-1),
        theta.reshape(-1),
        rad,
        domain='sph',
        convention='top_colat')

    return sampling


def sphere_gaussian(n_max):
    """Generate sampling of the sphere based on the Gaussian quadrature.

    Paramters
    ---------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

    """
    legendre, weights = np.polynomial.legendre.leggauss(n_max+1)
    theta_angles = np.arccos(legendre)
    n_phi = np.round((n_max+1)*2)
    phi_angles = np.arange(0,
                            2*np.pi,
                            2*np.pi/n_phi)
    theta, phi = np.meshgrid(theta_angles, phi_angles)
    rad = np.ones(theta.size)
    weights = np.tile(weights*np.pi/(n_max+1), 2*(n_max+1))

    sampling = Coordinates(phi.reshape(-1), theta.reshape(-1), rad,
                           domain='sph', convention='top_colat')
    sampling.weights = weights
    return sampling


def hyperinterpolation(n_max):
    """Gives the points of a Hyperinterpolation sampling grid
    after Sloan and Womersley [1]_.

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

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    sampling: SamplingSphere
        SamplingSphere object containing all sampling points
    """
    n_sh = (n_max+1)**2
    filename = "/Womersley/md%02d.%04d" % (n_max, n_sh)
    url = "http://www.ita-toolbox.org/Griddata"
    fileurl = url + filename

    http = urllib3.PoolManager()
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    file_data = np.fromstring(
        file_data,
        dtype='double',
        sep=' ').reshape((n_sh, 4))
    sampling = Coordinates(
        file_data[:, 0],
        file_data[:, 1],
        file_data[:, 2])
    sampling.weights = file_data[:, 3]

    return sampling


def spherical_t_design(n_max, criterion='const_energy'):
    r"""Return the sampling positions for a spherical t-design [1]_ .
    For a spherical harmonic order N, a t-Design of degree `:math: t=2N` for
    constant energy or `:math: t=2N+1` additionally ensuring a constant angular
    spread of energy is required [2]_. For a given degree t

    .. math::

        L = \lceil \frac{(t+1)^2}{2} \rceil+1,

    points will be generated, except for t = 3, 5, 7, 9, 11, 13, and 15.
    T-designs allow for a inverse spherical harmonic transform matrix
    calculated as `:math: D = \frac{4\pi}{L} \mathbf{Y}^\mathrm{H}`.

    Parameters
    ----------
    degree : integer
        T-design degree
    criterion : 'const_energy', 'const_angular_spread'
        Design criterion ensuring only a constant energy or additionally
        constant angular spread of energy

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

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
    if criterion == 'const_energy':
        degree = 2*n_max
    elif criterion == 'const_angular_spread':
        degree = 2*n_max + 1
    else:
        raise ValueError("Invalid design criterion.")

    n_points = np.int(np.ceil((degree + 1)**2 / 2) + 1)
    n_points_exceptions = {3:8, 5:18, 7:32, 9:50, 11:72, 13:98, 15:128}
    if degree in n_points_exceptions:
        n_points = n_points_exceptions[degree]

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

    points = np.fromstring(
        file_data,
        dtype=np.double,
        sep=' ').reshape((n_points, 3))

    sampling = Coordinates(points[...,0], points[...,1], points[...,2])

    return sampling

