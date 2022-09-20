import numpy as np
import urllib3
import os
import scipy.io as sio
import pyfar

from . import external


def cart_equidistant_cube(n_points):
    """
    Create a cuboid sampling with equidistant spacings in x, y, and z.

    The cube will have dimensions 1 x 1 x 1.

    Parameters
    ----------
    n_points : int, tuple
        Number of points in the sampling. If a single value is given, the
        number of sampling positions will be the same in every axis. If a
        tuple is given, the number of points will be set as
        ``(n_x, n_y, n_z)``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Does not contain sampling weights.

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

    sampling = pyfar.Coordinates(
        x_grid.flatten(),
        y_grid.flatten(),
        z_grid.flatten(),
        domain='cart',
        comment='equidistant cuboid sampling grid')

    return sampling


def sph_dodecahedron(radius=1.):
    """
    Generate a sampling based on the center points of the twelve
    dodecahedron faces.

    Parameters
    ----------
    radius : number, optional
        Radius of the sampling grid. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Sampling weights can be obtained from
        :py:func:`calculate_sph_voronoi_weights`.

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

    sampling = pyfar.Coordinates(
        phi, theta, rad, domain='sph', convention='top_colat',
        comment='dodecahedral sampling grid')
    return sampling


def sph_icosahedron(radius=1.):
    """
    Generate a sampling from the center points of the twenty icosahedron faces.

    Parameters
    ----------
    radius : number, optional
        Radius of the sampling grid. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Sampling weights can be obtained from
        :py:func:`calculate_sph_voronoi_weights`.

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
    sampling = pyfar.Coordinates(
        phi, theta, rad,
        domain='sph', convention='top_colat',
        comment='icosahedral spherical sampling grid')
    return sampling


def sph_equiangular(n_points=None, sh_order=None, radius=1.):
    """
    Generate an equiangular sampling of the sphere.

    For detailed information, see [#]_, Chapter 3.2.
    This sampling does not contain points at the North and South Pole and is
    typically used for spherical harmonics processing. See
    :py:func:`sph_equal_angle` and :py:func:`sph_great_circle` for samplings
    containing points at the poles.

    Parameters
    ----------
    n_points : int, tuple of two ints
        Number of sampling points in azimuth and elevation. Either `n_points`
        or `sh_order` must be provided. The default is ``None``.
    sh_order : int
        Maximum applicable spherical harmonic order. If this is provided,
        'n_points' is set to ``2 * sh_order + 1``. Either `n_points` or
        `sh_order` must be provided. The default is ``None``.
    radius : number, optional
        Radius of the sampling grid. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions including sampling weights.

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of spherical array processing, 1st ed.
           Berlin, Heidelberg, Germany: Springer, 2015.

    """
    if (n_points is None) and (sh_order is None):
        raise ValueError(
            "Either the n_points or sh_order needs to be specified.")

    # get number of points from required spherical harmonic order
    # ([#], equation 3.4)
    if sh_order is not None:
        n_points = 2 * (int(sh_order) + 1)

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

    # compute maximum applicable spherical harmonic order
    if sh_order is None:
        n_max = int(np.min([n_phi / 2 - 1, n_theta / 2 - 1]))
    else:
        n_max = int(sh_order)

    # compute sampling weights ([1], equation 3.11)
    q = 2 * np.arange(0, n_max + 1) + 1
    w = np.zeros_like(theta_angles)
    for nn, tt in enumerate(theta_angles):
        w[nn] = 2 * np.pi / (n_max + 1)**2 * np.sin(tt) \
            * np.sum(1 / q * np.sin(q * tt))

    # repeat and normalize sampling weights
    w = np.tile(w, n_phi)
    w = w / np.sum(w)

    # make Coordinates object
    sampling = pyfar.Coordinates(
        phi.reshape(-1), theta.reshape(-1), rad,
        domain='sph', convention='top_colat',
        comment='equiangular spherical sampling grid',
        weights=w, sh_order=n_max)

    return sampling


def sph_gaussian(n_points=None, sh_order=None, radius=1.):
    """
    Generate sampling of the sphere based on the Gaussian quadrature.

    For detailed information, see [#]_ (Section 3.3).
    This sampling does not contain points at the North and South Pole and is
    typically used for spherical harmonics processing. See
    :py:func:`sph_equal_angle` and :py:func:`sph_great_circle` for samplings
    containing points at the poles.

    Parameters
    ----------
    n_points : int, tuple of two ints
        Number of sampling points in azimuth and elevation. Either `n_points`
        or `sh_order` must be provided. The default is ``None``.
    sh_order : int
        Maximum applicable spherical harmonic order. If this is provided,
        `n_points` is set to ``(2 * (sh_order + 1), sh_order + 1)``. Either
        `n_points` or `sh_order` must be provided. The default is ``None``.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions including sampling weights.

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of spherical array processing, 1st ed.
           Berlin, Heidelberg, Germany: Springer, 2015.

    """
    if (n_points is None) and (sh_order is None):
        raise ValueError(
            "Either the n_points or sh_order needs to be specified.")

    # get number of points from required spherical harmonic order
    # ([1], equation 3.4)
    if sh_order is not None:
        n_points = [2 * (int(sh_order) + 1), int(sh_order) + 1]

    # get the number of points in both dimensions
    n_points = np.asarray(n_points)
    if n_points.size == 2:
        n_phi = n_points[0]
        n_theta = n_points[1]
    else:
        n_phi = n_points
        n_theta = n_points

    # compute the maximum applicable spherical harmonic order
    if sh_order is None:
        n_max = int(np.min([n_phi / 2 - 1, n_theta - 1]))
    else:
        n_max = int(sh_order)

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
    sampling = pyfar.Coordinates(
        phi.reshape(-1), theta.reshape(-1), rad,
        domain='sph', convention='top_colat',
        comment='gaussian spherical sampling grid',
        weights=weights, sh_order=n_max)

    return sampling


def sph_extremal(n_points=None, sh_order=None, radius=1.):
    """
    Return a Hyperinterpolation sampling grid.

    After Sloan and Womersley [#]_. The samplings are available for
    1 <= `sh_order` <= 200 (``n_points = (sh_order + 1)^2``).

    Parameters
    ----------
    n_points : int
        Number of sampling points in the grid. Related to the spherical
        harmonic order by ``n_points = (sh_order + 1)**2``. Either `n_points`
        or `sh_order` must be provided. The default is ``None``.
    sh_order : int
        Maximum applicable spherical harmonic order. Related to the number of
        points by ``sh_order = np.sqrt(n_points) - 1``. Either `n_points` or
        `sh_order` must be provided. The default is ``None``.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions including sampling weights.

    Notes
    -----
    This implementation uses precalculated sets of points from [#]_. The data
    up to ``sh_order = 99`` are loaded the first time this function is called.
    The remaining data is loaded upon request.

    References
    ----------
    .. [#]  I. H. Sloan and R. S. Womersley, “Extremal Systems of Points and
            Numerical Integration on the Sphere,” Advances in Computational
            Mathematics, vol. 21, no. 1/2, pp. 107–125, 2004.
    .. [#]  https://web.maths.unsw.edu.au/~rsw/Sphere/MaxDet/

    """

    if (n_points is None) and (sh_order is None):
        for o in range(1, 100):
            print(f"SH order {o}, number of points {(o + 1)**2}")
        return None

    # check input
    if n_points is not None and sh_order is not None:
        raise ValueError("Either n_points or sh_order must be None.")

    # get number of points or spherical harmonic order
    if sh_order is not None:
        if sh_order < 1 or sh_order > 200:
            raise ValueError('sh_order must be between 1 and 200')
        n_points = (sh_order + 1)**2
    else:
        if n_points not in [(n + 1)**2 for n in range(1, 200)]:
            raise ValueError('invalid value for n_points')
        sh_order = np.sqrt(n_points) - 1

    # download data if necessary
    filename = "samplings_extremal_md%03d.%05d" % (sh_order, n_points)
    filename = os.path.join(os.path.dirname(__file__), "external",  filename)
    if not os.path.exists(filename):
        if sh_order < 100:
            _sph_extremal_load_data('all')
        else:
            _sph_extremal_load_data(sh_order)

    # open data
    with open(filename, 'rb') as f:
        file_data = f.read()

    # format data
    file_data = file_data.decode()
    file_data = np.fromstring(
        file_data,
        dtype='double',
        sep=' ').reshape((int(n_points), 4))

    # normalize weights
    weights = file_data[:, 3] / 4 / np.pi

    # generate Coordinates object
    sampling = pyfar.Coordinates(
        file_data[:, 0] * radius,
        file_data[:, 1] * radius,
        file_data[:, 2] * radius,
        sh_order=sh_order, weights=weights,
        comment='extremal spherical sampling grid')

    return sampling


def sph_t_design(degree=None, sh_order=None, criterion='const_energy',
                 radius=1.):
    """
    Return spherical t-design sampling grid.

    For detailed information, see [#]_.
    For a spherical harmonic order :math:`n_{sh}`, a t-Design of degree
    :math:`t=2n_{sh}` for constant energy or :math:`t=2n_{sh}+1` additionally
    ensuring a constant angular spread of energy is required [#]_. For a given
    degree t

    .. math::

        L = \\lceil \\frac{(t+1)^2}{2} \\rceil+1,

    points will be generated, except for t = 3, 5, 7, 9, 11, 13, and 15.
    T-designs allow for an inverse spherical harmonic transform matrix
    calculated as :math:`D = \\frac{4\\pi}{L} \\mathbf{Y}^\\mathrm{H}` with
    :math:`\\mathbf{Y}^\\mathrm{H}` being the hermitian transpose of the
    spherical harmonics matrix.

    Parameters
    ----------
    degree : int
        T-design degree between ``1`` and ``180``. Either `degree` or
        `sh_order` must be provided. The default is ``None``.
    sh_order : int
        Maximum applicable spherical harmonic order. Related to the degree
        by ``degree = 2 * sh_order`` (``const_energy``) and
        ``degree = 2 * sh_order + 1`` (``const_angular_spread``). Either
        `degree` or `sh_order` must be provided. The default is ``None``.
    criterion : ``const_energy``, ``const_angular_spread``
        Design criterion ensuring only a constant energy or additionally
        constant angular spread of energy. The default is ``const_energy``.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Sampling weights can be obtained from
        :py:func:`calculate_sph_voronoi_weights`.

    Notes
    -----
    This function downloads a pre-calculated set of points from [#]_ . The data
    up to ``degree = 99`` are loaded the first time this function is called.
    The remaining data is loaded upon request.

    References
    ----------

    .. [#]  C. An, X. Chen, I. H. Sloan, and R. S. Womersley, “Well Conditioned
            Spherical Designs for Integration and Interpolation on the
            Two-Sphere,” SIAM Journal on Numerical Analysis, vol. 48, no. 6,
            pp. 2135–2157, Jan. 2010.
    .. [#]  F. Zotter, M. Frank, and A. Sontacchi, “The Virtual T-Design
            Ambisonics-Rig Using VBAP,” in Proceedings on the Congress on
            Sound and Vibration, 2010.
    .. [#]  http://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/sf.html

    """

    # check input
    if (degree is None) and (sh_order is None):
        print('Possible input values:')
        for d in range(1, 181):
            print(f"degree {d}, sh_order {int(d / 2)} ('const_energy'), \
                {int((d - 1) / 2)} ('const_angular_spread')")
        return None

    if criterion not in ['const_energy', 'const_angular_spread']:
        raise ValueError("Invalid design criterion. Must be 'const_energy' \
                         or 'const_angular_spread'.")

    if degree is not None and sh_order is not None:
        raise ValueError("Either n_points or sh_order must be None.")

    # get the degree
    if degree is None:
        if criterion == 'const_energy':
            degree = 2 * sh_order
        else:
            degree = 2 * sh_order + 1
    # get the SH order for the meta data entry in the Coordinates object
    else:
        if criterion == 'const_energy':
            sh_order = int(degree / 2)
        else:
            sh_order = int((degree - 1) / 2)

    if degree < 1 or degree > 180:
        raise ValueError('degree must be between 1 and 180.')

    # get the number of points
    n_points = int(np.ceil((degree + 1)**2 / 2) + 1)
    n_points_exceptions = {3: 8, 5: 18, 7: 32, 9: 50, 11: 72, 13: 98, 15: 128}
    if degree in n_points_exceptions:
        n_points = n_points_exceptions[degree]

    # download data if neccessary
    filename = "samplings_t_design_sf%03d.%05d" % (degree, n_points)
    filename = os.path.join(os.path.dirname(__file__), "external",  filename)
    if not os.path.exists(filename):
        if degree < 100:
            _sph_t_design_load_data('all')
        else:
            _sph_t_design_load_data(degree)

    # open data
    with open(filename, 'rb') as f:
        file_data = f.read()

    # format data
    file_data = file_data.decode()
    points = np.fromstring(
        file_data,
        dtype=np.double,
        sep=' ').reshape((n_points, 3))

    # generate Coordinates object
    sampling = pyfar.Coordinates(
        points[..., 0] * radius,
        points[..., 1] * radius,
        points[..., 2] * radius,
        sh_order=sh_order,
        comment='spherical T-design sampling grid')

    return sampling


def sph_equal_angle(delta_angles, radius=1.):
    """
    Generate sampling of the sphere with equally spaced angles.

    This sampling contain points at the North and South Pole. See
    :py:func:`sph_equiangular`, :py:func:`sph_gaussian`, and
    :py:func:`sph_great_circle` for samplings that do not contain points at the
    poles.


    Parameters
    ----------
    delta_angles : tuple, number
        Tuple that gives the angular spacing in azimuth and colatitude in
        degrees. If a number is provided, the same spacing is applied in both
        dimensions.
    radius : number, optional
        Radius of the sampling grid. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Sampling weights can be obtained from
        :py:func:`calculate_sph_voronoi_weights`.

    """

    # get the angles
    delta_angles = np.asarray(delta_angles)
    if delta_angles.size == 2:
        delta_phi = delta_angles[0]
        delta_theta = delta_angles[1]
    else:
        delta_phi = delta_angles
        delta_theta = delta_angles

    # check if the angles can be distributed
    eps = np.finfo('float').eps
    if not (np.abs(360 % delta_phi) < 2*eps):
        raise ValueError("delta_phi must be an integer divisor of 360")
    if not (np.abs(180 % delta_theta) < 2*eps):
        raise ValueError("delta_theta must be an integer divisor of 180")

    # get the angles
    phi_angles = np.arange(0, 360, delta_phi)
    theta_angles = np.arange(delta_theta, 180, delta_theta)

    # stack the angles
    phi = np.tile(phi_angles, theta_angles.size)
    theta = np.repeat(theta_angles, phi_angles.size)

    # add North and South Pole
    phi = np.concatenate(([0], phi, [0]))
    theta = np.concatenate(([0], theta, [180]))

    # make Coordinates object
    sampling = pyfar.Coordinates(
        phi, theta, radius,
        domain='sph', convention='top_colat', unit='deg',
        comment='equal angle spherical sampling grid')

    return sampling


def sph_great_circle(elevation=np.linspace(-90, 90, 19), gcd=10, radius=1,
                     azimuth_res=1, match=360):
    """
    Spherical sampling grid according to the great circle distance criterion.

    Sampling grid where neighboring points of the same elevation have approx.
    the same great circle distance across elevations [#]_.

    Parameters
    ----------
    elevation : array like, optional
        Contains the elevation from wich the sampling grid is generated, with
        :math:`-90^\\circ\\leq elevation \\leq 90^\\circ` (:math:`90^\\circ`:
        North Pole, :math:`-90^\\circ`: South Pole). The default is
        ``np.linspace(-90, 90, 19)``.
    gcd : number, optional
        Desired great circle distance (GCD). Note that the actual GCD of the
        sampling grid is equal or smaller then the desired GCD and that the GCD
        may vary across elevations. The default is ``10``.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.
    azimuth_res : number, optional
        Minimum resolution of the azimuth angle in degree. The default is
        ``1``.
    match : number, optional
        Forces azimuth entries to appear with a period of match degrees. E.g.,
        if ``match=90``, the grid includes the azimuth angles 0, 90, 180, and
        270 degrees. The default is ``360``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Sampling weights can be obtained from
        :py:func:`calculate_sph_voronoi_weights`.

    References
    ----------
    .. [#]  B. P. Bovbjerg, F. Christensen, P. Minnaar, and X. Chen, “Measuring
            the head-related transfer functions of an artificial head with a
            high directional resolution,” 109th AES Convention, Los Angeles,
            USA, Sep. 2000.

    """

    # check input
    assert 1 / azimuth_res % 1 == 0, "1/azimuth_res must be an integer."
    assert not 360 % match, "360/match must be an integer."
    assert match / azimuth_res % 1 == 0, "match/azimuth_res must be an \
                                         integer."

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
            while match % d_az[nn] > 1e-15:
                # round to precision of azimuth_res to avoid numerical errors
                d_az[nn] = np.round((d_az[nn] - azimuth_res)/azimuth_res) \
                           * azimuth_res

    # construct the full sampling grid
    azim = np.empty(0)
    elev = np.empty(0)
    for nn in range(elevation.size):
        azim = np.append(azim, np.arange(0, 360, d_az[nn]))
        elev = np.append(elev, np.full(int(360 / d_az[nn]), elevation[nn]))

    # round to precision of azimuth_res to avoid numerical errors
    azim = np.round(azim/azimuth_res) * azimuth_res

    # make Coordinates object
    sampling = pyfar.Coordinates(
        azim, elev, radius, 'sph', 'top_elev', 'deg',
        comment='spherical great circle sampling grid')

    return sampling


def sph_lebedev(n_points=None, sh_order=None, radius=1.):
    """
    Return Lebedev spherical sampling grid.

    For detailed information, see [#]_. For a list of available values
    for `n_points` and `sh_order` call :py:func:`sph_lebedev`.

    Parameters
    ----------
    n_points : int, optional
        Number of sampling points in the grid. Related to the spherical
        harmonic order by ``n_points = (sh_order + 1)**2``. Either `n_points`
        or `sh_order` must be provided. The default is ``None``.
    sh_order : int, optional
        Maximum applicable spherical harmonic order. Related to the number of
        points by ``sh_order = np.sqrt(n_points) - 1``. Either `n_points` or
        `sh_order` must be provided. The default is ``None``.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions including sampling weights.

    Notes
    -----
    This is a Python port of the Matlab Code written by Rob Parrish [#]_.

    References
    ----------
    .. [#] V.I. Lebedev, and D.N. Laikov
           "A quadrature formula for the sphere of the 131st
           algebraic order of accuracy"
           Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    .. [#] https://de.mathworks.com/matlabcentral/fileexchange/27097-\
        getlebedevsphere

    """

    # possible degrees
    degrees = np.array([6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230,
                        266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730,
                        2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294,
                        5810], dtype=int)

    # corresponding spherical harmonic orders
    orders = np.array((np.floor(np.sqrt(degrees / 1.3) - 1)), dtype=int)

    # list possible sh orders and degrees
    if n_points is None and sh_order is None:
        print('Possible input values:')
        for o, d in zip(orders, degrees):
            print(f"SH order {o}, number of points {d}")

        return None

    # check input
    if n_points is not None and sh_order is not None:
        raise ValueError("Either n_points or sh_order must be None.")

    # check if the order is available
    if sh_order is not None:
        if sh_order not in orders:
            str_orders = [f"{o}" for o in orders]
            raise ValueError("Invalid spherical harmonic order 'sh_order'. \
                             Valid orders are: {}.".format(
                             ', '.join(str_orders)))

        n_points = int(degrees[orders == sh_order])

    # check if n_points is available
    if n_points not in degrees:
        str_degrees = [f"{d}" for d in degrees]
        raise ValueError("Invalid number of points n_points. Valid degrees \
                         are: {}.".format(', '.join(str_degrees)))

    # calculate sh_order
    sh_order = int(orders[degrees == n_points])

    # get the samlpling
    leb = external.lebedev_sphere(n_points)

    # normalize the weights
    weights = leb["w"] / (4 * np.pi)

    # generate Coordinates object
    sampling = pyfar.Coordinates(
        leb["x"] * radius,
        leb["y"] * radius,
        leb["z"] * radius,
        sh_order=sh_order, weights=weights,
        comment='spherical Lebedev sampling grid')

    return sampling


def sph_fliege(n_points=None, sh_order=None, radius=1.):
    """
    Return Fliege-Maier spherical sampling grid.

    For detailed information, see [#]_. Call :py:func:`sph_fliege`
    for a list of possible values for `n_points` and `sh_order`.

    Parameters
    ----------
    n_points : int, optional
        Number of sampling points in the grid. Related to the spherical
        harmonic order by ``n_points = (sh_order + 1)**2``. Either `n_points`
        or `sh_order` must be provided. The default is ``None``.
    sh_order : int, optional
        Maximum applicable spherical harmonic order. Related to the number of
        points by ``sh_order = np.sqrt(n_points) - 1``. Either `n_points` or
        `sh_order` must be provided. The default is ``None``.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions including sampling weights.

    Notes
    -----
    This implementation uses pre-calculated points from the SOFiA
    toolbox [#]_. Possible combinations of `n_points` and `sh_order` are:

    +------------+------------+
    | `n_points` | `sh_order` |
    +============+============+
    | 4          | 1          |
    +------------+------------+
    | 9          | 2          |
    +------------+------------+
    | 16         | 3          |
    +------------+------------+
    | 25         | 4          |
    +------------+------------+
    | 36         | 5          |
    +------------+------------+
    | 49         | 6          |
    +------------+------------+
    | 64         | 7          |
    +------------+------------+
    | 81         | 8          |
    +------------+------------+
    | 100        | 9          |
    +------------+------------+
    | 121        | 10         |
    +------------+------------+
    | 144        | 11         |
    +------------+------------+
    | 169        | 12         |
    +------------+------------+
    | 196        | 13         |
    +------------+------------+
    | 225        | 14         |
    +------------+------------+
    | 256        | 15         |
    +------------+------------+
    | 289        | 16         |
    +------------+------------+
    | 324        | 17         |
    +------------+------------+
    | 361        | 18         |
    +------------+------------+
    | 400        | 19         |
    +------------+------------+
    | 441        | 20         |
    +------------+------------+
    | 484        | 21         |
    +------------+------------+
    | 529        | 22         |
    +------------+------------+
    | 576        | 23         |
    +------------+------------+
    | 625        | 24         |
    +------------+------------+
    | 676        | 25         |
    +------------+------------+
    | 729        | 26         |
    +------------+------------+
    | 784        | 27         |
    +------------+------------+
    | 841        | 28         |
    +------------+------------+
    | 900        | 29         |
    +------------+------------+

    References
    ----------
    .. [#] J. Fliege and U. Maier, "The distribution of points on the sphere
           and corresponding cubature formulae,” IMA J. Numerical Analysis,
           Vol. 19, pp. 317–334, Apr. 1999, doi: 10.1093/imanum/19.2.317.
    .. [#] https://audiogroup.web.th-koeln.de/SOFiA_wiki/DOWNLOAD.html

    """

    # possible values for n_points and sh_order
    points = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196,
                       225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625,
                       676, 729, 784, 841, 900], dtype=int)

    orders = np.array(np.floor(np.sqrt(points) - 1), dtype=int)

    # list possible sh orders and number of points
    if n_points is None and sh_order is None:
        for o, d in zip(orders, points):
            print(f"SH order {o}, number of points {d}")

        return None

    # check input
    if n_points is not None and sh_order is not None:
        raise ValueError("Either n_points or sh_order must be None.")

    if sh_order is not None:
        # check if the order is available
        if sh_order not in orders:
            str_orders = [f"{o}" for o in orders]
            raise ValueError("Invalid spherical harmonic order 'sh_order'. \
                              Valid orders are: {}.".format(
                              ', '.join(str_orders)))

        # assign n_points
        n_points = int(points[orders == sh_order])
    else:
        # check if n_points is available
        if n_points not in points:
            str_points = [f"{d}" for d in points]
            raise ValueError("Invalid number of points n_points. Valid points \
                            are: {}.".format(', '.join(str_points)))

        # assign sh_order
        sh_order = int(orders[points == n_points])

    # get the sampling points
    fliege = sio.loadmat(os.path.join(
        os.path.dirname(__file__), "external", "samplings_fliege.mat"),
        variable_names=f"Fliege_{int(n_points)}")
    fliege = fliege[f"Fliege_{int(n_points)}"]

    # generate Coordinates object
    sampling = pyfar.Coordinates(
        fliege[:, 0],
        fliege[:, 1],
        radius,
        domain='sph', convention='top_colat', unit='rad',
        sh_order=sh_order, weights=fliege[:, 2],
        comment='spherical Fliege sampling grid')

    # switch and invert coordinates in Cartesian representation to be
    # consistent with [1]
    xyz = sampling.get_cart(convention='right')
    sampling.set_cart(xyz[:, 1], xyz[:, 0], -xyz[:, 2])

    return sampling


def sph_equal_area(n_points, radius=1.):
    """
    Sampling based on partitioning into faces with equal area.

    For detailed information, see [#]_.

    Parameters
    ----------
    n_points : int
        Number of points corresponding to the number of partitions of the
        sphere.
    radius : number, optional
        Radius of the sampling grid in meters. The default is ``1``.

    Returns
    -------
    sampling : Coordinates
        Sampling positions. Sampling weights can be obtained from
        :py:func:`calculate_sph_voronoi_weights`.

    References
    ----------
    .. [#]  P. Leopardi, “A partition of the unit sphere into regions of equal
            area and small diameter,” Electronic Transactions on Numerical
            Analysis, vol. 25, no. 12, pp. 309–327, 2006.

    """

    point_set = external.eq_point_set(2, n_points)
    sampling = pyfar.Coordinates(
        point_set[0] * radius, point_set[1] * radius, point_set[2] * radius,
        domain='cart', convention='right',
        comment='Equal area partitioning of the sphere.')

    return sampling


def _sph_extremal_load_data(orders='all'):
    """Download extremal sampling grids.

    orders = 'all' : load all samplings up to SH order 99
    orders = int, list : load sampling of specified SH order(s)
    """

    # set the SH orders to be read
    if isinstance(orders, int):
        orders = [orders]
    elif isinstance(orders, str):
        orders = range(1, 100)
    elif not isinstance(orders, list):
        raise ValueError("orders must an int, list, or string.")

    print("Loading extremal sampling points from \
        https://web.maths.unsw.edu.au/~rsw/Sphere/MaxDet/. \
        This might take a while but is only done once.")

    http = urllib3.PoolManager(cert_reqs=False)
    prefix = 'samplings_extremal_'

    for sh_order in orders:
        # number of sampling points
        n_points = (sh_order + 1)**2

        # load the data
        filename = "md%03d.%05d" % (sh_order, n_points)
        url = "https://web.maths.unsw.edu.au/~rsw/Sphere/S2Pts/MD/"
        fileurl = url + filename

        http_data = http.urlopen('GET', fileurl)

        # save the data
        if http_data.status == 200:
            save_name = os.path.join(
                os.path.dirname(__file__), "external", prefix + filename)
            print(f'Loading file {sh_order}/{len(orders)}')
            with open(save_name, 'wb') as out:
                out.write(http_data.data)
        else:
            raise ConnectionError(
                "Connection error. Please check your internet connection.")


def _sph_t_design_load_data(degrees='all'):
    """Download t-design sampling grids.

    degrees = 'all' : load all samplings up to degree 99
    degrees = number : load sampling of specified degree
    """

    # set the degrees to be read
    if isinstance(degrees, int):
        degrees = [degrees]
    elif isinstance(degrees, str):
        degrees = range(1, 100)
    elif not isinstance(degrees, list):
        raise ValueError("degrees must an int, list, or string.")

    print("Loading t-design sampling points from \
        https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/sf.html. \
        This might take a while but is only done once.")

    http = urllib3.PoolManager(cert_reqs=False)
    prefix = 'samplings_t_design_'

    n_points_exceptions = {3: 8, 5: 18, 7: 32, 9: 50, 11: 72, 13: 98, 15: 128}

    for degree in degrees:
        # number of sampling points
        n_points = int(np.ceil((degree + 1)**2 / 2) + 1)
        if degree in n_points_exceptions:
            n_points = n_points_exceptions[degree]

        # load the data
        filename = "sf%03d.%05d" % (degree, n_points)
        url = "http://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/"\
              "SF29-Nov-2012/"
        fileurl = url + filename

        http_data = http.urlopen('GET', fileurl)

        # save the data
        if http_data.status == 200:
            save_name = os.path.join(
                os.path.dirname(__file__), "external", prefix + filename)
            print(f'Loading file {degree}/{len(degrees)}')
            with open(save_name, 'wb') as out:
                out.write(http_data.data)
        else:
            raise ConnectionError(
                "Connection error. Please check your internet connection.")
