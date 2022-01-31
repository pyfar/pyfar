# Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox.
# Release 1.10 2005-06-26
# COPYING
#
# For references, see AUTHORS.
# For revision history, see CHANGELOG.
#
# Copyright (c) 2004, 2005, University of New South Wales
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Python library partition the unit sphere into partition with equal area.
"""

from math import gcd
import numpy as np
import scipy.special as sps


def point_set(dimension, N):
    """Uniform distribution of points on the unit-sphere, by partitioning the
    unit-sphere into patches with equal area. This function is only a wrapper
    for point_set_polar and returns Cartesian coordinates instead of polar
    coordinates.

    Parameters
    ----------
    dimension : int
        The dimension, 2 represents is the 2-sphere and returns colatitude
        and azimuth angles.
    N : int
        The number of points to distribute.

    Returns
    -------
    points : array, double
        The resulting points in Cartesian coordinates. The dimensions of the
        array are (3, N)

    """
    dimension = np.asarray(dimension)
    N = np.asarray(N)
    return polar2cart(point_set_polar(dimension, N))


def point_set_polar(dimension, N):
    """Uniform distribution of points on the unit-sphere, by partitioning the
    unit-sphere into patches with equal area.

    Parameters
    ----------
    dimension : int
        The dimension, 2 represents is the 2-sphere and returns colatitude
        and azimuth angles.
    N : int
        The number of points to distribute.

    Returns
    -------
    points : array, double
        The resulting points in polar coordinates. The dimensions of the
        array are (dimension, N)

    """
    a_cap, n_regions = caps(dimension, N)

    if dimension == 1:
        points_s = a_cap - np.pi/N
    else:
        n_collars = np.size(n_regions) - 2

        points_s = np.zeros((dimension, N))
        point_n = 2

        offset = 0

        for collar_n in range(0, n_collars):
            # a_top is the colatitude of the top of the current collar.
            a_top = a_cap[collar_n]

            # a_bot is the colatitude of the bottom of the current collar.
            a_bot = a_cap[collar_n+1]

            # n_in_collar is the number of regions in the current collar.
            n_in_collar = n_regions[collar_n+1]

            points_l = point_set_polar(dimension - 1, n_in_collar)

            a_point = (a_top + a_bot)/2

            point_l_n = np.arange(0, np.size(points_l), dtype=int)
            # points_l = points_l[np.newaxis]

            if dimension == 2:
                points_s[0:dimension-1, point_n+point_l_n-1] = \
                    np.mod(points_l[point_l_n] + 2*np.pi*offset, 2*np.pi)

                offset += circle_offset(n_in_collar, n_regions[2+collar_n])
                offset -= np.floor(offset)
            else:
                points_s[0:dimension-2, point_n+point_l_n-1] = \
                    points_l[:, point_l_n]

            points_s[dimension-1, point_n+point_l_n-1] = a_point
            # point_n = point_n + points_l.shape[1]
            point_n += np.size(points_l)

        points_s[:, -1] = np.zeros(dimension)
        points_s[-1, -1] = np.pi

    return points_s


def caps(dimension, N):
    """Partition a sphere into to nested spherical caps

    Does the following:
    1)  partitions the unit sphere S^dim into a list of spherical caps of
        increasing colatitude and thus increasing area,
    2)  sets S_CAP to be an array of size (1 by N_COLLARS+2),
        containing increasing colatitudes of caps, and
    3)  sets N_REGIONS to be an array of size (1 by N_COLLARS+2),
        containing the intger number of regions in each corresponding zone of
        S^dim.

    Examples
    --------

    % > [s_cap,n_regions] = eq_caps(2,10)
    % s_cap =
    %     0.6435    1.5708    2.4981    3.1416
    % n_regions =
    %      1     4     4     1
    %
    % > [s_cap,n_regions] = eq_caps(3,6)
    % s_cap =
    %     0.9845    2.1571    3.1416
    % n_regions =
    %      1     4     1
    %

    Parameters
    ----------
    dimension : int
        The dimension, 2 represents is the 2-sphere and returns colatitude
        and azimuth angles.
    N : int
        The number of points to distribute.

    Returns
    -------

    """
    if dimension == 1:
        s_cap = np.linspace(2*np.pi/N, 2*np.pi, N)
        n_regions = np.ones(10, dtype=int)
    elif N == 1:
        s_cap = np.pi
        n_regions = 1
    else:
        c_polar = polar_colat(dimension, N)
        n_collars = num_collars(N, c_polar, ideal_collar_angle(dimension, N))
        r_regions = ideal_region_list(dimension, N, c_polar, n_collars)
        n_regions = round_to_naturals(N, r_regions)
        s_cap = cap_colats(dimension, N, c_polar, n_regions)

    return s_cap, n_regions


def polar_colat(dimension, N):
    """
    The colatitude of the North polar (top) spherical cap
    Given dim and N, determine the colatitude of the North polar spherical cap.

    Parameters
    ----------
    dimension : int
        The dimension
    N : int
        The number of partitions

    Returns
    -------
    colatitude : double
        The colatitude angle of the top cap.
    """
    # enough = N > 2
    # c_polar = np.empty()
    if N == 1:
        c_polar = np.pi
    elif N == 2:
        c_polar = np.pi/2
    else:
        ideal_region_area = area_of_ideal_region(dimension, N)
        c_polar = sradius_of_cap(dimension, ideal_region_area)

    return c_polar


def ideal_region_list(dimension, N, c_polar, n_collars):
    """The ideal real number of regions in each zone

    List the ideal real number of regions in each collar, plus the polar caps.
    Given dim, N, c_polar and n_collars, determine r_regions, a list of the
    ideal real number of regions in each collar, plus the polar caps.
    The number of elements is n_collars+2.
    r_regions[1] is 1.
    r_regions[n_collars+2] is 1.
    The sum of r_regions is N.

    Parameters
    ----------
    dimension : int
        The dimension
    N : int
        The number of points
    c_polar :
    n_collars : int
        The number of collar elements
    Returns
    -------
    ideal_regions : double
        The ideal number of regions in a collar

    """
    r_regions = np.zeros(2+n_collars)
    r_regions[0] = 1

    if n_collars > 0:
        a_fitting = (np.pi - 2*c_polar) / n_collars
        ideal_region_area = area_of_ideal_region(dimension, N)
        for collar_n in range(1, n_collars+1):
            ideal_collar_area = area_of_collar(
                dimension,
                c_polar + (collar_n - 1) * a_fitting,
                c_polar + collar_n * a_fitting)
            r_regions[collar_n] = ideal_collar_area / ideal_region_area

    r_regions[-1] = 1

    return r_regions


def round_to_naturals(N, r_regions):
    """Round off a given list of numbers of regions
    Given N and r_regions, determine n_regions, a list of the natural number
    of regions in each collar and the polar caps.
    This list is as close as possible to r_regions, using rounding.
    The number of elements is n_collars+2.
    n_regions[1] is 1.
    n_regions[n_collars+2] is 1.
    The sum of n_regions is N.

    Parameters
    ----------
    N : int
        The dimension
    r_regions : double
        The ideal number of regions per collar before rounding
    Returns
    -------
    n_regions : int
        The rounded integer number per collar.
    """
    r_regions = np.asarray(r_regions)
    n_regions = np.zeros(r_regions.shape, dtype=int)
    discrepancy = 0

    for zone_n in range(0, np.size(r_regions)):
        n_regions[zone_n] = np.rint(r_regions[zone_n] + discrepancy)
        discrepancy += (r_regions[zone_n] - n_regions[zone_n])

    return n_regions


def cap_colats(dimension, N, c_polar, n_regions):
    """Colatitudes of spherical caps enclosing cumulative sum of regions
    Given dim, N, c_polar and n_regions, determine c_caps, an increasing list
    of colatitudes of spherical caps which enclose the same area as that given
    by the cumulative sum of regions.
    The number of elements is n_collars+2.
    c_caps[1] is c_polar.
    c_caps[n_collars+1] is Pi-c_polar.
    c_caps[n_collars+2] is Pi.

    Parameters
    ----------
    dimension : int
        The dimension
    N : int
        Number of partitions
    c_polar : double
        Colatitude angles of the spherical caps
    n_regions: int
        Number of regions
    Returns
    -------
    c_caps : double
        Colatitude angles of the caps
    """
    c_caps = np.zeros(np.size(n_regions))
    c_caps[0] = c_polar
    ideal_region_area = area_of_ideal_region(dimension, N)
    n_collars = np.size(n_regions) - 2
    subtotal_n_regions = 1

    for collar_n in range(1, n_collars+1):
        subtotal_n_regions += n_regions[collar_n]
        c_caps[collar_n] = sradius_of_cap(
            dimension, subtotal_n_regions*ideal_region_area)

    c_caps[-1] = np.pi
    return c_caps


def num_collars(N, c_polar, a_ideal):
    """The number of collars between the polar caps
    Given N, an ideal angle, and c_polar, determine n_collars, the number of
    collars between the polar caps.

    Parameters
    ----------
    N : int
        The dimension
    c_polar : double
        The colatitude angle of the polar caps
    a_ideal : double
        The ideal collar angles.
    Returns
    -------
    n_collars : int
        The real number of collars.

    """
    if int((N > 2) and (a_ideal > 0)):
        n_collars = np.maximum(1, np.rint((np.pi - 2*c_polar) / a_ideal))
    else:
        n_collars = 0

    return int(n_collars)


def circle_offset(n_top, n_bot, extra_twist=False):
    """Try to maximize minimum distance of center points for S^2 collars
    Given n_top and n_bot, calculate an offset.
    The values n_top and n_bot represent the numbers of
    equally spaced points on two overlapping circles.
    The offset is given in multiples of whole rotations, and
    consists of three parts;
    1)  Half the difference between a twist of one sector on each of bottom
        and top. This brings the centre points into alignment.
    2)  A rotation which will maximize the minimum angle between
        points on the two circles.
    3)  An optional extra twist by a whole number of sectors on the second
        circle. The extra twist is added so that the location of the minimum
        angle between circles will progressively twist around the sphere
        with each collar.

    Parameters
    ----------
    n_top : int
        Number of points in the upper circle
    n_bot : int
        Number of points in the lower circle
    extra_twist : boolean
        Perform an additional rotation (see part 3)
    Returns
    -------
    offset : int
        The offset

    """
    offset = (1/n_bot - 1/n_top)/2 + gcd(n_top, n_bot) / (2*n_top*n_bot)
    if extra_twist:
        twist = 6
        offset += twist/n_bot

    return offset


def ideal_collar_angle(dimension, N):
    """Calculate the ideal angle for spherical collars.
    The ideal collar angle is determined by the side of a dim-dimensional
    hypercube of the same volume as the area of a single region of an N region
    equal area partition of S^dim.

    Since the EQ partition for N < 3 has no spherical collars,
    the recursive zonal equal area sphere partitioning algorithm does not use
    ideal_collar_angle(dim,N) for N < 3.

    dimension : int
        The dimension, 2 represents is the 2-sphere and returns colatitude
        and azimuth angles.
    N : int
        The number of points to distribute.

    Returns
    -------
    angle : double
        The ideal collar angle.

    """
    return area_of_ideal_region(dimension, N)**(1 / dimension)


def area_of_ideal_region(dimension, N):
    """The area a partition for an ideal partitioning of the sphere into
    N patches.

    Parameters
    ----------
    dimension : int
        The dimension
    N : int
        The number of partitions

    Returns
    -------
    area : double
        The area of an ideal partition.

    """
    return area_of_sphere(dimension)/N


def area_of_sphere(dimension):
    """Area of a sphere with the given dimension.

    Parameters
    ----------
    dimension : int

    Returns
    -------
    area : double
        The area.

    """
    power = (dimension + 1)/2
    return 2*np.pi**power / sps.gamma(power)


def area_of_collar(dimension, a_top, a_bot):
    """Area of spherical collar
    area_of_collar(dim, a_top, a_bot) calculates the area of
    an S^dim spherical collar specified by a_top, a_bot, where
    a_top is top (smaller) spherical radius,
    a_bot is bottom (larger) spherical radius.

    Parameters
    ----------
    dimension : int
        The dimension.
    a_top : array, double
        The top (smaller) spherical radius,
    a_bot : array, double
        The bottom (larger) spherical radius.

    Returns
    -------
    area : array, double
        The area of the spherical collar

    """
    return area_of_cap(dimension, a_bot) - area_of_cap(dimension, a_top)


def area_of_cap(dimension, s_cap):
    """Calculate the area of a spherical cap from the spherical radius s_cap.

    Parameters
    ----------
    dimension : int
        The dimension
    s_cap : double
        The spherical radius of the cap.

    Returns
    -------
    area : double
        The resulting area
    """
    if dimension == 1:
        area = 2 * s_cap
    elif dimension == 2:
        area = 4 * np.pi * np.sin(s_cap / 2)**2
    elif dimension == 3:
        raise ValueError("Dimension needs to be smaller or equal 2.")
    else:
        area = area_of_sphere(dimension) * sps.betainc(np.sin(s_cap/2)**2,
                                                       dimension/2,
                                                       dimension/2)

    return area


def sradius_of_cap(dimension, area):
    """Spherical radius of a spherical cap of a given area. It is assumed to be
    in the range [0, pi].
    The area is defined via the Lebesgue measure on S^dim inherited from
    its embedding in R^(dim+1).
    For dim <= 2, the spherical radius is calculated in closed form.

    Ref: [LeGS01 Lemma 4.1 p255].

    > s_cap=sradius_of_cap(2,area_of_sphere(2)/2)

    s_cap =
        1.5708

    s_cap=sradius_of_cap(3,(0:4)*area_of_sphere(3)/4)
    s_cap =
        [0, 1.1549, 1.5708, 1.9867, 3.1416]

    Parameters
    ----------
    dimension : int
        The dimension
    area : double
        The area of the spherical cap

    Returns
    -------
    s_rad : double
        The spherical radius

    Example
    -------
    >>> s_cap = sradius_of_cap(3,(0:4)*area_of_sphere(3)/4)
    >>> s_cap=sradius_of_cap(3,(0:4)*area_of_sphere(3)/4)

    """
    if dimension == 1:
        s_cap = area/2
    elif dimension == 2:
        s_cap = 2*np.arcsin(np.sqrt(area / np.pi) / 2)
    else:
        raise ValueError("Dimensions larger than 2 are not supported.")
    return np.asarray(s_cap)


def polar2cart(points_polar):
    """Comnversion from the polar angles theta and phi to Cartesian coordinates

        x = cos(phi) * sin(theta)
        y = sin(phi) * sin(theta)
        x = cos(theta)

    Parameters
    ----------
    points_polar : array, double
        The points in polar coordinates, with shape (2, N)

    Returns:
    points_cart : array, double
        The points in Cartesian coordinates with shape (3, N)

    """
    points_cart = np.zeros((points_polar.shape[0]+1, points_polar.shape[1]))
    points_cart[0, :] = np.cos(points_polar[0, :]) * np.sin(points_polar[1, :])
    points_cart[1, :] = np.sin(points_polar[0, :]) * np.sin(points_polar[1, :])
    points_cart[2, :] = np.cos(points_polar[1, :])
    return points_cart
