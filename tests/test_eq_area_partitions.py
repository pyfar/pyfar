import numpy as np
import numpy.testing as npt

import pyfar.samplings.external.eq_area_partitions as eq


def test_caps_dim_2():
    reference_cap = np.array(
        [0.643501108793284, 1.570796326794896,
         2.498091544796509, 3.141592653589793])
    reference_regions = np.array([1, 4, 4, 1])
    dim = 2
    N = 10
    s_cap, n_regions = eq.caps(dim, N)

    npt.assert_almost_equal(reference_cap, s_cap)
    npt.assert_almost_equal(reference_regions, n_regions)


def test_caps_dim_1():
    reference_cap = np.array(
        [0.628318530717959, 1.256637061435917, 1.884955592153876,
         2.513274122871834, 3.141592653589793, 3.769911184307752,
         4.398229715025710, 5.026548245743669, 5.654866776461628,
         6.283185307179586])
    reference_regions = np.ones(10, dtype=int)
    dim = 1
    N = 10
    s_cap, n_regions = eq.caps(dim, N)

    npt.assert_almost_equal(reference_cap, s_cap)
    npt.assert_almost_equal(reference_regions, n_regions)


def test_polar_colat():
    dim = 2
    N = 10
    reference = 0.643501108793284
    c_polar = eq.polar_colat(dim, N)
    npt.assert_almost_equal(reference, c_polar)


def test_num_collars():
    N = 10
    c_polar = 0.643501108793284
    angle = 1.120998243279586
    reference = 2
    n_collars = eq.num_collars(N, c_polar, angle)
    npt.assert_equal(n_collars, reference)


def test_ideal_region_list():
    dim = 2
    N = 10
    c_polar = 0.643501108793284
    n_collars = 8
    reference = np.array(
        [1.0, 0.796262803771163, 0.967669218729048, 1.087303374762984,
         1.148764602736805, 1.148764602736807, 1.087303374762983,
         0.967669218729050, 0.796262803771161, 1.000000000000000])
    r_regions = eq.ideal_region_list(dim, N, c_polar, n_collars)
    npt.assert_almost_equal(r_regions, reference)

    n_collars = 2
    r_regions = eq.ideal_region_list(dim, N, c_polar, n_collars)
    reference = np.array(
        [1.0, 3.999999999999999, 4.000000000000001, 1.0])
    npt.assert_almost_equal(r_regions, reference)


def test_cap_colats():
    dim = 2
    N = 10
    c_polar = 0.643501108793284
    r_regions = np.array([1, 4, 4, 1])
    reference = np.array(
        [0.643501108793284, 1.570796326794896,
         2.498091544796509, 3.141592653589793])
    s_cap = eq.cap_colats(dim, N, c_polar, r_regions)
    npt.assert_almost_equal(s_cap, reference)


def test_circle_offset():
    offset = eq.circle_offset(4, 4)
    npt.assert_almost_equal(offset, 0.125)

    offset = eq.circle_offset(4, 1)
    npt.assert_almost_equal(offset, 0.5)


def test_eq_point_set_polar():
    reference = np.array(
        [[0, 0.785398163397448, 2.356194490192345, 3.926990816987241,
          5.497787143782138, 1.570796326794897, 3.141592653589793,
          4.712388980384690, 0, 0],
         [0, 1.107148717794090, 1.107148717794090, 1.107148717794090,
          1.107148717794090, 2.034443935795703, 2.034443935795703,
          2.034443935795703, 2.034443935795703, 3.141592653589793]])
    dim = 2
    N = 10
    points_polar = eq.point_set_polar(dim, N)
    npt.assert_almost_equal(points_polar, reference)


def test_eq_point_set():
    reference = np.array(
        [[0, 0.632455532033676, -0.632455532033676, -0.632455532033676,
          0.632455532033676, 0.000000000000000, -0.894427190999916,
          -0.000000000000000, 0.894427190999916, 0.000000000000000],
         [0, 0.632455532033676, 0.632455532033676, -0.632455532033676,
          -0.632455532033676, 0.894427190999916, 0.000000000000000,
          -0.89442719099991, 0, 0],
         [1.0, 0.447213595499958, 0.447213595499958, 0.447213595499958,
          0.447213595499958, -0.447213595499958, -0.447213595499958,
          -0.44721359549995, -0.447213595499958, -1.000000000000000]])
    N = 10
    dim = 2
    points = eq.point_set(dim, N)
    npt.assert_almost_equal(points, reference)


def test_area_of_sphere():
    dim = 2
    area = eq.area_of_sphere(dim)
    reference = 12.566370614359171
    npt.assert_almost_equal(area, reference)


def test_area_of_ideal_region():
    dim = 2
    N = 10
    reference = 1.256637061435917
    area = eq.area_of_ideal_region(dim, N)
    npt.assert_almost_equal(area, reference)


def test_ideal_collar_angle():
    dim = 2
    N = 10
    angle = eq.ideal_collar_angle(dim, N)
    reference = 1.120998243279586
    npt.assert_almost_equal(angle, reference)


def test_area_of_collar():
    dim = 2
    a_top = 0.200334842323120
    a_bot = 0.542950213441064
    reference = 0.777932859551470
    c_area = eq.area_of_collar(dim, a_top, a_bot)
    npt.assert_almost_equal(c_area, reference)


def test_area_of_cap():
    dim = 2
    s_cap = np.pi/2
    reference = 2*np.pi
    cap_area = eq.area_of_cap(dim, s_cap)
    npt.assert_almost_equal(cap_area, reference)

    s_cap = 0.542950213441064
    cap_area = eq.area_of_cap(dim, s_cap)
    reference = 0.903596565695061
    npt.assert_almost_equal(cap_area, reference)

    s_cap = 0.200334842323120
    reference = 0.125663706143592
    cap_area = eq.area_of_cap(dim, s_cap)
    npt.assert_almost_equal(cap_area, reference)


def test_polar2cart():
    points = np.array([[0, np.pi/2, np.pi*3/2, 0],
                       [0, np.pi/2, np.pi/2, np.pi]])

    reference = np.array([[0, 0, 0, 0],
                          [0, 1, -1, 0],
                          [1, 0, 0, -1]])

    points_cart = eq.polar2cart(points)
    npt.assert_almost_equal(points_cart, reference)
