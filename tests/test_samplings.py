import numpy as np
import numpy.testing as npt
from pytest import raises

import pyfar
from pyfar import Coordinates
import pyfar.samplings as samplings


def test_cart_equidistant_cube():
    # test with int
    c = samplings.cart_equidistant_cube(3)
    assert isinstance(c, Coordinates)
    assert c.csize == 3**3

    # test with tuple
    c = samplings.cart_equidistant_cube((3, 2, 4))
    assert c.csize == 3*2*4


def test_sph_dodecahedron():
    # test with default radius
    c = samplings.sph_dodecahedron()
    assert isinstance(c, Coordinates)
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test with user radius
    c = samplings.sph_dodecahedron(1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)


def test_sph_icosahedron():
    # test with default radius
    c = samplings.sph_icosahedron()
    assert isinstance(c, Coordinates)
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test with user radius
    c = samplings.sph_icosahedron(1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)


def test_sph_equiangular():
    # test without parameters
    with raises(ValueError):
        samplings.sph_equiangular()

    # test with single number of points
    c = samplings.sph_equiangular(5)
    isinstance(c, Coordinates)
    assert c.csize == 5**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with tuple
    c = samplings.sph_equiangular((3, 5))
    assert c.csize == 3*5
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.sph_equiangular(sh_order=5)
    assert c.csize == 4 * (5 + 1)**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test user radius
    c = samplings.sph_equiangular(5, radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)


def test_sph_gaussian():
    # test without parameters
    with raises(ValueError):
        samplings.sph_gaussian()

    # test with single number of points
    c = samplings.sph_gaussian(5)
    isinstance(c, Coordinates)
    assert c.csize == 5**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with tuple
    c = samplings.sph_gaussian((3, 5))
    assert c.csize == 3*5
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.sph_gaussian(sh_order=5)
    assert c.csize == 2 * (5 + 1)**2
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test user radius
    c = samplings.sph_gaussian(5, radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)


def test_sph_extremal():
    # load test data
    pyfar.samplings.samplings._sph_extremal_load_data(1)

    # test without parameters
    assert samplings.sph_extremal() is None

    # test with n_points
    c = samplings.sph_extremal(4)
    isinstance(c, Coordinates)
    assert c.csize == 4
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.sph_extremal(sh_order=1)
    assert c.csize == 4
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test user radius
    c = samplings.sph_extremal(4, radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)

    # test loading SH order > 99
    c = samplings.sph_extremal(sh_order=100)

    # test exceptions
    with raises(ValueError):
        c = samplings.sph_extremal(4, 1)
    with raises(ValueError):
        c = samplings.sph_extremal(5)
    with raises(ValueError):
        c = samplings.sph_extremal(sh_order=0)


def test_sph_t_design():
    # load test data
    pyfar.samplings.samplings._sph_t_design_load_data([1, 2, 3])

    # test without parameters
    assert samplings.sph_t_design() is None

    # test with degree
    c = samplings.sph_t_design(2)
    isinstance(c, Coordinates)
    assert c.csize == 6

    # test with spherical harmonic order
    c = samplings.sph_t_design(sh_order=1)
    assert c.csize == 6
    c = samplings.sph_t_design(sh_order=1, criterion='const_angular_spread')
    assert c.csize == 8

    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test user radius
    c = samplings.sph_t_design(2, radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)

    # test loading degree order > 99
    c = samplings.sph_t_design(100)

    # test exceptions
    with raises(ValueError):
        c = samplings.sph_t_design(4, 1)
    with raises(ValueError):
        c = samplings.sph_t_design(0)
    with raises(ValueError):
        c = samplings.sph_t_design(sh_order=0)
    with raises(ValueError):
        c = samplings.sph_t_design(2, criterion='const_thread')


def test_sph_equal_angle():
    # test with tuple
    c = samplings.sph_equal_angle((10, 20))
    assert isinstance(c, Coordinates)
    # test with number
    c = samplings.sph_equal_angle(10)
    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)
    # test user radius
    c = samplings.sph_equal_angle(10, 1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)

    # test assertions
    with raises(ValueError):
        c = samplings.sph_equal_angle((11, 20))
    with raises(ValueError):
        c = samplings.sph_equal_angle((20, 11))


def test_sph_great_circle():
    # test with default values
    c = samplings.sph_great_circle()
    assert isinstance(c, Coordinates)
    # check default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test if azimuth matching angles work
    c = samplings.sph_great_circle(0, 4, match=90)
    azimuth = c.get_sph(unit='deg')[:, 0]
    for deg in [0, 90, 180, 270]:
        assert deg in azimuth

    # test user radius
    c = samplings.sph_great_circle(radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)

    # test fractional azimuth resolution
    c = samplings.sph_great_circle(60, 4,  azimuth_res=.1, match=90)
    npt.assert_allclose(c.get_sph(unit='deg')[1, 0], 7.5, atol=1e-15)

    # test assertion: 1 / azimuth_res is not an integer
    with raises(AssertionError):
        samplings.sph_great_circle(azimuth_res=.6)
    # test assertion: 360 / match is not an integer
    with raises(AssertionError):
        samplings.sph_great_circle(match=270)
    # test assertion: match / azimuth_res is not an integer
    with raises(AssertionError):
        samplings.sph_great_circle(azimuth_res=.5, match=11.25)


def test_sph_lebedev():
    # test without parameters
    assert samplings.sph_lebedev() is None

    # test with degree
    c = samplings.sph_lebedev(14)
    isinstance(c, Coordinates)
    assert c.csize == 14
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.sph_lebedev(sh_order=3)
    assert c.csize == 26
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test user radius
    c = samplings.sph_lebedev(6, radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)


def test_sph_fliege():
    # test without parameters
    assert samplings.sph_fliege() is None

    # test with degree
    c = samplings.sph_fliege(16)
    isinstance(c, Coordinates)
    assert c.csize == 16
    npt.assert_allclose(np.sum(c.weights), 1)

    # test with spherical harmonic order
    c = samplings.sph_fliege(sh_order=3)
    assert c.csize == 16
    npt.assert_allclose(np.sum(c.weights), 1)

    # test default radius
    npt.assert_allclose(c.get_sph()[..., 2], 1, atol=1e-15)

    # test user radius
    c = samplings.sph_fliege(4, radius=1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)

    # test exceptions
    with raises(ValueError):
        c = samplings.sph_fliege(9, 2)
    with raises(ValueError):
        c = samplings.sph_fliege(30)


def test_sph_equal_area():
    # test with points only
    c = samplings.sph_equal_area(10)
    assert isinstance(c, Coordinates)
    assert c.csize == 10
    npt.assert_allclose(c.get_sph()[..., 2], 1., atol=1e-15)

    # test with user radius
    c = samplings.sph_equal_area(10, 1.5)
    npt.assert_allclose(c.get_sph()[..., 2], 1.5, atol=1e-15)
