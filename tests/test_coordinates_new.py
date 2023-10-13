import numpy as np
import numpy.testing as npt
import pytest

from pyfar import Coordinates
from pyfar.classes.coordinates import (sph2cart, cart2sph, cyl2cart)
from pyfar import (deg2rad, rad2deg)


def test___eq___copy():
    coordinates = Coordinates(1, 2, 3, comment="Madre mia!")
    actual = coordinates.copy()
    assert coordinates == actual


@pytest.mark.parametrize(
    'x, y, z, radius, radius_z', [
        (1, 0, 0, 1, 1),
        (-1, 0, 0, 1, 1),
        (0, 2, 0, 2, 2),
        (0, 3, 4, 5, 3),
        (0, 0, 0, 0, 0),
    ])
def test_getter_radii_from_cart(x, y, z, radius, radius_z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.radius, radius, atol=1e-15)
    np.testing.assert_allclose(coords.rho, radius_z, atol=1e-15)
    np.testing.assert_allclose(coords.radius, radius, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z, azimuth, elevation', [
        (1, 0, 0, 0, 0),
        (-1, 0, 0, np.pi, 0),
        (0, 1, 0, np.pi/2, 0),
        (0, -1, 0, 3*np.pi/2, 0),
        (0, 0, 1, 0, np.pi/2),
        (0, 0, -1, 0, -np.pi/2),
    ])
def test_getter_sph_top_from_cart(x, y, z, azimuth, elevation):
    coords = Coordinates(x, y, z)
    colatitude = np.pi/2 - elevation
    np.testing.assert_allclose(coords.azimuth, azimuth, atol=1e-15)
    np.testing.assert_allclose(coords.elevation, elevation, atol=1e-15)
    np.testing.assert_allclose(coords.colatitude, colatitude, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_elevation,
        np.atleast_2d([azimuth, elevation, 1]), atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_colatitude,
        np.atleast_2d([azimuth, colatitude, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.azimuth = azimuth
    coords.elevation = elevation
    coords.radius = 1
    colatitude = np.pi/2 - elevation
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z, frontal, upper', [
        (0, 1, 0, 0, np.pi/2),
        (0, -1, 0, np.pi, np.pi/2),
        (0, 0, 1, np.pi/2, np.pi/2),
        (0, 0, -1, 3*np.pi/2, np.pi/2),
        (1, 0, 0, 0, 0),
        (-1, 0, 0, 0, np.pi),
    ])
def test_getter_sph_front_from_cart(x, y, z, frontal, upper):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.frontal, frontal, atol=1e-15)
    np.testing.assert_allclose(coords.upper, upper, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_front, np.atleast_2d([frontal, upper, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.frontal = frontal
    coords.upper = upper
    coords.radius = 1
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z, lateral, polar', [
        (0, 1, 0, np.pi/2, 0),
        (0, -1, 0, -np.pi/2, 0),
        (0, 0, 1, 0, np.pi/2),
        (0, 0, -1, 0, -np.pi/2),
        (1, 0, 0, 0, 0),
        (-1, 0, 0, 0, np.pi),
    ])
def test_getter_sph_side_from_cart(x, y, z, lateral, polar):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.lateral, lateral, atol=1e-15)
    np.testing.assert_allclose(coords.polar, polar, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_side, np.atleast_2d([lateral, polar, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.lateral = lateral
    coords.polar = polar
    coords.radius = 1
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)


@pytest.mark.parametrize(
    'x, actual', [
        (0, np.array([0])),
        (np.ones((1,)), np.ones((1,))),
        (np.ones((3,)), np.ones((3,))),
        (np.ones((1, 2)), np.ones((1, 2))),
        (np.ones((1, 1)), np.ones((1, 1))),
        (np.ones((2, 1)), np.ones((2, 1))),
        (np.ones((3, 2, 1)), np.ones((3, 2, 1))),
        (np.ones((1, 1, 1)), np.ones((1, 1, 1))),
        (np.ones((1, 2, 3)), np.ones((1, 2, 3))),
    ])
def test_coordinates_squeeze(x, actual):
    coords = Coordinates(x, 0, 1)
    np.testing.assert_allclose(coords.x, actual, atol=1e-15)
    np.testing.assert_allclose(coords.x.shape, actual.shape, atol=1e-15)
    np.testing.assert_allclose(coords.y.shape, actual.shape, atol=1e-15)
    np.testing.assert_allclose(coords.z.shape, actual.shape, atol=1e-15)
    np.testing.assert_allclose(coords.y, 0, atol=1e-15)
    np.testing.assert_allclose(coords.z, 1, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (0, 1, 0),
        (0, -1, 0),
        (0., 0, 1),
        (0, .0, -1),
        (1, 0, 0),
        (-1, 0, 0),
        (np.ones((2, 3, 1)), np.zeros((2, 3, 1)), np.ones((2, 3, 1))),
    ])
def test_cart_setter_same_size(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(
            coords.cartesian.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian.shape[-1], 3, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 0], coords.x, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 1], coords.y, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 2], coords.z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (np.ones((2, 3, 1)), 10, -1),
        (np.ones((2,)), 2, 1),
    ])
def test_cart_setter_different_size(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(
            coords.cartesian.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian.shape[-1], 3, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 0], coords.x, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 1], coords.y, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian[..., 2], coords.z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (np.ones((3, 1)), 7, 3),
        (np.ones((1, 2)), 5, 1),
        (np.ones((1, 1)), 5, 1),
    ])
def test_cart_setter_different_size_with_flatten(x, y, z):
    coords = Coordinates(x, y, z)
    shape = x.shape
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, np.ones(shape)*y, atol=1e-15)
    np.testing.assert_allclose(coords.z, np.ones(shape)*z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(
            coords.cartesian.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cartesian.shape[-1], 3, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (0, 1, 0),
        (0, -1, 0),
        (0., 0, 1),
        (0, .0, -1),
        (1, 0, 0),
        (-1, 0, 0),
        (np.ones((2, 3, 1)), 10, -1),
        (np.ones((2,)), 2, 1),
        (np.ones((2, 3, 1)), np.zeros((2, 3, 1)), np.ones((2, 3, 1))),
    ])
def test__array__getter(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(
        np.array(coords)[..., 0], x, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 2], z, atol=1e-15)


@pytest.mark.parametrize(
    'x, y, z', [
        (np.ones((3, 1)), 7, 3),
        (np.ones((1, 2)), 5, 1),
        (np.ones((1, 1)), 5, 1),
    ])
def test__array__getter_with_flatten(x, y, z):
    coords = Coordinates.from_cartesian(x, y, z)
    np.testing.assert_allclose(
        np.array(coords)[..., 0], x, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(
        np.array(coords)[..., 2], z, atol=1e-15)


def test__getitem__():
    """Test getitem with different parameters."""
    # test without weights
    coords = Coordinates([1, 2], 0, 0)
    new = coords[0]
    assert isinstance(new, Coordinates)
    np.testing.assert_allclose(new.x, 1)
    np.testing.assert_allclose(new.y, 0)
    np.testing.assert_allclose(new.z, 0)


def test__getitem__weights():
    # test with weights
    coords = Coordinates([1, 2], 0, 0, weights=[.1, .9])
    new = coords[0]
    assert isinstance(new, Coordinates)
    np.testing.assert_allclose(new.x, 1)
    np.testing.assert_allclose(new.y, 0)
    np.testing.assert_allclose(new.z, 0)
    assert new.weights == np.array(.1)


def test__getitem__3D_array():
    # test with 3D array
    coords = Coordinates([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], 0, 0)
    new = coords[0:1]
    assert isinstance(new, Coordinates)
    assert new.cshape == (1, 5)


def test__getitem__untouched():
    # test if sliced object stays untouched
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    new = coords[0]
    new.x = 2
    assert coords.cshape == (2,)
    np.testing.assert_allclose(coords.x[0], 0)
    np.testing.assert_allclose(coords.y[0], 0)
    np.testing.assert_allclose(coords.z[0], 0)


def test__repr__comment():
    coords = Coordinates([0, 1], [0, 1], [0, 1], comment="Madre Mia!")
    x = coords.__repr__()
    assert 'Madre Mia!' in x


def test_find_slice_cart():
    """Test different queries for find slice."""
    # test only for self.cdim = 1.
    # self.find_slice uses KDTree, which is tested with N-dimensional arrays
    # in test_find_nearest_k()
    d = np.linspace(-2, 2, 5)

    c = Coordinates(d, 0, 0)
    index, mask = c.find_slice('x', 'met', 0, 1)
    np.testing.assert_allclose(index[0], np.array([1, 2, 3]))
    np.testing.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    c = Coordinates(0, d, 0)
    index, mask = c.find_slice('y', 'met', 0, 1)
    np.testing.assert_allclose(index[0], np.array([1, 2, 3]))
    np.testing.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))

    c = Coordinates(0, 0, d)
    index, mask = c.find_slice('z', 'met', 0, 1)
    np.testing.assert_allclose(index[0], np.array([1, 2, 3]))
    np.testing.assert_allclose(mask, np.array([0, 1, 1, 1, 0]))


@pytest.mark.parametrize(
    'coordinate, unit, value, tol, des_index, des_mask', [
        ('azimuth', 'deg', 0, 1, np.array([1, 2, 3]),
            np.array([0, 1, 1, 1, 0])),
        ('azimuth', 'deg', 359, 2, np.array([0, 1, 2, 3]),
            np.array([1, 1, 1, 1, 0])),
        ('azimuth', 'deg', 1, 1, np.array([2, 3, 4]),
            np.array([0, 0, 1, 1, 1])),
    ])
def test_find_slice_sph(coordinate, unit, value, tol, des_index, des_mask):
    """Test different queries for find slice."""
    # spherical grid
    d = np.array([358, 359, 0, 1, 2]) * np.pi / 180
    c = Coordinates.from_spherical_elevation(d, 0, 1)

    index, mask = c.find_slice(coordinate, unit, value, tol)
    np.testing.assert_allclose(index[0], des_index)
    np.testing.assert_allclose(mask, des_mask)


def test_find_slice_error():
    d = np.array([358, 359, 0, 1, 2]) * np.pi / 180
    c = Coordinates.from_spherical_elevation(d, 0, 1)
    # out of range query
    # with pytest.raises(AssertionError):
    #     c.find_slice('azimuth', 'deg', -1, 1)
    # non existing coordinate query
    with pytest.raises(ValueError, match="does not exist"):
        c.find_slice('elevation', 'ged', 1, 1)
    with pytest.raises(ValueError, match="does not exist"):
        c.find_slice('Ola', 'red', 1, 1)


@pytest.mark.parametrize(
    'coordinate, min, max', [
        ('azimuth', 0, 2*np.pi),
        ('polar', -np.pi/2, 3*np.pi/2),
        ('frontal', 0, 2*np.pi),
    ])
def test_angle_limits_cyclic(coordinate, min, max):
    """Test different queries for find slice."""
    # spherical grid
    d = np.arange(-4*np.pi, 4*np.pi, np.pi/4)
    c = Coordinates(d, 0, 1)
    c.__setattr__(coordinate, d)
    attr = c.__getattribute__(coordinate)
    desired = (d - min) % (max - min) + min
    np.testing.assert_allclose(attr, desired, atol=2e-14)


@pytest.mark.parametrize(
    'coordinate, min, max', [
        ('azimuth', 0, 2*np.pi),
        ('polar', -np.pi/2, 3*np.pi/2),
        ('frontal', 0, 2*np.pi),
        ('radius', 0, np.inf),
        ('rho', 0, np.inf),
    ])
def test_angle_cyclic_limits(coordinate, min, max):
    """Test different queries for find slice."""
    # spherical grid
    d = np.arange(-4*np.pi, 4*np.pi, np.pi/4)
    c = Coordinates(d, 0, 1)
    c.__setattr__(coordinate, d)
    attr = c.__getattribute__(coordinate)
    assert all(attr <= max)
    assert all(attr >= min)


@pytest.mark.parametrize(
    'coordinate, min, max', [
        ('colatitude', 0, np.pi),
        ('upper', 0, np.pi),
        ('elevation', -np.pi/2, np.pi/2),
        ('lateral', -np.pi/2, np.pi/2),
        ('radius', 0, np.inf),
        ('rho', 0, np.inf),
    ])
@pytest.mark.parametrize(
    'eps_min', [
        (0),
        (np.finfo(float).eps),
    ])
@pytest.mark.parametrize(
    'eps_max', [
        (0),
        (np.finfo(float).eps),
    ])
def test_angle_limits_rounded_by_2eps(coordinate, min, max, eps_min, eps_max):
    """Test different queries for find slice."""
    # spherical grid
    if max == np.inf:
        d = np.arange(min, np.pi/4, 4*np.pi)
    else:
        d = np.arange(min, np.pi/4, max)
    d[0] -= eps_min
    d[-1] += eps_max
    c = Coordinates(d, 0, 1)
    c.__setattr__(coordinate, d)
    attr = c.__getattribute__(coordinate)
    assert all(attr <= max)
    assert all(attr >= min)


def test__repr__dim():
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    x = coords.__repr__()
    assert '1D' in x
    assert '2' in x
    assert '(2,)' in x


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
def test_coordinates_init_from_cartesian(x, y, z):
    coords = Coordinates.from_cartesian(x, y, z)
    npt.assert_allclose(coords._x, x)
    npt.assert_allclose(coords._y, y)
    npt.assert_allclose(coords._z, z)


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_coordinates_init_from_cartesian_with(x, y, z, weights, comment):
    coords = Coordinates.from_cartesian(x, y, z, weights, comment)
    npt.assert_allclose(coords._x, x)
    npt.assert_allclose(coords._y, y)
    npt.assert_allclose(coords._z, z)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
def test_coordinates_init_from_spherical_colatitude(x, y, z):
    upper, frontal, rad = cart2sph(x, y, z)
    coords = Coordinates.from_spherical_colatitude(upper, frontal, rad)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('x', [0, 1, -1.])
@pytest.mark.parametrize('y', [0, 1, -1.])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_coordinates_init_from_spherical_colatitude_with(
        x, y, z, weights, comment):
    upper, frontal, rad = cart2sph(x, y, z)
    coords = Coordinates.from_spherical_colatitude(
        upper, frontal, rad, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('elevation', [0, np.pi/2, -np.pi/2])
@pytest.mark.parametrize('radius', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_coordinates_init_from_spherical_elevation_with(
        azimuth, elevation, radius, weights, comment):
    coords = Coordinates.from_spherical_elevation(
        azimuth, elevation, radius, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('lateral', [0, np.pi/2, -np.pi/2])
@pytest.mark.parametrize('polar', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_coordinates_init_from_spherical_side_with(
        lateral, polar, radius, weights, comment):
    coords = Coordinates.from_spherical_side(
        lateral, polar, radius, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('frontal', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('upper', [0, np.pi/2, np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_coordinates_init_from_spherical_front_with(
        frontal, upper, radius, weights, comment):
    coords = Coordinates.from_spherical_front(
        frontal, upper, radius, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    y, z, x = sph2cart(frontal, upper, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('rho', [0, 1, -1.])
@pytest.mark.parametrize('weights', [1])
@pytest.mark.parametrize('comment', ['0'])
def test_coordinates_init_from_cylindrical_with(
        azimuth, z, rho, weights, comment):
    coords = Coordinates.from_cylindrical(
        azimuth, z, rho, weights, comment)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = cyl2cart(azimuth, z, rho)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)
    coords.comment == comment
    coords.weights == weights


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('elevation', [0, np.pi/2, -np.pi/2])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_coordinates_init_from_spherical_elevation(azimuth, elevation, radius):
    coords = Coordinates.from_spherical_elevation(azimuth, elevation, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('lateral', [0, np.pi/2, -np.pi/2])
@pytest.mark.parametrize('polar', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_coordinates_init_from_spherical_side(lateral, polar, radius):
    coords = Coordinates.from_spherical_side(lateral, polar, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('frontal', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('upper', [0, np.pi/2, np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_coordinates_init_from_spherical_front(frontal, upper, radius):
    coords = Coordinates.from_spherical_front(frontal, upper, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    y, z, x = sph2cart(frontal, upper, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi, 3*np.pi])
@pytest.mark.parametrize('z', [0, 1, -1.])
@pytest.mark.parametrize('radius_z', [0, 1, -1.])
def test_coordinates_init_from_cylindrical(azimuth, z, radius_z):
    coords = Coordinates.from_cylindrical(azimuth, z, radius_z)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = cyl2cart(azimuth, z, radius_z)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


def test_angle_conversion_wrong_input():
    '''Test input checks when converting from deg to rad and vice versa'''

    # test input checks (common functionality, needs to be tested for
    # only one function)
    # 1. input data has the wrong shape
    with pytest.raises(ValueError, match='coordinates must be of shape'):
        deg2rad(np.ones((2, 4)))
    # 2. invalid domain
    with pytest.raises(ValueError, match='domain must be'):
        deg2rad(np.ones((2, 3)), 'wrong')


@pytest.mark.parametrize('rad,deg', (
        # flat array
        [np.array([0, np.pi, 1]), np.array([0, 180, 1])],
        # 2D array
        [np.array([[0, np.pi, 1], [np.pi, 0, 2]]),
         np.array([[0, 180, 1], [180, 0, 2]])],
        # list
        [[0, np.pi, 1], np.array([0, 180, 1])]))
def test_angle_conversion_rad2deg_spherical(rad, deg):
    '''Test angle conversion from rad to deg and spherical coordinates'''

    # copy input
    rad_copy = rad.copy()
    # convert
    deg_actual = rad2deg(rad)
    # check that input did not change
    npt.assert_equal(rad_copy, rad)
    # check output values
    npt.assert_allclose(deg_actual, np.atleast_2d(deg))
    # check type and shape
    assert isinstance(deg_actual, np.ndarray)
    assert deg_actual.ndim >= 2
    assert deg_actual.shape[-1] == 3


def test_angle_conversion_rad2deg_cylindrical():
    '''
    Test angle conversion from rad to deg and cylindrical coordinates. Only
    the result is checked. Everything else is tested above.
    '''

    # copy input
    rad = np.array([np.pi, 1, 2])
    # convert
    deg = rad2deg(rad, 'cylindrical')
    # check output values
    npt.assert_allclose(deg, np.atleast_2d([180, 1, 2]))


def test_angle_conversion_deg2rad():
    '''
    Test angle conversion from rad to deg. Only spherical coordinates are used
    and the result is checked. Everything else is tested above.
    '''

    # copy input
    deg = np.array([180, 360, 1])
    # convert
    rad = deg2rad(deg)
    # check output values
    npt.assert_allclose(rad, np.atleast_2d([np.pi, 2*np.pi, 1]))
