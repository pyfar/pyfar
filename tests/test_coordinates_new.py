import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises

from pyfar import Coordinates
from pyfar.classes.coordinates import sph2cart, cart2sph, cyl2cart


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
    'x, y, z, phi, theta', [
        (0, 1, 0, 0, np.pi/2),
        (0, -1, 0, np.pi, np.pi/2),
        (0, 0, 1, np.pi/2, np.pi/2),
        (0, 0, -1, 3*np.pi/2, np.pi/2),
        (1, 0, 0, 0, 0),
        (-1, 0, 0, 0, np.pi),
    ])
def test_getter_sph_front_from_cart(x, y, z, phi, theta):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.phi, phi, atol=1e-15)
    np.testing.assert_allclose(coords.theta, theta, atol=1e-15)
    np.testing.assert_allclose(
        coords.spherical_front, np.atleast_2d([phi, theta, 1]), atol=1e-15)
    coords = Coordinates(0, 5, 0)
    coords.phi = phi
    coords.theta = theta
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
    shape = x.flatten().shape
    np.testing.assert_allclose(coords.x, x.flatten(), atol=1e-15)
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
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(
        np.array(coords)[..., 0], x.flatten(), atol=1e-15)
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


def test__getitem__untouced():
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
    d = np.array([358, 359, 0, 1, 2])
    c = Coordinates(d, 0, 1, 'sph', 'top_elev', 'deg')

    index, mask = c.find_slice(coordinate, unit, value, tol)
    np.testing.assert_allclose(index[0], des_index)
    np.testing.assert_allclose(mask, des_mask)


def test_find_slice_error():
    d = np.array([358, 359, 0, 1, 2])
    c = Coordinates(d, 0, 1, 'sph', 'top_elev', 'deg')
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
        ('phi', 0, 2*np.pi),
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
        ('phi', 0, 2*np.pi),
        ('colatitude', 0, np.pi),
        ('theta', 0, np.pi),
        ('elevation', -np.pi/2, np.pi/2),
        ('lateral', -np.pi/2, np.pi/2),
        ('radius', 0, np.inf),
        ('rho', 0, np.inf),
    ])
def test_angle_limits(coordinate, min, max):
    """Test different queries for find slice."""
    # spherical grid
    d = np.arange(-4*np.pi, 4*np.pi, np.pi/4)
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


@pytest.mark.parametrize(
    'coords', [
        (Coordinates(np.linspace(0, 1, 11), 0, 5)),
        (Coordinates(np.arange(10), 0, 5)),
    ])
def test_find_nearest_points_distance_1d(coords):
    for index in range(coords.csize):
        find = coords[index]
        d, i, m = coords.find_nearest_points(find, 1)
        npt.assert_array_almost_equal(d, 0)
        npt.assert_array_almost_equal(find.cartesian, coords[m].cartesian)
        npt.assert_array_almost_equal(coords[i].cartesian, find.cartesian)


@pytest.mark.parametrize(
    'coords', [
        (Coordinates(np.arange(9).reshape(3, 3), 0, 1)),
        (Coordinates(np.arange(8).reshape(2, 4), 5, 1))
    ])
def test_find_nearest_points_distance_2d(coords):
    for i in range(coords.cshape[0]):
        for j in range(coords.cshape[1]):
            find = coords[i, j]
            d, idx, m = coords.find_nearest_points(find, 1)
            npt.assert_array_almost_equal(d, 0)
            # assert find == coords[idx]
            npt.assert_array_almost_equal(find.cartesian, coords[m].cartesian)
            npt.assert_array_almost_equal(
                find.cartesian, coords[idx].cartesian)


# @pytest.mark.parametrize(
#     'azimuth', [
#         np.arange(0, 60, 10),
#         np.arange(0, 60, 10).reshape((2, 3)),
#         np.arange(0, 60, 10).reshape((3, 2))
#     ])
# @pytest.mark.parametrize(
#     'find_azimuth', [
#         20,
#         30,
#         np.array([20, 30]),
#         [[20, 20], [20, 20]]
#     ])
# def test_find_nearest_angular(azimuth, find_azimuth):
#     """Tests returns of find_nearest_sph."""
#     # test only 1D case since most of the code from self.find_nearest_k is
# used
#     coords = Coordinates.from_spherical_elevation(azimuth/180*np.pi, 0, 1)
#     find = Coordinates.from_spherical_elevation(find_azimuth/180*np.pi, 0, 1)
#     i = coords.find_nearest_angular(find, 5)
#     assert find == coords[i]


@pytest.mark.parametrize(
    'azimuth', [
        np.arange(0, 60, 10),
        np.arange(0, 60, 10).reshape((2, 3)),
        np.arange(0, 60, 10).reshape((3, 2))
    ])
@pytest.mark.parametrize(
    'find_azimuth', [0, 10, 20, 30, 40, 50])
def test_find_nearest_angular(azimuth, find_azimuth):
    # test normal condition with multiple inputs
    coords = Coordinates.from_spherical_elevation(azimuth/180*np.pi, 0, 1)
    find = Coordinates.from_spherical_elevation(find_azimuth/180*np.pi, 0, 1)
    i = coords.find_nearest_angular(find, 5)
    npt.assert_array_almost_equal(
        find.spherical_colatitude, coords[i].spherical_colatitude)


@pytest.mark.parametrize(
    'x', [
        np.arange(0, 60, 10),
        np.arange(0, 60, 10).reshape((2, 3)),
        np.arange(0, 60, 10).reshape((3, 2))
    ])
@pytest.mark.parametrize('find_x', [0, 10, 20, 30, 40, 50])
def test_find_nearest_euclidean(x, find_x):
    # test normal condition with multiple inputs
    coords = Coordinates(x, 0, 1)
    find = Coordinates(find_x, 0, 1)
    i = coords.find_nearest_euclidean(find, 5)
    npt.assert_array_almost_equal(find.cartesian, coords[i].cartesian)
    i = coords.find_nearest_euclidean(find, 500)
    actual = coords[i].cartesian
    coords = Coordinates(x.flatten(), 0, 1)
    npt.assert_array_almost_equal(coords.cartesian, actual)


@pytest.mark.parametrize(
    'azimuth', [
        np.arange(0, 60, 10),
        np.arange(0, 60, 10).reshape((2, 3)),
        np.arange(0, 60, 10).reshape((3, 2))
    ])
@pytest.mark.parametrize(
    'find_azimuth', [0, 10, 20, 30, 40, 50])
def test_find_nearest_angular_tol_radius(azimuth, find_azimuth):
    # test tolerance radius
    radius = np.ones(azimuth.shape)
    radius[..., -1] = 5
    coords = Coordinates.from_spherical_elevation(azimuth/180*np.pi, 0, radius)
    find = Coordinates.from_spherical_elevation(find_azimuth/180*np.pi, 0, 1)
    coords_copy = coords.copy()
    i = coords.find_nearest_angular(find, 5, atol_radius=5)
    npt.assert_array_almost_equal(find.theta, coords[i].theta)
    npt.assert_array_almost_equal(find.phi, coords[i].phi)
    assert coords_copy == coords
    with raises(ValueError, match='if all points have the same'):
        coords.find_nearest_angular(find, 5, atol_radius=1)


def test_find_nearest_angular_error():
    coords = Coordinates.from_spherical_elevation(
        np.arange(0, 60, 10)/180*np.pi, 0, 1)
    find_error = Coordinates.from_spherical_elevation(
        np.arange(0, 20, 10)/180*np.pi, 0, 1)
    find = Coordinates.from_spherical_elevation(
        0/180*np.pi, 0, 1)

    with raises(ValueError, match='only works for one input.'):
        coords.find_nearest_angular(find_error, 5)
    with raises(ValueError, match='distance must be >= 0 and <= 180.'):
        coords.find_nearest_angular(find, -5)
    with raises(
            ValueError,
            match='absolute radius tolerance \'atol_radius\' must be >= 0.'):
        coords.find_nearest_angular(find, 5, atol_radius=-5)
    with raises(ValueError, match='absolute tolerance \'atol\' must be >= 0.'):
        coords.find_nearest_angular(find, 5, atol=-5)


def test_find_nearest_by_distance_angular_error():
    az = np.linspace(0, 40, 5)
    coords = Coordinates(az, 0, 1, 'sph', 'top_elev', 'deg')
    # test out of range parameters
    with raises(AssertionError):
        find = Coordinates(1, 0, 0)
        coords.find_nearest_by_distance(find, -1, 'angular')
        # coords.find_nearest_sph(1, 0, 0, -1)
    with raises(AssertionError):
        find = Coordinates(1, 0, 0)
        coords.find_nearest_by_distance(find, 181, 'angular')
        # coords.find_nearest_sph(1, 0, 0, 181)

    # test assertion for multiple radii
    coords = Coordinates([1, 2], 0, 0)
    with raises(ValueError, match="find_nearest_sph only works if"):
        find = Coordinates(0, 0, 1)
        coords.find_nearest_by_distance(find, 1, 'angular')
        # coords.find_nearest_sph(0, 0, 1, 1)


def test_find_nearest_by_distance_direct():
    """Tests returns of find_nearest_cart."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(2.5, 0, 0)
    i, m = coords.find_nearest_by_distance(find, 1.5)
    npt.assert_allclose(i, np.array([[1, 2, 3, 4]]))
    npt.assert_allclose(m, np.array([0, 1, 1, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_by_distance(find, .1)
    assert len(i) == 1
    assert i[0].size == 0
    npt.assert_allclose(m, np.array([0, 0, 0, 0, 0, 0]))


def test_find_nearest_by_distance_direct_error():
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(2.5, 0, 0)
    # test out of range parameters
    with raises(AssertionError):
        coords.find_nearest_by_distance(find, -1)


def test_find_nearest_points():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # 1D cartesian, nearest point
    d, i, m = coords.find_nearest_points(find)
    assert i[0] == 1
    npt.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))

    # 1D spherical, nearest point
    find = Coordinates(0, 0, 1, 'sph', 'top_elev', 'deg')
    d, i, m = coords.find_nearest_points(find, 1)
    assert i[0] == 1
    npt.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))


def test_find_nearest_2_points():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # 1D cartesian, two nearest points
    find = Coordinates(1.2, 0, 0)
    d, i, m = coords.find_nearest_points(find, 2)
    npt.assert_allclose(i, np.array([[1, 2]]))
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))


def test_find_nearest_points_2():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # 1D cartesian query two points
    find = Coordinates([1, 2], 0, 0)
    d, i, m = coords.find_nearest_points(find)
    npt.assert_allclose(i, [[1, 2]])
    npt.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))


def test_find_nearest_points_2d():
    """Test returns of find_nearest_k"""
    x = np.arange(6)
    # 2D cartesian, nearest point
    coords = Coordinates(x.reshape(2, 3), 0, 0)
    find = Coordinates(1, 0, 0)
    d, i, m = coords.find_nearest_points(find)
    assert i[0] == 0
    assert i[1] == 1
    npt.assert_allclose(m, np.array([[0, 1, 0], [0, 0, 0]]))
    npt.assert_almost_equal(find.cartesian, coords[i].cartesian)
    npt.assert_almost_equal(find.cartesian, coords[m].cartesian)


def test_find_nearest_points_errors():
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    find = Coordinates(1, 0, 0)
    # test out of range parameters
    with raises(AssertionError, match='number of points must be'):
        coords.find_nearest_points(find, -1)


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
def test_coordinates_init_from_spherical_colatitude(x, y, z):
    theta, phi, rad = cart2sph(x, y, z)
    coords = Coordinates.from_spherical_colatitude(theta, phi, rad)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi])
@pytest.mark.parametrize('elevation', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_coordinates_init_from_spherical_elevation(azimuth, elevation, radius):
    coords = Coordinates.from_spherical_elevation(azimuth, elevation, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, y, z = sph2cart(azimuth, np.pi / 2 - elevation, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('lateral', [0, np.pi, -np.pi])
@pytest.mark.parametrize('polar', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_coordinates_init_from_spherical_side(lateral, polar, radius):
    coords = Coordinates.from_spherical_side(lateral, polar, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    x, z, y = sph2cart(polar, np.pi / 2 - lateral, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('phi', [0, np.pi, -np.pi])
@pytest.mark.parametrize('theta', [0, np.pi, -np.pi])
@pytest.mark.parametrize('radius', [0, 1, -1.])
def test_coordinates_init_from_spherical_front(phi, theta, radius):
    coords = Coordinates.from_spherical_front(phi, theta, radius)
    # use atol here because of numerical rounding issues introduced in
    # the coordinate conversion
    y, z, x = sph2cart(phi, theta, radius)
    npt.assert_allclose(coords._x, x, atol=1e-15)
    npt.assert_allclose(coords._y, y, atol=1e-15)
    npt.assert_allclose(coords._z, z, atol=1e-15)


@pytest.mark.parametrize('azimuth', [0, np.pi, -np.pi])
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
