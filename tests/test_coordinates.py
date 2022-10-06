import numpy as np
# import numpy.testing as npt
import pytest
# from pytest import raises
# import matplotlib.pyplot as plt

from pyfar import Coordinates
# import pyfar.classes.coordinates as coordinates


def test_coordinates_init():
    """Test initialization of empty coordinates object."""
    coords = Coordinates()
    assert isinstance(coords, Coordinates)


def test_coordinates_init_val():
    """Test initializing Coordinates with values of different type and size."""

    # test input: scalar
    c1 = 1
    # test input: 2 element vectors
    c2 = [1, 2]                        # list
    c3 = np.asarray(c2)                # flat np.array
    # test input: 3 element vector
    c6 = [1, 2, 3]
    # test input: 2D matrix
    c7 = np.array([[1, 2, 3], [1, 2, 3]])
    # test input: 3D matrix
    c8 = np.array([[[1, 2, 3], [1, 2, 3]],
                   [[1, 2, 3], [1, 2, 3]]])

    # tests that have to path
    # input scalar coordinate
    Coordinates(c1, c1, c1)
    # input list of coordinates
    Coordinates(c2, c2, c2)
    # input scalar and lists
    Coordinates(c1, c2, c2)
    # input flat np.arrays
    Coordinates(c3, c3, c3)
    # input non flat vectors
    # Coordinates(c3, c4, c5)
    # input 2D data
    Coordinates(c1, c1, c7)
    # input 3D data
    Coordinates(c1, c1, c8)

    # tests that have to fail
    with pytest.raises(AssertionError):
        Coordinates(c2, c2, c6)
    with pytest.raises(AssertionError):
        Coordinates(c6, c6, c7)
    with pytest.raises(AssertionError):
        Coordinates(c2, c2, c8)


def test_coordinates_init_val_and_comment():
    """Test initialization with comment."""
    coords = Coordinates(1, 1, 1, comment='try this')
    assert isinstance(coords, Coordinates)
    assert coords.comment == 'try this'


def test_coordinates_init_val_and_weights():
    """Test initialization with weights."""
    # correct number of weights
    coords = Coordinates([1, 2], 0, 0, weights=[.5, .5])
    assert isinstance(coords, Coordinates)
    np.testing.assert_allclose(coords.weights, [.5, .5])

    # incorrect number of weights
    with pytest.raises(AssertionError):
        Coordinates([1, 2], 0, 0, weights=.5)


def test_setter_weights():
    """Test setting weights."""
    coords = Coordinates([1, 2], 0, 0)
    coords.weights = [.5, .5]
    assert (coords.weights == np.array([.5, .5])).all()


def test_setter_comment():
    """Test setting the comment."""
    coords = Coordinates()
    coords.comment = 'now this'
    assert coords.comment == 'now this'


def test_assertion_for_getter():
    """Test assertion for empty Coordinates objects"""
    coords = Coordinates()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.get_cart()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.get_sph()
    with pytest.raises(ValueError, match="Object is empty"):
        coords.get_cyl()


@pytest.mark.parametrize(
    'x, y, z, actual, expected', [
        (1, 1, 1,                Coordinates(),                 False),
        (1, 1, 1,                Coordinates(1, 1, -1),                 False),
        (1, 1, 1,                Coordinates(1, 1, 1),                 True),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1], [1, 1], [1, 2]),   False),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1.0], [1, 1.0], [1, 1]), True)
    ])
def test___eq___differInPoints(x, y, z, actual, expected):
    """This function checks against 3 different pairings of Coordinates."""
    coordinates = Coordinates(x, y, z)
    comparison = coordinates == actual
    assert comparison == expected


def test___eq___differInWeigths_notEqual():
    coordinates = Coordinates(1, 2, 3, weights=.5)
    actual = Coordinates(1, 2, 3, weights=0.0)
    assert not coordinates == actual


def test___eq___differInShComment_notEqual():
    coordinates = Coordinates(1, 2, 3, comment="Madre mia!")
    actual = Coordinates(1, 2, 3, comment="Oh my woooooosh!")
    assert not coordinates == actual


def test___eq___copy():
    coordinates = Coordinates(1, 2, 3, comment="Madre mia!")
    actual = coordinates.copy()
    assert coordinates == actual


def test___eq___ForwardAndBackwardsDomainTransform_Equal():
    coordinates = Coordinates(1, 2, 3)
    actual = coordinates.copy()
    actual.get_sph()
    actual.get_cart()
    assert coordinates == actual


def test_cshape():
    """Test the cshape attribute."""
    # empty
    coords = Coordinates()
    assert coords.cshape == (0,)
    # 2D points
    coords = Coordinates([1, 0], [1, 1], [0, 1])
    assert coords.cshape == (2,)
    # 3D points
    coords = Coordinates([[1, 2, 3], [4, 5, 6]], 1, 1)
    assert coords.cshape == (2, 3)


def test_cdim():
    """Test the csim attribute."""
    # empty
    coords = Coordinates()
    assert coords.cdim == 0
    # 2D points
    coords = Coordinates([1, 0], 1, 1)
    assert coords.cdim == 1
    # 3D points
    coords = Coordinates([[1, 2, 3], [4, 5, 6]], 1, 1)
    assert coords.cdim == 2


def test_csize():
    """Test the csize attribute."""
    # 0 points
    coords = Coordinates()
    assert coords.csize == 0
    # two points
    coords = Coordinates([1, 0], 1, 1)
    assert coords.csize == 2
    # 6 points in two dimensions
    coords = Coordinates([[1, 2, 3], [4, 5, 6]], 1, 1)
    assert coords.csize == 6


@pytest.mark.parametrize("rot_type,rot", [
    ('quat', [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]),
    ('matrix',  [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ('rotvec', [0, 0, 90]),
    ('z', 90)])
def test_rotation(rot_type, rot):
    """Test rotation with different formats."""
    c = Coordinates(1, 0, 0)
    c.rotate(rot_type, rot)
    # np.testing.assert_allclose(
    #     c.get_cart(), np.atleast_2d([0, 1, 0]), atol=1e-15)
    np.testing.assert_allclose(c.x, 0, atol=1e-15)
    np.testing.assert_allclose(c.y, 1, atol=1e-15)
    np.testing.assert_allclose(c.z, 0, atol=1e-15)


def test_rotation_assertion():
    """Test rotation with unknown rotation type."""
    c = Coordinates(1, 0, 0)
    # test with unknown type
    with pytest.raises(ValueError):
        c.rotate('urgh', 90)


def test_inverse_rotation():
    """Test the inverse rotation."""
    x = np.ones((2, 4, 1))
    y = np.zeros((2, 4, 1))
    z = np.zeros((2, 4, 1))
    c = Coordinates(x.copy(), y.copy(), z.copy())
    c.rotate('z', 90)
    c.rotate('z', 90, inverse=True)
    np.testing.assert_allclose(c.x, x, atol=1e-15)
    np.testing.assert_allclose(c.y, y, atol=1e-15)
    np.testing.assert_allclose(c.z, z, atol=1e-15)


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
    np.testing.assert_allclose(coords.radius_z, radius_z, atol=1e-15)
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
        (np.ones((2, 3, 1)), 10, -1),
        (np.ones((3, 1)), 7, 3),
        (np.ones((1, 2)), 5, 1),
        (np.ones((2,)), 2, 1),
        (np.ones((2, 3, 1)), np.zeros((2, 3, 1)), np.ones((2, 3, 1))),
    ])
def test_xyz_setter(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(coords.xyz.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.xyz.shape[-1], 3, atol=1e-15)
    np.testing.assert_allclose(coords.xyz[..., 0], x, atol=1e-15)
    np.testing.assert_allclose(coords.xyz[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(coords.xyz[..., 2], z, atol=1e-15)
