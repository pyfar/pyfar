import numpy as np
import pytest
from pyfar import Coordinates
from packaging import version
import pyfar as pf
import matplotlib.pyplot as plt


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
    with pytest.raises(ValueError, match="Object is empty"):
        coords.x
    with pytest.raises(ValueError, match="Object is empty"):
        coords.y
    with pytest.raises(ValueError, match="Object is empty"):
        coords.z
    with pytest.raises(ValueError, match="Object is empty"):
        coords.azimuth
    with pytest.raises(ValueError, match="Object is empty"):
        coords.elevation
    with pytest.raises(ValueError, match="Object is empty"):
        coords.colatitude
    with pytest.raises(ValueError, match="Object is empty"):
        coords.radius
    with pytest.raises(ValueError, match="Object is empty"):
        coords.radius_z
    with pytest.raises(ValueError, match="Object is empty"):
        coords.phi
    with pytest.raises(ValueError, match="Object is empty"):
        coords.theta
    with pytest.raises(ValueError, match="Object is empty"):
        coords.lateral
    with pytest.raises(ValueError, match="Object is empty"):
        coords.polar


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


def test___eq___differInShOrder_notEqual():
    coordinates = Coordinates(1, 2, 3, sh_order=2)
    actual = Coordinates(1, 2, 3, sh_order=8)
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
    np.testing.assert_allclose(
        coords.sph_top_elev,
        np.atleast_2d([azimuth, elevation, 1]), atol=1e-15)
    np.testing.assert_allclose(
        coords.sph_top_colat,
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
        coords.sph_front, np.atleast_2d([phi, theta, 1]), atol=1e-15)
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
        coords.sph_side, np.atleast_2d([lateral, polar, 1]), atol=1e-15)
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
def test_cart_setter(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(coords.x, x, atol=1e-15)
    np.testing.assert_allclose(coords.y, y, atol=1e-15)
    np.testing.assert_allclose(coords.z, z, atol=1e-15)
    if x is np.array:
        np.testing.assert_allclose(coords.cart.shape[:-1], x.shape, atol=1e-15)
    np.testing.assert_allclose(coords.cart.shape[-1], 3, atol=1e-15)
    np.testing.assert_allclose(coords.cart[..., 0], x, atol=1e-15)
    np.testing.assert_allclose(coords.cart[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(coords.cart[..., 2], z, atol=1e-15)


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
def test__array__getter(x, y, z):
    coords = Coordinates(x, y, z)
    np.testing.assert_allclose(np.array(coords)[..., 0], x, atol=1e-15)
    np.testing.assert_allclose(np.array(coords)[..., 1], y, atol=1e-15)
    np.testing.assert_allclose(np.array(coords)[..., 2], z, atol=1e-15)


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
    new.set_cart(2, 2, 2)
    assert coords.cshape == (2,)
    np.testing.assert_allclose(coords.x[0], 0)
    np.testing.assert_allclose(coords.y[0], 0)
    np.testing.assert_allclose(coords.z[0], 0)


def test__repr__dim():
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    x = coords.__repr__()
    assert '1D' in x
    assert '2' in x
    assert '(2,)' in x


def test__repr__comment():
    coords = Coordinates([0, 1], [0, 1], [0, 1], comment="Madre Mia!")
    x = coords.__repr__()
    assert 'Madre Mia!' in x


def test_find_nearest_sph():
    """Tests returns of find_nearest_sph."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    az = np.linspace(0, 40, 5)
    coords = Coordinates(az, 0, 1, 'sph', 'top_elev', 'deg')
    i, m = coords.find_nearest_sph(25, 0, 1, 5, 'sph', 'top_elev', 'deg')
    np.testing.assert_allclose(i, np.array([2, 3]))
    np.testing.assert_allclose(m, np.array([0, 0, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_sph(25, 0, 1, 1, 'sph', 'top_elev', 'deg')
    assert len(i) == 0
    np.testing.assert_allclose(m, np.array([0, 0, 0, 0, 0]))

    # test out of range parameters
    with pytest.raises(AssertionError):
        coords.find_nearest_sph(1, 0, 0, -1)
    with pytest.raises(AssertionError):
        coords.find_nearest_sph(1, 0, 0, 181)

    # test assertion for multiple radii
    coords = Coordinates([1, 2], 0, 0)
    with pytest.raises(ValueError, match="find_nearest_sph only works if"):
        coords.find_nearest_sph(0, 0, 1, 1)


def test_get_nearest_deprecations():
    coords = Coordinates(np.arange(6), 0, 0)

    # nearest_k
    with pytest.warns(PendingDeprecationWarning,
                      match="This function will be deprecated"):
        coords.get_nearest_k(1, 0, 0)

    if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_nearest_k() from pyfar 0.5.0!
            coords.get_nearest_k(1, 0, 0)

    # nearest_cart
    with pytest.warns(PendingDeprecationWarning,
                      match="This function will be deprecated"):
        coords.get_nearest_cart(2.5, 0, 0, 1.5)

    if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_nearest_k() from pyfar 0.5.0!
            coords.get_nearest_cart(2.5, 0, 0, 1.5)

    # nearest_sph
    coords = Coordinates([1, 0, -1, 0], [0, 1, 0, -1], 0)
    with pytest.warns(PendingDeprecationWarning,
                      match="This function will be deprecated"):
        coords.get_nearest_sph(0, 0, 1, 1)

    if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_nearest_k() from pyfar 0.5.0!
            coords.get_nearest_sph(0, 0, 1, 1)

    # slice
    with pytest.warns(PendingDeprecationWarning,
                      match="This function will be deprecated"):
        coords.get_slice('x', 'met', 0, 1)

    if version.parse(pf.__version__) >= version.parse('0.5.0'):
        with pytest.raises(AttributeError):
            # remove get_slice() from pyfar 0.5.0!
            coords.get_slice('x', 'met', 0, 1)


@pytest.mark.parametrize(
    'statement', [
        ('coords.get_cart()'),
        ('coords.sh_order'),
        ('coords.set_cart(1,1,1)'),
        ('coords.get_cyl()'),
        ('coords.set_cyl(1,1,1)'),
        ('coords.get_sph()'),
        ('coords.set_sph(1,1,1)'),
        ('Coordinates(0, 0, 0, sh_order=1)'),
    ])
def test_get_nearest_deprecations_0_6_0(statement):
    coords = Coordinates(np.arange(6), 0, 0)
    coords.y = 1

    # sh_order getter
    with pytest.warns(PendingDeprecationWarning,
                      match="This function will be deprecated"):
        eval(statement)

    # remove statement from pyfar 0.6.0!
    if version.parse(pf.__version__) >= version.parse('0.6.0'):
        with pytest.raises(AttributeError):
            eval(statement)


def test_get_nearest_deprecations_0_6_0_set_sh_order():
    coords = Coordinates(np.arange(6), 0, 0)

    # sh_order getter
    with pytest.warns(PendingDeprecationWarning,
                      match="This function will be deprecated"):
        coords.sh_order = 1

    # remove statement from pyfar 0.6.0!
    if version.parse(pf.__version__) >= version.parse('0.6.0'):
        with pytest.raises(AttributeError):
            coords.sh_order = 1


def test_find_nearest_cart():
    """Tests returns of find_nearest_cart."""
    # test only 1D case since most of the code from self.find_nearest_k is used
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    i, m = coords.find_nearest_cart(2.5, 0, 0, 1.5)
    np.testing.assert_allclose(i, np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(m, np.array([0, 1, 1, 1, 1, 0]))

    # test search with empty results
    i, m = coords.find_nearest_cart(2.5, 0, 0, .1)
    assert len(i) == 0
    np.testing.assert_allclose(m, np.array([0, 0, 0, 0, 0, 0]))

    # test out of range parameters
    with pytest.raises(AssertionError):
        coords.find_nearest_cart(1, 0, 0, -1)


def test_find_nearest_k():
    """Test returns of find_nearest_k"""
    # 1D cartesian, nearest point
    x = np.arange(6)
    coords = Coordinates(x, 0, 0)
    i, m = coords.find_nearest_k(1, 0, 0)
    assert i == 1
    np.testing.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))

    # 1D spherical, nearest point
    i, m = coords.find_nearest_k(0, 0, 1, 1, 'sph', 'top_elev', 'deg')
    assert i == 1
    np.testing.assert_allclose(m, np.array([0, 1, 0, 0, 0, 0]))

    # 1D cartesian, two nearest points
    i, m = coords.find_nearest_k(1.2, 0, 0, 2)
    np.testing.assert_allclose(i, np.array([1, 2]))
    np.testing.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))

    # 1D cartesian query two points
    i, m = coords.find_nearest_k([1, 2], 0, 0)
    np.testing.assert_allclose(i, [1, 2])
    np.testing.assert_allclose(m, np.array([0, 1, 1, 0, 0, 0]))

    # 2D cartesian, nearest point
    coords = Coordinates(x.reshape(2, 3), 0, 0)
    i, m = coords.find_nearest_k(1, 0, 0)
    assert i == 1
    np.testing.assert_allclose(m, np.array([[0, 1, 0], [0, 0, 0]]))

    # test with plot
    coords = Coordinates(x, 0, 0)
    coords.find_nearest_k(1, 0, 0, show=True)

    # test object with a single point
    coords = Coordinates(1, 0, 0)
    coords.find_nearest_k(1, 0, 0, show=True)

    # test out of range parameters
    with pytest.raises(AssertionError):
        coords.find_nearest_k(1, 0, 0, -1)

    # plt.close("all")


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


def test_show():
    """Test if possible calls of show() pass."""
    coords = Coordinates([-1, 0, 1], 0, 0)
    # show without mask
    coords.show()
    # show with mask as list
    coords.show([1, 0, 1])
    # show with mask as ndarray
    coords.show(np.array([1, 0, 1], dtype=bool))
    # test assertion
    with pytest.raises(AssertionError):
        coords.show(np.array([1, 0], dtype=bool))

    plt.close("all")


def test_getter_with_degrees():
    """Test if getter return correct values also in degrees"""
    coords = Coordinates(0, 1, 0)

    sph = coords.get_sph(unit="deg")
    np.testing.assert_allclose(sph, np.atleast_2d(np.array([90, 90, 1])))

    cyl = coords.get_cyl(unit="deg")
    np.testing.assert_allclose(cyl, np.atleast_2d(np.array([90, 0, 1])))


def test_setter_sh_order():
    """Test setting the SH order."""
    coords = Coordinates()
    coords.sh_order = 10
    assert coords.sh_order == 10


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
        ('radius_z', 0, np.inf),
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
