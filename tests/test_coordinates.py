import numpy as np
import numpy.testing as npt
import pytest
import matplotlib.pyplot as plt

from pyfar import Coordinates
import pyfar.classes.coordinates as coordinates


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
    # input 2D data
    Coordinates(c1, c1, c7)
    # input 3D data
    Coordinates(c1, c1, c8)
    # input (3,) and (2, 3) data
    Coordinates(c6, c6, c7)

    # tests that have to fail
    with pytest.raises(ValueError, match="shape mismatch"):
        Coordinates(c2, c2, c6)
    with pytest.raises(ValueError, match="shape mismatch"):
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
    npt.assert_allclose(coords.weights, [.5, .5])

    # incorrect number of weights
    with pytest.raises(AssertionError):
        Coordinates([1, 2], 0, 0, weights=.5)


def test_show():
    """Test if possible calls of show() pass."""
    coords = Coordinates([-1, 0, 1], 0, 0)
    # show without mask
    coords.show()
    # show with mask as list
    coords.show([1, 0, 1])
    # show with index as list
    coords.show([0, 1])
    # show with mask as ndarray
    coords.show(np.array([1, 0, 1], dtype=bool))
    # show with index as ndarray
    coords.show(np.array([0, 1], dtype=int))

    plt.close("all")


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
    """Test the cdim attribute."""
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


def test_getitem():
    """Test getitem with different parameters."""
    # test without weights
    coords = Coordinates([1, 2], 0, 0)
    new = coords[0]
    assert isinstance(new, Coordinates)
    npt.assert_allclose(new.cartesian, np.atleast_2d([1, 0, 0]))

    # test with weights
    coords = Coordinates([1, 2], 0, 0, weights=[.1, .9])
    new = coords[0]
    assert isinstance(new, Coordinates)
    npt.assert_allclose(new.cartesian, np.atleast_2d([1, 0, 0]))
    assert new.weights == np.array(.1)

    # test with 3D array
    coords = Coordinates([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], 0, 0)
    new = coords[0:1]
    assert isinstance(new, Coordinates)
    assert new.cshape == (1, 5)

    # test if sliced object stays untouched
    coords = Coordinates([0, 1], [0, 1], [0, 1])
    new = coords[0]
    new.x = 2
    assert coords.cshape == (2,)
    npt.assert_allclose(coords.cartesian[0], np.array([0, 0, 0]))



@pytest.mark.parametrize(("rot_type", "rot"), [
    ('quat', [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]),
    ('matrix',  [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ('rotvec', [0, 0, 90]),
    ('z', 90)])
def test_rotation(rot_type, rot):
    """Test rotation with different formats."""
    c = Coordinates(1, 0, 0)
    c.rotate(rot_type, rot)
    npt.assert_allclose(c.cartesian, np.atleast_2d([0, 1, 0]), atol=1e-15)


def test_rotation_assertion():
    """Test rotation with unknown rotation type."""
    c = Coordinates(1, 0, 0)
    # test with unknown type
    match = "rotation must be 'quat', 'matrix', 'rotvec', or"
    with pytest.raises(ValueError, match=match):
        c.rotate('urgh', 90)


def test_inverse_rotation():
    """Test the inverse rotation."""
    xyz = np.concatenate((np.ones((2, 4, 1)),
                          np.zeros((2, 4, 1)),
                          np.zeros((2, 4, 1))), -1)
    c = Coordinates(xyz[..., 0].copy(), xyz[..., 1].copy(), xyz[..., 2].copy())
    c.rotate('z', 90)
    c.rotate('z', 90, inverse=True)
    npt.assert_allclose(c.cartesian, xyz, atol=1e-15)


def test_converters():
    """
    Test if converters can handle numbers (correctness of the conversion is
    tested in test_setter_and_getter_with_conversion).
    """
    coordinates.cart2sph(0, 0, 1)
    coordinates.sph2cart(0, 0, 1)
    coordinates.cart2cyl(0, 0, 1)
    coordinates.cyl2cart(0, 0, 1)


@pytest.mark.parametrize(
    ("points_1", "points_2", "points_3", "actual", "expected"), [
        (1, 1, 1,                Coordinates(1, 1, -1),                 False),
        ([1, 1], [1, 1], [1, 1], Coordinates([1, 1], [1, 1], [1, 2]),   False),
        (
            [1, 1], [1, 1], [1, 1],
            Coordinates([1, 1.0], [1, 1.0], [1, 1]),
            True),
    ])
def test___eq___differInPoints(
        points_1, points_2, points_3, actual, expected):
    """This function checks against 3 different pairings of Coordinates."""
    coordinates = Coordinates(points_1, points_2, points_3)
    comparison = coordinates == actual
    assert comparison == expected


def test___eq___differInWeights_notEqual():
    coordinates = Coordinates(1, 2, 3, weights=.5)
    actual = Coordinates(1, 2, 3, weights=0.0)
    assert not coordinates == actual


def test___eq___differInShComment_notEqual():
    coordinates = Coordinates(1, 2, 3, comment="Madre mia!")
    actual = Coordinates(1, 2, 3, comment="Oh my woooooosh!")
    assert not coordinates == actual
