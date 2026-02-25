import pytest
import re

import numpy as np
import numpy.testing as npt

from pyfar import Rotation
from pyfar import Coordinates


def test_rotation_init():
    """Init `Rotation` without optional parameters."""
    match = \
        "Rotation objects must be created using one of the `from_...` methods"
    with pytest.raises(RuntimeError, match=match):
        Rotation()

def test_class_properties():
    """Test Rotation.n_rotations parameter."""
    multidim_angles = np.zeros((3, 2, 3))
    multidim_rot = Rotation.from_euler('XYZ', multidim_angles)
    assert multidim_rot.cshape == (3, 2)
    assert multidim_rot.csize == 6


@pytest.mark.parametrize(
    ("views", "ups"),
    [
        # Single view and up vectors
        (
            [1, 0, 0],
            [0, 1, 0],
        ),
        # Multiple view and up vectors as lists
        (
            [[1, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 0]],
        ),
        # Provided as Coordinates
        (
            Coordinates([1, 0], [0, 0], [0, 1]),
            Coordinates([0, 0], [1, 1], [0, 0]),
        ),
        # N:1 shape broadcasting
        (
            [[1, 0, 0], [0, 0, 1]],
            [[0, 1, 0]],
        ),
        # 1:N shape broadcasting
        (
            [[1, 0, 0]],
            [[0, 1, 0], [0, 1, 0]],
        ),
        # Multi-dimensional shape (..., 3) broadcasting (2, 3)
        (
            np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]]),
            np.array([[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]]]),
        ),
        # Multi-dimensional shape (..., 3) broadcasting (2, 1, 3) with (1, 2, 3)
        (
            np.array([[[1, 0, 0]], [[0, 0, 1]]]),
            np.array([[[0, 1, 0], [0, 1, 0]]]),
        ),
    ]
)
def test_rotation_from_view_up(views, ups):
    """Create `Rotation` from view and up vectors."""
    Rotation.from_view_up(views, ups)


def test_rotation_from_view_up_shape_mismatch():
    """Test that incompatible shapes raise ValueError."""
    # M:N shape mismatch that cannot be broadcast
    views = [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    match = 'shape missmatch: `views` '
    with pytest.raises(ValueError, match=match):
        Rotation.from_view_up(views, ups)


def test_rotation_from_view_up_invalid():
    """Try to create `Rotation` from invalid view and up vectors."""
    # mal-formed lists
    views = [[1, 0, 0], [0, 0]]
    ups = [[0, 1, 0], [0, 0, 0]]
    match = 'setting an array element with a sequence.'
    with pytest.raises(ValueError, match=match):
        Rotation.from_view_up(views, ups)
    # any of views and ups has zero-length
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 0, 0]]
    match = 'View and Up Vectors must have a length.'
    with pytest.raises(ValueError, match=match):
        Rotation.from_view_up(views, ups)
    # views' and ups' shape must be (N, 3) or (3,)
    views = [0, 1]
    ups = [0, 1]
    match = 'Expected `views` and `ups` to have shape'
    with pytest.raises(ValueError, match=match):
        Rotation.from_view_up(views, ups)
    # view and up vectors must be orthogonal
    views = [1.0, 0.5, 0.1]
    ups = [0, 0, 1]
    match = 'View and Up vectors must be perpendicular'
    with pytest.raises(ValueError, match=match):
        Rotation.from_view_up(views, ups)


def test_as_view_up(views, ups, rotation):
    """
    Output of this method must be the normed input vectors.
    """
    views = np.atleast_2d(views).astype(float)
    # normalize view vectors
    views /= np.linalg.norm(views, axis=1)[:, np.newaxis]

    ups = np.atleast_2d(ups).astype(float)
    # normalize up vectors
    ups /= np.linalg.norm(ups, axis=1)[:, np.newaxis]

    # return view_, ups_ from Rotation (normalized in scipy Rotation).
    views_, ups_ = rotation.as_view_up()

    np.testing.assert_allclose(views_, views, atol=1e-15)
    np.testing.assert_allclose(ups_, ups, atol=1e-15)


@pytest.mark.parametrize(
    "v1",
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
)
@pytest.mark.parametrize(
    "v2",
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
)
def test_from_view_as_view_roundtrip(v1, v2):
    "Test from_view_up / as_view_up."
    if np.all(np.abs(v1) == np.abs(v2)):
        pytest.skip()

    views = np.atleast_2d(v1)
    ups = np.atleast_2d(v2)

    rotation = Rotation.from_view_up(views, ups)
    views_, ups_ = rotation.as_view_up()
    # indexed
    views_0, ups_0 = rotation[0].as_view_up()

    npt.assert_allclose(views, views_, atol=1e-15)
    npt.assert_allclose(ups, ups_, atol=1e-15)
    # indexed
    npt.assert_allclose(views[0], views_0, atol=1e-15)
    npt.assert_allclose(ups[0], ups_0, atol=1e-15)


def test_rotation_indexing(rotation):
    """
    Apply index-operator `[]` on `Rotation` to get a single quaternion.
    """
    rotation[0]
    rotation[1]
    with pytest.raises(IndexError):
        rotation[42]

    quats = rotation.as_quat()
    assert np.array_equal(quats[0], rotation[0].as_quat()), (
        "Indexed rotation are not the same")
    assert np.array_equal(quats[1], rotation[1].as_quat()), (
        "Indexed rotation are not the same")

def test_setitem_error(rotation):
    """
    Test if setting an item throws an error.
    """
    match = re.escape('Setting an item is disabled for pyfar Rotations. If '
                      'you want to modify the Rotation, use an array '
                      'representation like `as_quat()` or `as_matrix()` and '
                      'create a new object.')
    with pytest.raises(NotImplementedError, match=match):
        rotation[0] = [[0.5, 1, 0]]


def test_rotation_rotation(views, ups, positions, rotation):
    """Multiply a pyfar Roation with a scipy Rotation and visualize them."""
    # Rotate rotations around x-axis by 45°
    rot_x45 = Rotation.from_euler('x', 45, degrees=True)
    rotation = Rotation.from_view_up(views, ups)
    rotation = rotation * rot_x45


def test___eq___equal(rotation, views, ups):
    actual = Rotation.from_view_up(views, ups)
    assert rotation == actual


def test___eq___notEqual(rotation, views, ups):
    rot_z45 = Rotation.from_euler('z', 45, degrees=True)
    actual = rot_z45 * Rotation.from_view_up(views, ups)
    assert not rotation == actual


@pytest.mark.parametrize(
    ("method", "args"),
    [
        (
            Rotation.from_davenport,
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], "extrinsic", [90, 0, 0]),
        ),
        (
            Rotation.from_euler,
            ('zyx', [90, 45, 30]),
        ),
        (
            Rotation.from_matrix,
            ([[0, -1, 0], [1, 0, 0], [0, 0, 1]],),
        ),
        (
            Rotation.from_mrp,
            ([0, 0, 1],),
        ),
        (
            Rotation.from_quat,
            ([0, 0, 0, 1],),
        ),
        (
            Rotation.from_rotvec,
            (np.pi/2 * np.array([0, 0, 1]),),
        ),
        (
            Rotation.from_view_up,
            ([1, 0, 0], [0, 0, 1]),
        ),
        (
            Rotation.align_vectors,
            ([[0, 1, 0], [0, 1, 1], [0, 1, 1]],
             [[1, 0, 0], [1, 1.1, 0], [1, 0.9, 0]]),
        ),
        (
            Rotation.random,
            (),
        ),
        (
            Rotation.identity,
            (),
        ),
        (
            Rotation.concatenate,
            ([[Rotation.from_rotvec([0, 0, 1]),
             Rotation.from_rotvec([0, 0, 2])]]),
        ),
    ],
)
def test_methods_return_type(method, args):
    """Test if class-methods return Rotation instance."""
    obj = method(*args)

    if method.__name__ in ["align_vectors"]:
        assert isinstance(obj[0], Rotation)
    else:
        assert isinstance(obj, Rotation)


def test__pow__():
    """Test wrapped __pow__ method."""
    rotation = Rotation.from_rotvec([1, 0, 0])

    assert isinstance(rotation, Rotation)
    npt.assert_allclose((rotation**2).as_rotvec(), [2, 0, 0])
    npt.assert_allclose((rotation**0.5).as_rotvec(), [0.5, 0, 0])

def tests_instance_methods():
    """Test wrapped reduce method."""
    rotation = Rotation.from_rotvec([1, 0, 0])

    obj = rotation.reduce()
    assert isinstance(obj, Rotation)

    obj = obj.mean()
    assert isinstance(obj, Rotation)

    obj = obj.inv()
    assert isinstance(obj, Rotation)

def test__iter__():
    """Test iteration over Rotation."""
    rotations = Rotation.from_rotvec([[1, 0, 0], [2, 0, 0]])

    for i, rotation in enumerate(rotations):
        if i == 0:
            npt.assert_equal(rotation.as_rotvec(), [1, 0, 0])
        if i == 1:
            npt.assert_equal(rotation.as_rotvec(), [2, 0, 0])

        assert isinstance(rotation, Rotation)
