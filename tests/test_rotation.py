import pytest

import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation

from pyfar import Rotation
from pyfar import Coordinates


def test_rotation_init():
    """Init `Rotation` without optional parameters."""
    match = \
        "Rotation objects must be created using one of the `from_...` methods"
    with pytest.raises(RuntimeError, match=match):
        rotation = Rotation()


def test_rotation_from_view_up():
    """Create `Rotation` from view and up vectors."""
    # test with single view and up vectors
    view = [1, 0, 0]
    up = [0, 1, 0]
    Rotation.from_view_up(view, up)
    # test with multiple view and up vectors
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    Rotation.from_view_up(views, ups)
    # provided as numpy ndarrays
    views = np.atleast_2d(views).astype(np.float64)
    ups = np.atleast_2d(ups).astype(np.float64)
    Rotation.from_view_up(views, ups)
    # provided as Coordinates
    views = Coordinates(views[:, 0], views[:, 1], views[:, 2])
    ups = Coordinates(ups[:, 0], ups[:, 1], ups[:, 2])
    Rotation.from_view_up(views, ups)
    # number of views to ups N:1
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0]]
    Rotation.from_view_up(views, ups)
    # number of views to ups 1:N
    views = [[1, 0, 0]]
    ups = [[0, 1, 0], [0, 1, 0]]
    Rotation.from_view_up(views, ups)
    # number of views to ups M:N
    views = [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    match = 'Expected 1:1, 1:N or N:1 `views` and `ups` not M:N, got '
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


def test_rotation_show(positions, rotation):
    """
    Visualize rotation via `Rotation.show()`
    with and without `positions`.
    """
    # default orientation
    Rotation().show()
    # single vectors no position
    view = [1, 0, 0]
    up = [0, 1, 0]
    orientation_single = Rotation.from_view_up(view, up)
    orientation_single.show()
    # with position
    position = Coordinates(0, 1, 0)
    orientation_single.show(position)

    # multiple vectors no position
    rotation.show()
    # with matching number of positions
    rotation.show(positions)

    # select what to show
    rotation.show(show_views=False)
    rotation.show(show_ups=False)
    rotation.show(show_rights=False)
    rotation.show(show_views=False, show_ups=False)
    rotation.show(show_views=False, show_rights=False)
    rotation.show(show_ups=False, show_rights=False)
    rotation.show(positions=positions, show_views=False, show_ups=False)

    # with positions provided as Coordinates
    positions = np.asarray(positions)
    positions = Coordinates(positions[:, 0], positions[:, 1], positions[:, 2])
    rotation.show(positions)
    # with non-matching positions
    positions = Coordinates(0, 1, 0)
    match = 'If provided, there must be the same number'
    with pytest.raises(ValueError, match=match):
        rotation.show(positions)


def test_as_view_up_right(views, ups, rotation):
    """
    Output of this method must be the normed input vectors.
    """
    views = np.atleast_2d(views).astype(np.float64)
    views /= np.linalg.norm(views, axis=1)[:, np.newaxis]
    ups = np.atleast_2d(ups).astype(np.float64)
    ups /= np.linalg.norm(ups, axis=1)[:, np.newaxis]

    views_, ups_, _ = rotation.as_view_up_right()

    assert np.array_equal(views_, views), "views are not preserved"
    assert np.array_equal(ups_, ups), "ups are not preserved"


def test_from_view_as_view_roundtrip():
    vec = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]])

    for v1 in range(len(vec)):
        for v2 in range(len(vec)):
            if np.all(np.abs(vec[v1]) == np.abs(vec[v2])):
                continue
            print(f"Testing combination ({v1}, {v2})")
            views = np.atleast_2d(vec[v1])
            ups = np.atleast_2d(vec[v2])

            rotation = Rotation.from_view_up(views, ups)
            views_, ups_, _ = rotation.as_view_up_right()
            # indexed
            views_0, ups_0, _ = rotation[0].as_view_up_right()

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

def test_rotation_rotation(views, ups, positions, rotation):
    """Multiply a pyfar Roation with a scipy Rotation and visualize them."""
    rotation.show(positions)
    # Rotate first Orientation around z-axis by 45°
    rot_z45 = Rotation.from_euler('z', 45, degrees=True)
    rotation[0] = rotation[0] * rot_z45
    rotation.show(positions)
    # Rotate second Orientation around x-axis by 45°
    rot_x45 = Rotation.from_euler('x', 45, degrees=True)
    rotation[1] = rotation[1] * rot_x45
    rotation.show(positions)
    # Rotate both Rotation at once
    rotation = Rotation.from_view_up(views, ups)
    rotation = rotation * rot_x45
    rotation.show(positions)


def test___eq___equal(rotation, views, ups):
    actual = Rotation.from_view_up(views, ups)
    assert rotation == actual


def test___eq___notEqual(rotation, views, ups):
    rot_z45 = Rotation.from_euler('z', 45, degrees=True)
    actual = rot_z45 * Rotation.from_view_up(views, ups)
    assert not rotation == actual
