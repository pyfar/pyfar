from pytest import raises

import numpy as np

from haiopy.orientations import Orientations
from haiopy.coordinates import Coordinates


def test_orientations_init():
    """Test to init Orientations without optional parameters."""
    orient = Orientations()
    assert isinstance(orient, Orientations)
    
    
def test_orientations_from_view_up():
    # test with single view and up vectors
    view = [1, 0, 0]
    up = [0, 1, 0]
    Orientations.from_view_up(view, up)
    # test with multiple view and up vectors
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    Orientations.from_view_up(views, ups)
    # provided as ndarrays
    views = np.atleast_2d(views).astype(np.float64)
    ups = np.atleast_2d(ups).astype(np.float64)
    Orientations.from_view_up(views, ups)
    # provided as Coordinates
    views = Coordinates(views[:, 0], views[:, 1], views[:, 2])
    ups = Coordinates(ups[:, 0], ups[:, 1], ups[:, 2])
    Orientations.from_view_up(views, ups)
    

def test_orientations_from_view_up_invalid():
    # mal-formed lists
    views = [[1, 0, 0], [0, 0]]
    ups = [[0, 1, 0], [0, 0, 0]]
    with raises(ValueError):
        Orientations.from_view_up(views, ups)
    # any of views and ups has zero-length
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 0, 0]]
    with raises(ValueError):
        Orientations.from_view_up(views, ups)
    # view and up counts not matching
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0]]
    with raises(ValueError):
        Orientations.from_view_up(views, ups)
    # views' and ups' shape must be (N, 3) or (3,)
    views = [0, 1]
    ups = [0, 1]
    with raises(ValueError):
        Orientations.from_view_up(views, ups)
    # view and up vectors must be orthogonal
    views = [1.0, 0.5, 0.1]
    ups = [0, 0, 1]
    with raises(ValueError):
        Orientations.from_view_up(views, ups)


def test_orientations_show():
    """Test method show with and without positions"""
    # single vectors no position
    view = [1, 0, 0]
    up = [0, 1, 0]
    orientation = Orientations.from_view_up(view, up)
    orientation.show()
    # with position
    position = Coordinates(0, 1, 0)
    orientation.show(position)
    # multiple vectors no position
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    orientations = Orientations.from_view_up(views, ups)
    orientations.show()
    # with matching number of positions
    positions = [[0, 0, 0], [-2, 1, 3]]
    orientations.show(positions)
    # with positions provided as Coordinates
    positions = Coordinates([0, 5], [0, 0], [0, 0])
    orientations.show(positions)
    # with non-matching positions
    positions = Coordinates(0, 1, 0)
    with raises(ValueError):
        orientations.show(positions)