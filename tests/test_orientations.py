from pytest import raises

from haiopy.orientations import Orientations
from haiopy.coordinates import Coordinates


def test_orientations_init():
    """Test to init Orientations without optional parameters."""
    orient = Orientations()
    assert isinstance(orient, Orientations)
    

def test_orientations_show():
    """Test method show with and without positions"""
    orientations = Orientations()
    # TODO: raise specific exception
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    orientations = Orientations(views, ups)
    # without positions
    orientations.show()
    # wrong type
    with raises(TypeError):
        orientations.show([0, 1, 0])
    # with positions of differing size
    positions = Coordinates(0, 1, 0)
    with raises(ValueError):
        orientations.show(positions)
    positions = Coordinates([0, 5], [0, 0], [0, 0])
    orientations.show(positions)
    pass