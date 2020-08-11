from pytest import raises

from haiopy.orientations import Orientations
from haiopy.coordinates import Coordinates


def test_orientations_init():
    """Test to init Orientations without optional parameters."""
    orient = Orientations()
    assert isinstance(orient, Orientations)