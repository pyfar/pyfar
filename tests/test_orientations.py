import numpy as np
import numpy.testing as npt
from pytest import raises

import haiopy
from haiopy import Orientations


def test_orientations_init():
    """Test to init Orientations without optional parameters."""
    orient = Orientations()
    assert isinstance(orient, Orientations)


def test_orientations_init_view_up():
    """Test to init Orientations with view and up vector(s)."""
    # test with single view and up vectors
    views = [1, 0, 0]
    ups = [0, 1, 0]
    Orientations(views, ups)
    # test with multiple view and up vectors
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0], [0, 1, 0]]
    Orientations(views, ups)
    # view and up counts not matching
    views = [[1, 0, 0], [0, 0, 1]]
    ups = [[0, 1, 0]]
    with raises(ValueError):
        Orientations(views, ups)
    # views and ups must be either empty or
    
