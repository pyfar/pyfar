import numpy as np
import numpy.testing as npt
from pytest import raises

import haiopy
from haiopy import Orientations


def test_orientation_init():
    """Test to init Orientation without optional parameters."""
    orient = Orientations()
    assert isinstance(orient, Orientations)


def test_orientation_init_view_up():
    """Test to init Orientation with view and up vector(s)."""
    # test with view and up vector
    view = [1, 0, 0]
    up = [0, 1, 0]
    Orientations(view, up)
