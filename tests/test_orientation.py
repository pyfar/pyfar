import numpy as np
import numpy.testing as npt
import pytest

import haiopy
from haiopy import Orientation


def test_orientation_init():
    """Test to init Orientation without optional parameters."""
    orient = Orientation()
    assert isinstance(orient, Orientation)

def test_orientation_init_val():
    """Test to init Orientation with all parameters."""
    view = np.array([0, 1, 0])
    up = np.array([1, 0, 0])
    orient = Orientation(view, up)
    assert isinstance(orient, Orientation)
    

def test_coordinates_init_incomplete():
    """Test to init Orientation when view and up have different dimensions."""
    view = np.array([0, 1])
    up = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        Orientation(view, up)
        pytest.fail("Input arrays need to have same dimensions.")

def test_orientation_init_ortho():
    """Test to init Orientation when view and up vector are not orthogonal."""
    view = np.array([1, 1, 0])
    up = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        Orientation(view, up)
        pytest.fail("Input arrays need to be orthogonal to each other.")

def test_getter_view():
    view = np.array([0, 1, 0])
    view_test = np.array([0, 0, 1])
    up = np.array([1, 0, 0])
    orient = Orientation(view, up)
    orient._view = view_test
    npt.assert_allclose(orient.view, view_test)

def test_setter_view():
    view = np.array([0, 1, 0])
    view_test = np.array([0, 0, 1])
    up = np.array([1, 0, 0])
    orient = Orientation(view, up)
    orient.view = view_test
    npt.assert_allclose(view_test, orient._view)
    
def test_getter_up():
    view = np.array([0, 1, 0])
    up = np.array([1, 0, 0])
    up_test = np.array([0, 0, 1])
    orient = Orientation(view, up)
    orient._up = up_test
    npt.assert_allclose(orient.up, up_test)

def test_setter_up():
    view = np.array([0, 1, 0])
    up = np.array([1, 0, 0])
    up_test = np.array([0, 0, 1])
    orient = Orientation(view, up)
    orient.up = up_test
    npt.assert_allclose(up_test, orient._up)


        
        