import numpy as np
import numpy.testing as npt
import pytest

import haiopy
from haiopy import Orientation


def test_orientation_init():
    """Test to init Orientation without optional parameters."""
    orient = Orientation()
    assert isinstance(orient, Orientation)