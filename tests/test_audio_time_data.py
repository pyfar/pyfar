import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import TimeData


def test__repr__(capfd):
    """Test string representation."""
    print(TimeData([1, 2, 3], [1, 2, 3]))
    out, _ = capfd.readouterr()
    assert ("TimeData:\n"
            "(1,) channels with 3 samples") in out
