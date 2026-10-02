import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import TimeData


def test___eq___equal():
    """Check if copied TimeData is equal."""
    time_data = TimeData([1, 2, 3], [0.1, 0.2, 0.3])
    actual = time_data.copy()
    assert time_data == actual


def test___eq___notEqual():
    """Check if TimeData object is equal."""
    time_data = TimeData([1, 2, 3], [0.1, 0.2, 0.3])
    actual = TimeData([2, 3, 4], [0.1, 0.2, 0.3])
    assert not time_data == actual
    actual = TimeData([1, 2, 3], [0.2, 0.3, 0.4])
    assert not time_data == actual
    comment = f'{time_data.comment} A completely different thing'
    actual = TimeData([1, 2, 3], [0.1, 0.2, 0.3], comment=comment)
    assert not time_data == actual


def test__repr__(capfd):
    """Test string representation."""
    print(TimeData([1, 2, 3], [1, 2, 3]))
    out, _ = capfd.readouterr()
    assert ("TimeData:\n"
            "(1,) channels with 3 samples") in out
