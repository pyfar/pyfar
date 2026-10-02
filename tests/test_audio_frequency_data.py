import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import FrequencyData


def test__repr__(capfd):
    """Test string representation."""
    print(FrequencyData([1, 2, 3], [1, 2, 3]))
    out, _ = capfd.readouterr()
    assert ("FrequencyData:\n"
            "(1,) channels with 3 frequencies") in out
