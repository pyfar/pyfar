import numpy as np
import numpy.testing as npt
from haiopy import Signal


def test_simple_iter():
    data = np.ones((3, 1024)) * np.atleast_2d(np.arange(3)).T
    sig = Signal(data, 1)

    idx = 0
    for s in sig:
        npt.assert_array_equal(s._data, data[idx, :])
        idx += 1
