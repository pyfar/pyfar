import pytest
import numpy as np
import numpy.testing as npt
from pyfar import Signal


def test_simple_iter():
    data = np.ones((3, 1024)) * np.atleast_2d(np.arange(3)).T
    sig = Signal(data, 1)

    idx = 0
    for s in sig:
        npt.assert_array_equal(s._data, np.atleast_2d(data[idx, :]))
        idx += 1

    for idx, s in enumerate(sig):
        npt.assert_array_equal(s._data, np.atleast_2d(data[idx, :]))


def test_iter_write():
    data = np.ones((3, 1024)) * np.atleast_2d(np.arange(3)).T
    sig = Signal(data, 1)
    sig.domain = 'time'

    for idx, s in enumerate(sig):
        sig[idx] = s


def test_iter_domain_change():
    data = np.ones((3, 1024)) * np.atleast_2d(np.arange(3)).T
    sig = Signal(data, 1)
    sig.domain = 'time'

    with pytest.raises(RuntimeError, match='domain changes'):
        for s in sig:
            s.freq
