import numpy as np
import numpy.testing as npt
from haiopy import dsp


def test_filter_init():
    coeff = [[1, 0, 0], [1, 0, 0]]
    filt = dsp.Filter(coeff)
    npt.assert_array_equal(filt._coefficients, coeff)


def test_filter_init_state_empty():
    coeff = [[1, 0, 0], [1, 0, 0]]
    filt = dsp.Filter(coeff)
    npt.assert_array_equal(filt._state, np.zeros(2))


def test_filter_init_state():
    coeff = [[1, 1, 0], [1, 0, 0]]
    zi = [1., 0.]
    filt = dsp.Filter(coeff, state=zi)
    npt.assert_array_equal(filt._state, zi)


def test_reset():
    coeff = [[1, 1, 0], [1, 0, 0]]
    zi = [1., 0.]
    filt = dsp.Filter(coeff, state=zi)
    filt.reset()
    npt.assert_array_equal(filt._state, [0., 0.])
