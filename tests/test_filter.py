import pytest
import numpy as np
import numpy.testing as npt
import haiopy.dsp.filter as hpfilter
from haiopy import Signal

import matplotlib.pyplot as plt


def test_filter_init_empty_coefficients():
    filt = hpfilter.Filter(coefficients=None, state=None)
    assert filt._coefficients is None
    assert filt._state is None


def test_filter_init_empty_coefficients_with_state():
    with pytest.raises(ValueError):
        hpfilter.Filter(coefficients=None, state=[1, 0])


def test_filter_init():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    filt = hpfilter.Filter(coefficients=coeff)
    npt.assert_array_equal(filt._coefficients, coeff)


def test_filter_init_empty_state():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    filt = hpfilter.Filter(coefficients=coeff, state=None)
    npt.assert_array_equal(filt._coefficients, coeff)
    assert filt._state is None


def test_filter_init_with_state():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    state = np.array([[[1, 0]]])
    filt = hpfilter.Filter(coefficients=coeff, state=state)
    npt.assert_array_equal(filt._coefficients, coeff)
    npt.assert_array_equal(filt._state, state)


def test_filter_iir_init():
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = hpfilter.FilterIIR(coeff)
    npt.assert_array_equal(filt._coefficients, coeff[np.newaxis])


def test_filter_sos_init():
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    filt = hpfilter.FilterSOS(sos)
    npt.assert_array_equal(filt._coefficients, sos[np.newaxis])


def test_filter_sos_process():
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    filt = hpfilter.FilterSOS(sos)
    filt._state = np.array([[[0, 0]]])

    sr = 4e3
    times = np.arange(0, 2**10)/sr
    sig = Signal(np.sin(2*np.pi*1e3*times), sr, signal_type='power')

    filt.process(sig)

    a = 1

# def test_filter_init_state_empty():
#     coeff = [[1, 0, 0], [1, 0, 0]]
#     filt = hpfilter.Filter(coeff)
#     npt.assert_array_equal(filt._state, np.zeros(2))


# def test_filter_init_state():
#     coeff = [[1, 1, 0], [1, 0, 0]]
#     zi = [1., 0.]
#     filt = hpfilter.Filter(coeff, state=zi)
#     npt.assert_array_equal(filt._state, zi)


# def test_reset():
#     coeff = [[1, 1, 0], [1, 0, 0]]
#     zi = [1., 0.]
#     filt = hpfilter.Filter(coeff, state=zi)
#     filt.reset()
#     npt.assert_array_equal(filt._state, [0., 0.])


def test_atleast_3d_first_dim():
    arr = np.array([1, 0, 0])
    desired = np.array([[[1, 0, 0]]])

    arr_3d = hpfilter.atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)
    arr = np.array([[1, 0, 0], [2, 2, 2]])

    desired = np.array([[[1, 0, 0], [2, 2, 2]]])
    arr_3d = hpfilter.atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)

    arr = np.ones((2, 3, 5))
    desired = arr.copy()
    arr_3d = hpfilter.atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)
