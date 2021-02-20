import pytest
import numpy as np
import numpy.testing as npt
import pyfar.dsp.classes as fo
from pyfar import Signal
from scipy import signal as spsignal
from unittest import mock


def test_filter_init_empty_coefficients():
    filt = fo.Filter(coefficients=None, state=None, sampling_rate=None)
    assert filt._coefficients is None
    assert filt._state is None
    assert filt.comment is None


def test_filter_init_empty_coefficients_with_state():
    with pytest.raises(ValueError):
        fo.Filter(coefficients=None, state=[1, 0], sampling_rate=None)


def test_filter_init():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    filt = fo.Filter(coefficients=coeff, sampling_rate=None)
    npt.assert_array_equal(filt._coefficients, coeff)


def test_filter_init_empty_state():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    filt = fo.Filter(coefficients=coeff, state=None)
    npt.assert_array_equal(filt._coefficients, coeff)
    assert filt._state is None


def test_filter_init_with_state():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    state = np.array([[[1, 0]]])
    filt = fo.Filter(coefficients=coeff, state=state)
    npt.assert_array_equal(filt._coefficients, coeff)
    npt.assert_array_equal(filt._state, state)


def test_filter_comment():
    filt = fo.Filter(coefficients=None, state=None, comment='Bla')
    assert filt.comment == 'Bla'
    filt.comment = 'Blub'
    assert filt.comment == 'Blub'
    filt.comment = 500
    assert filt.comment == '500'


def test_filter_iir_init():
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterIIR(coeff, sampling_rate=2*np.pi)
    npt.assert_array_equal(filt._coefficients, coeff[np.newaxis])


def test_filter_fir_init():
    coeff = np.array([1, 1/2, 0])
    desired = np.array([[[1, 1/2, 0], [1, 0, 0]]])
    filt = fo.FilterFIR(coeff, sampling_rate=2*np.pi)
    npt.assert_array_equal(filt._coefficients, desired)


def test_filter_fir_init_multi_dim():
    coeff = np.array([
        [1, 1/2, 0],
        [1, 1/4, 1/8]])
    desired = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 1/8], [1, 0, 0]]
        ])
    filt = fo.FilterFIR(coeff, sampling_rate=2*np.pi)
    npt.assert_array_equal(filt._coefficients, desired)


def test_filter_sos_init():
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    filt = fo.FilterSOS(sos, sampling_rate=2*np.pi)
    npt.assert_array_equal(filt._coefficients, sos[np.newaxis])


def test_filter_iir_process(impulse):
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[:3], coeff[0])

    coeff = np.array([[1, 0, 0], [1, 1, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    npt.assert_allclose(res.time, desired)


def test_filter_iir_process_state(impulse):
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate, state=[0, 0])
    res = filt.process(impulse, reset=False)
    state = filt._state

    npt.assert_allclose([[0, 0]], state)
    npt.assert_allclose(res.time[:3], coeff[0])

    coeff = np.array([[1, 0, 0], [1, 1, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate,  state=[0, 0])
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    npt.assert_allclose(res.time, desired)
    npt.assert_allclose(filt._state, [[1, 0]])


def test_filter_fir_process(impulse):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[:3], coeff)


def test_filter_fir_process_state(impulse):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate, state=[0, 0])
    res = filt.process(impulse, reset=False)
    state = filt._state

    npt.assert_allclose([[0, 0]], state)
    npt.assert_allclose(res.time[:3], coeff)


def test_filter_fir_process_sampling_rate_mismatch(impulse):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate-1)
    with pytest.raises(ValueError):
        filt.process(impulse)


def test_filter_iir_process_multi_dim_filt(impulse):
    coeff = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 0], [1, 0, 0]]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)

    res = filt.process(impulse)

    npt.assert_allclose(res.time[:, :3], coeff[:, 0])


def test_filter_fir_process_multi_dim_filt(impulse):
    coeff = np.array([
        [1, 1/2, 0],
        [1, 1/4, 0]])

    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)
    npt.assert_allclose(res.time[:, :3], coeff)


def test_filter_sos_process(impulse):
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[:3], coeff[0])

    sos = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    npt.assert_allclose(res.time, desired)


def test_filter_sos_process_state(impulse):
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    desired = np.array([1, 1/2, 0])
    filt = fo.FilterSOS(sos, impulse.sampling_rate, state=[0, 0])
    res = filt.process(impulse, reset=False)
    state = filt._state

    npt.assert_allclose([[[0, 0]]], state)
    npt.assert_allclose(res.time[:3], desired)

    sos = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate,  state=[0, 0])
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    state = filt._state
    npt.assert_allclose(res.time, desired)
    npt.assert_allclose(filt._state, [[[1, 0]]])


def test_filter_sos_process_multi_dim_filt(impulse):
    sos = np.array([
        [[1, 1/2, 0, 1, 0, 0]],
        [[1, 1/4, 0, 1, 0, 0]]])
    coeff = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 0], [1, 0, 0]]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[:, :3], coeff[:, 0])


def test_atleast_3d_first_dim():
    arr = np.array([1, 0, 0])
    desired = np.array([[[1, 0, 0]]])

    arr_3d = fo.atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)
    arr = np.array([[1, 0, 0], [2, 2, 2]])

    desired = np.array([[[1, 0, 0], [2, 2, 2]]])
    arr_3d = fo.atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)

    arr = np.ones((2, 3, 5))
    desired = arr.copy()
    arr_3d = fo.atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)


def test_extend_sos_coefficients():
    sos = np.array([
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
    ])

    actual = fo.extend_sos_coefficients(sos, 2)
    npt.assert_allclose(actual, sos)

    expected = np.array([
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
    ])

    actual = fo.extend_sos_coefficients(sos, 4)
    npt.assert_allclose(actual, expected)

    # test if the extended filter has an ideal impulse response.
    imp = np.zeros(512)
    imp[0] = 1
    imp_filt = spsignal.sosfilt(actual, imp)
    npt.assert_allclose(imp_filt, imp)


def test_impulse_mock(impulse_mock):
    n_samples = 1000
    sampling_rate = 2000
    amplitude = 1

    signal = np.atleast_2d(np.zeros(n_samples, dtype=np.double))
    signal[:, 0] = amplitude

    assert impulse_mock.sampling_rate == sampling_rate
    assert impulse_mock.cshape == (1,)
    npt.assert_allclose(impulse_mock.time, signal)


@pytest.fixture
def impulse_mock():
    """ Generate a signal mock object.
    Returns
    -------
    signal : Signal
        The noise signal
    """
    n_samples = 1000
    sampling_rate = 2000
    amplitude = 1
    cshape = (1,)
    domain = 'time'

    signal = np.zeros(n_samples, dtype=np.double)
    signal[0] = amplitude

    # create a mock object of Signal class to test independently
    signal_object = mock.Mock(
        spec_set=Signal(signal, sampling_rate, n_samples, domain))
    signal_object.time = np.atleast_2d(signal)
    signal_object.sampling_rate = sampling_rate
    signal_object.domain = domain
    signal_object.cshape = cshape

    return signal_object


def test___eq___equal(filter):
    actual = filter.copy()
    assert filter == actual


def test___eq___notEqual(filter, coeffs, state):
    actual = fo.Filter(coefficients=2 * coeffs, state=state)
    assert not filter == actual
    actual = fo.Filter(coefficients=coeffs, state=2 * state)
    assert not filter == actual
    actual = filter.copy()
    actual.comment = f'{actual.comment} A completely different thing'
    assert not filter == actual
