import pytest
import numpy as np
import numpy.testing as npt
import pyfar.classes.filter as fo
from scipy import signal as spsignal


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

    npt.assert_allclose(res.time[0, :3], coeff[0])

    coeff = np.array([[1, 0, 0], [1, 1, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)
    desired = np.ones((1, impulse.n_samples))
    desired[:, 1::2] *= -1

    npt.assert_allclose(res.time, desired)


def test_filter_fir_process(impulse):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, :3], coeff)


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

    impulse.time = np.vstack((impulse.time, impulse.time))
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, 0, :3], coeff[0, 0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 0, :3], coeff[1, 0, :], atol=1e-16)

    npt.assert_allclose(res.time[0, 1, :3], coeff[0, 0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 1, :3], coeff[1, 0, :], atol=1e-16)


def test_filter_fir_process_multi_dim_filt(impulse):
    coeff = np.array([
        [1, 1/2, 0],
        [1, 1/4, 0]])

    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)
    npt.assert_allclose(res.time[:, :3], coeff)

    impulse.time = np.vstack((impulse.time, impulse.time))
    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, 0, :3], coeff[0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 0, :3], coeff[1, :], atol=1e-16)

    npt.assert_allclose(res.time[0, 1, :3], coeff[0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 1, :3], coeff[1, :], atol=1e-16)


def test_filter_sos_process(impulse):
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, :3], coeff[0])


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

    impulse.time = np.vstack((impulse.time, impulse.time))
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, 0, :3], coeff[0, 0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 0, :3], coeff[1, 0, :], atol=1e-16)

    npt.assert_allclose(res.time[0, 1, :3], coeff[0, 0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 1, :3], coeff[1, 0, :], atol=1e-16)


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
