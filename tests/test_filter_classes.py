import pytest
import numpy as np
import numpy.testing as npt
import pyfar.classes.filter as fo
import pyfar as pf
import re
from scipy import signal as spsignal


def test_filter_init_empty_coefficients():
    filt = fo.Filter(coefficients=None, state=None, sampling_rate=None)
    assert filt.coefficients is None
    assert filt.sampling_rate is None
    assert filt.state is None
    assert filt.comment == ''


def test_filter_init_empty_coefficients_with_state():
    with pytest.raises(ValueError, match="Cannot set a state without"):
        fo.Filter(coefficients=None, state=[1, 0], sampling_rate=None)


def test_filter_init():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    filt = fo.Filter(coefficients=coeff, sampling_rate=None)
    npt.assert_array_equal(filt._coefficients, coeff)
    assert filt.state is None


def test_filter_init_with_state():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    state = np.array([[[1, 0]]])
    filt = fo.Filter(coefficients=coeff, state=state)
    npt.assert_array_equal(filt._coefficients, coeff)
    npt.assert_array_equal(filt.state, state)


def test_filter_state_setter():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    state = np.array([[[1, 0]]])
    filt = fo.Filter(coefficients=coeff)
    filt.state = state
    npt.assert_array_equal(filt.state, state)


def test_filter_state_setter_errors():
    """Test errors when setting a state with wrong shape."""
    fs = 44100
    coeff_iir = np.array([[[1, 0, 0], [1, 0, 0]]])
    with pytest.raises(ValueError, match=re.escape(
        "The state does not match the filter structure. Required "
        "shape for FilterIIR is (n_channels, *cshape, order).")):
        fo.FilterIIR(coeff_iir, fs, state=[[[1, 0, 0]]])

    coeff_fir = np.array([1, 0, 0])
    with pytest.raises(ValueError, match=re.escape(
        "The state does not match the filter structure. Required "
        "shape for FilterFIR is (n_channels, *cshape, order).")):
        fo.FilterFIR(coeff_fir, fs, state=[[[1, 0, 0]]])

    coeff_sos = [[[1, .5, .25, 1, .5, .25], [1, .5, .25, 1, .5, .25]]]
    with pytest.raises(ValueError, match=re.escape(
        "The state does not match the filter structure. Required shape for "
        "FilterSOS is (n_channels, *cshape, n_sections, 2).")):
        fo.FilterSOS(coeff_sos, fs, state=[[[[0], [1]]]])


def test_filter_state_process_errors():
    """Test errors when processing a signal with wrong state shape."""
    fs = 44100
    impulse = pf.signals.impulse(256, amplitude=np.ones((4, 1)))
    coeff_iir = np.array([[[1, 0, 0], [1, 0, 0]]])
    with pytest.raises(ValueError, match=re.escape(
        "The initial state does not match the cshape of the signal. Required "
        "shape for `state` in FilterIIR is (n_channels, *cshape, order).")):
        fo.FilterIIR(coeff_iir, fs, state=[[[1, 0]]]).process(impulse)

    coeff_fir = np.array([1, 0, 0])
    with pytest.raises(ValueError, match=re.escape(
        "The initial state does not match the cshape of the signal. Required "
        "shape for `state` in FilterFIR is (n_channels, *cshape, order).")):
        fo.FilterFIR(coeff_fir, fs, state=[[[1, 0]]]).process(impulse)

    coeff_sos = [[[1, .5, .25, 1, .5, .25], [1, .5, .25, 1, .5, .25]]]
    with pytest.raises(ValueError, match=re.escape(
        "The initial state does not match the cshape of the signal. Required "
        "shape for `state` in FilterSOS is (n_channels, *cshape, n_sections,"
        " 2).")):
        fo.FilterSOS(
            coeff_sos, fs, state=[[[[0, 0], [1, 0]]]]).process(impulse)


def test_filter_comment():
    filt = fo.Filter(coefficients=None, state=None, comment='Bla')
    assert filt.comment == 'Bla'
    filt.comment = 'Blub'
    assert filt.comment == 'Blub'
    with pytest.raises(TypeError, match="comment has to be of type string."):
        pf.Signal([1, 2, 3], 44100, comment=[1, 2, 3])


@pytest.mark.parametrize('filter_object', [
    (fo.FilterFIR([[1, -1]], 48000)),
    (fo.FilterIIR([[1, -1], [1, -1]], 48000)),
    (fo.FilterSOS([[[1, -1, 1, 1, -1, 1]]], 48000)),
])
def test_filter_init_state_default_and_error(filter_object):
    """
    Test the default for the state keyword and if an error is raised if passing
    an invalid value.
    """

    # set state explicitly
    filter_object.init_state((1, ), 'zeros')
    state_explicit = filter_object.state.copy()
    # set state implicitly using the default
    filter_object.init_state((1, ))
    state_implicit = filter_object.state.copy()
    # assert states
    npt.assert_equal(state_implicit, state_explicit)

    # test raising the error for an invalid value
    with pytest.raises(ValueError, match="state is 'random' but must be"):
        filter_object.init_state((1, ), 'random')



def test_filter_iir_init():
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterIIR(coeff, sampling_rate=2*np.pi)
    npt.assert_array_equal(filt.coefficients, coeff[np.newaxis])


def test_filter_fir_init():
    coeff = np.array([1, 1/2, 0])
    desired = np.array([[[1, 1/2, 0], [1, 0, 0]]])
    filt = fo.FilterFIR(coeff, sampling_rate=2*np.pi)
    # seprately test internal coefficients and property because they differ
    npt.assert_array_equal(filt._coefficients, desired)
    npt.assert_array_equal(filt.coefficients, np.atleast_2d(coeff))


def test_filter_fir_init_multi_dim():
    coeff = np.array([
        [1, 1/2, 0],
        [1, 1/4, 1/8]])
    desired = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 1/8], [1, 0, 0]],
        ])
    filt = fo.FilterFIR(coeff, sampling_rate=2*np.pi)
    # seprately test internal coefficients and property because they differ
    npt.assert_array_equal(filt._coefficients, desired)
    npt.assert_array_equal(filt.coefficients, coeff)


def test_filter_sos_init():
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    filt = fo.FilterSOS(sos, sampling_rate=2*np.pi)
    npt.assert_array_equal(filt.coefficients, sos[np.newaxis])


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


def test_filter_iir_process_complex(impulse_complex):
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterIIR(coeff, impulse_complex.sampling_rate)
    res = filt.process(impulse_complex)

    npt.assert_allclose(res.time[0, :3], coeff[0])

    coeff = np.array([[1, 0, 0], [1, 1, 0]])
    filt = fo.FilterIIR(coeff, impulse_complex.sampling_rate)
    res = filt.process(impulse_complex)
    desired = np.ones((1, impulse_complex.n_samples))
    desired[:, 1::2] *= -1

    npt.assert_allclose(res.time, desired)


def test_filter_iir_process_state(impulse):
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate, state=[0, 0])
    res = filt.process(impulse, reset=False)
    state = filt.state

    npt.assert_allclose([[[0, 0]]], state)
    npt.assert_allclose(res.time[0, :3], coeff[0])

    coeff = np.array([[1, 0, 0], [1, 1, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate,  state=[0, 0])
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    npt.assert_allclose(res.time, np.atleast_2d(desired))
    npt.assert_allclose(filt.state, [[[1, 0]]])


def test_filter_iir_init_state(impulse):
    coeff = np.array([[1, 0, 0], [1, 1, 0]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)

    # init empty filter
    filt.init_state(impulse.cshape, state='zeros')
    desired = np.array([[[0, 0]]])
    npt.assert_allclose(filt.state, desired)

    # init with step function response
    filt.init_state(impulse.cshape, state='step')
    desired = np.array([[[-0.5, 0]]])
    npt.assert_allclose(filt.state, desired)

    # init with step function response multichannel
    filt.init_state((2, 3), state='zeros')
    desired = np.zeros((1, 2, 3, 2), dtype=float)
    npt.assert_allclose(filt.state, desired)

    # init with step function response multichannel
    filt.init_state((2, 3), state='step')
    desired = np.zeros((1, 2, 3, 2), dtype=float)
    desired[..., 0] = -0.5
    npt.assert_allclose(filt.state, desired)


def test_filter_fir_process(impulse):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, :3], coeff)


def test_filter_fir_process_complex(impulse_complex):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse_complex.sampling_rate)
    res = filt.process(impulse_complex)

    npt.assert_allclose(res.time[0, :3], coeff)


def test_filter_fir_process_state(impulse):
    coeff = np.array([1, 1/2, 0, 0, 0, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate, state=[0, 0, 0, 0, 0])
    res = filt.process(impulse, reset=False)
    state = filt.state

    npt.assert_allclose([[[0, 0, 0, 0, 0]]], state)
    npt.assert_allclose(res.time[:, :6], np.atleast_2d(coeff))


def test_filter_fir_init_state(impulse):
    coeff = np.array([1, 1/2, 0, 0, 0, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate)

    # init empty filter
    filt.init_state(impulse.cshape, state='zeros')
    desired = np.array([[[0, 0, 0, 0, 0]]])
    npt.assert_allclose(filt.state, desired)

    # init with step function response
    filt.init_state(impulse.cshape, state='step')
    desired = np.array([[[0.5, 0, 0, 0, 0]]])
    npt.assert_allclose(filt.state, desired)

    # init with step function response multichannel
    filt.init_state((2, 3), state='zeros')
    desired = np.zeros((1, 2, 3, 5), dtype=float)
    npt.assert_allclose(filt.state, desired)

    # init with step function response multichannel
    filt.init_state((2, 3), state='step')
    desired = np.zeros((1, 2, 3, 5), dtype=float)
    desired[..., 0] = 0.5
    npt.assert_allclose(filt.state, desired)


def test_filter_fir_process_sampling_rate_mismatch(impulse):
    coeff = np.array([1, 1/2, 0])
    filt = fo.FilterFIR(coeff, impulse.sampling_rate-1)
    match = 'The sampling rates of filter and signal do not match'
    with pytest.raises(ValueError, match=match):
        filt.process(impulse)


def test_filter_iir_process_multi_dim_filt(impulse):
    coeff = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 0], [1, 0, 0]]])
    filt = fo.FilterIIR(coeff, impulse.sampling_rate)

    res = filt.process(impulse)

    npt.assert_allclose(res.time[:, 0, :3], coeff[:, 0])

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
    npt.assert_allclose(res.time[:, 0, :3], coeff)

    impulse.time = np.vstack((impulse.time, impulse.time))
    filt = fo.FilterFIR(coeff, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, 0, :3], coeff[0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 0, :3], coeff[1, :], atol=1e-16)

    npt.assert_allclose(res.time[0, 1, :3], coeff[0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 1, :3], coeff[1, :], atol=1e-16)


def test_filter_sos_process(impulse):
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, :3], coeff[0])

    sos = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    npt.assert_allclose(res.time, np.atleast_2d(desired))


def test_filter_sos_process_complex(impulse_complex):
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    coeff = np.array([[1, 1/2, 0], [1, 0, 0]])
    filt = fo.FilterSOS(sos, impulse_complex.sampling_rate)
    res = filt.process(impulse_complex)

    npt.assert_allclose(res.time[0, :3], coeff[0])

    sos = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(sos, impulse_complex.sampling_rate)
    res = filt.process(impulse_complex)
    desired = np.ones(impulse_complex.n_samples)
    desired[1::2] *= -1

    npt.assert_allclose(res.time, np.atleast_2d(desired))


def test_filter_sos_process_state(impulse):
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    desired = np.array([[1, 1/2, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate, state=[0, 0])
    res = filt.process(impulse, reset=False)
    state = filt.state

    npt.assert_allclose([[[[0, 0]]]], state)
    npt.assert_allclose(res.time[:, :3], desired)

    sos = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate,  state=[0, 0])
    res = filt.process(impulse)
    desired = np.ones(impulse.n_samples)
    desired[1::2] *= -1

    state = filt.state
    npt.assert_allclose(res.time, np.atleast_2d(desired))
    npt.assert_allclose(filt.state, [[[[1, 0]]]])

    sos = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate,  state=[1, 0])
    res = filt.process(impulse)
    desired = 2 * np.ones(impulse.n_samples)
    desired[1::2] *= -1
    state = filt.state
    npt.assert_allclose(res.time, np.atleast_2d(desired))
    npt.assert_allclose(filt.state, [[[[2, 0]]]])

    sos = np.array([[[1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0]]])
    filt = fo.FilterSOS(
        sos, impulse.sampling_rate,  state=[[[[0, 0], [0, 0]]]])
    res = filt.process(impulse)
    state = filt.state
    desired = np.array([[[[1, 0], [0, 0]]]])
    npt.assert_allclose(state, desired)


def test_filter_sos_init_state(impulse):
    coeff = np.array([[1, 0, 0, 1, 1, 0]])
    filt = fo.FilterSOS(coeff, impulse.sampling_rate)

    # init empty filter
    filt.init_state(impulse.cshape, state='zeros')
    desired = np.array([[[[0, 0]]]])
    npt.assert_allclose(filt.state, desired)

    # init with step function response
    filt.init_state(impulse.cshape, state='step')
    desired = np.array([[[[-0.5, 0]]]])
    npt.assert_allclose(filt.state, desired)

    # init with step function response multichannel
    # sos states are saved as (*fshape, *cshape, n_sections, 2)
    n_sections = 1
    filt.init_state((2, 3), state='zeros')
    desired = np.zeros((1, 2, 3, n_sections, 2), dtype=float)
    npt.assert_allclose(filt.state, desired)

    # init with step function response multichannel
    filt.init_state((2, 3), state='step')
    desired = np.zeros((1, 2, 3, n_sections, 2), dtype=float)
    desired[..., 0] = -0.5
    npt.assert_allclose(filt.state, desired)


def test_filter_sos_process_multi_dim_filt(impulse):
    sos = np.array([
        [[1, 1/2, 0, 1, 0, 0]],
        [[1, 1/4, 0, 1, 0, 0]]])
    coeff = np.array([
        [[1, 1/2, 0], [1, 0, 0]],
        [[1, 1/4, 0], [1, 0, 0]]])
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[:, 0, :3], coeff[:, 0])

    impulse.time = np.vstack((impulse.time, impulse.time))
    filt = fo.FilterSOS(sos, impulse.sampling_rate)
    res = filt.process(impulse)

    npt.assert_allclose(res.time[0, 0, :3], coeff[0, 0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 0, :3], coeff[1, 0, :], atol=1e-16)

    npt.assert_allclose(res.time[0, 1, :3], coeff[0, 0, :], atol=1e-16)
    npt.assert_allclose(res.time[1, 1, :3], coeff[1, 0, :], atol=1e-16)


@pytest.mark.parametrize('Filter', [
    (fo.FilterFIR([[1, -1]], 44100)),
    (fo.FilterIIR([[1, -1], [1, 0]], 44100)),
    (fo.FilterSOS([[[1, -1, 0, 1, 0, 0]]], 44100))])
def test_blockwise_processing(Filter):

    # test signal
    signal = pf.Signal([1, 2, 3, 4, 5, 6], 44100)

    # filter entire signal
    complete = Filter.process(signal, reset=True)

    # filter in two blocks with correct handling of the state
    Filter.init_state(signal.cshape, 'zeros')
    block_a = Filter.process(pf.Signal(signal.time[0, :3], 44100), reset=False)
    block_b = Filter.process(pf.Signal(signal.time[0, 3:], 44100), reset=False)
    # outputs have to be identical in this case
    npt.assert_array_equal(np.atleast_2d(complete.time[0, :3]), block_a.time)
    npt.assert_array_equal(np.atleast_2d(complete.time[0, 3:]), block_b.time)


def test_blockwise_processing_with_coefficients_exchange():
    # input signal
    input_data = pf.Signal([1, 2, 3, 4, 0], 44100)

    coefficients_1 = [2, -2]
    coefficients_2 = [2, -1.9]

    # time variant filtering using Filter object
    # initialize filter and state
    filterObj = pf.FilterFIR([coefficients_1], 44100)
    filterObj.init_state(input_data.cshape, state='zeros')
    # process first block
    filter_1 = filterObj.process(pf.Signal(input_data.time[..., :2], 44100))
    # update filter coefficients
    filterObj.coefficients = [coefficients_2]
    # process second block
    filter_2 = filterObj.process(pf.Signal(input_data.time[..., 2:], 44100))

    # overlap and add time variant filtering
    # first block
    ola_1 = np.convolve([1, 2], coefficients_1)
    # second block
    ola_2 = np.convolve([3, 4], coefficients_2)
    ola_2[0] += ola_1[-1]

    # check equality
    npt.assert_allclose(filter_1.time.flatten(), ola_1[:2])
    npt.assert_allclose(filter_2.time.flatten(), ola_2)


def test_atleast_3d_first_dim():
    arr = np.array([1, 0, 0])
    desired = np.array([[[1, 0, 0]]])

    arr_3d = fo._atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)
    arr = np.array([[1, 0, 0], [2, 2, 2]])

    desired = np.array([[[1, 0, 0], [2, 2, 2]]])
    arr_3d = fo._atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)

    arr = np.ones((2, 3, 5))
    desired = arr.copy()
    arr_3d = fo._atleast_3d_first_dim(arr)
    npt.assert_array_equal(arr_3d, desired)


def test_extend_sos_coefficients():
    sos = np.array([
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
    ])

    actual = fo._extend_sos_coefficients(sos, 2)
    npt.assert_allclose(actual, sos)

    expected = np.array([
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
    ])

    actual = fo._extend_sos_coefficients(sos, 4)
    npt.assert_allclose(actual, expected)

    # test if the extended filter has an ideal impulse response.
    imp = np.zeros(512)
    imp[0] = 1
    imp_filt = spsignal.sosfilt(actual, imp)
    npt.assert_allclose(imp_filt, imp)


def test___eq___equal(filterObject):
    actual = filterObject.copy()
    assert filterObject == actual


def test___eq___notEqual(filterObject, coeffs, state):
    actual = fo.Filter(coefficients=2 * coeffs, state=state)
    assert not filterObject == actual
    actual = fo.Filter(coefficients=coeffs, state=2 * state)
    assert not filterObject == actual
    actual = filterObject.copy()
    actual.comment = f'{actual.comment} A completely different thing'
    assert not filterObject == actual


def test_repr(capfd):
    """Test the repr string of the filter classes."""

    print(fo.FilterFIR([[1, 0, 1]], 44100))
    out, _ = capfd.readouterr()
    assert out == \
        "2nd order FIR filter with 1 channel @ 44100 Hz sampling rate\n"

    print(fo.FilterIIR([[1, 0, 1], [1, 0, 0]], 44100))
    out, _ = capfd.readouterr()
    assert out == \
        "2nd order IIR filter with 1 channel @ 44100 Hz sampling rate\n"

    print(fo.FilterSOS([[[1, 0, 0, 1, 0, 0]]], 44100))
    out, _ = capfd.readouterr()
    assert out == \
        "SOS filter with 1 section and 1 channel @ 44100 Hz sampling rate\n"
