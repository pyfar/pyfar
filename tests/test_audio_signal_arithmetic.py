import numpy as np
import numpy.testing as npt
import pytest

import pyfar as pf
import pyfar.classes.audio as signal
from pyfar import Signal, TimeData, FrequencyData
import operator

SIGNAL1 = Signal([1, 2, 3, 4], 44100)
SIGNAL2 = Signal([1, 2, 3, 4], 48000)
SIGNAL3 = Signal([1, 2, 3], 44100)
SIGNAL4 = Signal([1, 2, 3, 4], 44100, fft_norm="rms")
SIGNAL5 = Signal([1+1j, 2+2j, 3+3j, 4+4j], 48000, is_complex=True)
SIGNAL6 = FrequencyData([1+1j, 2+2j, 3+3j, 4+4j], [10, 200, 1000, 20000])


@pytest.mark.parametrize(("domain", "x_complex", "y_complex", "desired"), [
    ('time', False, False, np.atleast_2d([2, 0, 0])),
    ('time', True, True, np.atleast_2d([2 + 0j, 0, 0])),
    ('time', True, False, np.atleast_2d([2 + 0j, 0 + 0j, 0 + 0j])),
    ('freq', False, False, np.atleast_2d([2, 2])),
    ('freq', True, True, np.atleast_2d([2 + 0j, 2 + 0j, 2 + 0j])),
    ('freq', True, False, np.atleast_2d([2 + 0j, 2 + 0j, 2 + 0j]))])
def test_add_two_signals_time_and_freq(domain, x_complex, y_complex, desired):
    x = Signal([1, 0, 0], 44100, is_complex=x_complex)
    y = Signal([1, 0, 0], 44100, is_complex=y_complex)
    z = pf.add((x, y), domain)

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)
    npt.assert_allclose(y.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    if domain == 'time':
        npt.assert_allclose(z.time, desired, atol=1e-15)
    else:
        npt.assert_allclose(z.freq, desired, atol=1e-15)

    assert isinstance(z, Signal)
    assert z.domain == domain
    assert z.complex == (x_complex or y_complex)


# test adding three signals
def test_add_three_signals():
    # generate and add signals
    x = Signal([1, 0, 0], 44100)
    y = pf.add((x, x, x), 'time')

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([3, 0, 0]), atol=1e-15)


@pytest.mark.parametrize(("y", "swap", "domain", "is_complex_x",
                           "is_complex_y", "desired"), [
    (1, False, 'time', False, False, np.atleast_2d([2, 1, 1])),
    (1 + 1j, False, 'time', False, True,
     np.atleast_2d([2 + 1j, 1 + 1j, 1 + 1j])),
    (1, True, 'time', False, False, np.atleast_2d([2, 1, 1])),
    (1, True, 'time', True, False, np.atleast_2d([2 + 0j, 1 + 0j, 1 + 0j])),
    (1, True, 'freq', True, False, np.atleast_2d([2 + 0j, 2 + 0j, 2 + 0j]))])
def test_add_signal_and_number(y, swap, domain, is_complex_x, is_complex_y,
                                desired):
    x = Signal([1, 0, 0], 44100, is_complex=is_complex_x)
    z = pf.add((y, x), domain) if swap else pf.add((x, y), domain)

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    if domain == 'time':
        npt.assert_allclose(z.time, desired, atol=1e-15)
    else:
        npt.assert_allclose(z.freq, desired, atol=1e-15)

    assert isinstance(z, Signal)
    assert z.domain == domain
    assert z.complex == (is_complex_x or is_complex_y)


@pytest.mark.parametrize(("x", "y", "domain", "desired_data",
                           "desired_instances"), [
    (TimeData([1, 0, 0], [0, .1, .5]), 1, 'time', np.atleast_2d([2, 1, 1]),
     np.atleast_1d([0, .1, .5])),
    (TimeData([1, 0, 0], [0, .1, .5]), TimeData([1, 0, 0], [0, .1, .5]),
     'time', np.atleast_2d([2, 0, 0]), np.atleast_1d([0, .1, .5])),
    (FrequencyData([1, 0, 0], [0, .1, .5]), 1, 'freq',
     np.atleast_2d([2, 1, 1]), np.atleast_1d([0, .1, .5])),
    (FrequencyData([1, 0, 0], [0, .1, .5]),
     FrequencyData([1, 0, 0], [0, .1, .5]), 'freq',
     np.atleast_2d([2, 0, 0]), np.atleast_1d([0, .1, .5]))])
def test_add_time_frequency_data(x, y, domain, desired_data,
                                 desired_instances):
    z = pf.add((x, y), domain)

    if domain == "time":
        assert isinstance(z, TimeData)
        x_data, x_instances = x.time, x.times
        z_data, z_instances = z.time, z.times
    else:
        assert isinstance(z, FrequencyData)
        x_data, x_instances = x.freq, x.frequencies
        z_data, z_instances = z.freq, z.frequencies

    npt.assert_allclose(x_data, np.atleast_2d([1, 0, 0]), atol=1e-15)
    npt.assert_allclose(x_instances, np.atleast_1d([0, .1, .5]), atol=1e-15)
    # check result
    npt.assert_allclose(z_data, desired_data, atol=1e-15)
    npt.assert_allclose(z_instances, desired_instances, atol=1e-15)


@pytest.mark.parametrize(("x", "y", "domain", "match"), [
    (TimeData([1, 0, 0], [0, .1, .5]), 1, 'freq',
     "The domain must be 'time'."),
    (TimeData([1, 0, 0], [0, .1, .5]), TimeData([1, 0, 0], [0, .1, .4]),
     'time', 'The times does not match.'),
    (FrequencyData([1, 0, 0], [0, .1, .5]), 1, 'time',
     "The domain must be 'freq'."),
    (FrequencyData([1, 0, 0], [0, .1, .5]),
     FrequencyData([1, 0, 0], [0, .1, .4]), 'freq',
     'The frequencies do not match.')])
def test_add_time_data_frequency_data_errors(x, y, domain, match):
    with pytest.raises(ValueError, match=match):
        pf.add((x, y), domain)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("x", [
    (np.arange(2 * 3 * 4).reshape((2, 3, 4))),
    (np.arange(3 * 4).reshape((3, 4)))])
def test_add_array_and_signal(x, swap):
    y = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    # shapes match
    z = pf.add((y, x)) if swap else pf.add((x, y))
    npt.assert_allclose(
        z.freq, np.ones_like(z.freq)*x[..., None] + 1, atol=1e-15)


def test_add_arrays():
    # With broadcasting
    x = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    y = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    z = pf.add((x, y))
    npt.assert_allclose(
        z, x + y, atol=1e-15)


@pytest.mark.parametrize('fft_norm', ['none', 'rms'])
def test_signal_inversion(fft_norm):
    """Test signal inversion with different FFT norms."""
    signal = pf.Signal([2, 0, 0], 44100, fft_norm=fft_norm)
    signal_inv = 1 / signal
    npt.assert_allclose(signal.time.flatten(), [2, 0, 0])
    npt.assert_allclose(signal_inv.time.flatten(), [.5, 0, 0])


def test_subtraction():
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100)
    y = Signal([0, 1, 0], 44100)
    z = pf.subtract((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([1, -1, 0]), atol=1e-15)


@pytest.mark.parametrize(('domain', 'desired'), [
    ("time", np.atleast_2d([0, 0, 0])),
    ("freq",  np.atleast_2d([1+0j, -0.5-0.8660254j]))])
def test_real_multiplication(domain, desired):
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100)
    y = Signal([0, 1, 0], 44100)
    z = pf.multiply((x, y), domain)

    # check result
    if domain == "time":
        npt.assert_allclose(z.time, desired, atol=1e-15)
    else:
        npt.assert_allclose(z.freq, desired, atol=1e-15)


@pytest.mark.parametrize(('domain', 'desired'), [
    ("time", np.atleast_2d([0 + 0j, 0 + 0j, 0 + 0j])),
    ("freq", np.atleast_2d([-0.5+0.8660254j,  1+0j, -0.5-0.8660254j]))])
@pytest.mark.parametrize(('is_complex_x', 'is_complex_y'), [
    (True, True), (False, True), (True, False)])
def test_complex_multiplication(domain, desired, is_complex_x, is_complex_y):
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100, is_complex=is_complex_x)
    y = Signal([0, 1, 0], 44100, is_complex=is_complex_y)
    z = pf.multiply((x, y), domain)

    # check result
    if domain == "time":
        npt.assert_allclose(z.time, desired, atol=1e-15)
    else:
        npt.assert_allclose(z.freq, desired, atol=1e-15)


@pytest.mark.parametrize('is_complex_y', [False, True])
def test_division(is_complex_y):
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100)
    y = Signal([2, 2, 2], 44100, is_complex=is_complex_y)
    z = pf.divide((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([0.5, 0, 0]), atol=1e-15)
    assert z.complex == is_complex_y


def test_power():
    # only test one case - everything else is tested below
    x = Signal([2, 1, 0], 44100)
    y = Signal([2, 2, 2], 44100)
    z = pf.power((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([4, 1, 0]), atol=1e-15)


@pytest.mark.parametrize(('x','y'),[
    (Signal([3, 2, 1], 44100, n_samples=5, domain='freq'),
     Signal([2, 2, 2], 44100, n_samples=5, domain='freq')),
    (Signal([3, 2, 1], 44100, n_samples=5, domain='freq'), 2),
    (TimeData([3, 2, 1], [0, 1, 2]),
     TimeData([2, 2, 2], [0, 1, 2])),
    (TimeData([3, 2, 1], [0, 1, 2]), 2),
    (FrequencyData([3, 2, 1], [0, 1, 2]),
     FrequencyData([2, 2, 2], [0, 1, 2])),
    (FrequencyData([3, 2, 1], [0, 1, 2]), 2)])
@pytest.mark.parametrize(("swap", "op", "desired"), [
    (False, operator.add, [5, 4, 3]),
    (False, operator.sub, [1, 0, -1]),
    (False, operator.mul, [6, 4, 2]),
    (False, operator.truediv, [1.5, 1, .5]),
    (False, operator.pow, [9, 4, 1]),
    (True, operator.add, [5, 4, 3]),
    (True, operator.sub, [-1, 0, 1]),
    (True, operator.mul, [6, 4, 2]),
    (True, operator.truediv, [2/3, 1, 2]),
    (True, operator.pow, [8, 4, 2])])
def test_overloaded_operators_signal_time_freq_data(x, y, swap, op, desired):
    z = op(y, x) if swap else op(x, y)
    if z.domain == 'time':
        npt.assert_allclose(z.time, np.array(desired, ndmin=2), atol=1e-15)
    else:
        npt.assert_allclose(z.freq, np.array(desired, ndmin=2), atol=1e-15)


@pytest.mark.parametrize(("swap", "op"), [
    (False, operator.add),
    (False, operator.sub),
    (False, operator.mul),
    (False, operator.truediv),
    (False, operator.pow),
    (True, operator.add),
    (True, operator.sub),
    (True, operator.mul),
    (True, operator.truediv),
    (True, operator.pow)])
def test_overloaded_operators_array_and_signal(swap, op):
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4) + 1
    y = Signal(np.ones((2, 3, 4, 5)), 44100, n_samples=8, domain='freq')

    n = np.broadcast_to(np.arange(1, 25).reshape(2, 3, 4, 1), (2, 3, 4, 5))
    desired = op(1, n) if swap else op(n, 1)
    z = op(y, x) if swap else op(x, y)

    npt.assert_allclose(z.freq, desired, atol=1e-15)


@pytest.mark.parametrize(("data", "domain", "is_complex"), [
    ((SIGNAL5, SIGNAL5), 'time', True),
    ((SIGNAL1, SIGNAL1), 'time', False),
    ((SIGNAL5, SIGNAL2), 'time', True),
    ((SIGNAL2, SIGNAL5), 'time', True),
    ((1 + 1j, SIGNAL5), 'time', True),
    ((SIGNAL5, 1 + 1j), 'time', True),
    ((1, SIGNAL1), 'time', False),
    ((SIGNAL1, 1), 'time', False),
    ((1, SIGNAL6), 'freq', False),
    ((SIGNAL6, 1), 'freq', False),
    ((1 + 1j, SIGNAL6), 'freq', False),
    ((SIGNAL6, 1 + 1j), 'freq', False)])
def test_assert_match_for_arithmetic_complex_flag(data, domain, is_complex):
    out = signal._assert_match_for_arithmetic(
        data, domain, division=False, matmul=False)
    assert out[7] == is_complex


@pytest.mark.parametrize(("data", "domain", "match"), [
    (SIGNAL1, 'time',"Input argument 'data' must be a tuple."),
    ((SIGNAL1, ['str', 'ing']), 'time',
     "Input must be of type Signal, int, float, or complex"),
    ((SIGNAL1, SIGNAL2), 'time', 'The sampling rates do not match'),
    ((SIGNAL1, SIGNAL3), 'time', 'The number of samples does not match')])
def test_assert_match_for_arithmetic_errors(data, domain,match):
    with pytest.raises(ValueError, match=match):
        signal._assert_match_for_arithmetic(
            data, domain, division=False, matmul=False)


@pytest.mark.parametrize("data", [
    (SIGNAL1, SIGNAL1),
    (SIGNAL1, [1, 2]),
    (SIGNAL1, SIGNAL1, SIGNAL1)])
def test_assert_match_for_arithmetic(data):
    signal._assert_match_for_arithmetic(data, 'time', division=False,
                                         matmul=False)


@pytest.mark.parametrize(("data", "index", "expected"), [
    ((SIGNAL1, SIGNAL1), [0,1,2,6,7], [44100, 4, 'none', (1,), False]),
    ((SIGNAL1, SIGNAL4), [0,1,2,6,7], [44100, 4, 'rms', (1,), False])])
def test_assert_match_for_arithmetic_output(data, index, expected):
    out = signal._assert_match_for_arithmetic(
       data, 'time', division=False, matmul=False)
    for exp_ind, ind in enumerate(index):
        assert out[ind] == expected[exp_ind]


def test_get_arithmetic_data_with_array():
    data_in = np.asarray(1)
    data_out = signal._get_arithmetic_data(
        data_in, None, (1,), False, type(None), contains_complex=False)
    npt.assert_allclose(data_in, data_out)


@pytest.mark.parametrize("domain", ["time", "freq"])
# all possible combinations of `domain`, `signal_type`, and `fft_norm`
@pytest.mark.parametrize("meta", [
        ['time', 'none'],
        ['freq', 'none'],
        ['time', 'unitary'],
        ['freq', 'unitary'],
        ['time', 'amplitude'],
        ['freq', 'amplitude'],
        ['time', 'rms'],
        ['freq', 'rms'],
        ['time', 'power'],
        ['freq', 'power'],
        ['time', 'psd'],
        ['freq', 'psd']])
def test_get_arithmetic_data_with_signal(domain, meta):
    # reference signal - _get_arithmetic_data should return the data without
    # any normalization regardless of the input data
    s_ref = Signal([1, 0, 0], 44100)
    m_in = meta

    # create input signal with current domain, type, and norm
    s_in = Signal([1, 0, 0], 44100, fft_norm=m_in[1])
    s_in.domain = m_in[0]
    # get output data
    data_out = signal._get_arithmetic_data(
        s_in, domain=domain, cshape=(1,), matmul=False,
        audio_type=Signal, contains_complex=False)
    if domain == 'time':
        npt.assert_allclose(s_ref.time, data_out, atol=1e-15)
    elif domain == 'freq':
        npt.assert_allclose(s_ref.freq, data_out, atol=1e-15)


def test_get_arithmetic_data_with_signal_complex_casting():
    s_in = Signal([1, 0, 0], 44100, is_complex=False)

    data_out = signal._get_arithmetic_data(
        s_in, 'time', (1,), False, type(None), contains_complex=True)

    assert data_out.dtype == 'complex'


def test_assert_match_for_arithmetic_data_different_audio_classes():
    match = 'The audio objects do not match.'
    with pytest.raises(ValueError, match=match):
        signal._assert_match_for_arithmetic(
            (Signal(1, 1), TimeData(1, 1)), 'time', division=False,
            matmul=False)


def test_assert_match_for_arithmetic_data_wrong_domain():
    match = 'domain must be time or freq but is space.'
    with pytest.raises(ValueError, match=match):
        signal._assert_match_for_arithmetic(
            (1, 1), 'space', division=False, matmul=False)


def test_assert_match_for_arithmetic_data_wrong_cshape():
    x = Signal(np.ones((2, 3, 4)), 44100)
    y = Signal(np.ones((5, 4)), 44100)
    with pytest.raises(ValueError, match="The cshapes"):
        signal._assert_match_for_arithmetic(
            (x, y), 'freq', division=False, matmul=False)


def test_get_arithmetic_data_wrong_domain():
    match = "domain must be 'time' or 'freq' but found space"
    with pytest.raises(ValueError, match=match):
        signal._get_arithmetic_data(
            Signal(1, 44100), 'space', (1,), False, Signal,
            contains_complex=False)


def test_array_broadcasting_dimension_error():
    x = np.arange(2 * 3 * 4 * 10).reshape((2, 3, 4, 10))
    y = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="array dimension"):
        pf.add((x, y), domain='time')


def test_array_broadcasting_shape_error():
    x = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    y = pf.signals.impulse(10, amplitude=np.ones((2, 3, 5)))
    match = 'operands could not be broadcast together with shapes'
    with pytest.raises(ValueError, match=match):
        pf.add((x, y))


def test_matrix_multiplication_default():
    """Test default behavior for signals."""
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = pf.signals.impulse(10, amplitude=np.array([[1, 2], [3, 4], [5, 6]]))
    z = pf.matrix_multiplication((x, y))
    desired = np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 6))
    npt.assert_allclose(z.freq, desired, atol=1e-15)


def test_matrix_multiplication_time_domain():
    """Time domain multiplication for signals."""
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = pf.signals.impulse(10, amplitude=np.array([[1, 2], [3, 4], [5, 6]]))
    z = pf.matrix_multiplication((x, y), domain='time')
    desired = np.zeros((2, 2, 10))
    desired[..., 0] = np.array([[22, 28], [49, 64]])
    npt.assert_allclose(z.time, desired, atol=1e-15)


@pytest.mark.parametrize(('swap', 'desired'),
    [(False, np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 6))),
     (True, np.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]])[..., None]*
                                                        np.ones((3, 3, 6)))])
def test_matrix_multiplication_operator(swap, desired):
    """Test overloaded @ operator."""
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = pf.signals.impulse(10, amplitude=np.array([[1, 2], [3, 4], [5, 6]]))
    z = y @ x if swap else x @ y
    npt.assert_allclose(z.freq, desired, atol=1e-15)


def test_matrix_multiplication_higher_shape():
    """Test correct multiplication nd signals."""
    x = pf.signals.impulse(10, amplitude=np.ones((2, 3, 4)))
    y = pf.signals.impulse(10, amplitude=np.ones((2, 4, 5)))
    z = pf.matrix_multiplication((x, y))
    desired = 4 * np.ones((2, 3, 5, 6))
    npt.assert_allclose(z.freq, desired, atol=1e-15)


def test_matrix_multiplication_shape_mismatch():
    """Test error for shape mismatch."""
    # Signals
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = pf.signals.impulse(10, amplitude=np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="matmul: Input operand 1"):
        pf.matrix_multiplication((x, y))
    # Signal and array
    y = np.ones((2, 2, 6)) * np.array([[1, 2], [3, 4]])[..., None]
    with pytest.raises(ValueError, match="matmul: Input operand 1"):
        pf.matrix_multiplication((x, y))


@pytest.mark.parametrize(("pf_class", "swap", "desired"), [
    (pf.TimeData, False,
     np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 10))),
    (pf.TimeData, True,
     np.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]])[..., None]*
     np.ones((3, 3, 10))),
    (pf.FrequencyData, False,
     np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 10))),
    (pf.FrequencyData, True,
     np.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]])[..., None]*
     np.ones((3, 3, 10)))])
def test_matrix_multiplication_TimeData_FrequencyData(pf_class, swap, desired):
    """Test @ operator for TimeData and FrequencyData."""
    times = np.arange(10)
    xdata = np.ones((2, 3, 10)) * np.array([[1, 2, 3], [4, 5, 6]])[..., None]
    ydata = np.ones((3, 2, 10)) * np.array([[1, 2], [3, 4], [5, 6]])[..., None]
    x = pf_class(xdata, times)
    y = pf_class(ydata, times)
    z = y @ x if swap else x @ y
    if z.domain == 'time':
        npt.assert_allclose(z.time, desired, atol=1e-15)
    else:
        npt.assert_allclose(z.freq, desired, atol=1e-15)


def test_matrix_multiplication_frequency_axis():
    """Test frequency dependent matrix explicitly."""
    freqs = np.arange(3)
    xdata = np.array([[[1, 2, 3], [4, 5, 6]]])
    ydata = np.array([[[1, 2, 3]], [[4, 5, 6]]])
    x = pf.FrequencyData(xdata, freqs)
    y = pf.FrequencyData(ydata, freqs)
    z = x @ y
    desired = np.array([[[17, 29, 45]]])
    npt.assert_allclose(z.freq, desired, atol=1e-15)
    assert isinstance(z, pf.FrequencyData)


def test_matrix_multiplication_signal_times_array():
    """Test multiplication of signal with array."""
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = np.ones((3, 2)) * np.array([[1, 2], [3, 4], [5, 6]])
    z = x @ y
    desired = np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 6))
    npt.assert_allclose(z.freq, desired, atol=1e-15)
    z = y @ x
    desired = np.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]])[..., None] \
        * np.ones((3, 3, 6))
    npt.assert_allclose(z.freq, desired, atol=1e-15)


@pytest.mark.parametrize(("pf_class", "swap", "desired"), [
    (pf.TimeData, False,
     np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 10))),
    (pf.TimeData, True,
     np.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]])[..., None]*
     np.ones((3, 3, 10))),
    (pf.FrequencyData, False,
     np.array([[22, 28], [49, 64]])[..., None] * np.ones((2, 2, 10))),
    (pf.FrequencyData, True,
     np.array([[9, 12, 15], [19, 26, 33], [29, 40, 51]])[..., None]*
     np.ones((3, 3, 10)))])
def test_matrix_multiplication_TimeData_FrequencyData_times_array(pf_class,
                                                                swap, desired):
    """Test multiplication of TimeData and FrequencyData with array."""
    times = np.arange(10)
    xdata = np.ones((2, 3, 10)) * np.array([[1, 2, 3], [4, 5, 6]])[..., None]
    x = pf_class(xdata, times)
    y = np.ones((3, 2)) * np.array([[1, 2], [3, 4], [5, 6]])
    z = y @ x if swap else x @ y
    if z.domain == 'time':
        npt.assert_allclose(z.time, desired, atol=1e-15)
    else:
        npt.assert_allclose(z.freq, desired, atol=1e-15)


def test_matrix_multiplication_axes():
    """Test axes parameter."""
    a = np.arange(2 * 3 * 5).reshape((2, 3, 5))
    b = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    x = pf.signals.impulse(10, amplitude=a)
    y = pf.signals.impulse(10, amplitude=b)
    z = pf.matrix_multiplication((x, y), axes=[(0, 1), (0, 1), (0, 1)])
    des = np.matmul(a, b, axes=[(0, 1), (0, 1), (0, 1)])[..., None] \
        * np.ones((2, 4, 5, 6))
    npt.assert_allclose(z.freq, des, atol=1e-15)


@pytest.mark.parametrize(('sx', 'sy', 'az', 'sz'), [
    ((1, 3, 5), (3, 5, 4), 5, (3, 3, 4)),
    ((2,), (3, 2, 4), 2, (3, 1, 4)),
    ((1, 2), (3, 2, 4), 2, (3, 1, 4)),
    ((2, 3, 4), (4,), 4, (2, 3, 1)),
    ((2, 3, 4), (4, 1), 4, (2, 3, 1))])
def test_matrix_multiplication_broadcasting(sx, sy, az, sz):
    """Test broadcasting."""
    x = pf.signals.impulse(10, amplitude=np.ones(sx))
    y = pf.signals.impulse(10, amplitude=np.ones(sy))
    z = pf.matrix_multiplication((x, y))
    des = az * np.ones(sz + (6,))
    npt.assert_allclose(z.freq, des, atol=1e-15)


def test_matrix_multiplication_multiple():
    """Test 3 arguments in data."""
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    c = np.ones((4, 5))
    x = pf.signals.impulse(10, amplitude=a)
    y = pf.signals.impulse(10, amplitude=b)
    z = pf.signals.impulse(10, amplitude=c)
    res = pf.matrix_multiplication((x, y, z))
    des = 12 * np.ones((2, 5, 6))
    npt.assert_allclose(res.freq, des, atol=1e-15)


@pytest.mark.parametrize(
    'x', [np.ones((2, 3)), pf.signals.impulse(10, amplitude=np.ones((2, 3)))])
@pytest.mark.parametrize(
    'y', [np.ones((3, 4)), pf.signals.impulse(10, amplitude=np.ones((3, 4)))])
@pytest.mark.parametrize(
    'z', [np.ones((4, 5)), pf.signals.impulse(10, amplitude=np.ones((4, 5)))])
def test_matrix_multiplication_multiple_arrays(x, y, z):
    """Test 2 arrays in 3 arguments."""
    if any(type(a) in (Signal, TimeData, FrequencyData) for a in [x, y, z]):
        des = 12 * np.ones((2, 5, 6))
        npt.assert_allclose(
            pf.matrix_multiplication((x, y, z)).freq, des, atol=1e-15)
    else:
        des = 12 * np.ones((2, 5))
        npt.assert_allclose(
            pf.matrix_multiplication((x, y, z)), des, atol=1e-15)


def test_matrix_multiplication_array_mismatch_errors():
    """Test errors for multiplication of signal with array."""
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = np.ones((3, 2, 1)) * np.array([[1, 2], [3, 4], [5, 6]])[..., None]
    with pytest.raises(ValueError, match='matmul'):
        x @ y
    with pytest.raises(ValueError, match='matmul'):
        y @ x


def test_matrix_multiplication_undocumented():
    """Test undesired, but not restricted multiplication along time axis."""
    x = pf.signals.impulse(10, amplitude=np.array([[1, 2, 3], [4, 5, 6]]))
    y = np.ones((3, 2, 10)) * np.array([[1, 2], [3, 4], [5, 6]])[..., None]
    pf.matrix_multiplication(
        (x, y), domain='time', axes=[(-2, -1), (-3, -2), (-2, -1)])


@pytest.mark.parametrize('audio_object', [
    pf.Signal([1, -1, 1], 1, fft_norm='none'),
    pf.Signal([1, -1, 1], 1, fft_norm='rms'),
    pf.FrequencyData([1, -1, 1], [0, 1, 3]),
    pf.TimeData([1, -1, 1], [1, 2, 3])])
@pytest.mark.parametrize('operation', [
    pf.add, pf.subtract, pf.multiply, pf.divide, pf.power])
def test_audio_object_and_number(audio_object, operation):
    """
    Test if arithmetic operations work regardless of the fft norm and
    audio object type if only one audio object is involved.
    """

    domain = 'time' if type(audio_object) is pf.TimeData else 'freq'

    result = operation((1, audio_object), domain=domain)
    assert type(result) is type(audio_object)

    result = operation((audio_object, 1), domain=domain)
    assert type(result) is type(audio_object)
