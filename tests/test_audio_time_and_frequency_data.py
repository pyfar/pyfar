import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf


def test_time_data_init_with_defaults():
    """Test the initialization without optional parameters for TimeData."""
    data = [1, 0, -1]
    times = [0, .1, .3]

    signal = pf.TimeData(data, times)
    assert isinstance(signal, pf.TimeData)
    npt.assert_allclose(signal.time, np.atleast_2d(np.asarray(data)))
    npt.assert_allclose(signal.times, np.atleast_1d(np.asarray(times)))
    assert signal.signal_length == .3
    assert signal.n_samples == 3
    assert signal.domain == 'time'
    assert not signal.complex


def test_frequency_data_init_with_defaults():
    """
    Test the initialization without optional parameters for FrequencyData.
    """
    data = [1, 0, -1]
    freqs = [0, .1, .3]

    signal = pf.FrequencyData(data, freqs)
    assert isinstance(signal, pf.FrequencyData)
    npt.assert_allclose(signal.freq, np.atleast_2d(np.asarray(data)))
    npt.assert_allclose(
        signal.frequencies, np.atleast_1d(np.asarray(freqs)))
    assert signal.n_bins == 3
    assert signal.domain == 'freq'


@pytest.mark.parametrize(("data_type", "match"), [
    (pf.TimeData, 'The length of times must be data.shape'),
    (pf.FrequencyData, 'Number of frequency values')])
def test_init_wrong_number_of_times_freqs(data_type, match):
    """
    Test that entering a wrong number of times/frequencies raises a ValueError.
    """
    data = [1, 0, -1]
    times_freqs = [0, .1]

    with pytest.raises(ValueError, match=match):
        data_type(data, times_freqs)


@pytest.mark.parametrize(("data_type", "match"), [
    (pf.TimeData, 'Times must be monotonously increasing'),
    (pf.FrequencyData, 'Frequencies must be monotonously increasing')])
def test_with_non_monotonically_increasing_freqs_times(data_type, match):
    """
    Test that non-monotonically increasing times/frequencies raise
    a ValueError.
    """
    data = [1, 0, -1]
    freqs_times = [0, .2, .1]

    with pytest.raises(ValueError, match=match):
        data_type(data, freqs_times)


@pytest.mark.parametrize(("data", "is_complex", "dtype"), [
    ([1, 2, 3], False, "f"),
    ([1., 2., 3.], False, "f"),
    ([1, 2, 3], True, "c"),
    ([1+1j, 2+2j, 3+3j], True, "c")])
def test_time_data_init_dtype(data, is_complex, dtype):
    """Test TimeData dtype casting."""
    signal = pf.TimeData(data, [0, .1, .3], is_complex=is_complex)
    assert signal.time.dtype.kind == dtype


@pytest.mark.parametrize(("data", "dtype"), [
    ([1, 2, 3], "f"),
    ([1., 2., 3.], "f"),
    ([1+1j, 2+2j, 3+3j], "c")])
def test_frequency_data_init_dtype(data, dtype):
    """Test FrequencyData dtype casting."""
    signal = pf.FrequencyData(data, [0, .1, .3])
    assert signal.freq.dtype.kind == dtype


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
def test_init_dtype_type_error(data_type):
    """Test TypeError for invalid time/frequency data."""
    with pytest.raises(TypeError, match="int, uint, float, or complex"):
        data_type(['1', '2'], [0, 1])


@pytest.mark.parametrize("data", [
    np.arange(2).astype(complex),
    [1+1j, 2+2j]])
def test_time_data_init_dtype_value_error(data):
    """Test ValueError for invalid time data."""
    with pytest.raises(ValueError, match="time data is complex,"
                       " set is_complex flag"):
        pf.TimeData(data, [0, 1])


@pytest.mark.parametrize(("data", "is_complex", "complex_flag", "dtype"), [
    ([1, 0, -1], False, True, "c"),
    ([1, 0, -1], True, False, "f"),
    ([1+1j, 2+2j, 3+3j], True, True, "c")])
def test_time_data_complex_flag_setter(data, is_complex, complex_flag, dtype):
    """Test the setter for the complex flag of TimeData."""
    time_data = pf.TimeData(data, times=[0, .1, .3], is_complex=is_complex)
    time_data.complex = complex_flag
    assert time_data.complex == complex_flag
    assert time_data.time.dtype.kind == dtype
    npt.assert_allclose(time_data.time, np.atleast_2d(data))


def test_time_data_complex_flag_value_error():
    """Test ValueError for invalid complex flag when data is complex."""
    time_data = pf.TimeData(data = [1+1j, 0+1j, -1+2j], times=[0, .1, .3],
                       is_complex=True)
    with pytest.raises(ValueError, match="Signal has complex-valued time data"
                                         " is_complex flag cannot be `False`"):
        time_data.complex = False


def test_time_data_complex_flag_type_error():
    """Test TypeError for invalid complex flag."""
    with pytest.raises(TypeError, match="but must be a boolean"):
        pf.TimeData(np.arange(2).astype(complex), [0, 1], is_complex=1)


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
def test_setter_time_freq(data_type):
    """Test the setter for the time/frequency data."""
    data_a = [1, 0, -1]
    data_b = [2, 0, -2]

    signal = data_type(data_a, [0, .1, .3])
    setattr(signal, signal.domain, data_b)
    npt.assert_allclose(getattr(signal, signal.domain),
                         np.atleast_2d(np.asarray(data_b)))


@pytest.mark.parametrize(("data_type", "match"), [
    pytest.param(pf.TimeData, "...", marks=pytest.mark.xfail(reason="The"
    " ValueError is not yet implemented in the file pyfar/classes/audio.py.")),
    (pf.FrequencyData, 'Number of frequency values')])
def test_setter_wrong_length_error(data_type, match):
    """Test that setting invalid number of time/freq raises a ValueError."""
    signal = data_type([1, 0, -1], [0, .1, .3])
    with pytest.raises(ValueError, match=match):
        setattr(signal, signal.domain, 1)


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
@pytest.mark.parametrize(("input_shape", "reshape_arg", "expected_shape"), [
    ((6, 256), (3, 2), (3, 2, -1)),
    ((6, 256), (3, -1), (3, 2, -1)),
    ((3, 2, 256), 6, (6, -1))])
def test_reshape(data_type, input_shape, reshape_arg, expected_shape):
    """Test the reshape method with tuple and int arguments."""
    rng = np.random.default_rng(seed=1111)
    x = rng.random(input_shape)
    signal_in = data_type(x, range(256))
    signal_out = signal_in.reshape(reshape_arg)
    signal_in_data = getattr(signal_in, signal_in.domain)
    signal_out_data = getattr(signal_out, signal_out.domain)

    npt.assert_allclose(
        signal_in_data.reshape(expected_shape), signal_out_data)
    assert id(signal_in) != id(signal_out)


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
@pytest.mark.parametrize(("new_shape", "match"), [
    ([3, 2], 'newshape must be an integer or tuple'),
    ((3, 4), 'Cannot reshape audio object')])
def test_reshape_exceptions(data_type, new_shape, match):
    """Test error handling of the reshape method with invalid arguments."""
    rng = np.random.default_rng(seed=1111)
    x = rng.random((6, 256))
    signal_in = data_type(x, range(256))
    with pytest.raises(ValueError, match=match):
        signal_in.reshape(new_shape)


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
def test_transpose(data_type):
    """Test the transpose method for TimeData and FrequencyData."""
    rng = np.random.default_rng(seed=1111)
    x = rng.random((6, 2, 5, 256))
    signal_in = data_type(x, range(256))
    signal_out = np.transpose(signal_in)
    expected = getattr(signal_out, signal_out.domain)
    npt.assert_allclose(
        getattr(signal_in.T, signal_in.domain), expected)
    npt.assert_allclose(
        getattr(signal_in, signal_in.domain).transpose(2, 1, 0, 3),
        expected)


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
@pytest.mark.parametrize('taxis', [(2, 0, 1), (-1, 0, -2)])
@pytest.mark.parametrize('unpacking', [False, True])
def test_transpose_args(data_type, taxis, unpacking):
    """Test the transpose method with argument unpacking."""
    rng = np.random.default_rng(seed=1111)
    x = rng.random((6, 2, 5, 256))
    signal_in = data_type(x, range(256))
    if unpacking:
        signal_out = signal_in.transpose(*taxis)
    else:
        signal_out = signal_in.transpose(taxis)
    npt.assert_allclose(
        getattr(signal_in, signal_in.domain).transpose(2, 0, 1, 3),
        getattr(signal_out, signal_out.domain))


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
@pytest.mark.parametrize(("input_shape", "expected_shape"), [
    ((2, 256), (2, -1)),
    ((3, 2, 256), (6, -1))])
def test_flatten(data_type, input_shape, expected_shape):
    """Test the flatten method for 2D and 3D input shapes."""
    rng = np.random.default_rng(seed=1111)
    x = rng.random(input_shape)
    signal_in = data_type(x, range(256))
    signal_out = signal_in.flatten()
    signal_in_data = getattr(signal_in, signal_in.domain)
    signal_out_data = getattr(signal_out, signal_out.domain)

    npt.assert_allclose(signal_in_data.reshape(expected_shape),
                        signal_out_data)
    assert id(signal_in) != id(signal_out)


@pytest.mark.parametrize(("data_type", "method"), [
    (pf.TimeData, "find_nearest_time"),
    (pf.FrequencyData, "find_nearest_frequency")])
@pytest.mark.parametrize(("value", "expected_idx"), [
    (.15, 1),
    ([.15, .4], [1, 2])])
def test_find_nearest_time_frequency(data_type, method, value, expected_idx):
    """
    Test that find_nearest_time/frequency returns the index of the
    closest value.
    """
    signal = data_type([1, 0, -1], [0, .1, .3])
    idx = getattr(signal, method)(value)
    npt.assert_array_equal(idx, expected_idx)
    assert np.ndim(idx) == np.ndim(expected_idx)


@pytest.mark.parametrize("data_type", [pf.TimeData, pf.FrequencyData])
@pytest.mark.parametrize("index", [0, -1, slice(None), slice(0, 1), (0, 1)])
def test_magic_getitem_slicing_indexing(data_type, index):
    """Test indexing and slicing of the channel dimensions."""
    rng = np.random.default_rng(seed=1111)
    data = rng.random((6, 2, 5, 256))
    signal = data_type(data, range(256))

    expected = np.atleast_2d((getattr(signal, signal.domain))[index])
    actual = getattr(signal[index], signal.domain)

    npt.assert_allclose(actual, expected)
