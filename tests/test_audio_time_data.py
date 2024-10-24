import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import TimeData


def test_data_time_init_with_defaults():
    """
    Test to init without optional parameters.
    Test getter for domain, time, times, length, and n_samples.
    """
    data = [1, 0, -1]
    times = [0, .1, .3]

    time = TimeData(data, times)
    assert isinstance(time, TimeData)
    npt.assert_allclose(time.time, np.atleast_2d(np.asarray(data)))
    npt.assert_allclose(time.times, np.atleast_1d(np.asarray(times)))
    assert time.signal_length == .3
    assert time.n_samples == 3
    assert time.domain == 'time'
    assert not time.complex


def test_data_time_init_wrong_dtype():
    """
    Test assertion from non integer/float data (also test time setter because
    it is called during initialization).
    """
    with pytest.raises(ValueError, match="time data is complex"):
        TimeData(np.arange(2).astype(complex), [0, 1])

    # pass array of invalid type
    with pytest.raises(TypeError, match="int, uint, float, or complex"):
        TimeData(['1', '2'], [0, 1])


def test_time_init_complex_flag():
    """
    Test assertion from non boolean complex flag.
    """
    complex_flag = 1
    with pytest.raises(TypeError, match="``is_complex`` flag is "
                                        f"{type(complex_flag).__name__}"
                                        f"but must be a boolean"):
        TimeData(np.arange(2).astype(complex), [0, 1], is_complex=complex_flag)


def test_data_time_init_wrong_number_of_times():
    """Test if entering a wrong number of times raises an assertion."""
    data = [1, 0, -1]
    times = [0, .1]

    match = 'The length of times must be data.shape'
    with pytest.raises(ValueError, match=match):
        TimeData(data, times)


def test_data_time_with_non_monotonously_increasing_time():
    """Test if non monotnously increasing of times raises an assertion."""
    data = [1, 0, -1]
    times = [0, .2, .1]

    match = 'Times must be monotonously increasing'
    with pytest.raises(ValueError, match=match):
        TimeData(data, times)


def test_data_time_setter_time():
    """Test the setter for the time data."""
    data_a = [1, 0, -1]
    data_b = [2, 0, -2]
    times = [0, .1, .3]

    time = TimeData(data_a, times)
    time.time = data_b
    npt.assert_allclose(time.time, np.atleast_2d(np.asarray(data_b)))


def test_data_time_setter_complex_casting():
    time = TimeData(data=[1, 0, -1], times=[0, .1, .3], is_complex=True)
    assert time.time.dtype.kind == "c"


def test_setter_complex():
    # test setting complex flag from False to True
    time = TimeData(data=[1, 0, -1], times=[0, .1, .3])
    time.complex = True
    assert time.time.dtype.kind == "c"

    # test setting complex flag from True to False
    time = TimeData(data=[1, 0, -1], times=[0, .1, .3], is_complex=True)
    time.complex = False
    assert time.time.dtype.kind == "f"

    time = TimeData(data=[1 + 1j, 0 + 1j, -1 + 2j], times=[0, .1, .3],
                    is_complex=True)
    with pytest.raises(ValueError, match="Signal has complex-valued time data"
                                         " is_complex flag cannot be `False`"):
        time.complex = False


def test_reshape():

    # test reshape with tuple
    rng = np.random.default_rng()
    rand = rng.random((6, 256))
    data_in = TimeData(rand, range(256))
    data_out = data_in.reshape((3, 2))
    npt.assert_allclose(data_in._data.reshape(3, 2, -1), data_out._data)
    assert id(data_in) != id(data_out)

    data_out = data_in.reshape((3, -1))
    npt.assert_allclose(data_in._data.reshape(3, 2, -1), data_out._data)
    assert id(data_in) != id(data_out)

    # test reshape with int
    rng = np.random.default_rng()
    rand = rng.random((3, 2, 256))
    data_in = TimeData(rand, range(256))
    data_out = data_in.reshape(6)
    npt.assert_allclose(data_in._data.reshape(6, -1), data_out._data)
    assert id(data_in) != id(data_out)


def test_reshape_exceptions():
    rng = np.random.default_rng()
    rand = rng.random((6, 256))
    data_in = TimeData(rand, range(256))
    data_out = data_in.reshape((3, 2))
    npt.assert_allclose(data_in._data.reshape(3, 2, -1), data_out._data)
    # test assertion for non-tuple input
    match = 'newshape must be an integer or tuple'
    with pytest.raises(ValueError, match=match):
        data_out = data_in.reshape([3, 2])

    # test assertion for wrong dimension
    with pytest.raises(ValueError, match='Cannot reshape audio object'):
        data_out = data_in.reshape((3, 4))


def test_transpose():
    rng = np.random.default_rng()
    rand = rng.random((6, 2, 5, 256))
    signal_in = TimeData(rand, range(256))
    signal_out = signal_in.transpose()
    npt.assert_allclose(signal_in.T._data, signal_out._data)
    npt.assert_allclose(
        signal_in._data.transpose(2, 1, 0, 3), signal_out._data)


@pytest.mark.parametrize('taxis', [(2, 0, 1), (-1, 0, -2)])
def test_transpose_args(taxis):
    rng = np.random.default_rng()
    rand = rng.random((6, 2, 5, 256))
    signal_in = TimeData(rand, range(256))
    signal_out = signal_in.transpose(taxis)
    npt.assert_allclose(
        signal_in._data.transpose(2, 0, 1, 3), signal_out._data)
    signal_out = signal_in.transpose(*taxis)
    npt.assert_allclose(
        signal_in._data.transpose(2, 0, 1, 3), signal_out._data)


def test_flatten():

    # test 2D signal (flatten should not change anything)
    rng = np.random.default_rng()
    x = rng.random((2, 256))
    data_in = TimeData(x, range(256))
    data_out = data_in.flatten()

    npt.assert_allclose(data_in._data, data_out._data)
    assert id(data_in) != id(data_out)

    # test 3D signal
    rng = np.random.default_rng()
    x = rng.random((3, 2, 256))
    data_in = TimeData(x, range(256))
    data_out = data_in.flatten()

    npt.assert_allclose(data_in._data.reshape((6, -1)), data_out._data)
    assert id(data_in) != id(data_out)


def test_data_time_find_nearest():
    """Test the find nearest function for a single number and list entry."""
    data = [1, 0, -1]
    times = [0, .1, .3]
    time = TimeData(data, times)

    # test for a single number
    idx = time.find_nearest_time(.15)
    assert idx == 1

    # test for a list
    idx = time.find_nearest_time([.15, .4])
    npt.assert_allclose(idx, np.asarray([1, 2]))


def test_magic_getitem_slice():
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([[1, 0, -1], [2, 0, -2]])
    times = [0, .1, .3]
    time = TimeData(data, times)
    npt.assert_allclose(TimeData(data[0], times)._data, time[0]._data)


def test_magic_getitem_complex():
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([[1, 0, -1], [2, 0, -2]], dtype=complex)
    times = [0, .1, .3]
    time = TimeData(data, times, is_complex=True)
    assert time[0].complex
    npt.assert_allclose(TimeData(data[0], times, is_complex=True)._data,
                        time[0]._data)


def test_magic_setitem():
    """Test the setimtem for TimeData."""
    times = [0, .1, .3]

    time_a = TimeData([[1, 0, -1], [1, 0, -1]], times)
    time_b = TimeData([2, 0, -2], times)
    time_a[0] = time_b

    npt.assert_allclose(time_a.time, np.asarray([[2, 0, -2], [1, 0, -1]]))


def test_magic_getitem_error():
    """
    Test if indexing that would return a subset of the frequency bins raises a
    key error.
    """
    time = pf.TimeData([[0, 0, 0], [1, 1, 1]], [0, 1, 3])
    # manually indexing too many dimensions
    with pytest.raises(IndexError, match='Indexed dimensions must not exceed'):
        time[0, 1]
    # indexing too many dimensions with ellipsis operator
    with pytest.raises(IndexError, match='Indexed dimensions must not exceed'):
        time[0, 0, ..., 1]


def test_magic_setitem_wrong_n_samples():
    """Test the setimtem for TimeData with wrong number of samples."""

    time_a = TimeData([1, 0, -1], [0, .1, .3])
    time_b = TimeData([2, 0, -2, 0], [0, .1, .3, .7])
    match = 'The number of samples does not match'
    with pytest.raises(ValueError, match=match):
        time_a[0] = time_b


@pytest.mark.parametrize("audio", [
    pf.FrequencyData([1, 2], [1, 2]), pf.Signal([1, 2], 44100)])
def test_magic_setitem_wrong_type(audio):
    time_data = TimeData([1, 2, 3, 4], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="Comparison only valid"):
        time_data[0] = audio


def test_separation_from_data_frequency():
    """Check if attributes from DataFrequency are really not available."""
    data = [1, 0, -1]
    times = [0, .1, .3]
    time = TimeData(data, times)

    with pytest.raises(AttributeError):
        assert time.freq
    with pytest.raises(AttributeError):
        assert time.frequencies
    with pytest.raises(AttributeError):
        assert time.n_bins
    with pytest.raises(AttributeError):
        assert time.find_nearest_frequency


def test_separation_from_signal():
    """Check if attributes from Signal are really not available."""
    data = [1, 0, -1]
    times = [0, .1, .3]
    time = TimeData(data, times)

    with pytest.raises(AttributeError):
        assert time.sampling_rate
    with pytest.raises(AttributeError):
        time.domain = 'time'


def test___eq___equal():
    """Check if copied TimeData is equal."""
    time_data = TimeData([1, 2, 3], [0.1, 0.2, 0.3])
    actual = time_data.copy()
    assert time_data == actual


def test___eq___notEqual():
    """Check if TimeData object is equal."""
    time_data = TimeData([1, 2, 3], [0.1, 0.2, 0.3])
    actual = TimeData([2, 3, 4], [0.1, 0.2, 0.3])
    assert not time_data == actual
    actual = TimeData([1, 2, 3], [0.2, 0.3, 0.4])
    assert not time_data == actual
    comment = f'{time_data.comment} A completely different thing'
    actual = TimeData([1, 2, 3], [0.1, 0.2, 0.3], comment=comment)
    assert not time_data == actual


def test__repr__(capfd):
    """Test string representation."""
    print(TimeData([1, 2, 3], [1, 2, 3]))
    out, _ = capfd.readouterr()
    assert ("TimeData:\n"
            "(1,) channels with 3 samples") in out
