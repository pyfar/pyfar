import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import FrequencyData


def test_data_frequency_init_with_defaults():
    """
    Test to init without optional parameters.
    Test getter for domain, freq, frequencies, and n_bins.
    """
    data = [1, 0, -1]
    freqs = [0, .1, .3]

    freq = FrequencyData(data, freqs)
    assert isinstance(freq, FrequencyData)
    npt.assert_allclose(freq.freq, np.atleast_2d(np.asarray(data)))
    npt.assert_allclose(freq.frequencies, np.atleast_1d(np.asarray(freqs)))
    assert freq.n_bins == 3
    assert freq.domain == 'freq'


def test_data_frequency_init_wrong_number_of_freqs():
    """Test if entering a wrong number of frequencies raises an assertion."""
    data = [1, 0, -1]
    freqs = [0, .1]

    match = 'Number of frequency values does not match the number'
    with pytest.raises(ValueError, match=match):
        FrequencyData(data, freqs)


def test_data_frequency_with_non_monotonously_increasing_frequencies():
    """Test if non monotnously increasing frequencies raises an assertion."""
    data = [1, 0, -1]
    freqs = [0, .2, .1]

    match = 'Frequencies must be monotonously increasing'
    with pytest.raises(ValueError, match=match):
        FrequencyData(data, freqs)


def test_data_frequency_init_dtype():
    """
    Test casting and assertions of dtype (also test freq setter because
    it is called during initialization).
    """

    # integer to float casting
    data = FrequencyData([1, 2, 3], [1, 2, 3])
    assert data.freq.dtype.kind == "f"

    # float
    data = FrequencyData([1., 2., 3.], [1, 2, 3])
    assert data.freq.dtype.kind == "f"

    # complex
    data = FrequencyData([1+1j, 2+2j, 3+3j], [1, 2, 3])
    assert data.freq.dtype.kind == "c"

    # object array
    with pytest.raises(TypeError, match="int, uint, float, or complex"):
        FrequencyData(["1", "2", "3"], [1, 2, 3])


def test_data_frequency_setter_freq():
    """Test the setter for the frequency data."""
    data_a = [1, 0, -1]
    data_b = [2, 0, -2]
    freqs = [0, .1, .3]

    freq = FrequencyData(data_a, freqs)
    freq.freq = data_b
    npt.assert_allclose(freq.freq, np.atleast_2d(np.asarray(data_b)))

    with pytest.raises(ValueError, match="Number of frequency values"):
        freq.freq = 1


def test_reshape():

    # test reshape with tuple
    rng = np.random.default_rng()
    x = rng.random((6, 256))
    data_in = FrequencyData(x, range(256))
    data_out = data_in.reshape((3, 2))
    npt.assert_allclose(data_in._data.reshape(3, 2, -1), data_out._data)
    assert id(data_in) != id(data_out)

    data_out = data_in.reshape((3, -1))
    npt.assert_allclose(data_in._data.reshape(3, 2, -1), data_out._data)
    assert id(data_in) != id(data_out)

    # test reshape with int
    rng = np.random.default_rng()
    x = rng.random((3, 2, 256))
    data_in = FrequencyData(x, range(256))
    data_out = data_in.reshape(6)
    npt.assert_allclose(data_in._data.reshape(6, -1), data_out._data)
    assert id(data_in) != id(data_out)


def test_reshape_exceptions():
    rng = np.random.default_rng()
    x = rng.random((6, 256))
    data_in = FrequencyData(x, range(256))
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
    x = rng.random((6, 2, 5, 256))
    signal_in = FrequencyData(x, range(256))
    signal_out = signal_in.transpose()
    npt.assert_allclose(signal_in.T._data, signal_out._data)
    npt.assert_allclose(
        signal_in._data.transpose(2, 1, 0, 3), signal_out._data)


@pytest.mark.parametrize('taxis', [(2, 0, 1), (-1, 0, -2)])
def test_transpose_args(taxis):
    rng = np.random.default_rng()
    x = rng.random((6, 2, 5, 256))
    signal_in = FrequencyData(x, range(256))
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
    data_in = FrequencyData(x, range(256))
    data_out = data_in.flatten()

    npt.assert_allclose(data_in._data, data_out._data)
    assert id(data_in) != id(data_out)

    # test 3D signal
    rng = np.random.default_rng()
    x = rng.random((3, 2, 256))
    data_in = FrequencyData(x, range(256))
    data_out = data_in.flatten()

    npt.assert_allclose(data_in._data.reshape((6, -1)), data_out._data)
    assert id(data_in) != id(data_out)


def test_data_frequency_find_nearest():
    """Test the find nearest function for a single number and list entry."""
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    freq = FrequencyData(data, freqs)

    # test for a single number
    idx = freq.find_nearest_frequency(.15)
    assert idx == 1

    # test for a list
    idx = freq.find_nearest_frequency([.15, .4])
    npt.assert_allclose(idx, np.asarray([1, 2]))


def test_magic_getitem_slice():
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([[1, 0, -1], [2, 0, -2]])
    freqs = [0, .1, .3]
    freq = FrequencyData(data, freqs)
    npt.assert_allclose(FrequencyData(data[0], freqs)._data, freq[0]._data)


def test_magic_getitem_error():
    """
    Test if indexing that would return a subset of the frequency bins raises a
    key error.
    """
    freq = pf.FrequencyData([[0, 0, 0], [1, 1, 1]], [0, 1, 3])
    # manually indexing too many dimensions
    with pytest.raises(IndexError, match='Indexed dimensions must not exceed'):
        freq[0, 1]
    # indexing too many dimensions with ellipsis operator
    with pytest.raises(IndexError, match='Indexed dimensions must not exceed'):
        freq[0, 0, ..., 1]


def test_magic_setitem():
    """Test the setitem for FrequencyData."""
    freqs = [0, .1, .3]

    freq_a = FrequencyData([[1, 0, -1], [1, 0, -1]], freqs)
    freq_b = FrequencyData([2, 0, -2], freqs)
    freq_a[0] = freq_b

    npt.assert_allclose(freq_a.freq, np.asarray([[2, 0, -2], [1, 0, -1]]))


def test_magic_setitem_wrong_n_bins():
    """Test the setitem for FrequencyData with wrong number of bins."""

    freq_a = FrequencyData([1, 0, -1], [0, .1, .3])
    freq_b = FrequencyData([2, 0, -2, 0], [0, .1, .3, .7])

    match = 'The number of frequency bins does not match'
    with pytest.raises(ValueError, match=match):
        freq_a[0] = freq_b


@pytest.mark.parametrize("audio", [
    pf.TimeData([1, 2], [1, 2]), pf.Signal([1, 2], 44100)])
def test_magic_setitem_wrong_type(audio):
    frequency_data = FrequencyData([1, 2, 3, 4], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="Comparison only valid"):
        frequency_data[0] = audio


def test_separation_from_time_data():
    """Check if attributes from FrequencyData are really not available."""
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    freq = FrequencyData(data, freqs)

    with pytest.raises(AttributeError):
        assert freq.time
    with pytest.raises(AttributeError):
        assert freq.times
    with pytest.raises(AttributeError):
        assert freq.n_samples
    with pytest.raises(AttributeError):
        assert freq.signal_length
    with pytest.raises(AttributeError):
        assert freq.find_nearest_time


def test_separation_from_signal():
    """Check if attributes from Signal are really not available."""
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    freq = FrequencyData(data, freqs)

    with pytest.raises(AttributeError):
        assert freq.sampling_rate
    with pytest.raises(AttributeError):
        freq.domain = 'freq'


def test___eq___equal():
    """Check if copied FrequencyData is equal."""
    frequency_data = FrequencyData([1, 2, 3], [1, 2, 3])
    actual = frequency_data.copy()
    assert frequency_data == actual


def test___eq___notEqual():
    """Check if FrequencyData is equal."""
    frequency_data = FrequencyData([1, 2, 3], [1, 2, 3])
    actual = FrequencyData([2, 3, 4], [1, 2, 3])
    assert not frequency_data == actual
    actual = FrequencyData([1, 2, 3], [2, 3, 4])
    assert not frequency_data == actual
    comment = f'{frequency_data.comment} A completely different thing'
    actual = FrequencyData([1, 2, 3], [1, 2, 3], comment=comment)
    assert not frequency_data == actual


def test__repr__(capfd):
    """Test string representation."""
    print(FrequencyData([1, 2, 3], [1, 2, 3]))
    out, _ = capfd.readouterr()
    assert ("FrequencyData:\n"
            "(1,) channels with 3 frequencies") in out
