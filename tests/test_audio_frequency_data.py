import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import FrequencyData


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
