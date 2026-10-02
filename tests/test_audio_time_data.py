import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import TimeData


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
