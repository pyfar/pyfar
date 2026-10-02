import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
from pyfar import FrequencyData


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
