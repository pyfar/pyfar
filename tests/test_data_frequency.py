import numpy as np
import numpy.testing as npt
import pytest
from pyfar.signal import DataFrequency as DataFrequency


def test_data_frequency_init_with_defaults():
    """
    Test to init without optional parameters.
    Test getter for domain, freq, frequencies, and n_bins.
    """
    data = [1, 0, -1]
    freqs = [0, .1, .3]

    freq = DataFrequency(data, freqs)
    assert isinstance(freq, DataFrequency)
    npt.assert_allclose(freq.freq, np.atleast_2d(np.asarray(data)))
    npt.assert_allclose(freq.frequencies, np.atleast_1d(np.asarray(freqs)))
    assert freq.n_bins == 3
    assert freq.domain == 'freq'


def test_data_frequency_init_wrong_number_of_freqs():
    """Test if entering a wrong number of frequencies raises an assertion."""
    data = [1, 0, -1]
    freqs = [0, .1]

    with pytest.raises(ValueError):
        DataFrequency(data, freqs)


def test_data_frequency_setter_freq():
    """Test the setter for the frequency data."""
    data_a = [1, 0, -1]
    data_b = [2, 0, -2]
    freqs = [0, .1, .3]

    freq = DataFrequency(data_a, freqs)
    freq.freq = data_b
    npt.assert_allclose(freq.freq, np.atleast_2d(np.asarray(data_b)))


def test_getter_fft_norm():
    data = [1, 0, -1]
    freqs = [0, .1, .3]

    freq = DataFrequency(data, freqs, fft_norm='psd')
    assert freq.fft_norm == 'psd'


def test_data_frequency_find_nearest():
    """Test the find nearest function for a single number and list entry."""
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    freq = DataFrequency(data, freqs)

    # test for a single number
    idx = freq.find_nearest_frequency(.15)
    assert idx == 1

    # test for a list
    idx = freq.find_nearest_frequency([.15, .4])
    npt.assert_allclose(idx, np.asarray([1, 2]))


def test_separation_from_data_frequency():
    """Check if attributes from DataFrequency are really not available."""
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    freq = DataFrequency(data, freqs)

    with pytest.raises(AttributeError):
        freq.time
    with pytest.raises(AttributeError):
        freq.times
    with pytest.raises(AttributeError):
        freq.n_samples
    with pytest.raises(AttributeError):
        freq.signal_length
    with pytest.raises(AttributeError):
        freq.find_nearest_time


def test_separation_from_signal():
    """Check if attributes from Signal are really not available."""
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    freq = DataFrequency(data, freqs)

    with pytest.raises(AttributeError):
        freq.sampling_rate
    with pytest.raises(AttributeError):
        freq.domain = 'freq'
    with pytest.raises(AttributeError):
        freq.fft_norm = 'amplitude'
