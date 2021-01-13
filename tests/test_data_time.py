import numpy as np
import numpy.testing as npt
import pytest
from pyfar.signal import DataTime as DataTime


def test_data_time_init_with_defaults():
    """
    Test to init without optional parameters.
    Test getter for domain, time, times, length, and n_samples.
    """
    data = [1, 0, -1]
    times = [0, .1, .3]

    time = DataTime(data, times)
    assert isinstance(time, DataTime)
    npt.assert_allclose(time.time, np.atleast_2d(np.asarray(data)))
    npt.assert_allclose(time.times, np.atleast_1d(np.asarray(times)))
    assert time.signal_length == .3
    assert time.n_samples == 3
    assert time.domain == 'time'


def test_data_time_init_wrong_number_of_times():
    """Test if entering a wrong number of times raises an assertion."""
    data = [1, 0, -1]
    times = [0, .1]

    with pytest.raises(ValueError):
        DataTime(data, times)


def test_data_time_setter_time():
    """Test the setter for the time data."""
    data_a = [1, 0, -1]
    data_b = [2, 0, -2]
    times = [0, .1, .3]

    time = DataTime(data_a, times)
    time.time = data_b
    npt.assert_allclose(time.time, np.atleast_2d(np.asarray(data_b)))


def test_data_time_find_nearest():
    """Test the find nearest function for a single number and list entry."""
    data = [1, 0, -1]
    times = [0, .1, .3]
    time = DataTime(data, times)

    # test for a single number
    idx = time.find_nearest_time(.15)
    assert idx == 1

    # test for a list
    idx = time.find_nearest_time([.15, .4])
    npt.assert_allclose(idx, np.asarray([1, 2]))


def test_separation_from_data_frequency():
    """Check if attributes from DataFrequency are really not available."""
    data = [1, 0, -1]
    times = [0, .1, .3]
    time = DataTime(data, times)

    with pytest.raises(AttributeError):
        time.freq
    with pytest.raises(AttributeError):
        time.frequencies
    with pytest.raises(AttributeError):
        time.n_bins
    with pytest.raises(AttributeError):
        time.find_nearest_frequency


def test_separation_from_signal():
    """Check if attributes from Signal are really not available."""
    data = [1, 0, -1]
    times = [0, .1, .3]
    time = DataTime(data, times)

    with pytest.raises(AttributeError):
        time.sampling_rate
    with pytest.raises(AttributeError):
        time.domain = 'time'
