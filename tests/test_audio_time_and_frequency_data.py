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