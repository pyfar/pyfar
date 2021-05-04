import pytest
import numpy as np
import numpy.testing as npt

from pyfar import Signal, TimeData, FrequencyData
from pyfar.testing import stub_utils


def test_signal_stub_properties():
    """ Test comparing properties of Signal stub
    with actual Signal implementation.
    """
    time = np.ones((1, 1024))
    freq = np.ones((1, 1024))
    sampling_rate = 44100
    fft_norm = 'none'

    signal_stub = stub_utils.signal_stub(time, freq, sampling_rate, fft_norm)
    stub_dir = dir(signal_stub)
    signal_dir = dir(Signal(time, sampling_rate))

    assert stub_dir.sort() == signal_dir.sort()


def test_signal_stub_times():
    """ Test times vector of Signal stub.
    """
    time = np.ones((1, 4))
    freq = np.ones((1, 3))
    sampling_rate = 1
    fft_norm = 'none'
    times = np.array([0., 1., 2., 3.])

    signal_stub = stub_utils.signal_stub(time, freq, sampling_rate, fft_norm)

    npt.assert_allclose(signal_stub.times, times, rtol=1e-10)


def test_signal_stub_frequencies_even():
    """ Test frequencies vector of Signal stub,
    even number of samples.
    """
    time = np.ones((1, 4))
    freq = np.ones((1, 3))
    sampling_rate = 1
    fft_norm = 'none'
    frequencies = np.array([0., 0.25, 0.5])

    signal_stub = stub_utils.signal_stub(time, freq, sampling_rate, fft_norm)

    npt.assert_allclose(signal_stub.frequencies, frequencies, rtol=1e-10)


def test_signal_stub_frequencies_odd():
    """ Test frequencies vector of Signal stub,
    odd number of samples.
    """
    time = np.ones((1, 5))
    freq = np.ones((1, 3))
    sampling_rate = 1
    fft_norm = 'none'
    frequencies = np.array([0., 0.2, 0.4])

    signal_stub = stub_utils.signal_stub(time, freq, sampling_rate, fft_norm)

    npt.assert_allclose(signal_stub.frequencies, frequencies, rtol=1e-10)


def test_time_data_stub_properties():
    """ Test comparing properties of TimeData stub
    with actual TimeData implementation.
    """
    time = [1, 0, -1]
    times = [0, .1, .4]

    time_data_stub = stub_utils.time_data_stub(time, times)
    stub_dir = dir(time_data_stub)
    time_data_dir = dir(TimeData(time, times))

    assert stub_dir.sort() == time_data_dir.sort()


def test_time_data_stub_data():
    """Test the data contained in the TimeData stub."""

    time = [1, 0, -1]
    times = [0, .1, .4]

    time_data_stub = stub_utils.time_data_stub(time, times)

    npt.assert_allclose(time_data_stub.time, np.atleast_2d(time))
    npt.assert_allclose(time_data_stub.times, np.atleast_1d(times))


def test_time_data_stub_slice():
    """Test slicing the TimeData stub."""

    time = [[1, 0, -1], [2, 0, -2]]
    times = [0, .1, .4]

    time_data_stub = stub_utils.time_data_stub(time, times)
    stub_slice = time_data_stub[0]

    npt.assert_allclose(stub_slice.time, np.atleast_2d(time[0]))
    npt.assert_allclose(stub_slice.times, np.atleast_1d(times))


def test_frequency_data_stub_properties():
    """ Test comparing properties of FrequencyData stub
    with actual FrequencyData implementation.
    """
    freq = [1, 0, -1]
    frequencies = [0, .1, .4]

    frequency_data_stub = stub_utils.frequency_data_stub(freq, frequencies)
    stub_dir = dir(frequency_data_stub)
    frequency_data_dir = dir(FrequencyData(freq, frequencies))

    assert stub_dir.sort() == frequency_data_dir.sort()


def test_frequency_data_stub_data():
    """Test the data contained in the FrequencyData stub."""

    freq = [1, 0, -1]
    frequencies = [0, .1, .4]

    frequency_data_stub = stub_utils.frequency_data_stub(freq, frequencies)

    npt.assert_allclose(frequency_data_stub.freq, np.atleast_2d(freq))
    npt.assert_allclose(
        frequency_data_stub.frequencies, np.atleast_1d(frequencies))


def test_frequency_data_stub_slice():
    """Test slicing the FrequencyData stub."""

    freq = [[1, 0, -1], [2, 0, -2]]
    frequencies = [0, .1, .4]

    frequency_data_stub = stub_utils.frequency_data_stub(freq, frequencies)
    stub_slice = frequency_data_stub[0]

    npt.assert_allclose(stub_slice.freq, np.atleast_2d(freq[0]))
    npt.assert_allclose(stub_slice.frequencies, np.atleast_1d(frequencies))


def test_impulse_func_single_channel():
    """Test generation of delta impulse, single channel.
    """
    n_samples = 4
    fft_norm = 'none'
    cshape = (1,)
    delay = 0
    time_truth = np.array([[1, 0, 0, 0]])
    freq_truth = np.array([[1, 1, 1]], dtype=complex)

    time, freq = stub_utils.impulse_func(delay, n_samples, fft_norm, cshape)

    npt.assert_allclose(time, time_truth, rtol=1e-10)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)


def test_impulse_func_rms():
    """Test generation of delta impulse,
    RMS FFT normalization.
    """
    n_samples = 6
    cshape = (1,)
    delay = 0
    fft_norm = 'rms'
    freq_truth = np.array([[1, np.sqrt(2), np.sqrt(2), 1]], dtype=complex)
    freq_truth /= n_samples
    _, freq = stub_utils.impulse_func(delay, n_samples, fft_norm, cshape)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10, atol=1e-15)


def test_impulse_func_value_error():
    """Test generation of delta impulse, value errors.
    """
    n_samples = 4
    fft_norm = 'none'
    # Delay too large
    cshape = (1,)
    delay = n_samples
    with pytest.raises(ValueError):
        stub_utils.impulse_func(delay, n_samples, fft_norm, cshape)
    # Inconsistent input shape
    cshape = (2, 2)
    delay = [1, 1]
    with pytest.raises(ValueError):
        stub_utils.impulse_func(delay, n_samples, fft_norm, cshape)


def test_impulse_func_multi_channel():
    """Test generation of delta impulse, multiple channels.
    """
    n_samples = 4
    fft_norm = 'none'
    cshape = (2, 2)
    delay = np.array([[0, 1], [2, 3]])
    time_truth = np.array([[[1, 0, 0, 0], [0, 1, 0, 0]],
                           [[0, 0, 1, 0], [0, 0, 0, 1]]])
    freq_truth = np.array([[[1, 1, 1],
                            [1, -1j, -1]],
                           [[1, -1, 1],
                            [1, 1j, -1]]])

    time, freq = stub_utils.impulse_func(delay, n_samples, fft_norm, cshape)

    npt.assert_allclose(time, time_truth, rtol=1e-10)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)


def test_normalization_none():
    """ Test unitary FFT normalization implemented in stubs_utils.py"""
    n_samples = 4
    freq = np.array([1, 1, 1], dtype=complex)
    freq_norm = stub_utils._normalization(freq, n_samples, 'none')
    npt.assert_allclose(freq, freq_norm, rtol=1e-10)


def test_normalization_rms_even():
    """ Test RMS FFT normalization implemented in stubs_utils.py,
    even number of samples."""
    n_samples = 4
    freq = np.array([1, 1, 1], dtype=complex)
    freq_norm_truth = np.array([1, np.sqrt(2), 1], dtype=complex)
    freq_norm_truth /= n_samples
    freq_norm = stub_utils._normalization(freq, n_samples, 'rms')
    npt.assert_allclose(freq_norm, freq_norm_truth, rtol=1e-10)


def test_normalization_rms_odd():
    """ Test RMS FFT normalization implemented in stubs_utils.py,
    odd number of samples."""
    n_samples = 5
    freq = np.array([1, 1, 1], dtype=complex)
    freq_norm_truth = np.array([1, np.sqrt(2), np.sqrt(2)], dtype=complex)
    freq_norm_truth /= n_samples
    freq_norm = stub_utils._normalization(freq, n_samples, 'rms')
    npt.assert_allclose(freq_norm, freq_norm_truth, rtol=1e-10)


def test_sine_func():
    """Test generation of sine data, single channel.
    """
    n_samples = 4
    sampling_rate = 4
    fft_norm = 'none'
    cshape = (1,)

    frequency_truth = 1
    time_truth = np.array([[0, 1, 0, -1]])
    freq_truth = np.array([[0, -2.j, 0]], dtype=complex)

    time, freq, frequency = stub_utils.sine_func(
        frequency_truth,
        sampling_rate,
        n_samples,
        fft_norm,
        cshape)

    npt.assert_allclose(
        time, time_truth, rtol=1e-10, atol=10 * np.finfo(float).eps)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)
    assert frequency == frequency_truth


def test_sine_func_rms():
    """Test generation of sine,
    RMS FFT normalization.
    """
    n_samples = 4
    sampling_rate = 4
    fft_norm = 'rms'
    cshape = (1,)

    frequency_truth = 1
    freq_truth = np.array(
        [[0, -2j * np.sqrt(2) / n_samples, 0]], dtype=complex)

    _, freq, _ = stub_utils.sine_func(
        frequency_truth,
        sampling_rate,
        n_samples,
        fft_norm,
        cshape)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)


def test_sine_func_multi_channel():
    """Test generation of sine data, multiple channels.
    """
    sampling_rate = 12
    n_samples = 12
    fft_norm = 'none'
    cshape = (2, 2)

    frequency_truth = np.array([[1, 2], [3, 4]])
    sq3_2 = np.sqrt(3) / 2
    time_truth = np.array(
        [[[0, 0.5, sq3_2, 1, sq3_2, 0.5,
           0, -0.5, -sq3_2, -1, -sq3_2, -0.5],
          [0, sq3_2, sq3_2, 0, -sq3_2, -sq3_2,
           0, sq3_2, sq3_2, 0, -sq3_2, -sq3_2]],
         [[0, 1, 0, -1, 0, 1,
          0, -1, 0, 1, 0, -1],
          [0, sq3_2, -sq3_2, 0, sq3_2, -sq3_2,
           0, sq3_2, -sq3_2, 0, sq3_2, -sq3_2]]])
    freq_truth = np.array(
        [[[0, -6j, 0, 0, 0, 0, 0],
          [0, 0, -6j, 0, 0, 0, 0]],
         [[0, 0, 0, -6j, 0, 0, 0],
          [0, 0, 0, 0, -6j, 0, 0]]],
        dtype=complex)

    time, freq, frequency = stub_utils.sine_func(
        frequency_truth,
        sampling_rate,
        n_samples,
        fft_norm,
        cshape)

    npt.assert_allclose(
        time, time_truth, rtol=1e-10, atol=10 * np.finfo(float).eps)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)
    npt.assert_allclose(frequency, frequency_truth, rtol=1e-10)


def test_sine_func_frequency_adjustment():
    """Test generation of sine data, adjusted frequency.
    """
    n_samples = 4
    sampling_rate = 4
    fft_norm = 'none'
    cshape = (1,)
    frequency_in = 1.5

    frequency_truth = 1.
    time_truth = np.array([[0, 1, 0, -1]])
    freq_truth = np.array([[0, -2.j, 0]], dtype=complex)

    time, freq, frequency = stub_utils.sine_func(
        frequency_in,
        sampling_rate,
        n_samples,
        fft_norm,
        cshape)

    npt.assert_allclose(
        time, time_truth, rtol=1e-10, atol=10 * np.finfo(float).eps)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)
    assert frequency == frequency_truth


def test_sine_func_value_error():
    """Test generation of sine data, value errors.
    """
    n_samples = 4
    fft_norm = 'none'
    # Sampling theorem
    cshape = (1,)
    sampling_rate = 4
    frequency_in = 2
    with pytest.raises(ValueError):
        stub_utils.sine_func(
            frequency_in,
            sampling_rate,
            n_samples,
            fft_norm,
            cshape)

    # Inconsistent input shape
    cshape = (2, 2)
    sampling_rate = 4
    frequency_in = [1, 1]
    with pytest.raises(ValueError):
        stub_utils.sine_func(
            frequency_in,
            sampling_rate,
            n_samples,
            fft_norm,
            cshape)


def test_noise_func():
    """Test generation of noise data, single channel.
    """
    n_samples = 2**18
    sigma = 1
    cshape = (1,)
    time, freq = stub_utils.noise_func(sigma, n_samples, cshape)

    npt.assert_array_almost_equal(np.mean(time), 0, decimal=1)
    npt.assert_array_almost_equal(np.std(time, ddof=1), sigma, decimal=1)
    e_time = np.mean(np.abs(time)**2)
    e_freq = np.sum(np.abs(freq)**2)
    npt.assert_array_almost_equal(e_time, e_freq, decimal=4)


def test_noise_func_multi_channel():
    """Test generation of noise data, multiple channels.
    """
    n_samples = 2**10
    sigma = 1
    cshape = (2, 2)
    time, freq = stub_utils.noise_func(sigma, n_samples, cshape)

    npt.assert_array_almost_equal(np.mean(time), 0, decimal=1)
    npt.assert_array_almost_equal(np.std(time, ddof=1), sigma, decimal=1)
    e_time = np.mean(np.abs(time)**2, axis=-1)
    e_freq = np.sum(np.abs(freq)**2, axis=-1)
    npt.assert_array_almost_equal(e_time, e_freq, decimal=2)


def test__eq___dict__flat_data(flat_data):
    """ Test equality for stub. """
    actual = flat_data.copy()
    assert actual == flat_data


def test__eq___dict__nested_data(nested_data):
    """ Test equality for stub. """
    actual = nested_data.copy()
    assert actual == nested_data
