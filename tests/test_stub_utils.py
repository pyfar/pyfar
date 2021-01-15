import pytest
import numpy as np
import numpy.testing as npt

from pyfar import Signal

import stub_utils


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
    n_samples = 4
    cshape = (1,)
    delay = 0
    fft_norm = 'rms'
    freq_truth = np.array([[1, np.sqrt(2), 1]], dtype=complex) / n_samples
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
                           [[1, -1,  1],
                            [1,  1j, -1]]])

    time, freq = stub_utils.impulse_func(delay, n_samples, fft_norm, cshape)

    npt.assert_allclose(time, time_truth, rtol=1e-10)
    npt.assert_allclose(freq, freq_truth, rtol=1e-10)


def test_normalization_none():
    """ Test unitary FFT normalization implemented in stubs_utils.py"""
    n_samples = 4
    freq = np.array([1, 1, 1], dtype=np.complex)
    freq_norm = stub_utils._normalization(freq, n_samples, 'none')
    npt.assert_allclose(freq, freq_norm, rtol=1e-10)


def test_normalization_rms_even():
    """ Test RMS FFT normalization implemented in stubs_utils.py,
    even number of samples."""
    n_samples = 4
    freq = np.array([1, 1, 1], dtype=np.complex)
    freq_norm_truth = np.array([1, np.sqrt(2), 1], dtype=np.complex)
    freq_norm_truth /= n_samples
    freq_norm = stub_utils._normalization(freq, n_samples, 'rms')
    npt.assert_allclose(freq_norm, freq_norm_truth, rtol=1e-10)


def test_normalization_rms_odd():
    """ Test RMS FFT normalization implemented in stubs_utils.py,
    odd number of samples."""
    n_samples = 5
    freq = np.array([1, 1, 1], dtype=np.complex)
    freq_norm_truth = np.array([1, np.sqrt(2), np.sqrt(2)], dtype=np.complex)
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

    npt.assert_allclose(time, time_truth,
        rtol=1e-10, atol=10*np.finfo(float).eps)
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
    freq_truth = np.array([[0, -2j*np.sqrt(2)/n_samples, 0]], dtype=complex)

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
    sq3 = np.sqrt(3)
    time_truth = np.array(
        [[[0, 0.5, sq3/2, 1, sq3/2, 0.5,
           0, -0.5, -sq3/2, -1, -sq3/2, -0.5],
          [0, sq3/2, sq3/2, 0, -sq3/2, -sq3/2,
           0, sq3/2, sq3/2, 0, -sq3/2, -sq3/2]],
         [[0, 1, 0, -1, 0, 1,
          0, -1, 0, 1, 0, -1],
          [0, sq3/2, -sq3/2, 0, sq3/2, -sq3/2,
           0, sq3/2, -sq3/2, 0, sq3/2, -sq3/2]]])
    freq_truth = np.array(
        [[[0, -1j, 0, 0, 0, 0, 0],
          [0, 0, -1j, 0, 0, 0, 0]],
         [[0, 0, 0, -1j, 0, 0, 0],
          [0, 0, 0, 0, -1j, 0, 0]]],
        dtype=complex)

    time, freq, frequency = stub_utils.sine_func(
                                frequency_truth,
                                sampling_rate,
                                n_samples,
                                fft_norm,
                                cshape)

    npt.assert_allclose(time, time_truth,
        rtol=1e-10, atol=10*np.finfo(float).eps)
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
    freq_truth = np.array([[0, -1.j, 0]], dtype=complex)

    time, freq, frequency = stub_utils.sine_func(
                                frequency_in,
                                sampling_rate,
                                n_samples,
                                fft_norm,
                                cshape)

    npt.assert_allclose(time, time_truth,
        rtol=1e-10, atol=10*np.finfo(float).eps)
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
    time = stub_utils.noise_func(sigma, n_samples, cshape)

    npt.assert_array_almost_equal(np.mean(time), 0, decimal=1)
    npt.assert_array_almost_equal(np.std(time, ddof=1), sigma, decimal=1)


def test_noise_func_multi_channel():
    """Test generation of noise data, multiple channels.
    """
    n_samples = 2**10
    sigma = 1
    cshape = (2, 2)
    time = stub_utils.noise_func(sigma, n_samples, cshape)

    npt.assert_array_almost_equal(np.mean(time), 0, decimal=1)
    npt.assert_array_almost_equal(np.std(time, ddof=1), sigma, decimal=1)
