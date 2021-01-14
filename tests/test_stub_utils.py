import pytest
import numpy as np
import numpy.testing as npt
import scipy.stats as stats

import stub_utils


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

    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)


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
   
    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)


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
    freq_truth = np.array([[0, -1.j, 0]], dtype=complex)
  
    time, freq, frequency = stub_utils.sine_func(
                                frequency_truth,
                                sampling_rate,
                                n_samples,
                                fft_norm,
                                cshape)

    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)
    assert frequency == frequency_truth


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

    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)
    npt.assert_allclose(frequency, frequency_truth, atol=1e-10)


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

    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)
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
