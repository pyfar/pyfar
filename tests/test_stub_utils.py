import pytest
import numpy as np
import numpy.testing as npt

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
    # Non consistent input
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
    """Test generation of sine data.
    """
    n_samples = 4
    sampling_rate = 4
    fft_norm = 'none'
    
    frequency_truth = 1
    time_truth = np.array([[0, 1, 0, -1]])
    freq_truth = np.array([[0, -1.j, 0]], dtype=complex)
  
    time, freq, frequency = stub_utils.sine_func(
                                frequency_truth,
                                sampling_rate,
                                n_samples,
                                fft_norm)

    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)
    assert frequency == frequency_truth


def test_sine_func_frequency_adjustment():
    """Test generation of sine data, adjusted frequency.
    """
    n_samples = 4
    sampling_rate = 4
    fft_norm = 'none'
    frequency_in = 1.5

    frequency_truth = 1.
    time_truth = np.array([[0, 1, 0, -1]])
    freq_truth = np.array([[0, -1.j, 0]], dtype=complex)
  
    time, freq, frequency = stub_utils.sine_func(
                                frequency_in,
                                sampling_rate,
                                n_samples,
                                fft_norm)

    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)
    assert frequency == frequency_truth


def test_sine_func_value_error():
    """Test generation of sine data, value errors.
    """
    # Sampling theorem
    n_samples = 4
    sampling_rate = 4
    fft_norm = 'none'
    frequency_in = 2
  
    with pytest.raises(ValueError):
        stub_utils.sine_func(
                            frequency_in,
                            sampling_rate,
                            n_samples,
                            fft_norm)


# TO DO
def test_cos_func_multi_channel():
    """Test function to cosine, full period as input"""
    # Full period
    n_samples = 12
    sampling_rate = 12
    cshape = (2, 2)
    frequency = np.array([[1, 2], [4, 8]])
    fullperiod = False

    time_truth = np.array([[[1, np.sqrt(3)/2, 0.5, 0, -0.5, -np.sqrt(3)/2],
                            [1, 0, 0, 0, 0, 0],],
                           [[1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],]])
    freq_truth = np.array([[[0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],],
                           [[0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],]],
                          dtype=complex)
    fft_norm_truth = 'rms'
    
    time, freq, fft_norm = mocks.cos_func(cshape,
                                          frequency,
                                          fullperiod,
                                          n_samples,
                                          sampling_rate)
    npt.assert_allclose(time, time_truth, atol=1e-10)
    npt.assert_allclose(np.real(freq), np.real(freq_truth), atol=1e-10)
    npt.assert_allclose(np.imag(freq), np.imag(freq_truth), atol=1e-10)
    assert fft_norm == fft_norm_truth