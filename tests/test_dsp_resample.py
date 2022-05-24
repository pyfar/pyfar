import pyfar as pf
import pytest
import numpy as np


def test_upsampling():
    # Tests the upsampling of a noise signal to the double sampling rate.
    fs_1 = 48000
    fs_2 = 2*fs_1
    signal = pf.signals.noise(1024, sampling_rate=fs_1)
    resampled_sig = pf.dsp.resample(signal, fs_2)
    # asserts the sample length
    assert 2*signal.n_samples == resampled_sig.n_samples
    # asserts the length of time of the signals
    assert signal.n_samples/fs_1 == resampled_sig.n_samples/fs_2


def test_downsampling():
    # Tests the upsampling of a noise signal to the double sampling rate.
    fs_1 = 48000
    fs_2 = 0.5*fs_1
    signal = pf.signals.noise(1024, sampling_rate=fs_1)
    resampled_sig = pf.dsp.resample(signal, fs_2)
    # asserts the sample length
    assert 0.5*signal.n_samples == resampled_sig.n_samples
    # asserts the length of time of the signals
    assert signal.n_samples/fs_1 == resampled_sig.n_samples/fs_2


def test_upsampling_delayed_impulse():
    """
    Compares a upsampled delayed impulse with the analytic result of a
    sync function.
    """
    fs_1 = 48000
    fs_2 = 96000
    N = 128
    signal = pf.signals.impulse(N, 64, sampling_rate=fs_1)
    # Get resampled Signal with function
    resampled = pf.dsp.resample(signal, fs_2, match_amplitude="time")
    # Calculated the analytic signal with sinc function
    L = fs_2 / fs_1
    n = np.arange(-N/2, N/2, 1/L)
    c = int(20*fs_2/fs_1)
    sinc = pf.Signal(np.sinc(n), fs_2)
    sinc = pf.dsp.time_window(sinc, [N*L/2-int(c/2), N*L/2+int(c/2)],
                              window='hamming')
    np.testing.assert_almost_equal(resampled.time, sinc.time, decimal=3)


def test_downsampling_delayed_impulse():
    """
    Compares a downsampled delayed impulse with the analytic result of a
    sync function.
    """
    fs_1 = 48000
    fs_2 = 24000
    N = 128
    signal = pf.signals.impulse(N, 64, sampling_rate=fs_1)
    # Get resampled Signal with function
    resampled = pf.dsp.resample(signal, fs_2, match_amplitude="time")
    # Calculated the analytic signal with sinc function
    L = fs_2 / fs_1
    n = np.arange(-N/2*L, N/2*L)
    c = 20
    sinc = pf.Signal(L*np.sinc(n), fs_2)
    sinc = pf.dsp.time_window(sinc, [N*L/2-int(c/2), N*L/2+int(c/2)],
                              window='hamming')
    np.testing.assert_almost_equal(resampled.time, sinc.time, decimal=3)


def test_wrong_aplitude_matching():
    # Tests ValueError for wrong aplitude matching for power signals.
    fs_1 = 48000
    fs_2 = 96000
    # Use power signal, which needs match_amplitude="time"
    signal = pf.signals.sine(3000, 128, full_period=True, sampling_rate=fs_1)
    signal.fft_norm = "amplitude"
    with pytest.raises(ValueError, match='match_aplitude must be "time"'):
        pf.dsp.resample(signal, fs_2, match_amplitude="freq")


def test_frequency_matching():
    """
    Tests the amplitude matching in the frequency domain for the first N/2-10
    samples.
    """
    fs_1 = 48000
    fs_2 = 96000
    N = 128
    signal = pf.signals.impulse(N, 64, sampling_rate=fs_1)
    resampled = pf.dsp.resample(signal, fs_2, match_amplitude="freq")
    np.testing.assert_almost_equal(resampled.freq[0][:int(N/2)-10],
                                   signal.freq[0][:int(N/2)-10], decimal=2)
