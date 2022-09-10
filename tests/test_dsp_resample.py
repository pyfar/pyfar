import pyfar as pf
import pytest
import numpy as np


@pytest.mark.parametrize('L', [2, 0.5])
def test_resampling(L):
    """
    Tests the up and downsampling of a noise signal to the double/half sampling
    rate.
    """
    fs_1 = 48000
    fs_2 = L*fs_1
    signal = pf.signals.noise(1024, sampling_rate=fs_1)
    resampled_sig = pf.dsp.resample(signal, fs_2, post_filter=False)
    # asserts the sample length
    assert L*signal.n_samples == resampled_sig.n_samples
    # asserts the length of time of the signals
    assert signal.n_samples/fs_1 == resampled_sig.n_samples/fs_2


def test_upsampling_delayed_impulse():
    """
    Compares an upsampled delayed impulse with the analytic result of a
    sinc function.
    """
    fs_1 = 48000
    fs_2 = 96000
    N = 128
    signal = pf.signals.impulse(N, 64, sampling_rate=fs_1)
    # Get resampled Signal with function
    resampled = pf.dsp.resample(signal, fs_2, match_amplitude="time",
                                post_filter=False)
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
    sinc function.
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


def test_resample_assertions():
    # test the remaining errors and warnings
    fs_1 = 48000
    fs_2 = 96000
    # use power signal, which needs match_amplitude="time"
    signal = pf.signals.sine(3000, 128, full_period=True, sampling_rate=fs_1)
    signal.fft_norm = "amplitude"
    # test ValueError with wrong value for match_amplitude
    with pytest.raises(ValueError, match='match_amplitude must be "time"'):
        pf.dsp.resample(signal, fs_2, match_amplitude="freq")
    # test TypeError for input is not a pyfar.Signal
    with pytest.raises(TypeError,
                       match="Input data has to be of type pyfar.Signal"):
        pf.dsp.resample([0, 1, 0], fs_2, match_amplitude="freq")
    # test ValueError for invalid match_amplitude, must be "time" or "freq"
    with pytest.raises(ValueError,
                       match="match_amplitude is 'invalid_match_amplitude'"):
        pf.dsp.resample(signal, fs_2,
                        match_amplitude="invalid_match_amplitude")
    # test warning for target sampling rate is not divisible by 10
    with pytest.warns(UserWarning,
                      match="At least one sampling rate is not divisible"):
        pf.dsp.resample(signal, 12345, match_amplitude="time")
    # test warning for target sampling rate realisation with an error
    signal2 = pf.signals.impulse(128, 64, sampling_rate=48000)
    with pytest.warns(UserWarning,
                      match="The target sampling rate was realized with"):
        pf.dsp.resample(signal2, 420, frac_limit=100)


def test_frequency_matching():
    """
    Tests the amplitude matching in the frequency domain for the first N/2-10
    samples.
    """
    fs_1 = 48000
    fs_2 = 96000
    N = 128
    signal = pf.signals.impulse(N, 64, sampling_rate=fs_1)
    resampled = pf.dsp.resample(signal, fs_2, match_amplitude="freq",
                                post_filter=False)
    np.testing.assert_almost_equal(resampled.freq[0][:int(N/2)-10],
                                   signal.freq[0][:int(N/2)-10], decimal=2)


def test_resample_multidimensional_impulse():
    """
    Compares an upsampled multidimensional delayed impulse with cshape = (3,2)
    with the analytic result of a sinc function.
    """
    fs_1 = 48000
    fs_2 = 96000
    N = 128
    signal = pf.signals.impulse(N, 64, amplitude=[[1, 2, 3], [4, 5, 6]],
                                sampling_rate=fs_1)
    # Get resampled Signal with function
    resampled = pf.dsp.resample(signal, fs_2, 'time', post_filter=False)
    # Test the cshape
    assert signal.cshape == resampled.cshape
    # Calculated the analytic signal with sinc function
    L = fs_2 / fs_1
    n = np.arange(-N/2, N/2, 1/L)
    c = int(20*fs_2/fs_1)
    s = np.sinc(n)
    sinc = pf.Signal([[1*s, 2*s, 3*s], [4*s, 5*s, 6*s]], fs_2)
    sinc = pf.dsp.time_window(sinc, [N*L/2-int(c/2), N*L/2+int(c/2)],
                              window='hamming')
    np.testing.assert_almost_equal(resampled.time, sinc.time, decimal=2)


def test_resample_suppress_aliasing():
    """Test the aliasing suppression filter"""

    # test signal
    signal = pf.signals.impulse(1024, 512)
    # measure impulse response of anti-aliasing filter
    true = pf.dsp.resample(signal, 48000, post_filter=True)
    false = pf.dsp.resample(signal, 48000, post_filter=False)
    diff = true / false

    phase = pf.dsp.phase(diff)
    mag = pf.dsp.decibel(diff)

    # check zero-phase in pass band (should be exactly 0, but is not possibly
    # due to the limited length of the signal)
    idx_pass = diff.find_nearest_frequency(22050)
    assert np.all(np.abs(phase[..., :idx_pass - 1]) <= .02)
    # check unity gain in pass band (deviations should be <= 0.1 dB, but are
    # in the range of 0.25 dB possibly due to the finite length of the signal)
    assert np.all(np.abs(mag[..., :idx_pass - 1]) <= .25)

    # check stop band damping (magnitudes should be below -60 dB, but are
    # below -50 dB possibly due to the finite length of the signal)
    idx_stop = diff.find_nearest_frequency(22050 * 1.05)
    assert np.all(mag[..., idx_stop + 1:] < -50)
