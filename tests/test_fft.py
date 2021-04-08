import numpy as np
import numpy.testing as npt
from pytest import raises

from pyfar import fft


def test_n_bins_even():
    n_samples = 6
    truth = int(n_samples/2 + 1)
    n_bins = fft._n_bins(n_samples)
    assert n_bins == truth


def test_n_bins_odd():
    n_samples = 7
    truth = int((n_samples + 1)/2)
    n_bins = fft._n_bins(n_samples)
    assert n_bins == truth


def test_fft_orthogonality_sine_even_np(sine, fft_lib_np):
    signal_spec = fft.rfft(
        sine.time, sine.n_samples, sine.sampling_rate, sine.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, sine.n_samples, sine.sampling_rate, sine.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, sine.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_sine_even_fftw(sine, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        sine.time, sine.n_samples, sine.sampling_rate, sine.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, sine.n_samples, sine.sampling_rate, sine.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, sine.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_sine_odd_np(sine_odd, fft_lib_np):
    signal_spec = fft.rfft(
        sine_odd.time, sine_odd.n_samples, sine_odd.sampling_rate,
        sine_odd.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, sine_odd.n_samples, sine_odd.sampling_rate,
        sine_odd.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, sine_odd.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_sine_odd_fftw(sine_odd, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        sine_odd.time, sine_odd.n_samples, sine_odd.sampling_rate,
        sine_odd.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, sine_odd.n_samples, sine_odd.sampling_rate,
        sine_odd.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, sine_odd.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_noise_even_np(noise, fft_lib_np):
    signal_spec = fft.rfft(
        noise.time, noise.n_samples, noise.sampling_rate, noise.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, noise.n_samples, noise.sampling_rate, noise.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, noise.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_noise_even_fftw(noise, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        noise.time, noise.n_samples, noise.sampling_rate, noise.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, noise.n_samples, noise.sampling_rate, noise.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, noise.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_noise_odd_np(noise_odd, fft_lib_np):
    signal_spec = fft.rfft(
        noise_odd.time, noise_odd.n_samples, noise_odd.sampling_rate,
        noise_odd.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, noise_odd.n_samples, noise_odd.sampling_rate,
        noise_odd.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, noise_odd.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_orthogonality_noise_odd_fftw(noise_odd, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        noise_odd.time, noise_odd.n_samples, noise_odd.sampling_rate,
        noise_odd.fft_norm)
    transformed_signal_time = fft.irfft(
        signal_spec, noise_odd.n_samples, noise_odd.sampling_rate,
        noise_odd.fft_norm)
    npt.assert_allclose(
        transformed_signal_time, noise_odd.time,
        rtol=1e-10, atol=10*np.finfo(float).eps)


def test_fft_parsevaL_theorem_sine_even_np(sine_rms, fft_lib_np):
    signal_spec = fft.rfft(
        sine_rms.time, sine_rms.n_samples, sine_rms.sampling_rate,
        sine_rms.fft_norm)

    e_time = np.mean(np.abs(sine_rms.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_sine_even_fftw(sine_rms, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        sine_rms.time, sine_rms.n_samples, sine_rms.sampling_rate,
        sine_rms.fft_norm)

    e_time = np.mean(np.abs(sine_rms.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_sine_odd_np(sine_odd_rms, fft_lib_np):
    signal_spec = fft.rfft(
        sine_odd_rms.time, sine_odd_rms.n_samples, sine_odd_rms.sampling_rate,
        sine_odd_rms.fft_norm)

    e_time = np.mean(np.abs(sine_odd_rms.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_sine_odd_fftw(sine_odd_rms, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        sine_odd_rms.time, sine_odd_rms.n_samples, sine_odd_rms.sampling_rate,
        sine_odd_rms.fft_norm)

    e_time = np.mean(np.abs(sine_odd_rms.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_even_np(noise, fft_lib_np):
    signal_spec = fft.rfft(
        noise.time, noise.n_samples, noise.sampling_rate, noise.fft_norm)

    e_time = np.mean(np.abs(noise.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_even_fftw(noise, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        noise.time, noise.n_samples, noise.sampling_rate, noise.fft_norm)

    e_time = np.mean(np.abs(noise.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_odd_np(noise_odd, fft_lib_np):
    signal_spec = fft.rfft(
        noise_odd.time, noise_odd.n_samples, noise_odd.sampling_rate,
        noise_odd.fft_norm)

    e_time = np.mean(np.abs(noise_odd.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_odd_fftw(noise_odd, fft_lib_pyfftw):
    signal_spec = fft.rfft(
        noise_odd.time, noise_odd.n_samples, noise_odd.sampling_rate,
        noise_odd.fft_norm)

    e_time = np.mean(np.abs(noise_odd.time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_freq, e_time, rtol=1e-10)


def test_is_odd():
    num = 3
    assert fft._is_odd(num)


def test_is_not_odd():
    num = 4
    assert not fft._is_odd(num)


def test_normalization_none(impulse):
    spec_out = fft.normalization(
        impulse.freq.copy(), impulse.n_samples, impulse.sampling_rate,
        impulse.fft_norm, inverse=False)
    npt.assert_allclose(spec_out, impulse.freq, atol=10*np.finfo(float).eps)

    spec_out = fft.normalization(
        impulse.freq.copy(), impulse.n_samples, impulse.sampling_rate,
        impulse.fft_norm, inverse=True)
    npt.assert_allclose(spec_out, impulse.freq, atol=10*np.finfo(float).eps)


def test_normalization_single_sided_single_channel_even_samples():
    # single sided test spectrum
    v = 1/3 + 1/3j
    vsq = v * np.abs(v)
    spec_single = np.array([v, v, v])
    # valid number of samples of time signal corresponding to spec_single
    N = 4       # time signal with even number of samples
    Nsq = N**2  # factor for power and psd normalization
    fs = 40     # arbitrary sampling frequency for psd normalization

    # expected results for even number of samples
    sqrt2 = np.sqrt(2)
    truth = {
        'unitary': np.array([v, v * 2, v]),
        'amplitude': np.array([v / N,
                               v / N * 2,
                               v / N]),
        'rms': np.array([v / N,
                         v / N / sqrt2 * 2,
                         v / N]),
        'power': np.array([vsq / Nsq,
                           vsq / Nsq * 2,
                           vsq / Nsq]),
        'psd': np.array([vsq / N / fs,
                         vsq / N / fs * 2,
                         vsq / N / fs])
    }

    for normalization in truth:
        print(f"Assesing normalization: '{normalization}'")
        spec_out = fft.normalization(spec_single.copy(), N, fs,
                                     normalization, inverse=False)
        npt.assert_allclose(spec_out, truth[normalization], atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs,
                                         normalization, inverse=True)
        npt.assert_allclose(spec_out_inv, spec_single, atol=1e-15)


def test_normalization_single_sided_single_channel_odd_samples():
    # single sided test spectrum
    v = 1/3 + 1/3j
    vsq = v * np.abs(v)
    spec_single = np.array([v, v, v])
    # valid number of samples of time signal corresponding to spec_single
    N = 5       # time signal with even number of samples
    Nsq = N**2  # factor for power and psd normalization
    fs = 50     # arbitrary sampling frequency for psd normalization

    # expected results for even number of samples
    sqrt2 = np.sqrt(2)
    truth = {
        'unitary': np.array([v, v * 2, v * 2]),
        'amplitude': np.array([v / N,
                               v / N * 2,
                               v / N * 2]),
        'rms': np.array([v / N,
                         v / N / sqrt2 * 2,
                         v / N / sqrt2 * 2]),
        'power': np.array([vsq / Nsq,
                           vsq / Nsq * 2,
                           vsq / Nsq * 2]),
        'psd': np.array([vsq / N / fs,
                         vsq / N / fs * 2,
                         vsq / N / fs * 2])
    }

    for normalization in truth:
        print(f"Assesing normalization: '{normalization}'")
        spec_out = fft.normalization(spec_single.copy(), N, fs,
                                     normalization, inverse=False)
        npt.assert_allclose(spec_out, truth[normalization], atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs,
                                         normalization, inverse=True)
        npt.assert_allclose(spec_out_inv, spec_single, atol=1e-15)


def test_normalization_both_sided_single_channel():
    # single sided test spectrum
    v = 1/3 + 1/3j
    vsq = v * np.abs(v)
    spec_single = np.array([v, v, v])
    # valid number of samples of time signal corresponding to spec_single
    N = 3       # time signal with even number of samples
    Nsq = N**2  # factor for power and psd normalization
    fs = 30     # arbitrary sampling frequency for psd normalization

    # expected results for even number of samples
    truth = {
        'unitary': np.array([v, v, v]),
        'amplitude': np.array([v / N,
                               v / N,
                               v / N]),
        'power': np.array([vsq / Nsq,
                           vsq / Nsq,
                           vsq / Nsq]),
        'psd': np.array([vsq / N / fs,
                         vsq / N / fs,
                         vsq / N / fs])
    }

    for normalization in truth:
        print(f"Assesing normalization: '{normalization}'")
        spec_out = fft.normalization(spec_single.copy(), N, fs,
                                     normalization, inverse=False,
                                     single_sided=False)
        npt.assert_allclose(spec_out, truth[normalization], atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs,
                                         normalization, inverse=True,
                                         single_sided=False)
        npt.assert_allclose(spec_out_inv, spec_single, atol=1e-15)


def test_normalization_single_sided_multi_channel_even_samples():
    # single sided test spectrum
    v = 1/3 + 1/3j
    vsq = v * np.abs(v)
    tile = (4, 2, 1)
    spec_single = np.tile(np.array([v, v, v]), tile)
    # valid number of samples of time signal corresponding to spec_single
    N = 4       # time signal with even number of samples
    Nsq = N**2  # factor for power and psd normalization
    fs = 40     # arbitrary sampling frequency for psd normalization

    # expected results for even number of samples
    sqrt2 = np.sqrt(2)
    truth = {
        'unitary': np.array([v, v * 2, v]),
        'amplitude': np.array([v / N,
                               v / N * 2,
                               v / N]),
        'rms': np.array([v / N,
                         v / N / sqrt2 * 2,
                         v / N]),
        'power': np.array([vsq / Nsq,
                           vsq / Nsq * 2,
                           vsq / Nsq]),
        'psd': np.array([vsq / N / fs,
                         vsq / N / fs * 2,
                         vsq / N / fs])
    }

    for normalization in truth:
        print(f"Assesing normalization: '{normalization}'")
        spec_out = fft.normalization(spec_single.copy(), N, fs,
                                     normalization, inverse=False)
        npt.assert_allclose(spec_out, np.tile(truth[normalization], tile),
                            atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs,
                                         normalization, inverse=True)
        npt.assert_allclose(spec_out_inv, spec_single, atol=1e-15)


def test_normalization_with_window():
    """
    Test if the window cancels out if applying the normalization and
    inverse normalization.
    """

    # test with window as list and numpy array
    windows = [[1, 1, 1, 1], np.array([1, 1, 1, 1])]

    fft_norms = ['unitary', 'amplitude', 'rms', 'power', 'psd']
    for window in windows:
        for fft_norm in fft_norms:
            print(f"testing: {window}, {fft_norm}")
            spec = fft.normalization(np.array([.5, 1, .5]), 4, 44100,
                                     fft_norm, window=window)
            spec = fft.normalization(spec, 4, 44100, fft_norm,
                                     inverse=True, window=window)
            npt.assert_allclose(spec, np.array([.5, 1, .5]), atol=1e-15)


def test_normalization_with_window_value_error():
    """
    Test if normalization throws a ValueError if the window has the
    wrong length.
    """

    with raises(ValueError):
        # n_samples=5, and len(window)=5
        fft.normalization(np.array([.5, 1, .5]), 4, 44100,
                          'amplitude', window=[1, 1, 1, 1, 1])


def test_normalization_exceptions():
    # Call without numpy array
    with raises(ValueError):
        fft.normalization(1, 1, 44100, 'rms')
    # Invalid normalization
    with raises(ValueError):
        fft.normalization(np.array([1]), 1, 44100, 'goofy')


def test_rfft_normalization_impulse(impulse):
    """ Test for call of normalization in rfft.
    """
    signal_spec = fft.rfft(
        impulse.time, impulse.n_samples, impulse.sampling_rate,
        impulse.fft_norm)

    npt.assert_allclose(
        signal_spec, impulse.freq,
        rtol=1e-10, atol=1e-10)


def test_rfft_normalization_impulse_rms(impulse_rms):
    """ Test for call of normalization in rfft.
    """
    signal_spec = fft.rfft(
        impulse_rms.time, impulse_rms.n_samples, impulse_rms.sampling_rate,
        impulse_rms.fft_norm)

    npt.assert_allclose(
        signal_spec, impulse_rms.freq,
        rtol=1e-10, atol=1e-10)


def test_rfft_normalization_sine(sine):
    """ Test for correct call of normalization in rfft.
    """
    signal_spec = fft.rfft(
        sine.time, sine.n_samples, sine.sampling_rate,
        sine.fft_norm)

    npt.assert_allclose(
        signal_spec, sine.freq,
        rtol=1e-10, atol=1e-10)


def test_rfft_normalization_sine_rms(sine_rms):
    """ Test for correct call of normalization in rfft.
    """
    signal_spec = fft.rfft(
        sine_rms.time, sine_rms.n_samples, sine_rms.sampling_rate,
        sine_rms.fft_norm)

    npt.assert_allclose(
        signal_spec, sine_rms.freq,
        rtol=1e-10, atol=1e-10)


def test_fft_mock_numpy(fft_lib_np):
    assert 'numpy.fft' in fft.fft_lib.__name__


def test_fft_mock_pyfftw(fft_lib_pyfftw):
    assert 'pyfftw' in fft.fft_lib.__name__
