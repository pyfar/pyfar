import numpy as np
import numpy.testing as npt
import pytest
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


def test_fft_orthogonality_sine_even_lib():
    num_samples = 2**10
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = np.fft.rfft(signal_time, n=num_samples, axis=-1)
    transformed_signal_time = np.fft.irfft(signal_spec, n=num_samples, axis=-1)

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_sine_odd_lib():
    num_samples = 2**10+3
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = np.fft.rfft(signal_time, n=num_samples, axis=-1)
    transformed_signal_time = np.fft.irfft(signal_spec, n=num_samples, axis=-1)

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_noise_even_lib():
    n_samples = 2**18
    np.random.seed(450)
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = np.fft.rfft(signal_time, n=n_samples, axis=-1)
    transformed_signal_time = np.fft.irfft(signal_spec, n=n_samples, axis=-1)

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_odd_lib():
    n_samples = 2**18+1
    np.random.seed(450)
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = np.fft.rfft(signal_time, n=n_samples, axis=-1)
    transformed_signal_time = np.fft.irfft(signal_spec, n=n_samples, axis=-1)

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_sine_even_np(fft_lib_np):
    num_samples = 2**10
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, num_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_sine_even_fftw(fft_lib_pyfftw):
    num_samples = 2**10
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, num_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_sine_odd_np(fft_lib_np):
    num_samples = 2**10+3
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, num_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_sine_odd_fftw(fft_lib_pyfftw):
    num_samples = 2**10+3
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, num_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_noise_even_np(fft_lib_np):
    n_samples = 2**18
    np.random.seed(450)
    samplingrate = 40e3
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(
        signal_time, n_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, n_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_even_fftw(fft_lib_pyfftw):
    n_samples = 2**18
    np.random.seed(450)
    samplingrate = 40e3
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(
        signal_time, n_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, n_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_odd_np(fft_lib_np):
    n_samples = 2**18+1
    np.random.seed(450)
    samplingrate = 40e3
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(
        signal_time, n_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, n_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_odd_fftw(fft_lib_pyfftw):
    n_samples = 2**18+1
    np.random.seed(450)
    samplingrate = 40e3
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(
        signal_time, n_samples, samplingrate, 'power', 'rms')
    transformed_signal_time = fft.irfft(
        signal_spec, n_samples, samplingrate, 'power', 'rms')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_parsevaL_theorem_sine_even_np(fft_lib_np):
    num_samples = 2**10
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(signal_time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_sine_even_fftw(fft_lib_pyfftw):
    num_samples = 2**10
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(signal_time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_sine_odd_np(fft_lib_np):
    num_samples = 2**10+3
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(signal_time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_sine_odd_fftw(fft_lib_pyfftw):
    num_samples = 2**10+3
    frequency = 10e3
    samplingrate = 40e3
    num_periods = np.floor(num_samples / samplingrate * frequency)
    # round to the nearest frequency resulting in a fully periodic sine signal
    # in the given time interval
    frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate

    signal_time = 1 * np.cos(2 * np.pi * frequency * times)
    signal_spec = fft.rfft(
        signal_time, num_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(signal_time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_even_np(fft_lib_np):
    n_samples = 2**20
    np.random.seed(450)
    samplingrate = 40e3
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(
        noise_time, n_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_even_fftw(fft_lib_pyfftw):
    n_samples = 2**20
    np.random.seed(450)
    samplingrate = 40e3
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(
        noise_time, n_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_odd_np(fft_lib_np):
    n_samples = 2**20+1
    np.random.seed(450)
    samplingrate = 40e3
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(
        noise_time, n_samples, samplingrate, 'power', 'rms')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_is_odd():
    num = 3
    assert fft._is_odd(num)


def test_is_not_odd():
    num = 4
    assert not fft._is_odd(num)


def test_normalization_energy_signal():
    spec_single = np.array([1, 1, 1])
    N = 4       # time signal with even number of samples
    fs = 40     # arbitrary sampling frequency for psd normalization

    spec_out = fft.normalization(spec_single.copy(), N, fs, "energy",
                                 "rms", inverse=False)
    npt.assert_allclose(spec_out, spec_single, atol=1e-15)

    spec_out = fft.normalization(spec_out, N, fs, "energy",
                                 "rms", inverse=True)
    npt.assert_allclose(spec_out, spec_single, atol=1e-15)


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
        spec_out = fft.normalization(spec_single.copy(), N, fs, "power",
                                     normalization, inverse=False)
        npt.assert_allclose(spec_out, truth[normalization], atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs, "power",
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
        spec_out = fft.normalization(spec_single.copy(), N, fs, "power",
                                     normalization, inverse=False)
        npt.assert_allclose(spec_out, truth[normalization], atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs, "power",
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
        spec_out = fft.normalization(spec_single.copy(), N, fs, "power",
                                     normalization, inverse=False,
                                     single_sided=False)
        npt.assert_allclose(spec_out, truth[normalization], atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs, "power",
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
        spec_out = fft.normalization(spec_single.copy(), N, fs, "power",
                                     normalization, inverse=False)
        npt.assert_allclose(spec_out, np.tile(truth[normalization], tile),
                            atol=1e-15)

        print(f"Assesing normalization: '{normalization}' (inverse)")
        spec_out_inv = fft.normalization(spec_out, N, fs, "power",
                                         normalization, inverse=True)
        npt.assert_allclose(spec_out_inv, spec_single, atol=1e-15)


def test_normalization_with_window():
    """
    Test if the window cancels out if applying the normalization and
    inverse normalization.
    """

    # test with window as list and numpy array
    windows = [[1, 1, 1, 1], np.array([1, 1, 1, 1])]

    # test power signals
    fft_norms = ['unitary', 'amplitude', 'rms', 'power', 'psd']
    for window in windows:
        for fft_norm in fft_norms:
            print(f"testing: {window}, {fft_norm}")
            spec = fft.normalization(np.array([.5, 1, .5]), 4, 44100,
                                     'power', fft_norm, window=window)
            spec = fft.normalization(spec, 4, 44100, 'power', fft_norm,
                                     inverse=True, window=window)
            npt.assert_allclose(np.array([.5, 1, .5]), spec, atol=1e-15)

    # test with energy signals
    for window in windows:
        print(f"testing: {window}, energy signal (unitary)")
        spec = fft.normalization(np.array([.5, 1, .5]), 4, 44100,
                                 'energy', 'unitary', window=window)
        spec = fft.normalization(spec, 4, 44100, 'energy', 'unitary',
                                 inverse=True, window=window)
        npt.assert_allclose(np.array([.5, 1, .5]), spec, atol=1e-15)


def test_normalization_with_window_value_error():
    """
    Test if normalization throws a ValueError if the window has the
    wrong length.
    """

    with raises(ValueError):
        # n_samples=5, and len(window)=5
        fft.normalization(np.array([.5, 1, .5]), 4, 44100, 'power',
                          'amplitude', window=[1, 1, 1, 1, 1])


def test_normalization_exceptions():
    # try without numpy array
    with raises(ValueError):
        fft.normalization(1, 1, 44100, 'power')
    # try rms normalization for power signal
    with raises(ValueError):
        fft.normalization(np.array([1]), 1, 44100, 'power',
                          'rms', single_sided=False)


def test_rfft_energy_imp_even_samples(impulse):
    n_samples = 1024
    samplingrate = 40e3
    spec = fft.rfft(impulse, n_samples, samplingrate, 'energy', 'unitary')

    truth = np.ones(int(n_samples/2+1), dtype=np.complex)
    npt.assert_allclose(spec, truth)


def test_rfft_energy_imp_even_samples_fftw(impulse, fft_lib_pyfftw):
    n_samples = 1024
    samplingrate = 40e3
    spec = fft.rfft(impulse, n_samples, samplingrate, 'energy', 'unitary')

    truth = np.ones(int(n_samples/2+1), dtype=np.complex)
    npt.assert_allclose(spec, truth)


def test_irfft_energy_imp_even_samples_np(impulse, fft_lib_np):
    n_samples = 1024
    samplingrate = 40e3
    spec = np.ones(int(n_samples/2+1), dtype=np.complex)
    data = fft.irfft(spec, n_samples, samplingrate, 'energy', 'unitary')

    truth = impulse
    npt.assert_allclose(data, truth)


def test_irfft_energy_imp_even_samples_np_fftw(impulse, fft_lib_pyfftw):
    n_samples = 1024
    samplingrate = 40e3
    spec = np.ones(int(n_samples/2+1), dtype=np.complex)
    data = fft.irfft(spec, n_samples, samplingrate, 'energy', 'unitary')

    truth = impulse
    npt.assert_allclose(data, truth)


def test_rfft_power_imp_even_samples_np(sine, fft_lib_np):
    n_samples = 1024
    sampling_rate = 2e3
    spec = fft.rfft(sine, n_samples, sampling_rate, 'power', 'rms')

    truth = np.zeros(int(n_samples/2+1), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_rfft_power_imp_even_samples_fftw(sine, fft_lib_pyfftw):
    n_samples = 1024
    sampling_rate = 2e3
    spec = fft.rfft(sine, n_samples, sampling_rate, 'power', 'rms')

    truth = np.zeros(int(n_samples/2+1), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_irfft_power_imp_even_samples_np(sine, fft_lib_np):
    n_samples = 1024
    samplingrate = 40e3
    spec = np.zeros(int(n_samples/2+1), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, samplingrate, 'power', 'rms')

    truth = sine
    npt.assert_allclose(data, truth, atol=1e-10)


def test_irfft_power_imp_even_samples_fftw(sine, fft_lib_pyfftw):
    n_samples = 1024
    samplingrate = 40e3
    spec = np.zeros(int(n_samples/2+1), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, samplingrate, 'power', 'rms')

    truth = sine
    npt.assert_allclose(data, truth, atol=1e-10)


def test_irfft_power_imp_odd_samples_np(sine_odd, fft_lib_np):
    n_samples = 1023
    samplingrate = 40e3
    spec = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, samplingrate, 'power', 'rms')

    truth, f = sine_odd
    npt.assert_allclose(data, truth, atol=1e-10)


def test_irfft_power_imp_odd_samples_fftw(sine_odd, fft_lib_pyfftw):
    n_samples = 1023
    samplingrate = 40e3
    spec = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, samplingrate, 'power', 'rms')

    truth, f = sine_odd
    npt.assert_allclose(data, truth, atol=1e-10)


def test_rfft_power_imp_odd_samples_np(sine_odd, fft_lib_np):
    n_samples = 1023
    s, f = sine_odd
    sampling_rate = 40e3
    spec = fft.rfft(s, n_samples, sampling_rate, 'power', 'rms')

    truth = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_rfft_power_imp_odd_samples_fftw(sine_odd, fft_lib_pyfftw):
    n_samples = 1023
    s, f = sine_odd
    sampling_rate = 40e3
    spec = fft.rfft(s, n_samples, sampling_rate, 'power', 'rms')

    truth = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_fft_mock_numpy(fft_lib_np):
    assert 'numpy.fft' in fft.fft_lib.__name__


@pytest.fixture
def fft_lib_np(monkeypatch):
    # from pyfar.fft import fft_lib
    import pyfar.fft
    import numpy as np
    monkeypatch.setattr(pyfar.fft, 'fft_lib', np.fft)


def test_fft_mock_pyfftw(fft_lib_pyfftw):
    assert 'pyfftw' in fft.fft_lib.__name__


@pytest.fixture
def fft_lib_pyfftw(monkeypatch):
    # from pyfar.fft import fft_lib
    import pyfar.fft
    from pyfftw.interfaces import numpy_fft as npi_fft
    monkeypatch.setattr(pyfar.fft, 'fft_lib', npi_fft)


@pytest.fixture
def impulse():
    """Generate an impulse, also known as the Dirac delta function

    .. math::

        s(n) =
        \\begin{cases}
        a,  & \\text{if $n$ = 0} \\newline
        0, & \\text{else}
        \\end{cases}

    Returns
    -------
    signal : ndarray, double
        The impulse signal

    """
    amplitude = 1
    num_samples = 1024

    signal = np.zeros(num_samples, dtype=np.double)
    signal[0] = amplitude

    return signal


@pytest.fixture
def sine():
    """Generate a sine signal with f = 440 Hz and samplingrate = 44100 Hz.

    Returns
    -------
    signal : ndarray, double
        The sine signal

    """
    amplitude = 1
    frequency = 125
    samplingrate = 2e3
    num_samples = 1024
    fullperiod = False

    if fullperiod:
        num_periods = np.floor(num_samples / samplingrate * frequency)
        # round to the nearest frequency resulting in a fully periodic
        # sine signal in the given time interval
        frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate
    signal = amplitude * np.cos(2 * np.pi * frequency * times)

    return signal


@pytest.fixture
def sine_odd():
    """Generate a sine signal with f = 440 Hz and samplingrate = 44100 Hz.

    Returns
    -------
    signal : ndarray, double
        The sine signal

    """
    amplitude = 1
    frequency = 125
    samplingrate = 2e3
    num_samples = 1023
    fullperiod = True

    if fullperiod:
        num_periods = np.floor(num_samples / samplingrate * frequency)
        # round to the nearest frequency resulting in a fully periodic
        # sine signal in the given time interval
        frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate
    signal = amplitude * np.cos(2 * np.pi * frequency * times)

    return signal, frequency
