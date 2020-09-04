import numpy as np
import numpy.testing as npt
import pytest

from haiopy import fft


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

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1.5e-10)


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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, num_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, num_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, num_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, num_samples, 'power')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-10)


def test_fft_orthogonality_noise_even_np(fft_lib_np):
    n_samples = 2**18
    np.random.seed(450)
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(signal_time, n_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, n_samples, 'power')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_even_fftw(fft_lib_pyfftw):
    n_samples = 2**18
    np.random.seed(450)
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(signal_time, n_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, n_samples, 'power')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_odd_np(fft_lib_np):
    n_samples = 2**18+1
    np.random.seed(450)
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(signal_time, n_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, n_samples, 'power')

    npt.assert_allclose(signal_time, transformed_signal_time, rtol=1e-8)


def test_fft_orthogonality_noise_odd_fftw(fft_lib_pyfftw):
    n_samples = 2**18+1
    np.random.seed(450)
    signal_time = np.random.normal(0, 1, n_samples)
    signal_spec = fft.rfft(signal_time, n_samples, 'power')
    transformed_signal_time = fft.irfft(signal_spec, n_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')

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
    signal_spec = fft.rfft(signal_time, num_samples, 'power')

    e_time = np.mean(np.abs(signal_time)**2)
    e_freq = np.sum(np.abs(signal_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_even_np(fft_lib_np):
    n_samples = 2**20
    np.random.seed(450)
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(noise_time, n_samples, 'power')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_even_fftw(fft_lib_pyfftw):
    n_samples = 2**20
    np.random.seed(450)
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(noise_time, n_samples, 'power')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_odd_np(fft_lib_np):
    n_samples = 2**20+1
    np.random.seed(450)
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(noise_time, n_samples, 'power')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_fft_parsevaL_theorem_noise_odd_fftw(fft_lib_pyfftw):
    n_samples = 2**20+1
    np.random.seed(450)
    noise_time = np.random.normal(0, 1, n_samples)
    noise_spec = fft.rfft(noise_time, n_samples, 'power')

    e_time = np.mean(np.abs(noise_time)**2)
    e_freq = np.sum(np.abs(noise_spec)**2)

    npt.assert_allclose(e_time, e_freq, rtol=1e-10)


def test_is_odd():
    num = 3
    assert fft._is_odd(num)


def test_is_not_odd():
    num = 4
    assert not fft._is_odd(num)


def test_rfft_energy_imp_even_samples_np(impulse, fft_lib_np):
    n_samples = 1024
    spec = fft.rfft(impulse, n_samples, 'energy')

    truth = np.ones(int(n_samples/2+1), dtype=np.complex)
    npt.assert_allclose(spec, truth)


def test_rfft_energy_imp_even_samples_fftw(impulse, fft_lib_pyfftw):
    n_samples = 1024
    spec = fft.rfft(impulse, n_samples, 'energy')

    truth = np.ones(int(n_samples/2+1), dtype=np.complex)
    npt.assert_allclose(spec, truth)


def test_irfft_energy_imp_even_samples_np(impulse, fft_lib_np):
    n_samples = 1024
    spec = np.ones(int(n_samples/2+1), dtype=np.complex)
    data = fft.irfft(spec, n_samples, 'energy')

    truth = impulse
    npt.assert_allclose(data, truth)


def test_irfft_energy_imp_even_samples_np_fftw(impulse, fft_lib_pyfftw):
    n_samples = 1024
    spec = np.ones(int(n_samples/2+1), dtype=np.complex)
    data = fft.irfft(spec, n_samples, 'energy')

    truth = impulse
    npt.assert_allclose(data, truth)


def test_rfft_power_imp_even_samples_np(sine, fft_lib_np):
    n_samples = 1024
    spec = fft.rfft(sine, n_samples, 'power')

    truth = np.zeros(int(n_samples/2+1), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_rfft_power_imp_even_samples_fftw(sine, fft_lib_pyfftw):
    n_samples = 1024
    spec = fft.rfft(sine, n_samples, 'power')

    truth = np.zeros(int(n_samples/2+1), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_irfft_power_imp_even_samples_np(sine, fft_lib_np):
    n_samples = 1024
    spec = np.zeros(int(n_samples/2+1), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, 'power')

    truth = sine
    npt.assert_allclose(data, truth, atol=1e-10)


def test_irfft_power_imp_even_samples_fftw(sine, fft_lib_pyfftw):
    n_samples = 1024
    spec = np.zeros(int(n_samples/2+1), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, 'power')

    truth = sine
    npt.assert_allclose(data, truth, atol=1e-10)


def test_irfft_power_imp_odd_samples_np(sine_odd, fft_lib_np):
    n_samples = 1023
    spec = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, 'power')

    truth, f = sine_odd
    npt.assert_allclose(data, truth, atol=1e-10)


def test_irfft_power_imp_odd_samples_fftw(sine_odd, fft_lib_pyfftw):
    n_samples = 1023
    spec = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, 'power')

    truth, f = sine_odd
    npt.assert_allclose(data, truth, atol=1e-10)


def test_rfft_power_imp_odd_samples_np(sine_odd, fft_lib_np):
    n_samples = 1023
    s, f = sine_odd
    spec = fft.rfft(s, n_samples, 'power')

    truth = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_rfft_power_imp_odd_samples_fftw(sine_odd, fft_lib_pyfftw):
    n_samples = 1023
    s, f = sine_odd
    spec = fft.rfft(s, n_samples, 'power')

    truth = np.zeros(int((n_samples+1)/2), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_fft_mock_numpy(fft_lib_np):
    assert 'numpy.fft' in fft.fft_lib.__name__


@pytest.fixture
def fft_lib_np(monkeypatch):
    # from haiopy.fft import fft_lib
    import haiopy.fft
    import numpy as np
    monkeypatch.setattr(haiopy.fft, 'fft_lib', np.fft)


def test_fft_mock_pyfftw(fft_lib_pyfftw):
    assert 'pyfftw' in fft.fft_lib.__name__


@pytest.fixture
def fft_lib_pyfftw(monkeypatch):
    # from haiopy.fft import fft_lib
    import haiopy.fft
    from pyfftw.interfaces import numpy_fft as npi_fft
    monkeypatch.setattr(haiopy.fft, 'fft_lib', npi_fft)


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
        # round to the nearest frequency resulting in a fully periodic sine signal
        # in the given time interval
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
        # round to the nearest frequency resulting in a fully periodic sine
        # signal in the given time interval
        frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate
    signal = amplitude * np.cos(2 * np.pi * frequency * times)

    return signal, frequency
