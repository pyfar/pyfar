import pytest
import numpy as np

import stub_utils


# Sine stubs
# test_dsp.py
# test_fft.py
# test_plot.py
# test_signal.py
@pytest.fixture
def sine():
    """Sine signal stub.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
                        frequency,
                        sampling_rate,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)

    return signal


# test_fft.py
@pytest.fixture
def sine_odd():
    """Sine signal stub,
    odd number of samples.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 9999
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
                        frequency,
                        sampling_rate,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)

    return signal


# Impulse stubs
# test_fft.py
# test_dsp.py
@pytest.fixture
def impulse():
    """Delta impulse signal stub.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    """
    delay = 0
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
                        delay,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)

    return signal


# test_dsp.py
@pytest.fixture
def impulse_group_delay():
    """Delayed delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    delay = 100
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
                        delay,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)
    group_delay = delay * np.ones_like(freq, dtype=float)

    return signal, group_delay


# test_dsp.py
@pytest.fixture
def impulse_group_delay_two_channel():
    """Delayed 2 channel delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    delay = np.atleast_1d([100, 200])
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2,)

    time, freq = stub_utils.impulse_func(
                        delay,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)
    group_delay = delay[..., np.newaxis] * np.ones_like(freq, dtype=float)

    return signal, group_delay


# test_dsp.py
@pytest.fixture
def impulse_group_delay_two_by_two_channel():
    """Delayed 2-by-2 channel delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    delay = np.array([[100, 200], [300, 400]])
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2, 2)

    time, freq = stub_utils.impulse_func(
                        delay,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)
    group_delay = delay[..., np.newaxis] * np.ones_like(freq, dtype=float)

    return signal, group_delay


# test_fft.py
@pytest.fixture
def impulse_rms():
    """Delta impulse signal stub with static properties, rms-FFT normalization.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    """
    pass


# test_dsp.py
@pytest.fixture
def sine_plus_impulse():
    """Combined sine and delta impulse signal stub.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    delay = 100
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time_sine, freq_sine, frequency = stub_utils.sine_func(
                        frequency,
                        sampling_rate,
                        n_samples,
                        fft_norm,
                        cshape)
    time_imp, freq_imp = stub_utils.impulse_func(
                        delay,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time_sine + time_imp,
                        freq_sine + freq_imp,
                        sampling_rate,
                        fft_norm)

    return signal


# test_fft.py
@pytest.fixture
def noise():
    """Gaussian white noise signal stub.
    The frequency spectrum is set to dummy value None.

    Returns
    -------
    signal : Signal
        Stub of noise signal
    """
    sigma = 1
    n_samples = int(1e5)
    cshape = (1,)
    sampling_rate = 44100
    fft_norm = 'rms'
    freq = None

    time = stub_utils.noise_func(sigma, n_samples, cshape)

    signal = stub_utils.signal_stub(
                    time,
                    freq,
                    sampling_rate,
                    fft_norm)
    return signal


@pytest.fixture
def noise_odd():
    """Gaussian white noise signal stub,
    odd number of samples.
    The frequency spectrum is set to dummy value None.

    Returns
    -------
    signal : Signal
        Stub of noise signal
    """
    sigma = 1
    n_samples = int(1e5-1)
    cshape = (1,)
    sampling_rate = 44100
    fft_norm = 'rms'
    freq = None

    time = stub_utils.noise_func(sigma, n_samples, cshape)

    signal = stub_utils.signal_stub(
                    time,
                    freq,
                    sampling_rate,
                    fft_norm)
    return signal


# test_fft.py
@pytest.fixture
def fft_lib_np(monkeypatch):
    """Set numpy.fft as fft library.
    """
    import pyfar.fft
    monkeypatch.setattr(pyfar.fft, 'fft_lib', np.fft)


@pytest.fixture
def fft_lib_pyfftw(monkeypatch):
    """Set pyfftw as fft library.
    """
    import pyfar.fft
    from pyfftw.interfaces import numpy_fft as npi_fft
    monkeypatch.setattr(pyfar.fft, 'fft_lib', npi_fft)
