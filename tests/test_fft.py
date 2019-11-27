import numpy as np
import numpy.testing as npt
import pytest

from haiopy import fft


def test_rfft_energy_imp_even_samples(impulse):
    n_samples = 1024
    spec = fft.rfft(impulse, n_samples, 'energy')

    truth = np.ones(int(n_samples/2+1), dtype=np.complex)
    npt.assert_allclose(spec, truth)


def test_irfft_energy_imp_even_samples(impulse):
    n_samples = 1024
    spec = np.ones(int(n_samples/2+1), dtype=np.complex)
    data = fft.irfft(spec, n_samples, 'energy')

    truth = impulse
    npt.assert_allclose(data, truth)


def test_rfft_power_imp_even_samples(sine):
    n_samples = 1024
    sampling_rate = 2e3
    spec = fft.rfft(sine, n_samples, 'power')

    truth = np.zeros(int(n_samples/2+1), dtype=np.complex)
    truth[int(n_samples/16)] = 1/np.sqrt(2)
    npt.assert_allclose(np.real(spec), np.real(truth), atol=1e-10)
    npt.assert_allclose(np.imag(spec), np.imag(truth), atol=1e-10)


def test_irfft_power_imp_even_samples(sine):
    n_samples = 1024
    spec = np.zeros(int(n_samples/2+1), dtype=np.complex)
    spec[int(n_samples/16)] = 1/np.sqrt(2)

    data = fft.irfft(spec, n_samples, 'power')

    truth = sine
    npt.assert_allclose(data, truth, atol=1e-10)


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
