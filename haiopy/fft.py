import numpy as np

try:
    import pyfftw
except ImportError:
    from numpy import fft as fft


def rfftfreq(n_samples, sampling_rate):
    return fft.rfftfreq(n_samples, d=1/sampling_rate)


def rfft(data, n_samples, signal_type):
    """

    """
    if signal_type == 'energy':
        norm = 1
    elif signal_type == 'power':
        norm = 1/n_samples*np.sqrt(2)

    spec = fft.rfft(data, n=n_samples, axis=-1) * norm

    return spec


def irfft(spec, n_samples, signal_type):
    if signal_type == 'energy':
        norm = 1
    elif signal_type == 'power':
        norm = n_samples/np.sqrt(2)

    data = fft.irfft(spec, n=n_samples, axis=-1) * norm

    return data
