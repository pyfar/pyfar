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
    sqrt_two = np.sqrt(2)

    spec = fft.rfft(data, n=n_samples, axis=-1)

    if signal_type == 'energy':
        norm = 1
    elif signal_type == 'power':
        norm = 1/n_samples * sqrt_two
        spec[..., 0] = spec[..., 0] / sqrt_two
        if not _is_odd(n_samples):
            spec[..., -1] = spec[..., -1] / sqrt_two

    spec *= norm

    return spec


def irfft(spec, n_samples, signal_type):
    """
    """
    sqrt_two = np.sqrt(2)

    if signal_type == 'energy':
        norm = 1
    elif signal_type == 'power':
        norm = n_samples/np.sqrt(2)
        spec[..., 0] = spec[..., 0] * sqrt_two
        if not _is_odd(n_samples):
            spec[..., -1] = spec[..., -1] * sqrt_two

    spec *= norm

    data = fft.irfft(spec, n=n_samples, axis=-1)

    return data


def _is_odd(num):
    """
    Check if a number is even or odd. Returns True if odd and False if even.

    Parameters
    ----------
    num : int
        Integer number to check

    Returns
    -------
    condition : bool
        True if odd and False if even

    """
    return bool(num & 0x1)
