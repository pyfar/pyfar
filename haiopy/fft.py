r"""

The implemented discrete Fourier spectrum is defined as

.. math::
        X(\mu) = \sum_{n=0}^{N-1} x(n) e^{-i 2 \pi \frac{\mu n}{N}}

using a negative sign convention in the transform kernel `math::\kappa(\mu, n) = e^{-i 2 \pi \mu \frac{n}{N}`.
Analogously, the discrete inverse Fourier transform is implemented as

.. math::
        x(n) = \frac{1}{N} \sum_{\mu = 0}^{N-1} X(\mu) e^{i 2 \pi \frac{\mu n}{N}}

Normalization
-------------
Haiopy uses the assumption of real-valued time signals resulting
in Fourier spectra with complex conjugate symmetry for negative and
positive frequencies, enabling the use of single sided spectra.
In order for Parseval's theorem to remain valid, the single sided
needs to be multiplied by a factor of 2, compensating for the energy
missing.

>>> import numpy as np
>>> from haiopy import fft
>>> import matplotlib.pyplot as plt
>>> sine = np.sin(np.linspace(0, 2*np.pi, 256))
>>> spec = fft.rfft(sine, 256, 'power')
>>> plt.plot(np.abs(spec))
>>> plt.show()

.. plot::

    import numpy as np
    from haiopy import fft
    import matplotlib.pyplot as plt
    n_samples = 1024
    sine = np.sin(np.linspace(0, 10, n_samples) * 2*np.pi * 100)
    spec = fft.rfft(sine, n_samples, 'power')
    freqs = fft.rfftfreq(n_samples, 48e3)
    plt.subplot(1, 2, 1)
    plt.plot(sine)
    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(spec))
    plt.show()


References
----------
.. [1]  J.-R. Ohm and H. D. Lüke, Signalübertragung: Grundlagen der
        digitalen und analogen Nachrichtenübertragungssysteme. Springer DE, 2002.

.. [2]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on Scaling of
        the Discrete Fourier Transform and the Implied Physical Units of the
        Spectra of Time-Discrete Signals,” p. 5, 2020.


"""
import multiprocessing
import warnings

import numpy as np

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    from pyfftw.interfaces import numpy_fft as fft_lib
except ImportError:
    warnings.warn(
        "Using numpy FFT implementation.\
        Install pyfftw for improved performance.")
    from numpy import fft as fft_lib


def rfftfreq(n_samples, sampling_rate):
    """
    Returns the positive discrete frequencies in the range `:math:[0, f_s/2]`
    for which the FFT of a real valued time-signal with n_samples is
    calculated. If the number of samples `:math:N` is even the number of
    frequency bins will be `:math:2N+1`, if `:math:N` is odd, the number of
    bins will be `:math:(N+1)/2`.

    Parameters
    ----------
    n_samples : int
        The number of samples in the signal
    sampling_rate : int
        The sampling rate of the signal

    Returns
    -------
    frequencies : array, double
        The positive discrete frequencies for which the FFT is calculated.
    """
    return fft_lib.rfftfreq(n_samples, d=1/sampling_rate)


def rfft(data, n_samples, signal_type):
    """
    Calculate the FFT of a real-valued time-signal. The function returns only
    the right-hand side of the axis-symmetric spectrum. The normalization
    distinguishes between energy and power signals. Energy signals are not
    normalized in the forward transform. Power signals are normalized by their
    number of samples and to their effective value as well as compensated for
    the missing energy from the left-hand side of the spectrum. This ensures
    that the energy of the time signal and the right-hand side of the spectrum
    are equal and thus fulfill Parseval's theorem.

    Parameters
    ----------
    data : array, double
        Array containing the time domain signal with dimensions
        (..., n_samples)
    n_samples : int
        The number of samples
    signal_type : 'energy', 'power'
        The signal type for normalization.

    Returns
    -------
    spec : array, complex
        The complex valued right-hand side of the spectrum with dimensions
        (..., n_bins)

    """

    sqrt_two = np.sqrt(2)
    spec = fft_lib.rfft(data, n=n_samples, axis=-1)

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
    Calculate the IFFT of a axis-symmetric Fourier spectum. The function
    takes only the right-hand side of the spectrum and returns a real-valued
    time signal. The normalization distinguishes between energy and power
    signals. Energy signals are not normalized by their number of samples.
    Power signals are normalized by their effective value as well as
    compensated for the missing energy from the left-hand side of the spectrum.
    This ensures that the energy of the time signal and the right-hand side
    of the spectrum are equal and thus fulfill Parseval's theorem.

    Parameters
    ----------
    spec : array, complex
        The complex valued right-hand side of the spectrum with dimensions
        (..., n_bins)
    n_samples : int
        The number of samples in the corresponding tim signal. This is crucial
        to allow for the correct transform of time signals with an odd number
        of samples.
    signal_type : 'energy', 'power'
        The signal type for normalization.

    Returns
    -------
    data : array, double
        Array containing the time domain signal with dimensions
        (..., n_samples)
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

    data = fft_lib.irfft(spec, n=n_samples, axis=-1)

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


def _n_bins(n_samples):
    """
    Helper function to calculate the number of bins resulting from a FFT
    with n_samples

    Paramters
    ---------
    n_samples : int
        Number of samples

    Returns
    -------
    n_bins : int
        Resulting number of frequency bins

    """
    if _is_odd(n_samples):
        n_bins = (n_samples+1)/2
    else:
        n_bins = n_samples/2+1

    return int(n_bins)
