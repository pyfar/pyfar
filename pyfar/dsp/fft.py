r"""
The discrete Fourier spectrum of an arbitrary, but band-limited signal
:math:`x(n)` is defined as

.. math::
        X(\mu) = \sum_{n=0}^{N-1} x(n) e^{-i 2 \pi \frac{\mu n}{N}}

using a negative sign convention in the transform kernel
:math:`\kappa(\mu, n) = e^{-i 2 \pi \mu \frac{n}{N}}`.
Analogously, the discrete inverse Fourier transform is implemented as

.. math::
        x(n) = \frac{1}{N} \sum_{\mu=0}^{N-1} X(\mu) e^{i2\pi\frac{\mu n}{N}}

Pyfar uses a DFT implementation for purely real-valued time signals resulting
in Fourier spectra with complex conjugate symmetry for negative and
positive frequencies :math:`X(\mu) = X(-\mu)^*`. As a result,
the left-hand side of the spectrum is discarded, yielding
:math:`X_R(\mu) = X(\mu) \mbox{ }\forall 0 \le \mu \le N/2`. Complex valued
time signals can be implemented, if required.

Normalization [1]_
------------------
Bases on a signal FFT norm - namely 'unitary', 'amplitude', 'rms', 'power' or
'psd', pyfar implements five different normalization variants, whereby
'unitary' denotes that no additional normalization is performed.

Energy Signals
==============

For energy signals with finite energy,
such as impulse responses, no additional normalization is required, that is
the spectrum of a energy signal is equivalent to the right-hand spectrum
of a real-valued time signal defined above.

Power Signals
=============

For power signals however, which possess a finite power but infinite energy,
a normalization for the time interval in which the signal is sampled, is
chosen. In order for Parseval's theorem to remain valid, the single sided
needs to be multiplied by a factor of 2, compensating for the discarded part
of the spectrum (cf. [1]_, Eq. 8). Additionally, the implemented DFT uses
different introduced above.

>>> import numpy as np
>>> from pyfar.dsp import fft
>>> import matplotlib.pyplot as plt
>>> frequency = 100
>>> sampling_rate = 1000
>>> n_samples = 1024
>>> sampling_rate = 48e3
>>> sine = np.sin(np.linspace(0, 2*np.pi*frequency/sampling_rate, n_samples))
>>> spectrum = fft.rfft(sine, n_samples, sampling_rate, 'rms')

.. plot::

    import numpy as np
    from pyfar.dsp import fft
    import matplotlib.pyplot as plt
    n_samples = 1024
    sampling_rate = 48e3
    times = np.linspace(0, 10, n_samples)
    sine = np.sin(times * 2*np.pi * 100)
    spec = fft.rfft(sine, n_samples, sampling_rate, 'rms')
    freqs = fft.rfftfreq(n_samples, 48e3)
    plt.subplot(1, 2, 1)
    plt.plot(times, sine)
    ax = plt.gca()
    ax.set_xlabel('Time in s')
    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(spec))
    ax = plt.gca()
    ax.set_xlabel('Frequency in Hz')
    plt.show()


References
----------
.. [1]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
        Scaling of the Discrete Fourier Transform and the Implied Physical
        Units of the Spectra of Time-Discrete Signals,” Vienna, Austria,
        May 2020, p. e-Brief 600.


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


def rfft(data, n_samples, sampling_rate, fft_norm):
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
    sampling_rate : number
        sampling rate in Hz
    fft_norm : 'unitary', 'amplitude', 'rms', 'power', 'psd'
        See documentation of :py:func:`~pyfar.dsp.fft.normalization`.

    Returns
    -------
    spec : array, complex
        The complex valued right-hand side of the spectrum with dimensions
        (..., n_bins)

    """

    # DFT
    spec = fft_lib.rfft(data, n=n_samples, axis=-1)
    # Normalization
    spec = normalization(spec, n_samples, sampling_rate, fft_norm,
                         inverse=False, single_sided=True)

    return spec


def irfft(spec, n_samples, sampling_rate, fft_norm):
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
    sampling_rate : number
        sampling rate in Hz
    fft_norm : 'unitary', 'amplitude', 'rms', 'power', 'psd'
        See documentaion of :py:func:`~pyfar.dsp.fft.normalization`.

    Returns
    -------
    data : array, double
        Array containing the time domain signal with dimensions
        (..., n_samples)
    """

    # Inverse normalization
    spec = normalization(spec, n_samples, sampling_rate, fft_norm,
                         inverse=True, single_sided=True)
    # Inverse DFT
    data = fft_lib.irfft(spec, n=n_samples, axis=-1)

    return data


def normalization(spec, n_samples, sampling_rate, fft_norm='none',
                  inverse=False, single_sided=True, window=None):
    """
    Normalize spectrum of power signal.

    Apply normalizations defined in [2]_ to DFT spectrum of power signals.
    No normalization is applied to energy signals. Note that, the phase is
    maintained in all cases, i.e., instead of taking the squared absolute
    spectra in Eq. (5-6), the complex spectra are multiplied with their
    absolute values.

    Parameters
    ----------
    spec : numpy array
        N dimensional array which has the frequency bins in the last
        dimension. E.g., spec.shape == (10,2,129) holds 10 times 2 spectra with
        129 frequencies each
    n_samples : int
        number of samples of the corresponding time signal
    sampling_rate : number
        sampling rate of the corresponding time signal in Hz
    fft_norm : string, optional
        'none'
            Do not apply any normalization
        'unitary'
            Multiplied single sided spectra by factor two (except for 0 Hz and
            half the sampling rate)
        'amplitude'
            as in _[2] Eq. (4)
        'rms'
            as in _[2] Eq. (10)
        'power'
            as in _[2] Eq. (5)
        'psd'
            as in _[2] Eq. (6)
    inverse : bool, optional
        apply the inverse normalization. The default is false.
    single_sided : bool, optional
        denotes if `spec` is a single sided spectrum up to half the sampling
        rate or a both sided (full) spectrum. If `single_sided==True` the
        normalization according to _[2] Eq. (8) is applied for power signals.
        The default is True.
    window : None, array like
        window that was applied to the time signal before performing the FFT.
        window must be an array like with `n_samples` and affects the
        normalization as in _[2] Eqs. (11-13). The default is None, which
        denotes that no window was applied.

    Returns
    -------
    spec : numpy array
        normalized version of the input spectrum

    References
    ----------
    .. [2] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
           Scaling of the Discrete Fourier Transform and the Implied Physical
           Units of the Spectra of Time-Discrete Signals,” Vienna, Austria,
           May 2020, p. e-Brief 600.
    """

    # check if normalization should be applied
    if fft_norm == 'none':
        return spec

    # check input
    if not isinstance(spec, np.ndarray):
        raise ValueError("Input 'spec' must be a numpy array.")
    if window is not None:
        if len(window) != n_samples:
            raise ValueError((f"window must be {n_samples} long "
                              f"but is {len(window)} long."))

    n_bins = spec.shape[-1]
    norm = np.ones(n_bins)

    # account for type of normalization
    if fft_norm == "amplitude":
        if window is None:
            # Equation 4 in Ahrens et al. 2020
            norm /= n_samples
        else:
            # Equation 11 in Ahrens et al. 2020
            norm /= np.sum(window)
    elif fft_norm == 'rms':
        if not single_sided:
            raise ValueError(
                "'rms' normalization does only exist for single-sided spectra")
        if window is None:
            # Equation 10 in Ahrens et al. 2020
            norm /= n_samples
        else:
            # Equation 11 in Ahrens et al. 2020
            norm /= np.sum(window)
        if _is_odd(n_samples):
            norm[1:] /= np.sqrt(2)
        else:
            norm[1:-1] /= np.sqrt(2)
    elif fft_norm == 'power':
        if window is None:
            # Equation 5 in Ahrens et al. 2020
            norm /= n_samples**2
        else:
            # Equation 12 in Ahrens et al. 2020
            norm /= np.sum(window)**2
        # the phase is kept for being able to switch between normalizations
        # altoug the power spectrum does usually not have phase information,
        # i.e., spec = np.abs(spec)**2
        if not inverse:
            spec *= np.abs(spec)
    elif fft_norm == 'psd':
        if window is None:
            # Equation 6 in Ahrens et al. 2020
            norm /= (n_samples * sampling_rate)
        else:
            # Equation 13 in Ahrens et al. 2020
            norm /= (np.sum(window)**2 * sampling_rate)
        # the phase is kept for being able to switch between normalizations
        # altoug the power spectrum does usually not have phase information,
        # i.e., spec = np.abs(spec)**2
        if not inverse:
            spec *= np.abs(spec)
    elif fft_norm != 'unitary':
        raise ValueError(("norm type must be 'unitary', 'amplitude', 'rms', "
                          f"'power', or 'psd' but is '{fft_norm}'"))

    # account for inverse
    if inverse:
        norm = 1 / norm

    # apply normalization
    spec = spec * norm

    # scaling for single sided spectrum, i.e., to account for the lost
    # energy in the discarded half of the spectrum. Only the bins at 0 Hz
    # and Nyquist remain as they are (Equation 8 in Ahrens et al. 2020).
    if single_sided:
        scale = 2 if not inverse else 1 / 2
        if _is_odd(n_samples):
            spec[..., 1:] *= scale
        else:
            spec[..., 1:-1] *= scale

    # reverse the squaring in case of 'power' and 'psd' normalization
    if inverse and fft_norm in ["power", "psd"]:
        spec /= np.sqrt(np.abs(spec))

    return spec


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
