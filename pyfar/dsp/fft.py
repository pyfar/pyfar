"""
The following documents the FFT functionality. More details and background is
given in the :py:mod:`FFT concepts <pyfar._concepts.fft>`.
"""
import multiprocessing

import numpy as np
from scipy import fft


def rfftfreq(n_samples, sampling_rate):
    """
    Returns the positive discrete frequencies for which the FFT is calculated.

    If the number of samples
    :math:`N` is even the number of frequency bins will be :math:`2/N+1`, if
    :math:`N` is odd, the number of bins will be :math:`(N+1)/2`.

    Parameters
    ----------
    n_samples : int
        The number of samples in the signal
    sampling_rate : int
        The sampling rate of the signal

    Returns
    -------
    frequencies : array, double
        The positive discrete frequencies in Hz for which the FFT is
        calculated.
    """
    return fft.rfftfreq(n_samples, d=1/sampling_rate)


def rfft(data, n_samples, sampling_rate, fft_norm):
    """
    Calculate the FFT of a real-valued time-signal.

    The function returns only the right-hand side of the axis-symmetric
    spectrum. The normalization is considered according to
    ``'fft_norm'`` as described in :py:func:`~pyfar.dsp.fft.normalization`
    and :py:mod:`FFT concepts <pyfar._concepts.fft>`.

    Parameters
    ----------
    data : array, double
        Array containing the time domain signal with dimensions
        (..., ``'n_samples'``)
    n_samples : int
        The number of samples
    sampling_rate : number
        sampling rate in Hz
    fft_norm : 'none', 'unitary', 'amplitude', 'rms', 'power', 'psd'
        See documentation of :py:func:`~pyfar.dsp.fft.normalization`.

    Returns
    -------
    spec : array, complex
        The complex valued right-hand side of the spectrum with dimensions
        (..., n_bins)

    """

    # DFT
    spec = fft.rfft(
        data, n=n_samples, axis=-1, workers=multiprocessing.cpu_count())
    # Normalization
    spec = normalization(spec, n_samples, sampling_rate, fft_norm,
                         inverse=False, single_sided=True)

    return spec


def irfft(spec, n_samples, sampling_rate, fft_norm):
    """
    Calculate the IFFT of a single-sided Fourier spectrum.

    The function takes only the right-hand side of the spectrum and returns a
    real-valued time signal. The normalization is considered according to
    ``'fft_norm'`` as described in :py:func:`~pyfar.dsp.fft.normalization`
    and :py:mod:`FFT concepts <pyfar._concepts.fft>`.

    Parameters
    ----------
    spec : array, complex
        The complex valued right-hand side of the spectrum with dimensions
        (..., n_bins)
    n_samples : int
        The number of samples of the corresponding time signal. This is crucial
        to allow for the correct transform of time signals with an odd number
        of samples.
    sampling_rate : number
        sampling rate in Hz
    fft_norm : 'none', 'unitary', 'amplitude', 'rms', 'power', 'psd'
        See :py:func:`~pyfar.dsp.fft.normalization`.

    Returns
    -------
    data : array, double
        Array containing the time domain signal with dimensions
        (..., ``'n_samples'``)
    """

    # Inverse normalization
    spec = normalization(spec, n_samples, sampling_rate, fft_norm,
                         inverse=True, single_sided=True)
    # Inverse DFT
    data = fft.irfft(
        spec, n=n_samples, axis=-1, workers=multiprocessing.cpu_count())

    return data


def normalization(spec, n_samples, sampling_rate, fft_norm='none',
                  inverse=False, single_sided=True, window=None):
    """
    Normalize a Fourier spectrum.

    Apply normalizations defined in [1]_ to the DFT spectrum.
    Note that the phase is maintained in all cases, i.e., instead of taking
    the squared absolute values for ``'power'`` and ``'psd'``, the complex
    spectra are multiplied with their absolute values to ensure a correct
    renormalization.
    For detailed information and explanations, refer to
    :py:mod:`FFT concepts <pyfar._concepts.fft>`.

    Parameters
    ----------
    spec : numpy array
        N dimensional array which has the frequency bins in the last
        dimension. E.g., ``spec.shape == (10,2,129)`` holds 10 times 2 spectra
        with 129 frequency bins each.
    n_samples : int
        number of samples of the corresponding time signal
    sampling_rate : number
        sampling rate of the corresponding time signal in Hz
    fft_norm : string, optional
        ``'none'``
            Do not apply any normalization. Appropriate for energy signals
            such as impulse responses.
        ``'unitary'``
            Multiply `spec` by factor of two as in [1]_ Eq. (8)
            (except for 0 Hz and the Nyquist frequency at half the sampling
            rate) to obtain the single-sided spectrum.
        ``'amplitude'``
            Scale spectrum by ``1/n_samples`` as in [1]_ Eq. (4)
            to obtain the amplitude spectrum.
        'rms'
            Scale spectrum by :math:`1/\\sqrt{2}` as in [1]_
            Eq.(10) to obtain the RMS spectrum.
        'power'
            Power spectrum, which equals the squared RMS spectrum
            (except for the retained phase).
        'psd'
            The power spectrum is scaled by ``n_samples/sampling_rate`` as in
            [1]_ Eq. (6)

        Note that the `unitary` normalization is also applied for `amplitude`,
        `rms`, `power`, and `psd` if the input spectrum is single sided (see
        `single_sided`).
    inverse : bool, optional
        apply the inverse normalization. The default is ``False``.
    single_sided : bool, optional
        denotes if `spec` is a single sided spectrum up to half the sampling
        rate or a both sided (full) spectrum. If ``single_sided=True`` the
        `unitary` normalization according to [1]_ Eq. (8) is applied unless
        ``fft_norm='none'``.
        The default is ``True``.
    window : None, array like
        window that was applied to the time signal before performing the FFT.
        Affects the normalization as in [1]_ Eqs. (11-13). The window must be
        an array-like with `n_samples` length and. The default is ``None``,
        which denotes that no window was applied.

    Returns
    -------
    spec : numpy array
        normalized input spectrum

    References
    ----------
    .. [1]  J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
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
