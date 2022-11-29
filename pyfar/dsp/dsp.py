import multiprocessing
import numpy as np
from scipy import signal as sgn
import pyfar
from pyfar.dsp import fft
import warnings


def phase(signal, deg=False, unwrap=False):
    """Returns the phase for a given signal object.

    Parameters
    ----------
    signal : Signal, FrequencyData
        pyfar Signal or FrequencyData object.
    deg : Boolean
        Specifies, whether the phase is returned in degrees or radians.
    unwrap : Boolean
        Specifies, whether the phase is unwrapped or not.
        If set to ``'360'``, the phase is wrapped to 2 pi.

    Returns
    -------
    phase : numpy array
        The phase of the signal.
    """

    if not isinstance(signal, pyfar.Signal) and \
            not isinstance(signal, pyfar.FrequencyData):
        raise TypeError(
            'Input data has to be of type: Signal or FrequencyData.')

    phase = np.angle(signal.freq)

    if np.isnan(phase).any() or np.isinf(phase).any():
        raise ValueError('Your signal has a point with NaN or Inf phase.')

    if unwrap is True:
        phase = np.unwrap(phase)
    elif unwrap == '360':
        phase = wrap_to_2pi(np.unwrap(phase))

    if deg:
        phase = np.degrees(phase)
    return phase


def group_delay(signal, frequencies=None, method='fft'):
    """Returns the group delay of a signal in samples.

    Parameters
    ----------
    signal : Signal
        An audio signal object from the pyfar signal class
    frequencies : array-like
        Frequency or frequencies in Hz at which the group delay is calculated.
        The default is ``None``, in which case signal.frequencies is used.
    method : 'scipy', 'fft', optional
        Method to calculate the group delay of a Signal. Both methods calculate
        the group delay using the method presented in [#]_ avoiding issues
        due to discontinuities in the unwrapped phase. Note that the scipy
        version additionally allows to specify frequencies for which the
        group delay is evaluated. The default is ``'fft'``, which is faster.

    Returns
    -------
    group_delay : numpy array
        Frequency dependent group delay in samples. The array is flattened if
        a single channel signal was passed to the function.

    References
    ----------
    .. [#]  https://www.dsprelated.com/showarticle/69.php
    """

    # check input and default values
    if not isinstance(signal, pyfar.Signal):
        raise TypeError('Input data has to be of type: Signal.')

    if frequencies is not None and method == 'fft':
        raise ValueError(
            "Specifying frequencies is not supported for the 'fft' method.")

    frequencies = signal.frequencies if frequencies is None \
        else np.asarray(frequencies, dtype=float)

    if method == 'scipy':
        # get time signal and reshape for easy looping
        time = signal.time
        time = time.reshape((-1, signal.n_samples))

        # initialize group delay
        group_delay = np.zeros((np.prod(signal.cshape), frequencies.size))

        # calculate the group delay
        for cc in range(time.shape[0]):
            group_delay[cc] = sgn.group_delay(
                (time[cc], 1), frequencies, fs=signal.sampling_rate)[1]

        # reshape to match signal
        group_delay = group_delay.reshape(signal.cshape + (-1, ))

    elif method == 'fft':
        freq_k = fft.rfft(signal.time * np.arange(signal.n_samples),
                          signal.n_samples, signal.sampling_rate,
                          fft_norm='none')

        group_delay = np.real(freq_k / signal.freq_raw)

        # catch zeros in the denominator
        group_delay[np.abs(signal.freq_raw) < 1e-15] = 0

    else:
        raise ValueError(
            "Invalid method, needs to be either 'scipy' or 'fft'.")

    # flatten in numpy fashion if a single channel is returned
    if signal.cshape == (1, ):
        group_delay = np.squeeze(group_delay)

    return group_delay


def wrap_to_2pi(x):
    """Wraps phase to 2 pi.

    Parameters
    ----------
    x : double
        Input phase to be wrapped to 2 pi.

    Returns
    -------
    x : double
        Phase wrapped to 2 pi`.
    """
    positive_input = (x > 0)
    zero_check = np.logical_and(positive_input, (x == 0))
    x = np.mod(x, 2*np.pi)
    x[zero_check] = 2*np.pi
    return x


def linear_phase(signal, group_delay, unit="samples"):
    """
    Set the phase to a linear phase with a specified group delay.

    The linear phase signal is computed as

    .. math:: H_{\\mathrm{lin}} = |H| \\mathrm{e}^{-j \\omega \\tau}\\,,

    with :math:`H` the complex spectrum of the input data, :math:`|\\cdot|` the
    absolute values, :math:`\\omega` the frequency in radians and :math:`\\tau`
    the group delay in seconds.

    Parameters
    ----------
    signal : Signal
        input data
    group_delay : float, array like
        The desired group delay of the linear phase signal according to `unit`.
        A reasonable value for most cases is ``signal.n_samples / 2`` samples,
        which results in a time signal that is symmetric around the center. If
        group delay is a list or array it must broadcast with the channel
        layout of the signal (``signal.cshape``).
    unit : string, optional
        Unit of the group delay. Can be ``'samples'`` or ``'s'`` for seconds.
        The default is ``'samples'``.

    Returns
    -------
    signal: Signal
        linear phase copy of the input data
    """

    if not isinstance(signal, pyfar.Signal):
        raise TypeError("signal must be a pyfar Signal object.")

    # group delay in seconds
    if unit == "samples":
        tau = np.asarray(group_delay) / signal.sampling_rate
    elif unit == "s":
        tau = np.asarray(group_delay)
    else:
        raise ValueError(f"unit is {unit} but must be 'samples' or 's'.")

    # linear phase
    phase = 2 * np.pi * signal.frequencies * tau[..., np.newaxis]

    # construct linear phase spectrum
    signal_lin = signal.copy()
    signal_lin.freq_raw = \
        np.abs(signal_lin.freq_raw).astype(complex) * np.exp(-1j * phase)

    return signal_lin


def zero_phase(signal):
    """Calculate zero phase signal.

    The zero phase signal is obtained by taking the absolute values of the
    spectrum

    .. math:: H_z = |H| = \\sqrt{\\mathrm{real}(H)^2 + \\mathrm{imag}(H)^2},

    where :math:`H` is the complex valued spectrum of the input data and
    :math:`H_z` the real valued zero phase spectrum.

    The time domain data of a zero phase signal is symmetric around the first
    sample, e.g., ``signal.time[0, 1] == signal.time[0, -1]``.

    Parameters
    ----------
    signal : Signal, FrequencyData
        input data

    Returns
    -------
    signal : Signal, FrequencyData
        zero phase copy of the input data
    """

    if not isinstance(signal, (pyfar.Signal, pyfar.FrequencyData)):
        raise TypeError(
            'Input data has to be of type Signal or FrequencyData.')

    signal_zero = signal.copy()
    signal_zero.freq_raw = np.atleast_2d(np.abs(signal_zero.freq_raw))

    return signal_zero


def nextpow2(x):
    """Returns the exponent of next higher power of 2.

    Parameters
    ----------
    x : double
        Input variable to determine the exponent of next higher power of 2.

    Returns
    -------
    nextpow2 : double
        Exponent of next higher power of 2.
    """
    return np.ceil(np.log2(x))


def spectrogram(signal, window='hann', window_length=1024,
                window_overlap_fct=0.5, normalize=True):
    """Compute the magnitude spectrum versus time.

    This is a wrapper for ``scipy.signal.spectogram`` with two differences.
    First, the returned times refer to the start of the FFT blocks, i.e., the
    first time is always 0 whereas it is window_length/2 in scipy. Second, the
    returned spectrogram is normalized according to ``signal.fft_norm`` if the
    ``normalize`` parameter is set to ``True``.

    Parameters
    ----------
    signal : Signal
        Signal to compute spectrogram of.
    window : str
        Specifies the window (see ``scipy.signal.windows``). The default is
        ``'hann'``.
    window_length : integer
        Window length in samples, the default ist 1024.
    window_overlap_fct : double
        Ratio of points to overlap between FFT segments [0...1]. The default is
        ``0.5``.
    normalize : bool
        Flag to indicate if the FFT normalization should be applied to the
        spectrogram according to `signal.fft_norm`. The default is ``True``.

    Returns
    -------
    frequencies : numpy array
        Frequencies in Hz at which the magnitude spectrum was computed
    times : numpy array
        Times in seconds at which the magnitude spectrum was computed
    spectrogram : numpy array
    """

    # check input
    if not isinstance(signal, pyfar.Signal):
        raise TypeError('Input data has to be of type: Signal.')

    if window_length > signal.n_samples:
        raise ValueError("window_length exceeds signal length")

    if not isinstance(normalize, bool):
        raise TypeError("The normalize parameter needs to be boolean")

    # get spectrogram from scipy.signal
    window_overlap = int(window_length * window_overlap_fct)
    window = sgn.get_window(window, window_length)

    frequencies, times, spectrogram = sgn.spectrogram(
        x=signal.time.squeeze(), fs=signal.sampling_rate, window=window,
        noverlap=window_overlap, mode='magnitude', scaling='spectrum')

    # remove normalization from scipy.signal.spectrogram
    spectrogram /= np.sqrt(1 / window.sum()**2)

    # apply normalization from signal
    if normalize:
        spectrogram = fft.normalization(
            spectrogram, window_length, signal.sampling_rate,
            signal.fft_norm, window=window)

    # scipy.signal takes the center of the DFT blocks as time stamp we take the
    # beginning (looks nicer in plots, both conventions are used)
    times -= times[0]

    return frequencies, times, spectrogram


def time_window(signal, interval, window='hann', shape='symmetric',
                unit='samples', crop='none', return_window=False):
    """Apply time window to signal.

    This function uses the windows implemented in ``scipy.signal.windows``.

    Parameters
    ----------
    signal : Signal
        Signal object to be windowed.
    interval : array_like
        If `interval` has two entries, these specify the beginning and the end
        of the symmetric window or the fade-in / fade-out (see parameter
        `shape`).
        If `interval` has four entries, a window with fade-in between
        the first two entries and a fade-out between the last two is created,
        while it is constant in between (ignores `shape`).
        The unit of `interval` is specified by the parameter `unit`.
        See below for more details.
    window : string, float, or tuple, optional
        The type of the window. See below for a list of implemented
        windows. The default is ``'hann'``.
    shape : string, optional
        ``'symmetric'``
            General symmetric window, the two values in `interval` define the
            first and last samples of the window.
        ``'symmetric_zero'``
            Symmetric window with respect to t=0, the two values in `interval`
            define the first and last samples of fade-out. `crop` is ignored.
        ``'left'``
            Fade-in, the beginning and the end of the fade is defined by the
            two values in `interval`. See Notes for more details.
        ``'right'``
            Fade-out, the beginning and the end of the fade is defined by the
            two values in `interval`. See Notes for more details.

        The default is ``'symmetric'``.
    unit : string, optional
        Unit of `interval`. Can be set to ``'samples'`` or ``'s'`` (seconds).
        Time values are rounded to the nearest sample. The default is
        ``'samples'``.
    crop : string, optional
        ``'none'``
            The length of the windowed signal stays the same.
        ``'window'``
            The signal is truncated to the windowed part.
        ``'end'``
            Only the zeros at the end of the windowed signal are
            cropped, so the original phase is preserved.

        The default is ``'none'``.
    return_window: bool, optional
        If ``True``, both the windowed signal and the time window are returned.
        The default is ``False``.

    Returns
    -------
    signal_windowed : Signal
        Windowed signal object
    window : Signal
        Time window used to create the windowed signal, only returned if
        ``return_window=True``.

    Notes
    -----
    For a fade-in, the indexes of the samples given in `interval` denote the
    first sample of the window which is non-zero and the first which is one.
    For a fade-out, the samples given in `interval` denote the last sample
    which is one and the last which is non-zero.

    This function calls `scipy.signal.windows.get_window` to create the
    window.
    Available window types:

    - ``boxcar``
    - ``triang``
    - ``blackman``
    - ``hamming``
    - ``hann``
    - ``bartlett``
    - ``flattop``
    - ``parzen``
    - ``bohman``
    - ``blackmanharris``
    - ``nuttall``
    - ``barthann``
    - ``kaiser`` (needs beta, see :py:func:`~pyfar.dsp.kaiser_window_beta`)
    - ``gaussian`` (needs standard deviation)
    - ``general_gaussian`` (needs power, width)
    - ``dpss`` (needs normalized half-bandwidth)
    - ``chebwin`` (needs attenuation)
    - ``exponential`` (needs center, decay scale)
    - ``tukey`` (needs taper fraction)
    - ``taylor`` (needs number of constant sidelobes, sidelobe level)

    If the window requires no parameters, then `window` can be a string.
    If the window requires parameters, then `window` must be a tuple
    with the first argument the string name of the window, and the next
    arguments the needed parameters.

    Examples
    --------

    Options for parameter `shape`.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> signal = pf.Signal(np.ones(100), 44100)
        >>> for shape in ['symmetric', 'symmetric_zero', 'left', 'right']:
        >>>     signal_windowed = pf.dsp.time_window(
        ...         signal, interval=[25,45], shape=shape)
        >>>     ax = pf.plot.time(signal_windowed, label=shape)
        >>> ax.legend(loc='right')

    Window with fade-in and fade-out defined by four values in `interval`.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> signal = pf.Signal(np.ones(100), 44100)
        >>> signal_windowed = pf.dsp.time_window(
        ...         signal, interval=[25, 40, 60, 90], window='hann')
        >>> pf.plot.time(signal_windowed)


    """
    # Check input
    if not isinstance(signal, pyfar.Signal):
        raise TypeError("The parameter signal has to be of type: Signal.")
    if shape not in ('symmetric', 'symmetric_zero', 'left', 'right'):
        raise ValueError(
            "The parameter shape has to be 'symmetric', 'symmetric_zero' "
            "'left' or 'right'.")
    if crop not in ('window', 'end', 'none'):
        raise TypeError(
            "The parameter crop has to be 'none', 'window' or 'end'.")
    if not isinstance(interval, (list, tuple)):
        raise TypeError(
            "The parameter interval has to be of type list, tuple or None.")
    if not isinstance(return_window, bool):
        raise TypeError(
            "The parameter return_window needs to be boolean.")

    interval = np.array(interval)
    if not np.array_equal(interval, np.sort(interval)):
        raise ValueError("Values in interval need to be in ascending order.")
    # Convert to samples
    if unit == 's':
        interval = np.round(interval*signal.sampling_rate).astype(int)
    elif unit == 'samples':
        interval = interval.astype(int)
    else:
        raise ValueError(f"unit is {unit} but has to be 'samples' or 's'.")
    # Check window size
    if interval[-1] > signal.n_samples:
        raise ValueError(
            "Values in interval require window to be longer than signal.")

    # Create window
    # win_start and win_stop define the first and last sample of the window
    if len(interval) == 2:
        if shape == 'symmetric':
            win, win_start, win_stop = _time_window_symmetric_interval_two(
                interval, window)
        elif shape == 'symmetric_zero':
            win, win_start, win_stop = _time_window_symmetric_zero(
                signal.n_samples, interval, window)
        elif shape == 'left':
            win, win_start, win_stop = _time_window_left(
                signal.n_samples, interval, window)
        elif shape == 'right':
            win, win_start, win_stop = _time_window_right(
                interval, window)
    elif len(interval) == 4:
        win, win_start, win_stop = _time_window_symmetric_interval_four(
            interval, window)
    else:
        raise ValueError(
            "interval needs to contain two or four values.")

    # Apply window
    signal_win = signal.copy()
    if crop == 'window':
        signal_win.time = signal_win.time[..., win_start:win_stop+1]*win
        if return_window:
            window_fin = pyfar.Signal(win, signal_win.sampling_rate)
    if crop == 'end':
        # Add zeros before window
        window_zeropadded = np.zeros(win_stop+1)
        window_zeropadded[win_start:win_stop+1] = win
        signal_win.time = signal_win.time[..., :win_stop+1]*window_zeropadded
        if return_window:
            window_fin = pyfar.Signal(
                window_zeropadded, signal_win.sampling_rate)
    elif crop == 'none':
        # Create zeropadded window
        window_zeropadded = np.zeros(signal.n_samples)
        window_zeropadded[win_start:win_stop+1] = win
        signal_win.time = signal_win.time*window_zeropadded
        if return_window:
            window_fin = pyfar.Signal(
                window_zeropadded, signal_win.sampling_rate)

    if return_window:
        window_fin.comment = (
            f"Time window with parameters interval={tuple(interval)},"
            f"window='{window}', shape='{shape}', unit='{unit}', "
            f"crop='{crop}'")
        return signal_win, window_fin
    else:
        return signal_win


def kaiser_window_beta(A):
    """ Return a shape parameter beta to create kaiser window based on desired
    side lobe suppression in dB.

    This function can be used to call :py:func:`~pyfar.dsp.time_window` with
    ``window=('kaiser', beta)``.

    Parameters
    ----------
    A : float
        Side lobe suppression in dB

    Returns
    -------
    beta : float
        Shape parameter beta after [#]_, Eq. 7.75

    References
    ----------
    .. [#]  A. V. Oppenheim and R. W. Schafer, Discrete-time signal processing,
            Third edition, Upper Saddle, Pearson, 2010.
    """
    A = np.abs(A)
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.0

    return beta


def _time_window_symmetric_interval_two(interval, window):
    """ Symmetric time window between 2 values given in interval.

    Parameters
    ----------
    interval : array_like
        Boundaries of the window
    window : string
        Window type, see :py:func:`~pyfar.dsp.time_window`

    Returns
    -------
    win : numpy array
        Time window
    win_start : int
        Index of first sample of window
    win_stop : int
        Index of last sample of window
    """
    win_samples = interval[1]-interval[0]+1
    win = sgn.windows.get_window(window, win_samples, fftbins=False)
    win_start = interval[0]
    win_stop = interval[1]
    return win, win_start, win_stop


def _time_window_left(n_samples, interval, window):
    """ Left-sided time window. ""

    Parameters
    ----------
    n_samples : int
        Number of samples of signal to be windowed
    interval : array_like
        First and last sample of fade-in
    window : string
        Window type, see :py:func:`~pyfar.dsp.time_window`

    Returns
    -------
    win : numpy array
        Time window
    win_start : int
        Index of first sample of window
    win_stop : int
        Index of last sample of window
    """
    fade_samples = int(2*(interval[1]-interval[0]))
    fade = sgn.windows.get_window(window, fade_samples, fftbins=False)
    win = np.ones(n_samples-interval[0])
    win[0:interval[1]-interval[0]] = fade[:int(fade_samples/2)]
    win_start = interval[0]
    win_stop = n_samples-1
    return win, win_start, win_stop


def _time_window_right(interval, window):
    """ Right-sided time window. ""

    Parameters
    ----------
    interval : array_like
        First and last sample of fade-out
    window : string
        Window type, see :py:func:`~pyfar.dsp.time_window`

    Returns
    -------
    win : numpy array
        Time window
    win_start : int
        Index of first sample of window
    win_stop : int
        Index of last sample of window
    """
    fade_samples = int(2*(interval[1]-interval[0]))
    fade = sgn.windows.get_window(window, fade_samples, fftbins=False)
    win = np.ones(interval[1]+1)
    win[interval[0]+1:] = fade[int(fade_samples/2):]
    win_start = 0
    win_stop = interval[1]
    return win, win_start, win_stop


def _time_window_symmetric_zero(n_samples, interval, window):
    """ Symmetric time window with respect to t=0. ""

    Parameters
    ----------
    n_samples : int
        Number of samples of signal to be windowed
    interval : array_like
        First and last sample of fade-out.
    window : string
        Window type, see :py:func:`~pyfar.dsp.time_window`

    Returns
    -------
    win : numpy array
        Time window
    win_start : int
        Index of first sample of window
    win_stop : int
        Index of last sample of window
    """
    fade_samples = int(2*(interval[1]-interval[0]))
    fade = sgn.windows.get_window(window, fade_samples, fftbins=False)
    win = np.zeros(n_samples)
    win[:interval[0]+1] = 1
    win[interval[0]+1:interval[1]+1] = fade[int(fade_samples/2):]
    win[-interval[0]:] = 1
    win[-interval[1]:-interval[0]] = fade[:int(fade_samples/2)]
    win_start = 0
    win_stop = n_samples
    return win, win_start, win_stop


def _time_window_symmetric_interval_four(interval, window):
    """ Symmetric time window with two fades and constant range in between.

    Parameters
    ----------
    interval : array_like
        Indexes of fade-in and fade-out
    window : string
        Window type, see :py:func:`~pyfar.dsp.time_window`

    Returns
    -------
    win : numpy array
        Time window
    win_start : int
        Index of first sample of window
    win_stop : int
        Index of last sample of window
    """
    fade_in_samples = int(2*(interval[1]-interval[0]))
    fade_in = sgn.windows.get_window(
        window, fade_in_samples, fftbins=False)
    fade_in = fade_in[:int(fade_in_samples/2)]
    fade_out_samples = int(2*(interval[3]-interval[2]))
    fade_out = sgn.windows.get_window(
        window, fade_out_samples, fftbins=False)
    fade_out = fade_out[int(fade_out_samples/2):]
    win = np.ones(interval[-1]-interval[0]+1)
    win[0:interval[1]-interval[0]] = fade_in
    win[interval[2]-interval[0]+1:interval[3]-interval[0]+1] = fade_out
    win_start = interval[0]
    win_stop = interval[3]
    return win, win_start, win_stop


def regularized_spectrum_inversion(
        signal, freq_range,
        regu_outside=1., regu_inside=10**(-200/20), regu_final=None,
        normalized=True):
    r"""Invert the spectrum of a signal applying frequency dependent
    regularization.

    Regularization can either be specified within a given
    frequency range using two different regularization factors, or for each
    frequency individually using the parameter `regu_final`. In the first case
    the regularization factors for the frequency regions are cross-faded using
    a raised cosine window function with a width of :math:`\sqrt{2}f` above and
    below the given frequency range. Note that the resulting regularization
    function is adjusted to the quadratic maximum of the given signal.
    In case the `regu_final` parameter is used, all remaining options are
    ignored and an array matching the number of frequency bins of the signal
    needs to be given. In this case, no normalization of the regularization
    function is applied.

    Finally, the inverse spectrum is calculated as [#]_, [#]_,

    .. math::

        S^{-1}(f) = \frac{S^*(f)}{S^*(f)S(f) + \epsilon(f)}


    Parameters
    ----------
    signal : Signal
        The signals which spectra are to be inverted.
    freq_range : tuple, array_like, double
        The upper and lower frequency limits outside of which the
        regularization factor is to be applied.
    regu_outside : float, optional
        The normalized regularization factor outside the frequency range.
        The default is ``1``.
    regu_inside : float, optional
        The normalized regularization factor inside the frequency range.
        The default is ``10**(-200/20)`` (-200 dB).
    regu_final : float, array_like, optional
        The final regularization factor for each frequency, default ``None``.
        If this parameter is set, the remaining regularization factors are
        ignored.
    normalized : bool
        Flag to indicate if the normalized spectrum (according to
        `signal.fft_norm`) should be inverted. The default is ``True``.


    Returns
    -------
    Signal
        The resulting signal after inversion.

    References
    ----------
    .. [#]  O. Kirkeby and P. A. Nelson, “Digital Filter Design for Inversion
            Problems in Sound Reproduction,” J. Audio Eng. Soc., vol. 47,
            no. 7, p. 13, 1999.

    .. [#]  P. C. Hansen, Rank-deficient and discrete ill-posed problems:
            numerical aspects of linear inversion. Philadelphia: SIAM, 1998.

    """
    if not isinstance(signal, pyfar.Signal):
        raise ValueError("The input signal needs to be of type pyfar.Signal.")

    if not isinstance(normalized, bool):
        raise TypeError("The normalized parameter needs to be boolean")

    if normalized:
        data = signal.freq
    else:
        data = signal.freq_raw

    freq_range = np.asarray(freq_range)

    if freq_range.size < 2:
        raise ValueError(
            "The frequency range needs to specify lower and upper limits.")

    if regu_final is None:
        regu_inside = np.ones(signal.n_bins, dtype=np.double) * regu_inside
        regu_outside = np.ones(signal.n_bins, dtype=np.double) * regu_outside

        idx_xfade_lower = signal.find_nearest_frequency(
            [freq_range[0]/np.sqrt(2), freq_range[0]])

        regu_final = _cross_fade(regu_outside, regu_inside, idx_xfade_lower)

        if freq_range[1] < signal.sampling_rate/2:
            idx_xfade_upper = signal.find_nearest_frequency([
                freq_range[1],
                np.min([freq_range[1]*np.sqrt(2), signal.sampling_rate/2])])

            regu_final = _cross_fade(regu_final, regu_outside, idx_xfade_upper)

        regu_final *= np.max(np.abs(data)**2)

    inverse = signal.copy()
    inverse.freq = np.conj(data) / (np.conj(data)*data + regu_final)

    return inverse


def _cross_fade(first, second, indices):
    """Cross-fade two numpy arrays by multiplication with a raised cosine
    window inside the range specified by the indices. Outside the range, the
    result will be the respective first or second array, without distortions.

    Parameters
    ----------
    first : array, double
        The first array.
    second : array, double
        The second array.
    indices : array-like, tuple, int
        The lower and upper cross-fade indices.

    Returns
    -------
    result : array, double
        The resulting array after cross-fading.
    """
    indices = np.asarray(indices)
    if np.shape(first)[-1] != np.shape(second)[-1]:
        raise ValueError("Both arrays need to be of same length.")
    len_arrays = np.shape(first)[-1]
    if np.any(indices > np.shape(first)[-1]):
        raise IndexError("Index is out of range.")

    len_xfade = np.squeeze(np.abs(np.diff(indices)))
    window = sgn.windows.hann(len_xfade*2 + 1, sym=True)
    window_rising = window[:len_xfade]
    window_falling = window[len_xfade+1:]

    window_first = np.concatenate(
        (np.ones(indices[0]), window_falling, np.zeros(len_arrays-indices[1])))
    window_second = np.concatenate(
        (np.zeros(indices[0]), window_rising, np.ones(len_arrays-indices[1])))

    result = first * window_first + second * window_second

    return result


def minimum_phase(signal, n_fft=None, truncate=True):
    """
    Calculate the minimum phase equivalent of a finite impulse response.

    The method is based on the Hilbert transform of the real-valued cepstrum
    of the finite impulse response, that is the cepstrum of the magnitude
    spectrum only. As a result the magnitude spectrum is not distorted.
    Potential aliasing errors can occur due to the Fourier transform based
    calculation of the magnitude spectrum, which however are negligible if the
    length of Fourier transform ``n_fft`` is sufficiently high. [#]_
    (Section 8.5.4)

    Parameters
    ----------
    signal : Signal
        The finite impulse response for which the minimum-phase version is
        computed.
    n_fft : int, optional
        The FFT length used for calculating the cepstrum. Should be at least a
        few times larger than ``signal.n_samples``. The default ``None`` uses
        eight times the signal length rounded up to the next power of two,
        that is: ``2**int(np.ceil(np.log2(n_samples * 8)))``.
    truncate : bool, optional
        If ``truncate`` is ``True``, the resulting minimum phase impulse
        response is truncated to a length of
        ``signal.n_samples//2 + signal.n_samples % 2``. This avoids
        aliasing described above in any case but might distort the magnitude
        response if ``signal.n_samples`` is to low. If truncate is ``False``
        the output signal has the same length as the input signal. The default
        is ``True``.

    Returns
    -------
    signal_minphase : Signal
        The minimum phase version of the input data.

    References
    ----------
    .. [#]  J. S. Lim and A. V. Oppenheim, Advanced topics in signal
            processing, pp. 472-473, First Edition. Prentice Hall, 1988.

    Examples
    --------

    Create a minimum phase equivalent of a linear phase FIR low-pass filter

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> from scipy.signal import remez
        >>> import matplotlib.pyplot as plt
        >>> freq = [0, 0.2, 0.3, 1.0]
        >>> h_linear = pf.Signal(remez(151, freq, [1, 0], Hz=2.), 44100)
        >>> # create minimum phase impulse responses
        >>> h_min = pf.dsp.minimum_phase(h_linear, truncate=False)
        >>> # plot the result
        >>> fig, axs = plt.subplots(3, figsize=(8, 6))
        >>> pf.plot.time(h_linear, ax=axs[0])
        >>> pf.plot.time(h_min, ax=axs[0])
        >>> axs[0].grid(True)
        >>> pf.plot.freq(h_linear, ax=axs[1])
        >>> pf.plot.group_delay(h_linear, ax=axs[2], unit="ms")
        >>> pf.plot.freq(h_min, ax=axs[1])
        >>> pf.plot.group_delay(h_min, ax=axs[2], unit="ms")
        >>> axs[2].legend(['Linear', 'Minimum'], loc=3, ncol=2)
        >>> axs[2].set_ylim(-2.5, 2.5)

    """
    from scipy.fft import fft, ifft

    workers = multiprocessing.cpu_count()
    # center the energy by taking the linear phase signal (using n_samples//2
    # performs better than using n_samples/2)
    signal = pyfar.dsp.linear_phase(
        signal, signal.n_samples // 2, unit='samples')

    if n_fft is None:
        n_fft = 2**int(np.ceil(np.log2(signal.n_samples * 8)))
    elif n_fft < signal.n_samples:
        raise ValueError((
            f"n_fft is {n_fft} but must be at least {signal.n_samples}, "
            "which is the length of the input signal"))

    # add eps to the magnitude spectrum to avoid nans in log
    H = np.abs(fft(signal.time, n=n_fft, workers=workers, axis=-1))
    H[H == 0] = np.finfo(float).eps

    # calculate the minimum phase using the Hilbert transform
    phase = -np.imag(sgn.hilbert(np.log(H), N=n_fft, axis=-1))
    data = ifft(H*np.exp(1j*phase), axis=-1, workers=workers).real

    # cut to length
    if truncate:
        N = signal.n_samples // 2 + signal.n_samples % 2
        data = data[..., :N]
    else:
        data = data[..., :signal.n_samples]

    return pyfar.Signal(data, signal.sampling_rate)


def pad_zeros(signal, pad_width, mode='after'):
    """Pad a signal with zeros in the time domain.

    Parameters
    ----------
    signal : Signal
        The signal which is to be extended.
    pad_width : int
        The number of samples to be padded.
    mode : str, optional
        The padding mode:

        ``'after'``
            Append zeros to the end of the signal
        ``'before'``
            Pre-pend zeros before the starting time of the signal
        ``'center'``
            Insert the number of zeros in the middle of the signal.
            This mode can be used to pad signals with a symmetry with respect
            to the time ``t=0``.

        The default is ``'after'``.

    Returns
    -------
    Signal
        The zero-padded signal.

    Examples
    --------
    >>> import pyfar as pf
    >>> impulse = pf.signals.impulse(512, amplitude=1)
    >>> impulse_padded = pf.dsp.pad_zeros(impulse, 128, mode='after')

    """

    if not isinstance(signal, pyfar.Signal):
        raise TypeError('Input data has to be of type: Signal.')

    padded_signal = signal.flatten()

    if mode in ['after', 'center']:
        pad_array = ((0, 0), (0, pad_width))
    elif mode == 'before':
        pad_array = ((0, 0), (pad_width, 0))
    else:
        raise ValueError("Unknown padding mode.")

    if mode == 'center':
        shift_samples = int(np.round(signal.n_samples/2))
        padded_signal.time = np.roll(
            padded_signal.time, shift_samples, axis=-1)

    padded_signal.time = np.pad(
        padded_signal.time, pad_array, mode='constant')

    if mode == 'center':
        padded_signal.time = np.roll(
            padded_signal.time, -shift_samples, axis=-1)

    padded_signal = padded_signal.reshape(signal.cshape)

    return padded_signal


def time_shift(
        signal, shift, mode='cyclic', unit='samples', pad_value=0.):
    """Apply a cyclic or linear time-shift to a signal.

    This function only allows integer value sample shifts. If unit ``'time'``
    is used, the shift samples will be rounded to the nearest integer value.
    For a shift using fractional sample values see
    :py:func:`~pf.dsp.fractional_time_shift`.

    Parameters
    ----------
    signal : Signal
        The signal to be shifted
    shift : int, float
        The time-shift value. A positive value will result in right shift on
        the time axis (delaying of the signal), whereas a negative value
        yields a left shift on the time axis (non-causal shift to a earlier
        time). If a single value is given, the same time shift will be applied
        to each channel of the signal. Individual time shifts for each channel
        can be performed by passing an array matching the signals channel
        dimensions ``cshape``.
    mode : str, optional
        The shifting mode

        ``"linear"``
            Apply linear shift, i.e., parts of the signal that are shifted to
            times smaller than 0 samples and larger than ``signal.n_samples``
            disappear. To maintain the shape of the signal, the signal is
            padded at the respective other end. The pad value is determined by
            ``pad_type``.
        ``"cyclic"``
            Apply a cyclic shift, i.e., parts of the signal that are shifted to
            values smaller than 0 are wrapped around to the end, and parts that
            are shifted to values larger than ``signal.n_samples`` are wrapped
            around to the beginning.

        The default is ``"cyclic"``
    unit : str, optional
        Unit of the shift variable, this can be either ``'samples'`` or ``'s'``
        for seconds. By default ``'samples'`` is used. Note that in the case
        of specifying the shift time in seconds, the value is rounded to the
        next integer sample value to perform the shift.
    pad_type : numeric, optional
        The pad value for linear shifts, by default ``0.`` is used.
        Pad ``numpy.nan`` to the respective channels if the rms value of the
        signal is to be maintained for block-wise rms estimation of the noise
        power of a signal. Note that if NaNs are padded, the returned data
        will be a :py:class:`~pyfar.classes.audio.TimeData` instead of
        :py:class:`~pyfar.classes.audio.Signal` object.

    Returns
    -------
    Signal, TimeData
        The time-shifted signal. This is a
        :py:class:`~pyfar.classes.audio.TimeData` object in case a linear shift
        was done and the signal was padded with Nans. In all other cases, a
        :py:class:`~pyfar.classes.audio.Signal` object is returend.

    Examples
    --------
    Individually do a cyclic shift of a set of ideal impulses stored in three
    different channels and plot the resulting signals

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> # generate and shift the impulses
        >>> impulse = pf.signals.impulse(
        ...     32, amplitude=(1, 1.5, 1), delay=(14, 15, 16))
        >>> shifted = pf.dsp.time_shift(impulse, [-2, 0, 2])
        >>> # time domain plot
        >>> pf.plot.use('light')
        >>> _, axs = plt.subplots(2, 1)
        >>> pf.plot.time(impulse, ax=axs[0])
        >>> pf.plot.time(shifted, ax=axs[1])
        >>> axs[0].set_title('Original signals')
        >>> axs[1].set_title('Shifted signals')
        >>> plt.tight_layout()

    Perform a linear time shift instead and pad with NaNs

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> # generate and shift the impulses
        >>> impulse = pf.signals.impulse(
        ...     32, amplitude=(1, 1.5, 1), delay=(14, 15, 16))
        >>> shifted = pf.dsp.time_shift(
        ...     impulse, [-2, 0, 2], mode='linear', pad_value=np.nan)
        >>> # time domain plot
        >>> pf.plot.use('light')
        >>> _, axs = plt.subplots(2, 1)
        >>> pf.plot.time(impulse, ax=axs[0])
        >>> pf.plot.time(shifted, ax=axs[1])
        >>> axs[0].set_title('Original signals')
        >>> axs[1].set_title('Shifted signals')
        >>> plt.tight_layout()

    """
    if mode not in ["linear", "cyclic"]:
        raise ValueError(f"mode is '{mode}' but mist be 'linear' or cyclic'")

    shift = np.broadcast_to(shift, signal.cshape)
    if unit == 's':
        shift_samples = np.round(shift*signal.sampling_rate).astype(int)
    elif unit == 'samples':
        shift_samples = shift.astype(int)
    else:
        raise ValueError(
            f"unit is '{unit}' but must be 'samples' or 's'.")

    if np.any(np.abs(shift_samples) > signal.n_samples) and mode == "linear":
        raise ValueError(("Can not shift by more samples than signal.n_samples"
                          " if mode is 'linear'"))

    shifted = signal.copy()
    for ch in np.ndindex(signal.cshape):
        shifted.time[ch] = np.roll(
            shifted.time[ch],
            shift_samples[ch],
            axis=-1)

        if mode == 'linear':
            if shift_samples[ch] > 0:
                samples = slice(0, shift_samples[ch])
                shifted.time[ch + (samples, )] = pad_value
            elif shift_samples[ch] < 0:
                samples = slice(shifted.n_samples + shift_samples[ch],
                                shifted.n_samples)
                shifted.time[ch + (samples, )] = pad_value

    if np.any(np.isnan(shifted.time)):
        shifted = pyfar.TimeData(
            shifted.time, shifted.times, comment=shifted.comment)

    return shifted


def deconvolve(system_output, system_input, fft_length=None, **kwargs):
    r"""Calculate transfer functions by spectral deconvolution of two signals.

    The transfer function :math:`H(\omega)` is calculated by spectral
    deconvolution (spectral division).

    .. math::

        H(\omega) = \frac{Y(\omega)}{X(\omega)},

    where :math:`X(\omega)` is the system input signal and :math:`Y(\omega)`
    the system output. Regularized inversion is used to avoid numerical issues
    in calculating :math:`X(\omega)^{-1} = 1/X(\omega)` for small values of
    :math:`X(\omega)`
    (see :py:func:`~pyfar.dsp.regularized_spectrum_inversion`).
    The system response (transfer function) is thus calculated as

    .. math::

        H(\omega) = Y(\omega)X(\omega)^{-1}.

    For more information, refer to [#]_.

    Parameters
    ----------
    system_output : Signal
        The system output signal (e.g., recorded after passing a device under
        test).
        The system output signal is zero padded, if it is shorter than the
        system input signal.
    system_input : Signal
        The system input signal (e.g., used to perform a measurement).
        The system input signal is zero padded, if it is shorter than the
        system output signal.
    fft_length: int or None
        The length the signals system_output and system_input are zero padded
        to before deconvolving. The default is None. In this case only the
        shorter signal is padded to the length of the longer signal, no padding
        is applied when both signals have the same length.
    kwargs : key value arguments
        Key value arguments to control the inversion of :math:`H(\omega)` are
        passed to to :py:func:`~pyfar.dsp.regularized_spectrum_inversion`.


    Returns
    -------
    system_response : Signal
        The resulting signal after deconvolution, representing the system
        response (the transfer function).
        The ``fft_norm`` of is set to ``'none'``.

    References
    -----------
    .. [#] S. Mueller and P. Masserani "Transfer function measurement with
           sweeps. Directors cut." J. Audio Eng. Soc. 49(6):443-471,
           (2001, June).
    """

    # Check if system_output and system_input are both type Signal
    if not isinstance(system_output, pyfar.Signal):
        raise TypeError('system_output has to be of type pyfar.Signal')
    if not isinstance(system_input, pyfar.Signal):
        raise TypeError('system_input has to be of type pyfar.Signal')

    # Check if both signals have the same sampling rate
    if not system_output.sampling_rate == system_input.sampling_rate:
        raise ValueError("The two signals have different sampling rates!")

    # Set fft_length to the max n_samples of both signals,
    # if it is not explicitly set to a value
    if fft_length is None:
        fft_length = np.max([system_output.n_samples, system_input.n_samples])
    # Check if both signals length are shorter or the same as fft_length
    if fft_length < system_output.n_samples:
        raise ValueError("The fft_length can not be shorter than" +
                         "system_output.n_samples.")
    if fft_length < system_input.n_samples:
        raise ValueError("The fft_length can not be shorter than" +
                         "system_input.n_samples.")

    # Check if both signals have the same length as ftt_length,
    # if not: bring them to the same length by padding with zeros
    system_output = pyfar.dsp.pad_zeros(system_output,
                                        (fft_length - system_output.n_samples))
    system_input = pyfar.dsp.pad_zeros(system_input,
                                       (fft_length - system_input.n_samples))

    # multiply system_output signal with regularized inversed system_input
    # signal to get the system response
    system_response = (system_output *
                       regularized_spectrum_inversion(system_input, **kwargs))

    # Check if the signals have any comments,
    # if yes: concatenate the comments for the system_response
    system_response.comment = "Calculated with pyfar.dsp.deconvolve."
    if system_output.comment != 'none':
        system_response.comment += f" system input: {system_output.comment}."
    if system_input.comment != 'none':
        system_response.comment += f" system output: {system_input.comment}."

    # return the impulse resonse
    system_response.fft_norm = pyfar.classes.audio._match_fft_norm(
        system_output.fft_norm, system_input.fft_norm, division=True)

    return system_response


def convolve(signal1, signal2, mode='full', method='overlap_add'):
    """Convolve two signals.

    Parameters
    ----------
    signal1 : Signal
        The first signal
    signal2 : Signal
        The second signal
    mode : string, optional
        A string indicating the size of the output:

        ``'full'``
            Compute the full discrete linear convolution of
            the input signals. The output has the length
            ``'signal1.n_samples + signal2.n_samples - 1'`` (Default).
        ``'cut'``
            Compute the complete convolution with ``full`` and truncate the
            result to the length of the longer signal.
        ``'cyclic'``
            The output is the cyclic convolution of the signals, where the
            shorter signal is zero-padded to fit the length of the longer
            one. This is done by computing the complete convolution with
            ``'full'``, adding the tail (i.e., the part that is truncated
            for ``mode='cut'`` to the beginning of the result) and
            truncating the result to the length of the longer signal.

    method : str {'overlap_add', 'fft'}, optional
        A string indicating which method to use to calculate the convolution:

        ``'overlap_add'``
            Convolve using  the overlap-add algorithm based
            on ``scipy.signal.oaconvolve``. (Default)
        ``'fft'``
            Convolve using FFT based on ``scipy.signal.fftconvolve``.

        See Notes for more details.

    Returns
    -------
    Signal
        The convolution result as a Signal object.

    Notes
    -----
    The overlap-add method is generally much faster than fft convolution when
    one signal is much larger than the other, but can be slower when only a few
    output values are needed or when the signals have a very similar length.
    For ``method='overlap_add'``, integer data will be cast to float.

    Examples
    --------
    Illustrate the different modes.

    .. plot::

        >>> import pyfar as pf
        >>> s1 = pf.Signal([1, 0.5, 0.5], 1000)
        >>> s2 = pf.Signal([1,-1], 1000)
        >>> full = pf.dsp.convolve(s1, s2, mode='full')
        >>> cut = pf.dsp.convolve(s1, s2, mode='cut')
        >>> cyc = pf.dsp.convolve(s1, s2, mode='cyclic')
        >>> # Plot input and output
        >>> with pf.plot.context():
        >>>     fig, ax = plt.subplots(2, 1, sharex=True)
        >>>     pf.plot.time(s1, ax=ax[0], label='Signal 1', marker='o')
        >>>     pf.plot.time(s2, ax=ax[0], label='Signal 2', marker='o')
        >>>     ax[0].set_title('Input Signals')
        >>>     ax[0].legend()
        >>>     pf.plot.time(full, ax=ax[1], label='full', marker='o')
        >>>     pf.plot.time(cut, ax=ax[1], label='cut', ls='--',  marker='o')
        >>>     pf.plot.time(cyc, ax=ax[1], label='cyclic', ls=':', marker='o')
        >>>     ax[1].set_title('Convolution Result')
        >>>     ax[1].set_ylim(-1.1, 1.1)
        >>>     ax[1].legend()
        >>>     fig.tight_layout()


    """
    if not signal1.sampling_rate == signal2.sampling_rate:
        raise ValueError("The sampling rates do not match")
    fft_norm = pyfar.classes.audio._match_fft_norm(
        signal1.fft_norm, signal2.fft_norm)
    if mode not in ['full', 'cut', 'cyclic']:
        raise ValueError(
            f"Invalid mode {mode}, needs to be "
            "'full', 'cut' or 'cyclic'.")

    if method == 'overlap_add':
        res = sgn.oaconvolve(signal1.time, signal2.time, mode='full', axes=-1)
    elif method == 'fft':
        res = sgn.fftconvolve(signal1.time, signal2.time, mode='full', axes=-1)
    else:
        raise ValueError(
            f"Invalid method {method}, needs to be 'overlap_add' or 'fft'.")

    if mode == 'cut':
        res = res[..., :np.max((signal1.n_samples, signal2.n_samples))]
    elif mode == 'cyclic':
        n_min = np.min((signal1.n_samples, signal2.n_samples))
        n_max = np.max((signal1.n_samples, signal2.n_samples))
        res[..., :n_min-1] += res[..., -n_min+1:]
        res = res[..., :n_max]

    return pyfar.Signal(
        res, signal1.sampling_rate, domain='time', fft_norm=fft_norm)


def decibel(signal, domain='freq', log_prefix=None, log_reference=1,
            return_prefix=False):
    r"""Convert data of the selected signal domain into decibels (dB).

    The converted data is calculated by the base 10 logarithmic scale:
    ``data(dB) = log_prefix * numpy.log10(data/log_reference)``. By using a
    logarithmic scale, the deciBel is able to compare quantities that
    may have vast ratios between them. As an example, the sound pressure in
    dB can be calculated as followed:

    .. math::

        L_p = 20\log_{10}\biggl(\frac{p}{p_0}\biggr),

    where :math:`20` is the logarithmic prefix for sound field quantities and
    :math:`p_0` would be the reference for the sound pressure level. A list
    of commonly used reference values can be found in the 'log_reference'
    parameters section.

    Parameters
    ----------
    signal : Signal, TimeData, FrequencyData
        The signal which is converted into decibel
    domain : str
        The domain, that is converted to decibels:

        ``'freq'``
            Convert normalized frequency domain data. Signal must be of type
            'Signal' or 'FrequencyData'.
        ``'time'``
            Convert time domain data. Signal must be of type
            'Signal' or 'TimeData'.
        ``'freq_raw'``
            Convert frequency domain data without normalization. Signal must be
            of type 'Signal'.

        The default is ``'freq'``.
    log_prefix : int
        The prefix for the dB calculation. The default ``None``, uses ``10``
        for signals with ``'psd'`` and ``'power'`` FFT normalization and
        ``20`` otherwise.
    log_reference : int or float
        Reference for the logarithm calculation.
        List of commonly used values:

        +---------------------------------+--------------+
        | log_reference                   | value        |
        +=================================+==============+
        | Digital signals (dBFs)          | 1            |
        +---------------------------------+--------------+
        | Sound pressure :math:`L_p` (dB) | 2e-5 Pa      |
        +---------------------------------+--------------+
        | Voltage :math:`L_V` (dBu)       | 0.7746 volt  |
        +---------------------------------+--------------+
        | Sound intensity :math:`L_I` (dB)| 1e-12 W/m²   |
        +---------------------------------+--------------+
        | Voltage :math:`L_V` (dBV)       | 1 volt       |
        +---------------------------------+--------------+
        | Electric power :math:`L_P` (dB) | 1 watt       |
        +---------------------------------+--------------+

        The default is 1.
    return_prefix : bool, optional
        If return_prefix is ``True``, the function will also return the
        `log_prefix` value. This can be used to delogrithmize the data. The
        default is ``False``.
    Returns
    -------
    decibel : numpy.ndarray
        The given signal in decibel in chosen domain.
    log_prefix : int or float
        Will be returned if `return_prefix` is set to ``True``.

    Examples
    --------
    >>> import pyfar as pf
    >>> signal = pf.signals.noise(41000, rms=[1, 1])
    >>> decibel_data = decibel(signal, domain='time')
    """
    if log_prefix is None:
        if isinstance(signal, pyfar.Signal) and signal.fft_norm in ('power',
                                                                    'psd'):
            log_prefix = 10
        else:
            log_prefix = 20
    if domain == 'freq':
        if isinstance(signal, (pyfar.FrequencyData, pyfar.Signal)):
            data = signal.freq.copy()
        else:
            raise ValueError(
                f"Domain is '{domain}' and signal is type '{signal.__class__}'"
                " but must be of type 'Signal' or 'FrequencyData'.")
    elif domain == 'time':
        if isinstance(signal, (pyfar.TimeData, pyfar.Signal)):
            data = signal.time.copy()
        else:
            raise ValueError(
                f"Domain is '{domain}' and signal is type '{signal.__class__}'"
                " but must be of type 'Signal' or 'TimeData'.")
    elif domain == 'freq_raw':
        if isinstance(signal, (pyfar.Signal)):
            data = signal.freq_raw.copy()
        else:
            raise ValueError(
                f"Domain is '{domain}' and signal is type '{signal.__class__}'"
                " but must be of type 'Signal'.")
    else:
        raise ValueError(
            f"Domain is '{domain}', but has to be 'time', 'freq',"
            " or 'freq_raw'.")
    data[data == 0] = np.finfo(float).eps
    if return_prefix is True:
        return log_prefix * np.log10(np.abs(data) / log_reference), log_prefix
    else:
        return log_prefix * np.log10(np.abs(data) / log_reference)


def average(signal, mode='time', axis=None, keepdims=False, weights=None):
    """
    Average multichannel Signals.

    Parameters
    ----------
    signal: Signal
        Input signal of the Signal class
    mode: string

        ``'time'``
            average in time domain. Note that this might cause artifacts if
            the data is not aligned across channels.
        ``'complex'``
            average the complex spectra. Note that this might cause artifacts
            if the data is not aligned across channels.
        ``'magnitude_zerophase'``
            average the magnitude spectra and discard the phase
        ``'magnitude_phase'``
            average the magnitude spectra and keep the phase
        ``'power'``
            average the power spectra $|X|^2$ and discard the phase(zerophase).
            The squaring of the spectra is reversed before returning the
            averaged signal.
        ``'log_magnitude_zerophase'``
            average the log. magnitude spectra $20 \\log_{10}(X)$ and discard
            the phase. The logarithm is reversed before returning the averaged
            signal.

        The default is ``'time'``
    axis: None or int or tuple of ints, optional
        Axis or axes along which the averaging is done. Can be None, which will
        take all channels to average the multichannel signal.  If axis is
        negative it counts from the last to the first axis. If axis is a tuple
        of ints, average will perform on the channels specified in the tuple.
        The default is ``None``.
    keepdims: bool, optional
        If this is true, the axis which are reduced during the averaging are
        kept as a dimension with size one. Also, the output will broadcast
        correctly with the input Signal. If keepdims is False and the converted
        domain data after summing up got more then two dimensions, the data
        will be also squeezed to the shape of axis.
        The default is ``False``.
    weights: array like
        array that gives channel weights for averaging the data. Must be of
        shape which can be broadcasted to ``signal.cshape``.
        The default is ``None``, which applies equal weights to all channels.
    Returns
    --------
    averaged_signal: Signal
        averaged input Signal

    Notes
    -----
    The functions :py:func:`~pyfar.dsp.linear_phase` and
    :py:func:`~pyfar.dsp.minimum_phase` can be used to obtain non-zero phase
    responses. This can be usefull if the average mode discards the phase.
    """

    # check input
    if not isinstance(signal, (pyfar.Signal, pyfar.FrequencyData,
                               pyfar.TimeData)):
        raise TypeError(("Input data has to be of type 'Signal', 'TimeData' "
                         "or 'FrequencyData'."))
    if type(signal) == pyfar.FrequencyData and mode == 'time':
        raise ValueError((
            f"mode is '{mode}' and signal is type '{signal.__class__}'"
            " but must be of type 'Signal' or 'TimeData'."))
    if type(signal) == pyfar.TimeData and mode in ('log_magnitude_zerophase',
                                                   'magnitude_zerophase',
                                                   'magnitude_phase', 'power',
                                                   'complex'):
        raise ValueError((
            f"mode is '{mode}' and signal is type '{signal.__class__}'"
            " but must be of type 'Signal' or 'FrequencyData'."))
    # check for axis
    if axis and np.max(axis) > len(signal.cshape):
        raise ValueError('The maximum of axis needs to be smaller then '
                         'len(signal.cshape).')

    # set axis default
    if axis is None:
        axis = tuple([i for i in range(len((signal.cshape)))])
    # check if averaging over one dimensional axis
    if 1 in signal.cshape:
        for i, ax in enumerate(axis):
            if signal.cshape[ax] == 1:
                warnings.warn(f"Averaging one dimensional axis={axis}.")
    # set weights default
    if weights is None:
        weights = 1
        for idx, size in enumerate(signal.cshape):
            if idx in axis:
                weights /= signal.cshape[idx]
    # apply weights
    signal *= weights

    if not isinstance(axis, int):
        axis = tuple([ax-1 if ax < 0 else ax for ax in axis])
    # convert data to desired domain
    if mode == 'time':
        data = signal.time
    elif mode == 'complex':
        data = signal.freq
    elif mode == 'magnitude_zerophase':
        data = np.abs(signal.freq)
    elif mode == 'magnitude_phase':
        data = [np.abs(signal.freq), pyfar.dsp.phase(signal, unwrap=True)]
    elif mode == 'power':
        data = np.abs(signal.freq)**2
    elif mode == 'log_magnitude_zerophase':
        data, log_prefix = pyfar.dsp.decibel(signal, 'freq',
                                             prefix_return=True)
    else:
        raise ValueError(
            """mode must be 'time', 'complex', 'magnitude_zerophase', 'power',
            'magnitude_phase' or 'log_magnitude_zerophase'."""
            )
    # average the data
    if mode == 'magnitude_phase':
        data = [np.sum(d, axis=axis, keepdims=False) for d in data]
        data = data[0]*np.exp(1j*data[1])
    else:
        data = np.sum(data, axis=axis, keepdims=keepdims)
    if mode == 'power':
        data = np.sqrt(data)
    elif mode == 'log_magnitude_zerophase':
        data = 10**(data/log_prefix)
    # input data into averaged_signal
    averaged_signal = signal.copy()
    if mode == 'time':
        averaged_signal.time = data
    else:
        averaged_signal.freq = data

    return averaged_signal
