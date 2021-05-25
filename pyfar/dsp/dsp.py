import warnings
import numpy as np
from scipy import signal as sgn
import pyfar
from pyfar.dsp import fft


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
        If set to "360", the phase is wrapped to 2 pi.

    Returns
    -------
    phase : np.array()
        Phase.
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
    signal : Signal object
        An audio signal object from the pyfar signal class
    frequencies : number array like
        Frequency or frequencies in Hz at which the group delay is calculated.
        The default is None, in which case signal.frequencies is used.
    method : 'scipy', 'fft', optional
        Method to calculate the group delay of a Signal. Both methods calculate
        the group delay using the method presented in [#]_ avoiding issues
        due to discontinuities in the unwrapped phase. Note that the scipy
        version additionally allows to specify frequencies for which the
        group delay is evaluated. The default is 'fft', which is faster.

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

        freq = fft.normalization(
            signal.freq, signal.n_samples, signal.sampling_rate,
            signal.fft_norm, inverse=True)

        group_delay = np.real(freq_k / freq)

        # catch zeros in the denominator
        group_delay[np.abs(freq) < 1e-15] = 0

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
        Phase wrapped to 2 pi.
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
        Unit of the group delay. Can be ``'s'`` for seconds, ``'ms'`` for
        milliseconds, ``'mus'`` for microseconds, or ``'samples'``. The
        default is ``'samples'``.

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
    elif unit == 'ms':
        tau = np.asarray(group_delay) / 1e3
    elif unit == 'mus':
        tau = np.asarray(group_delay) / 1e6
    else:
        raise ValueError("unit must be 'samples', 's', 'ms', or 'mus'.")

    # linear phase
    phase = 2 * np.pi * signal.frequencies * tau[..., np.newaxis]

    # construct linear phase spectrum
    signal_lin = signal.copy()
    signal_lin.freq = \
        np.abs(signal_lin.freq).astype(complex) * np.exp(-1j * phase)

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
    signal_zero.freq = np.atleast_2d(np.abs(signal_zero.freq))

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


def spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                window='hann', window_length=1024, window_overlap_fct=0.5):
    """Compute the magnitude spectrum versus time.

    This is a wrapper for scipy.signal.spectogram with two differences. First,
    the returned times refer to the start of the FFT blocks, i.e., the first
    time is always 0 whereas it is window_length/2 in scipy. Second, the
    returned spectrogram is normalized accroding to `signal.signal_type` and
    `signal.fft_norm`.

    Parameters
    ----------
    signal : Signal
        pyfar Signal object.
    db : Boolean
        False to plot the logarithmic magnitude spectrum. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is 1.
    window : str
        Specifies the window (See scipy.signal.get_window). The default is
        'hann'.
    window_length : integer
        Specifies the window length in samples. The default ist 1024.
    window_overlap_fct : double
        Ratio of points to overlap between fft segments [0...1]. The default is
        0.5

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

    # get spectrogram from scipy.signal
    window_overlap = int(window_length * window_overlap_fct)
    window = sgn.get_window(window, window_length)

    frequencies, times, spectrogram = sgn.spectrogram(
        x=signal.time.squeeze(), fs=signal.sampling_rate, window=window,
        noverlap=window_overlap, mode='magnitude', scaling='spectrum')

    # remove normalization from scipy.signal.spectrogram
    spectrogram /= np.sqrt(1 / window.sum()**2)

    # apply normalization from signal
    spectrogram = fft.normalization(
        spectrogram, window_length, signal.sampling_rate,
        signal.fft_norm, window=window)

    # scipy.signal takes the center of the DFT blocks as time stamp we take the
    # beginning (looks nicer in plots, both conventions are used)
    times -= times[0]

    return frequencies, times, spectrogram


def time_window(signal, interval, window='hann', shape='symmetric',
                unit='samples', crop='none'):
    """Apply time window to signal.

    This function uses the windows implemented in ``scipy.signal.windows``.

    Parameters
    ----------
    signal : Signal
        pyfar Signal object to be windowed
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
        Unit of `interval`. Can be set to ``'s'`` (seconds), ``'ms'``
        (milliseconds) or ``'samples'``.
        Time values are rounded to the nearest sample.
        The default is ``'samples'``.
    crop : string, optional
        ``'none'``
            The length of the windowed signal stays the same.
        ``'window'``
            The signal is truncated to the windowed part
        ``'end'``
            Only the zeros at the end of the windowed signal are
            cropped, so the original phase is preserved.

        The default is ``'none'``.

    Returns
    -------
    signal_windowed : Signal
        Windowed signal object

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

    >>> import pyfar
    >>> import numpy as np
    >>>
    >>> signal = pyfar.Signal(np.ones(100), 44100)
    >>> for shape in ['symmetric', 'symmetric_zero', 'left', 'right']:
    >>>     signal_windowed = pyfar.dsp.time_window(
    >>>         signal, interval=[25,45], shape=shape)
    >>>     ax = pyfar.plot.time(signal_windowed, label=shape)
    >>> ax.legend(loc='right')

    .. plot::

        import pyfar
        import numpy as np

        signal = pyfar.Signal(np.ones(100), 44100)
        for shape in ['symmetric', 'symmetric_zero', 'left', 'right']:
            signal_windowed = pyfar.dsp.time_window(
                signal, interval=[25,45], shape=shape)
            ax = pyfar.plot.time(signal_windowed, label=shape)
        ax.legend(loc='right')

    Window with fade-in and fade-out defined by four values in `interval`.

    >>> import pyfar
    >>> import numpy as np
    >>>
    >>> signal = pyfar.Signal(np.ones(100), 44100)
    >>> signal_windowed = pyfar.dsp.time_window(
    >>>         signal, interval=[25, 40, 60, 90], window='hann')
    >>> pyfar.plot.time(signal_windowed)

    .. plot::

        import pyfar
        import numpy as np

        signal = pyfar.Signal(np.ones(100), 44100)
        signal_windowed = pyfar.dsp.time_window(
                signal, interval=[25, 40, 60, 90], window='hann')
        pyfar.plot.time(signal_windowed)


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

    interval = np.array(interval)
    if not np.array_equal(interval, np.sort(interval)):
        raise ValueError("Values in interval need to be in ascending order.")
    # Convert to samples
    if unit == 's':
        interval = np.round(interval*signal.sampling_rate).astype(int)
    elif unit == 'ms':
        interval = np.round(interval*signal.sampling_rate/1e3).astype(int)
    elif unit == 'samples':
        interval = interval.astype(int)
    else:
        raise ValueError(f"unit is {unit} but has to be"
                         f" 'samples', 's' or 'ms'.")
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
    if crop == 'end':
        # Add zeros before window
        window_zeropadded = np.zeros(win_stop+1)
        window_zeropadded[win_start:win_stop+1] = win
        signal_win.time = signal_win.time[..., :win_stop+1]*window_zeropadded
    elif crop == 'none':
        # Create zeropadded window
        window_zeropadded = np.zeros(signal.n_samples)
        window_zeropadded[win_start:win_stop+1] = win
        signal_win.time = signal_win.time*window_zeropadded

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
        regu_outside=1., regu_inside=10**(-200/20), regu_final=None):
    r"""Invert the spectrum of a signal applying frequency dependent
    regularization. Regularization can either be specified within a given
    frequency range using two different regularization factors, or for each
    frequency individually using the parameter `regu_final`. In the first case
    the regularization factors for the frequency regions are cross-faded using
    a raised cosine window function with a width of `math:f*\sqrt(2)` above and
    below the given frequency range. Note that the resulting regularization
    function is adjusted to the quadratic maximum of the given signal.
    In case the `regu_final` parameter is used, all remaining options are
    ignored and an array matching the number of frequency bins of the signal
    needs to be given. In this case, no normalization of the regularization
    function is applied. Finally, the inverse spectrum is calculated as
    [#]_, [#]_,

    .. math::

        S^{-1}(f) = \frac{S^*(f)}{S^*(f)S(f) + \epsilon(f)}


    Parameters
    ----------
    signal : pyfar.Signal
        The signals which spectra are to be inverted.
    freq_range : tuple, array_like, double
        The upper and lower frequency limits outside of which the
        regularization factor is to be applied.
    regu_outside : float, optional
        The normalized regularization factor outside the frequency range.
        The default is 1.
    regu_inside : float, optional
        The normalized regularization factor inside the frequency range.
        The default is 10**(-200/20).
    regu_final : float, array_like, optional
        The final regularization factor for each frequency, by default None.
        If this parameter is set, the remaining regularization factors are
        ignored.

    Returns
    -------
    pyfar.Signal
        The resulting signal after inversion.

    References
    ----------
    .. [#]  O. Kirkeby and P. A. Nelson, “Digital Filter Designfor Inversion
            Problems in Sound Reproduction,” J. Audio Eng. Soc., vol. 47,
            no. 7, p. 13, 1999.

    .. [#]  P. C. Hansen, Rank-deficient and discrete ill-posed problems:
            numerical aspects of linear inversion. Philadelphia: SIAM, 1998.

    """
    if not isinstance(signal, pyfar.Signal):
        raise ValueError("The input signal needs to be of type pyfar.Signal.")

    data = signal.freq
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
    window = sgn.windows.windows.hann(len_xfade*2 + 1, sym=True)
    window_rising = window[:len_xfade]
    window_falling = window[len_xfade+1:]

    window_first = np.concatenate(
        (np.ones(indices[0]), window_falling, np.zeros(len_arrays-indices[1])))
    window_second = np.concatenate(
        (np.zeros(indices[0]), window_rising, np.ones(len_arrays-indices[1])))

    result = first * window_first + second * window_second

    return result


def time_shift(signal, shift, unit='samples'):
    """Apply a time-shift to a signal.

    The shift is performed as a cyclic shift on the time axis, potentially
    resulting in non-causal signals for negative shift values.

    Parameters
    ----------
    signal : Signal
        The signal to be shifted
    shift : int, float
        The time-shift value. A positive value will result in right shift on
        the time axis (delaying of the signal), whereas a negative value
        yields a left shift on the time axis (non-causal shift to a earlier
        time).
    unit : str, optional
        Unit of the shift variable, this can be either samples or seconds.
        By default 'samples' is used. Note that in the case of specifying
        the shift time in seconds, the value is rounded to the next integer
        sample value to perform the shift.


    Returns
    -------
    Signal
        The time-shifted signal.
    """
    shift = np.atleast_1d(shift)
    if shift.size == 1:
        shift = np.ones(signal.cshape) * shift

    if unit == 'seconds':
        shift_samples = np.round(shift*signal.sampling_rate).astype(int)
    elif unit == 'samples':
        shift_samples = shift.astype(int)
    else:
        raise ValueError(
            f"Unit is: {unit}, but has to be 'samples' or 'seconds'.")

    if np.any(shift_samples > signal.n_samples):
        warnings.warn(
            "Shifting by more samples than the length of the signal")

    shifted = signal.flatten()
    shift_samples = shift_samples.flatten()
    for ch in range(shifted.cshape[0]):
        shifted.time[ch] = np.roll(
            shifted.time[ch],
            shift_samples[ch],
            axis=-1)

    return shifted.reshape(signal.cshape)
