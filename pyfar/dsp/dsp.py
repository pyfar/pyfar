import numpy as np
from scipy import signal as sgn
import pyfar
from pyfar import Signal, FrequencyData
import pyfar.fft as fft


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

    if not isinstance(signal, Signal) and \
            not isinstance(signal, FrequencyData):
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
        the group delay using the method presented in [1]_ avoiding issues
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
    .. [1]  https://www.dsprelated.com/showarticle/69.php
    """

    # check input and default values
    if not isinstance(signal, Signal):
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
        Falg to plot the logarithmic magnitude specturm. The default is True.
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
    if not isinstance(signal, Signal):
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


def normalize(signal, normalize='time', normalize_to='max',
              channel_handling='max', value=None, freq_range=None,
              return_values=False):
    """
    Normalize signal in the time or frequency domain.

    Parameters
    ----------
    signal: Signal
        Input signal of the signal class
    normalize: string
        'time' - Normalize the time signal to 'value'
        'magnitude' - Normalize the magnitude spectrum to 'value'
        'log_magnitude' - Normalize the log magnitude spectrum to 'value'
        The default is 'time'
    normalize_to: string
        'max' - Normalize to the absolute maximum of the signal data
        'mean' - Normalize to the mean of the signal data
        'rms' - Normalize to the rms of the signal data
        The default is 'max'
    channel_handling: string
        'each' - Normalize each channel separately
        'max' - Normalize to the max of 'normalize_to' across all channels
        'min' - Normalize to min of 'normalize_to' across all channels
        'mean' - Normalize to mean of 'normalize_to' across all channels
       The default is 'max'
    value: scalar, array
        Normalizes to `value` which can be a scalar or an array with
        shape equal to signal cshape. The unit of `value`
        is defined by `norm_type`, i.e., it is either dB or linear.
        The default is 0 for normalize='log_magnitude' and 1 otherwise
    freq_range: tuple
        Two element vector specifying upper and lower frequency bounds
        for normalization or scalar specifying the centre frequency for
        normalization
    Returns
    --------
    normalized_signal: Signal
        The normalized signal
    values: numpy array
        If return_values=True returns values, the values of all channels
        before normalization.
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # set default values
    if value is None:
        value = 0 if normalize == 'log_magnitude' else 1
    if freq_range is None:
        freq_range = (0, signal.frequencies[-1])

    # copy and transform data to the desired domain
    if normalize == 'time':
        input_data = np.abs(signal.time.copy())
    elif normalize == 'magnitude':
        input_data = np.abs(signal.freq.copy())
    elif normalize == 'log_magnitude':
        input_data = 20 * np.log10(signal.freq.copy())
    else:
        raise ValueError(
                    "normalize must be 'time', 'magnitude' or 'log_magnitude'")

    # get bounds for normalization
    if normalize == 'time':
        lim = (0, signal.n_samples)

    else:
        lim = signal.find_nearest_frequency(freq_range)
        if signal.n_samples % 2:
            lim[0] = np.max([lim[0], 1])
        else:
            lim = np.clip(lim, 1, signal.n_bins-1)

    # get values for normalization
    if normalize_to == 'max':
        values = np.max(input_data[..., lim[0]:lim[1]], axis=-1,
                        keepdims=True)
    elif normalize_to == 'mean':
        values = np.mean(input_data[..., lim[0]:lim[1]], axis=-1,
                         keepdims=True)
    elif normalize_to == 'rms':
        values = np.sqrt(np.mean(input_data[..., lim[0]:lim[1]]**2,
                         axis=-1, keepdims=True))
    else:
        raise ValueError("normalize_to must be 'max', 'mean' or 'rms'")

    # manipulate values
    if channel_handling == 'each':
        pass
    elif channel_handling == 'max':
        values = np.max(values)
    elif channel_handling == 'min':
        values = np.min(values)
    elif channel_handling == 'mean':
        values = np.mean(values)
    else:
        raise ValueError(
                    "channel_handling must be 'each', 'max', 'min' or 'mean'")

    # de-logarthimize value
    if normalize == 'log_magnitude':
        value = 10**(value/20)

    # replace input with normalized_input
    normalized_signal = signal.copy()
    if normalize == 'time':
        normalized_signal.time = signal.time.copy() / values * value
    else:
        normalized_signal.freq = signal.freq.copy() / values * value

    if return_values:
        return normalized_signal, values
    else:
        return normalized_signal


def average(signal, average_mode='time', phase_copy=None,
            weights=None):
    """
    Used to average multichannel Signals in different ways. You may want to
    align your data first.

    Parameters
    ----------
    signal: Signal
        Input signal of the Signal class
    average_mode: string
        'time' - averages in time domain
        'complex' - averages the complex spectra
        'magnitude' - averages the magnitude spectra
        'power' - averages the power spectra
        'log_magnitude' - averages the log magnitude spectra
        The default is 'time'
    phase_copy: vector
        indicates signal channel from which phase is to be copied to the
        averaged signal
        None - ignores the phase. Resulting in zero phase
        The default is None
    weights: numpy array
        array that gives channel weighting for averaging the data. Must
        have same shape as channel shape of signal.
        The default is None
    Returns
    --------
    averaged_signal: Signal
        averaged input Signal
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal')

    # set weights default
    if weights is None:
        weights = 1/(np.prod(signal.cshape))
    else:
        weights = weights/np.sum(weights)

    # convert data to desired domain
    if average_mode == 'time':
        data = signal.time.copy()
    elif average_mode == 'complex':
        data = signal.freq.copy()
    elif average_mode == 'magnitude':
        data = np.abs(signal.freq.copy())
    elif average_mode == 'power':
        data = np.abs(signal.freq.copy())**2
    elif average_mode == 'log_magnitude':
        data = 20 * np.log10(signal.freq.copy())

    # apply weights
    # NOT SURE IF THIS WORKS WITH MORE THAN FOR SIGNALS GREATER THAN 3D
    data = data * np.transpose(np.array([weights, ]*len(data[-1])))

    # average the data
    if (average_mode == 'time' or
            average_mode == 'complex' or average_mode == 'magnitude'):
        data = np.sum(data, axis=-2, keepdims=True)
    elif average_mode == 'power':
        data = np.sum(data, axis=-2, keepdims=True)
        data = np.sqrt(data)
    elif average_mode == 'log_magnitude':
        data = np.sum(data, axis=-2, keepdims=True)
        data = 10**(data/20)
    else:
        raise ValueError(
            """average_mode must be 'time', 'complex', 'magnitude', 'power' or
            'log_magnitude'"""
            )

    # phase handling
    if phase_copy is None:
        pass
    else:
        if average_mode == 'time':
            data_ang = signal.time.copy()
            # NOT SURE IF THIS COPPIES PHASE_COPY INDEX IN CORRECTLY
            data_ang = np.angle(data_ang[phase_copy, ...])
        else:
            data_ang = signal.freq.copy()
            data_ang = np.angle(data_ang[phase_copy, ...])

        data = data * np.exp(1j * data_ang)

    # input data into averaged_signal
    averaged_signal = signal.copy()
    if average_mode == 'time':
        averaged_signal.time = data
    else:
        averaged_signal.freq = data

    return averaged_signal


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
    [1]_, [2]_,

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
    .. [1]  O. Kirkeby and P. A. Nelson, “Digital Filter Designfor Inversion
            Problems in Sound Reproduction,” J. Audio Eng. Soc., vol. 47,
            no. 7, p. 13, 1999.

    .. [2]  P. C. Hansen, Rank-deficient and discrete ill-posed problems:
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
