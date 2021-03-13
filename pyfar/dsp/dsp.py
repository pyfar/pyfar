import numpy as np
from scipy import signal as sgn
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
