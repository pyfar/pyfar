import numpy as np
from scipy import signal as sgn
from pyfar import Signal
import pyfar.fft as fft


def phase(signal, deg=False, unwrap=False):
    """Returns the phase for a given signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar signal class
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

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

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


def group_delay(signal, frequencies=None):
    """Returns the group delay of a signal in samples.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar signal class
    frequencies : number array like
        Frequency or frequencies in Hz at which the group delay is calculated.
        The default is None, in which case signal.frequencies is used.

    Returns
    -------
    group_delay : numpy array
        Frequency dependent group delay in samples. The array is flattened if
        a single channel signal was passed to the function.
    """

    # check input and default values
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    frequencies = signal.frequencies if frequencies is None \
        else np.asarray(frequencies)

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


def normalize(signal, norm_type='time', operation='max', channelwise='max',
              value=None, freq_range=(20, 22000)):
    """
    Normalize signal in the time or frequency domain.

    Parameters
    ----------
    signal: Signal
        Input signal of the signal class
    norm_type: string
        'time' - Normalize the time signal to 'value'
        'abs' - Normalize the magnitude spectrum to 'value'
        'dB' - Normalize the db magnitude spectrum to 'value'
        The default is 'time'
    operation: string
        'max' - Normalize to the absolute maximum of the signal data
        'mean' - Normalize to the mean of the signal data
        'rms' - Normalize to the rms of the signal data
        The default is 'max'
    channelwise: string
        'each' - Normalize each channel separately
        'max' - Normalize to the max or operation across all channels
        'min' - Normalize to min or operation across all channels
        'mean' - Normalize to mean of operation across all channels
       The default is 'max'
    value: int
        Normalizes to `value` which can be a scalor or a vector with
        a number of elements equal to channels
        The default is 0 for norm_type='db' and 1 otherwise
    freq_range: tuple
        two element vector specifiying upper and lower frequenzy bounds
        for normalization or scalar specifying the centre frequency for
        normalization
        The default is (20,22000)
    Returns
    --------
    normalized_signal: Signal
        The normalized signal
    """
    # set default
    if value is None:
        value = 0 if norm_type == 'dB' else 1

    # transform data to the desired domain
    if norm_type == 'time':
        normalized_input = np.abs(signal.time.copy())
    elif norm_type == 'abs':
        normalized_input = np.abs(signal.freq.copy())
    elif norm_type == 'dB':
        normalized_input = np.log(signal.freq.copy())
    else:
        raise ValueError(("norm _type must be 'time', 'abs' or 'dB'"))

    # take logarithm of the data here instead of later?

    # get bounds for normalization
    if norm_type == 'time':
        lim = (0, signal.n_samples - 1)

    else:
        lim = signal.find_nearest_frequency(f)

        # remove 0 hz and nyquist due to normalization dependency
        if signal.n_samples % 2:
            lim[0] = np.max([lim[0, 1])
        else:
            lim = np.clip(lim, 0)

    # get values for normalization
    if operation == 'max':
        values = np.max(...,normalized_input[...,lim[0]:lim[1]], axis=-1,
        keepdims=True)
    elif operation == 'mean':
        values = np.mean(...,normalized_input[...,lim[0]:lim[1]],axis=-1,
        keepdims=True)
    elif operation == 'rms':
        values = np.sqrt(np.mean(...,(normalized_input[...,lim[0]:lim[1]])**2,
        axis,=-1, keepdims=True)
    else:
        raise ValueError(("operation must be 'abs', 'mean' or 'rms'"))

    # manipulate values
    if channelwise == 'each':
        pass
    elif channelwise == 'max':
        values = np.max(values)
    elif channelwise == 'min':
        values = np.min(values)
    elif channelwise == 'mean':
        values = np.mean(values)
    else:
        raise ValueError(("channelwise must be 'each', 'max', 'min' or 'mean'")
    )
    # apply normalization
    normalized_input = (input_normalized / values) * value

    # replace input with normlaized_input
    normalized_signal = signal.copy()
    normalized_signal.time = normalized_input

    return normalized_signal
