import numpy as np
from scipy import signal as sgn
from pyfar import Signal


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


def group_delay(signal):
    """Returns the group delay of a signal in samples.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar signal class

    Returns
    -------
    group_delay : numpy array
        Frequency dependent group delay in samples. The array is flattened if
        a single channel signal was passed to the function.
    """

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    # get time signal and reshape for easy looping
    time = signal.time
    time = time.reshape((np.prod(signal.cshape), signal.n_samples))
    # initialize group delay
    group_delay = np.zeros((np.prod(signal.cshape), signal.n_bins))
    # calculate the group delay
    for cc in range(time.shape[0]):
        group_delay[cc] = sgn.group_delay(
            (time[cc], 1), signal.frequencies, fs=signal.sampling_rate)[1]
    # reshape to match signal
    group_delay = group_delay.reshape(signal.cshape + (signal.n_bins, ))

    # flatten in numpy fashion if a single channel is returned
    if signal.cshape == (1, ):
        group_delay = np.squeeze(group_delay)

    return group_delay


def spectrogram(signal,
                window='hann',
                window_length='auto',
                window_overlap_fct=0.5,
                log_prefix=20,
                log_reference=1,
                log=False,
                nodb=False,
                cut=False,
                clim=None):
    """TODO: This function might not be necessary, if clipping functionallity
     is not desired. It is already provided by scipy.signal.spectrogram().

    Generates the spectrogram for a given signal object including frequency,
    time and colorbar-limit vectors.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar signal class
    window : String (Default: 'hann')
        Specifies the window type. See scipy.signal.get_window for details.
    window_length : integer
        Specifies the window length. If not set, it will be automatically
        calculated.
    window_overlap_fct : double
        Ratio of points to overlap between fft segments [0...1]
    log_prefix : integer
        Prefix for Decibel calculation.
    log_reference : double
        Prefix for Decibel calculation.
    log : Boolean
        Speciefies, whether the y axis is plotted logarithmically.
        Defaults to False.
    nodb : Boolean
        Speciefies, whether the spectrogram is plotted in decibels.
        Defaults to False.
    cut : Boolean
        Cut results to specified clim vector to avoid sparcles.
        Defaults to False.
    clim : np.array()
        Array of limits for the colorbar [lower, upper].

    **kwargs
        Keyword arguments that are piped to matplotlib.pyplot.plot

    Returns
    -------
    frequencies : np.array()
        Array of sample frequencies.
    times : np.array()
        Array of segment times.
    spectrogram : np.array()
        Spectrogram of signal. By default, the last axis of Sxx corresponds to
        the segment times.
    clim : np.array()
        Array of limits for the colorbar [lower, upper]

    See Also
    --------
    scipy.signal.spectrogram() : Generate the spectrogram for a given signal.
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    eps = np.finfo(float).tiny

    # If not set, define window length for FFT automatically
    if window_length == 'auto':
        window_length = 2**nextpow2(signal.n_samples / 2000)
        if window_length < 1024:
            window_length = 1024
    window_overlap = int(window_length * window_overlap_fct)

    frequencies, times, spectrogram = sgn.spectrogram(
            x=signal.time,
            fs=signal.sampling_rate,
            window=sgn.get_window(window, window_length),
            noverlap=window_overlap,
            mode='magnitude',
            scaling='spectrum')

    spectrogram = spectrogram/(window_length/2)

    if nodb:
        spectrogram = np.abs(spectrogram)
    else:
        spectrogram = log_prefix*np.log10(
            np.abs(spectrogram) / log_reference + eps)

    # Get CLIMs, TODO: Verify, whether clim was passed correctly.
    if not clim:
        upper = 10 * np.ceil(np.max(np.max(spectrogram))/10)
        lower = upper - 70
        clim = np.array([lower, upper])

    # Cut results to CLIM to avoid sparcles?
    if cut:
        spectrogram[spectrogram < clim[0]-20] = clim[0] - 20
        spectrogram[spectrogram > clim[1]+20] = clim[1] + 20
    spectrogram = np.squeeze(spectrogram)

    return frequencies, times, spectrogram, clim


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
