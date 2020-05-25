import numpy as np
from scipy import signal as sgn
from haiopy import Signal

def group_delay(signal):
    """Generates the group delay for a given signal object including frequency,
    time and colorbar-limit vectors.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class

    Returns
    -------
    group_delay : np.array()
        Group delay.
    """

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    phase_vec = phase(signal, deg=False, unwrap=True)
    bin_dist = signal.sampling_rate / signal.n_samples
    group_delay = - np.diff(phase_vec,1,-1, prepend=0) / (bin_dist * 2*np.pi);
    return np.squeeze(group_delay)

def phase(signal, deg=False, unwrap=False):
    """Generates the spectrogram for a given signal object including frequency,
    time and colorbar-limit vectors.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
    deg : Boolean
        Specifies, whether the phase is plotted in degrees or radians.
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

    if(unwrap==True):
        phase = np.unwrap(phase)
    elif(unwrap=='360'):
        phase = wrap_to_2pi(np.unwrap(phase))

    if deg:
        phase = dsp.rad_to_deg(phase)
    return phase

def spectrogram(signal, window='hann', window_length='auto',
                         window_overlap_fct=0.5, log_prefix=20, log_reference=1,
                         log=False, nodb=False, cut=False, clim=None):
    """Generates the spectrogram for a given signal object including frequency,
    time and colorbar-limit vectors.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy signal class
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

    Examples
    --------
    """
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    eps = np.finfo(float).tiny
    # TODO: Verify against ITAToolbox implementation!

    # If not set, define window length for FFT automatically
    if window_length == 'auto':
        window_length  = 2**nextpow2(signal.n_samples / 2000)
        if window_length < 1024: window_length = 1024
    window_overlap = int(window_length * window_overlap_fct)

    frequencies, times, spectrogram = sgn.spectrogram(
        x=signal.time, fs=signal.sampling_rate,
        window=sgn.get_window(window, window_length), noverlap=window_overlap,
        mode='magnitude', scaling='spectrum')

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
        spectrogram[spectrogram < clim[0]-20] = clim[0]-20;
        spectrogram[spectrogram > clim[1]+20] = clim[1]+20;
    spectrogram = np.squeeze(spectrogram)

    return frequencies, times, spectrogram, clim


### HELPER FUNCTIONS ###

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
    zero_check = np.logical_and(positive_input,(x == 0))
    x = np.mod(x, 2*np.pi)
    x[zero_check] = 2*np.pi
    return x

def rad_to_deg(x):
    """Converts radians to degrees.

    Parameters
    ----------
    x : double
        Input variable to convert to degrees.

    Returns
    -------
    x : double
        Value in degrees.
    """
    return x*180/np.pi

def nextpow2(x):
    """Returns the exponent of next higher power of 2.

    Parameters
    ----------
    x : double
        Input variable to determine the exponent of next higher power of 2.

    Returns
    -------
    nextpower2 : double
        Exponent of next higher power of 2.
    """
    nextpower2 = np.ceil(np.log2(x))
    return nextpower2
