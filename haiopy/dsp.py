import numpy as np
from scipy import signal as sgn

def wrap_to_2pi(data):
    positive_input = (data > 0)
    zero_check = np.logical_and(positive_input,(data == 0))
    data = np.mod(data, 2*np.pi)
    data[zero_check] = 2*np.pi
    return data

def rad_to_deg(data):
    return data*180/np.pi

def groupdelay(signal): # TODO
    return

def phase(signal, deg=False, unwrap=False): # TODO
    return

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

    See Also
    --------

    Examples
    --------
    """
    nextpower2 = np.ceil(np.log2(x))
    return nextpower2

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
        spectrogram = log_prefix*np.log10(np.abs(spectrogram)/log_reference + eps)

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
