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
