"""
Signal generation
"""
import numpy as np


def sine(
        amplitude,
        frequency,
        sampling_rate,
        num_samples,
        full_period=False):
    """Generate a sine signal.

    Parameters
    ----------
    amplitude : double
        The amplitude
    frequency : double
        Frequency of the sine signal. f < f_s/2
    sampling_rate : int
        The sampling rate f_s
    num_samples : int
        Length of the signal in samples
    full_period : bool, optional
        If True, the returned sine signal will have an integer number of
        periods resulting in a periodic signal.

    Returns
    -------
    signal : ndarray, double
        The sine signal

    """

    if full_period:
        num_periods = np.floor(num_samples / sampling_rate * frequency)
        # round to the nearest frequency resulting in a fully periodic sine
        # signal in the given time interval
        frequency = num_periods * sampling_rate / num_samples
    times = np.arange(0, num_samples) / sampling_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * times)

    return signal


def exponential_sweep(
        amplitude,
        sampling_rate,
        num_samples,
        frequency_range,
        sweeprate=None):
    """Generate an exponential sine sweep signal.
    The sine sweep is generated in the time domain according to:

    .. math::

        s(t) = \\sin(2\\pi f_l L \\left( \\mathrm{e}^{t/L} - 1 \\right))

    Parameters
    ----------
    amplitude : double
        The envelope amplitude of the signal
    sampling_rate : integer
        Sampling rate of the signal
    num_samples : integer
        Length of the signal in samples
    frequency_range : tuple, double, (20, 20e3)
        Lower and upper frequency range limits of the sweep signal
    sweeprate : double, optional
        Rate at which the sine frequency increases over time

    Returns
    -------
    signal : ndarray, float
        Time domain data of the sweep signal

    """
    freq_upper = frequency_range[1]
    freq_lower = frequency_range[0]

    if sweeprate:
        L = 1 / sweeprate / np.log(2)
        T = L * np.log(freq_upper / freq_lower)
        num_samples = np.round(T * sampling_rate)
    else:
        L = (num_samples - 1) / sampling_rate / np.log(freq_upper / freq_lower)
        sweeprate = 1 / L / np.log(2)

    times = np.arange(0, num_samples) / sampling_rate
    signal = np.sin(2 * np.pi * freq_lower * L * (np.exp(times / L ) - 1))
    signal *= amplitude

    return signal


def impulse(amplitude, num_samples):
    """Generate an impulse, also known as the Dirac delta function

    .. math::

        s(n) =
        \\begin{cases}
        a,  & \\text{if $n$ = 0} \\newline
        0, & \\text{else}
        \\end{cases}

    Parameters
    ----------
    amplitude : double
        The peak amplitude of the impulse
    num_samples : int
        Length of the impulse in samples

    Returns
    -------
    signal : ndarray, double
        The impulse signal

    """
    signal = np.zeros(num_samples, dtype=np.double)
    signal[0] = amplitude

    return signal


def white_noise(amplitude, num_samples, num_channels=1):
    """Generate white noise.

    Parameters
    ----------
    amplitude : double
        The RMS amplitude of the white noise signal.
    num_samples : int
        The length of the signal in samples

    Returns
    -------
    signal : ndarray, double
        The white noise signal

    """
    signal = np.random.randn(num_samples, num_channels)
    rms = np.sqrt(np.mean(signal**2, axis=0))
    signal = signal / rms * amplitude
    return signal.T
