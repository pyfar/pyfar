"""
This module contains functions for generating common audio signals such as
sine, impulse, and noise signals.

Note
----
All signal length are given in samles. The value for the length are caseted to
integer numbers in all cases. This makes it possible to pass float numbers for
convenience, e.g., `n_samples=.015 * sampling_rate`.
"""
import numpy as np
from pyfar import Signal
import pyfar.fft as fft


def sine(frequency, n_samples, amplitude=1, phase=0, sampling_rate=44100,
         full_period=False):
    """Generate a sine signal.

    Parameters
    ----------
    frequency : double, array like
        Frequency of the sine in Hz (0 <= frequency <= sampling_rate/2).
    n_samples : int
        Length of the signal in samples.
    amplitude : double, array like, optional
        The amplitude. The default is 1.
    phase : double, array like, optional
        The phase in radians. The default is 0.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.
    full_period : boolean, optional
        Make sure that the returned signal contains an integer number of
        periods resulting in a periodic signal. This is done by adjusting the
        frequency of the sine. The default is False.

    Returns
    -------
    signal : Signal
        The sine as a Signal object. The Signal is in the time domain and has
        the 'rms' FFT normalization (see pyfar.fft.normalization). The exact
        frequency is written to Signal.commtent.

    Note
    ----
    The parameters frequency, amplitude, and samples must all be scalars
    or of the same shape.
    """

    # check and match the shape
    shape = _get_common_shape(frequency, amplitude, phase)
    frequency, amplitude, phase = _match_shape(
        shape, frequency, amplitude, phase)

    if np.any(frequency < 0) or np.any(frequency > sampling_rate/2):
        raise ValueError(
            f"The frequencies but must be between 0 and {sampling_rate/2} Hz")

    # generate the sine signal
    n_samples = int(n_samples)
    times = np.arange(n_samples) / sampling_rate
    sine = np.zeros(shape + (n_samples, ))
    for idx in np.ndindex(shape):
        if full_period:
            # nearest number of full periods
            num_periods = np.round(
                n_samples / sampling_rate * frequency[idx])
            # corresponding frequency
            frequency[idx] = num_periods * sampling_rate / n_samples

        sine[idx] = amplitude[idx] * \
            np.sin(2 * np.pi * frequency[idx] * times + phase[idx])

    # save to Signal
    if shape == (1, ):
        frequency = frequency[0]  # nicer comments :)

    signal = Signal(sine, sampling_rate, fft_norm="rms",
                    comment=f"f = {frequency} Hz")

    return signal


def impulse(n_samples, delay=0, amplitude=1, sampling_rate=44100):
    """Generate an impulse signal, also known as the Dirac delta function

    .. math::
        s(n) =
        \\begin{cases}
        amplitude,  & \\text{if $n$ = delay} \\newline
        0, & \\text{else}
        \\end{cases}

    Parameters
    ----------
    n_samples : int
        Length of the impulse in samples
    delay : double, array like
        Delay in samples. The default is 0.
    amplitude : double
        The peak amplitude of the impulse. The default is 1.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.

    Returns
    -------
    signal : Signal
        The impulse as a Signal object. The Signal is in the time domain and
        has the 'none' FFT normalization (see pyfar.fft.normalization).

    Note
    ----
    The parameters delay and amplitude must all be scalars or of the same
    shape.
    """
    # check and match the shape
    shape = _get_common_shape(delay, amplitude)
    delay, amplitude = _match_shape(shape, delay, amplitude)

    # generate the impulse signal
    n_samples = int(n_samples)
    signal = np.zeros(shape + (n_samples, ), dtype=np.double)
    for idx in np.ndindex(shape):
        signal[idx + (delay[idx], )] = amplitude[idx]

    # get pyfar Signal
    signal = Signal(signal, sampling_rate)

    return signal


def white_noise(n_samples, amplitude=1, sampling_rate=44100, seed=None):
    """Generate normally distributed white noise.

    Parameters
    ----------
    n_samples : int
        The length of the signal in samples
    amplitude : double, array like, optional
        The RMS amplitude of the white noise signal. A multi channel noise
        signal is generated if an array of amplitudes is passed. The default
        is 1.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.
    seed : int, None
        The seed for the random generator. Pass a seed to obtain identical
        results for multiple calls of white noise. The default is None, which
        will yield different results with every call.

    Returns
    -------
    signal : Signal
        The noise as a Signal object. The Signal is in the time domain and
        has the 'rms' FFT normalization (see pyfar.fft.normalization).
    """

    # generate the noise
    amplitude = np.atleast_1d(amplitude)
    signal = _generate_normal_noise(n_samples, amplitude, seed)

    # level the noise
    signal = _normalize_rms(signal, amplitude)

    # save to Signal
    signal = Signal(signal, sampling_rate, fft_norm="rms")

    return signal


def pink_noise(n_samples, amplitude=1, sampling_rate=44100, seed=None):
    """Generate normally distributed pink noise.

    The pink noise is generated by applying a sqrt(1/f) filter to the spectrum.

    Parameters
    ----------
    n_samples : int
        The length of the signal in samples
    amplitude : double, array like, optional
        The RMS amplitude of the white noise signal. A multi channel noise
        signal is generated if an array of amplitudes is passed. The default
        is 1.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.
    seed : int, None
        The seed for the random generator. Pass a seed to obtain identical
        results for multiple calls of white noise. The default is None, which
        will yield different results with every call.

    Returns
    -------
    signal : Signal
        The noise as a Signal object. The Signal is in the time domain and
        has the 'rms' FFT normalization (see pyfar.fft.normalization).
    """

    # generate the noise
    amplitude = np.atleast_1d(amplitude)
    signal = _generate_normal_noise(n_samples, amplitude, seed)

    # apply 1/f filter in the frequency domain
    signal = fft.rfft(signal, n_samples, sampling_rate, 'none')
    signal /= np.sqrt(np.arange(1, signal.shape[-1]+1))
    signal = fft.irfft(signal, n_samples, sampling_rate, 'none')

    # level the noise
    signal = _normalize_rms(signal, amplitude)

    # save to Signal
    signal = Signal(signal, sampling_rate, fft_norm="rms")

    return signal


def pulsed_noise(n_pulse, n_pause, n_fade=90, repetitions=5, amplitude=1,
                 color="pink", frozen=True, sampling_rate=44100, seed=None):
    """Generate normally distributed pulsed white and pink noise.

    The pink noise is generated by applying a sqrt(1/f) filter to the spectrum.

    Parameters
    ----------
    n_pulse : int
        The length of the pulses in samples
    n_pause : int
        The length of the pauses between the pulses in samples.
    n_fade : int, optional
        The length of the squared sine/cosine fade-in and fade outs in samples.
        The default is 90, which equals approximately 2 ms at sampling rates of
        44.1 and 48 kHz.
    repetitions : int, optional
    amplitude : double, array like, optional
        The RMS amplitude of the white noise signal. A multi channel noise
        signal is generated if an array of amplitudes is passed. The default
        is 1.
    color: string, optional
        The noise color, which can be 'pink' or 'white'. The default is 'pink'.
    frozen : boolena, optional
        If True, all noise pulses are identical. If False each noise pulse is
        a separate stochastic process. The default is True.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.
    seed : int, None
        The seed for the random generator. Pass a seed to obtain identical
        results for multiple calls of white noise. The default is None, which
        will yield different results with every call.

    Returns
    -------
    signal : Signal
        The noise as a Signal object. The Signal is in the time domain and
        has the 'rms' FFT normalization (see pyfar.fft.normalization).
    """

    if n_pulse < 2 * n_fade:
        raise ValueError(
            "n_fade too large. It must be smaller than n_pulse/2.")

    # get the noise sample
    n_pulse = int(n_pulse)
    repetitions = int(repetitions)
    n_samples = n_pulse if frozen else n_pulse * repetitions

    if color == "pink":
        noise = pink_noise(n_samples, amplitude, sampling_rate, seed).time
    elif color == "white":
        noise = white_noise(n_samples, amplitude, sampling_rate, seed).time
    else:
        raise ValueError(f"color is {color} but must be 'pink' or 'white'.")

    noise = np.tile(noise, (repetitions, 1)) if frozen else \
        noise.reshape((repetitions, n_pulse))

    # fade the noise
    if n_fade > 0:
        n_fade = int(n_fade)
        fade = np.sin(np.linspace(0, np.pi/4, n_fade))
        noise[..., 0:n_fade] *= fade
        noise[..., -n_fade:] *= fade[::-1]

    # add the pause
    noise = np.concatenate((noise, np.zeros((repetitions, int(n_pause)))), -1)

    # reshape to single channel signal and discard final pause
    noise = noise.reshape((1, -1))[..., :-int(n_pause)]

    # save to Signal
    signal = Signal(noise, sampling_rate, fft_norm="rms")

    return signal


def _generate_normal_noise(n_samples, amplitude, seed=None):
    """Generate normally distributed noise."""

    n_samples = int(n_samples)
    shape = np.atleast_1d(amplitude).shape
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(np.prod(shape + (n_samples, )))
    noise = noise.reshape(shape + (n_samples, ))

    return noise


def _normalize_rms(signal, amplitude):
    """Level signal to RMS amplitude."""

    rms = np.atleast_1d(np.sqrt(np.mean(signal**2, axis=-1)))
    for idx in np.ndindex(amplitude.shape):
        signal[idx] = signal[idx] / rms[idx] * amplitude[idx]
    return signal


def _get_common_shape(*data):
    """Check if all entries in data have the same shape or shape (1, )

    Parameters
    ----------
    data : *args
       Numbers and array likes for which the shape is checked.

    Returns
    -------
    shape : tuple
        Common shape of data, e.g., (1, ) if al entries in data are numbers or
        (2, 3) if data has entries with shape (2, 3) (and (1, )).
    """

    shape = None
    for d in data:
        d = np.atleast_1d(d)
        if shape is None or shape == (1, ):
            shape = d.shape
        elif d.shape != (1, ):
            if d.shape != shape:
                raise ValueError(
                    "Input data must be of the same shape or of shape (1, ).")

    return shape


def _match_shape(shape, *args):
    """
    Match the shape of *args by using np.tile(shape) if the shape is not (1, 0)

    Note that calling _get_common_shape before might be a good idea.

    Parameters
    ----------
    shape : tuple
        The desired shape
    *args : number array likes
        All *args must be of shape (1, ) or shape

    Returns
    -------
    args : tuple
        (*arg_1, *arg_2, ..., arg_N)
    """

    result = []
    for arg in args:
        arg = np.atleast_1d(arg)
        if arg.shape == (1, ):
            arg = np.tile(arg, shape)

        result.append(arg)

    return tuple(result)
