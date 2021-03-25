"""
This module contains functions for generating common audio signals such as
sine, impulse, and noise signals.

Note
----
All signal length are given in samples. The value for the length are casted to
integer numbers in all cases. This makes it possible to pass float numbers for
convenience, e.g., `n_samples=.015 * sampling_rate`.
"""
import numpy as np
from pyfar import Signal
import pyfar.fft as fft


def sine(frequency, n_samples, amplitude=1, phase=0, sampling_rate=44100,
         full_period=False):
    """Generate a single or multi channel sine signal.

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
        frequency is written to Signal.comment.

    Note
    ----
    The parameters frequency, amplitude, and samples must all be scalars
    or of the same shape.
    """

    # check and match the cshape
    cshape = _get_common_shape(frequency, amplitude, phase)
    frequency, amplitude, phase = _match_shape(
        cshape, frequency, amplitude, phase)

    if np.any(frequency < 0) or np.any(frequency > sampling_rate/2):
        raise ValueError(
            f"The frequency must be between 0 and {sampling_rate/2} Hz")

    # generate the sine signal
    n_samples = int(n_samples)
    times = np.arange(n_samples) / sampling_rate
    sine = np.zeros(cshape + (n_samples, ))
    for idx in np.ndindex(cshape):
        if full_period:
            # nearest number of full periods
            num_periods = np.round(
                n_samples / sampling_rate * frequency[idx])
            # corresponding frequency
            frequency[idx] = num_periods * sampling_rate / n_samples

        sine[idx] = amplitude[idx] * \
            np.sin(2 * np.pi * frequency[idx] * times + phase[idx])

    # save to Signal
    nl = "\n"  # required as variable because f-strings cannot contain "\"
    comment = (f"Sine signal (f = {str(frequency).replace(nl, ',')} Hz, "
               f"amplitude = {str(amplitude).replace(nl, ',')})")

    signal = Signal(sine, sampling_rate, fft_norm="rms",
                    comment=comment)

    return signal


def impulse(n_samples, delay=0, amplitude=1, sampling_rate=44100):
    """
    Generate an single or multi channel impulse signal, also known as the
    Dirac delta function.

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
    # check and match the cshape
    cshape = _get_common_shape(delay, amplitude)
    delay, amplitude = _match_shape(cshape, delay, amplitude)

    # generate the impulse
    n_samples = int(n_samples)
    impulse = np.zeros(cshape + (n_samples, ), dtype=np.double)
    for idx in np.ndindex(cshape):
        impulse[idx + (delay[idx], )] = amplitude[idx]

    # save to Signal
    nl = "\n"  # required as variable because f-strings cannot contain "\"
    comment = (f"Impulse signal (delay = {str(delay).replace(nl, ',')} "
               f"samples, amplitude = {str(amplitude).replace(nl, ',')})")

    signal = Signal(impulse, sampling_rate, comment=comment)

    return signal


def white_noise(n_samples, rms=1, sampling_rate=44100, seed=None):
    """Generate single or multi channel normally distributed white noise.

    Parameters
    ----------
    n_samples : int
        The length of the signal in samples
    rms : double, array like, optional
        The route mean square (RMS) value of the noise signal. A multi channel
        noise signal is generated if an array of RMS values is passed.
        The default is 1.
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
    rms = np.atleast_1d(rms)
    noise = _generate_white_noise(n_samples, rms, seed)

    # level the noise
    noise = _normalize_rms(noise, rms)

    # save to Signal
    nl = "\n"  # required as variable because f-strings cannot contain "\"
    comment = f"White noise signal (rms = {str(rms).replace(nl, ',')})"

    signal = Signal(noise, sampling_rate, fft_norm="rms", comment=comment)

    return signal


def pink_noise(n_samples, rms=1, sampling_rate=44100, seed=None):
    """Generate single or multi channel normally distributed pink noise.

    The pink noise is generated by applying a sqrt(1/f) filter to the spectrum.

    Parameters
    ----------
    n_samples : int
        The length of the signal in samples
    rms : double, array like, optional
        The route mean square (RMS) value of the noise signal. A multi channel
        noise signal is generated if an array of RMS values is passed.
        The default is 1.
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
    rms = np.atleast_1d(rms)
    noise = _generate_white_noise(n_samples, rms, seed)

    # apply 1/f filter in the frequency domain
    noise = fft.rfft(noise, n_samples, sampling_rate, 'none')
    noise /= np.sqrt(np.arange(1, noise.shape[-1]+1))
    noise = fft.irfft(noise, n_samples, sampling_rate, 'none')

    # level the noise
    noise = _normalize_rms(noise, rms)

    # save to Signal
    nl = "\n"  # required as variable because f-strings cannot contain "\"
    comment = f"Pink noise signal (rms = {str(rms).replace(nl, ',')})"

    signal = Signal(noise, sampling_rate, fft_norm="rms", comment=comment)

    return signal


def pulsed_noise(n_pulse, n_pause, n_fade=90, repetitions=5, rms=1,
                 color="pink", frozen=True, sampling_rate=44100, seed=None):
    """
    Generate single channel normally distributed pulsed white and pink noise.

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
    rms : double, array like, optional
        The RMS amplitude of the white noise signal. The default is 1.
    color: string, optional
        The noise color, which can be 'pink' or 'white'. The default is 'pink'.
    frozen : boolean, optional
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
        noise = pink_noise(n_samples, rms, sampling_rate, seed).time
    elif color == "white":
        noise = white_noise(n_samples, rms, sampling_rate, seed).time
    else:
        raise ValueError(f"color is {color} but must be 'pink' or 'white'.")

    noise = np.tile(noise, (repetitions, 1)) if frozen else \
        noise.reshape((repetitions, n_pulse))

    # fade the noise
    if n_fade > 0:
        n_fade = int(n_fade)
        fade = np.sin(np.linspace(0, np.pi/2, n_fade))**2
        noise[..., 0:n_fade] *= fade
        noise[..., -n_fade:] *= fade[::-1]

    # add the pause
    noise = np.concatenate((noise, np.zeros((repetitions, int(n_pause)))), -1)

    # reshape to single channel signal and discard final pause
    noise = noise.reshape((1, -1))[..., :-int(n_pause)]

    # save to Signal
    frozen_str = "frozen" if frozen else ""
    comment = (f"{frozen_str} {color} pulsed noise signal (rms = {rms}, "
               f"{repetitions} repetitions, {n_pulse} samples pulse duration, "
               f"{n_pause} samples pauses, and {n_fade} samples fades.")

    signal = Signal(noise, sampling_rate, fft_norm="rms", comment=comment)

    return signal


def linear_sweep(n_samples, frequency_range, n_fade=90, amplitude=1,
                 sampling_rate=44100):
    """Generate single channel sine sweep with linearly increasing frequency.

    Time domain sweep generation according to _[1]:

    .. math::
        x(t) = \\sin(2\\pi\f_{low} t + 2\\pi (f_{high}-f_{low}) / T t^2 / 2),

    with T the duration in seconds and t the sampling points in seconds.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz.
    n_fade : int, optional
        The length of the squared cosine fade-out in samples. This is done to
        avoid discontinuities at the end of the sweep. The default is 90, which
        equals approximately 2 ms at sampling rates of 44.1 and 48 kHz.
    amplitude : double, optional
        The amplitude of the signal. The default is 1.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.

    Returns
    -------
    sweep : pyfar Signal
        The sweep as a Signal object. The Signal is in the time domain and
        has the 'rms' FFT normalization (see pyfar.fft.normalization).

    References
    ----------
    .. [1]  Farina, Angelo (2000): "Simultaneous measurement of impulse
            response and distortion with a swept-sine technique." 108th AES
            Convention, Paris: France.
    """

    signal = _time_domain_sweep(
        n_samples, frequency_range, n_fade, amplitude, sampling_rate, "linear")

    return signal


def exponential_sweep(n_samples, frequency_range, n_fade=90, amplitude=1,
                      sweep_rate=None, sampling_rate=44100):
    """
    Generate single channel sine sweep with exponentially increasing frequency.

    Time domain sweep generation according to _[1]:

    .. math::
        s(t) = \\sin(2\\pi f_{low} L \\left( \\mathrm{e}^{t/L} - 1 \\right))

    with

    .. math::
        L = T / \\log(f_{high}/f_{low}),

    T the duration in seconds, and t the sampling points in seconds.

    Parameters
    ----------
     n_samples : int
        The length of the sweep in samples
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz.
    n_fade : int, optional
        The length of the squared cosine fade-out in samples. This is done to
        avoid discontinuities at the end of the sweep. The default is 90, which
        equals approximately 2 ms at sampling rates of 44.1 and 48 kHz.
    amplitude : double, optional
        The amplitude of the signal. The default is 1.
    sweep_rate : double, optional
        Rate at which the sine frequency increases over time. If this is given
        n_samples is calculated according to the sweep rate. The default is
        None, which uses n_samples without modifications.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is 44100.

    Returns
    -------
    sweep : pyfar Signal
        The sweep as a Signal object. The Signal is in the time domain and
        has the 'rms' FFT normalization (see pyfar.fft.normalization).

    References
    ----------
    .. [1]  Farina, Angelo (2000): "Simultaneous measurement of impulse
            response and distortion with a swept-sine technique." 108th AES
            Convention, Paris: France.
    """

    signal = _time_domain_sweep(
        n_samples, frequency_range, n_fade, amplitude, sampling_rate,
        "exponential", sweep_rate)

    return signal


def _time_domain_sweep(n_samples, frequency_range, n_fade, amplitude,
                       sampling_rate, sweep_type, sweep_rate=None):

    # check input
    if np.atleast_1d(frequency_range).size != 2:
        raise ValueError(
            "frequency_range must be an array like with to elements.")
    if frequency_range[1] > sampling_rate/2:
        raise ValueError(
            "Upper frequency limit is larger than half the sampling rate.")
    if frequency_range[0] == 0 and sweep_type == "exponential":
        raise ValueError("The exponential sweep can not start at 0 Hz.")

    # generate sweep
    if sweep_type == "linear":
        sweep = _linear_sweep(
            n_samples, frequency_range, amplitude, sampling_rate)
    elif sweep_type == 'exponential':
        sweep = _exponential_sweep(
            n_samples, frequency_range, amplitude, sweep_rate, sampling_rate)

    # fade out
    n_fade = int(n_fade)
    if n_fade > 0:
        # check must be done here because n_samples might not be defined if
        # using the sweep_rate for exponential sweeps
        if sweep.size < n_fade:
            raise ValueError("The sweep must be longer than n_fade.")

        sweep[-n_fade:] *= np.cos(np.linspace(0, np.pi/2, n_fade))**2

    # save to signal
    comment = (f"{sweep_type} sweep between {frequency_range[0]} "
               f"and {frequency_range[1]} Hz "
               f"with {n_fade} samples squared cosine fade-out.")
    signal = Signal(sweep, sampling_rate, fft_norm="rms", comment=comment)

    return signal


def _linear_sweep(n_samples, frequency_range, amplitude, sampling_rate):

    # generate sweep
    n_samples = int(n_samples)
    t = np.arange(n_samples) / sampling_rate
    T = n_samples / sampling_rate

    # [1, page 5]
    sweep = amplitude * np.sin(
        2 * np.pi * frequency_range[0] * t +
        2 * np.pi * (frequency_range[1]-frequency_range[0]) / T * t**2 / 2)

    return sweep


def _exponential_sweep(n_samples, frequency_range, amplitude, sweep_rate,
                       sampling_rate):

    c = np.log(frequency_range[1] / frequency_range[0])

    # get n_samples
    if sweep_rate is not None:
        L = 1 / sweep_rate / np.log(2)
        T = L * c
        n_samples = np.round(T * sampling_rate)
    else:
        n_samples = int(n_samples)

    # L for actual n_samples
    L = n_samples / sampling_rate / c

    # make the sweep
    times = np.arange(n_samples) / sampling_rate
    sweep = amplitude * np.sin(
        2 * np.pi * frequency_range[0] * L * (np.exp(times / L) - 1))

    return sweep


def _generate_white_noise(n_samples, amplitude, seed=None):
    """Generate normally distributed noise."""

    n_samples = int(n_samples)
    cshape = np.atleast_1d(amplitude).shape
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(np.prod(cshape + (n_samples, )))
    noise = noise.reshape(cshape + (n_samples, ))

    return noise


def _normalize_rms(data, amplitude):
    """Level data to RMS amplitude."""

    rms = np.atleast_1d(np.sqrt(np.mean(data**2, axis=-1)))
    for idx in np.ndindex(amplitude.shape):
        data[idx] = data[idx] / rms[idx] * amplitude[idx]
    return data


def _get_common_shape(*data):
    """Check if all entries in data have the same shape or shape (1, )

    Parameters
    ----------
    data : *args
       Numbers and array likes for which the shape is checked.

    Returns
    -------
    cshape : tuple
        Common shape of data, e.g., (1, ) if al entries in data are numbers or
        (2, 3) if data has entries with shape (2, 3) (and (1, )).
    """

    cshape = None
    for d in data:
        d = np.atleast_1d(d)
        if cshape is None or cshape == (1, ):
            cshape = d.shape
        elif d.shape != (1, ):
            if d.shape != cshape:
                raise ValueError(
                    "Input data must be of the same shape or of shape (1, ).")

    return cshape


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
