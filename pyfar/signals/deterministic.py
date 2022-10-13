import numpy as np
import pyfar


def sine(frequency, n_samples, amplitude=1, phase=0, sampling_rate=44100,
         full_period=False):
    """Generate a single or multi channel sine signal.

    Parameters
    ----------
    frequency : double, array like
        Frequency of the sine in Hz (0 <= `frequency` <= `sampling_rate`/2).
    n_samples : int
        Length of the signal in samples.
    amplitude : double, array like, optional
        The amplitude. The default is ``1``.
    phase : double, array like, optional
        The phase in radians. The default is ``0``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.
    full_period : boolean, optional
        Make sure that the returned signal contains an integer number of
        periods resulting in a periodic signal. This is done by adjusting the
        frequency of the sine. The default is ``False``.

    Returns
    -------
    signal : Signal
        The sine signal. The Signal is in the time domain and has the ``rms``
        FFT normalization (see :py:func:`~pyfar.dsp.fft.normalization`).
        The exact frequency, amplitude and phase are written to `comment`.

    Notes
    -----
    The parameters `frequency`, `amplitude`, and `phase` are
    `broadcasted <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
    to the parameter that contains the most elements. For example `frequency`
    could be of shape ``(2, 4)``, `amplitude` of shape ``(2, 1)``, and `phase`
    could be a scalar. In this case all parameters would be broadcasted to a
    shape of ``(2, 4)``.
    """

    # check and match the cshape
    try:
        cshape, (frequency, amplitude, phase) = _match_shape(
            frequency, amplitude, phase)
    except ValueError:
        raise ValueError(("The parameters frequency, amplitude, and phase can "
                          "not be broadcasted to the same shape"))

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
               f"amplitude = {str(amplitude).replace(nl, ',')}, "
               f"phase = {str(phase).replace(nl, ',')} rad)")

    signal = pyfar.Signal(
        sine, sampling_rate, fft_norm="rms", comment=comment)

    return signal


def impulse(n_samples, delay=0, amplitude=1, sampling_rate=44100):
    """
    Generate a single or multi channel impulse signal, also known as the
    Dirac delta function.

    .. math::
        s(n) =
        \\begin{cases}
        \\text{amplitude},  & \\text{if $n$ = delay}\\\\
        0, & \\text{else}
        \\end{cases}


    Parameters
    ----------
    n_samples : int
        Length of the impulse in samples
    delay : double, array like, optional
        Delay in samples. The default is ``0``.
    amplitude : double, optional
        The peak amplitude of the impulse. The default is ``1``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.

    Returns
    -------
    signal : Signal
        The impulse signal. The Signal is in the time domain and has the
        ``none`` FFT normalization (see
        :py:func:`~pyfar.dsp.fft.normalization`). The delay and amplitude
        are written to `comment`.

    Notes
    -----
    The parameters `delay` and `amplitude` are
    `broadcasted <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
    to the parameter that contains the most elements. For example `delay`
    could be of shape ``(2, 4)``, `amplitude` of shape ``(2, 1)`` or a scalar.
    In this case all parameters would be broadcasted to a shape of ``(2, 4)``.
    """
    # check and match the cshape
    try:
        cshape, (delay, amplitude) = _match_shape(delay, amplitude)
    except ValueError:
        raise ValueError(("The parameters delay and amplitude can not be "
                          "broadcasted to the same shape"))

    # generate the impulse
    n_samples = int(n_samples)
    impulse = np.zeros(cshape + (n_samples, ), dtype=np.double)
    for idx in np.ndindex(cshape):
        impulse[idx + (delay[idx], )] = amplitude[idx]

    # save to Signal
    nl = "\n"  # required as variable because f-strings cannot contain "\"
    comment = (f"Impulse signal (delay = {str(delay).replace(nl, ',')} "
               f"samples, amplitude = {str(amplitude).replace(nl, ',')})")

    signal = pyfar.Signal(impulse, sampling_rate, comment=comment)

    return signal


def linear_sweep_time(n_samples, frequency_range, n_fade_out=90, amplitude=1,
                      sampling_rate=44100):
    """Generate single channel sine sweep with linearly increasing frequency.

    Time domain sweep generation according to [#]_:

    .. math::
        s(t) = \\sin(2\\pi f_\\mathrm{low} t + 2\\pi (f_\\mathrm{high}-
        f_\\mathrm{low}) / T \\cdot t^2 / 2),

    with :math:`T` the duration in seconds, :math:`t` the sampling points in
    seconds, and the frequency limits :math:`f_\\mathrm{low}` and
    :math:`f_\\mathrm{high}`.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz.
    n_fade_out : int, optional
        The length of the squared cosine fade-out in samples. This is done to
        avoid discontinuities at the end of the sweep. The default is ``90``,
        which equals approximately 2 ms at sampling rates of 44.1 and 48 kHz.
    amplitude : double, optional
        The amplitude of the signal. The default is ``1``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain and has the ``none``
        FFT normalization (see :py:func:`~pyfar.dsp.fft.normalization`). The
        sweep type, frequency range, and length of the fade our are written to
        `comment`.

    References
    ----------
    .. [#]  Farina, Angelo (2000): "Simultaneous measurement of impulse
            response and distortion with a swept-sine technique." 108th AES
            Convention, Paris: France.
    """

    signal = _time_domain_sweep(
        n_samples, frequency_range, n_fade_out, amplitude,
        sampling_rate, "linear")

    return signal


def exponential_sweep_time(n_samples, frequency_range, n_fade_out=90,
                           amplitude=1, sweep_rate=None, sampling_rate=44100):
    """
    Generate single channel sine sweep with exponentially increasing frequency.

    Time domain sweep generation according to [#]_:

    .. math::
        s(t) = \\sin(2\\pi f_\\mathrm{low} L \\left( e^{t/L} - 1 \\right))

    with

    .. math::
        L = T / \\log(f_\\mathrm{high}/f_\\mathrm{low}),

    :math:`T` the duration in seconds, :math:`t` the sampling points in
    seconds, and the frequency limits :math:`f_\\mathrm{low}` and
    :math:`f_\\mathrm{high}`.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz.
    n_fade_out : int, optional
        The length of the squared cosine fade-out in samples. This is done to
        avoid discontinuities at the end of the sweep. The default is ``90``,
        which equals approximately 2 ms at sampling rates of 44.1 and 48 kHz.
    amplitude : double, optional
        The amplitude of the signal. The default is ``1``.
    sweep_rate : double, optional
        Rate at which the sine frequency increases over time. If this is given,
        `n_samples` is calculated according to the sweep rate. The default is
        ``None``, which uses `n_samples` without modifications.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain and has the ``none``
        FFT normalization (see :py:func:`~pyfar.dsp.fft.normalization`). The
        sweep type, frequency range, and length of the fade our are written to
        `comment`.

    References
    ----------
    .. [#]  Farina, Angelo (2000): "Simultaneous measurement of impulse
            response and distortion with a swept-sine technique." 108th AES
            Convention, Paris: France.
    """

    signal = _time_domain_sweep(
        n_samples, frequency_range, n_fade_out, amplitude, sampling_rate,
        "exponential", sweep_rate)

    return signal


def _time_domain_sweep(n_samples, frequency_range, n_fade_out, amplitude,
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
    n_fade_out = int(n_fade_out)
    if n_fade_out > 0:
        # check must be done here because n_samples might not be defined if
        # using the sweep_rate for exponential sweeps
        if sweep.size < n_fade_out:
            raise ValueError("The sweep must be longer than n_fade_out.")

        sweep[-n_fade_out:] *= np.cos(np.linspace(0, np.pi/2, n_fade_out))**2

    # save to signal
    comment = (f"{sweep_type} sweep between {frequency_range[0]} "
               f"and {frequency_range[1]} Hz "
               f"with {n_fade_out} samples squared cosine fade-out.")
    signal = pyfar.Signal(
        sweep, sampling_rate, fft_norm="none", comment=comment)

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


def _match_shape(*args):
    """
    Match the shape of *args to the shape of the arg with the largest size
    using np.broadcast_to()

    Parameters
    ----------
    *args :
        data for matching shape

    Returns
    -------
    shape : tuple
        new common shape of the args
    args : list
        args with new common shape
        (*arg_1, *arg_2, ..., *arg_N)
    """

    # find the shape of the largest array
    size = 1
    shape = (1, )
    for arg in args:
        arg = np.asarray(arg)
        if arg.size > size:
            size = arg.size
            shape = arg.shape

    # try to match the shape
    result = []
    for arg in args:
        arg = np.broadcast_to(arg, shape)
        arg.setflags(write=1)
        result.append(np.atleast_1d(arg))

    return shape, result
