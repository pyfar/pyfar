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
    The parameters `frequency`, `amplitude`, and `phase` must must be scalars
    and/or array likes of the same shape.
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
    The parameters `delay` and `amplitude` must be scalars and/or array likes
    of the same shape.
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

    signal = pyfar.Signal(impulse, sampling_rate, comment=comment)

    return signal


def linear_sweep(n_samples, frequency_range, n_fade_out=90, amplitude=1,
                 sampling_rate=44100):
    """Generate single channel sine sweep with linearly increasing frequency.

    Time domain sweep generation according to [#]_:

    .. math::
        s(t) = \\sin(2\\pi f_\\mathrm{low} t + 2\\pi (f_\\mathrm{high}-
        f_\\mathrm{low}) / T \\cdot t^2 / 2),

    with :math:`T` the duration in seconds, :math:`t` the sampling points in
    seconds, and the frequency limits :math:`f_\\mathrm{low}` and
    :math:`f_\\mathrm{high}`.

    .. note::
        The linear sweep can also be generated in the frequency domain
        (see :py:func:`~general_sweep_synthesis`). Time domain synthesis
        exhibits a constant temporal envelope in trade of slight ripples in the
        magnitude response. Frequency domain synthesis exhibits smooth
        magnitude spectra and in trade of a slightly irregular temporal
        envelope.

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


def exponential_sweep(n_samples, frequency_range, n_fade_out=90, amplitude=1,
                      sweep_rate=None, sampling_rate=44100):
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

    .. note::
        The exponential sweep can also be generated in the frequency domain
        (see :py:func:`~general_sweep_synthesis`). Time domain synthesis
        exhibits a constant temporal envelope in trade of slight ripples in the
        magnitude response. Frequency domain synthesis exhibits smooth
        magnitude spectra and in trade of a slightly irregular temporal
        envelope.

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


def general_sweep_synthesis(
        n_samples, spectrum, start_margin=None, stop_margin=None,
        requency_range=None, double=True, sampling_rate=44100):
    """
    Frequency domain sweep synthesis with arbitrary magnitude response.

    TODO Link to fractional octave smoothing in Notes

    TODO Implement calculation of group delay of impulse responses

    TODO Examples

    Sweep sweep synthesis according to [#]_

    .. note::
        The linear and exponential sweep can also be generated in the time
        domain (see :py:func:`~linear_sweep`, :py:func:`~exponential_sweep`).
        Frequency domain synthesis exhibits smooth magnitude spectra and in
        trade of a slightly irregular temporal envelope. Time domain synthesis
        exhibits a constant temporal envelope in trade of slight ripples in the
        magnitude response.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples.
    spectrum : Signal, string
        Specify the magnitude response of the sweep.

        signal
            The magnitude response as :py:class:`~pyfar.classes.audio.Signal`
            object. If ``signal.n_samples`` is smaller than `n_samples`, zeros
            are padded to the end of `signal`. Note that `frequency_range` is
            not required in this case.
        ``'linear'``
            Design a sweep with linearly increasing frequency and a constant
            magnitude spectrum.
        ``'exponential'``
            Design a sweep with exponentially increasing frequency. The
            magnitude decreases by 3 dB per frequency doubling and has constant
            energy in fiters of relative constant bandwidth (e.g. octaves).
        ``'perfect'``
            Perfect sweep according to [#]_. Note that the parameters
            `start_margin`, `stop_margin`, `frequency_range` and `double` are
            not required in this case.

    start_margin : int, optional
        The time in samples, at which the sweep starts. The start margin is
        required because the frequency domain sweep synthesis has pre-ringing
        in the time domain. Not required if `spectrum` is ``'perfect'``.
    stop_margin : int, optional
        Time in samples, at which the sweep stops. This is relative to
        `n_samples`, e.g., a stop margin of 100 samples means that the sweep
        ends at sample ``n_samples-10``. This is required, because the
        frequency domain sweep synthesis has post-ringing in the time domain.
        Not required if `spectrum` is ``'perfect'``.
    frequency_range : array like, optional
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz. The restriction of the frequency range is realized
        by appling 8th order Butterworth filters at the specified frequencies.
        Not required if `spectrum` is ``'perfect'`` or `signal`.
    double : bool, optional
        Double `n_samples` during the sweep calculation (recommended). The
        default is  ``True``. Not required if `spectrum` is ``'perfect'``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain and has the ``none``
        FFT normalization (see :py:func:`~pyfar.dsp.fft.normalization`). The
        sweep parameters are written to `comment`.
    group_delay_sweep : FrequencyData
        The group delay of the sweep as a single sided spectrum between 0 Hz
        and ``sampling_rate/2``.

        TODO add this after implementation is complete:

        This can be used to calculate the group delay of the impulse responses
        of linear and harmonic distortion products after deconvoloution (see
        :py:func:`~pyfar.dsp...`).

    Notes
    -----
    The envelope of the sweep time signal should be constant, appart from
    slight overshoots at the beginning and end. If this is not the case, try to
    provide a smoother spectrum (if `spectrum` is `signal`) or increase
    `n_samples`.

    References
    ----------
    .. [#] S. Müller, P. Massarani. 'Transfer Function Measurement with Sweeps.
           Directors Cut Including Previously Unreleased Material and some
           Corrections. J. Audio Eng. Soc. 2001, 49 (6), 443–471.
    .. [#] C. Antweiler, A. Telle, P. Vary, G. Enzner. 'Perfect-Sweep NLMS for
           Time-Variant Acoustic System Identification,' IEEE Int. Conf.
           Acoustics, Speech and Signal Processing (ICASSP),
           Prague, Czech Republic, 2011. doi: 10.1109/ICASSP.2012.6287930.

    Examples
    --------
    TODO Example with spectrum=singal
          (e.g., Bass emphasis by means of low shelve filter)

    TODO Examples with spectrum="linear"

    TODO Examples with spectrum="exponential"

    TODO Examples with spectrum="perfect"
    """
    pass


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
