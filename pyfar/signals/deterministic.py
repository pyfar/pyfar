"""Module for generating deterministic signals as part of pyfar.signals."""
import numpy as np
import warnings
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
    The parameters `frequency`, `amplitude`, and `phase` are broadcasted using
    the :doc:`numpy rules<numpy:user/basics.broadcasting>`. For example
    `frequency` could be of shape ``(2, 4)``, `amplitude` of shape ``(2, 1)``,
    and `phase` could be a scalar. In this case all parameters would be
    broadcasted to a shape of ``(2, 4)``.
    """

    # check and match the cshape
    try:
        cshape, (frequency, amplitude, phase) = _match_shape(
            frequency, amplitude, phase)
    except ValueError as error:
        raise ValueError(("The parameters frequency, amplitude, and phase can "
                          "not be broadcasted to the same shape")) from error

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
    r"""
    Generate a single or multi channel impulse signal, also known as the
    Dirac delta function.

    .. math::
        s(n) =
        \begin{cases}
        \text{amplitude},  & \text{if $n$ = delay}\\
        0, & \text{else}
        \end{cases}


    Parameters
    ----------
    n_samples : int
        Length of the impulse in samples
    delay : double, array like, optional
        Delay in samples. The default is ``0``.
    amplitude : double, array like, optional
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
    The parameters `delay` and `amplitude` are broadcasted using the
    :doc:`numpy rules<numpy:user/basics.broadcasting>`. For example `delay`
    could be of shape ``(2, 4)``, `amplitude` of shape ``(2, 1)`` or a scalar.
    In this case all parameters would be broadcasted to a shape of ``(2, 4)``.
    """
    # check and match the cshape
    try:
        cshape, (delay, amplitude) = _match_shape(delay, amplitude)
    except ValueError as error:
        raise ValueError(
            ("The parameters delay and amplitude can not be "
            "broadcasted to the same shape")) from error

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
    r"""
    Generate sine sweep with linearly increasing frequency in the time domain.

    Time domain sweep generation according to [#]_:

    .. math::
        s(t) = \sin(2\pi f_\mathrm{low} t + 2\pi (f_\mathrm{high}-
        f_\mathrm{low}) / T \cdot t^2 / 2),

    with :math:`T` the duration in seconds, :math:`t` the sampling points in
    seconds, and the frequency limits :math:`f_\mathrm{low}` and
    :math:`f_\mathrm{high}`.

    The linear sweep can also be generated in the frequency domain (see
    :py:func:`~linear_sweep_freq`). Time domain synthesis exhibits a constant
    temporal envelope at the cost of slight ripples in the magnitude response.
    Frequency domain synthesis exhibits smooth magnitude spectra at the cost
    of a slightly irregular temporal envelope.

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

    Examples
    --------
    Linear sweep between 0 and 22050 Hz

    .. plot::

        >>> import pyfar as pf
        >>> sweep = pf.signals.linear_sweep_time(2**16, [0, 22050])
        >>> pf.plot.time_freq(sweep)
    """

    signal = _time_domain_sweep(
        n_samples, frequency_range, n_fade_out, amplitude,
        sampling_rate, "linear")

    return signal


def linear_sweep_freq(
        n_samples, frequency_range, start_margin, stop_margin, n_fade_in=0,
        n_fade_out=0, bandpass_order=8, sampling_rate=44100,
        return_group_delay=False):
    """
    Generate sine sweep with linearly increasing frequency in the frequency
    domain.

    Sine sweep synthesis according to [#]_.

    The linear sweep can also be generated in the time domain
    (:py:func:`~linear_sweep_time`). Frequency domain synthesis exhibits
    smooth magnitude spectra at the cost of a slightly irregular temporal
    envelope. Time domain synthesis exhibits a constant temporal envelope in
    trade of slight ripples in the magnitude response.

    .. note::
        The envelope of the sweep time signal should be constant, apart from
        slight overshoots at the beginning and end. If this is not the case,
        try to increase `n_samples`, `start_margin`, `stop_margin`, `n_fade_in`
        or `n_fade_out`.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples.
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz. The restriction of the frequency range is realized
        by applying a Butterworth high-pass if ``frequency_range[0] > 0``
        and/or by a low-pass if ``frequency_range[1] < sampling_rate / 2``.
    start_margin : int, float
        The time in samples, at which the sweep starts. The start margin is
        required because the frequency domain sweep synthesis has pre-ringing
        in the time domain.
    stop_margin : int, float
        Time in samples, at which the sweep stops. This is relative to
        `n_samples`, e.g., a stop margin of 100 samples means that the sweep
        ends at sample ``n_samples-100``. This is required, because the
        frequency domain sweep synthesis has post-ringing in the time domain.
    n_fade_in : int, optional
        Duration of a squared sine fade-in in samples. The fade starts at the
        first sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal. The default ``0`` does not apply
        a fade-in.
    n_fade_out : int, optional
        Duration of a squared cosine fade-out in samples. The fade ends at the
        last sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal. The default ``0`` does not apply
        a fade-out.
    bandpass_order : int, optional
        The order of the Butterworth filters that are applied to limit the
        frequency range (see above). The default is ``8``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.
    return_group_delay : boolean, optional
        Return the analytical group delay of the sweep. This can be used to
        compute the times at which distortion products appear. The default is
        ``False``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain, has a maximum
        absolute amplitude of 1 and the ``none`` FFT normalization
        (see :py:func:`~pyfar.dsp.fft.normalization`).
    group_delay_sweep : FrequencyData
        The analytical group delay of the sweep in seconds as a single sided
        spectrum. Only returned if `return_group_delay` is ``True``.

    References
    ----------
    .. [#] S. Müller, P. Massarani. 'Transfer Function Measurement with Sweeps.
           Directors Cut Including Previously Unreleased Material and some
           Corrections.' J. Audio Eng. Soc. 2001, 49 (6), 443-471.

    Examples
    --------
    Linear sweep between 50 and 22050 Hz

    .. plot::

        >>> import pyfar as pf
        >>> sweep = pf.signals.linear_sweep_freq(
        ...     2**16, [50, 22050], 1000, 1000)
        >>> pf.plot.time_freq(sweep)
    """

    return _frequency_domain_sweep(
        n_samples=n_samples,
        sweep_type='linear',
        frequency_range=frequency_range,
        bandpass_order=bandpass_order,
        start_margin=start_margin,
        stop_margin=stop_margin,
        n_fade_in=n_fade_in,
        n_fade_out=n_fade_out,
        sampling_rate=sampling_rate,
        return_group_delay=return_group_delay)


def exponential_sweep_time(n_samples, frequency_range, n_fade_out=90,
                           amplitude=1, sweep_rate=None, sampling_rate=44100):
    r"""
    Generate sine sweep with exponentially increasing frequency in the time
    domain.

    Time domain sweep generation according to [#]_:

    .. math::
        s(t) = \sin(2\pi f_\mathrm{low} L \left( e^{t/L} - 1 \right))

    with

    .. math::
        L = T / \log(f_\mathrm{high}/f_\mathrm{low}),

    :math:`T` the duration in seconds, :math:`t` the sampling points in
    seconds, and the frequency limits :math:`f_\mathrm{low}` and
    :math:`f_\mathrm{high}`.

    The exponential sweep can also be generated in the frequency domain (see
    see :py:func:`~exponential_sweep_freq`). Time domain synthesis exhibits a
    constant temporal envelope at the cost of slight ripples in the magnitude
    response. Frequency domain synthesis exhibits smooth magnitude spectra
    at the cost of a slightly irregular temporal envelope.

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
        Rate at which the sine frequency increases over time in octaves per
        second. If this is given, `n_samples` is calculated according to the
        `sweep_rate` and `frequency_range`. The default is ``None``, which uses
        `n_samples` without modifications.
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

    Examples
    --------
    Exponential sweep between 50 and 22050 Hz

    .. plot::

        >>> import pyfar as pf
        >>> sweep = pf.signals.exponential_sweep_time(2**16, [50, 22050])
        >>> pf.plot.time_freq(sweep)
    """

    signal = _time_domain_sweep(
        n_samples, frequency_range, n_fade_out, amplitude, sampling_rate,
        "exponential", sweep_rate)

    return signal


def exponential_sweep_freq(
        n_samples, frequency_range, start_margin, stop_margin, n_fade_in=0,
        n_fade_out=0, bandpass_order=8, sampling_rate=44100,
        return_group_delay=False):
    """
    Generate sine sweep with exponentially increasing frequency in the
    frequency domain.

    Sweep synthesis according to [#]_.

    The exponential sweep can also be generated in the time domain
    (:py:func:`~exponential_sweep_time`). Frequency domain synthesis
    exhibits smooth magnitude spectra at the cost of a slightly irregular
    temporal envelope. Time domain synthesis exhibits a constant temporal
    envelope at the cost of slight ripples in the magnitude response.

    .. note::
        The envelope of the sweep time signal should be constant, apart from
        slight overshoots at the beginning and end. If this is not the case,
        try to increase `n_samples`, `start_margin`, `stop_margin`, `n_fade_in`
        or `n_fade_out`.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples.
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz. The restriction of the frequency range is realized
        by applying a Butterworth high-pass if ``frequency_range[0] > 0``
        and/or by a low-pass if ``frequency_range[1] < sampling_rate / 2``.
        Note that the exponential sweep can not start at 0 Hz, because its
        magnitude is defined by 1/frequency.
    start_margin : int, float
        The time in samples, at which the sweep starts. The start margin is
        required because the frequency domain sweep synthesis has pre-ringing
        in the time domain.
    stop_margin : int, float
        Time in samples, at which the sweep stops. This is relative to
        `n_samples`, e.g., a stop margin of 100 samples means that the sweep
        ends at sample ``n_samples-100``. This is required, because the
        frequency domain sweep synthesis has post-ringing in the time domain.
    n_fade_in : int, optional
        Duration of a squared sine fade-in in samples. The fade starts at the
        first sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal. The default ``0`` does not apply
        a fade-in.
    n_fade_out : int, optional
        Duration of a squared cosine fade-out in samples. The fade ends at the
        last sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal. The default ``0`` does not apply
        a fade-out.
    bandpass_order : int, optional
        The order of the Butterworth filters that are applied to limit the
        frequency range (see above). The default is ``8``.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.
    return_group_delay : boolean, optional
        Return the analytical group delay of the sweep. This can be used to
        compute the times at which distortion products appear. The default is
        ``False``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain, has a maximum
        absolute amplitude of 1 and the ``none`` FFT normalization
        (see :py:func:`~pyfar.dsp.fft.normalization`).
    group_delay_sweep : FrequencyData
        The analytical group delay of the sweep in seconds as a single sided
        spectrum. Only returned if `return_group_delay` is ``True``.

    References
    ----------
    .. [#] S. Müller, P. Massarani. 'Transfer Function Measurement with Sweeps.
           Directors Cut Including Previously Unreleased Material and some
           Corrections.' J. Audio Eng. Soc. 2001, 49 (6), 443-471.

    Examples
    --------
    Exponential sweep between 50 and 22050 Hz

    .. plot::

        >>> import pyfar as pf
        >>> sweep = pf.signals.exponential_sweep_freq(
        ...     2**16, [50, 22050], 5000, 1000)
        >>> pf.plot.time_freq(sweep)
    """

    return _frequency_domain_sweep(
        n_samples=n_samples,
        sweep_type='exponential',
        frequency_range=frequency_range,
        bandpass_order=bandpass_order,
        start_margin=start_margin,
        stop_margin=stop_margin,
        n_fade_in=n_fade_in,
        n_fade_out=n_fade_out,
        sampling_rate=sampling_rate,
        return_group_delay=return_group_delay)


def magnitude_spectrum_weighted_sweep(
        n_samples, magnitude_spectrum, start_margin, stop_margin,
        n_fade_in=0, n_fade_out=0, sampling_rate=44100,
        return_group_delay=False):
    """
    Generate sine sweep with arbitrary magnitude spectrum in the frequency
    domain.

    Frequency domain sine sweep synthesis according to [#]_. There is currently
    no method to design sine sweeps with arbitrary magnitude response in the
    time domain.

    .. note::
        The envelope of the sweep time signal should be constant, apart from
        slight overshoots at the beginning and end. If this is not the case,
        try to increase `n_samples`, `start_margin`, `stop_margin`, `n_fade_in`
        or `n_fade_out`, or provide a more smooth magnitude spectrum.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples.
    magnitude_spectrum : Signal
        The magnitude spectrum as a Signal object. Must be at least `n_samples`
        long and is zero-padded to `n_samples` otherwise.
    start_margin : int, float
        The time in samples, at which the sweep starts. The start margin is
        required because the frequency domain sweep synthesis has pre-ringing
        in the time domain.
    stop_margin : int, float
        Time in samples, at which the sweep stops. This is relative to
        `n_samples`, e.g., a stop margin of 100 samples means that the sweep
        ends at sample ``n_samples-100``. This is required, because the
        frequency domain sweep synthesis has post-ringing in the time domain.
    n_fade_in : int, optional
        Duration of a squared sine fade-in in samples. The fade starts at the
        first sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal. The default ``0`` does not apply
        a fade-in.
    n_fade_out : int, optional
        Duration of a squared cosine fade-out in samples. The fade ends at the
        last sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal. The default ``0`` does not apply
        a fade-out.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.
    return_group_delay : boolean, optional
        Return the analytical group delay of the sweep. This can be used to
        compute the times at which distortion products appear. The default is
        ``False``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain, has a maximum
        absolute amplitude of 1 and the ``none`` FFT normalization
        (see :py:func:`~pyfar.dsp.fft.normalization`).
    group_delay_sweep : FrequencyData
        The analytical group delay of the sweep in seconds as a single sided
        spectrum. Only returned if `return_group_delay` is ``True``.

    References
    ----------
    .. [#] S. Müller, P. Massarani. 'Transfer Function Measurement with Sweeps.
           Directors Cut Including Previously Unreleased Material and some
           Corrections.' J. Audio Eng. Soc. 2001, 49 (6), 443-471.

    Examples
    --------
    .. plot::

        >>> import pyfar as pf
        >>> magnitude = pf.dsp.filter.low_shelf(
        ...     pf.signals.impulse(2**16), 500, 20, 2)
        >>> magnitude = pf.dsp.filter.butterworth(magnitude, 8, 50, 'highpass')
        >>> sweep = pf.signals.magnitude_spectrum_weighted_sweep(
        ...     2**16, magnitude, 5000, 1000)
        >>> pf.plot.time_freq(sweep)
    """

    return _frequency_domain_sweep(
        n_samples=n_samples,
        sweep_type=magnitude_spectrum,
        frequency_range=[0, sampling_rate / 2],
        bandpass_order=0,
        start_margin=start_margin,
        stop_margin=stop_margin,
        n_fade_in=n_fade_in,
        n_fade_out=n_fade_out,
        sampling_rate=sampling_rate,
        return_group_delay=return_group_delay)


def linear_perfect_sweep(
        n_samples, sampling_rate=44100, return_group_delay=False):
    """
    Generate a perfect linear sweep in the frequency domain.

    The perfect sweep is generated according to [#]_ and is used for adaptive
    system identification. It is orthogonal to delayed versions of itself.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.
    return_group_delay : boolean, optional
        Return the analytical group delay of the sweep. This can be used to
        compute the times at which distortion products appear. The default is
        ``False``.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain, has a maximum
        absolute amplitude of 1 and the ``none`` FFT normalization
        (see :py:func:`~pyfar.dsp.fft.normalization`).
    group_delay_sweep : FrequencyData
        The analytical group delay of the sweep in seconds as a single sided
        spectrum. Only returned if `return_group_delay` is ``True``.

    References
    ----------
    .. [#] C. Antweiler, A. Telle, P. Vary, and G. Enzner, “Perfect-sweep NLMS
           for time-variant acoustic system identification,” in IEEE
           Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), Prague,
           Czech Republic, May 2011. doi: 10.1109/ICASSP.2012.6287930.

    Examples
    --------
    Plot a shifted perfect sweep to show that it can be looped

    .. plot::

        >>> import pyfar as pf
        >>> sweep = pf.signals.linear_perfect_sweep(2**8)
        >>> sweep_shifted = pf.dsp.time_shift(sweep, 2**7)
        >>> ax = pf.plot.time_freq(sweep_shifted, unit='samples')
        >>> ax[0].axvline(2**7, ls='--', c='r', label='start/end')
        >>> ax[0].legend(loc=4)

    Show that the sweep is orthogonal to all shifted version of itself. This
    property is important for adaptive system identification.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> sweep = pf.signals.linear_perfect_sweep(2**8)
        >>>
        >>> # compute auto-correlation
        >>> auto_correlation = np.empty(2**8)
        >>> for idx, shift in enumerate(range(-2**7, 2**7)):
        >>>     auto_correlation[idx] = np.dot(
        ...         sweep.time.flatten(),
        ...         np.roll(sweep.time.flatten(), shift))
        >>>
        >>> auto_correlation /= pf.dsp.energy(sweep)
        >>>
        >>> # plot auto-correlation
        >>> with pf.plot.context():
        >>>     plt.plot(np.arange(-2**7, 2**7), auto_correlation)
        >>>     plt.gca().set_xlim(-2**7, 2**7)
        >>>     plt.gca().set_xlabel('time lag in samples')
        >>>     plt.gca().set_ylabel('auto correlation')
    """

    return _frequency_domain_sweep(
        n_samples=n_samples,
        sweep_type='perfect_linear',
        frequency_range=[0, sampling_rate / 2],
        bandpass_order=0,
        start_margin=0,
        stop_margin=0,
        n_fade_in=0,
        n_fade_out=0,
        sampling_rate=sampling_rate,
        return_group_delay=return_group_delay)


def _frequency_domain_sweep(
        n_samples, sweep_type, frequency_range, bandpass_order,
        start_margin, stop_margin, n_fade_in, n_fade_out, sampling_rate,
        return_group_delay):
    """
    Frequency domain sweep synthesis with arbitrary magnitude response [#]_.

    Non-unique parameters are documented in ``linear_sweep_freq``,
    ``exponential_sweep_freq``, ``linear_perfect_sweep``, and
    ``magnitude_spectrum_weighted_sweep``.

    Parameters
    ----------
    n_samples : int
        The length of the sweep in samples.
    sweep_type : Signal, string
        Specify the magnitude response of the sweep.

        signal
            The magnitude response as :py:class:`~pyfar.Signal`
            object. If ``signal.n_samples`` is smaller than `n_samples`, zeros
            are padded to the end of `signal`. Note that `frequency_range` is
            not required in this case.
        ``'linear'``
            Design a sweep with linearly increasing frequency and a constant
            magnitude spectrum.
        ``'exponential'``
            Design a sweep with exponentially increasing frequency. The
            magnitude decreases by 3 dB per frequency doubling and has constant
            energy in filters of relative constant bandwidth (e.g. octaves).
        ``'perfect_linear'``
            Perfect linear sweep. Note that the parameters `start_margin`,
            `stop_margin`, and `frequency_range` are not required in this case.
    frequency_range : array like
        Frequency range of the sweep given by the lower and upper cut-off
        frequency in Hz. The restriction of the frequency range is realized
        by applying a Butterworth high-pass if ``frequency_range[0] > 0``
        and/or by a low-pass if ``frequency_range[1] < sampling_rate / 2``.
        Note that the exponential sweep can not start at 0 Hz, because its
        magnitude is defined by 1/frequency.
    bandpass_order : int
        The order of the Butterworth filters that are applied to limit the
        frequency range (see above).
    start_margin : int, float
        The time in samples, at which the sweep starts. The start margin is
        required because the frequency domain sweep synthesis has pre-ringing
        in the time domain.
    stop_margin : int, float
        Time in samples, at which the sweep stops. This is relative to
        `n_samples`, e.g., a stop margin of 100 samples means that the sweep
        ends at sample ``n_samples-100``. This is required, because the
        frequency domain sweep synthesis has post-ringing in the time domain.
    n_fade_in : int
        Duration of a squared sine fade-in in samples. The fade starts at the
        first sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal.
    n_fade_out : int
        Duration of a squared cosine fade-out in samples. The fade ends at the
        last sample of the sweep that is closer than 60 dB to the absolute
        maximum of the sweep time signal.
    sampling_rate : int
        The sampling rate in Hz.
    return_group_delay : boolean
        Return the analytical group delay of the sweep. This can be used to
        compute the times at which distortion products appear.

    Returns
    -------
    sweep : Signal
        The sweep signal. The Signal is in the time domain and has the ``none``
        FFT normalization (see :py:func:`~pyfar.dsp.fft.normalization`). The
        sweep parameters are written to `comment`.
    group_delay_sweep : FrequencyData
        The analytical group delay of the sweep in seconds as a single sided
        spectrum. Only returned if `return_group_delay` is ``True``.

    References
    ----------
    .. [#] S. Müller, P. Massarani. 'Transfer Function Measurement with Sweeps.
           Directors Cut Including Previously Unreleased Material and some
           Corrections.' J. Audio Eng. Soc. 2001, 49 (6), 443-471. (all
           equations referenced in the code refer to this).
    """

    # check input (only checks for user input, no checks for calling the
    # private function from the public functions)
    if isinstance(sweep_type, pyfar.Signal):
        magnitude = sweep_type
        sweep_type = 'signal'

    if np.atleast_1d(frequency_range).size != 2:
        raise ValueError(
            "Frequency_range must be an array like with two elements.")
    if frequency_range[0] < 0 or frequency_range[1] > sampling_rate/2:
        raise ValueError(
            ("Lower frequency limit must be at least 0 Hz and upper frequency "
             "limit must be below half the sampling rate."))
    if frequency_range[0] == 0 and sweep_type == "exponential":
        warnings.warn((
            "The exponential sweep has a 1/sqrt(frequency) magnitude spectrum."
            " The magnitude is set to 0 at 0 Hz to avoid a division by zero."),
            stacklevel=2)

    # initialize basic parameters ---------------------------------------------

    # spacing between frequency bins of FFT
    df = sampling_rate / n_samples

    # get number of bins (works for even and odd n_samples)
    n_bins = n_samples // 2 + 1

    # compute magnitude spectrum ----------------------------------------------
    if sweep_type == 'signal':
        # zero pad magnitude Signal or raise error if needed
        if n_samples > magnitude.n_samples:
            magnitude = pyfar.dsp.pad_zeros(
                magnitude, n_samples - magnitude.n_samples)
        elif magnitude.n_samples > n_samples:
            raise ValueError(
                (f'magnitude_spectrum has {magnitude.n_samples} samples '
                 f'but must not be longer than {n_samples}'))
        sweep_abs = np.abs(magnitude.freq_raw.flatten())
    elif sweep_type in ['linear', 'perfect_linear']:
        # constant spectrum
        sweep_abs = np.ones(n_bins)
    elif sweep_type == 'exponential':
        # 1/sqrt(f) spectrum
        sweep_abs = np.zeros(n_bins)
        sweep_abs[1:] = 1 / np.sqrt(2 * np.pi * np.arange(1, n_bins) * df)

    # band limit to magnitude spectrum
    if sweep_type in ['linear', 'exponential']:
        if frequency_range[0] > 0 and frequency_range[1] < sampling_rate / 2:
            band_limit = pyfar.dsp.filter.butterworth(
                pyfar.signals.impulse(n_samples, sampling_rate=sampling_rate),
                bandpass_order, frequency_range, 'bandpass')
        elif frequency_range[0] > 0:
            band_limit = pyfar.dsp.filter.butterworth(
                pyfar.signals.impulse(n_samples, sampling_rate=sampling_rate),
                bandpass_order, frequency_range[0], 'highpass')
        elif frequency_range[1] < sampling_rate / 2:
            band_limit = pyfar.dsp.filter.butterworth(
                pyfar.signals.impulse(n_samples, sampling_rate=sampling_rate),
                bandpass_order, frequency_range[1], 'lowpass')
        else:
            band_limit = pyfar.Signal(np.ones_like(sweep_abs), sampling_rate,
                                      n_samples, 'freq')
        sweep_abs *= np.abs(band_limit.freq.flatten())

    # compute group delay -----------------------------------------------------
    # group delay at 0 Hz must be 0
    sweep_gd = np.zeros(n_bins)
    # group delay at df equals starting time unless it's 0
    if start_margin > 0:
        sweep_gd[1] = start_margin / sampling_rate
        tg_start = 2
    else:
        tg_start = 1
    # group delay at Nyquist equals stopping time
    sweep_gd[-1] = (n_samples - stop_margin) / sampling_rate

    # FORMULA (11, p.40, in Müller and Massarani 2001)
    sweep_power = np.sum(np.abs(sweep_abs**2))
    C = (sweep_gd[-1] - sweep_gd[1]) / sweep_power

    # FORMULA (10, p.40, in Müller and Massarani 2001)
    for k in range(tg_start, n_bins):  # index 2 to nyq
        sweep_gd[k] = sweep_gd[k-1] + C * np.abs(sweep_abs[k])**2

    # compute phase from group delay ------------------------------------------
    sweep_ang = -1 * np.cumsum(sweep_gd) * 2 * np.pi * df

    # wrap and correct phase to be real 0 at Nyquist
    sweep_ang = pyfar.dsp.wrap_to_2pi(sweep_ang)
    sweep_ang[sweep_ang > np.pi] -= 2*np.pi

    if sweep_type == 'perfect_linear':
        sweep_ang[-1] = 0
    elif sweep_ang[-1] != 0 and not n_samples % 2:
        factor = np.cumsum(np.ones_like(sweep_ang)) - 1
        offset = df * sweep_ang[-1] / (sampling_rate / 2)
        sweep_ang -= factor * offset
        sweep_ang[-1] = np.abs(sweep_ang[-1])

    # compute and finalize return data ----------------------------------------
    # put group delay on pyfar FrequencyData
    sweep_gd = pyfar.FrequencyData(
        sweep_gd, pyfar.dsp.fft.rfftfreq(n_samples, sampling_rate))

    # combine magnitude and phase of sweep
    sweep = sweep_abs * np.exp(1j * sweep_ang)

    # put sweep in pyfar.Signal an transform to time domain
    sweep = pyfar.Signal(sweep, sampling_rate, n_samples, 'freq', 'none')

    # find windowing end points heuristically. The start and stop margin are
    # not reliable, because freq. domain synthesis causes pre/post-ringing
    if n_fade_in or n_fade_out:
        threshold = 10**(-60/20) * np.max(np.abs(sweep.time))
        fade_start = np.argmax(np.abs(sweep.time) > threshold)
        fade_end = n_samples - \
            np.argmax(np.abs(sweep.time.flatten()[::-1]) > threshold)
        print([fade_start, fade_end])

    # generate windows for fade in and/or out
    if n_fade_in and n_fade_out:
        fade = [fade_start, fade_start + n_fade_in,
                fade_end - n_fade_out, fade_end]
        shape = 'symmetric'
    elif n_fade_in:
        fade = [fade_start, fade_start + n_fade_in]
        shape = 'left'
    elif n_fade_out:
        fade = [fade_end - n_fade_out, fade_end]
        shape = 'right'
    else:
        fade = None

    # apply window
    if fade is not None:
        sweep = pyfar.dsp.time_window(sweep, fade, shape=shape)

    # normalize to time domain amplitude
    sweep = pyfar.dsp.normalize(sweep)

    if return_group_delay:
        return sweep, sweep_gd
    else:
        return sweep


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
    using np.broadcast_shapes and np.broadcast_to().

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

    # broadcast shapes
    shape = np.broadcast_shapes(*[np.atleast_1d(arg).shape for arg in args])

    # match the shape
    result = []
    for arg in args:
        arg = np.broadcast_to(arg, shape)
        arg.setflags(write=1)
        result.append(arg)

    return shape, result
