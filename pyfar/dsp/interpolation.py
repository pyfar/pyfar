import numpy as np
from scipy.special import iv as bessel_first_mod
from scipy.interpolate import interp1d
import scipy.signal as sgn
import matplotlib.pyplot as plt
import pyfar as pf
from scipy.ndimage import generic_filter1d
from fractions import Fraction
from decimal import Decimal
import warnings


def _weighted_moving_average(input, output, weights):
    """Moving average filter of length N and arbitrary.

    Parameters
    ----------
    input : numpy.ndarray
        The input array
    output : numpy.ndarray
        The output buffer
    N : int
        Length of the filter
    weights : numpy.ndarray
        The weights used for averaging. The length of the weights also
        specifies the length of the filter.

    Note
    ----
    This function is primarily intended to be used in combination with
    ``scipy.ndimage.generic_filter1d``. The input is strided instead of
    reshaped, leaving the memory layout unchanged. The function does also not
    return it's output but requires the output buffer as function input, which
    is required by ``scipy.ndimage.generic_filter1d``.

    """
    strided = np.lib.stride_tricks.as_strided(
        input, strides=input.strides*2,
        shape=(weights.size, input.size - (weights.size-1)))
    output[:] = np.average(strided, weights=weights, axis=0)


def smooth_fractional_octave(signal, num_fractions, mode="magnitude_zerophase",
                             window="boxcar"):
    """
    Smooth spectrum with a fractional octave width.

    The smoothing is done according to Tylka et al. 2017 [#]_ (method 2) in
    three steps:

    1. Interpolate the spectrum to a logarithmically spaced frequency scale
    2. Smooth the spectrum by convolution with a smoothing window
    3. Interpolate the spectrum to the original linear frequency scale


    Parameters
    ----------
    signal : pyfar.Signal
        The input data.
    num_fractions : number
        The width of the smoothing window in fractional octaves, e.g., 3 will
        apply third octave smoothing and 1 will apply octave smoothing.
    mode : str, optional

        ``"magnitude_zerophase"``
            Only the magnitude response, i.e., the absolute spectrum is
            smoothed. Note that this return a zero-phase signal. It might be
            necessary to generate a minimum or linear phase if the data is
            subject to further processing after the smoothing (cf.
            :py:func:`~pyfar.dsp.minimum_phase` and
            :py:func:`~pyfar.dsp.linear_phase`)
        ``"magnitude"``
            Smooth the magnitude and keep the phase of the input signal.
        ``"magnitude_phase"``
            Separately smooth the magnitude and unwrapped phase response.
        ``"complex"``
            Separately smooth the real and imaginary part of the spectrum.

        Note that the modes `magnitude_zerophase` and `magnitude` make sure
        that the smoothed magnitude response is as expected at the cost of an
        artificial phase response. This is often desired, e.g., when plotting
        signals or designing compensation filters. The modes `magnitude_phase`
        and `complex` smooth all information but might cause a high frequency
        energy loss in the smoothed magnitude response. The default is
        ``"magnitude_zerophase"``.
    window : str, optional
        String that defines the smoothing window. All windows from
        :py:func:`~pyfar.dsp.time_window` that do not require an additional
        parameter can be used. The default is "boxcar", which uses the
        most commonly used rectangular window.

    Returns
    -------
    signal : pyfar.Signal
        The smoothed output data
    window_stats : tuple
        A tuple containing information about the smoothing process

        `n_window`
            The window length in (logarithmically spaced) samples
        `num_fractions`
            The actual width of the window in fractional octaves. This can
            deviate from the desired width because the smoothing window must
            have an integer sample length

    Notes
    -----
    Method 3 in Tylka at al. 2017 is mathematically more elegant at the
    price of a largely increased computational and memory cost. In most
    practical cases, methods 2 and 3 yield close to identical results (cf. Fig.
    2 and 3 in Tylka et al. 2017). If the spectrum contains extreme
    discontinuities, however, method 3 is superior (see examples below).

    References
    ----------
    .. [#] J. G. Tylka, B. B. Boren, and E. Y. Choueiri, “A Generalized Method
           for Fractional-Octave Smoothing of Transfer Functions that Preserves
           Log-Frequency Symmetry (Engineering Report),” J. Audio Eng. Soc. 65,
           239-245 (2017). doi:10.17743/jaes.2016.0053

    Examples
    --------

    Octave smoothing of continuous spectrum consisting of two bell filters.

    .. plot::

        >>> import pyfar as pf
        >>> signal = pf.signals.impulse(441)
        >>> signal = pf.dsp.filter.bell(signal, 1e3, 12, 1, "III")
        >>> signal = pf.dsp.filter.bell(signal, 10e3, -60, 100, "III")
        >>> smoothed, _ = pf.dsp.smooth_fractional_octave(signal, 1)
        >>> ax = pf.plot.freq(signal, label="input")
        >>> pf.plot.freq(smoothed, label="smoothed")
        >>> ax.legend(loc=3)

    Octave smoothing of the discontinuous spectrum of a sine signal causes
    artifacts at the edges due to the intermediate interpolation steps (cf.
    Tylka et al. 2017, Fig. 4). However this is a rather unusual application
    and is mentioned only for the sake of completeness.

    .. plot::

        >>> import pyfar as pf
        >>> signal = pf.signals.sine(1e3, 4410)
        >>> signal.fft_norm = "amplitude"
        >>> smoothed, _ = pf.dsp.smooth_fractional_octave(signal, 1)
        >>> ax = pf.plot.freq(signal, label="input")
        >>> pf.plot.freq(smoothed, label="smoothed")
        >>> ax.set_xlim(200, 4e3)
        >>> ax.set_ylim(-45, 5)
        >>> ax.legend(loc=3)
    """

    if not isinstance(signal, pf.Signal):
        raise TypeError("Input signal has to be of type pyfar.Signal")

    if mode in ["magnitude_zerophase", "magnitude"]:
        data = [np.atleast_2d(np.abs(signal.freq_raw))]
    elif mode == "complex":
        data = [np.atleast_2d(np.real(signal.freq_raw)),
                np.atleast_2d(np.imag(signal.freq_raw))]
    elif mode == "magnitude_phase":
        data = [np.atleast_2d(np.abs(signal.freq_raw)),
                np.atleast_2d(pf.dsp.phase(signal, unwrap=True))]
    else:
        raise ValueError((f"mode is '{mode}' but must be 'magnitude_zerophase'"
                          ", 'magnitude_phase', 'magnitude', or 'complex'"))

    # linearly and logarithmically spaced frequency bins ----------------------
    N = signal.n_bins
    n_lin = np.arange(N)
    n_log = N**(n_lin/(N-1))

    # frequency bin spacing in octaves: log2(n_log[n]/n_log[n-1])
    # Note: n_log[0] = 1
    delta_n = np.log2(n_log[1])

    # width of the window in logarithmically spaced samples
    # Note: Forcing the window to have an odd length increases the deviation
    #       from the exact width, but makes sure that the delay introduced in
    #       the convolution is integer and can be easily compensated
    n_window = int(2 * np.floor(1 / (num_fractions * delta_n * 2)) + 1)

    if n_window == 1:
        raise ValueError((
            "The smoothing width given by num_fractions is below the frequency"
            " resolution of the signal. Increase the signal length or decrease"
            " num_fractions"))

    # generate the smoothing window
    if isinstance(window, str):
        window = sgn.windows.get_window(window, n_window, fftbins=False)
    elif isinstance(window, (list, np.ndarray)):
        # undocumented possibility for testing
        window = np.asanyarray(window, dtype=float)
        if window.shape != (n_window, ):
            raise ValueError(
                f"window.shape is {window.shape} but must be ({n_window}, )")
    else:
        raise ValueError(f"window is of type {str(type(window))} but must be "
                         "of type string")

    for nn in range(len(data)):
        # interpolate to logarithmically spaced frequencies
        interpolator = interp1d(
            n_lin + 1, data[nn], "cubic", copy=False, assume_sorted=True)
        data[nn] = interpolator(n_log)

        # apply a moving average filter based on the window function
        data[nn] = generic_filter1d(
            data[nn],
            function=_weighted_moving_average,
            filter_size=n_window,
            mode='nearest',
            extra_arguments=(window,))

        # interpolate to original frequency axis
        interpolator = interp1d(
            n_log, data[nn], "cubic", copy=False, assume_sorted=True)
        data[nn] = interpolator(n_lin + 1)

    # generate return signal --------------------------------------------------
    if mode == "magnitude_zerophase":
        data = data[0]
    elif mode == "complex":
        data = data[0] + 1j * data[1]
    elif mode == "magnitude_phase":
        data = data[0] * np.exp(1j * data[1])
    elif mode == "magnitude":
        data = data[0] * np.exp(1j * np.angle(signal.freq_raw))

    # force 0 Hz and Nyquist to be real if it might not be the case
    if mode in ["complex", "magnitude_phase", "magnitude"]:
        data[..., 0] = np.abs(data[..., 0])
        data[..., -1] = np.abs(data[..., -1])

    signal = signal.copy()
    signal.freq_raw = data

    return signal, (n_window, 1 / (n_window * delta_n))


def fractional_time_shift(signal, shift, unit="samples", order=30,
                          side_lobe_suppression=60, mode="linear"):
    """
    Apply fractional time shift to input data.

    This function uses a windowed Sinc filter (Method FIR-2 in [#]_ according
    to Equations 21 and 22) to apply fractional delays, i.e., non-integer
    delays to an input signal. A Kaiser window according to [#]_ Equations
    (10.12) and (10.13) is used, which offers the possibility to control the
    side lobe suppression.

    Parameters
    ----------
    signal : Signal
        The input data
    shift : float, array like
        The fractional shift in samples (positive or negative). If this is a
        float, the same shift is applied to all channels of `signal`. If this
        is an array like different delays are applied to the channels of
        `signal`. In this case it must broadcast to `signal.cshape` (see
        `Numpy broadcasting
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_)
    order : int, optional
        The order of the fractional shift (sinc) filter. The precision of the
        filter increases with the order. High frequency errors decrease with
        increasing order. The order must be smaller than
        ``signal.n_samples``. The default is ``30``.
    side_lobe_suppression : float, optional
        The side lobe suppression of the Kaiser window in dB. The default is
        ``60``.
    mode : str, optional
        The filtering mode

        ``"linear"``
            Apply linear shift, i.e., parts of the signal that are shifted to
            times smaller than 0 samples and larger than ``signal.n_samples``
            disappear.
        ``"cyclic"``
            Apply a cyclic shift, i.e., parts of the signal that are shifted to
            values smaller than 0 are wrapped around to the end, and parts that
            are shifted to values larger than ``signal.n_samples`` are wrapped
            around to the beginning.

        The default is ``"linear"``

    Returns
    -------
    signal : Signal
        The delayed input data

    References
    ----------

    .. [#] T. I. Laakso, V. Välimäki, M. Karjalainen, and U. K. Laine,
           'Splitting the unit delay,' IEEE Signal Processing Magazine 13,
           30-60 (1996). doi:10.1109/79.482137
    .. [#] A. V. Oppenheim and R. W. Schafer, Discrete-time signal processing,
           (Upper Saddle et al., Pearson, 2010), Third edition.


    Examples
    --------

    Apply a fractional shift of 2.3 samples using filters of orders 6 and 30

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> signal = pf.signals.impulse(64, 10)
        >>>
        >>> pf.plot.use()
        >>> _, ax = plt.subplots(3, 1, figsize=(8, 8))
        >>> pf.plot.time_freq(signal, ax=ax[:2], label="input")
        >>> pf.plot.group_delay(signal, ax=ax[2], unit="samples")
        >>>
        >>> for order in [30, 6]:
        >>>     delayed = pf.dsp.fractional_time_shift(
        ...         signal, 2.3, order=order)
        >>>     pf.plot.time_freq(delayed, ax=ax[:2],
        ...                       label=f"delayed, order={order}")
        >>>     pf.plot.group_delay(delayed, ax=ax[2], unit="samples")
        >>>
        >>> ax[1].set_ylim(-15, 5)
        >>> ax[2].set_ylim(8, 14)
        >>> ax[0].legend()

    Apply a shift that exceeds the signal length using the modes ``"linear"``
    and ``"cyclic"``

    .. plot::

        >>> import pyfar as pf
        >>>
        >>> signal = pf.signals.impulse(32, 16)
        >>>
        >>> ax = pf.plot.time(signal, label="input")
        >>>
        >>> for mode in ["cyclic", "linear"]:
        >>>     delayed = pf.dsp.fractional_time_shift(
        ...         signal, 25.3, order=10, mode=mode)
        >>>     pf.plot.time(delayed, label=f"delayed, mode={mode}")
        >>>
        >>> ax.legend()
    """

    # check input -------------------------------------------------------------
    if not isinstance(signal, (pf.Signal)):
        raise TypeError("Input data has to be of type pyfar.Signal")
    if order <= 0:
        raise ValueError("The order must be > 0")
    if side_lobe_suppression <= 0:
        raise ValueError("The side lobe suppression must be > 0")
    if mode not in ["linear", "cyclic"]:
        raise ValueError(
            f"The mode is '{mode}' but must be 'linear' or 'cyclic'")
    if order + 1 > signal.n_samples:
        raise ValueError((f"The order is {order} but must not exceed "
                          f"{signal.n_samples-1} (signal.n_samples-1)"))

    if unit == 's':
        shift = shift*signal.sampling_rate
    elif unit != 'samples':
        raise ValueError(
            f"Unit is '{unit}' but has to be 'samples' or 's'.")

    # separate integer and fractional shift -----------------------------------
    delay_int = np.atleast_1d(shift).astype(int)
    delay_frac = np.atleast_1d(shift - delay_int)
    # force delay_frac >= 0 as required by Laakso et al. 1996 Eq. (2)
    mask = delay_frac < 0
    delay_int[mask] -= 1
    delay_frac[mask] += 1

    # compute the sinc functions (fractional shift filters) -------------------
    # Laakso et al. 1996 Eq. (21) applied to the fractional part of the shift
    # M_opt essentially sets the center of the sinc function in the FIR filter.
    # NOTE: This is also  the shift that is added when applying the fractional
    #       part of the shift and has thus to be accounted for when realizing
    #       delay_int
    if order % 2:
        M_opt = delay_frac.astype("int") - (order-1)/2
    else:
        M_opt = np.round(delay_frac) - order / 2

    # get matrix versions of the fractional shift and M_opt
    delay_frac_matrix = np.tile(
        delay_frac[..., np.newaxis],
        tuple(np.ones(delay_frac.ndim, dtype="int")) + (order + 1, ))
    M_opt_matrix = np.tile(
        M_opt[..., np.newaxis],
        tuple(np.ones(M_opt.ndim, dtype="int")) + (order + 1, ))

    # discrete time vector
    n = np.arange(order + 1) + M_opt_matrix - delay_frac_matrix

    sinc = np.sinc(n)

    # get the Kaiser windows --------------------------------------------------
    # (dsp.time_window can not be used because we need to evaluate the window
    #  for non integer values)

    # beta parameter for side lobe rejection according to
    # Oppenheim (2010) Eq. (10.13)
    beta = pf.dsp.kaiser_window_beta(abs(side_lobe_suppression))

    # Kaiser window according to Oppenheim (2010) Eq. (10.12)
    alpha = order / 2
    L = np.arange(order + 1).astype("float") - delay_frac_matrix
    # required to counter operations on M_opt and make sure that the maxima
    # of the underlying continuous sinc function and Kaiser window appear at
    # the same time
    if order % 2:
        L += .5
    else:
        L[delay_frac_matrix > .5] += 1
    Z = beta * np.sqrt(np.array(1 - ((L - alpha) / alpha)**2, dtype="complex"))
    # suppress small imaginary parts
    kaiser = np.real(bessel_first_mod(0, Z)) / bessel_first_mod(0, beta)

    # apply fractional shift --------------------------------------------------
    # compute filter and match dimensions
    frac_delay_filter = sinc * kaiser
    while frac_delay_filter.ndim < signal.time.ndim:
        frac_delay_filter = frac_delay_filter[np.newaxis]
    # apply filter
    convolve_mode = mode if mode == "cyclic" else "full"
    n_samples = signal.n_samples

    signal = pf.dsp.convolve(
        signal, pf.Signal(frac_delay_filter, signal.sampling_rate),
        mode=convolve_mode)

    # apply integer shift -----------------------------------------------------
    # account for shift from applying the fractional filter
    delay_int += M_opt.astype("int")
    signal = pf.dsp.time_shift(signal, delay_int, mode)

    # truncate signal (got padded during convolution with mode='full')
    if mode == "linear":
        signal.time = signal.time[..., :n_samples]

    return signal


def resample(signal, sampling_rate, match_amplitude="auto", frac_limit=None,
             post_filter=False):
    """Resample signal to new sampling rate.

    The SciPy function ``scipy.signal.resample_poly`` is used for resampling.
    The resampling ratio ``L = sampling_rate/signal.sampling_rate``
    is approximated by a fraction of two integer numbers `up/down` to first
    upsample the signal by `up` and then downsample by `down`. This way `up`
    and `down` are smaller than the respective new and old sampling rates.

    .. note ::

        `sampling_rate` should be divisible by 10, otherwise it can cause an
        infinite loop in the ``resample_poly`` function.

        The amplitudes of the resampled signal can match the amplitude of the
        input signal in the time or frequency domain. See the parameter
        `match_amplitude` and the examples for more information.

    Parameters
    ----------
    signal : Signal
        Input data to be resampled
    sampling_rate : number
        The new sampling rate in Hz
    match_amplitude : string
        Define the domain to match the amplitude of the resampled data.

        ``'auto'``
            Chooses domain to maintain the amplitude automatically, depending
            on the ``signal.signal_type``. Sets ``match_amplitude == 'freq'``
            for ``signal.signal_type = 'energy'`` like impulse responses and
            ``match_amplitude == 'time'`` for other signals.
        ``'time'``
            Maintains the amplitude in the time domain. This is useful for
            recordings such as speech or music and must be used if
            ``signal.signal_type = 'power'``.
        ``'freq'``
            Maintains the amplitude in the frequency domain by multiplying the
            resampled signal by ``1/L`` (see above). This is often desired
            when resampling impulse responses.

        The default is ``'auto'``.
    frac_limit : int
        Limit the denominator for approximating the resampling factor `L`
        (see above). This can be used in case the resampling gets stuck in an
        infinite loop (see note above) at the potenital cost of not exactly
        realizing the target sampling rate.

        The default is ``None``, which uses ``frac_limit = 1e6``.
    post_filter : bool, optional
        In some cases the up-sampling causes artifacts above the Nyquist
        frequency of the input signal, i.e., ``signal.sampling_rate/2``. If
        ``True`` the artifacts are suppressed by applying a zero-phase Elliptic
        filter with a pass band ripple of 0.1 dB, a stop band attenuation of 60
        dB. The pass band edge frequency is ``signal.sampling_rate/2``. The
        stop band edge frequency is the minimum of 1.05 times the pass band
        frequency and the new Nyquist frequency (``sampling_rate/2``). The
        default is ``False``. Note that this is only applied in case of
        up-sampling.

    Returns
    -------
    signal : pyfar.Signal
        The resampled signal of the input data with a length of
        `up/down * signal.n_samples` samples.

    Examples
    --------
    For power signals, the amplitude of the resampled signal is automatically
    correct in the time `and` frequency domain if ``match_amplitude="time"``

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> signal = pf.signals.sine(200, 4800, sampling_rate=48000)
        >>> resampled = pf.dsp.resample(signal, 96000)
        >>>
        >>> pf.plot.time_freq(signal, label="original")
        >>> pf.plot.time_freq(resampled, c="y", ls=":",
        ...                   label="resampled (time domain matched)")
        >>> plt.legend()

    With some energy signals, such as impulse responses, the amplitude can only
    be correct in the time `or` frequency domain due to the lack of
    normalization by the number of samples. In such cases, it is often desired
    to match the amplitude in the frequency domain

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> signal = pf.signals.impulse(128, 64, sampling_rate=48000)
        >>> resampled_time = pf.dsp.resample(
        ...     signal, 96000, match_amplitude = "time")
        >>> resampled_freq = pf.dsp.resample(
        ...     signal, 96000, match_amplitude = "freq")
        >>>
        >>> pf.plot.time_freq(signal, label="original")
        >>> pf.plot.time_freq(resampled_freq, dashes=[2, 3],
        ...                   label="resampled (freq. domain matched)")
        >>> ax = pf.plot.time_freq(resampled_time, ls=":",
        ...                   label="resampled (time domain matched)", c='y')
        >>> ax[0].set_xlim(1.2,1.46)
        >>> plt.legend()
    """
    # check input
    if not isinstance(signal, (pf.Signal)):
        raise TypeError("Input data has to be of type pyfar.Signal")
    # calculate factor L for up- or downsampling
    sampling_rate_old = signal.sampling_rate
    L = sampling_rate / sampling_rate_old
    # set match_amplitude domain depending on signal.signal_type
    if match_amplitude == "auto":
        match_amplitude = "freq" if signal.signal_type == "energy" else "time"
    # set gain depending on domain to match aplitude in
    if match_amplitude == "time":
        gain = 1
    elif match_amplitude == "freq":
        gain = 1/L
        # the aplitude of signals with signal_type "power" must be matched in
        # the time domain
        if signal.signal_type == "power":
            raise ValueError((
                'match_amplitude must be "time" if signal.signal_type is '
                '"power".'))
    else:
        raise ValueError((f"match_amplitude is '{match_amplitude}' but must be"
                          " 'auto', 'time' or 'freq'"))
    # check if one of the sampling rates is not divisible by 10
    if sampling_rate % 10 or sampling_rate_old % 10:
        warnings.warn((
            'At least one sampling rate is not divisible by 10, , which can '
            'cause a infinite loop in `scipy.resample_poly`. If this occurs, '
            'interrupt and choose different sampling rates or decrease '
            'frac_limit. However, this can cause an error in the target '
            'sampling rate realisation.'))
    # give the numerator and denomitor of the fraction for factor L
    if frac_limit is None:
        frac = Fraction(Decimal(L)).limit_denominator()
    else:
        frac = Fraction(Decimal(L)).limit_denominator(frac_limit)
    up, down = frac.numerator, frac.denominator
    # calculate an error depending on samplings rates and fraction
    error = abs(sampling_rate_old * up / down - sampling_rate)
    if error != 0.0:
        warnings.warn((
            f'The target sampling rate was realized with an error of {error}.'
            f'The error might be decreased by setting `frac_limit` to a value '
            f'larger than {down} (This warning is not shown, if the target '
            'sampling rate can exactly be realized).'))
    # resample data with scipy resampe_poly function
    data = sgn.resample_poly(signal.time, up, down, axis=-1)
    data = pf.Signal(data * gain, sampling_rate, fft_norm=signal.fft_norm,
                     comment=signal.comment)

    if post_filter and L > 1:

        # Design elliptic filter
        # (pass band is given by nyquist frequency of input signal, other
        # parameters are freely chosen)
        wp = sampling_rate_old / 2 / sampling_rate * 2
        ws = min(1, 1.05 * wp)
        gpass = .1
        gstop = 60

        # calculate the required order and -3 dB cut-off frequency
        N, f_c = sgn.ellipord(wp, ws, gpass, gstop/2, fs=sampling_rate)
        f_c *= sampling_rate / 2

        # apply zero-phase filter
        data = pf.dsp.filter.elliptic(data, N, gpass, gstop/2, f_c, 'lowpass')
        data.time = np.flip(data.time, axis=-1)
        data = pf.dsp.filter.elliptic(data, N, gpass, gstop/2, f_c, 'lowpass')
        data.time = np.flip(data.time, axis=-1)

    return data


class InterpolateSpectrum():
    """
    Interpolate an incomplete spectrum to a complete single sided spectrum.

    This is intended to interpolate transfer functions, for example sparse
    spectra that are defined only at octave frequencies or incomplete spectra
    from numerical simulations.

    Parameters
    ----------
    data : FrequencyData
        Input data to be interpolated. `data.fft_norm` must be `'none'`.
    method : string
        Specifies the input data for the interpolation

        ``'complex'``
            Separate interpolation of the real and imaginary part
        ``'magnitude_phase'``
            Separate interpolation if the magnitude and unwrapped phase values
        ``'magnitude'``
            Interpolate the magnitude values only. Results in a zero phase
            signal, which is symmetric around the first sample. This phase
            response might not be ideal for many applications. Minimum and
            linear phase responses can be generated with
            :py:func:`~pyfar.dsp.minimum_phase` and
            :py:func:`~pyfar.dsp.linear_phase`.

    kind : tuple
        Three element tuple ``('first', 'second', 'third')`` that specifies the
        kind of inter/extrapolation below the lowest frequency (first), between
        the lowest and highest frequency (second), and above the highest
        frequency (third).

        The individual strings have to be

        ``'zero'``, ``slinear``, ``'quadratic'``, ``'cubic'``
            Spline interpolation of zeroth, first, second or third order
        ``'previous'``, ``'next'``
            Simply return the previous or next value of the point
        ``'nearest-up'``, ``'nearest'``
            Differ when interpolating half-integers (e.g. 0.5, 1.5) in that
            ``'nearest-up'`` rounds up and ``'nearest'`` rounds down.

        The interpolation is done using ``scipy.interpolate.interp1d``.
    fscale : string, optional

        ``'linear'``
            Interpolate on a linear frequency axis.
        ``'log'``
            Interpolate on a logarithmic frequency axis. Note that 0 Hz can
            not be interpolated on a logarithmic scale because the logarithm
            of 0 does not exist. Frequencies of 0 Hz are thus replaced by the
            next highest frequency before interpolation.

        The default is ``'linear'``.
    clip : bool, tuple
        The interpolated magnitude response is clipped to the range specified
        by this two element tuple. E.g., ``clip=(0, 1)`` will assure that no
        values smaller than 0 and larger than 1 occur in the interpolated
        magnitude response. The clipping is applied after the interpolation.
        The default is ``False`` which does not clip the data.

    Returns
    -------
    interpolator : :py:class:`InterpolateSpectrum`
        The interpolator can be called to interpolate the data (see examples
        below). It returns a :py:class:`~pyfar.classes.audio.Signal` and has
        the following parameters

        `n_samples` : int
            Length of the interpolated time signal in samples
        `sampling_rate`: int
            Sampling rate of the output signal in Hz
        `show` : bool, optional
            Show a plot of the input and output data. The default is ``False``.

    Examples
    --------
    Interpolate a magnitude spectrum, add an artificial linear phase and
    inspect the results.
    Note that a similar plot can be created by the interpolator object by
    ``signal = interpolator(64, 44100, show=True)``

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> # generate data
        >>> data = pf.FrequencyData([1, 0], [5e3, 20e3])
        >>> interpolator = pf.dsp.InterpolateSpectrum(
        ...     data, 'magnitude', ('nearest', 'linear', 'nearest'))
        >>> # interpolate 64 samples at a sampling rate of 44100
        >>> signal = interpolator(64, 44100)
        >>> # add linear phase
        >>> signal = pf.dsp.linear_phase(signal, 32)
        >>> # plot input and output data
        >>> with pf.plot.context():
        >>>     _, ax = plt.subplots(2, 2)
        >>>     # time signal (linear and logarithmic amplitude)
        >>>     pf.plot.time(signal, ax=ax[0, 0])
        >>>     pf.plot.time(signal, ax=ax[1, 0], dB=True)
        >>>     # frequency plot (linear x-axis)
        >>>     pf.plot.freq(signal, dB=False, freq_scale="linear",
        ...                  ax=ax[0, 1])
        >>>     pf.plot.freq(data, dB=False, freq_scale="linear",
        ...                  ax=ax[0, 1], c='r', ls='', marker='.')
        >>>     ax[0, 1].set_xlim(0, signal.sampling_rate/2)
        >>>     # frequency plot (log x-axis)
        >>>     pf.plot.freq(signal, dB=False, ax=ax[1, 1], label='input')
        >>>     pf.plot.freq(data, dB=False, ax=ax[1, 1],
        ...                  c='r', ls='', marker='.', label='output')
        >>>     min_freq = np.min([signal.sampling_rate / signal.n_samples,
        ...                        data.frequencies[0]])
        >>>     ax[1, 1].set_xlim(min_freq, signal.sampling_rate/2)
        >>>     ax[1, 1].legend(loc='best')

    """

    def __init__(self, data, method, kind, fscale='linear',
                 clip=False, group_delay=None, unit='samples'):

        # check input ---------------------------------------------------------
        # ... data
        if not isinstance(data, pf.FrequencyData):
            raise TypeError('data must be a FrequencyData object.')
        if data.n_bins < 2:
            raise ValueError("data.n_bins must be at least 2")

        # ... method
        methods = ['complex', 'magnitude_phase', 'magnitude']
        if method not in methods:
            raise ValueError((f"method is '{method}'' but must be on of the "
                              f"following: {', '.join(methods)}"))

        # ... kind
        if not isinstance(kind, tuple) or len(kind) != 3:
            raise ValueError("kind must be a tuple of length 3")
        kinds = ['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                 'quadratic', 'cubic', 'previous', 'next']
        for k in kind:
            if k not in kinds:
                raise ValueError((f"kind contains '{k}' but must only contain "
                                  f"the following: {', '.join(kinds)}"))

        # ... fscale
        if fscale not in ["linear", "log"]:
            raise ValueError(
                f"fscale is '{fscale}'' but must be linear or log")

        # ... clip
        if clip:
            if not isinstance(clip, tuple) or len(clip) != 2:
                raise ValueError("clip must be a tuple of length 2")

        # initialize the interpolators ----------------------------------------
        # store required parameters
        self._method = method
        self._clip = clip
        self._fscale = fscale

        # flatten input data to work with scipy interpolators
        self._cshape = data.cshape
        data = data.flatten()
        self._input = data

        # get the required data for interpolation
        if method == 'complex':
            self._data = [np.real(data.freq), np.imag(data.freq)]
        elif method == 'magnitude_phase':
            self._data = [np.abs(data.freq),
                          pf.dsp.phase(data, unwrap=True)]
        else:
            self._data = [np.abs(data.freq)]

        # frequencies for interpolation (store for testing)
        self._f_in = self._get_frequencies(data.frequencies.copy())

        # frequency range
        self._freq_range = [self._f_in[0], self._f_in[-1]]

        # get the interpolators
        self._interpolators = []
        for d in self._data:
            interpolators = []
            for idx, k in enumerate(kind):
                if idx == 1:
                    interpolators.append(interp1d(self._f_in, d, k))
                else:
                    interpolators.append(interp1d(
                        self._f_in, d, k, fill_value="extrapolate"))
            self._interpolators.append(interpolators)

    def __call__(self, n_samples, sampling_rate, show=False):
        """
        Interpolate a Signal with n_samples length.
        (see class docstring) for more information.
        """

        # get the query frequencies (store for testing)
        self._f_query = self._get_frequencies(
            pf.dsp.fft.rfftfreq(n_samples, sampling_rate))

        # get interpolation ranges
        id_below = self._f_query < self._freq_range[0]
        id_within = np.logical_and(self._f_query >= self._freq_range[0],
                                   self._f_query <= self._freq_range[1])
        id_above = self._f_query > self._freq_range[1]

        # interpolate the data
        interpolated = []
        for data in self._interpolators:
            data_interpolated = np.concatenate((
                (data[0](self._f_query[id_below])),
                (data[1](self._f_query[id_within])),
                (data[2](self._f_query[id_above]))),
                axis=-1)
            interpolated.append(data_interpolated)

        # get half sided spectrum
        if self._method == "complex":
            freq = interpolated[0] + 1j * interpolated[1]
        elif self._method == 'magnitude_phase':
            freq = interpolated[0] * np.exp(-1j * interpolated[1])
        else:
            freq = interpolated[0]

        # get initial signal
        signal = pf.Signal(freq, sampling_rate, n_samples, "freq")

        # clip the magnitude
        if self._clip:
            signal.freq = np.clip(
                np.abs(signal.freq),
                self._clip[0],
                self._clip[1]) * np.exp(-1j * pf.dsp.phase(signal))

        if show:
            # plot input and output data
            with pf.plot.context():
                _, ax = plt.subplots(2, 2)
                # time signal (linear amplitude)
                pf.plot.time(signal, ax=ax[0, 0])
                # time signal (log amplitude)
                pf.plot.time(signal, ax=ax[1, 0], dB=True)
                # frequency plot (linear x-axis)
                pf.plot.freq(signal, dB=False, freq_scale="linear",
                             ax=ax[0, 1])
                pf.plot.freq(self._input, dB=False, freq_scale="linear",
                             ax=ax[0, 1], c='r', ls='', marker='.')
                ax[0, 1].set_xlim(0, sampling_rate/2)
                # frequency plot (log x-axis)
                pf.plot.freq(signal, dB=False, ax=ax[1, 1], label='input')
                pf.plot.freq(self._input, dB=False, ax=ax[1, 1],
                             c='r', ls='', marker='.', label='output')
                min_freq = np.min([sampling_rate / n_samples,
                                   self._input.frequencies[0]])
                ax[1, 1].set_xlim(min_freq, sampling_rate/2)
                ax[1, 1].legend(loc='best')

        return signal

    def _get_frequencies(self, frequencies):
        """
        Return frequencies for creating or quering interpolation objects.

        In case logfrequencies are requested, 0 Hz entries are replaced by
        the next highest frequency, because the logarithm of 0 does not exist.
        """
        if self._fscale == "log":
            if frequencies[0] == 0:
                frequencies[0] = frequencies[1]
            frequencies = np.log(frequencies)

        return frequencies
