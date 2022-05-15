import numpy as np
from scipy.special import iv as bessel_first_mod
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pyfar as pf


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
        >>>     delayed = pf.dsp.fractional_time_shift(signal, 2.3, order)
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
    n_samples_full = signal.n_samples

    # apply integer shift -----------------------------------------------------
    # account for shift from applying the fractional filter
    delay_int += M_opt.astype("int")
    # broadcast to required shape for easier looping
    delay_int = np.broadcast_to(delay_int, signal.cshape)

    for idx in np.ndindex(signal.cshape):
        if mode == "cyclic":
            signal.time[idx] = np.roll(signal.time[idx], delay_int[idx],
                                       axis=-1)
        else:
            d = delay_int[idx]

            # select correct part of time signal
            if d < 0:
                if d + n_samples > 0:
                    # discard d starting samples
                    time = signal.time[
                        idx + (slice(abs(d), n_samples_full), )].flatten()
                else:
                    # we are left with a zero vector (strictly spoken we might
                    # have some tail left from 'full' convolution but zeros
                    # seem the more reasonable choice here)
                    time = np.zeros(n_samples)
            elif d > 0:
                # add d zeros
                time = np.concatenate((np.zeros(d), signal.time[idx]))

            # adjust length to n_samples
            if time.size >= n_samples:
                # discard samples at end
                time = time[:n_samples]
            else:
                time = np.concatenate(
                    (time, np.zeros(n_samples - time.size)))

            signal.time[idx + (slice(0, n_samples), )] = time

    # truncate signal
    if mode == "linear":
        signal.time = signal.time[..., :n_samples]

    return signal


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
