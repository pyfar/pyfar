import numpy as np
from scipy.special import iv as bessel_first_mod
import pyfar as pf


def fractional_delay_sinc(signal, delay, order=30, side_lobe_suppression=60,
                          mode="cut"):
    """
    Apply fractional delay to input data.

    This function uses a windowed Sinc filter (Method FIR-2 in [#]_ according
    to Equations 21 and 22) to apply fractional delays, i.e., non-integer
    delays to an input signal. A Kaiser window according to [#]_ Equations
    (10.12) and (10.13) is used, which offers the possibility to control the
    side lobe suppression.

    Parameters
    ----------
    signal : Signal
        The input data
    delay : float, array like
        The fractional delay (positive or negative). If this is a float, the
        same delay is applied to all channels of `signal`. If this is an array
        like different delays are applied to the channels of `signal`. In this
        case it must broadcast to `signal` (see `Numpy broadcasting
        <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_)
    order : int, optional
        The order of the fractional delay (sinc) filter. The precision of the
        filter increases with the order. High frequency errors decrease with
        increasing order. The order must be smaller than
        ``signal.n_samples``. The default is ``30``.
    side_lobe_suppression : float, optional
        The side lobe suppression of the Kaiser window in dB. The default is
        ``60``.
    mode : str, optional
        The filtering mode

        ``"cut"``
            The delayed signal has the same length as the input signal but
            parts of the signal that are shifted to values smaller than 0
            samples and larger than ``signal.n_samples`` are removed from the
            output
        ``"cyclic"``
            The delayed signal has the same length as the input signal. Parts
            of the signal that are shifted to values smaller than 0 are wrapped
            around the end. Parts that are shifted to values larger than
            ``signal.n_samples`` are wrapped around to the beginning.

        The default is ``"cut"``

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

    Apply a fractional delay of 2.3 samples using filters of orders 6 and 30

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
        >>>     delayed = pf.dsp.fractional_delay_sinc(signal, 2.3, order)
        >>>     pf.plot.time_freq(delayed, ax=ax[:2],
        ...                       label=f"delayed, order={order}")
        >>>     pf.plot.group_delay(delayed, ax=ax[2], unit="samples")
        >>>
        >>> ax[1].set_ylim(-15, 5)
        >>> ax[2].set_ylim(8, 14)
        >>> ax[0].legend()

    Apply a delay that exceeds the signal length using the modes ``"cut"`` and
    ``"cyclic"``

    .. plot::

        >>> import pyfar as pf
        >>>
        >>> signal = pf.signals.impulse(32, 16)
        >>>
        >>> ax = pf.plot.time(signal, label="input")
        >>>
        >>> for mode in ["cyclic", "cut"]:
        >>>     delayed = pf.dsp.fractional_delay_sinc(
        ...         signal, 25.3, order=10, mode=mode)
        >>>     pf.plot.time(delayed, label=f"delayed, mode={mode}")
        >>>
        >>> ax.legend()
    """

    # check input -------------------------------------------------------------
    if not isinstance(signal, (pf.Signal)):
        raise ValueError("Input data has to be of type pyfar.Signal")
    if side_lobe_suppression <= 0:
        raise ValueError("The order must be > 0")
    if side_lobe_suppression <= 0:
        raise ValueError("The side lobe rejection must be > 0")
    if mode not in ["cut", "cyclic"]:
        raise ValueError(f"mode is '{mode}' but must be 'cut' or 'cyclic'")
    if order + 1 > signal.n_samples:
        raise ValueError((f"order is {order} but must be smaller than "
                          f"{signal.n_samples-1} (signal.n_samples-1)"))

    # separate integer and fractional delay -----------------------------------
    delay_int = np.atleast_1d(delay).astype(int)
    delay_frac = np.atleast_1d(delay - delay_int)
    # force delay_frac >= 0 as required by Laakso et al. 1996 Eq. (2)
    mask = delay_frac < 0
    delay_int[mask] -= 1
    delay_frac[mask] += 1

    # compute the sinc functions (fractional delay filters) -------------------
    # Laakso et al. 1996 Eq. (21) applied to the fractional part of the delay
    # M_opt essentially sets the center of the sinc function in the FIR filter.
    # NOTE: This is also  the delay that is added when applying the fractional
    #       part of the delay and has thus to be accounted for when realizing
    #       delay_int
    if order % 2:
        M_opt = delay_frac.astype("int") - (order-1)/2
    else:
        M_opt = np.round(delay_frac) - order / 2

    # get matrix versions of the fractional delay and M_opt
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

    # apply fractional delay --------------------------------------------------
    # compute filter and broadcast to signal shape
    frac_delay_filter = np.broadcast_to(
        sinc * kaiser, signal.cshape + (order + 1, ))
    n_samples = signal.n_samples
    # calculate full concolution, and cut later
    convolve_mode = mode if mode == "cyclic" else "full"
    # apply filter
    delayed = pf.dsp.convolve(
        signal, pf.Signal(frac_delay_filter, signal.sampling_rate),
        mode=convolve_mode)

    # apply integer delay -----------------------------------------------------
    # account for delay from applying the fractional filter
    delay_int += M_opt.astype("int")
    # broadcast to required shape for easier looping
    delay_int = np.broadcast_to(delay_int, delayed.cshape)

    for idx in np.ndindex(delayed.cshape):
        if mode == "cyclic":
            delayed.time[idx] = np.roll(delayed.time[idx], delay_int[idx],
                                        axis=-1)
        else:
            d = delay_int[idx]

            # select correct part of time signal
            if d < 0:
                if d + n_samples > 0:
                    # discard d starting samples
                    time = delayed.time[idx, abs(d):].flatten()
                else:
                    # we are left with a zero vector (strictly spoken we might
                    # have some tail left from 'full' convolution but zeros
                    # seem the more reasonable choice here)
                    time = np.zeros(n_samples)
            elif d > 0:
                # add d zeros
                time = np.concatenate((np.zeros(d), delayed.time[idx]))

            # adjust length to n_samples
            if time.size >= n_samples:
                # discard samples at end
                time = time[:n_samples]
            else:
                time = np.concatenate(
                    (time, np.zeros(n_samples - time.size)))

            delayed.time[idx, :n_samples] = time

    # truncate signal
    if mode == "cut":
        delayed.time = delayed.time[..., :n_samples]

    return delayed
