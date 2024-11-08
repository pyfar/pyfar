"""Audio filter design and application."""
import warnings
from pyfar.classes.warnings import PyfarDeprecationWarning
import numpy as np
import pyfar as pf
from . import _audiofilter as iir


def allpass(signal, frequency, order, coefficients=None, sampling_rate=None):
    r"""
    Create and apply first or second order allpass filter.

    Allpass filters have an almost constant group delay below their cut-off
    frequency and are often used in analogue loudspeaker design.
    The filter transfer function is based on Tietze et al. [#]_:

    .. math:: A(s) = \frac{1-\frac{a_i}{\omega_c} s+\frac{b_i}
                {\omega_c^2} s^2}{1+\frac{a_i}{\omega_c} s
                +\frac{b_i}{\omega_c^2} s^2},


    where :math:`\omega_c = 2 \pi f_c` with the cut-off frequency :math:`f_c`
    and :math:`s=\mathrm{i} \omega`.

    By definition the ``bi`` coefficient of a first order allpass is ``0``.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Cutoff frequency of the allpass in Hz.
    order : number
        Order of the allpass filter. Must be ``1`` or ``2``.
    coefficients:  number, list, optional
        Filter characteristic coefficients ``bi`` and ``ai``.

        -   For 1st order allpass provide ai-coefficient as single value.\n
            The default is ``ai = 0.6436``.

        -   For 2nd order allpass provide coefficients as list ``[bi, ai]``.\n
            The default is ``bi = 0.8832``, ``ai = 1.6278``.

        Defaults are chosen according to Tietze et al. (Fig. 12.66)
        for maximum flat group delay.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterIIR
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [#] Tietze, U., Schenk, C. & Gamm, E. (2019). Halbleiter-
        Schaltungstechnik (16th ed.). Springer Vieweg
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_design/audiofilter.py

    Examples
    --------
    First and second order allpass filter with ``fc = 1000`` Hz.

    .. plot::

        import pyfar as pf
        import matplotlib.pyplot as plt

        # impulse to be filtered
        impulse = pf.signals.impulse(256)

        orders = [1, 2]
        labels = ['First order', 'Second order']

        fig, (ax1, ax2) = plt.subplots(2,1, layout='constrained')

        for (order, label) in zip(orders, labels):
            # create and apply allpass filter
            sig_filt = pf.dsp.filter.allpass(impulse, 1000, order)
            pf.plot.group_delay(sig_filt, unit='samples', label=label, ax=ax1)
            pf.plot.phase(sig_filt, label=label, ax=ax2, unwrap = True)

        ax1.set_title('1. and 2. order allpass filter with fc = 1000 Hz')
        ax2.legend()
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # check if coefficients match filter order
    if coefficients is not None and (
            (order == 1 and np.isscalar(coefficients) is False) or
            (order == 2 and (
                not isinstance(coefficients, (list, np.ndarray)) or
                len(coefficients) != 2))):
        print(type(coefficients), order)
        raise ValueError('Coefficients must match the allpass order')

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    if order == 1:
        if coefficients is None:
            coefficients = 0.6436
        # get filter coefficients for first order allpass
        b, a = iir.biquad_ap1st(frequency, fs, ai=coefficients)[2:]
    elif order == 2:
        if coefficients is None:
            coefficients = [0.8832, 1.6278]
        # get filter coefficients for second order allpass
        b, a = iir.biquad_ap2nd(
            frequency, fs, bi=coefficients[0], ai=coefficients[1])[2:]
    else:
        raise ValueError('Order must be 1 or 2')

    filter_coeffs = np.stack((b, a), axis=0)
    filt = pf.FilterIIR(filter_coeffs, fs)
    filt.comment = (f"Allpass of order {order} with cutoff frequency "
                    f"{frequency} Hz.")

    if signal is None:
        # return the filter-object
        return filt
    else:
        # return filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def bell(signal, center_frequency, gain, quality, bell_type='II',
         quality_warp='cos', sampling_rate=None):
    """
    Create and apply second order bell (parametric equalizer) filter.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    center_frequency : number
        Center frequency of the parametric equalizer in Hz
    gain : number
        Gain of the parametric equalizer in dB
    quality : number
        Quality of the parametric equalizer, i.e., the inverse of the
        bandwidth
    bell_type : str
        Defines the bandwidth/quality. The default is ``'II'``.

        ``'I'``
            not recommended. Also known as 'constant Q'.
        ``'II'``
            defines the bandwidth by the points 3 dB below the maximum if the
            gain is positive and 3 dB above the minimum if the gain is
            negative. Also known as 'symmetric'.
        ``'III'``
            defines the bandwidth by the points at gain/2. Also known as
            'half pad loss'.
    quality_warp : str
        Sets the pre-warping for the quality (``'cos'``, ``'sin'``, or
        ``'tan'``). The default is ``'cos'``.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterIIR
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_design/audiofilter.py
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    if bell_type not in ['I', 'II', 'III']:
        raise ValueError(("bell_type must be 'I', 'II' or "
                          f"'III' but is '{bell_type}'.'"))

    if quality_warp not in ['cos', 'sin', 'tan']:
        raise ValueError(("quality_warp must be 'cos', 'sin' or "
                          f"'tan' but is '{quality_warp}'.'"))

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((2, 3))
    _, _, b, a = iir.biquad_peq2nd(
        center_frequency, gain, quality, fs, bell_type, quality_warp)
    ba[0] = b
    ba[1] = a

    # generate filter object
    filt = pf.FilterIIR(ba, fs)
    filt.comment = ("Second order bell (parametric equalizer) "
                    f"of type {bell_type} with {gain} dB gain at "
                    f"{center_frequency} Hz (Quality = {quality}).")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def high_shelve(signal, frequency, gain, order, shelve_type='I',
                sampling_rate=None):
    """
    :py:func:`~pyfar.dsp.filter.high_shelve` will be deprecated in
    pyfar 0.9.0 in favor of :py:func:`~pyfar.dsp.filter.high_shelf`.
    Create and/or apply first or second order high shelf filter.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelf in Hz.
    gain : number
        Gain of the shelf in dB.
    order : number
        The shelf order. Must be ``1`` or ``2``.
    shelve_type : str
        Defines the characteristic frequency. The default is ``'I'``.

        ``'I'``
            Defines the characteristic frequency 3 dB below the `gain` value if
            the `gain` is positive and 3 dB above the `gain` value if the
            `gain` is negative.
        ``'II'``
            Defines the characteristic frequency at 3 dB if the `gain` is
            positive and at -3 dB if the `gain` is negative.
        ``'III'``
            Defines the characteristic frequency at `gain`/2 dB.

        For types ``I`` and ``II`` the absolute value of the `gain` must be
        sufficiently large (> 9 dB) to set the characteristic
        frequency according to the above rules with an error below 0.5 dB.
        For smaller absolute `gain` values the gain at the characteristic
        frequency becomes less accurate.
        For type ``III`` the characteristic frequency is always set correctly.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterIIR
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_design/audiofilter.py
    """

    warnings.warn(("'high_shelve' will be deprecated in pyfar 0.9.0 in favor"
                   " of 'high_shelf'"), PyfarDeprecationWarning, stacklevel=2)

    return high_shelf(signal, frequency, gain, order, shelve_type,
                      sampling_rate)


def high_shelf(signal, frequency, gain, order, shelf_type='I',
               sampling_rate=None):
    """
    Create and/or apply first or second order high shelf filter.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelf in Hz.
    gain : number
        Gain of the shelf in dB.
    order : number
        The shelf order. Must be ``1`` or ``2``.
    shelf_type : str
        Defines the characteristic frequency. The default is ``'I'``.

        ``'I'``
            Defines the characteristic frequency 3 dB below the `gain` value if
            the `gain` is positive and 3 dB above the `gain` value if the
            `gain` is negative.
        ``'II'``
            Defines the characteristic frequency at 3 dB if the `gain` is
            positive and at -3 dB if the `gain` is negative.
        ``'III'``
            Defines the characteristic frequency at `gain`/2 dB.

        For types ``I`` and ``II`` the absolute value of the `gain` must be
        sufficiently large (> 9 dB) to set the characteristic
        frequency according to the above rules with an error below 0.5 dB.
        For smaller absolute `gain` values the gain at the characteristic
        frequency becomes less accurate.
        For type ``III`` the characteristic frequency is always set correctly.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterIIR
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_design/audiofilter.py
    """

    output = _shelf(
        signal, frequency, gain, order, shelf_type, sampling_rate, 'high')

    return output


def low_shelve(signal, frequency, gain, order, shelve_type='I',
               sampling_rate=None):
    """
    :py:func:`~pyfar.dsp.filter.low_shelve` will be deprecated in
    pyfar 0.9.0 in favor of :py:func:`~pyfar.dsp.filter.low_shelf`.
    Create and apply first or second order low shelf filter.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelf in Hz.
    gain : number
        Gain of the shelf in dB.
    order : number
        The shelf order. Must be ``1`` or ``2``.
    shelve_type : str
        Defines the characteristic frequency. The default is ``'I'``.

        ``'I'``
            Defines the characteristic frequency 3 dB below the `gain` value if
            the `gain` is positive and 3 dB above the `gain` value if the
            `gain` is negative.
        ``'II'``
            Defines the characteristic frequency at 3 dB if the `gain` is
            positive and at -3 dB if the `gain` is negative.
        ``'III'``
            Defines the characteristic frequency at `gain`/2 dB.

        For types ``I`` and ``II`` the absolute value of the `gain` must be
        sufficiently large (> 9 dB) to set the characteristic
        frequency according to the above rules with an error below 0.5 dB.
        For smaller absolute `gain` values the gain at the characteristic
        frequency becomes less accurate.
        For type ``III`` the characteristic frequency is always set correctly.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterIIR
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_design/audiofilter.py
    """

    warnings.warn(("'low_shelve' will be deprecated in pyfar 0.9.0 in favor "
                   "of 'low_shelf'"), PyfarDeprecationWarning, stacklevel=2)

    return low_shelf(signal, frequency, gain, order, shelve_type,
                     sampling_rate)


def low_shelf(signal, frequency, gain, order, shelf_type='I',
              sampling_rate=None):
    """
    Create and apply first or second order low shelf filter.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelf in Hz.
    gain : number
        Gain of the shelf in dB.
    order : number
        The shelf order. Must be ``1`` or ``2``.
    shelf_type : str
        Defines the characteristic frequency. The default is ``'I'``.

        ``'I'``
            Defines the characteristic frequency 3 dB below the `gain` value if
            the `gain` is positive and 3 dB above the `gain` value if the
            `gain` is negative.
        ``'II'``
            Defines the characteristic frequency at 3 dB if the `gain` is
            positive and at -3 dB if the `gain` is negative.
        ``'III'``
            Defines the characteristic frequency at `gain`/2 dB.

        For types ``I`` and ``II`` the absolute value of the `gain` must be
        sufficiently large (> 9 dB) to set the characteristic
        frequency according to the above rules with an error below 0.5 dB.
        For smaller absolute `gain` values the gain at the characteristic
        frequency becomes less accurate.
        For type ``III`` the characteristic frequency is always set correctly.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterIIR
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/blob/master/filter_design/audiofilter.py
    """

    output = _shelf(
        signal, frequency, gain, order, shelf_type, sampling_rate, 'low')

    return output


def high_shelve_cascade(
        signal, frequency, frequency_type="lower", gain=None, slope=None,
        bandwidth=None, N=None, sampling_rate=None):
    """
    :py:func:`~pyfar.dsp.filter.high_shelve_cascade` will be deprecated in
    pyfar 0.9.0 in favor of :py:func:`~pyfar.dsp.filter.high_shelf_cascade`.
    Create and apply constant slope filter from cascaded 2nd order high
    shelves.

    The filters - also known as High-Schultz filters (cf. [#]_) - are defined
    by their characteristic frequency, gain, slope, and bandwidth. Two out of
    the three parameter `gain`, `slope`, and `bandwidth` must be specified,
    while the third parameter is calculated as

    ``gain = bandwidth * slope``

    ``bandwidth = abs(gain/slope)``

    ``slope = gain/bandwidth``

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency in Hz (see `frequency_type`)
    frequency_type : string
        Defines how `frequency` is used

        ``'upper'``
            `frequency` gives the upper characteristic frequency. In this case
            the lower characteristic frequency is given by
            ``2**bandwidth / frequency``
        ``'lower'``
            `frequency` gives the lower characteristic frequency. In this case
            the upper characteristic frequency is given by
            ``2**bandwidth * frequency``
    gain : number
        The filter gain in dB. The default is ``None``, which calculates the
        gain from the `slope` and `bandwidth` (must be given if `gain` is
        ``None``).
    slope : number
        Filter slope in dB per octave, with positive values denoting a rising
        filter slope and negative values denoting a falling filter slope. The
        default is ``None``, which calculates the slope from the `gain` and
        `bandwidth` (must be given if `slope` is ``None``).
    bandwidth : number
        The bandwidth of the filter in octaves. The default is ``None``, which
        calculates the bandwidth from `gain` and `slope` (must be given if
        `bandwidth` is ``None``).
    N : int
        Number of shelf filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : :py:class:`~pyfar.Signal`, :py:class:`~pyfar.FilterSOS`
        The filtered signal (returned if ``sampling_rate = None``) or the
        Filter object (returned if ``signal = None``).
    N : int
        The number of shelf filters that were cascaded
    ideal : :py:class:`~pyfar.FrequencyData`
        The ideal, piece-wise magnitude response of the filter

    References
    ----------
    .. [#] F. Schultz, N. Hahn, and S. Spors, “Shelving Filter Cascade with
           Adjustable Transition Slope and Bandwidth,” in 148th AES Convention
           (Vienna, Austria, 2020).

    Examples
    --------
    Generate a filter with a bandwith of 4 octaves and a gain of -60 dB and
    compare it to the piece-wise constant idealized magnitude response.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> impulse = pf.signals.impulse(40e3, sampling_rate=40000)
        >>> impulse, N, ideal = pf.dsp.filter.high_shelf_cascade(
        >>>     impulse, 250, "lower", -60, None, 4)
        >>>
        >>> pf.plot.freq(ideal, c='k', ls='--', label="ideal")
        >>> pf.plot.freq(impulse, label="actual")
        >>> plt.legend()
    """

    warnings.warn(("'high_shelve_cascade' will be deprecated in pyfar 0.9.0 "
                   "in favor of 'high_shelf_cascade'"),
                  PyfarDeprecationWarning, stacklevel=2)

    return high_shelf_cascade(signal, frequency, frequency_type, gain, slope,
                              bandwidth, N, sampling_rate)


def high_shelf_cascade(
        signal, frequency, frequency_type="lower", gain=None, slope=None,
        bandwidth=None, N=None, sampling_rate=None):
    """
    Create and apply constant slope filter from cascaded 2nd order high
    shelves.

    The filters - also known as High-Schultz filters (cf. [#]_) - are defined
    by their characteristic frequency, gain, slope, and bandwidth. Two out of
    the three parameter `gain`, `slope`, and `bandwidth` must be specified,
    while the third parameter is calculated as

    ``gain = bandwidth * slope``

    ``bandwidth = abs(gain/slope)``

    ``slope = gain/bandwidth``

    .. note::

        The `bandwidth` must be at least 1 octave to obtain a good
        approximation of the desired frequency response. Make sure to specify
        the parameters `gain`, `slope`, and `bandwidth` accordingly.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency in Hz (see `frequency_type`)
    frequency_type : string
        Defines how `frequency` is used

        ``'upper'``
            `frequency` gives the upper characteristic frequency. In this case
            the lower characteristic frequency is given by
            ``2**bandwidth / frequency``
        ``'lower'``
            `frequency` gives the lower characteristic frequency. In this case
            the upper characteristic frequency is given by
            ``2**bandwidth * frequency``
    gain : number
        The filter gain in dB. The default is ``None``, which calculates the
        gain from the `slope` and `bandwidth` (must be given if `gain` is
        ``None``).
    slope : number
        Filter slope in dB per octave, with positive values denoting a rising
        filter slope and negative values denoting a falling filter slope. The
        default is ``None``, which calculates the slope from the `gain` and
        `bandwidth` (must be given if `slope` is ``None``).
    bandwidth : number
        The bandwidth of the filter in octaves. The default is ``None``, which
        calculates the bandwidth from `gain` and `slope` (must be given if
        `bandwidth` is ``None``).
    N : int
        Number of shelf filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : :py:class:`~pyfar.Signal`, :py:class:`~pyfar.FilterSOS`
        The filtered signal (returned if ``sampling_rate = None``) or the
        Filter object (returned if ``signal = None``).
    N : int
        The number of shelf filters that were cascaded
    ideal : :py:class:`~pyfar.FrequencyData`
        The ideal, piece-wise magnitude response of the filter

    References
    ----------
    .. [#] F. Schultz, N. Hahn, and S. Spors, “Shelving Filter Cascade with
           Adjustable Transition Slope and Bandwidth,” in 148th AES Convention
           (Vienna, Austria, 2020).

    Examples
    --------
    Generate a filter with a bandwith of 4 octaves and a gain of -60 dB and
    compare it to the piece-wise constant idealized magnitude response.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> impulse = pf.signals.impulse(40e3, sampling_rate=40000)
        >>> impulse, N, ideal = pf.dsp.filter.high_shelf_cascade(
        >>>     impulse, 250, "lower", -60, None, 4)
        >>>
        >>> pf.plot.freq(ideal, c='k', ls='--', label="ideal")
        >>> pf.plot.freq(impulse, label="actual")
        >>> plt.legend()
    """
    signal, N, ideal_response = _shelf_cascade(
        signal, frequency, frequency_type, gain, slope, bandwidth, N,
        sampling_rate, shelf_type="high")

    return signal, N, ideal_response


def low_shelve_cascade(
        signal, frequency, frequency_type="upper", gain=None, slope=None,
        bandwidth=None, N=None, sampling_rate=None):
    """
    :py:func:`~pyfar.dsp.filter.low_shelve_cascade` will be deprecated in
    pyfar 0.9.0 in favor of :py:func:`~pyfar.dsp.filter.low_shelf_cascade`.
    Create and apply constant slope filter from cascaded 2nd order low shelves.

    The filters - also known as Low-Schultz filters (cf. [#]_) - are defined
    by their characteristic frequency, gain, slope, and bandwidth. Two out of
    the three parameter `gain`, `slope`, and `bandwidth` must be specified,
    while the third parameter is calculated as

    ``gain = -bandwidth * slope``

    ``bandwidth = abs(gain/slope)``

    ``slope = -gain/bandwidth``



    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency in Hz (see `frequency_type`)
    frequency_type : string
        Defines how `frequency` is used

        ``'upper'``
            `frequency` gives the upper characteristic frequency. In this case
            the lower characteristic frequency is given by
            ``2**bandwidth / frequency``
        ``'lower'``
            `frequency` gives the lower characteristic frequency. In this case
            the upper characteristic frequency is given by
            ``2**bandwidth * frequency``
    gain : number
        The filter gain in dB. The default is ``None``, which calculates the
        gain from the `slope` and `bandwidth` (must be given if `gain` is
        ``None``).
    slope : number
        Filter slope in dB per octave, with positive values denoting a rising
        filter slope and negative values denoting a falling filter slope. The
        default is ``None``, which calculates the slope from the `gain` and
        `bandwidth` (must be given if `slope` is ``None``).
    bandwidth : number
        The bandwidth of the filter in octaves. The default is ``None``, which
        calculates the bandwidth from `gain` and `slope` (must be given if
        `bandwidth` is ``None``).
    N : int
        Number of shelf filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : :py:class:`~pyfar.Signal`, :py:class:`~pyfar.FilterSOS`
        The filtered signal (returned if ``sampling_rate = None``) or the
        Filter object (returned if ``signal = None``).
    N : int
        The number of shelf filters that were cascaded
    ideal : :py:class:`~pyfar.FrequencyData`
        The ideal, piece-wise magnitude response of the filter

    References
    ----------
    .. [#] F. Schultz, N. Hahn, and S. Spors, “Shelving Filter Cascade with
           Adjustable Transition Slope and Bandwidth,” in 148th AES Convention
           (Vienna, Austria, 2020).

    Examples
    --------
    Generate a filter with a bandwith of 4 octaves and a gain of -60 dB and
    compare it to the piece-wise constant idealized magnitude response.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> impulse = pf.signals.impulse(40e3, sampling_rate=40000)
        >>> impulse, N, ideal = pf.dsp.filter.low_shelf_cascade(
        >>>     impulse, 4000, "upper", -60, None, 4)
        >>>
        >>> pf.plot.freq(ideal, c='k', ls='--', label="ideal")
        >>> pf.plot.freq(impulse, label="actual")
        >>> plt.legend()
    """

    warnings.warn(("'low_shelve_cascade' will be deprecated in pyfar 0.9.0 "
                   "in favor of 'low_shelf_cascade'"),
                   PyfarDeprecationWarning, stacklevel=2)

    return low_shelf_cascade(signal, frequency, frequency_type, gain, slope,
                             bandwidth, N, sampling_rate)


def low_shelf_cascade(
        signal, frequency, frequency_type="upper", gain=None, slope=None,
        bandwidth=None, N=None, sampling_rate=None):
    """
    Create and apply constant slope filter from cascaded 2nd order low shelves.

    The filters - also known as Low-Schultz filters (cf. [#]_) - are defined
    by their characteristic frequency, gain, slope, and bandwidth. Two out of
    the three parameter `gain`, `slope`, and `bandwidth` must be specified,
    while the third parameter is calculated as

    ``gain = -bandwidth * slope``

    ``bandwidth = abs(gain/slope)``

    ``slope = -gain/bandwidth``

    .. note::

        The `bandwidth` must be at least 1 octave to obtain a good
        approximation of the desired frequency response. Make sure to specify
        the parameters `gain`, `slope`, and `bandwidth` accordingly.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency in Hz (see `frequency_type`)
    frequency_type : string
        Defines how `frequency` is used

        ``'upper'``
            `frequency` gives the upper characteristic frequency. In this case
            the lower characteristic frequency is given by
            ``2**bandwidth / frequency``
        ``'lower'``
            `frequency` gives the lower characteristic frequency. In this case
            the upper characteristic frequency is given by
            ``2**bandwidth * frequency``
    gain : number
        The filter gain in dB. The default is ``None``, which calculates the
        gain from the `slope` and `bandwidth` (must be given if `gain` is
        ``None``).
    slope : number
        Filter slope in dB per octave, with positive values denoting a rising
        filter slope and negative values denoting a falling filter slope. The
        default is ``None``, which calculates the slope from the `gain` and
        `bandwidth` (must be given if `slope` is ``None``).
    bandwidth : number
        The bandwidth of the filter in octaves. The default is ``None``, which
        calculates the bandwidth from `gain` and `slope` (must be given if
        `bandwidth` is ``None``).
    N : int
        Number of shelf filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : :py:class:`~pyfar.Signal`, :py:class:`~pyfar.FilterSOS`
        The filtered signal (returned if ``sampling_rate = None``) or the
        Filter object (returned if ``signal = None``).
    N : int
        The number of shelf filters that were cascaded
    ideal : :py:class:`~pyfar.FrequencyData`
        The ideal, piece-wise magnitude response of the filter

    References
    ----------
    .. [#] F. Schultz, N. Hahn, and S. Spors, “Shelving Filter Cascade with
           Adjustable Transition Slope and Bandwidth,” in 148th AES Convention
           (Vienna, Austria, 2020).

    Examples
    --------
    Generate a filter with a bandwith of 4 octaves and a gain of -60 dB and
    compare it to the piece-wise constant idealized magnitude response.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> impulse = pf.signals.impulse(40e3, sampling_rate=40000)
        >>> impulse, N, ideal = pf.dsp.filter.low_shelf_cascade(
        >>>     impulse, 4000, "upper", -60, None, 4)
        >>>
        >>> pf.plot.freq(ideal, c='k', ls='--', label="ideal")
        >>> pf.plot.freq(impulse, label="actual")
        >>> plt.legend()
    """
    signal, N, ideal_response = _shelf_cascade(
        signal, frequency, frequency_type, gain, slope, bandwidth, N,
        sampling_rate, shelf_type="low")

    return signal, N, ideal_response


def _shelf(signal, frequency, gain, order, shelf_type, sampling_rate, kind):
    """
    First and second order high and low shelves.

    For the documentation refer to high_shelf and low_shelf. The only
    additional parameter is `kind`, which has to be 'high' or 'low'.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    if shelf_type not in ['I', 'II', 'III']:
        raise ValueError(("shelf_type must be 'I', 'II' or "
                          f"'III' but is '{shelf_type}'.'"))

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((2, 3))

    if order == 1 and kind == 'high':
        shelf = iir.biquad_hshv1st
    elif order == 2 and kind == 'high':
        shelf = iir.biquad_hshv2nd
    elif order == 1 and kind == 'low':
        shelf = iir.biquad_lshv1st
    elif order == 2 and kind == 'low':
        shelf = iir.biquad_lshv2nd
    else:
        raise ValueError(f"order must be 1 or 2 but is {order}")

    _, _, b, a = shelf(frequency, gain, fs, shelf_type)
    ba[0] = b
    ba[1] = a

    # generate filter object
    filt = pf.FilterIIR(ba, fs)
    kind = "High" if kind == "high" else "Low"
    filt.comment = (f"{kind}-shelf of order {order} and type "
                    f"{shelf_type} with {gain} dB gain at {frequency} Hz.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def _shelf_cascade(signal, frequency, frequency_type, gain, slope, bandwidth,
                   N, sampling_rate, shelf_type):
    """Design constant slope filter from shelf filter cascade.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency in Hz (see `frequency_type`)
    frequency_type : string
        Defines how `frequency` is used

        ``'upper'``
            `frequency` gives the upper characteristic frequency. In this case
            the lower characteristic frequency is given by
            ``2**bandwidth / frequency``
        ``'lower'``
            `frequency` gives the lower characteristic frequency. In this case
            the upper characteristic frequency is given by
            ``2**bandwidth * frequency``
    gain : number
        The filter gain in dB. The default is ``None``, which calculates the
        gain from the `slope` and `bandwidth` (must be given if `gain` is
        ``None``).
    slope : number
        Filter slope in dB per octave, with positive values denoting a rising
        filter slope and negative values denoting a falling filter slope. The
        default is ``None``, which calculates the slope from the `gain` and
        `bandwidth` (must be given if `slope` is ``None``).
    bandwidth : number
        The bandwidth of the filter in octaves. The default is ``None``, which
        calculates the bandwidth from `gain` and `slope` (must be given if
        `bandwidth` is ``None``).
    N : int
        Number of shelf filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.
    shelf_type : string
        ``'low'`` or ``'high'`` for low- or high-shelf.

    [1] F. Schultz, N. Hahn, and S. Spors, “Shelving Filter Cascade with
        Adjustable Transition Slope and Bandwidth,” in 148th AES Convention
        (Vienna, Austria, 2020).
    """

    # check input -------------------------------------------------------------
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')
    if not isinstance(signal, (pf.Signal, type(None))):
        raise ValueError("signal must be a pyfar Signal object or None")

    # check and set filter slope parameters according to Eq. (4)
    gain, slope, bandwidth = _shelving_cascade_slope_parameters(
        gain, slope, bandwidth, shelf_type)
    if bandwidth < 1:
        warnings.warn((
            f"The bandwidth is {bandwidth} octaves but should be at least 1 "
            "to obtain a good approximation of the desired frequency response."
        ), stacklevel=2)

    # get sampling rate
    sampling_rate = sampling_rate if signal is None else signal.sampling_rate

    # get upper and lower cut-off frequency
    if frequency_type == "upper":
        frequency = [frequency / 2**bandwidth, frequency]
    elif frequency_type == "lower":
        frequency = [frequency, 2**bandwidth * frequency]
    else:
        raise ValueError((f"frequency_type is '{frequency_type}' but must be "
                          "'lower' or 'upper'"))

    # check characteristic frequencies
    if frequency[0] == 0:
        raise ValueError("The lower characteristic frequency must not be 0 Hz")
    if frequency[0] > sampling_rate/2:
        raise ValueError(("The lower characteristic frequency must be smaller "
                          "than half the sampling rate"))
    if frequency[1] > sampling_rate/2 and shelf_type == "low":
        raise ValueError(("The upper characteristic frequency must be smaller "
                          "than half the sampling rate"))
    if frequency[1] > sampling_rate/2:
        frequency[1] = sampling_rate/2
        gain *= np.log2(frequency[1]/frequency[0]) / bandwidth
        bandwidth = np.log2(frequency[1]/frequency[0])
        warnings.warn((f"The upper frequency exceeded the Nyquist frequency "
                       f"It was set to {sampling_rate/2} Hz, which equals "
                       f"a restriction of the bandwidth to {bandwidth} "
                       f"octaves and a reduction of the gain to {gain} dB to "
                       f"maintain the intended slope of {slope} dB/octave."),
                       stacklevel=2)

    # determine number of shelf filters per octave ---------------------------

    # recommended minimum shelf filters per octave according to Eq. (11.2)
    N_octave_min = 1 if abs(slope) < 12.04 else abs(slope) / 12.04
    # minimum total shelf filters according to Eq. (9)
    N_min = np.ceil(N_octave_min*bandwidth).astype(int)

    # actual total shelf filters either from user input or recommended minimum
    N = int(N) if N else N_min

    if N < N_min:
        warnings.warn((
            f"N is {N} but should be at least {N_min} to obtain an good "
            "approximation of the desired frequency response"), stacklevel=2)

    # used shelf filters per octave
    N_octave = N / bandwidth

    # get the filter ----------------------------------------------------------

    # initialize variables
    filter_func = high_shelf if shelf_type == "high" else low_shelf
    shelf_gain = gain / N
    SOS = np.zeros((1, N, 6))

    # get the filter coefficients
    for n in range(N):
        # current frequency according to Eq. (5)
        f = 2**(-(n+.5)/N_octave) * frequency[1]
        # get shelf and cascade coefficients
        shelf = filter_func(None, f, shelf_gain, 2, 'III', sampling_rate)
        SOS[:, n] = shelf.coefficients.flatten()

    # make filter object
    comment = (f"Constant slope filter cascaded from {N} {shelf_type}-shelf "
               f"filters ({frequency_type} frequency: {frequency} Hz, "
               f"bandwidth: {bandwidth} octaves, gain: {gain} dB, {N_octave} "
               "shelf filters per octave")
    filt = pf.FilterSOS(SOS, sampling_rate, comment=comment)

    # get the ideal filter response -------------------------------------------
    magnitudes = np.array([10**(gain/20), 10**(gain/20), 1, 1])
    if shelf_type == "high":
        magnitudes = np.flip(magnitudes)
    frequencies = [0, frequency[0], frequency[1], sampling_rate/2]

    # remove duplicate entries (happens if the slope ends at Nyquist)
    if frequencies[-2] == frequencies[-1]:
        magnitudes = magnitudes[:-1]
        frequencies = frequencies[:-1]

    ideal_response = pf.FrequencyData(
        magnitudes, frequencies,
        "ideal magnitude response of cascaded shelf filter")

    # return parameter --------------------------------------------------------
    if signal is None:
        return filt, N, ideal_response
    else:
        return filt.process(signal), N, ideal_response


def _shelving_cascade_slope_parameters(gain, slope, bandwidth, shelf_type):
    """Compute the third parameter from the given two.

    Parameters
    ----------
    gain : float
        Desired gain of the stop band in decibel.
    slope : float
        Desired shelving slope in decibel per octave.
    bandwidth : float
        Desired bandwidth of the slope in octave.
    shelf_type : string
        ``'low'`` or ``'high'`` for low- or high-shelf.

    """
    if slope == 0:
        raise ValueError("slope must be non-zero.")

    if gain is None and slope is not None and bandwidth is not None:
        bandwidth = abs(bandwidth)
        gain = -bandwidth * slope if shelf_type == "low" \
            else bandwidth * slope
    elif slope is None and gain is not None and bandwidth is not None:
        bandwidth = abs(bandwidth)
        slope = -gain / bandwidth if shelf_type == "low" \
            else gain / bandwidth
    elif bandwidth is None and gain is not None and slope is not None:
        if shelf_type == "low" and np.sign(gain * slope) == 1:
            raise ValueError("gain and slope must have different signs")
        if shelf_type == "high" and np.sign(gain * slope) == -1:
            raise ValueError("gain and slope must have the same signs")
        bandwidth = abs(gain / slope)
    else:
        raise ValueError(("Exactly two out of the parameters gain, slope, and "
                          "bandwidth must be given."))

    return gain, slope, bandwidth
