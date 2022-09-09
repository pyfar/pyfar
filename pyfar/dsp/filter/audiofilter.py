import warnings
import numpy as np
import pyfar as pf
from . import _audiofilter as iir


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
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/\
blob/master/filter_design/audiofilter.py
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
    Create and/or apply first or second order high shelve filter.

    Uses the implementation of [#]_.


    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelve in Hz.
    gain : number
        Gain of the shelve in dB.
    order : number
        The shelve order. Must be ``1`` or ``2``.
    shelve_type : str
        Defines the characteristic frequency. The default is ``'I'``

        ``'I'``
            defines the characteristic frequency 3 dB below the gain value if
            the gain is positive and 3 dB above the gain value otherwise
        ``'II'``
            defines the characteristic frequency at 3 dB if the gain is
            positive and at -3 dB if the gain is negative.
        ``'III'``
            defines the characteristic frequency at gain/2 dB.
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
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/\
blob/master/filter_design/audiofilter.py
    """

    output = _shelve(
        signal, frequency, gain, order, shelve_type, sampling_rate, 'high')

    return output


def low_shelve(signal, frequency, gain, order, shelve_type='I',
               sampling_rate=None):
    """
    Create and apply first or second order low shelve filter.

    Uses the implementation of [#]_.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelve in Hz.
    gain : number
        Gain of the shelve in dB.
    order : number
        The shelve order. Must be ``1`` or ``2``.
    shelve_type : str
        Defines the characteristic frequency. The default is ``'I'``

        ``'I'``
            defines the characteristic frequency 3 dB below the gain value if
            the gain is positive and 3 dB above the gain value otherwise.
        ``'II'``
            defines the characteristic frequency at 3 dB if the gain is
            positive and at -3 dB if the gain is negative.
        ``'III'``
            defines the characteristic frequency at gain/2 dB.
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
    .. [#] https://github.com/spatialaudio/digital-signal-processing-lecture/\
blob/master/filter_design/audiofilter.py
    """

    output = _shelve(
        signal, frequency, gain, order, shelve_type, sampling_rate, 'low')

    return output


def high_shelve_cascade(
        signal, frequency, frequency_type="lower", gain=None, slope=None,
        bandwidth=None, N=None, sampling_rate=None):
    """
    Create and apply constant slope filter from cascaded 2nd order high shelves.

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
        Number of shelve filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : :py:class:`~pyfar.classes.audio.Signal`, :py:class:`~pyfar.classes.filter.FilterSOS`
        The filtered signal (returned if ``sampling_rate = None``) or the
        Filter object (returned if ``signal = None``).
    N : int
        The number of shelve filters that were cascaded
    ideal : :py:class:`~pyfar.classes.audio.FrequencyData`
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
        >>> impulse, N, ideal = pf.dsp.filter.high_shelve_cascade(
        >>>     impulse, 250, "lower", -60, None, 4)
        >>>
        >>> pf.plot.freq(ideal, c='k', ls='--', label="ideal")
        >>> pf.plot.freq(impulse, label="actual")
        >>> plt.legend()
    """  # noqa E501
    signal, N, ideal_response = _shelve_cascade(
        signal, frequency, frequency_type, gain, slope, bandwidth, N,
        sampling_rate, shelve_type="high")

    return signal, N, ideal_response


def low_shelve_cascade(
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
        Number of shelve filters that are cascaded. The default is ``None``,
        which calculated the minimum ``N`` that is required to satisfy Eq. (11)
        in Schultz et al. 2020, i.e., the minimum ``N`` that is required for
        a good approximation of the ideal filter response.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : :py:class:`~pyfar.classes.audio.Signal`, :py:class:`~pyfar.classes.filter.FilterSOS`
        The filtered signal (returned if ``sampling_rate = None``) or the
        Filter object (returned if ``signal = None``).
    N : int
        The number of shelve filters that were cascaded
    ideal : :py:class:`~pyfar.classes.audio.FrequencyData`
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
        >>> impulse, N, ideal = pf.dsp.filter.low_shelve_cascade(
        >>>     impulse, 4000, "upper", -60, None, 4)
        >>>
        >>> pf.plot.freq(ideal, c='k', ls='--', label="ideal")
        >>> pf.plot.freq(impulse, label="actual")
        >>> plt.legend()
    """  # noqa E501
    signal, N, ideal_response = _shelve_cascade(
        signal, frequency, frequency_type, gain, slope, bandwidth, N,
        sampling_rate, shelve_type="low")

    return signal, N, ideal_response


def _shelve(signal, frequency, gain, order, shelve_type, sampling_rate, kind):
    """
    First and second order high and low shelves.

    For the documentation refer to high_shelve and low_shelve. The only
    additional parameter is `kind`, which has to be 'high' or 'low'.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    if shelve_type not in ['I', 'II', 'III']:
        raise ValueError(("shelve_type must be 'I', 'II' or "
                          f"'III' but is '{shelve_type}'.'"))

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((2, 3))

    if order == 1 and kind == 'high':
        shelve = iir.biquad_hshv1st
    elif order == 2 and kind == 'high':
        shelve = iir.biquad_hshv2nd
    elif order == 1 and kind == 'low':
        shelve = iir.biquad_lshv1st
    elif order == 2 and kind == 'low':
        shelve = iir.biquad_lshv2nd
    else:
        raise ValueError(f"order must be 1 or 2 but is {order}")

    _, _, b, a = shelve(frequency, gain, fs, shelve_type)
    ba[0] = b
    ba[1] = a

    # generate filter object
    filt = pf.FilterIIR(ba, fs)
    kind = "High" if kind == "high" else "Low"
    filt.comment = (f"{kind}-shelve of order {order} and type "
                    f"{shelve_type} with {gain} dB gain at {frequency} Hz.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def _shelve_cascade(signal, frequency, frequency_type, gain, slope, bandwidth,
                    N, sampling_rate, shelve_type):
    """Design constant slope filter from shelve filter cascade.

    Parameters
    ----------
    shelve_type : string
        ``'low'``, or ``'high'`` for low- or high-shelve
    other : see high_shelve_cascade and low_shelve_cascade

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
        gain, slope, bandwidth, shelve_type)
    if bandwidth < 1:
        warnings.warn((
            f"The bandwidth is {bandwidth} octaves but should be at least 1 "
            "to obtain an good approximation of the desired frequency response"
        ))

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
    if frequency[1] > sampling_rate/2 and shelve_type == "low":
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
                       f"maintain the intended slope of {slope} dB/octave."))

    # determine number of shelve filters per octave ---------------------------

    # recommended minimum shelve filters per octave according to Eq. (11.2)
    N_octave_min = 1 if abs(slope) < 12.04 else abs(slope) / 12.04
    # minimum total shelve filters according to Eq. (9)
    N_min = np.ceil(N_octave_min*bandwidth).astype(int)

    # actual total shelve filters either from user input or recommended minimum
    N = int(N) if N else N_min

    if N < N_min:
        warnings.warn((
            f"N is {N} but should be at least {N_min} to obtain an good "
            "approximation of the desired frequency response"))

    # used shelve filters per octave
    N_octave = N / bandwidth

    # get the filter ----------------------------------------------------------

    # initialize variables
    filter_func = high_shelve if shelve_type == "high" else low_shelve
    shelve_gain = gain / N
    SOS = np.zeros((1, N, 6))

    # get the filter coefficients
    for n in range(N):
        # current frequency according to Eq. (5)
        f = 2**(-(n+.5)/N_octave) * frequency[1]
        # get shelve and cascade coefficients
        shelve = filter_func(None, f, shelve_gain, 2, 'III', sampling_rate)
        SOS[:, n] = shelve.coefficients.flatten()

    # make filter object
    comment = (f"Constant slope filter cascaded from {N} {shelve_type}-shelve "
               f"filters ({frequency_type} frequency: {frequency} Hz, "
               f"bandwidth: {bandwidth} octaves, gain: {gain} dB, {N_octave} "
               "shelve filters per octave")
    filt = pf.FilterSOS(SOS, sampling_rate, comment=comment)

    # get the ideal filter response -------------------------------------------
    magnitudes = np.array([10**(gain/20), 10**(gain/20), 1, 1])
    if shelve_type == "high":
        magnitudes = np.flip(magnitudes)
    frequencies = [0, frequency[0], frequency[1], sampling_rate/2]

    # remove duplicate entries (happens if the slope ends at Nyquist)
    if frequencies[-2] == frequencies[-1]:
        magnitudes = magnitudes[:-1]
        frequencies = frequencies[:-1]

    ideal_response = pf.FrequencyData(
        magnitudes, frequencies,
        "ideal magnitude response of cascaded shelve filter")

    # return parameter --------------------------------------------------------
    if signal is None:
        return filt, N, ideal_response
    else:
        return filt.process(signal), N, ideal_response


def _shelving_cascade_slope_parameters(gain, slope, bandwidth, shelve_type):
    """Compute the third parameter from the given two.

    Parameters
    ----------
    slope : float
        Desired shelving slope in decibel per octave.
    bandwidth : float
        Desired bandwidth of the slope in octave.
    gain : float
        Desired gain of the stop band in decibel.

    """
    if slope == 0:
        raise ValueError("slope must be non-zero.")

    if gain is None and slope is not None and bandwidth is not None:
        bandwidth = abs(bandwidth)
        gain = -bandwidth * slope if shelve_type == "low" \
            else bandwidth * slope
    elif slope is None and gain is not None and bandwidth is not None:
        bandwidth = abs(bandwidth)
        slope = -gain / bandwidth if shelve_type == "low" \
            else gain / bandwidth
    elif bandwidth is None and gain is not None and slope is not None:
        if shelve_type == "low" and np.sign(gain * slope) == 1:
            raise ValueError("gain and slope must have different signs")
        if shelve_type == "high" and np.sign(gain * slope) == -1:
            raise ValueError("gain and slope must have the same signs")
        bandwidth = abs(gain / slope)
    else:
        raise ValueError(("Exactly two out of the parameters gain, slope, and "
                          "bandwidth must be given."))

    return gain, slope, bandwidth
