import warnings

import numpy as np
import scipy.signal as spsignal

import pyfar

from . import _audiofilter as iir


def butter(signal, N, frequency, btype='lowpass', sampling_rate=None):
    """
    Create and apply a digital Butterworth IIR filter.

    This is a wrapper for ``scipy.signal.butter``. Which creates digital
    Butterworth filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass None to create the filter without
        applying it.
    N : int
        The order of the Butterworth filter
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is lowpass or highpass. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following ``'lowpass'``, ``'highpass'``, ``'bandpass'``,
        ``'bandstop'``. The default is ``'lowpass'``.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        SOS Filter object. Only returned if ``signal = None``.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate
    # normalized frequency (half-cycle / per sample)
    frequency_norm = np.asarray(frequency) / fs * 2

    # get filter coefficients
    sos = spsignal.butter(N, frequency_norm, btype, analog=False, output='sos')

    # generate filter object
    filt = pyfar.FilterSOS(sos, fs)
    filt.comment = (f"Butterworth {btype} of order {N}. "
                    f"Cut-off frequency {frequency} Hz.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def cheby1(signal, N, ripple, frequency, btype='lowpass', sampling_rate=None):
    """
    Create and apply digital Chebyshev Type I IIR filter.

    This is a wrapper for ``scipy.signal.cheby1``. Which creates digital
    Chebyshev Type I filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass None to create the filter without
        applying it.
    N : int
        The order of the Chebychev filter
    ripple : number
        The passband ripple in dB.
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is ``'lowpass'`` or
        ``'highpass'``. An array like containing the lower and upper cut-off
        frequencies in Hz if `btype` is ``'bandpass'`` or ``'bandstop'``.
    btype : str
        One of the following ``'lowpass'``, ``'highpass'``, ``'bandpass'``,
        ``'bandstop'``. The default is ``'lowpass'``.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        SOS Filter object. Only returned if ``signal = None``.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate
    # normalized frequency (half-cycle / per sample)
    frequency_norm = np.asarray(frequency) / fs * 2

    # get filter coefficients
    sos = spsignal.cheby1(N, ripple, frequency_norm, btype, analog=False,
                          output='sos')

    # generate filter object
    filt = pyfar.FilterSOS(sos, fs)
    filt.comment = (f"Chebychev Type I {btype} of order {N}. "
                    f"Cut-off frequency {frequency} Hz. "
                    f"Pass band ripple {ripple} dB.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def cheby2(signal, N, attenuation, frequency, btype='lowpass',
           sampling_rate=None):
    """
    Create and apply digital Chebyshev Type II IIR filter.

    This is a wrapper for ``scipy.signal.cheby2``. Which creates digital
    Chebyshev Type II filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass None to create the filter without
        applying it.
    N : int
        The order of the Chebychev filter
    attenuation : number
        The minimum stop band attenuation in dB.
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is ``'lowpass'`` or
        ``'highpass'``. An array like containing the lower and upper cut-off
        frequencies in Hz if `btype` is ``'bandpass'`` or ``'bandstop'``.
    btype : str
        One of the following ``'lowpass'``, ``'highpass'``, ``'bandpass'``,
        ``'bandstop'``. The default is ``'lowpass'``.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        SOS Filter object. Only returned if ``signal = None``.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate
    # normalized frequency (half-cycle / per sample)
    frequency_norm = np.asarray(frequency) / fs * 2

    # get filter coefficients
    sos = spsignal.cheby2(N, attenuation, frequency_norm, btype, analog=False,
                          output='sos')

    # generate filter object
    filt = pyfar.FilterSOS(sos, fs)
    filt.comment = (f"Chebychev Type II {btype} of order {N}. "
                    f"Cut-off frequency {frequency} Hz. "
                    f"Stop band attenuation {attenuation} dB.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def ellip(signal, N, ripple, attenuation, frequency, btype='lowpass',
          sampling_rate=None):
    """
    Create and apply digital Elliptic (Cauer) IIR filter.

    This is a wrapper for ``scipy.signal.ellip``. Which creates digital
    Chebyshev Type II filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass None to create the filter without
        applying it.
    N : int
        The order of the Elliptic filter
    ripple : number
        The passband ripple in dB.
    attenuation : number
        The minimum stop band attenuation in dB.
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is ``'lowpass'`` or
        ``'highpass'``. An array like containing the lower and upper cut-off
        frequencies in Hz if `btype` is ``'bandpass'`` or ``'bandstop'``.
    btype : str
        One of the following ``'lowpass'``, ``'highpass'``, ``'bandpass'``,
        ``'bandstop'``. The default is ``'lowpass'``.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        SOS Filter object. Only returned if ``signal = None``.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate
    # normalized frequency (half-cycle / per sample)
    frequency_norm = np.asarray(frequency) / fs * 2

    # get filter coefficients
    sos = spsignal.ellip(N, ripple, attenuation, frequency_norm, btype,
                         analog=False, output='sos')

    # generate filter object
    filt = pyfar.FilterSOS(sos, fs)
    filt.comment = (f"Elliptic (Cauer) {btype} of order {N}. "
                    f"Cut-off frequency {frequency} Hz. "
                    f"Pass band ripple {ripple} dB. "
                    f"Stop band attenuation {attenuation} dB.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def bessel(signal, N, frequency, btype='lowpass', norm='phase',
           sampling_rate=None):
    """
    Create and apply digital Bessel/Thomson IIR filter.

    This is a wrapper for ``scipy.signal.bessel``. Which creates digital
    Butterworth filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass None to create the filter without
        applying it.
    N : int
        The order of the Bessel/Thomson filter
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is ``'lowpass'`` or
        ``'highpass'``. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following ``'lowpass'``, ``'highpass'``, ``'bandpass'``,
        ``'bandstop'``. The default is ``'lowpass'``.
    norm : str
        Critical frequency normalization:

        ``'phase'``
            The filter is normalized such that the phase response reaches its
            midpoint at angular (e.g. rad/s) frequency `Wn`. This happens for
            both low-pass and high-pass filters, so this is the
            "phase-matched" case.
            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.
            This is the default, and matches MATLAB's implementation.
        ``'delay'``
            The filter is normalized such that the group delay in the passband
            is 1/`Wn` (e.g., seconds). This is the "natural" type obtained by
            solving Bessel polynomials.
        ``'mag'``
            The filter is normalized such that the gain magnitude is -3 dB at
            the angular frequency `Wn`.

        The default is 'phase'.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        SOS Filter object. Only returned if ``signal = None``.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate
    # normalized frequency (half-cycle / per sample)
    frequency_norm = np.asarray(frequency) / fs * 2

    # get filter coefficients
    sos = spsignal.bessel(N, frequency_norm, btype, analog=False,
                          output='sos', norm=norm)

    # generate filter object
    filt = pyfar.FilterSOS(sos, fs)
    filt.comment = (f"Bessel/Thomson {btype} of order {N} and '{norm}' "
                    f"normalization. Cut-off frequency {frequency} Hz.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def peq(signal, center_frequency, gain, quality, peq_type='II',
        quality_warp='cos', sampling_rate=None):
    """
    Create and apply second order parametric equalizer filter.

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
    peq_type : str
        Defines the bandwidth/quality. The default is ``'II'``

        ``'I'``
            not recommended. Also known as 'constant Q'
        ``'II'``
            defines the bandwidth by the points 3 dB below the maximum if the
            gain is positive and 3 dB above the minimum if the gain is
            negative. Also known as 'symmetric'
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

    if peq_type not in ['I', 'II', 'III']:
        raise ValueError(("peq_type must be 'I', 'II' or "
                          f"'III' but is '{peq_type}'.'"))

    if quality_warp not in ['cos', 'sin', 'tan']:
        raise ValueError(("quality_warp must be 'cos', 'sin' or "
                          f"'tan' but is '{quality_warp}'.'"))

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((2, 3))
    _, _, b, a = iir.biquad_peq2nd(
        center_frequency, gain, quality, fs, peq_type, quality_warp)
    ba[0] = b
    ba[1] = a

    # generate filter object
    filt = pyfar.FilterIIR(ba, fs)
    filt.comment = ("Second order parametric equalizer (PEQ) "
                    f"of type {peq_type} with {gain} dB gain at "
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
        Characteristic frequency of the shelve in Hz
    gain : number
        Gain of the shelve in dB
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
        The Signal to be filtered. Pass None to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelve in Hz
    gain : number
        Gain of the shelve in dB
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
            defines the characteristic frequency at gain/2 dB
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


def crossover(signal, N, frequency, sampling_rate=None):
    """
    Create and apply Linkwitz-Riley crossover network  [1]_, [2]_.

    Linkwitz-Riley crossover filters are designed by cascading Butterworth
    filters of order `N/2`. where `N` must be even.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    N : int
        The order of the Linkwitz-Riley crossover network, must be even.
    frequency : number, array-like
        Characteristic frequencies of the crossover network. If a single number
        is passed, the network consists of a single lowpass and highpass. If
        `M` frequencies are passed, the network consists of 1 lowpass, M-1
        bandpasses, and 1 highpass.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if `signal` is ``None``. The
        default is ``None``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        Filter object. Only returned if ``signal = None``.

    References
    ----------
    .. [1]  S. H. Linkwitz, 'Active crossover networks for noncoincident
            drivers,' J. Audio Eng. Soc., vol. 24, no. 1, pp. 2–8, Jan. 1976.
    .. [2]  D. Bohn, 'Linkwitz Riley crossovers: A primer,' Rane, RaneNote 160,
            2005.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    if N % 2:
        raise ValueError("The order 'N' must be an even number.")

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate
    # order of Butterworth filters
    N = int(N/2)
    # normalized frequency (half-cycle / per sample)
    freq = np.atleast_1d(np.asarray(frequency)) / fs * 2

    # init neutral SOS matrix of shape (freq.size+1, SOS_dim_2, 6)
    n_sos = int(np.ceil(N / 2))  # number of lowpass sos
    SOS_dim_2 = n_sos if freq.size == 1 else N
    SOS = np.tile(np.array([1, 0, 0, 1, 0, 0], dtype='float64'),
                  (freq.size + 1, SOS_dim_2, 1))

    # get filter coefficients for lowpass
    # (and bandpass if more than one frequency is provided)
    for n in range(freq.size):
        # get coefficients
        kind = 'lowpass' if n == 0 else 'bandpass'
        f = freq[n] if n == 0 else freq[n-1:n+1]
        sos = spsignal.butter(N, f, kind, analog=False, output='sos')
        # write to sos matrix
        if n == 0:
            SOS[n, 0:n_sos] = sos
        else:
            SOS[n] = sos

    # get filter coefficients for the highpass
    sos = spsignal.butter(N, freq[-1], 'highpass', analog=False, output='sos')
    # write to sos matrix
    SOS[-1, 0:n_sos] = sos

    # Apply every Butterworth filter twice
    SOS = np.tile(SOS, (1, 2, 1))

    # invert phase in every second channel if the Butterworth order is odd
    # (realized by reversing b-coefficients of the first sos)
    if N % 2:
        SOS[np.arange(1, freq.size + 1, 2), 0, 0:3] *= -1

    # generate filter object
    filt = pyfar.FilterSOS(SOS, fs)
    freq_list = [str(f) for f in np.array(frequency, ndmin=1)]
    filt.comment = (f"Linkwitz-Riley cross over network of order {N*2} at "
                    f"{', '.join(freq_list)} Hz.")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


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
    filt = pyfar.FilterIIR(ba, fs)
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


def fractional_octave_frequencies(
        num_fractions=1, frequency_range=(20, 20e3), return_cutoff=False):
    """Return the octave center frequencies according to the IEC 61260:1:2014
    standard. For numbers of fractions other than ``1`` and ``3``, only the
    exact center frequencies are returned, since nominal frequencies are not
    specified by corresponding standards.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., ``1`` refers to
        octave bands and ``3`` to third octave bands. The default is ``1``.
    frequency_range : array, tuple
        The lower and upper frequency limits, the default is
        ``frequency_range=(20, 20e3)``.

    Returns
    -------
    nominal : array, float
        The nominal center frequencies in Hz specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands
    exact : array, float
        The exact center frequencies in Hz, resulting in a uniform distribution
        of frequency bands over the frequency range.
    cutoff_freq : tuple, array, float
        The lower and upper critical frequencies in Hz of the bandpass filters
        for each band as a tuple corresponding to ``(f_lower, f_upper)``
    """
    nominal = None

    f_lims = np.asarray(frequency_range)
    if f_lims.size != 2:
        raise ValueError(
            "You need to specify a lower and upper limit frequency.")
    if f_lims[0] > f_lims[1]:
        raise ValueError(
            "The second frequency needs to be higher than the first.")

    if num_fractions in [1, 3]:
        nominal, exact = _center_frequencies_fractional_octaves_iec(
            nominal, num_fractions)

        mask = (nominal >= f_lims[0]) & (nominal <= f_lims[1])
        nominal = nominal[mask]
        exact = exact[mask]

    else:
        exact = _exact_center_frequencies_fractional_octaves(
            num_fractions, f_lims)

    if return_cutoff:
        octave_ratio = 10**(3/10)
        freqs_upper = exact * octave_ratio**(1/2/num_fractions)
        freqs_lower = exact * octave_ratio**(-1/2/num_fractions)
        f_crit = (freqs_lower, freqs_upper)
        return nominal, exact, f_crit
    else:
        return nominal, exact


def _exact_center_frequencies_fractional_octaves(
        num_fractions, frequency_range):
    """Calculate the center frequencies of arbitrary fractional octave bands.

    Parameters
    ----------
    num_fractions : int
        The number of fractions
    frequency_range
        The upper and lower frequency limits

    Returns
    -------
    exact : array, float
        An array containing the center frequencies of the respective fractional
        octave bands

    """
    ref_freq = 1e3
    Nmax = np.around(num_fractions*(np.log2(frequency_range[1]/ref_freq)))
    Nmin = np.around(num_fractions*(np.log2(ref_freq/frequency_range[0])))

    indices = np.arange(-Nmin, Nmax+1)
    exact = ref_freq * 2**(indices / num_fractions)

    return exact


def _center_frequencies_fractional_octaves_iec(nominal, num_fractions):
    """Returns the exact center frequencies for fractional octave bands
    according to the IEC 61260:1:2014 standard.
    octave ratio
    .. G = 10^{3/10}
    center frequencies
    .. f_m = f_r G^{x/b}
    .. f_m = f_e G^{(2x+1)/(2b)}
    where b is the number of octave fractions, f_r is the reference frequency
    chosen as 1000Hz and x is the index of the frequency band.

    Parameters
    ----------
    num_fractions : 1, 3
        The number of octave fractions. 1 returns octave center frequencies,
        3 returns third octave center frequencies.

    Returns
    -------
    nominal : array, float
        The nominal (rounded) center frequencies specified in the standard.
        Nominal frequencies are only returned for octave bands and third octave
        bands
    exact : array, float
        The exact center frequencies, resulting in a uniform distribution of
        frequency bands over the frequency range.
    """
    if num_fractions == 1:
        nominal = np.array([
            31.5, 63, 125, 250, 500, 1e3,
            2e3, 4e3, 8e3, 16e3], dtype=float)
    elif num_fractions == 3:
        nominal = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000,
            6300, 8000, 10000, 12500, 16000, 20000], dtype=float)

    reference_freq = 1e3
    octave_ratio = 10**(3/10)

    iseven = np.mod(num_fractions, 2) == 0
    if ~iseven:
        indices = np.around(
            num_fractions * np.log(nominal/reference_freq)
            / np.log(octave_ratio))
        exponent = (indices/num_fractions)
    else:
        indices = np.around(
            2.0*num_fractions *
            np.log(nominal/reference_freq) / np.log(octave_ratio) - 1)/2
        exponent = ((2*indices + 1) / num_fractions / 2)

    exact = reference_freq * octave_ratio**exponent

    return nominal, exact


def fractional_octave_bands(
        signal,
        num_fractions,
        sampling_rate=None,
        freq_range=(20.0, 20e3),
        order=14):
    """Create and/or apply a fractional octave filter bank.

    Parameters
    ----------
    signal : Signal, None
        The signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., ``1`` refers to
        octave bands and ``3`` to third octave bands. The default is ``1``.
    sampling_rate : None, int
        The sampling rate in Hz. Only required if signal is ``None``. The
        default is ``None``.
    frequency_range : array, tuple, optional
        The lower and upper frequency limits. The default is
         ``frequency_range=(20, 20e3)``
    order : int, optional
        Order of the Butterworth filter. The default is ``14``.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter : FilterSOS
        Filter object. Only returned if ``signal = None``.

    Notes
    -----
    This function uses second order sections of Butterworth filters for
    increased numeric accuracy and stability.
    """
    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    sos = _coefficients_fractional_octave_bands(
        sampling_rate=fs, num_fractions=num_fractions,
        freq_range=freq_range, order=order)

    filt = pyfar.FilterSOS(sos, fs)
    filt.comment = (
        "Second order section 1/{num_fractions} fractional octave band"
        "filter of order {order}")

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def _coefficients_fractional_octave_bands(
        sampling_rate, num_fractions,
        freq_range=(20.0, 20e3), order=14):
    """Calculate the second order section filter coefficients of a fractional
    octave band filter bank.

    Parameters
    ----------
    num_fractions : int, optional
        The number of bands an octave is divided into. Eg., 1 refers to octave
        bands and 3 to third octave bands. The default is 1.
    sampling_rate : None, int
        The sampling rate in Hz. Only required if signal is None. The default
        is None.
    frequency_range : array, tuple, optional
        The lower and upper frequency limits. The default is (20, 20e3)
    order : integer, optional
        Order of the Butterworth filter. The default is 14.


    Returns
    -------
    sos : array, float
        Second order section filter coefficients with shape (.., 6)

    Notes
    -----
    This function uses second order sections of butterworth filters for
    increased numeric accuracy and stability.
    """

    f_crit = fractional_octave_frequencies(
        num_fractions, freq_range, return_cutoff=True)[2]

    freqs_upper = f_crit[1]
    freqs_lower = f_crit[0]

    # normalize interval such that the Nyquist frequency is 1
    Wns = np.vstack((freqs_lower, freqs_upper)).T / sampling_rate * 2

    mask_skip = Wns[:, 0] >= 1
    if np.any(mask_skip):
        Wns = Wns[~mask_skip]
        warnings.warn("Skipping bands above the Nyquist frequency")

    num_bands = np.sum(~mask_skip)
    sos = np.zeros((num_bands, order, 6), np.double)

    for idx, Wn in enumerate(Wns):
        # in case the upper frequency limit is above Nyquist, use a highpass
        if Wn[-1] > 1:
            warnings.warn('The upper frequency limit {} Hz is above the \
                Nyquist frequency. Using a highpass filter instead of a \
                bandpass'.format(np.round(freqs_upper[idx], decimals=1)))
            Wn = Wn[0]
            btype = 'highpass'
            sos_hp = spsignal.butter(order, Wn, btype=btype, output='sos')
            sos_coeff = pyfar.classes.filter.extend_sos_coefficients(
                sos_hp, order)
        else:
            btype = 'bandpass'
            sos_coeff = spsignal.butter(
                order, Wn, btype=btype, output='sos')
        sos[idx, :, :] = sos_coeff
    return sos
