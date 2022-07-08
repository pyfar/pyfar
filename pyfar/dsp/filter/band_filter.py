import numpy as np
import scipy.signal as spsignal
import pyfar as pf


def butterworth(signal, N, frequency, btype='lowpass', sampling_rate=None):
    """
    Create and apply a digital Butterworth IIR filter.

    This is a wrapper for ``scipy.signal.butter``. Which creates digital
    Butterworth filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
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
    filt = pf.FilterSOS(sos, fs)
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


def chebyshev1(signal, N, ripple, frequency, btype='lowpass',
               sampling_rate=None):
    """
    Create and apply digital Chebyshev Type I IIR filter.

    This is a wrapper for ``scipy.signal.cheby1``. Which creates digital
    Chebyshev Type I filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    N : int
        The order of the Chebychev filter.
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
    filt = pf.FilterSOS(sos, fs)
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


def chebyshev2(signal, N, attenuation, frequency, btype='lowpass',
               sampling_rate=None):
    """
    Create and apply digital Chebyshev Type II IIR filter.

    This is a wrapper for ``scipy.signal.cheby2``. Which creates digital
    Chebyshev Type II filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    N : int
        The order of the Chebychev filter.
    attenuation : number
        The minimum stop band attenuation in dB.
    frequency : number, array like
        The frequency in Hz where the `attenuatoin` is first reached if `btype`
        is ``'lowpass'`` or ``'highpass'``. An array like containing the lower
        and upper frequencies in Hz if `btype` is ``'bandpass'`` or
        ``'bandstop'``.
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
    filt = pf.FilterSOS(sos, fs)
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


def elliptic(signal, N, ripple, attenuation, frequency, btype='lowpass',
             sampling_rate=None):
    """
    Create and apply digital Elliptic (Cauer) IIR filter.

    This is a wrapper for ``scipy.signal.ellip``. Which creates digital
    Elliptic (Cauer) filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    N : int
        The order of the Elliptic filter.
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
    filt = pf.FilterSOS(sos, fs)
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
    Bessel filter coefficients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filtered. Pass ``None`` to create the filter without
        applying it.
    N : int
        The order of the Bessel/Thomson filter.
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
    filt = pf.FilterSOS(sos, fs)
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


def crossover(signal, N, frequency, sampling_rate=None):
    """
    Create and apply Linkwitz-Riley crossover network.

    Linkwitz-Riley crossover filters ([#]_, [#]_) are designed by cascading
    Butterworth filters of order `N/2`. where `N` must be even.

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
    .. [#]  S. H. Linkwitz, 'Active crossover networks for noncoincident
            drivers,' J. Audio Eng. Soc., vol. 24, no. 1, pp. 2–8, Jan. 1976.
    .. [#]  D. Bohn, 'Linkwitz Riley crossovers: A primer,' Rane, RaneNote 160,
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
    SOS_dim_2 = n_sos if freq.size == 1 else 2 * n_sos

    SOS = np.tile(np.array([1, 0, 0, 1, 0, 0], dtype='float64'),
                  (freq.size + 1, SOS_dim_2, 1))

    # get filter coefficients for lowpass
    sos = spsignal.butter(N, freq[0], 'lowpass', analog=False, output='sos')
    SOS[0, 0:n_sos] = sos

    # get filter coefficients for the bandpass if more than one frequency is
    # provided
    for n in range(1, freq.size):
        sos_high = spsignal.butter(
            N, freq[n-1], 'highpass', analog=False, output='sos')
        sos_low = spsignal.butter(
            N, freq[n], 'lowpass', analog=False, output='sos')
        SOS[n] = np.concatenate((sos_high, sos_low))

    # get filter coefficients for the highpass
    sos = spsignal.butter(
        N, freq[-1], 'highpass', analog=False, output='sos')
    SOS[-1, 0:n_sos] = sos

    # Apply every Butterworth filter twice
    SOS = np.tile(SOS, (1, 2, 1))

    # invert phase in every second channel if the Butterworth order is odd
    # (realized by reversing b-coefficients of the first sos)
    if N % 2:
        SOS[np.arange(1, freq.size + 1, 2), 0, 0:3] *= -1

    # generate filter object
    filt = pf.FilterSOS(SOS, fs)
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
