import copy
import warnings

import numpy as np
import scipy.signal as spsignal

from pyfar import Signal
import pyfar.dsp._audiofilter as iir


def butter(signal, N, frequency, btype='lowpass', sampling_rate=None):
    """
    Create and apply digital Butterworth IIR filter.

    This is a wrapper for scipy.signal.butter(). Which creates digitial
    Butterworth filter coeffiecients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    N : int
        The order of the Butterworth filter
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is lowpass or highpass. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following 'lowpass', 'highpass', 'bandpass', 'bandstop'. The
        default is 'lowpass'.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        SOS Filter object. Only returned if `signal = None`.
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
    filt = FilterSOS(sos)

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

    This is a wrapper for scipy.signal.cheby1(). Which creates digitial
    Chebyshev Type I filter coeffiecients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    N : int
        The order of the Chebychev filter
    ripple : number
        The passband ripple in dB.
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is lowpass or highpass. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following 'lowpass', 'highpass', 'bandpass', 'bandstop'. The
        default is 'lowpass'.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        SOS Filter object. Only returned if `signal = None`.
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
    filt = FilterSOS(sos)

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

    This is a wrapper for scipy.signal.cheby2(). Which creates digitial
    Chebyshev Type II filter coeffiecients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    N : int
        The order of the Chebychev filter
    attenuation : number
        The minimum stop band attenuation in dB.
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is lowpass or highpass. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following 'lowpass', 'highpass', 'bandpass', 'bandstop'. The
        default is 'lowpass'.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        SOS Filter object. Only returned if `signal = None`.
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
    filt = FilterSOS(sos)

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

    This is a wrapper for scipy.signal.ellip(). Which creates digitial
    Chebyshev Type II filter coeffiecients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    N : int
        The order of the Elliptic filter
    ripple : number
        The passband ripple in dB.
    attenuation : number
        The minimum stop band attenuation in dB.
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is lowpass or highpass. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following 'lowpass', 'highpass', 'bandpass', 'bandstop'. The
        default is 'lowpass'.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        SOS Filter object. Only returned if `signal = None`.
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
    filt = FilterSOS(sos)

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

    This is a wrapper for scipy.signal.bessel(). Which creates digitial
    Butterworth filter coeffiecients in second-order sections (SOS).

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    N : int
        The order of the Bessel/Thomson filter
    frequency : number, array like
        The cut off-frequency in Hz if `btype` is lowpass or highpass. An array
        like containing the lower and upper cut-off frequencies in Hz if
        `btype` is bandpass or bandstop.
    btype : str
        One of the following 'lowpass', 'highpass', 'bandpass', 'bandstop'. The
        default is 'lowpass'.
    norm : str
        Critical frequency normalization:
        ``phase``
            The filter is normalized such that the phase response reaches its
            midpoint at angular (e.g. rad/s) frequency `Wn`. This happens for
            both low-pass and high-pass filters, so this is the
            "phase-matched" case.
            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.
            This is the default, and matches MATLAB's implementation.
        ``delay``
            The filter is normalized such that the group delay in the passband
            is 1/`Wn` (e.g., seconds). This is the "natural" type obtained by
            solving Bessel polynomials.
        ``mag``
            The filter is normalized such that the gain magnitude is -3 dB at
            angular frequency `Wn`.

        The default is ``phase``.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        SOS Filter object. Only returned if `signal = None`.
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
    filt = FilterSOS(sos)

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

    Uses the implementation of
    https://github.com/spatialaudio/digital-signal-processing-lecture
    (audiofilter.py in the filter_design lecture)

    Paramters
    ---------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    center_frequency : number
        Center frequency of the parametric equalizer in Hz
    gain : number
        Gain of the parametric equalizer in dB
    quality : number
        Quality of the parametric equalizer, i.e., the inverse of the
        bandwidth
    peq_type : str
        Defines the bandwidth/quality. The default is 'II'

        I   - not recommended. Also known as 'constant Q'
        II  - defines the bandwidth by the points 3 dB below the maximum if the
              gain is positve and 3 dB above the minimum if the gain is
              negative. Also known as 'symmetric'
        III - defines the bandwidth by the points at gain/2. Also known as
              'half pad loss'.
    qualtiy_warp : str
        Sets the pre-warping for the quality ('cos', 'sin', or 'tan'). The
        default is 'cos'.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        Filter object. Only returned if `signal = None`.
    """

    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    if peq_type not in ['I', 'II', 'III']:
        raise ValueError(("peq_type must be 'I', 'II' or "
                          f"'III' but is '{peq_type}'.'"))

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((2, 3))
    _, _, b, a = iir.biquad_peq2nd(
        center_frequency, gain, quality, fs, peq_type, quality_warp)
    ba[0] = b
    ba[1] = a

    # generate filter object
    filt = FilterIIR(ba)

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
    Create and apply first or second order high shelve filter.

    Uses the implementation of
    https://github.com/spatialaudio/digital-signal-processing-lecture
    (audiofilter in the filter_design lecture)

    Paramters
    ---------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelve in Hz
    gain : number
        Gain of the shelve in dB
    order : number
        The shelve order. Must be 1 or 2.
    shelve_type : str
        Defines the characteristik frequency. The default is 'I'

        I   - defines the characteristic frequency 3 dB below the gain value if
              the gain is positive and 3 dB above the gain value otherwise
        II  - defines the characteristic frequency at 3 dB if the gain is
              positive and at -3 dB if the gain is negative.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        Filter object. Only returned if `signal = None`.
    """

    output = _shelve(
        signal, frequency, gain, order, shelve_type, sampling_rate, 'high')

    return output


def low_shelve(signal, frequency, gain, order, shelve_type='I',
               sampling_rate=None):
    """
    Create and apply first or second order low shelve filter.

    Uses the implementation of
    https://github.com/spatialaudio/digital-signal-processing-lecture
    (audiofilter in the filter_design lecture)

    Paramters
    ---------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    frequency : number
        Characteristic frequency of the shelve in Hz
    gain : number
        Gain of the shelve in dB
    order : number
        The shelve order. Must be 1 or 2.
    shelve_type : str
        Defines the characteristik frequency. The default is 'I'

        I   - defines the characteristic frequency 3 dB below the gain value if
              the gain is positive and 3 dB above the gain value otherwise
        II  - defines the characteristic frequency at 3 dB if the gain is
              positive and at -3 dB if the gain is negative.
        III - defines the characteristic frequency at gain/2 dB
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        Filter object. Only returned if `signal = None`.
    """

    output = _shelve(
        signal, frequency, gain, order, shelve_type, sampling_rate, 'low')

    return output


def crossover(signal, N, frequency, sampling_rate=None):
    """
    Create and apply Linkwitz-Riley crossover network  [1]_, [2]_.

    Linwitz-Riley crossover filters are desined by cascading Butterworth
    filters of order `N/2`. `N` must this be even.

    Parameters
    ----------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    N : int
        The order of the Linkwitz-Riley crossover network
    frequency : number, array like
        Characteristic frequencies of the crossover network. If a single number
        is passed, the network consists of a single lowpass and highpass. If
        `M` frequencies are passed, the network consists of 1 lowpass, M-1
        bandpasses, and 1 highpass.
    order : number
        The filter order. Must be an even number.
    sampling_rate : None, number
        The sampling rate in Hz. Only required if signal is None. The default
        is None.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if `sampling_rate = None`.
    filter : Filter
        Filter object. Only returned if `signal = None`.

    References
    ----------
    .. [1] S. H. Linkwitz, 'Active crossover networks for noncoincident
           drivers,' J. Audio Eng. Soc., vol. 24, no. 1, pp. 2–8, Jan. 1976.
    .. [2] D. Bohn, 'Linkwitz Riley crossovers: A primer,' Rane, RaneNote 160,
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
    # order of Butterwirth filters
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
    filt = FilterSOS(SOS)

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt


def atleast_3d_first_dim(arr):
    arr = np.asarray(arr)
    ndim = np.ndim(arr)

    if ndim < 2:
        arr = np.atleast_2d(arr)
    if ndim < 3:
        return arr[np.newaxis]
    else:
        return arr


def pop_state_from_kwargs(**kwargs):
    kwargs.pop('zi', None)
    warnings.warn(
        "This filter function does not support saving the filter state")
    return kwargs


def lfilter(coefficients, signal, zi):
    return spsignal.lfilter(coefficients[0], coefficients[1], signal, zi=zi)


def filtfilt(coefficients, signal, **kwargs):
    kwargs = pop_state_from_kwargs(kwargs)
    return spsignal.filtfilt(
        coefficients[0], coefficients[1], signal, **kwargs)


def sosfilt(sos, signal, zi):
    return spsignal.sosfilt(sos, signal, zi=zi)


def sosfiltfilt(sos, signal, **kwargs):
    kwargs = pop_state_from_kwargs(kwargs)
    return spsignal.sosfiltfilt(sos, signal, **kwargs)


class Filter(object):
    """
    Container class for digital filters.
    This is an abstract class method, only used for the shared processing
    method used for the application of a filter on a signal.
    """
    def __init__(
            self,
            coefficients=None,
            filter_func=None,
            state=None):
        """
        Initialize a general Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array.
        filter_func : default, zerophase
            Default applies a direct form II transposed time domain filter
            based on the standard difference equation. Zerophase uses
            the same filter twice, first forward, then backwards resulting
            in zero phase.
        state : array, optional
            The state of the filter from a priory knowledge.


        Returns
        -------
        filter : Filter
            The filter object

        """
        super().__init__()
        if coefficients is not None:
            coefficients = atleast_3d_first_dim(coefficients)
        self._coefficients = coefficients
        if state is not None:
            if coefficients is None:
                raise ValueError(
                    "Cannot set a state without filter coefficients")
            state = atleast_3d_first_dim(state)
            self._initialized = True
        else:
            self._initialized = False
        self._state = state
        self._filter_func = None

        self._FILTER_FUNCS = {
            'default': None,
            'zerophase': None}

    def initialize(self):
        raise NotImplementedError("Abstract class method")

    @property
    def shape(self):
        """
        The shape of the filter.
        """
        return self._coefficients.shape[:-2]

    @property
    def size(self):
        """
        The size of the filter, that is all elements in the filter object.
        """
        return np.prod(self.shape)

    @property
    def state(self):
        """
        The current state of the filter as an array with dimensions
        corresponding to the order of the filter and number of filter channels.
        """
        return self._state

    @property
    def filter_func(self):
        raise NotImplementedError("Abstract class method")

    @filter_func.setter
    def filter_func(self, filter_func):
        raise NotImplementedError("Abstract class method")

    def process(self, signal, reset=True):
        """Apply the filter to a signal.

        Parameters
        ----------
        signal : Signal
            The data to be filtered as Signal object.
        reset : bool, True
            If set to true, the filter state will be reset to zero before the
            filter is applied to the signal.

        Returns
        -------
        filtered : Signal
            A filtered copy of the input signal.
        """
        if not isinstance(signal, Signal):
            raise ValueError("The input needs to be a haiopy.Signal object.")

        if reset is True:
            self.reset()

        if self.size > 1:
            filtered_signal_data = np.broadcast_to(
                signal.time,
                (self.shape[0], *signal.time.shape))
            filtered_signal_data = filtered_signal_data.copy()
        else:
            filtered_signal_data = signal.time.copy()

        if self.state is not None:
            for idx, (coeff, state) in enumerate(
                    zip(self._coefficients, self._state)):
                filtered_signal_data[idx, ...], new_state = self.filter_func(
                    coeff, filtered_signal_data[idx, ...], state)
        else:
            for idx, coeff in enumerate(self._coefficients):
                filtered_signal_data[idx, ...] = self.filter_func(
                    coeff, filtered_signal_data[idx, ...], zi=None)

        filtered_signal = copy.deepcopy(signal)
        if (signal.time.ndim == 2) and (signal.shape[0] == 1):
            filtered_signal_data = np.squeeze(filtered_signal_data)
        filtered_signal.time = filtered_signal_data

        return filtered_signal

    def reset(self):
        if self._state is not None:
            self._state = np.zeros_like(self._state)
        else:
            warnings.warn(
                "No previous state was set. Initialize a filter state first.")


class FilterFIR(Filter):
    """
    Filter object for FIR filters.
    """
    def __init__(
            self,
            coefficients,
            filter_func=lfilter):
        """
        Initialize a general Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array with dimensions
            (n_channels_filter, num_coefficients)
        filter_func : default, zerophase
            Default applies a direct form II transposed time domain filter
            based on the standard difference equation. Zerophase uses
            the same filter twice, first forward, then backwards resulting
            in zero phase.
        state : array, optional
            The state of the filter from a priory knowledge.
        """
        b = np.atleast_2d(coefficients)
        a = np.zeros_like(b)
        a[..., 0] = 1
        coeff = np.stack((b, a), axis=-2)

        super().__init__(coefficients=coeff)

        self._FILTER_FUNCS = {
            'default': lfilter,
            'zerophase': filtfilt}
        self._filter_func = filter_func

    @property
    def filter_func(self):
        return self._filter_func

    @filter_func.setter
    def filter_func(self, filter_func):
        if type('filter_func') == str:
            filter_func = self._FILTER_FUNCS[filter_func]
        self._filter_func = filter_func


class FilterIIR(Filter):
    """
    Filter object for IIR filters. For IIR filters with high orders, second
    order section IIR filters using FilterSOS should be considered.
    """
    def __init__(
            self,
            coefficients,
            filter_func=lfilter):
        """IIR filter
        Initialize a general Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array, with shape
            (n_filter_channels, n_coefficients_num, n_coefficients_denom)
        filter_func : default, zerophase
            Default applies a direct form II transposed time domain filter
            based on the standard difference equation. Zerophase uses
            the same filter twice, first forward, then backwards resulting
            in zero phase.
        state : array, optional
            The state of the filter from a priory knowledge.
        """
        coeff = np.atleast_2d(coefficients)
        super().__init__(coefficients=coeff)

        self._FILTER_FUNCS = {
            'default': lfilter,
            'zerophase': filtfilt}
        self._filter_func = filter_func

    @property
    def filter_func(self):
        return self._filter_func

    @filter_func.setter
    def filter_func(self, filter_func):
        if type('filter_func') == str:
            filter_func = self._FILTER_FUNCS[filter_func]
        self._filter_func = filter_func


class FilterSOS(Filter):
    """
    Filter object for IIR filters as second order sections.
    """
    def __init__(
            self,
            coefficients,
            filter_func=sosfilt):
        """
        Initialize a general Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array with dimensions
            (n_filter_chan, n_sections, 6)
        filter_func : default, zerophase
            Default applies a direct form II transposed time domain filter
            based on the standard difference equation. Zerophase uses
            the same filter twice, first forward, then backwards resulting
            in zero phase.
        state : array, optional
            The state of the filter from a priory knowledge.

        """
        coeff = np.atleast_2d(coefficients)
        if coeff.shape[-1] != 6:
            raise ValueError(
                "The coefficients are not in line with a second order",
                "section filter structure.")
        super().__init__(
            coefficients=coeff)

        self._FILTER_FUNCS = {
            'default': sosfilt,
            'zerophase': sosfiltfilt
        }
        self._filter_func = filter_func

    @property
    def filter_func(self):
        return self._filter_func


def _shelve(signal, frequency, gain, order, shelve_type, sampling_rate, kind):
    """
    First and second order high and low shelves.

    For the documentation refer to high_shelve and low_shelve. The only
    additional parameteris `kind`, which has to be 'high' or 'low'.
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
    elif order != 2 and kind == 'high':
        shelve = iir.biquad_hshv2nd
    elif order == 1 and kind == 'low':
        shelve = iir.biquad_lshv1st
    elif order != 2 and kind == 'low':
        shelve = iir.biquad_lshv2nd
    else:
        raise ValueError(f"order must be 1 or 2 but is {order}")

    _, _, b, a = shelve(frequency, gain, fs, shelve_type)
    ba[0] = b
    ba[1] = a

    # generate filter object
    filt = FilterIIR(ba)

    # return the filter object
    if signal is None:
        # return the filter object
        return filt
    else:
        # return the filtered signal
        signal_filt = filt.process(signal)
        return signal_filt
