import copy
import warnings

import numpy as np
import scipy.signal as spsignal

from pyfar import Signal


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
    frequency_norm = frequency / fs * 2

    # get filter coefficients
    sos = spsignal.butter(N, frequency_norm, btype, analog=False,
                          output='sos', fs=sampling_rate)

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
    frequency_norm = frequency / fs * 2

    # get filter coefficients
    sos = spsignal.cheby1(N, ripple, frequency_norm, btype, analog=False,
                          output='sos', fs=sampling_rate)

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
    frequency_norm = frequency / fs * 2

    # get filter coefficients
    sos = spsignal.cheby2(N, attenuation, frequency_norm, btype, analog=False,
                          output='sos', fs=sampling_rate)

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
    frequency_norm = frequency / fs * 2

    # get filter coefficients
    sos = spsignal.ellip(N, ripple, attenuation, frequency_norm, btype,
                         analog=False, output='sos', fs=sampling_rate)

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
    frequency_norm = frequency / fs * 2

    # get filter coefficients
    sos = spsignal.bessel(N, frequency_norm, btype, analog=False,
                          output='sos', norm=norm, fs=sampling_rate)

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

    All input parameters except `signal` and `sampling_rate` can also be array
    likes of the same length to create multiple filters. Uses the
    implementation of
    https://github.com/spatialaudio/digital-signal-processing-lecture
    (audiofilter in the filter_design lecture)

    Paramters
    ---------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    center_frequency : number, array like
        Center frequency of the parametric equalizer in Hz
    gain : number, array like
        Gain of the parametric equalizer in dB
    quality : number, array like
        Quality of the parametric equalizer, i.e., the inverse of the
        bandwidth
    peq_type : str, array like of strings
        Defines the bandwidth/quality. The default is 'II'

        I   - not recommended. Also known as 'constant Q'
        II  - defines the bandwidth by the points 3 dB below the maximum if the
              gain is positve and 3 dB above the minimum if the gain is
              negative. Also known as 'symmetric'
        III - defines the bandwidth by the points at gain/2. Also known as
              'half pad loss'.
    qualtiy_war : str, array like of strings
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

    # check and tile filter parameter
    p = _tile_filter_parameters(
        [center_frequency, gain, quality, peq_type, quality_warp])

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((p[0].size, 2, 3))
    for n, (freq, gain, q, peq_type, warp) in \
            enumerate(zip(p[0], p[1], p[2], p[3], p[4])):

        if peq_type not in ['I', 'II', 'III']:
            raise ValueError(("peq_type must be 'I', 'II' or "
                              f"'III' but is '{peq_type}'.'"))

        _, _, b, a = iir.biquad_peq2nd(freq, gain, q, fs, peq_type, warp)
        ba[n, 0] = b
        ba[n, 1] = a

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

    All input parameters except `signal` and `sampling_rate` can also be array
    likes of the same length to create multiple filters. Uses the
    implementation of
    https://github.com/spatialaudio/digital-signal-processing-lecture
    (audiofilter in the filter_design lecture)

    Paramters
    ---------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    frequency : number, array like
        Characteristic frequency of the shelve in Hz
    gain : number, array like
        Gain of the shelve in dB
    order : number, array like
        The shelve order. Must be 1 or 2.
    shelve_type : str, array like of strings
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


def low_shelve(signal, frequency, gain, order, shelve_type, sampling_rate):
    """
    Create and apply first or second order low shelve filter.

    All input parameters except `signal` and `sampling_rate` can also be array
    likes of the same length to create multiple filters. Uses the
    implementation of
    https://github.com/spatialaudio/digital-signal-processing-lecture
    (audiofilter in the filter_design lecture)

    Paramters
    ---------
    signal : Signal, None
        The Signal to be filterd. Pass None to create the filter without
        applying it.
    frequency : number, array like
        Characteristic frequency of the shelve in Hz
    gain : number, array like
        Gain of the shelve in dB
    order : number, array like
        The shelve order. Must be 1 or 2.
    shelve_type : str, array like of strings
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
    def __init__(
            self,
            coefficients=None,
            filter_func=None,
            state=None):
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
            'minimumphase': None}

    def initialize(self):
        raise NotImplementedError("Abstract class method")

    @property
    def shape(self):
        return self._coefficients.shape[:-2]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def state(self):
        return self._state

    @property
    def filter_func(self):
        raise NotImplementedError("Abstract class method")

    @filter_func.setter
    def filter_func(self, filter_func):
        raise NotImplementedError("Abstract class method")

    def process(self, signal, reset=True):
        """Apply the filter to a signal.
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
    def __init__(
            self,
            coefficients,
            filter_func=lfilter):

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
    def __init__(
            self,
            coefficients,
            filter_func=lfilter):
        """IIR filter
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
    def __init__(
            self,
            coefficients,
            filter_func=sosfilt):
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

    # check and tile filter parameter
    p = _tile_filter_parameters([frequency, gain, order, shelve_type])

    # sampling frequency in Hz
    fs = signal.sampling_rate if sampling_rate is None else sampling_rate

    # get filter coefficients
    ba = np.zeros((p[0].size, 2, 3))
    for n, (freq, gain, order, shelve_type) in \
            enumerate(zip(p[0], p[1], p[2], p[3])):

        if shelve_type not in ['I', 'II', 'III']:
            raise ValueError(("shelve_type must be 'I', 'II' or "
                              f"'III' but is '{shelve_type}'.'"))

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

        _, _, b, a = shelve(freq, gain, fs, shelve_type)
        ba[n, 0] = b
        ba[n, 1] = a

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


def _tile_filter_parameters(parameters: list):
    """
    Return parameters as list of flattend numpy arrays if sizes agree and raise
    ValueError otherwise.

    Parameters
    ----------
    parameters : list
        parameters can be a mixture of single elements and array likes of the
        same length, e.g., [parameter_1, ..., parameter_N]

    Returns
    -------
    parameters : list
        flattened list of input parameters tiled to be of the same length,
        e.g., [np.array(parameter_1), ..., np.array(parameter_N)]
    """

    # cast to numpy arrays
    params = [np.asarray(p).flatten() for p in parameters]
    # check number of elements in each parameter
    len_params = [p.size for p in params]
    n_filter = max(len_params)
    # check for equal length
    for length in len_params:
        if length > 1 and length != n_filter:
            raise ValueError(
                "All parameters must be scalar or have the same length.")
    # tile
    params = [np.tile(p, n_filter) if p.size == 1 else p for p in params]

    return params
