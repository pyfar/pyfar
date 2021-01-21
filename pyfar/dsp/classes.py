import copy
import warnings

import numpy as np
import scipy.signal as spsignal

from pyfar import Signal
from .. import utils


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


def extend_sos_coefficients(sos, order):
    """Extend a set of SOS filter coefficients to match a required filter order
    by adding sections with coefficients resulting in an ideal frequency
    response.
    """
    sos_order = sos.shape[0]
    pad_len = order-sos_order
    sos_ext = np.zeros((pad_len, 6))
    sos_ext[:, 3] = 1.
    sos_ext[:, 0] = 1.

    return np.vstack((sos, sos_ext))


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
            sampling_rate=None,
            state=None,
            comment=None):
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

        self._sampling_rate = sampling_rate

        self._comment = comment

    def initialize(self):
        raise NotImplementedError("Abstract class method")

    @property
    def sampling_rate(self):
        """Sampling rate of the filter in Hz. The sampling rate is set upon
        initialization and cannot be changed after the object has been created.
        """
        return self._sampling_rate

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

        if self.sampling_rate != signal.sampling_rate:
            raise ValueError(
                "The sampling rates of filter and signal do not match")

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
        if (signal.time.ndim == 2) and (signal.cshape[0] == 1):
            filtered_signal_data = np.squeeze(filtered_signal_data)
        filtered_signal.time = filtered_signal_data

        return filtered_signal

    def reset(self):
        if self._state is not None:
            self._state = np.zeros_like(self._state)
        else:
            warnings.warn(
                "No previous state was set. Initialize a filter state first.")

    @property
    def comment(self):
        """Get comment."""
        return self._comment

    @comment.setter
    def comment(self, value):
        """Set comment."""
        self._comment = str(value)

    def copy(self):
        """Return a deep copy of the Filter object."""
        return utils.copy(self)


class FilterFIR(Filter):
    """
    Filter object for FIR filters.
    """
    def __init__(
            self,
            coefficients,
            sampling_rate,
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

        super().__init__(coefficients=coeff, sampling_rate=sampling_rate)

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
            sampling_rate,
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
        super().__init__(coefficients=coeff, sampling_rate=sampling_rate)

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
            sampling_rate,
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
            coefficients=coeff, sampling_rate=sampling_rate)

        self._FILTER_FUNCS = {
            'default': sosfilt,
            'zerophase': sosfiltfilt
        }
        self._filter_func = filter_func

    @property
    def filter_func(self):
        return self._filter_func
