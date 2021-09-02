import deepdiff
import warnings

import numpy as np
import scipy.signal as spsignal

import pyfar as pf
from copy import deepcopy


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


def extend_sos_coefficients(sos, order):
    """
    Extend a set of SOS filter coefficients to match a required filter order
    by adding sections with coefficients resulting in an ideal frequency
    response.

    Parameters
    ----------
    sos : array-like
        The second order section filter coefficients.
    order : int
        The order to which the coefficients are to be extended.

    Returns
    -------
    sos_ext : array-like
        The extended second order section coefficients.

    """
    sos_order = sos.shape[0]
    if sos_order == order:
        return sos
    pad_len = order - sos_order
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
            sampling_rate=None,
            state=None,
            comment=None):
        """
        Initialize a general Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array.
        sampling_rate : number
            The sampling rate of the filter in Hz.
        state : array, optional
            The state of the filter from a priory knowledge.

        Returns
        -------
        Filter
            The filter object.

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

    @staticmethod
    def _process(coefficients, data, zi=None):
        raise NotImplementedError("Abstract class method.")

    def process(self, signal, reset=True):
        """Apply the filter to a signal.

        Parameters
        ----------
        signal : Signal
            The data to be filtered as Signal object.
        reset : bool, optional
            If set to ``True``, the filter state will be reset to zero before
            the filter is applied to the signal. The default is ``'True'``.

        Returns
        -------
        filtered : Signal
            A filtered copy of the input signal.
        """
        if not isinstance(signal, pf.Signal):
            raise ValueError("The input needs to be a haiopy.Signal object.")

        if self.sampling_rate != signal.sampling_rate:
            raise ValueError(
                "The sampling rates of filter and signal do not match")

        if reset is True:
            self.reset()

        filtered_signal_data = np.zeros(
            (self.size, *signal.time.shape),
            dtype=signal.time.dtype)

        if self.state is not None:
            for idx, (coeff, state) in enumerate(
                    zip(self._coefficients, self._state)):
                filtered_signal_data[idx, ...], new_state = self._process(
                    coeff, filtered_signal_data[idx, ...], state)
        else:
            for idx, coeff in enumerate(self._coefficients):
                filtered_signal_data[idx, ...] = self._process(
                    coeff, signal.time, zi=None)

        filtered_signal = deepcopy(signal)
        filtered_signal.time = np.squeeze(filtered_signal_data)

        return filtered_signal

    def reset(self):
        """Reset the filter state by filling it with zeros."""
        if self._state is not None:
            self._state = np.zeros_like(self._state)
        else:
            self._state = None

    @property
    def comment(self):
        """Get comment."""
        return self._comment

    @comment.setter
    def comment(self, value):
        """Set comment."""
        self._comment = str(value)

    def copy(self):
        """Return a copy of the Filter object."""
        return deepcopy(self)

    def _encode(self):
        """Return dictionary for the encoding."""
        return self.copy().__dict__

    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective object dictionary."""
        # initializing this way satisfies FIR, IIR and SOS initialization
        obj = cls(np.zeros((1, 6)), None)
        obj.__dict__.update(obj_dict)
        return obj

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(self, other)


class FilterFIR(Filter):
    """
    Filter object for FIR filters.
    """
    def __init__(
            self,
            coefficients,
            sampling_rate):
        """
        Initialize an finite impulse response (FIR) Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array with dimensions
            (number of channels, number of filter coefficients)
        sampling_rate : number
            The sampling rate of the filter in Hz.
        state : array, optional
            The state of the filter from a priory knowledge.

        Returns
        -------
        FilterFIR
            The FIR filter object.
        """
        b = np.atleast_2d(coefficients)
        a = np.zeros_like(b)
        a[..., 0] = 1
        coeff = np.stack((b, a), axis=-2)

        super().__init__(coefficients=coeff, sampling_rate=sampling_rate)

    @staticmethod
    def _process(coefficients, data, zi=None):
        return spsignal.lfilter(coefficients[0], 1, data, zi=zi)


class FilterIIR(Filter):
    """
    Filter object for IIR filters. For IIR filters with high orders, second
    order section IIR filters using FilterSOS should be considered.
    """
    def __init__(
            self,
            coefficients,
            sampling_rate):
        """IIR filter
        Initialize an infinite impulse response (IIR) Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array, with shape
            (number of channels, number of coefficients in the nominator,
            number of coefficients in the denominator)
        sampling_rate : number
            The sampling rate of the filter in Hz.
        state : array, optional
            The state of the filter from a priory knowledge.

        Returns
        -------
        FilterIIR
            The IIR filter object.
        """
        coeff = np.atleast_2d(coefficients)
        super().__init__(coefficients=coeff, sampling_rate=sampling_rate)

    @staticmethod
    def _process(coefficients, data, zi=None):
        return spsignal.lfilter(coefficients[0], coefficients[1], data, zi=zi)


class FilterSOS(Filter):
    """
    Filter object for IIR filters as second order sections (SOS).
    """
    def __init__(
            self,
            coefficients,
            sampling_rate):
        """
        Initialize a second order sections (SOS) Filter object.

        Parameters
        ----------
        coefficients : array, double
            The filter coefficients as an array with dimensions
            (n_filter_chan, n_sections, 6)
        sampling_rate : number
            The sampling rate of the filter in Hz.
        state : array, optional
            The state of the filter from a priory knowledge.

        Returns
        -------
        FilterSOS
            The SOS filter object.
        """
        coeff = np.atleast_2d(coefficients)
        if coeff.shape[-1] != 6:
            raise ValueError(
                "The coefficients are not in line with a second order",
                "section filter structure.")
        super().__init__(
            coefficients=coeff, sampling_rate=sampling_rate)

    @staticmethod
    def _process(sos, data, zi=None):
        return spsignal.sosfilt(sos, data, zi=zi)
