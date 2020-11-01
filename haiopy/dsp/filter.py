import copy
import warnings

import numpy as np
import scipy.signal as spsignal

from haiopy import Signal


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
                raise ValueError("Cannot set a state without filter coefficients")
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
            raise ValueError("The coefficients are not in line with a second order section filter structure.")
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
