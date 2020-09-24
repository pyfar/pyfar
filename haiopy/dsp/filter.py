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


def lfilter(coefficients, signal, zi):
    return spsignal.lfilter(coefficients[0], coefficients[1], signal, zi=zi)


def filtfilt(coefficients, signal, **kwargs):
    kwargs.pop('zi', None)
    return spsignal.filtfilt(
        coefficients[0], coefficients[1], signal, **kwargs)


def sosfilt(sos, signal, zi):
    return spsignal.sosfilt(sos, signal, zi=zi)


def sosfiltfilt(sos, signal, **kwargs):
    kwargs.pop('zi', None)
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
        # if not isinstance(signal, Signal):
        #     raise ValueError("The input needs to be a haiopy.Signal object.")

        for coeff, state in zip(self._coefficients, self._state):
            filtered, new_state = self.filter_func(
                coeff, signal.time, state)

        # filtered_signal = copy.deepcopy(signal)
        # filtered_signal.time = filtered
        return filtered

    def reset(self):
        if self._state is not None:
            self._state = np.zeros_like(self._state)


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
            'minimumphase': filtfilt}
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
            'minimumphase': sosfiltfilt
        }
        self._filter_func = filter_func

    @property
    def filter_func(self):
        return self._filter_func


class FilterFIR(Filter):
    def __init__(self, coefficients):
        coeff = np.atleast_2d(coefficients)
        super().__init__()
