import copy

import numpy as np
import scipy.signal as spsignal


class Filter(object):
    def __init__(
            self,
            coefficients,
            filter_func=spsignal.lfilter,
            state=None):
        super().__init__()
        self._coefficients = coefficients
        self._filter_func = filter_func
        if state is None:
            zi = spsignal.lfilter_zi(coefficients[0], coefficients[1])
            self._state = np.zeros_like(zi)
        else:
            self._state = state

    @property
    def state(self):
        return self._state

    @property
    def filter_func(self):
        return self._filter_func

    @filter_func.setter
    def filter_func(self, filter_func):
        self._filter_func = filter_func

    def process(self, signal):
        filtered, state = self.filter_func(signal.time, self.state)
        self.state = state
        signal.time = filtered
        filtered_signal = copy.deepcopy(signal)
        filtered_signal.time = filtered
        return filtered_signal

    def reset(self):
        self._state = np.zeros_like(self._state)


class FilterIIR(Filter):
    def __init__(self, coefficients, form='sos'):
        super().__init__()


class FilterFIR(Filter):
    def __init__(self):
        super().__init__()


class FilterSOS(Filter):
    def __init__(self):
        super().__init__()
