#%%
"""
Documentation for regularization class.
"""

from pyfar import FrequencyData, Signal
from pyfar.dsp.dsp import _cross_fade
import numpy as np

class Regularization():
    """Class for regularization.
    """

    def __init__(self,
                 frequency_range: np.array=np.asarray([]),
                 regu_outside: float=None,
                 regu_inside: float=None,
                 regu_type: str="") -> None:

        # Initialize empty Regularization object
        super(Regularization, self).__init__()
        self.frequency_range = frequency_range
        self.regu_outside = regu_outside
        self.regu_inside = regu_inside
        self.regu_type = regu_type

        # attributes for internal use
        self._signal = None
        self._regu = None


    @property
    def regu_data(self):
        return self._regu

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        """
        Setter method for signal attribute.

        Used to call the _get_regularization_factors method to calculate the
        regularization factors for the given signal.
        """

        if not isinstance(signal, (FrequencyData, Signal)):
            raise TypeError(
                "signal must be an instance of FrequencyData or Signal")

        self._signal = signal

        self._regu = self._get_regularization_factors()

    def __repr__(self):
        return f"Regularization object with regularization "\
            f"of type {self.regu_type}"


    @classmethod
    def from_frequency_range(cls, frequency_range, regu_outside=1.,
                             regu_inside=10**(-200/20)):
        """
        Regularization from frequency range.

        Parameters
        ----------
        frequency_range : array, tuple
            Tuple with two values for the lower and upper frequency limit.
        regu_outside : float
            Regularization factor outside the frequency range.
        regu_inside : float
            Regularization factor inside the frequency range.
        """

        return cls(frequency_range, regu_outside, regu_inside,
                   regu_type="frequency range")


    def _get_regularization_factors(self):
        """
        Aranges the regularization factors depending on regularization type
        and number of bins in the signal.

        Returns
        -------
        regu_final : FrequencyData
            Regularization factors per frequency as pf.FrequencyData object.
        """

        if self.regu_type == "frequency range":
            regu_inside = np.ones(self._signal.n_bins,
                                  dtype=np.double) * self.regu_inside
            regu_outside = np.ones(self._signal.n_bins,
                                   dtype=np.double) * self.regu_outside

            idx_xfade_lower = self._signal.find_nearest_frequency(
                [self.frequency_range[0]/np.sqrt(2), self.frequency_range[0]])

            regu_final = _cross_fade(regu_outside, regu_inside,
                                     idx_xfade_lower)

            if self.frequency_range[1] < self._signal.sampling_rate/2:
                idx_xfade_upper = self._signal.find_nearest_frequency([
                    self.frequency_range[1],
                    np.min([self.frequency_range[1]*np.sqrt(2),
                            self._signal.sampling_rate/2])])

                regu_final = _cross_fade(regu_final, regu_outside,
                                         idx_xfade_upper)
            return FrequencyData(regu_final, self._signal.frequencies)


#%%
import pyfar as pf
import matplotlib.pyplot as plt

regu = Regularization.from_frequency_range((20, 20e3))

sig = pf.signals.linear_sweep_time(512, (100, 20e3))


regu.signal = sig
# #regu.signal = sig.time
plt.plot(sig.frequencies, regu.regu_data.freq[0,:])
plt.show()

regu
# %%
