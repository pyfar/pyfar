"""
Documentation for regularization class.
"""

import pyfar as pf
from pyfar.dsp.dsp import _cross_fade
import numpy as np

class Regularization():
    """Class for regularization."""

    def __init__(self, regu_type: str="") -> None:

        # throw error if object is instanced without classmethod
        if regu_type == "":
            raise RuntimeError("Regularization objects must be created "
                             "using a classmethod.")

    def __repr__(self):
        return f"Regularization object with regularization "\
            f"of type '{self._regu_type}'."

    @classmethod
    def from_frequency_range(cls, frequency_range, beta=1, target=None):
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
        instance = cls(regu_type = "frequency range")

        if len(frequency_range) < 2:
            raise ValueError(
                "The frequency range needs to specify lower and upper limits.")

        if target and not isinstance(target, pf.Signal):
            raise ValueError(
                "Target function must be a pyfar.Signal object.")

        instance._frequency_range = np.asarray(frequency_range)
        instance._target = target
        instance._beta = beta
        instance._regu_type = "frequency range"

        return instance

    @classmethod
    def from_signal(cls, regularization, beta=1, target=None):
        """
        Regularization from signal.

        Parameters
        ----------
        regularization : pf.Signal
            Regularization as pyfar Signal
        """
        if not isinstance(regularization, pf.Signal):
            raise ValueError(
                "Regularization must be a pyfar.Signal object.")

        if target and not isinstance(target, pf.Signal):
            raise ValueError(
                "Target function must be a pyfar.Signal object.")

        instance = cls(regu_type = "signal")
        instance._regu = regularization
        instance._target = target
        instance._beta = beta

        return instance

    def invert(self, signal):
        """Method to get inverse of signal"""

        # get regularization factors
        regu = self.get_regularization(signal)
        regu_final = regu.freq

        data = self._signal.freq

        # calculate inverse filter
        inverse = self._signal.copy()
        if self._target:
            # Norcross 2006 eq. 2.13
            inverse.freq = np.conj(data) * regu_final / (np.conj(data) * data)
        else:
            # normalize to maximum of signal's magnitude spectrum
            regu_final *= np.max(np.abs(data)**2)
            inverse.freq = np.conj(data) / (np.conj(data) * data + regu_final)

        return inverse

    def get_regularization(self, signal):
        """
        Method to get the regularization factors based on regularization type.

        Parameters
        ----------
        signal : pf.Signal, optional
            Signal object to get the regularization factors from. Depending on
            the regularization type, a signal must be passed.
        """

        # zero pad, if target function is given
        if self._target and self._target.n_samples != signal.n_samples:
            n_samples = max(signal.n_samples, self._target.n_samples)
            signal = pf.dsp.pad_zeros(signal, (n_samples-signal.n_samples))
            self._target = pf.dsp.pad_zeros(self._target,
                                            (n_samples-self._target.n_samples))

        # Assign signal to class attribute, so further processing methods can
        # use zero padded signal (e.g. invert())
        self._signal = signal

        # Call private method to get regularization factors
        if self._regu_type == "frequency range":
            regu = self._get_regularization_from_frequency_range(signal)
        elif self._regu_type == "signal":
            regu = self._get_regularization_from_regularization_final(signal)

        if self._target:
            regu = self._get_regularization_with_target(signal, regu)

        return regu

    def _get_regularization_with_target(self, signal, regu):
        """Get regularization using a target function"""
        # calculate target function regularization (Norcross 2006)
        regu_abs_sq = regu.freq * np.conj(regu.freq)
        sig_avs_sq = signal.freq * np.conj(signal.freq)

        A_eq = 1 / (1 + self._beta * (regu_abs_sq / sig_avs_sq))
        A = A_eq * np.abs(self._target.freq)

        return pf.FrequencyData(A, signal.frequencies)


    def _get_regularization_from_frequency_range(self, signal):
        """Get regularization factors from frequency range."""

        if not isinstance(signal, pf.Signal):
            raise TypeError(f"Regularization of type '{self._regu_type}' "
                         "requires an input signal of type pyfar.Signal.")

        # arrange regularization factors for inside and outside frequency range
        regu_inside = np.zeros(signal.n_bins, dtype=np.double)
        regu_outside = np.ones(signal.n_bins, dtype=np.double) * self._beta

        # index for crossfade at lower frequency limit
        idx_xfade_lower = signal.find_nearest_frequency(
            [self._frequency_range[0]/np.sqrt(2), self._frequency_range[0]])

        regu_final = _cross_fade(regu_outside, regu_inside, idx_xfade_lower)

        # index for crossfade at upper frequency limit
        if self._frequency_range[1] < signal.sampling_rate/2:
            idx_xfade_upper = signal.find_nearest_frequency([
                self._frequency_range[1],
                np.min([self._frequency_range[1]*np.sqrt(2),
                        signal.sampling_rate/2])])

            # crossfade regularization factors at frequency limits
            regu_final = _cross_fade(regu_final, regu_outside, idx_xfade_upper)

        if self._target is None:
            # control amount of regularization using beta
            regu_final *= self._beta

        return pf.FrequencyData(regu_final, signal.frequencies)

    def _get_regularization_from_regularization_final(self, signal):
        """
        Get regularization from final regularization factors.
        """

        if not isinstance(signal, pf.Signal):
            raise TypeError(f"Regularization of type '{self._regu_type}' "
                            "requires an input signal of type pyfar.Signal.")

        if signal.n_bins != len(self._regu_final):
            raise ValueError(
                "The number of bins in the signal and the regularization "
                "factors must be equal.")
        return pf.FrequencyData(self._regu_final, signal.frequencies)

