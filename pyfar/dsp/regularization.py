"""
Documentation for regularization class.
"""

import pyfar as pf
from pyfar.dsp.dsp import _cross_fade
import numpy as np

class RegularizedSpectrumInversion():
    r"""Class for frequency dependent regularization and inversion.

    Regularization is used in inverse filtering methods to limit the gain
    applied by an inverse filter. This, for example, is nescessary to
    avoid extreme amplification caused by notches and peaks in the original
    signal's spectrum or by the limit of a system's bandwidth.

    The frequency dependent regularization :math:`\epsilon(f)` can be defined
    using one of the class methods.
    The regularization is then applied when calculating the inverse of a
    signal as [1]_:

    .. math::

        S^{-1}(f) = \frac{S^*(f)}{S^*(f)S(f) + \beta * |\epsilon(f)|^2} * D(f)

    with :math:`S(f)` being the input signal's Fourier transform,
    :math:`S^*(f)` the complex conjugate, and :math:`\beta` a parameter to
    control the influence of the regularization function on the inversion.
    :math:`D(f)` denotes an optional arbitrary target function.
    This function can be used to manipulate the frequency response and phase
    of the inversion [1]_.

    The inversion of signals is handled using the class' :meth:`invert` method.

    Examples
    --------

    Regularized inversion from a frequency range. For better control over
    the slopes at the lower and upper frequency limits a bandpass target
    function is also passed.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # get regularization object
        >>> Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range([20, 15e3])
        >>> # create linear sweep and invert it:
        >>> sweep = pf.signals.linear_sweep_time(1000, (20, 20e3))
        >>> inv = Regu.invert(sweep)
        >>>
        >>> # design bandpass target function
        >>> bp = pf.dsp.filter.butterworth(None, 4, (20, 15e3), 'bandpass', 44.1e3)
        >>> target = bp.process(pf.signals.impulse(1000))
        >>> # get regularization object with target function
        >>> Regu_target = pf.dsp.RegularizedSpectrumInversion.from_frequency_range([20, 15e3])
        >>> # invert signal
        >>> inv_target = Regu_target.invert(sweep, target=target)
        >>>
        >>> # plot results
        >>> pf.plot.freq(sweep, label="Original signal")
        >>> pf.plot.freq(inv, label="Regularized inversion")
        >>> pf.plot.freq(inv_target, label="Regularized inversion with target")
        >>> plt.legend()

    References
    ----------
    .. [1]  S. G. Norcross, M. Bouchard, and G. A. Soulodre, “Inverse Filtering
            Design Using a Minimal-Phase Target Function from Regularization,”
            Convention Paper 6929 Presented at the 121st Convention, 2006
            October 5–8, San Francisco, CA, USA.
    """ # noqa: E501

    def __init__(self, regu_type: str="") -> None:

        self._regu_type = regu_type

        # throw error if object is instanced without classmethod
        if regu_type == "":
            raise RuntimeError("Regularization objects must be created "
                             "using a classmethod.")

    def __repr__(self):
        return f"Regularization object with regularization "\
            f"of type '{self._regu_type}'."

    @classmethod
    def from_frequency_range(cls, frequency_range):
        r"""
        Regularization from frequency range.
        Defines a frequency range within which the regularization factor is set
        to ``0``. Outside the frequency range the regularization factor is
        ``1`` and can be scaled using the ``beta`` parameter.
        The regularization factors are cross-faded using a raised cosine
        window function with a width of :math:`\sqrt{2}f` above and below the
        given frequency range.

        Parameters
        ----------
        frequency_range : array, tuple
            Tuple with two values for the lower and upper frequency limit.
        """
        instance = cls(regu_type = "frequency range")

        if len(frequency_range) != 2:
            raise ValueError(
                "The frequency range needs to specify lower and upper limits.")

        instance._frequency_range = np.asarray(frequency_range)

        return instance

    @classmethod
    def from_magnitude_spectrum(cls, regularization):
        """
        Regularization from magnitude spectrum.
        Regularization factors passed as :py:class:`~pyfar.Signal`. The factors
        have to match the number of bins of the signal to be inverted.

        Parameters
        ----------
        regularization : pf.Signal
            Regularization as pyfar Signal.
        """
        if not isinstance(regularization, pf.Signal):
            raise ValueError(
                "Regularization must be a pyfar.Signal object.")

        instance = cls(regu_type = "magnitude spectrum")
        instance._regu = regularization

        return instance

    def invert(self, signal, beta=1, target=None,
               normalize_regularization=None):
        r"""Invert the spectrum of a signal applying frequency dependent
        regularization.

        Parameters
        ----------
        signal : pf.Signal
            Signal to be inverted.
        beta : float, optional
            Beta parameter to control the amount of regularization.
        target : pf.Signal, optional
            Target function for the regularization.
        normalize_regularization : str
            Method of normalization.

            ``'max'``
                Normalize the regularization :math:`\epsilon(f)` to the maximum
                magnitude of a given signal :math:`S(f)`.

                :math:`Normalization Factor = \frac{\max(|S(f)|)}{\max(|\epsilon(f)|)}`
            ``'mean'``
                Normalize the regularization :math:`\epsilon(f)` to the mean
                magnitude of a given signal :math:`S(f)`.

                :math:`Normalization Factor = \frac{\text{mean}(|S(f)|)}{\text{mean}(|\epsilon(f)|)}`


        Returns
        -------
        inverse : pf.Signal
            Resulting signal after regularized inversion.
        """ # noqa: E501
        if target is not None and not isinstance(target, pf.Signal):
            raise ValueError(
                "Target function must be a pyfar.Signal object.")

        # zero pad, if target function is given and has different n_samples
        if target is not None and \
            target.n_samples != signal.n_samples:
            n_samples = max(signal.n_samples, target.n_samples)
            signal = pf.dsp.pad_zeros(signal, (n_samples-signal.n_samples))
            target = pf.dsp.pad_zeros(target, (n_samples-target.n_samples))

        # get regularization factors
        regu = self.regularization(signal)

        # Apply optional normalization
        if normalize_regularization is not None:
            normalization_factor_ = \
                self.normalization_factor(signal, normalize_regularization)
            regu.freq *= normalization_factor_

        data = signal.freq

        # calculate inverse filter
        inverse = signal.copy()
        inverse.freq = \
            np.conj(data) / (np.conj(data) * data + beta *
                             np.abs(regu.freq)**2)

        # Apply target function
        if target is not None:
            inverse.freq *= target.freq

        return inverse


    def regularization(self, signal):
        r"""
        Get the frequency dependent regularization factors :math:`\epsilon(f)`.

        Returns the regularization factors without any normalization apllied.

        Parameters
        ----------
        signal : pf.Signal, optional
            Signal on which the regularization will be used on. This is
            nescessary to assure that the regularization factors have the same
            number of bins as the signal.

        Returns
        -------
        regu : pf.FrequencyData
            Frequency dependent regularization as FrequencyData.
        """
        # Call private method to get regularization factors
        if self._regu_type == "frequency range":
            regu = self._get_regularization_from_frequency_range(signal)
        elif self._regu_type == "magnitude spectrum":
            regu = self._get_regularization_from_regularization_signal(signal)

        return regu


    def normalization_factor(self, signal, normalize_regularization):
        r"""Returns the normalization factor by which the regularization is
        scaled during inversion.

        Parameters
        ----------
        signal : pf.Signal
            Signal on which the regularization will be used on. The
            normalization is being calculated in relation to the signal.
        normalize_regularization : str
            Method of normalization.

            ``'max'``
                Normalize the regularization :math:`\epsilon` to the maximum
                magnitude of a given signal :math:`S(f)`.

                :math:`Normalization Factor = \frac{\max(|S(f)|)}{\max(|\epsilon(f)|)}`
            ``'mean'``
                Normalize the regularization :math:`\epsilon` to the mean
                magnitude of a given signal :math:`S(f)`.

                :math:`Normalization Factor = \frac{\text{mean}(|S(f)|)}{\text{mean}(|\epsilon(f)|)}`

        Returns
        -------
        normalization_factor : float
            Normalization factor by which the regularization is scaled.
        """ # noqa: E501
        regu = self.regularization(signal)

        if normalize_regularization == 'mean':
            self._normalization_factor = \
                np.mean(np.abs(signal.freq)) / np.mean(np.abs(regu.freq))
        if normalize_regularization == 'max':
            self._normalization_factor = \
                np.max(np.abs(signal.freq)) / np.max(np.abs(regu.freq))

        return self._normalization_factor

    def _get_regularization_from_frequency_range(self, signal):
        """Get regularization factors from frequency range."""

        if not isinstance(signal, pf.Signal):
            raise TypeError(f"Regularization of type '{self._regu_type}' "
                         "requires an input signal of type pyfar.Signal.")

        # arrange regularization factors for inside and outside frequency range
        regu_inside = np.zeros(signal.n_bins, dtype=np.double)
        regu_outside = np.ones(signal.n_bins, dtype=np.double)

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

        return pf.FrequencyData(regu_final, signal.frequencies)

    def _get_regularization_from_regularization_signal(self, signal):
        """
        Get regularization from final regularization factors.
        """

        if not isinstance(signal, pf.Signal):
            raise TypeError(f"Regularization of type '{self._regu_type}' "
                            "requires an input signal of type pyfar.Signal.")

        if signal.n_bins != self._regu.n_bins:
            raise ValueError(
                "The number of bins in the signal and the regularization "
                "factors must be equal.")

        return pf.FrequencyData(np.abs(self._regu.freq), signal.frequencies)

