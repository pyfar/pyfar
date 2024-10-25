"""
Documentation for regularization class.
"""

import pyfar as pf
from pyfar.dsp.dsp import _cross_fade
import numpy as np

class RegularizedSpectrumInversion():
    r"""Class for frequency-dependent regularized inversion.

    Regularization is used in inverse filtering methods to limit the gain
    applied by an inverse filter. This, for example, is necessary to
    avoid extreme amplification to compensate a signal of limited bandwidth or
    containing notches in the spectrum.

    The inverse is computed as [#]_:

    .. math::

        S^{-1}(f) = \frac{S^*(f)}{S^*(f)S(f) + \beta |\epsilon(f)|^2} D(f)

    with :math:`S(f)` being the input signal's spectrum, :math:`(\cdot)^*` the
    complex conjugate, :math:`\epsilon(f)` the regularization, and
    :math:`\beta` a scalar to control the influence of the regularization on
    the inversion. :math:`D(f)` denotes an optional target function.

    The compensated system :math:`C(f) = S(f)S^{-1}(f)` approaches the target
    function in magnitude and phase. In many applications, the target function
    should contain a delay to make sure that :math:`S^{-1}(f)` is causal. The
    larger :math:`\beta` and :math:`\epsilon(f)` are, the larger the deviation
    of :math:`C(f)` from the target :math:`D(f)`.

    Examples
    --------

    Invert a sine sweep with limited bandwidth and apply maximum normalization
    to the regularization function.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        ...
        >>> sweep = pf.signals.exponential_sweep_freq(
        ...     2**16, [50, 16e3], 1000, 10e3)
        ...
        >>> # inversion
        >>> Inversion = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
        >>>     [50, 16e3])
        >>> inverted = Inversion.invert(sweep, normalize_regularization='max')
        ...
        >>> # obtain the normalized regularization function
        >>> regularization = Inversion.regularization(sweep) * \
        >>>     Inversion.normalization_factor(sweep, 'max')
        ...
        >>> pf.plot.use()
        >>> fig, axes = plt.subplots(2,1)
        >>> pf.plot.freq(sweep, ax=axes[0], label='sweep')
        >>> pf.plot.freq(regularization, ax=axes[0], label='regularization')
        >>> axes[0].axvline(50, color='k', linestyle='--',
        ...     label='frequency range')
        >>> axes[0].axvline(16e3, color='k', linestyle='--')
        >>> axes[0].legend(loc='lower center')
        ...
        >>> pf.plot.freq(inverted, ax=axes[1], color='p',
        ...     label='inverted with regulariztion')
        >>> pf.plot.freq(1 / (sweep+1e-10), ax=axes[1], color='y',
        ...     linestyle=':', label='inverted without regulariztion')
        >>> axes[1].axvline(50, color='k', linestyle='--',
        ...     label='frequency range')
        >>> axes[1].axvline(16e3, color='k', linestyle='--')
        >>> axes[1].set_ylim(-120, -20)
        >>> axes[1].legend(loc='lower center')

    Invert a headphone transfer function (HpTF), regularize the inversion at
    high frequencies, and use a band-pass as target function. Note that the
    equalized HpTF, which is obtained from a convolution of the HpTF with its
    inverse filter, approaches the target function in both, time and frequency.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>>
        >>> # Hedphone transfer function to be inverted
        >>> hptf = pf.signals.files.headphone_impulse_responses()[0, 0].flatten()
        >>> # Regularize inversion at high frequencies
        >>> regularization = pf.dsp.filter.low_shelf(
        ...     pf.signals.impulse(hptf.n_samples), 4e3, -20, 2, 'II')
        >>> # Use band-pass as target function
        >>> target = pf.dsp.filter.butterworth(
        ...     pf.signals.impulse(hptf.n_samples), 4, [100, 16e3], 'bandpass')
        >>> target = pf.dsp.time_shift(target, 125, 'cyclic')
        >>>
        >>> # Regulated inversion
        >>> Inversion = pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(
        ...     regularization)
        >>> inverted = Inversion.invert(hptf, beta=.1, target=target)
        >>>
        >>> # Plot the result
        >>> ax = pf.plot.time_freq(hptf, label='HpTF')
        >>> pf.plot.time_freq(inverted, label='HpTF inverted')
        >>> pf.plot.time_freq(hptf * inverted, label='Equalized HpTF')
        >>> pf.plot.time_freq(target, linestyle='--', label='Target')
        >>> pf.plot.freq(regularization, ax=ax[1], label='Regularization')
        >>> ax[0].set_xlim(0 , .005)
        >>> ax[1].set_ylim(-40 , 15)
        >>> ax[1].legend(loc='lower center', ncols=3)

    References
    ----------
    .. [#]  S. G. Norcross, M. Bouchard, and G. A. Soulodre, “Inverse Filtering
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
        Compute regularization from frequency range.

        Defines a frequency range within which the regularization factor is set
        to ``0``. Outside the frequency range the regularization factor is
        ``1`` and can be scaled using the `beta` parameter.
        The regularization factors are cross-faded using a raised cosine
        window function with a width of :math:`\sqrt{2}f` above and below the
        given frequency range.

        Parameters
        ----------
        frequency_range : array like
            Array like containing the lower and upper frequency limit in Hz.
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
        Compute regularization from magnitude spectrum.

        Regularization passed as :py:class:`~pyfar.Signal`. The length of
        `regularization` must match the length the signal to be inverted.

        Parameters
        ----------
        regularization : Signal
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
        r"""
        Invert the input using frequency-dependent regularized inversion.

        Parameters
        ----------
        signal : Signal
            Signal to be inverted.
        beta : float, optional
            Beta parameter to control the amount of regularization. The default
            is ``1``.
        target : Signal, optional
            Target function for the regularization. The default ``None`` uses
            constant spectrum with an amplitude of 1 as target.
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

            The default ``None`` does not apply any normalization.


        Returns
        -------
        inverse : Signal
            The inverted input signal.
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
        Compute the frequency dependent regularization :math:`\epsilon(f)`.

        Returns the regularization function without normalization.

        Parameters
        ----------
        signal : pf.Signal, optional
            Signal on which the regularization will be used on. This is
            necessary to assure that the regularization function has the same
            number of frequency bins as the signal.

        Returns
        -------
        regu : FrequencyData
            The frequency dependent regularization.
        """
        # Call private method to get regularization factors
        if self._regu_type == "frequency range":
            regu = self._get_regularization_from_frequency_range(signal)
        elif self._regu_type == "magnitude spectrum":
            regu = self._get_regularization_from_regularization_signal(signal)

        return regu


    def normalization_factor(self, signal, normalize_regularization):
        r"""
        Compute the normalization factor by which the regularization is
        scaled during inversion in :py:func:`~RegularizedSpectrumInversion.invert`.

        Parameters
        ----------
        signal : Signal
            Signal on which the regularization will be used on. The
            normalization factor is being calculated in relation to the signal.
        normalize_regularization : str
            Method of normalization.

            ``'max'``
                Normalize the regularization :math:`\epsilon` to the maximum
                magnitude of a given signal :math:`S(f)`. In this case the
                normalization factor is given by
                :math:`\frac{\max(|S(f)|)}{\max(|\epsilon(f)|)}`
            ``'mean'``
                Normalize the regularization :math:`\epsilon` to the mean
                magnitude of a given signal :math:`S(f)`. In this case the
                normalization factor is given by
                :math:`\frac{\text{mean}(|S(f)|)}{\text{mean}(|\epsilon(f)|)}`

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
        """Get regularization from signal."""

        if not isinstance(signal, pf.Signal):
            raise TypeError(f"Regularization of type '{self._regu_type}' "
                            "requires an input signal of type pyfar.Signal.")

        if signal.n_bins != self._regu.n_bins:
            raise ValueError(
                "The number of bins in the signal and the regularization "
                "factors must be equal.")

        return pf.FrequencyData(np.abs(self._regu.freq), signal.frequencies)

