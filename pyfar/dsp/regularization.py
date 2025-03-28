"""
Documentation for regularization class.
"""

import pyfar as pf
from pyfar.dsp.dsp import _cross_fade
from pyfar.dsp.fft import _n_samples_from_n_bins
import numpy as np
import numbers

class RegularizedSpectrumInversion():
    r"""Class for frequency-dependent regularized inversion.

    Regularized inversion limits the gain applied by an inverse filter.
    This can be useful to avoid extreme amplification when inverting a signal
    of limited bandwidth or containing notches in the spectrum.

    The inverse is computed as [#]_:

    .. math::
        :label: regularized_inversion

        S^{-1}(f) = \frac{S^*(f)}{S^*(f)S(f) + \beta |\epsilon(f)|^2} D(f)

    with :math:`f` being the frequency, :math:`S(f)` the spectrum of the signal
    to be inverted, :math:`^*` the complex conjugate,
    :math:`\epsilon(f)` the regularization, and :math:`\beta` a scalar to
    control the amount of regularization. :math:`D(f)` denotes an optional
    target function.

    The compensated system :math:`C = S(f)S^{-1}(f)` approaches the target
    function in magnitude and phase. In many applications, the target function
    should contain a delay to make sure that :math:`S^{-1}(f)` is causal. The
    larger :math:`\beta` and :math:`\epsilon(f)` are, the larger the deviation
    of :math:`C(f)` from the target :math:`D(f)`.

    The inversion is done in two steps:

    1. Define :math:`S(f)`, :math:`\epsilon(f)`, :math:`\beta`, and
       :math:`D(f)` using one of the ``from_...()`` methods listed below.
    2. Compute the inverse :math:`S(f)^{-1}` using :py:func:`~invert`

    The parameters that are defined in the first step are often iteratively
    adjusted. Examples are given in the documentation of the ``from_...()``
    methods.

    References
    ----------
    .. [#]  S. G. Norcross, M. Bouchard, and G. A. Soulodre, “Inverse Filtering
            Design Using a Minimal-Phase Target Function from Regularization,”
            Convention Paper 6929 Presented at the 121st Convention, 2006
            October 5–8, San Francisco, CA, USA.
    """

    def __init__(self) -> None:

        # throw error if object is instanced without classmethod
        raise RuntimeError("Regularization objects must be created "
                           "using one of the 'from_()' classmethods.")

    def __repr__(self):
        """String representation of RegularizedSpectrumInversion class."""
        return f"Regularization object with regularization "\
            f"of type '{self._regularization_type}'."

    @classmethod
    def from_frequency_range(cls, signal, frequency_range,
                             regularization_within=0, beta=1, target=None):
        r"""
        Regularization from a given frequency range.

        Defines a frequency range within which the regularization
        :math:`\epsilon(f)` is set to `regularization_within`.
        Outside the frequency range the regularization is
        :math:`\epsilon(f)=1` and can be controlled using the `beta`
        parameter.
        The regularization factors are cross-faded using a raised cosine window
        function with a width of :math:`\sqrt{2}f` above and below the given
        frequency range.

        Parameters
        ----------
        signal : Signal
            Signal to be inverted.
        frequency_range : array like
            Array like containing the lower and upper frequency limit in Hz.
        regularization_within: float, optional
            Set regularization inside frequency range. The default is `0`.
        beta : float, string, optional
            Beta parameter to control the scaling of the regularization as in
            equation :eq:`regularized_inversion`. Can be a

            ``numerical value``
                Usually between ``0`` and ``1``, with ``0`` being no
                regularization applied.
            ``'energy'``
                Normalize the regularization to match the signal's energy.

                :math:`\beta = \frac{\frac{1}{N}\sum_{k=0}^{N-1}|S[k]|^2}{\frac{1}{N}\sum_{k=0}^{N-1}|\epsilon[k]|^2}`
            ``'max'``
                Normalize the regularization :math:`\epsilon(f)` to the maximum
                magnitude of a given signal :math:`S(f)`.

                :math:`\beta = \frac{\max(|S(f)|)}{\max(|\epsilon(f)|)}`
            ``'mean'``
                Normalize the regularization :math:`\epsilon(f)` to the mean
                magnitude of a given signal :math:`S(f).`

                :math:`\beta = \frac{\text{mean}(|S(f)|)}{\text{mean}(|\epsilon(f)|)}`

            The default is ``1``.
        target : Signal, optional
            Target function for the regularization. The default ``None`` uses a
            zero-phase spectrum with an amplitude of 1 as target, equal to an
            impulse in the time domain.


        Examples
        --------
        Invert a sine sweep with limited bandwidth and apply maximum
        normalization to the regularization function.

        .. plot::

            >>> import pyfar as pf
            >>> import matplotlib.pyplot as plt
            ...
            >>> sweep = pf.signals.exponential_sweep_freq(
            ...     2**16, [50, 16e3], 1000, 10e3)
            ...
            >>> # Inversion
            >>> Inversion = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            >>>     sweep, [50, 16e3], beta='max')
            >>> inverted = Inversion.invert
            ...
            >>> # Obtain the scaled regularization function
            >>> regularization = Inversion.regularization * Inversion.beta_value
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
        """# noqa: E501
        instance = cls.__new__(cls)

        instance._frequency_range = np.asarray(frequency_range).flatten()

        if instance._frequency_range.shape != (2, ):
            raise ValueError(
                "The frequency range needs to specify lower and upper limits.")

        instance._regularization_type = "frequency range"
        instance._regularization_within = regularization_within
        instance.signal = signal
        instance.beta = beta
        instance.target = target

        return instance

    @classmethod
    def from_magnitude_spectrum(cls, signal, regularization, beta=1,
                                target=None):
        r"""
        Regularization from a given magnitude spectrum.

        Regularization passed as :py:class:`~pyfar.Signal`. The length and
        sampling rate of `regularization` must match the length and sampling
        rate of the signal to be inverted.

        Parameters
        ----------
        signal : Signal
            Signal to be inverted.
        regularization : Signal
            Signal defining the regularization :math:`\epsilon(f)` in
            ``regularization.freq``.
        beta : float, string, optional
            Beta parameter to control the scaling of the regularization as in
            :eq:`regularized_inversion`. Can be a

            ``numerical value``
                Usually between ``0`` and ``1``, with ``0`` being no
                regularization applied.
            ``'energy'``
                Normalize the regularization to match the signal's energy.

                :math:`\beta = \frac{\frac{1}{N}\sum_{k=0}^{N-1}|S[k]|^2}{\frac{1}{N}\sum_{k=0}^{N-1}|\epsilon[k]|^2}`
            ``'max'``
                Normalize the regularization :math:`\epsilon(f)` to the maximum
                magnitude of a given signal :math:`S(f)`.

                :math:`\beta = \frac{\max(|S(f)|)}{\max(|\epsilon(f)|)}`
            ``'mean'``
                Normalize the regularization :math:`\epsilon(f)` to the mean
                magnitude of a given signal :math:`S(f).`

                :math:`\beta = \frac{\text{mean}(|S(f)|)}{\text{mean}(|\epsilon(f)|)}`

            The default is ``1``.
        target : Signal, optional
            Target function for the regularization. The default ``None`` uses a
            zero-phase spectrum with an amplitude of 1 as target, equal to an
            impulse in the time domain.

        Examples
        --------
        Invert a headphone transfer function (HpTF), regularize the inversion at
        high frequencies, and use a band-pass as target function. Note that the
        equalized HpTF, which is obtained from a convolution of the HpTF with its
        inverse filter, approaches the target function in both, time and frequency.

        .. plot::

            >>> import pyfar as pf
            >>> import numpy as np
            >>>
            >>> # Headphone transfer function to be inverted
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
            ...     hptf, regularization, beta=.1, target=target)
            >>> inverted = Inversion.invert
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
        """# noqa: E501
        if not isinstance(regularization, pf.Signal):
            raise ValueError(
                "Regularization must be a pyfar.Signal object.")
        if signal.n_samples != regularization.n_samples:
            raise ValueError(
                "The number of samples in the signal and the regularization "
                "function differs but must be equal.")
        if signal.sampling_rate != regularization.sampling_rate:
            raise ValueError(
                "The sampling rate of the signal and the regularization "
                "function differs but must be equal.")

        instance = cls.__new__(cls)
        instance._regularization_signal = regularization
        instance._regularization_type = "magnitude spectrum"
        instance.signal = signal
        instance.beta = beta
        instance.target = target

        return instance

    @property
    def beta(self):
        r"""
        Scaling :math:`\beta` of regularization function :math:`\epsilon(f)`.

        This can be ``'energy'``, ``'max'``, ``'mean'``, or a number.

        To return the numeric :math:`\beta` value used in the inversion as in
        equation :eq:`regularized_inversion`, use the property
        :py:attr:`beta_value`.
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        """Set beta parameter for scaling of regularization function."""
        if not isinstance(beta, numbers.Number) \
            and beta not in ['energy', 'mean', 'max']:
            raise ValueError("Beta must be a scalar or 'energy', 'mean' or "
                             "'max'.")
        self._beta = beta
        return self._beta

    @property
    def beta_value(self):
        r"""Numeric :math:`\beta` value."""
        if self.beta == 'mean':
            self._beta_value = np.mean(np.abs(self.signal.freq)) / np.mean(
                np.abs(self.regularization.freq))
        elif self.beta == 'max':
            self._beta_value = np.max(np.abs(self.signal.freq)) / np.max(
                np.abs(self.regularization.freq))
        elif self.beta == 'energy':
            self._beta_value = \
                pf.dsp.energy(self.signal) / pf.dsp.energy(self.regularization)
        else:
            self._beta_value = self.beta
        return self._beta_value

    @property
    def regularization(self):
        r"""
        Regularization :math:`\epsilon(f)` without scaling by :math:`\beta`.
        """
        # Call private method to get regularization factors
        if self._regularization_type == "frequency range":
            self._regularization = \
                self._get_regularization_from_frequency_range(self.signal)
        elif self._regularization_type == "magnitude spectrum":
            self._regularization =  \
                self._get_regularization_from_regularization_signal(
                    self._regularization_signal)
        return self._regularization

    @property
    def signal(self):
        r"""Signal :math:`S(f)` to be inverted."""
        return self._signal

    @signal.setter
    def signal(self, signal):
        """Set signal to be inverted."""
        if not isinstance(signal, pf.Signal):
            raise TypeError("Regularization of type "
                            f"'{self._regularization_type}' requires an input"
                            " signal of type pyfar.Signal.")
        self._signal = signal

    @property
    def target(self):
        """Target function :math:`D(f)`."""
        return self._target

    @target.setter
    def target(self, target):
        """Set target function."""
        if target is not None and not isinstance(target, pf.Signal):
            raise ValueError(
                "Target function must be a pyfar.Signal object.")
        if target is not None and target.n_samples != self.signal.n_samples:
            raise ValueError(
                "The number of samples in the signal and the target function"
                " differs but must be equal.")
        if (target is not None and
            target.sampling_rate != self.signal.sampling_rate):
            raise ValueError(
                "The sampling rate of the signal and the target function"
                " differs but must be equal.")
        self._target = target

    @property
    def invert(self):
        """
        Invert signal using frequency-dependent regularized inversion.

        Returns
        -------
        inverse : Signal
            The inverted signal.
        """# noqa: E501
        data = self.signal.freq

        # calculate inverse filter
        inverse = self.signal.copy()
        inverse.freq = \
            np.conj(data) / (np.abs(data)**2 + self.beta_value *
                             np.abs(self.regularization.freq)**2)

        # Apply target function
        if self.target is not None:
            inverse.freq *= self.target.freq

        return inverse


    def _get_regularization_from_frequency_range(self, signal):
        """Get regularization factors from frequency range."""
        # arrange regularization factors for inside and outside frequency range
        regu_inside = \
            np.ones(signal.n_bins, dtype=np.double)*self._regularization_within
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

        # get number of samples according to the number of bins
        n_samples = _n_samples_from_n_bins(signal.n_bins, signal.complex)

        return pf.Signal(regu_final, signal.sampling_rate,
                         n_samples, domain='freq')


    def _get_regularization_from_regularization_signal(self, signal):
        """Get regularization from signal."""
        # get number of samples according to the number of bins
        n_samples = _n_samples_from_n_bins(signal.n_bins, signal.complex)

        return pf.Signal(np.abs(self._regularization_signal.freq),
                         signal.sampling_rate, n_samples, domain='freq')
