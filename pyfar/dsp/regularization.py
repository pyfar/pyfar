"""
Documentation for regularization class.
"""

import pyfar as pf
from pyfar.dsp.dsp import _cross_fade
import numpy as np

class Regularization():
    r"""Class for frequency dependent regularization and inversion.

    Regularization is used in inverse filtering methods to limit the gain
    applied by an inverse filter. This, for example, is nescessary to
    avoid extreme amplification caused by notches and peaks in the original
    signal's spectrum or by the limit of a system's bandwidth.

    The frequency dependent regularization :math:`\epsilon(f)` can be defined
    using one of the class methods. Note that the resulting regularization
    function is adjusted to the quadratic maximum of the given signal.
    The regularization is then applied when calculating the inverse of a
    signal as [1]_, [2]_:

    .. math::

        S^{-1}(f) = \frac{S^*(f)}{S^*(f)S(f) + \beta * \epsilon(f)},

    with :math:`S(f)` being the input signal's Fourier transform,
    :math:`S^*(f)` the complex conjugate, and :math:`\beta` a parameter to
    control the influence of the regularization function on the inversion.

    Further, the class methods offer the possibility of passing an arbitrary
    target function :math:`D(f)`. This function can be used to manipulate
    the frequency response of the inversion. Therefore the chosen
    regularization :math:`\epsilon(f)` is first converted into a target
    function :math:`A_{eq}(f)` with the same effect as the regularization [3]_:

    .. math::

        A_{eq}(f)=\frac{1}{1+\beta \frac{|\epsilon(f)|^2}{|S(f)|^2}},

    and then magnitude multiplied with the passed target :math:`D(f)`:

    .. math::

        A(f)=A_{eq}(f)*|D(f)|.

    Finally the inverse signal is calculated as [3]_:

    .. math::

        S^{-1}(f)=\frac{S^*(f) A(f)}{|S(f)|^2}.

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
        >>> Regu = pf.dsp.Regularization.from_frequency_range([20, 15e3])
        >>> # create linear sweep and invert it:
        >>> sweep = pf.signals.linear_sweep_time(1000, (20, 20e3))
        >>> inv = Regu.invert(sweep)
        >>>
        >>> # design bandpass target function
        >>> bp = pf.dsp.filter.butterworth(None, 4, (20, 15e3), 'bandpass', 44.1e3)
        >>> target = bp.process(pf.signals.impulse(1000))
        >>> # get regularization object with target function
        >>> Regu_target = pf.dsp.Regularization.from_frequency_range([20, 15e3], target=target)
        >>> # invert signal
        >>> inv_target = Regu_target.invert(sweep)
        >>>
        >>> # plot results
        >>> pf.plot.freq(sweep, label="Original signal")
        >>> pf.plot.freq(inv, label="Regularized inversion")
        >>> pf.plot.freq(inv_target, label="Regularized inversion with target")
        >>> plt.legend()

    References
    ----------
    .. [1]  O. Kirkeby and P. A. Nelson, “Digital Filter Design for Inversion
            Problems in Sound Reproduction,” J. Audio Eng. Soc., vol. 47,
            no. 7, p. 13, 1999.

    .. [2]  P. C. Hansen, Rank-deficient and discrete ill-posed problems:
            numerical aspects of linear inversion. Philadelphia: SIAM, 1998.

    .. [3]  S. G. Norcross, M. Bouchard, and G. A. Soulodre, “Inverse Filtering
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
    def from_frequency_range(cls, frequency_range, beta=1, target=None):
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
        beta : float, optional
            Beta parameter to control the amount of regularization.
        target : pf.Signal, optional
            Target function for the regularization.
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

        return instance

    @classmethod
    def from_signal(cls, regularization, beta=1, target=None):
        """
        Regularization from signal.
        Regularization factors passed as pf.Signal. The factors have to
        match the number of bins of the signal to be inverted.

        Parameters
        ----------
        regularization : pf.Signal
            Regularization as pyfar Signal.
        beta : float, optional
            Beta parameter to control the amount of regularization.
        target : pf.Signal, optional
            Target function for the regularization.
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
        """Invert the spectrum of a signal applying frequency dependent
        regularization.

        Parameters
        ----------
        signal : pf.Signal
            Signal to be inverted.

        Returns
        -------
        inverse : pf.Signal
            Resulting signal after regularized inversion.
        """

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
        Method to get the frequency dependent regularization.

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
            regu = self._get_regularization_from_regularization_signal(signal)

        if self._target:
            regu = self._get_regularization_from_target(signal, regu)

        return regu

    def _get_regularization_from_target(self, signal, regu):
        """Get regularization using a target function"""
        # scale regularization to signal's magnitude spectrum
        regu_scaled = regu.freq * np.max(np.abs(signal.freq))
        # calculate target function from regularization (Norcross 2006)
        regu_abs_sq = regu_scaled * np.conj(regu_scaled)
        sig_abs_sq = signal.freq * np.conj(signal.freq)

        A_eq = 1 / (1 + self._beta * (regu_abs_sq / sig_abs_sq))

        # Apply passed target function by magnitude multiplication

        A = A_eq * np.abs(self._target.freq)

        return pf.FrequencyData(A, signal.frequencies)


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

        if self._target is None:
            # control amount of regularization using beta
            regu_final *= self._beta

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

