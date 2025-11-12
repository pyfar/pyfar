"""
Functions and helpers to create frequency weighting filters
according to IEC 61672-1.
"""
import warnings
from typing import Callable, Literal, Optional
import numpy as np
import pyfar as pf
import pyfar.constants as pfc
import scipy.signal as sps
import scipy.optimize as spo
from pyfar.constants.frequency_weighting import _F_1, _F_2, _F_3, _F_4


_A_WEIGHTING_ZEROS = [0, 0, 0, 0]
_A_WEIGHTING_POLES = [_F_1, _F_1, _F_4, _F_4, _F_2, _F_3]
_C_WEIGHTING_ZEROS = [0, 0]
_C_WEIGHTING_POLES = [_F_1, _F_1, _F_4, _F_4]


def frequency_weighting_filter(
        signal,
        target_weighting: Literal["A", "C"]="A",
        n_bins=100,
        error_weighting: Optional[Callable[[
            np.ndarray], np.ndarray]] = None,
        sampling_rate: Optional[float] = None,
        **kwargs,
        ):
    """
    Create and apply an A or C weighting filter.

    The created SOS filter approximates the A or C weighting defined
    in IEC 61672-1 [#]_.

    .. note::
        This function will run a least-squares algorithm to iteratively
        approach the target weighting curve. Therefore, it may be much
        faster to create the filter once and reuse that filter when
        repeatedly weighting audio data of identical sampling rates.

    .. note::
        When using default parameters for `n_frequencies` and `error_weighting`
        and no `kwargs`, the returned filter is compliant with a class 1 sound
        level meter as described in the standard for the sampling rates
        48 kHz, 44.1 kHz, 16 kHz as well as these sampling rates multiplied by
        2, 1/2, 4, 1/4, 8, and 1/8 each.
        For other arguments or sampling rates, the returned filter may not
        comply with class 1 requirements, in which case a warning will be
        printed.

    Parameters
    ----------
    signal : Signal, None
        The signal to be filtered. Pass ``None`` to create the filter without
        applying it.

    target_weighting: str, optional
        Specifies which frequency weighting curve to approximate.
        Must be either ``"A"`` or ``"C"``. The default is ``"A"``.

    n_bins: int, optional
        At how many frequencies to evaluate the filter coefficients during
        optimization. Less frequencies means faster iterations, but
        potentially worse results. The evaluation frequencies are
        logarithmically spaced between 10 Hz and the Nyquist frequency.
        The default is ``100``.

    error_weighting: callable
        A function that can be used to emphasize the approximation errors in
        specific frequency ranges. The function should take a float array as
        argument, specifying the normalized frequencies between 0 and 1
        (where 1 is the Nyquist frequency) to weight, and returns a float
        array as output containing the weights.
        By default the errors of all
        (logarithmically spaced) frequencies are equally weighted. This
        usually leads to larger errors for higher frequencies. By passing
        a function that emphasizes high frequencies, it is possible to reduce
        this effect and get a filter potentially closer to the target curve
        (see examples below). The default is ``None``.

    sampling_rate: float, conditionally optional
        The sampling rate of the returned filter. The default is ``None``.

    **kwargs: dict
        Keyword args that are passed to the
        :py:func:`scipy.optimize.least_squares` call.

    Returns
    -------
    signal : Signal
        The filtered signal. Only returned if ``sampling_rate = None``.
    filter: FilterSOS
        The frequency weighting filter as SOS Filter object.
        Only returned if ``signal = None``.
        If weighting is 'A' the filter order will be 6. 'C' weighting will
        return a filter of order 4.

    References
    ----------
    .. [#] International Electrotechnical Commission,
           "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
           Specifications", IEC, 2013.

    Examples
    --------
    Create and apply an A weighting filter to white noise.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> noise = pf.dsp.normalize(pf.signals.noise(10000), domain="freq")
        >>> weighting_filter = pf.dsp.filter.frequency_weighting_filter(
        ...     None, "A", sampling_rate=noise.sampling_rate)
        >>> weighted_noise = weighting_filter.process(noise)
        >>> pf.plot.freq(noise, label="unweighted noise")
        >>> pf.plot.freq(
        ...     weighting_filter.impulse_response(10000),
        ...     label="weighting filter")
        >>> pf.plot.freq(weighted_noise, label="A weighted noise")
        >>> plt.ylim(-45, 5)
        >>> plt.legend()

    Use the ``error_weighting`` parameter to improve filter accuracy
    near the nyquist frequency. Doing so sometimes leads to better results
    for typical sampling rates, but much worse for very high rates.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> impulse = pf.signals.impulse(1000, sampling_rate=48000)
        >>> no_weighting = pf.dsp.filter.frequency_weighting_filter(impulse)
        >>> with_weighting = pf.dsp.filter.frequency_weighting_filter(
        ...     impulse, error_weighting=lambda nf: 100**nf)
        >>> pf.plot.freq(no_weighting, c="b", label="without weighting")
        >>> pf.plot.freq(with_weighting, c="o", label="with weighting")
        >>> f = np.logspace(1, 4.4, 100)
        >>> ideal = pf.constants.frequency_weighting_curve("A", f)
        >>> plt.plot(f, ideal, linestyle="--", c="black", label="ideal")
        >>> plt.legend()
        >>> plt.ylim(-13, 3)
        >>> plt.xlim(4e3, 24e3)
        >>> plt.show()
    """
    # check input
    if (signal is None and sampling_rate is None) \
            or (signal is not None and sampling_rate is not None):
        raise ValueError('Either signal or sampling_rate must be none.')

    # obtain sampling rate from signal if not given
    if sampling_rate is None:
        sampling_rate: float = signal.sampling_rate

    sos = _design_frequency_weighting_filter(sampling_rate, target_weighting,
                                             n_bins, error_weighting,
                                             **kwargs)
    filt = pf.FilterSOS([sos], sampling_rate)
    filt.comment = (f"Frequency weighting SOS filter of order {filt.order} "
                    f"to approximate {target_weighting} weighting according "
                    "to IEC 61672-1")

    # apply or return the filter object
    if signal is None:
        return filt
    else:
        signal_filt = filt.process(signal)
        return signal_filt


def _design_frequency_weighting_filter(sampling_rate: float,
                                       target_weighting: Literal["A", "C"]="A",
                                       n_bins=100,
                                       error_weighting: Optional[Callable[[
                                           np.ndarray], np.ndarray]] = None,
                                       **kwargs,
                                       ) -> np.ndarray:
    """
    Designs SOS filter coefficients approximating the A or C weighting defined
    in IEC/DIN-EN 61672-1 for an arbitrary sampling rate.

    .. note::
        This function will run a least-squares algorithm to iteratively
        approach the target weighting curve. Therefore, it may be much faster
        to create the filter once and reuse that filter when repeatedly
        weighting audio data of identical sampling rates.

    .. note::
        When using default parameters for `n_frequencies` and `error_weighting`
        and no `kwargs`, the returned filter is compliant with a class 1 sound
        level meter as described in the standard for the sampling rates
        48 kHz, 44.1 kHz, 16 kHz as well as these sampling rates multiplied by
        2, 1/2, 4, 1/4, 8, and 1/8 each.
        For other arguments or sampling rates, the returned filter may not
        comply with class 1 requirements, in which case a warning will be
        printed.

    Parameters
    ----------
    sampling_rate: float
        The sampling rate of the returned filter.

    target_weighting: str, optional
        Specifies which frequency weighting curve to approximate.
        Must be either "A" or "C". The default is "A".

    n_bins: int, optional
        At how many frequencies to evaluate the filter coefficients during
        optimization. Less frequencies means faster iterations, but
        potentially worse results. The evaluation frequencies are
        logarithmically spaced between 10 Hz and the Nyquist frequency.
        The default is 100.

    error_weighting: callable
        A function that can be used to emphasize the approximation errors in
        specific frequency ranges. The function should take a float array as
        argument, specifying the normalized frequencies between 0 and 1
        (where 1 is the Nyquist frequency) to weight, and returns a float
        array as output containing the weights.
        The default value is None, in which case the errors of all
        (logarithmically spaced) frequencies are equally weighted. This
        usually leads to larger errors for higher frequencies. By passing
        a function that emphasizes high frequencies, it is possible to reduce
        this effect and get a filter potentially closer to the target curve.
        Example: `error_weighting=lambda nf: 100**nf`. This example often
        leads to better results for typical sampling rates, but much worse
        for very high rates. The default is `None`.

    **kwargs: dict
        Keyword args that are passed to the
        :py:func:`scipy.optimize.least_squares` call.

    Returns
    -------
    sos_coefficients: numpy.ndarray[float]
        The coefficients of the designed filter in scipy's sos format.
    """

    if target_weighting not in ["A", "C"]:
        raise ValueError("Allowed literals for weighting are 'A' and 'C'")

    # Build the initial guess x0 (the variables to optimize) from
    # zeros, poles and gain. Use the poles from the bilinear filter,
    # since they are not too far from the approximation.
    _z, p0, _k = _zpk_from_analog(target_weighting, sampling_rate)
    # The zeros at (1, 0) (=0 Hz) are fixed. The remaining two are wrong from
    # the bilinear transform, so we optimize them starting from 0
    # (the z-plane origin).
    z0 = [0, 0]
    k0 = 1
    x0 = np.concat([z0, p0, [k0]])

    # build the cost function (as a closure, so that passing it to
    # least_squares() is easier and cleaner)
    frequencies = np.logspace(1, np.log10(sampling_rate/2), n_bins)
    target_levels = pfc.frequency_weighting_curve(target_weighting,
                                                  frequencies)

    def compute_residuals(x: np.ndarray) -> np.ndarray:
        """
        Lets least_squares() compute how close the current approximation is to
        the target. Treats x (z, p and k) as real values, limiting them to
        the real axis.
        """
        z, p, k = _x2zpk(x, target_weighting)
        _, freq_response = sps.freqz_zpk(
            z, p, k, frequencies, fs=sampling_rate)
        approximated_levels = 20 * np.log10(abs(freq_response))

        # use level differences (in dB) as cost
        level_diffs = target_levels - approximated_levels
        if error_weighting:
            level_diffs *= error_weighting(frequencies / (sampling_rate/2))
        return level_diffs

    # Bounds are necessary to keep poles inside the unit circle,
    # ensuring the resulting filter is stable.
    kwargs.setdefault("bounds", (-1, 1))
    # The actual optimization happens here. This iteratively moves
    # zeros and poles along the real axis on the z-plane until
    # a local optimum is reached.
    optimization_result = spo.least_squares(compute_residuals, x0, **kwargs)

    # turn the result into a filter object
    z, p, k = _x2zpk(optimization_result.x, target_weighting)
    sos = sps.zpk2sos(z, p, k)

    # check if the filter is class 1 compliant
    is_class_1, max_err, mean_err = _check_filter(
        sampling_rate, sos, target_weighting)
    if not is_class_1:
        warnings.warn(f"The generated {target_weighting} weighting filter "
                      "is not class 1 compliant; Max error: "
                      f"{max_err:.2} dB, mean error: {mean_err:.2} dB.",
                      stacklevel=2)
    return sos



def _zpk_from_analog(weighting: str, fs: float,
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Transforms the analog (s-plane) A or C weighting filter as specified in
    IEC/DIN-EN 61672-1 to a digital filter (z-plane) via bilinear transform
    and returns it as zeros poles and gain.
    The resulting filter is heavily warped near the Nyquist frequency.
    """
    if weighting == "A":
        zeros, poles = (_A_WEIGHTING_ZEROS, _A_WEIGHTING_POLES)
    elif weighting == "C":
        zeros, poles = (_C_WEIGHTING_ZEROS, _C_WEIGHTING_POLES)
    else:
        raise ValueError("Unrecognized weighting:", weighting)

    poles_angular = np.array(poles) * -2 * np.pi
    return sps.bilinear_zpk(zeros, poles_angular, 1, fs)


def _x2zpk(x: np.ndarray,
           target_weighting: Literal["A", "C"],
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Reobtain z, p and k from x (the 1-D array of optimizer inputs).
    Adds back the fixed zeros at 0 Hz that are excluded from optimization.
    """
    fixed_zeros = [1, 1, 1, 1] if target_weighting == "A" else [1, 1]
    zeros = np.concat([fixed_zeros, x[0:2]])
    return (zeros, x[2:-1], x[-1])



def _get_error_margins(f: np.ndarray):
    """
    Returns the upper and lower error margin in dB for every frequency in the
    input array for the A and C weighting according to IEC/DIN-EN 61672-1.
    Margins are taken from table 3 in the standard.
    """
    upper = np.zeros_like(f)
    upper[f <= 10] = 3.0
    upper[(f > 10) & (f <= 12.5)] = 2.5
    upper[(f > 12.5) & (f <= 25)] = 2.0
    upper[(f > 25) & (f <= 31.5)] = 1.5
    upper[(f > 31.5) & (f <= 800)] = 1.0
    upper[(f > 800) & (f < 1250)] = 0.7
    upper[(f >= 1250) & (f < 5000)] = 1.0
    upper[(f >= 5000) & (f < 10e3)] = 1.5
    upper[(f >= 10e3) & (f < 16e3)] = 2.0
    upper[(f >= 16e3) & (f < 20e3)] = 2.5
    upper[f >= 20e3] = 3.0

    lower = np.zeros_like(f)
    lower[f <= 12.5] = -np.inf
    lower[(f > 12.5) & (f <= 16)] = -4.0
    lower[(f > 16) & (f <= 20)] = -2.0
    lower[(f > 20) & (f <= 31.5)] = -1.5
    lower[(f > 31.5) & (f <= 800)] = -1.0
    lower[(f > 800) & (f < 1250)] = -0.7
    lower[(f >= 1250) & (f < 5000)] = -1.0
    lower[(f >= 5000) & (f < 6300)] = -1.5
    lower[(f >= 6300) & (f < 8000)] = -2.0
    lower[(f >= 8000) & (f < 10e3)] = -2.5
    lower[(f >= 10e3) & (f < 12500)] = -3.0
    lower[(f >= 12500) & (f < 16e3)] = -5.0
    lower[(f >= 16e3) & (f < 20e3)] = -16.0
    lower[f >= 20e3] = -np.inf

    return upper, lower


def _check_filter(sampling_rate: float,
                  sos: np.ndarray,
                  weighting: Literal["A", "C"],
                  ) -> tuple[bool, float, float]:
    """
    Checks whether the filter is class 1 compliant according to the norm and
    provides simple error statistics.
    """
    test_freqs = np.logspace(1, np.log10(sampling_rate / 2), 100_000)

    # calculate the magnitude differences between the filter response
    # and the target weighting
    z, p, k = sps.sos2zpk(sos)
    _, freq_resp = sps.freqz_zpk(z, p, k, test_freqs, fs=sampling_rate)
    mags = np.abs(freq_resp)
    mags_dB = 10 * np.log10(mags**2)
    mags_target = pfc.frequency_weighting_curve(weighting, test_freqs)
    mag_diffs = mags_dB - mags_target

    upper, lower = _get_error_margins(test_freqs)
    is_class_1 = np.all(mag_diffs < upper) and np.all(mag_diffs >= lower)
    max_diff = np.max(np.abs(mag_diffs))
    mean_diff = np.mean(np.abs(mag_diffs))
    return is_class_1, max_diff, mean_diff


