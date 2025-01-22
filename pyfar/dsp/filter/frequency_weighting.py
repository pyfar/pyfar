"""
Frequency weighting (A, C) according to IEC/DIN-EN 61672-1.
"""

# %%
from collections.abc import Callable

import pyfar as pf
import scipy.signal as sps
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt


# constants for the formulas
F_1 = 20.598997057618316
F_2 = 107.65264864304629
F_3 = 737.8622307362901
F_4 = 12194.217147998012
A_1000 = -2
C_1000 = -0.062
A_WEIGHTING_ZEROS = [0, 0, 0, 0]
A_WEIGHTING_POLES = [F_1, F_1, F_4, F_4, F_2, F_3]
C_WEIGHTING_ZEROS = [0, 0]
C_WEIGHTING_POLES = [F_1, F_1, F_4, F_4]


def calculate_A_weighted_level(f):
    """
    Calculates the level correction in dB of a frequency component
    when using the A weighting.
    """
    bracket_term = (F_4**2 * f**4) / ((f**2 + F_1**2) * (f**2 + F_2**2)
                                      ** 0.5 * (f**2 + F_3**2)**0.5
                                      * (f**2 + F_4**2))
    return 10 * np.log10(bracket_term**2) - A_1000


def calculate_C_weighted_level(f):
    """
    Calculates the level correction in dB of a frequency component
    when using the C weighting.
    """
    bracket_term = (F_4**2 * f**2) / ((f**2 + F_1**2) * (f**2 + F_4**2))
    return 10 * np.log10(bracket_term**2) - C_1000


def calculate_weighted_level(weighting: str,
                             frequency: float | np.ndarray,
                             ) -> float | np.ndarray:
    """
    Calculates the level correction in dB of a frequency component
    when using the A or C weighting.
    """
    if weighting == "A":
        return calculate_A_weighted_level(frequency)
    elif weighting == "C":
        return calculate_C_weighted_level(frequency)
    else:
        raise ValueError("Unrecognized weighting:", weighting)


def _zpk_from_analog(weighting: str, fs: float,
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Transforms the analog (s-plane) A or C weighting filter as specified in
    IEC/DIN-EN 61672-1 to a digital filter (z-plane) via bilinear transform
    and returns it as zeros poles and gain.
    The resulting filter is heavily warped near the Nyquist frequency.
    """
    if weighting == "A":
        zeros, poles = (A_WEIGHTING_ZEROS, A_WEIGHTING_POLES)
    elif weighting == "C":
        zeros, poles = (C_WEIGHTING_ZEROS, C_WEIGHTING_POLES)
    else:
        raise ValueError("Unrecognized weighting:", weighting)

    poles_angular = np.array(poles) * -2 * np.pi
    return sps.bilinear_zpk(zeros, poles_angular, 1, fs)


def design_frequency_weighting_filter(sampling_rate: float,
                                      target_weighting="A",
                                      n_frequencies=100,
                                      error_weighting: Callable[[
                                          np.ndarray], np.ndarray] = None,
                                      **kwargs,
                                      ) -> pf.FilterSOS:
    """
    Creates an efficient SOS filter approximating the A or C weighting defined
    in IEC/DIN-EN 61672-1.
    When using default parameters, the returned filter is compliant with a
    class 1 sound level meter as described in the norm for all tested
    sampling rates (see Notes).
    This function will run a least-squares algorithm to iteratively approach
    the target weighting curve.
    Therefore, it may be much faster to create the filter once and reuse that
    filter when repeatedly weighting audio data of identical sampling rates.

    Parameters
    ----------
    sampling_rate: float
        The sampling rate of the returned filter.

    target_weighting: str, optional
        Specifies which frequency weighting curve to approximate.
        Must be either "A" or "C". The default is "A".

    n_frequencies: int, optional
        At how many frequencies to evaluate the filter coefficients during
        optimization. Less frequencies means faster iterations, but
        potentially worse results. The evaluation frequencies are
        logarithmically spaced between 10 Hz and the Nyquist frequency.

    error_weighting: callable
        A function that can be used to emphasize the approximation errors in
        specific frequency ranges. The function should take a float array as
        argument, specifying the normalized frequencies between 0 and 1
        (where 1 is the Nyquist frequency) to weight, and returns a float
        array as output containing the weights.
        The default value is None, in which case the errors of all
        (logarithmically spaced) frequencies are equally weighted. This
        usually leds to larger errors in the higher frequencies. By passing
        a function that emphasizes high frequencies, it is possible to reduce
        this effect and get a filter potentially closer to the target curve.
        Example: `error_weighting=lambda nf: 100**nf`. This example often
        leads to better results for typical sampling rates, but much worse
        for very high rates.

    **kwargs: dict
        Keyword args that are passed to the
        scipy.optimize.least_squares() call.

    Returns
    -------
    filter: FilterSOS
        The frequency weighting filter in second order sections format.
        If weighting is 'A' the filter order will be 6. 'C' weighting will
        return a filter of order 4.

    Notes
    -----
    Since this function performs an iterative approximation, results may not
    be perfect depending on the input parameters. With the default parameters,
    the filter will be at least class 1 compliant for the sampling rates
    48 kHz, 44.1 kHz, 16 kHz as well as these sampling rates multiplied by
    2, 4, 8, 0.5, 0.25 and 0.125 each.
    With non-default parameters and/or other sampling rates, the returned
    filter may not comply with class 1 requirements,
    in which case a warning will be printed.
    """

    if target_weighting not in ["A", "C"]:
        raise ValueError(f"Unrecognized weighting: '{target_weighting}'.",
                         " Only 'A' and 'C' are implemented")

    # Build the initial guess x0 (the variables to optimize) from
    # zeros, poles and gain. Use the poles from the bilinear filter,
    # since they are not too far from the approximation.
    _z, p0, _k = _zpk_from_analog(target_weighting, sampling_rate)
    # The zeros at (1, 0) (=0 Hz) are fixed. The remaining two are
    # wrong from the bilinear transform, so we optimize them starting from 0.
    z0 = [0, 0]
    k0 = 1
    x0 = np.concat([z0, p0, [k0]])

    def x2zpk(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Reobtain z, p and k from x (the 1-D array of optimizer inputs).
        Adds back the fixed zeros at 0 Hz that are excluded from optimization.
        """
        fixed_zeros = [1, 1, 1, 1] if target_weighting == "A" else [1, 1]
        zeros = np.concat([fixed_zeros, x[0:2]])
        return (zeros, x[2:-1], x[-1])

    # build the cost function
    frequencies = np.logspace(1, np.log10(sampling_rate/2), n_frequencies)
    target_levels = calculate_weighted_level(target_weighting, frequencies)

    def compute_residuals(x: np.ndarray) -> np.ndarray:
        """
        Lets least_squares() compute how close the current approximation is to
        the target. Treats x (z, p and k) as real values, limiting them to
        the real axis.
        """
        z, p, k = x2zpk(x)
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
    z, p, k = x2zpk(optimization_result.x)
    sos = sps.zpk2sos(z, p, k)
    weighting_filter = pf.FilterSOS([sos], sampling_rate)

    # ensure the filter is correct
    is_class_1, max_err, mean_err = _check_filter(
        weighting_filter, target_weighting)
    if not is_class_1:
        print(f"Warning: The generated {target_weighting} weighting filter is"
              + "not class 1 compliant; "
              + f"Max error: {max_err:.2} dB, mean error: {mean_err:.2} dB.")
    return weighting_filter


def _get_error_margins(f: np.ndarray):
    """
    Returns the upper and lower error margin in dB for every frequency in the
    input array for the A and C weighting according to IEC/DIN-EN 61672-1.
    """
    upper = np.zeros_like(f)
    lower = np.zeros_like(f)

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


def _check_filter(filt: pf.FilterSOS, weighting,
                  ) -> tuple[bool, float, float]:
    """
    Checks whether the filter is class 1 compliant according to the norm and
    provides simple error statistics.
    """
    test_freqs = np.logspace(1, np.log10(filt.sampling_rate/2), 1e5)

    # calculate the magnitude differences between the filter response
    # and the target weighting
    z, p, k = sps.sos2zpk(filt.coefficients[0])
    _, freq_resp = sps.freqz_zpk(z, p, k, test_freqs, fs=filt.sampling_rate)
    mags = np.abs(freq_resp)
    mags_dB = 10 * np.log10(mags**2)
    mag_diffs = mags_dB - calculate_weighted_level(weighting, test_freqs)

    upper, lower = _get_error_margins(test_freqs)
    is_class_1 = np.all(mag_diffs < upper) and np.all(mag_diffs >= lower)
    max_diff = np.max(np.abs(mag_diffs))
    mean_diff = np.mean(np.abs(mag_diffs))
    return is_class_1, max_diff, mean_diff


###############################################################
# simple test to check whether the filters are compliant
# to be improved later
# %%
def _test_design_function():
    base_rates = np.array([48000, 44100, 16000])
    test_rates = np.array([
        base_rates,
        base_rates / 2, base_rates * 2,
        base_rates / 4, base_rates * 4,
        base_rates / 8, base_rates * 8,
    ]).flatten().tolist()

    for weighting in ["A", "C"]:
        for fs in test_rates:
            filt = design_frequency_weighting_filter(fs, weighting)
            is_class_1, _, _ = _check_filter(filt, weighting)
            print(f"{int(fs)}, {weighting}: "
                  + f"{"ok" if is_class_1 else "!!!"}")


_test_design_function()


#############################################################################
# manual testing section that allows easy inspection of the
# filter response through plots.
# best used with Jupyter.
# to be deleted after we figured out how to improve the automated tests
# %%
fs = 48000
weighting = "A"
kwargs = {}


def _err_wgt(f):
    return 100**f


sos_filter = design_frequency_weighting_filter(fs, weighting,
                                               error_weighting=_err_wgt)
dirac = pf.signals.impulse(fs, sampling_rate=fs)
response = sos_filter.process(dirac)

plot_frequencies = np.logspace(1, np.log10(fs/2), 100000)
z, p, k = sps.sos2zpk(sos_filter.coefficients[0])
_, h = sps.freqz_zpk(z, p, k, plot_frequencies, fs=fs)
h_db = 10 * np.log10(abs(h) ** 2)
diffs = h_db - calculate_weighted_level(weighting, plot_frequencies)

# plot target curve vs filter response
target_curve = calculate_weighted_level(weighting, plot_frequencies)
plt.plot(plot_frequencies, target_curve, label="target",
         linestyle="dashdot", c="r")
pf.plot.freq(response, label="impulse response", c="c", alpha=0.8)
plt.xlim(10, fs/2)
plt.legend()
plt.show()


def _plot_differences(frequencies, differences):
    plt.plot([0, fs/2], [0, 0], c="gray")
    plt.plot(frequencies, differences)
    plt.title("difference filter vs target in dB")
    plt.grid()
    plt.ylim(-1, 1)
    plt.semilogx()
    plt.xlim(10, fs/2)
    plt.show()


_plot_differences(plot_frequencies, diffs)
print(f"max. diff: {np.max(np.abs(diffs)):.2} dB")
print(f"mean diff: {np.mean(np.abs(diffs)):.2} dB")
print(_check_filter(sos_filter, weighting)[0])
