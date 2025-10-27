"""File containing all frequency weighting functions."""
from typing import Literal, Union
import numpy as np
import pyfar as pf

# Constants for the weighting curve formulas.
# See appendix E in IEC 61672-1.
_F_1 = 20.598997057618316
_F_2 = 107.65264864304629
_F_3 = 737.8622307362901
_F_4 = 12194.217147998012
_A_1000 = -2
_C_1000 = -0.062

# Constants for nominal band corrections, taken from table 3 in IEC 61672-1.
_NOMINAL_A_WEIGHTING_CORRECTIONS = {
    10: -70.4, 12.5: -63.4, 16: -56.7, 20: -50.5, 25: -44.7, 31.5: -39.4,
    40: -34.6, 50: -30.2, 63: -26.2, 80: -22.5, 100: -19.1, 125: -16.1,
    160: -13.4, 200: -10.9, 250: -8.6, 315: -6.6, 400: -4.8, 500: -3.2,
    630: -1.9, 800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2,
    2500: 1.3, 3150: 1.2, 4000: 1.0, 5000: 0.5, 6300: -0.1, 8000: -1.1,
    10000: -2.5, 12500: -4.3, 16000: -6.6, 20000: -9.3,
}
_NOMINAL_C_WEIGHTING_CORRECTIONS = {
    10: -14.3, 12.5: -11.2, 16: -8.5, 20: -6.2, 25: -4.4, 31.5: -3.0,
    40: -2.0, 50: -1.3, 63: -0.8, 80: -0.5, 100: -0.3, 125: -0.2,
    160: -0.1, 200: 0.0, 250: 0.0, 315: 0.0, 400: 0.0, 500: 0.0,
    630: 0.0, 800: 0.0, 1000: 0.0, 1250: 0.0, 1600: -0.1, 2000: -0.2,
    2500: -0.3, 3150: -0.5, 4000: -0.8, 5000: -1.3, 6300: -2.0, 8000: -3.0,
    10000: -4.4, 12500: -6.2, 16000: -8.5, 20000: -11.2,
}


def _calculate_A_weighted_level(f: Union[float, np.ndarray],
                                ) -> Union[float, np.ndarray]:
    """
    Calculates the level correction in dB of a frequency component
    when using the A weighting according to IEC 61672-1.
    """
    # formula (E.6) in the standard
    bracket_term = (_F_4**2 * f**4) / ((f**2 + _F_1**2) * (f**2 + _F_2**2)
                                      ** 0.5 * (f**2 + _F_3**2)**0.5
                                      * (f**2 + _F_4**2))
    return 10 * np.log10(bracket_term**2) - _A_1000


def _calculate_C_weighted_level(f: Union[float, np.ndarray],
                                ) -> Union[float, np.ndarray]:
    """
    Calculates the level correction in dB of a frequency component
    when using the C weighting according to IEC 61672-1.
    """
    # formula (E.1) in the standard
    bracket_term = (_F_4**2 * f**2) / ((f**2 + _F_1**2) * (f**2 + _F_4**2))
    return 10 * np.log10(bracket_term**2) - _C_1000


def frequency_weighting_curve(weighting: Literal["A", "C"],
                              frequencies: list[float]):
    """
    Calculates the level correction in dB of a frequency component when using
    the A or C weighting defined in IEC 61672-1.

    Parameters
    ----------
    weighting: str ('A' or 'C')
        Which frequency weighting to use.

    frequencies: array-like
        The frequencies at which to evaluate the weighting curve.

    Returns
    -------
    weights: numpy.ndarray[float]
        The weights in dB in the same shape as `frequencies`.

    References
    ----------
    .. [#] International Electrotechnical Commission,
           "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
           Specifications", IEC, 2013.

    Examples
    --------
    Plot the weighting curves.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> plot_frequencies = np.logspace(1, 4.5, 100, base=10)
        >>> weights_A = pf.constants.frequency_weighting_curve(
        ...     "A", plot_frequencies)
        >>> weights_C = pf.constants.frequency_weighting_curve(
        ...     "C", plot_frequencies)
        >>> plt.plot(plot_frequencies, weights_A, label="A weighting")
        >>> plt.plot(plot_frequencies, weights_C, label="C weighting")
        >>> plt.semilogx()
        >>> plt.xlabel("f in Hz")
        >>> plt.ylabel("Weights in dB")
        >>> ticks = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
        >>> plt.xticks(ticks, ticks)
        >>> plt.grid()
        >>> plt.legend()
    """
    if weighting == "A":
        weights = _calculate_A_weighted_level(np.array(frequencies))
    elif weighting == "C":
        weights = _calculate_C_weighted_level(np.array(frequencies))
    else:
        raise ValueError("Allowed literals for weighting are 'A' and 'C'")

    return weights


def frequency_weighting_band_corrections(
    weighting: Literal["A", "C"],
    num_fractions: Literal[1, 3],
    frequency_range: tuple[float, float],
):
    """
    Returns the A or C frequency weighting band corrections as specified in
    IEC 62672-1 for the given frequency range.

    Parameters
    ----------
    weighting: str ('A' or 'C')
        Which frequency weighting type to use.

    num_fractions: {1, 3}
        The number of octave fractions. ``1`` returns octave band
        corrections, ``3`` returns third-octave band corrections.

    frequency_range: (float, float)
        A range of what nominal center frequencies to include. The lowest
        defined center frequency is 10 Hz and the highest is 20 kHz.

    Returns
    -------
    nominal_frequencies: numpy.ndarray[float]
        The nominal center frequencies included in the given range.

    weights: numpy.ndarray[float]
        The correction values in dB for the specific frequency weighting.

    References
    ----------
    .. [#] International Electrotechnical Commission,
           "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
           Specifications", IEC, 2013.

    Examples
    --------
    Plot the band correction levels.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> f_range = (10, 20000)
        >>> nominals_third, weights_A = pf.constants.frequency_weighting_band_corrections(
        ...     "A", 3, f_range)
        >>> nominals_octave, weights_C = pf.constants.frequency_weighting_band_corrections(
        ...     "C", 1, f_range)
        >>> # plotting
        >>> plt.plot(nominals_third, weights_A, "--", c=(0.5, 0.5, 0.5, 0.5))
        >>> plt.plot(nominals_third, weights_A, "bo", label="A weighting in third bands")
        >>> plt.plot(nominals_octave, weights_C, "--", c=(0.5, 0.5, 0.5, 0.5))
        >>> plt.plot(nominals_octave, weights_C, "go", label="C weighting in octave bands")
        >>> plt.legend()
        >>> ticks = nominals_octave[::2].astype(int)
        >>> plt.semilogx()
        >>> plt.xticks(ticks, ticks)
        >>> plt.xlabel("f in Hz")
        >>> plt.ylabel("Corrections in dB")
        >>> plt.grid()
    """ # noqa: E501
    if weighting == "A":
        all_weights = _NOMINAL_A_WEIGHTING_CORRECTIONS
    elif weighting == "C":
        all_weights = _NOMINAL_C_WEIGHTING_CORRECTIONS
    else:
        raise ValueError("Allowed literals for weighting are 'A' and 'C'")

    nominals_in_range = pf.constants.fractional_octave_frequencies_nominal(
        num_fractions, frequency_range)
    weights_in_range = np.array([all_weights[n] for n in nominals_in_range])
    return (nominals_in_range, weights_in_range)
