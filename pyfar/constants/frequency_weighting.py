"""File containing all frequency weighting functions."""
from typing import Literal, Union
import numpy as np
import pyfar

# Constants for the weighting curve formulas.
# See appendix E in IEC 61672-1.
_F_1 = 20.598997057618316
_F_2 = 107.65264864304629
_F_3 = 737.8622307362901
_F_4 = 12194.217147998012
_A_1000 = -2
_C_1000 = -0.062


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
    weights: FrequencyData
        The weights in dB and their frequencies as a Frequency Data object.

    Examples
    --------
    Plot the weighting curves.

    .. plot::
        import pyfar as pf
        import matplotlib.pyplot as plt
        import numpy as np

        plot_frequencies = np.logspace(1, 4.5, 100, base=10)
        weights_A = pf.constants.frequency_weighting_curve(
            "A", plot_frequencies)
        weights_C = pf.constants.frequency_weighting_curve(
            "C", plot_frequencies)

        plt.plot(plot_frequencies, weights_A.freq[0], label="A weighting")
        plt.plot(plot_frequencies, weights_C.freq[0], label="C weighting")
        plt.semilogx()
        plt.xlabel("f in Hz")
        plt.ylabel("Weights in dB")
        ticks = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
        plt.xticks(ticks, ticks)
        plt.grid()
        plt.legend()
    """
    if weighting == "A":
        weights = _calculate_A_weighted_level(np.array(frequencies))
    elif weighting == "C":
        weights = _calculate_C_weighted_level(np.array(frequencies))
    else:
        raise ValueError("Allowed literals for weighting are 'A' and 'C'")

    comment = f"Level corrections for {weighting} weighting according to" \
               "IEC 61672-1"
    return pyfar.FrequencyData(weights, frequencies, comment)


# Constants for nominal band corrections, taken from table 3 in IEC 61672-1.
# This is somewhat redundant with dsp.filter.fractional_octaves, but
# implementing different standards. Here, IEC 61672-1 defines bands as low as
# 10 Hz, which is not part of fractional_octaves.
_NOMINAL_THIRDBAND_FREQUENCIES = np.array([
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
    400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500, 16000, 20000,
])
_THIRDBAND_WEIGHTINGS_A = np.array([
    -70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5,
    -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8,
    0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1,
    -2.5, -4.3, -6.6, -9.3,
])
_THIRDBAND_WEIGHTINGS_C = np.array([
    -14.3, -11.2, -8.5, -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5,
    -0.3, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.4,
    -6.2, -8.5, -11.2,
])


def frequency_weighting_band_corrections(
    weighting: Literal["A", "C"],
    bands: Literal["third", "octave"],
    frequency_range: tuple[float, float],
):
    """
    Returns the A or C frequency weighting band corrections as specified in
    IEC 62672-1 for the given frequency range.

    Parameters
    ----------
    weighting: str ('A' or 'C')
        Which frequency weighting type to use.

    bands: str ('octave' or 'third')
        Whether to use octave bands or third bands.

    frequency_range: (float, float)
        A range of what nominal center frequencies to include. The lowest
        defined center frequency is 10 Hz and the highest is 20 kHz.

    Returns
    -------
    weights_with_nominals: FrequencyData
        The correction values in dB for the specific frequency weighting
        over their nominal frequencies as a Frequency Data object.

    Examples
    --------
    Plot the band correction weights.

    .. plot::

        import pyfar as pf
        import matplotlib.pyplot as plt

        frequency_range = (10, 20000)
        weights_A = pf.constants.frequency_weighting_band_corrections(
            "A", "third", frequency_range)
        weights_C = pf.constants.frequency_weighting_band_corrections(
            "C", "octave", frequency_range)

        # plotting
        plt.plot(weights_A.frequencies, weights_A.freq[0], "--",
                c=(0.5, 0.5, 0.5, 0.5))
        plt.plot(weights_A.frequencies, weights_A.freq[0], "bo",
                label="A weighting in third bands")
        plt.plot(weights_C.frequencies, weights_C.freq[0], "--",
                c=(0.5, 0.5, 0.5, 0.5))
        plt.plot(weights_C.frequencies, weights_C.freq[0], "go",
                label="C weighting in octave bands")
        plt.legend()
        ticks = [10, 30, 100, 300, 1000, 3000, 10000]
        plt.semilogx()
        plt.xticks(ticks, ticks)
        plt.grid()
    """
    if weighting == "A":
        all_weights = _THIRDBAND_WEIGHTINGS_A
    elif weighting == "C":
        all_weights = _THIRDBAND_WEIGHTINGS_C
    else:
        raise ValueError("Allowed literals for weighting are 'A' and 'C'")

    if bands == "octave":
        nominals = _NOMINAL_THIRDBAND_FREQUENCIES[2::3]
        band_weights = all_weights[2::3]
    elif bands == "third":
        nominals =_NOMINAL_THIRDBAND_FREQUENCIES
        band_weights = all_weights
    else:
        raise ValueError("Allowed literals for bands are 'octave' and 'third'")

    mask = (nominals >= frequency_range[0]) & (nominals <= frequency_range[1])
    nominals_in_range = nominals[mask]
    weights_in_range = band_weights[mask]

    if len(nominals_in_range) == 0:
        raise ValueError("Frequency range must include at least one value " \
                         "between 10 and 20000")

    comment = f"Level corrections for the {weighting} weighting in {bands}" \
               "bands in dB and the bands' nominal center frequencies"
    return pyfar.FrequencyData(weights_in_range, nominals_in_range, comment)
