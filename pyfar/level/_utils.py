"""Reusable helper functions for the individual processing steps shared
between many of the standardized level functions.
"""

import numpy as np
import pyfar as pf


def _check_signal_type(signal):
    """Raises a TypeError if not a pyfar.Signal, otherwise returns
    the signal with type hinting.
    """
    if not isinstance(signal, pf.Signal):
        raise TypeError("'signal' parameter must be a pyfar.Signal")
    signal: pf.Signal = signal
    return signal


def _apply_frequency_weighting(signal, frequency_weighting):
    """Applies frequency weighting to the signal
    or raises a ValueError if the weighting name is invalid.
    """
    if frequency_weighting == "Z":
        return signal
    elif frequency_weighting in ["A", "C"]:
        return pf.dsp.filter.frequency_weighting_filter(
            signal, frequency_weighting)
    else:
        raise ValueError("Frequency weighting must be 'A', 'C', or 'Z'")


def _apply_multi_band(signal, num_octave_band_fractions: int | None):
    """Applies fractional octave band filtering to the signal or does
    nothing if `num_octave_band_fractions` is None.
    """
    if num_octave_band_fractions is None:
        return signal
    return pf.dsp.filter.fractional_octave_bands(
        signal, num_octave_band_fractions)


def _energies_to_levels(energies: np.ndarray,
                        reference_pressure: float,
                        replace_zeros: bool = False):
    """Converts energies to levels in dB relative to the reference pressure.
    Raises an error if the reference pressure is zero.
    If `replace_zeros` is ``True``, zeros in the energies are replaced with
    the float epsilon to avoid ``-inf`` values and resulting numpy warnings.
    """
    if reference_pressure == 0:
        raise ValueError("Reference pressure must not be zero.")
    if replace_zeros:
        energies = np.where(
            energies == 0, np.finfo(energies.dtype).eps, energies)
    return 10 * np.log10(energies / reference_pressure**2)
