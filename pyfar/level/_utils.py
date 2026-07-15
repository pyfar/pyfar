"""Reusable helper functions for the individual processing steps shared
between many of the standardized level functions.
"""

import numpy as np
import pyfar as pf
import scipy.ndimage


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


def _energies_to_levels(energies: np.ndarray, reference_pressure: float):
    """Converts energies to levels in dB relative to the reference pressure.
    """
    if reference_pressure == 0:
        raise ValueError("Reference pressure must not be zero.")
    return 10 * np.log10(energies / reference_pressure**2)


def _moving_average(array: np.ndarray,
                    window_size: int,
                    axis: int = -1,
                    cyclic: bool = False,
                    center_window: bool = False,
                    ):
    """Efficiently calculates the moving average of an array along
    an axis.
    If `cyclic` is ``True``, the array is treated as if it were periodic
    (like a cyclic convolution). If ``False`` (default), edges are
    zero-padded, leading to fade-like boundary effects.
    If `center_window` is ``True``, the window is centered on the current
    sample. If ``False`` (default), the window is causal, meaning it only
    considers past values, same as the `convolve` functions in `numpy`,
    `scipy.signal`, and `pyfar.dsp`.
    """
    if (not isinstance(window_size, int)):
        raise TypeError("Window size must be an integer.")
    if window_size < 1:
        raise ValueError("Window size must be a positive integer.")

    # If the input are integers, the outputs are too (rounded), which
    # we never want in floating point audio.
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64)

    if center_window:
        origin = 0  # 0 means the window is centered
    else:
        # For a causal window, the window needs to be shifted back
        if window_size % 2 == 1:
            origin = window_size // 2
        else:
            # since scipy.ndimage is designed for images, which normally
            # use odd window sizes, even sizes are finnicky...
            origin = window_size // 2 - 1

    mode = "wrap" if cyclic else "constant"

    # This is effectively the same as a convolution with a rectangular window,
    # but more efficient, especially for long windows.
    # Unlike convolve, this function has a built-in "wrap" mode, but no
    # "full" mode
    result = scipy.ndimage.uniform_filter1d(
        array, size=window_size, axis=axis, mode=mode, origin=origin)
    return result
