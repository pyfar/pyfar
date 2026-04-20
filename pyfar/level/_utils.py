import numpy as np
import pyfar as pf
import math
import scipy.ndimage

def _check_signal_type(signal):
    if not isinstance(signal, pf.Signal):
        raise ValueError("'signal' parameter must be a pyfar.Signal")
    signal: pf.Signal = signal
    return signal


def _apply_frequency_weighting(signal, frequency_weighting):
    if frequency_weighting == "Z":
        return signal
    elif frequency_weighting in ["A", "C"]:
        return pf.dsp.filter.frequency_weighting_filter(
            signal, frequency_weighting)
    else:
        raise ValueError("Frequency weighting must be 'A', 'C', or 'Z'")


def _apply_multi_band(signal, num_octave_band_fractions: int | None):
    if num_octave_band_fractions is None:
        return signal
    elif (isinstance(num_octave_band_fractions, int)
          and num_octave_band_fractions > 0):
        return pf.dsp.filter.fractional_octave_bands(
            signal, num_octave_band_fractions)
    else:
        raise ValueError("num_octave_band_fractions must be a positive "
                         "integer or None")


def _apply_oversampling(signal, oversampling: int | None):
    if oversampling is None:
        return signal
    elif isinstance(oversampling, int) and oversampling > 1:
        return pf.dsp.resample(signal,
                               signal.sampling_rate * oversampling,
                               "time")
    else:
        raise ValueError(
            "Oversampling must be an integer greater than 1 or None")


def _energies_to_levels(energies: np.ndarray, reference_pressure: float):
    return 10 * np.log10(energies / reference_pressure**2)


def _moving_average(array: np.ndarray,
                    window_size: int,
                    axis: int = -1,
                    cyclic: bool = False,
                    center_window: bool = False,
                    ):
    """Efficiently calculates the moving average of an array along
    an axis.
    If cyclic is True, the array is treated as if it were periodic
    (like a cyclic convolution). If False (default), edges are
    zero-padded, leading to fade-like boundary effects.
    If center_window is True, the window is centered on the current
    sample. If False (default), the window is causal, meaning it only
    considers past values.
    """
    if window_size < 1:
        raise ValueError("Window size must be a positive integer.")

    # since scipy.ndimage.uniform_filter1d is intended for images,
    # it assumes odd window sizes and therefore needs special handling
    # for even sizes.
    origin = 0 if center_window else (math.ceil(window_size/2)-1)
    mode = "wrap" if cyclic else "constant"

    # This is effectively the same as a convolution with a rectangular window,
    # but more efficient, especially for long windows.
    # Unlike convolve, this function has a built-in "wrap" mode, but no
    # "full" mode
    result = scipy.ndimage.uniform_filter1d(
        array, size=window_size, mode=mode, origin=origin, axis=axis)
    return result
