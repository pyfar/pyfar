import numpy as np
import pyfar as pf
import scipy.ndimage


def _check_signal_type(signal):
    if not isinstance(signal, pf.Signal):
        raise TypeError("'signal' parameter must be a pyfar.Signal")
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
    error_message = "Number of octave band fractions must be " \
                    "a positive integer or None"
    if num_octave_band_fractions is None:
        return signal
    elif not isinstance(num_octave_band_fractions, int):
        raise TypeError(error_message)
    elif num_octave_band_fractions <= 0:
        raise ValueError(error_message)
    else:
        return pf.dsp.filter.fractional_octave_bands(
            signal, num_octave_band_fractions)


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
    considers past values, same as numpy/scipy.signal/pyfar.dsp convolve.
    """
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
