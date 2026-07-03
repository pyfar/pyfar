"""Puclic utility function for working with levels."""

from typing import Literal

import numpy as np


def _combine_levels(levels, axis: int | None, operation: Literal["average", "sum"]):
    levels = np.asarray(levels)
    if np.issubdtype(levels.dtype, np.complexfloating):
        raise ValueError("Levels must be real-valued.")

    energies = 10 ** (levels / 10)
    if operation == "average":
        combined_energy = np.mean(energies, axis=axis)
    elif operation == "sum":
        combined_energy = np.sum(energies, axis=axis)
    else:
        raise ValueError("Operation must be 'average' or 'sum'.")
    combined_level = 10 * np.log10(combined_energy)
    return combined_level


def average_levels(levels, axis: int | None = None) -> np.ndarray | float:
    """Average levels energetically across channels or time.
    The input levels must be in dB and use the same reference pressure.

    Parameters
    ----------
    levels : array-like
        The levels in dB. Must be a numpy array or a type that can be
        converted to one.
    axis : int or None, optional
        Axis along which to average. If None (default), the average is
        taken over all values.

    Returns
    -------
    np.ndarray or scalar
        Average level(s) in dB. This is a scalar when averaging over all axes
        and an array otherwise.

    Examples
    --------
    Calculate the average level across two channels of a stereo signal.

    >>> signal = pf.signals.noise(48000, rms=(0.01, 0.02), seed=0)
    >>> level_per_channel = pf.level.equivalent_continuous_level(signal, "A")
    >>> average_level = pf.level.average_levels(level_per_channel)
    >>> print(level_per_channel)    # [51.57310831 57.61345662]
    >>> print(average_level)        # 55.56831429364634
    """
    return _combine_levels(levels, axis=axis, operation="average")


def sum_levels(levels, axis: int | None = None) -> np.ndarray | float | int:
    """Sum levels energetically across channels or time.
    The input levels must be in dB and use the same reference pressure.

    Parameters
    ----------
    levels : array-like
        The levels in dB. Must be a numpy array or a type that can be
        converted to one.
    axis : int or None, optional
        Axis along which to sum. If None (default), the sum is taken over
        all values.

    Returns
    -------
    np.ndarray or scalar
        Sum level(s) in dB. This is a scalar when summing over all axes
        and an array otherwise.

    Examples
    --------
    Calculate the total level from levels measured in octave bands.

    >>> import pyfar as pf
    >>> levels_in_bands = [70.2, 63.4, 55.2, 56.1, 67.8, 50.8]
    >>> total_level = pf.level.sum_levels(levels_in_bands)
    >>> print(total_level)  # 72.9099969445094
    """
    return _combine_levels(levels, axis=axis, operation="sum")
