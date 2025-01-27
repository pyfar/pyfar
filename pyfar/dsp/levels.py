"""
Functions used to calculate the levels of signals.
"""

import numpy as np
import scipy.signal as sps


def time_weighted_levels(signal, weighting: str,
                         reference=1) -> np.ndarray:
    """
    Calculates sound pressure levels with exponential time weighting
    according to IEC / DIN EN 61672-1.

    Paramters:
    ---------

    signal: Signal
        The signal object to calculate the levels for

    type: str
        The time weighting type. Options are "F" (="fast") and "S" (="slow"),
        which correspond to level decays of -34.7 dB and -4.3 dB,
        respectively.

    reference: float
        The reference sound pressure in Pascal to calculate the level
        for. Defaults to 1, in which case the returned level is in dBFS
        (decibels to full scale). If the input signal is in Pascal, use
        `2e-5` to obtain a sound pressure level (SPL).

    Returns
    -------
    A numpy array of the same shape as `signal.time` with the current time
    weighted level for each point in time and channel.
    """
    weighting = weighting.upper()
    if weighting in ["F", "FAST"]:
        time_constant = 0.125
    elif weighting in ["S", "SLOW"]:
        time_constant = 1
    else:
        raise ValueError(f"Unknown 'type' parameter: {weighting}")

    energies = signal.time**2

    # the e-function defined in the norm, but while the norm defines it
    # as an integral, we can implement it recursively as a scaling factor
    # for performance.
    sample_duration = 1 / signal.sampling_rate
    exp_func_decay = np.exp(-sample_duration / time_constant)

    # this is effectively the same as looping over all samples and doing
    # > weighted[i] = exp_func_decay * weighted[i-1] + energies[i]
    # but it's faster to use an lfilter call to make use of scipy's speed
    weighted = sps.lfilter([1, 0], [1, -exp_func_decay], energies)

    # normalize the integral
    normalized = weighted / time_constant / signal.sampling_rate

    with np.errstate(divide="ignore"):
        levels = 10 * np.log10(normalized / reference**2)
    return levels
