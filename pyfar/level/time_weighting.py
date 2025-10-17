"""Fast and Slow time weighting as found in standardized sound level meters."""
from typing import Literal
import numpy as np
import scipy.signal
import pyfar as pf

def time_weighted_sound_pressure(signal, time_weighting: Literal["F", "S"]):
    r"""
    Calculates sound pressure with exponential time weighting
    according to IEC / DIN EN 61672-1, returning as a Signal of sound pressure
    instead of levels.

    The standard defines the time weighting F as:
    .. math:: L_\text{F}(t) = 10 \lg \left[ \frac{(1/\tau_F) \int_{-\infty}^{t}
        p^2(\xi) e^{-(t-\xi)/\tau_\text{F}} d\xi} {p_0^2}\right] \text{ dB}

    This function works on finite, discrete signals and only calculates the
    weighted pressure:
    .. math:: p_\text{F}[n] = \sqrt{ (1/\tau_F) \sum_{0}^{n} p^2(n)
        e^{-(t-n)/\tau_\text{F}} }

    Parameters:
    ---------
    signal: Signal
        The signal object to apply the weighting to

    time_weighting: "F" or "S"
        The time weighting type. Options are "F" (="fast") and "S" (="slow"),
        which correspond to level decays of -34.7 dB and -4.3 dB per second,
        respectively.

    Returns
    -------
    weighted: Signal
        A pressure signal to which the weighting was applied

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.
    """
    weighting = time_weighting.upper()
    if weighting in ["F", "FAST"]:
        time_constant = 0.125
    elif weighting in ["S", "SLOW"]:
        time_constant = 1
    else:
        raise ValueError(f"Unknown 'time_weighting' value: {time_weighting}")

    if not isinstance(signal, pf.Signal):
        raise ValueError("'signal' parameter must be a pyfar.Signal")

    energies = signal.time**2

    # the exponential function defined in the norm, but while the norm defines
    # it as an integral, we can implement it recursively as a scaling factor
    # for performance.
    sample_duration = 1 / signal.sampling_rate
    exp_func_decay = np.exp(-sample_duration / time_constant)

    # this is effectively the same as looping over all samples and doing
    # > weighted[i] = exp_func_decay * weighted[i-1] + energies[i]
    # but it's faster to use an lfilter call to make use of scipy's speed
    weighted = scipy.signal.lfilter([1, 0], [1, -exp_func_decay], energies)

    # normalize the integral
    normalized = weighted / time_constant / signal.sampling_rate

    # turn energy to back to pressure
    time_weighted_pressure = np.sqrt(normalized)
    return pf.Signal(time_weighted_pressure, signal.sampling_rate)
