"""Public utility functions related to sound level calculations."""

from typing import Literal
import numpy as np
import scipy.signal
import pyfar as pf

from ._utils import _check_signal_type

def time_weighted_pressure(signal, time_weighting: Literal["F", "S"]):
    r"""
    Calculates sound pressure with exponential time weighting.
    This uses the method defined in IEC 61672-1 [#]_, but returns
    sound pressure values instead of levels.

    The standard defines the time weighting F as:

    .. math::
        L_\text{F}(t) = 10 \log_{10} \left[ \frac{(1 / \tau_{\text{F}})
        \int_{-\infty}^{t} p^2(\xi) e^{-(t-\xi)/\tau_\text{F}} d\xi}
        {p_0^2}\right] \text{ dB}

    This function works on finite, discrete signals and only calculates the
    weighted pressure:

    .. math::
        p_\text{F}[n] = \sqrt{ (1/\tau_F) \sum_{0}^{N-1} p^2(n)
        e^{-(t-n)/\tau_\text{F}} }

    .. note::
        While this function appears similar to functions in
        `pyfar.dsp.filter`, it is not a linear system like actual filters,
        since the time data is squared in this algorithm, removing the
        sign of each sample. This function therefore exists mainly as a
        helper function for other functions in `pyfar.level` as well as
        for plotting purposes.

    Parameters
    ----------
    signal: Signal
        The signal object to apply the weighting to.

    time_weighting: ``"F"`` or ``"S"``
        The time weighting type. Options are ``"F"`` (fast) and ``"S"`` (slow),
        which correspond to level decays of -34.7 dB and -4.3 dB per second,
        respectively.

    Returns
    -------
    weighted: TimeData
        A series of positive sound pressure values to which the
        weighting was applied.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.

    Examples
    --------
    Plot the effect of time weighting on example audio.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> audio = pf.signals.files.drums()
        >>> fast_weighted = pf.level.time_weighted_pressure(audio, "F")
        >>> slow_weighted = pf.level.time_weighted_pressure(audio, "S")
        >>> pf.plot.time(audio, dB=True, label="Audio content", alpha=0.7)
        >>> pf.plot.time(fast_weighted, dB=True, label="Fast-weighted level")
        >>> pf.plot.time(slow_weighted, dB=True, label="Slow-weighted level")
        >>> plt.ylim(-45, 5)
        >>> plt.legend()
        >>> plt.show()
    """
    if not isinstance(time_weighting, str):
        raise TypeError("Time weighting must be a string.")
    weighting = time_weighting.upper()
    if weighting in ["F"]:
        time_constant = 0.125
    elif weighting in ["S"]:
        time_constant = 1
    else:
        raise ValueError("Time weighting must be 'F' or 'S'")

    signal = _check_signal_type(signal)
    energies = signal.time**2

    # the exponential function defined in the norm, but while the norm defines
    # it as an integral, we can implement it recursively as a scaling factor
    # for performance.
    sample_duration = 1 / signal.sampling_rate
    exp_func_decay = np.exp(-sample_duration / time_constant)

    # this is effectively the same as looping over all samples and doing
    # > weighted[i] = exp_func_decay * weighted[i-1] + energies[i]
    # but it's faster to use an lfilter call to make use of scipy's speed
    # and channel handling
    weighted = scipy.signal.lfilter([1, 0], [1, -exp_func_decay], energies)

    # normalize the integral
    normalized = weighted / time_constant / signal.sampling_rate

    # turn energy to back to pressure
    time_weighted_pressure = np.sqrt(normalized)
    return pf.TimeData(time_weighted_pressure, signal.times)
