"""Fast and Slow time weighting as found in standardized sound level meters."""
from typing import Literal
import numpy as np
import scipy.signal
import pyfar as pf

def time_weighted_sound_pressure(signal, time_weighting: Literal["F", "S"]):
    r"""
    Calculates sound pressure with exponential time weighting
    according to IEC 61672-1 [#]_, returning as a Signal of sound pressure
    instead of levels.

    The standard defines the time weighting F as:

    .. math::
        L_\text{F}(t) = 10 \log_{10} \left[ \frac{(1 / \tau_{\text{F}})
        \int_{-\infty}^{t} p^2(\xi) e^{-(t-\xi)/\tau_\text{F}} d\xi}
        {p_0^2}\right] \text{ dB}

    This function works on finite, discrete signals and only calculates the
    weighted pressure:

    .. math::
        p_\text{F}[n] = \sqrt{ (1/\tau_F) \sum_{0}^{n} p^2(n)
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
        The signal object to apply the weighting to

    time_weighting: ``"F"`` or ``"S"``
        The time weighting type. Options are ``"F"`` (fast) and ``"S"`` (slow),
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

    Examples
    --------
    Plot the effect of time weighting on example audio.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> audio = pf.signals.files.drums()
        >>> fast_weighted = pf.level.time_weighted_sound_pressure(audio, "F")
        >>> slow_weighted = pf.level.time_weighted_sound_pressure(audio, "S")
        >>> pf.plot.time(audio, dB=True, label="Audio content", alpha=0.7)
        >>> pf.plot.time(fast_weighted, dB=True, label="Fast-weighted level")
        >>> pf.plot.time(slow_weighted, dB=True, label="Slow-weighted level")
        >>> plt.ylim(-45, 5)
        >>> plt.legend()
        >>> plt.show()
    """
    weighting = time_weighting.upper()
    if weighting in ["F"]:
        time_constant = 0.125
    elif weighting in ["S"]:
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
    # and channel handling
    weighted = scipy.signal.lfilter([1, 0], [1, -exp_func_decay], energies)

    # normalize the integral
    normalized = weighted / time_constant / signal.sampling_rate

    # turn energy to back to pressure
    time_weighted_pressure = np.sqrt(normalized)
    return pf.Signal(time_weighted_pressure, signal.sampling_rate)


def time_weighted_level(signal,
                        frequency_weighting: Literal["A", "C", "Z"],
                        time_weighting: Literal["F", "S"],
                        num_octave_band_fractions: int | None = None,
                        reference_pressure: float = 20e-6,
                        replace_zeros: bool = True,
                        ):
    r"""
    Calculates the frequency and time weighted sound pressure level of a
    signal according to IEC 61672-1 [#]_. The returned values are the levels
    at each sample in dB relative to the `reference_pressure`.

    For instance, the A-weighted, F-time-weighted level is defined as:

    .. math::
        L_\text{AF}(t) = 10 \log_{10} \left[ \frac{(1 / \tau_{\text{F}})
        \int_{-\infty}^{t} p_{\text{A}}^2(\xi) e^{-(t-\xi)/\tau_\text{F}} d\xi}
        {p_0^2}\right] \text{ dB}

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        .. note::
            The frequency weighting is applied using
            `pyfar.dsp.filter.frequency_weighting_filter` with its
            (standard-compliant) default parameters. If you need more control,
            you can set this parameter to ``"Z"`` and apply the frequency
            weighting filter yourself before calling this function.

    time_weighting: ``"F"`` or ``"S"``
        The time weighting type. Options are ``"F"`` (fast) and
        ``"S"`` (slow), which correspond to level decays of -34.7 dB and
        -4.3 dB per second, respectively.

    num_octave_band_fractions: int or ``None``
        Can be used to calculate the level in octave (``1``), third-octave
        (``3``), or other positive integer fractional octave bands.
        If ``None``, levels are calculated for the full-band signal.

        .. note::
            The fraction octave band filtering is applied using
            `pyfar.dsp.filter.fractional_octave_bands` with its
            (standard-compliant) default parameters. If you need more control,
            you can set this parameter to ``None`` and apply the filter bank
            yourself before calling this function.

    reference_pressure: float
        The reference pressure to calculate levels relative to. The default
        value, ``20e-6``, corresponds to the standard reference pressure of
        20 micropascals, which assumes the signal is in units of pascals (Pa).
        To compute the level in dBFS of a digital signal, or if you plan
        to correct for the recording setup afterwards, this parameter
        should be ``1``.

    replace_zeros: bool
        If ``False``, the function will return ``-inf`` for samples where the
        time-weighted energy is zero. If ``True``, these energy values will be
        replaced with a very small number (the array type epsilon)
        to avoid ``-inf`` values and corresponding numpy warnings.
        The default is ``True``.

    Returns
    -------
    levels: NDArray
        The calculated levels at each sample in dB relative to the
        `reference_pressure`.

        .. note::
            If `num_octave_band_fractions` is not ``None``, the returned array
            will have an additional first dimension for the individual bands.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.

    Examples
    --------
    Shows the unweighted and time-weighted levels of a noise burst.
    It shows the transient behavior of the time weighting during the start
    of the noise burst, as well as the constant decay after the noise stops.
    The initial silence shows the effect of the `replace_zeros` parameter,
    setting the energy to ``2.22e-16``, which is the machine epsilon of the
    ``np.float64`` type.

    .. plot::

        >>> import pyfar as pf
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> noise = pf.signals.noise(
        >>>     24000, rms=0.001, sampling_rate=48000, seed=0)
        >>> signal = pf.dsp.pad_zeros(noise, 24000, "beginning")
        >>> signal = pf.dsp.pad_zeros(signal, 3 * 48000, "end")
        >>> levels = pf.level.time_weighted_level(signal, "A", "F",
        >>>                                       reference_pressure=1)
        >>> noise_times = noise.times + 0.5
        >>> noise_levels = 10 * np.log10(noise.time[0]**2)
        >>> plt.plot(noise_times, noise_levels, label='unweighted', alpha=0.7)
        >>> plt.plot(signal.times, levels[0], label='time-weighted')
        >>> plt.ylabel("Level in dbFS")
        >>> plt.xlabel("Time in seconds")
        >>> plt.grid()
        >>> plt.legend()
        >>> plt.show()
    """
    if not isinstance(signal, pf.Signal):
        raise TypeError("'signal' parameter must be a pyfar.Signal")

    if frequency_weighting == "Z":
        pass
    elif frequency_weighting in ["A", "C"]:
        signal = pf.dsp.filter.frequency_weighting_filter(
            signal, frequency_weighting)
    else:
        raise ValueError(f"Unknown 'frequency_weighting' value: "
                         f"{frequency_weighting}")

    if num_octave_band_fractions is not None:
        signal = pf.dsp.filter.fractional_octave_bands(
            signal, num_octave_band_fractions)

    signal = time_weighted_sound_pressure(signal, time_weighting)
    energies = signal.time**2
    if replace_zeros:
        energies = np.where(
            energies == 0, np.finfo(energies.dtype).eps, energies)

    levels = 10 * np.log10(energies / reference_pressure**2)
    return levels
