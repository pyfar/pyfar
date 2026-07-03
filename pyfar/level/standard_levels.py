from typing import Literal
import numpy as np

from .time_weighting import time_weighted_sound_pressure
from ._utils import (
    _check_signal_type,
    _apply_frequency_weighting,
    _apply_multi_band,
    _apply_oversampling,
    _energies_to_levels,
    _moving_average,
)


def time_weighted_level(signal,
                        frequency_weighting: Literal["A", "C", "Z"],
                        time_weighting: Literal["F", "S"],
                        num_octave_band_fractions: int | None = None,
                        reference_pressure: float = 20e-6,
                        replace_zeros: bool = True,
                        ):
    r"""
    Calculates frequency and time weighted sound pressure levels for
    each sample in each channel of a signal.
    The levels are calculated according to IEC 61672-1 [#]_.
    The returned array has the same shape as the input in the time domain,
    containing sound pressure levels in dB relative to the
    `reference_pressure`.

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
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_multi_band(signal, num_octave_band_fractions)
    weighted = time_weighted_sound_pressure(signal, time_weighting)
    energies = weighted.time**2
    if replace_zeros:
        energies = np.where(
            energies == 0, np.finfo(energies.dtype).eps, energies)

    levels = _energies_to_levels(energies, reference_pressure)
    return levels


def equivalent_continuous_level(signal,
                                frequency_weighting: Literal["A", "C", "Z"],
                                num_octave_band_fractions: int | None = None,
                                reference_pressure: float = 20e-6,
                                ):
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_multi_band(signal, num_octave_band_fractions)
    mean_energy_per_channel = np.mean(signal.time**2, axis=-1)
    levels = _energies_to_levels(mean_energy_per_channel, reference_pressure)
    return levels


def sliding_equivalent_continuous_level(
    signal,
    frequency_weighting: Literal["A", "C", "Z"],
    num_octave_band_fractions: int | None = None,
    window_duration: float = 1,
    cyclic: bool = False,
    center_window: bool = False,
    reference_pressure: float = 20e-6,
):
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_multi_band(signal, num_octave_band_fractions)

    window_size = round(signal.sampling_rate * window_duration)
    energies = signal.time**2
    mean_energies = _moving_average(energies, window_size, cyclic=cyclic,
                                    center_window=center_window)
    levels = _energies_to_levels(mean_energies, reference_pressure)
    return levels


def exposure_level(signal,
                   frequency_weighting: Literal["A", "C", "Z"],
                   reference_energy: float = 400e-12):
    signal = _check_signal_type(signal)
    eq_level = equivalent_continuous_level(
        signal, frequency_weighting,
        reference_pressure=np.sqrt(reference_energy))
    duration_term = 10 * np.log10(signal.signal_length)
    return eq_level + duration_term


def peak_level(signal,
               frequency_weighting: Literal["A", "C", "Z"],
               reference_pressure: float = 20e-6,
               oversampling: int | None = 4,
               ):
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_oversampling(signal, oversampling)
    energies = signal.time**2
    maxima = np.max(energies, axis=-1)
    levels = _energies_to_levels(maxima, reference_pressure)
    indexes = np.argmax(energies, axis=-1)
    times = indexes / signal.sampling_rate
    return levels, times


def maximum_time_weighted_level(signal,
                                frequency_weighting: Literal["A", "C", "Z"],
                                time_weighting: Literal["F", "S"],
                                reference_pressure: float = 20e-6,
                                oversampling: int | None = 4,
                                ):

    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_oversampling(signal, oversampling)
    time_weighted = time_weighted_sound_pressure(signal, time_weighting)
    maxima = np.max(time_weighted.time, axis=-1)
    levels = _energies_to_levels(maxima**2, reference_pressure)
    indexes = np.argmax(time_weighted.time, axis=-1)
    times = indexes / signal.sampling_rate
    return levels, times
