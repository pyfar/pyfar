"""Level calculation functions according to IEC 61672-1."""

from typing import Literal
import numpy as np
import pyfar as pf

from .utils import time_weighted_pressure
from ._utils import (
    _check_signal_type,
    _apply_frequency_weighting,
    _apply_multi_band,
    _apply_oversampling,
    _time_weighting_to_constant,
    _energies_to_levels,
    _moving_average,
)


def equivalent_continuous_level(
        signal,
        frequency_weighting: Literal["A", "C", "Z"],
        num_octave_band_fractions: int | None = None,
        reference_pressure: float = pf.constants.reference_sound_pressure,
):
    r"""Calculate the frequency-weighted equivalent continuous sound pressure
    level (Leq).

    The levels are calculated per channel and according to IEC 61672-1 [#]_.
    For instance, the A-weighted equivalent continuous level is calculated as:

    .. math::
        L_\text{Aeq} = 10 \log_{10} \left[ \frac{(1/N) \sum_{n=0}^{N-1}
        p_{\text{A}}^2[n]} {p_0^2} \right] \text{ dB}

    where :math:`N` is the number of samples in the signal,
    :math:`p_\mathrm{A}` the A-weighted sound pressure at index :math:`n`,
    and :math:`p_0` is the `reference_pressure`.

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of.

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        The frequency weighting is applied using
        :py:func:`~pyfar.dsp.filter.frequency_weighting_filter` with its
        (standard-compliant) default parameters. If you need more control,
        you can set this parameter to ``"Z"`` and apply the frequency
        weighting filter yourself before calling this function.

    num_octave_band_fractions: int or ``None``
        Can be used to calculate the level in octave (``1``), third-octave
        (``3``), or other positive integer fractional octave bands.
        If ``None``, levels are calculated for the full-band signal.

        The fraction octave band filtering is applied using
        :py:func:`~pyfar.dsp.filter.fractional_octave_bands` with its
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

    Returns
    -------
    levels: NDArray
        The calculated levels of each channel in dB relative to the
        `reference_pressure`.

        `levels` has shape ``signal.cshape`` if `num_octave_band_fractions`
        is ``None`` and ``(n_bands, signal.cshape)`` otherwise, where
        `n_bands` denotes the number of (fractional) octave bands.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.
    """
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_multi_band(signal, num_octave_band_fractions)
    mean_energy_per_channel = np.mean(signal.time**2, axis=-1)
    levels = _energies_to_levels(mean_energy_per_channel, reference_pressure)
    return levels


def exposure_level(
        signal,
        frequency_weighting: Literal["A", "C", "Z"],
        duration: float | None = None,
        reference_pressure: float = pf.constants.reference_sound_pressure,
):
    r"""Calculate the frequency-weighted sound exposure level.

    The levels are calculated per channel and according to IEC 61672-1 [#]_.
    For instance, the A-weighted sound exposure level is calculated as:

    .. math::
        L_{\text{A}E,T} = 10 \log_{10} \left[ \frac{T/N \sum_{n=0}^{N-1}
        p_{\text{A}}^2[n]} {p_0^2 T_0} \right] \text{ dB}
        = L_\text{Aeq} + 10 \log_{10}(T) \text{ dB}

    where :math:`N` is the number of samples in the signal,
    :math:`p_\mathrm{A}` the A-weighted sound pressure at index :math:`n`,
    :math:`p_0` is the `reference_pressure`,
    :math:`T_0` is the reference duration of 1 second, and
    :math:`T` is the duration of the sound exposure (see below).

    .. note::
        The standard defines the sound exposure level relative to
        a reference *energy* :math:`E_0 = p_0^2 T_0
        = 20\,\mu\text{Pa}^2 \cdot 1 \text{ s}
        = 400 \cdot 10^{-12} \text{ J}`.
        However, this function uses a reference *pressure* instead to remain
        consistent with the other level functions.
        You can obtain the `reference_pressure` by taking the square root of of
        the desired reference energy.

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of.

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        The frequency weighting is applied using
        :py:func:`~pyfar.dsp.filter.frequency_weighting_filter` with its
        (standard-compliant) default parameters. If you need more control,
        you can set this parameter to ``"Z"`` and apply the frequency
        weighting filter yourself before calling this function.

    duration: float or ``None``
        The duration of the signal in seconds. If ``None``, the duration is
        set to the length of the signal in seconds to calculate the exposure
        of just the signal. You can specify a different duration to extrapolate
        the exposure level to a longer (or shorter) time period. This
        extrapolation is valid if the equivalent sound pressure level
        :math:`L_\text{eq}` over the longer (or shorter) time period is the
        same as that of this signal.
        The duration must be a positive number.

    reference_pressure: float
        The reference pressure to calculate levels relative to. The default
        value, ``20e-6``, corresponds to the standard reference pressure of
        20 micropascals, which assumes the signal is in units of pascals (Pa).
        To compute the level in dBFS of a digital signal, or if you plan
        to correct for the recording setup afterwards, this parameter
        should be ``1``.

    Returns
    -------
    levels: NDArray
        The calculated levels of each channel in dB relative to the
        `reference_pressure`.

        `levels` has shape ``signal.cshape``.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.
    """
    signal = _check_signal_type(signal)
    duration = signal.signal_length if duration is None else duration
    if not isinstance(duration, (int, float, np.number)):
        raise TypeError("Duration must be a number.")
    if duration <= 0:
        raise ValueError("Duration must be a positive number.")

    eq_level = equivalent_continuous_level(
        signal, frequency_weighting, None, reference_pressure)
    duration_term = 10 * np.log10(duration)
    return eq_level + duration_term


def sliding_equivalent_continuous_level(
    signal,
    frequency_weighting: Literal["A", "C", "Z"],
    num_octave_band_fractions: int | None = None,
    window_duration: float = 1,
    cyclic: bool = False,
    center_window: bool = False,
    reference_pressure: float = pf.constants.reference_sound_pressure,
    replace_zeros: bool = True,
):
    r"""Calculate the frequency-weighted equivalent continuous sound
    pressure level with a sliding time window.

    The levels are calculated per channel and according to IEC 61672-1 [#]_.
    See :py:func:`~pyfar.level.equivalent_continuous_level` for the definition
    of the equivalent continuous level.

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of.

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        The frequency weighting is applied using
        :py:func:`~pyfar.dsp.filter.frequency_weighting_filter` with its
        (standard-compliant) default parameters. If you need more control,
        you can set this parameter to ``"Z"`` and apply the frequency
        weighting filter yourself before calling this function.

    num_octave_band_fractions: int or ``None``
        Can be used to calculate the level in octave (``1``), third-octave
        (``3``), or other positive integer fractional octave bands.
        If ``None``, levels are calculated for the full-band signal.

        The fraction octave band filtering is applied using
        :py:func:`~pyfar.dsp.filter.fractional_octave_bands` with its
        (standard-compliant) default parameters. If you need more control,
        you can set this parameter to ``None`` and apply the filter bank
        yourself before calling this function.

    window_duration: float
        The duration of the sliding window in seconds, which determines
        the time over which the energy is averaged at each sample.
        The default value is ``1``.

    cyclic: bool
        If ``True``, the signal is treated as if it were periodic (like a
        cyclic convolution). If ``False``, edges are zero-padded,
        leading to fade-like boundary effects. The default is ``False``.

    center_window: bool
        If ``True``, the window is centered on the current sample.
        If ``False``, the window is causal, meaning it only considers
        past values, just like a convolution with a rectangular window.
        The default is ``False``.

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
        The calculated levels of each channel in dB relative to the
        `reference_pressure`.

        `levels` has shape ``signal.time.shape`` if `num_octave_band_fractions`
        is ``None`` and ``(n_bands, signal.time.shape)`` otherwise, where
        `n_bands` denotes the number of (fractional) octave bands.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.

    Examples
    --------
    Obtain the equivalent continuous sound pressure level
    over a 2-second window in dbFS(A). Note how the sliding level is rising
    monotonically for the first 2 seconds, due to the 2 seconds before the
    signal start being treated as zero.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> signal: pf.Signal = pf.signals.files.guitar()
        >>> sliding_levels = pf.level.sliding_equivalent_continuous_level(
        >>>         signal, "A", window_duration=2, reference_pressure=1)
        >>> pf.plot.time(signal, True, alpha=0.6, label="Signal")
        >>> plt.plot(signal.times, sliding_levels[0], label="Sliding Leq")
        >>> plt.legend()
        >>> plt.ylim(-50, 0)
        >>> plt.show()

    Demonstrate the difference between centered vs causal windows and
    cyclic vs non-cyclic processing.
    Note how the centered (acausal) window leads to peaks at the most
    energy-dense times of the signal, but rises before the energy is present.
    The causal (non-centered) window rises only after the energy is present,
    but lags behind the most energy-dense times.
    Also note how the cyclic processing leads to a wrap-around of the energy
    at the edges. With non-cyclic processing, a causal window will always
    lead to levels starting at zero and remaining identical to the cyclic ones
    after reaching the window duration.
    In contrast, with a non-cyclic, centered window, the
    levels will start with the energy of the right half of the window, and
    diverge from the cyclic ones at both edges.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> fs = 44100
        >>> signal = pf.signals.pulsed_noise(
        >>>     fs, fs, repetitions=2, rms=0.2, sampling_rate=fs, seed=0)
        >>> signal = pf.dsp.pad_zeros(signal, fs // 4, "beginning")
        >>> pf.plot.time(signal, True, alpha=0.6, label="Signal")
        >>> variants = [
        >>>     ("Causal, non-cyclic", False, False),
        >>>     ("Causal, cyclic", False, True),
        >>>     ("Centered, non-cyclic", True, False),
        >>>     ("Centered, cyclic", True, True),
        >>> ]
        >>> for title, center_window, cyclic in variants:
        >>>     ls = "dashed" if cyclic else "solid"
        >>>     s = pf.level.sliding_equivalent_continuous_level(
        >>>         signal, "Z", None, 1.1,
        >>>         cyclic, center_window, 1, True
        >>>     )
        >>>     plt.plot(signal.times, s[0], label=title, linestyle=ls)
        >>> plt.legend()
        >>> plt.ylim(-40, 0)
        >>> plt.show()
    """
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_multi_band(signal, num_octave_band_fractions)

    window_size = round(signal.sampling_rate * window_duration)
    energies = signal.time**2
    mean_energies = _moving_average(energies, window_size, cyclic=cyclic,
                                    center_window=center_window)
    levels = _energies_to_levels(
        mean_energies, reference_pressure, replace_zeros)
    return levels


def peak_level(
        signal,
        frequency_weighting: Literal["A", "C", "Z"],
        oversampling: float | None = 4,
        reference_pressure: float = pf.constants.reference_sound_pressure,
):
    """Calculate the frequency-weighted peak sound pressure level.

    The levels are calculated per channel; and according to IEC 61672-1 [#]_.

    Applies optional oversampling to find the "true" inter-sample peak.
    The actual peak amplitude of a digital signal may occur between
    discrete time samples, especially for high frequencies. This can cause
    the true analog peak to exceed the digital peak value unless oversampling
    is applied.
    Please refer to ITU-R BS.1770-5 [#]_ Annex 2 for further information and
    on why oversampling is used for true-peak detection as well as for
    recommendations on the oversampling factor.

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of.

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        The frequency weighting is applied using
        :py:func:`~pyfar.dsp.filter.frequency_weighting_filter` with its
        (standard-compliant) default parameters. If you need more control,
        you can set this parameter to ``"Z"`` and apply the frequency
        weighting filter yourself before calling this function.

    oversampling: float or ``None``
        The oversampling factor to apply before calculating the peak level.
        The default value of ``4`` matches the true-peak detection
        recommendation from ITU-R BS.1770-5 for 48 kHz signals.

    reference_pressure: float
        The reference pressure to calculate levels relative to. The default
        value, ``20e-6``, corresponds to the standard reference pressure of
        20 micropascals, which assumes the signal is in units of pascals (Pa).
        To compute the level in dBFS of a digital signal, or if you plan
        to correct for the recording setup afterwards, this parameter
        should be ``1``.

    Returns
    -------
    levels: np.ndarray
        The peak sound pressure levels in dB, one per channel.
    times: np.ndarray
        The times of the peak levels in seconds, one per channel. If
        `oversampling` is ``None``, the times match sample times of the input.
        Otherwise, `times` may contain inter-sample times.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.

    .. [#] International Telecommunication Union,
        Recommendation ITU-R BS.1770-5 (11/2023): "`Algorithms to measure
        audio programme loudness and true-peak audio level <https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf>`_".
    """
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_oversampling(signal, oversampling)
    energies = signal.time**2

    maxima = np.max(energies, axis=-1)
    indexes = np.argmax(energies, axis=-1)
    levels = _energies_to_levels(maxima, reference_pressure)
    times: np.ndarray = indexes / signal.sampling_rate
    return levels, times


def time_weighted_level(signal,
                        frequency_weighting: Literal["A", "C", "Z"],
                        time_weighting: Literal["F", "S"],
                        num_octave_band_fractions: int | None = None,
                        reference_pressure: float = 20e-6,
                        replace_zeros: bool = True,
                        ):
    r"""
    Calculates frequency and time weighted sound pressure levels for
    each sample of a signal.

    The levels are calculated per channel according to IEC 61672-1 [#]_.
    The returned array has the same shape as the input in the time domain,
    containing sound pressure levels in dB relative to the
    `reference_pressure`.

    For instance, the A-weighted, F-time-weighted level is defined as:

    .. math::
        L_\text{AF}(t) = 10 \log_{10} \left[ \frac{(1 / \tau_{\text{F}})
        \int_{-\infty}^{t} p_{\text{A}}^2(\xi) e^{-(t-\xi)/\tau_\text{F}} d\xi}
        {p_0^2}\right] \text{ dB}

    where :math:`p_\text{A}` is the A-weighted sound pressure,
    :math:`p_0` is the reference sound pressure,
    :math:`t` is the time at which to calculate the level,
    and :math:`\tau_\text{F}` is the time constant for the F weighting.

    Because this function works on finite, discrete signals, this
    formula becomes:

    .. math::
        L_\text{AF}[n] = 10 \log_{10} \left[
            \frac{1/(f_\text{s}\tau_F) \sum_{i=0}^{n} p_\text{A}^2[i]
            e^{-(n-i)/(f_\text{s} \tau_\text{F})} }{p_0^2}
        \right] \text{ dB}

    where :math:`n` is the sample index currently being calculated,
    :math:`p_\text{A}[i]` is the A-weighted sound pressure at index :math:`i`,
    :math:`f_\text{s}` is the sampling rate,
    and :math:`\tau_\text{F}` is the time constant for the F weighting.

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of.

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        The frequency weighting is applied using
        :py:func:`~pyfar.dsp.filter.frequency_weighting_filter` with its
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

        The fraction octave band filtering is applied using
        :py:func:`~pyfar.dsp.filter.fractional_octave_bands` with its
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

        `levels` has shape ``signal.time.shape`` if `num_octave_band_fractions`
        is ``None`` and ``(n_bands, signal.time.shape)`` otherwise, where
        `n_bands` denotes the number of (fractional) octave bands.

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
        >>> levels = pf.level.time_weighted_level(signal, "Z", "F",
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
    time_constant = _time_weighting_to_constant(time_weighting)
    weighted = time_weighted_pressure(signal, time_constant)
    energies = weighted.time**2
    levels = _energies_to_levels(energies, reference_pressure, replace_zeros)
    return levels


def maximum_time_weighted_level(
        signal,
        frequency_weighting: Literal["A", "C", "Z"],
        time_weighting: Literal["F", "S"],
        oversampling: float | None = 4,
        reference_pressure: float = pf.constants.reference_sound_pressure,
):
    r"""Calculate the maximum time-weighted sound pressure level.

    The levels are calculated per channel; and according to IEC 61672-1 [#]_,
    where this is defined as the maximum time-weighted level of the signal,
    such as :math:`L_{\text{AFmax}}` for the ``A`` frequency weighting and
    ``F`` time weighting.

    For more information on the oversampling and time weighting, please refer
    to :py:func:`~pyfar.level.peak_level` and
    :py:func:`~pyfar.level.time_weighted_level`, respectively.

    Parameters
    ----------
    signal: Signal
        The signal object to calculate the levels of.

    frequency_weighting: ``"A"``, ``"C"``, or ``"Z"``
        The frequency weighting type. If ``"A"`` or ``"C"``, the corresponding
        frequency weighting filter is applied before level
        calculation. If ``"Z"``, no frequency weighting is applied.

        The frequency weighting is applied using
        :py:func:`~pyfar.dsp.filter.frequency_weighting_filter` with its
        (standard-compliant) default parameters. If you need more control,
        you can set this parameter to ``"Z"`` and apply the frequency
        weighting filter yourself before calling this function.

    time_weighting: ``"F"`` or ``"S"``
        The time weighting type. Options are ``"F"`` (fast) and
        ``"S"`` (slow), which correspond to level decays of -34.7 dB and
        -4.3 dB per second, respectively.

    oversampling: float or ``None``
        The oversampling factor to apply before calculating the peak level.
        The default value of ``4`` matches the true-peak detection
        recommendation from ITU-R BS.1770-5 for 48 kHz signals.

    reference_pressure: float
        The reference pressure to calculate levels relative to. The default
        value, ``20e-6``, corresponds to the standard reference pressure of
        20 micropascals, which assumes the signal is in units of pascals (Pa).
        To compute the level in dBFS of a digital signal, or if you plan
        to correct for the recording setup afterwards, this parameter
        should be ``1``.

    Returns
    -------
    levels: np.ndarray
        The peak sound pressure levels in dB, one per channel.
    times: np.ndarray
        The times of the peak levels in seconds, one per channel. If
        `oversampling` is ``None``, the times match sample times of the input.
        Otherwise, `times` may contain inter-sample times.

    References
    ----------
    .. [#] International Electrotechnical Commission,
        "IEC 61672-1:2013 - Electroacoustics - Sound level meters - Part 1:
        Specifications", IEC, 2013.
    """
    signal = _check_signal_type(signal)
    signal = _apply_frequency_weighting(signal, frequency_weighting)
    signal = _apply_oversampling(signal, oversampling)
    time_constant = _time_weighting_to_constant(time_weighting)
    time_weighted = time_weighted_pressure(signal, time_constant)
    maxima = np.max(time_weighted.time, axis=-1)
    levels = _energies_to_levels(maxima**2, reference_pressure)
    indexes = np.argmax(time_weighted.time, axis=-1)
    times = indexes / signal.sampling_rate
    return levels, times
