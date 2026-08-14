"""Level calculation functions according to IEC 61672-1."""

from typing import Literal
import numpy as np
import pyfar as pf

from ._utils import (
    _check_signal_type,
    _apply_frequency_weighting,
    _apply_multi_band,
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
        The signal object to calculate the levels of

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
