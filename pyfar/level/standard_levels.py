"""Level calculation functions according to IEC 61672-1."""

from typing import Literal
import numpy as np

from ._utils import (
    _check_signal_type,
    _apply_frequency_weighting,
    _apply_multi_band,
    _energies_to_levels,
)


def equivalent_continuous_level(signal,
                                frequency_weighting: Literal["A", "C", "Z"],
                                num_octave_band_fractions: int | None = None,
                                reference_pressure: float = 20e-6,
                                ):
    r"""Calculates the frequency-weighted equivalent continuous sound pressure
    level (Leq) for each channel of a signal.

    The levels are calculated according to IEC 61672-1 [#]_.
    For instance, the A-weighted equivalent continuous level is calculated as:

    .. math::
        L_\text{Aeq} = 10 \log_{10} \left[ \frac{(1/N) \sum_{n=0}^{N-1}
        p_{\text{A}}^2[n]} {p_0^2} \right] \text{ dB}

    where :math:`N` is the number of samples in the signal and
    :math:`p_0` is the `reference_pressure`.

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

    Returns
    -------
    levels: NDArray
        The calculated levels of each channel in dB relative to the
        `reference_pressure`.

        .. note::
            If `num_octave_band_fractions` is not ``None``, the returned array
            will have an additional first dimension for the individual bands.

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
