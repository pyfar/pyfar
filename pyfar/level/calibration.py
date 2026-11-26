"""Functions for calibrating recordings to absolute sound pressure."""

import numpy as np
import pyfar as pf

from ._utils import _check_signal_type, _check_finite_number


def sensitivity_correction(
        signal,
        calibrator_level: float = 94.0,
        reference_pressure: float = pf.constants.reference_sound_pressure,
):
    r"""Calculate the correction factor and level offset of a recording
    chain from a calibrator recording.

    A sound calibrator produces a sine tone (typically at 1 kHz) with a known
    sound pressure level, e.g., 94 dB SPL, which corresponds to an RMS
    sound pressure of approximately 1 Pa. Recording this tone with a
    microphone allows to relate the digital amplitude of the recording to the
    absolute sound pressure at the microphone.

    The correction factor :math:`c` and the level offset :math:`\Delta L`
    are calculated as

    .. math::
        c = \frac{p_\text{RMS}}{x_\text{RMS}}
        = \frac{p_0 \cdot 10^{L_\text{cal} / 20}}{x_\text{RMS}}, \qquad
        \Delta L = 20 \log_{10} \left( \frac{c}{p_0} \right) \text{ dB}
        = L_\text{cal} - L_\text{FS}

    where :math:`p_\text{RMS}` is the RMS sound pressure produced by the
    calibrator, :math:`x_\text{RMS}` is the RMS value of the recording in
    digital full scale units, :math:`L_\text{cal}` is the
    `calibrator_level`, :math:`p_0` is the `reference_pressure`, and
    :math:`L_\text{FS}` is the level of the recording in dB relative
    to full scale, i.e., with a reference value of ``1``.

    The correction factor is the inverse of the combined sensitivity of the
    whole recording chain between the microphone and the digital signal.
    Multiplying a recording made with the same setup by :math:`c` converts
    it to sound pressure in Pa, so it can be passed to the level functions
    in :py:mod:`pyfar.level` with their default `reference_pressure`.
    Alternatively, adding :math:`\Delta L` to levels calculated with a
    `reference_pressure` of ``1`` (dBFS) converts them to dB SPL.

    Parameters
    ----------
    signal: Signal
        The recording of the calibrator tone in digital full scale units.
        Each channel is treated as a separate recording chain. The signal
        should only contain the stationary calibrator tone, so cut away
        any parts before the calibrator was attached or after it was
        removed. Any DC offset is removed before calculation.

    calibrator_level: float
        The sound pressure level produced by the calibrator in dB relative
        to the `reference_pressure`. Check the calibrator's specification
        for the correct value. The default is ``94``.

    reference_pressure: float
        The reference pressure of the `calibrator_level` in Pa. The default
        value, ``20e-6``, corresponds to the standard reference pressure of
        20 micropascals for sound in air, in which calibrators are usually
        specified. Other values are only useful for other media or
        standards, e.g., ``1e-6`` for underwater acoustics.

    Returns
    -------
    correction_factor: NDArray
        The factor :math:`c` in Pa per full scale unit that converts
        recordings to sound pressure for each channel, with shape
        ``signal.cshape``.

    level_offset: NDArray
        The offset :math:`\Delta L` in dB that converts levels relative to
        full scale to levels relative to the `reference_pressure` for each
        channel, with shape ``signal.cshape``.

    Examples
    --------
    Calibrate a recording chain with a simulated calibrator recording and
    use the result to calculate the sound pressure level of another
    recording made with the same setup.

    >>> import pyfar as pf
    >>> # calibrator tone of 94 dB SPL (about 1 Pa RMS) recorded at -20 dBFS
    >>> recording = pf.signals.sine(
    ...     1000, 48000, amplitude=0.1 * 2**0.5, sampling_rate=48000)
    >>> correction_factor, level_offset = pf.level.sensitivity_correction(
    ...     recording, calibrator_level=94)
    >>> print(correction_factor)    # [10.02374467]
    >>> print(level_offset)         # [114.]
    >>>
    >>> # the noise level is identical for both approaches
    >>> noise = pf.signals.noise(48000, rms=0.01, seed=0)
    >>> level = pf.level.equivalent_continuous_level(
    ...     noise * correction_factor, "Z")
    >>> level_fs = pf.level.equivalent_continuous_level(
    ...     noise, "Z", reference_pressure=1)
    >>> print(level)                    # [74.]
    >>> print(level_fs + level_offset)  # [74.]
    """
    signal = _check_signal_type(signal)
    _check_finite_number(calibrator_level, "calibrator_level")
    _check_finite_number(
        reference_pressure, "reference_pressure", positive=True)

    # microphones do not pick up static pressure, so any DC offset stems from
    # the electronics and would otherwise bias the RMS value
    time = signal.time - np.mean(signal.time, axis=-1, keepdims=True)
    recorded_rms = np.sqrt(np.mean(time**2, axis=-1))
    if np.any(recorded_rms == 0):
        raise ValueError("'signal' must not contain silent channels.")

    calibrator_rms = reference_pressure * 10**(calibrator_level / 20)
    correction_factor = calibrator_rms / recorded_rms
    level_offset = 20 * np.log10(correction_factor / reference_pressure)

    return correction_factor, level_offset
