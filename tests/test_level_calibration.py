import pytest
import pyfar as pf
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize("calibrator_level", [94, 114])
@pytest.mark.parametrize("rms", [0.1, 0.5])
def test_level_sensitivity_correction_values(calibrator_level, rms):
    recording = pf.signals.sine(
        1000, 48000, amplitude=rms * np.sqrt(2), sampling_rate=48000)
    correction_factor, level_offset = pf.level.sensitivity_correction(
        recording, calibrator_level)

    calibrator_rms = 20e-6 * 10**(calibrator_level / 20)
    npt.assert_allclose(correction_factor, calibrator_rms / rms)
    npt.assert_allclose(level_offset, calibrator_level - 20 * np.log10(rms))


def test_level_sensitivity_correction_reference_pressure():
    """A calibrator level relative to 1 Pa must give the same correction
    factor as the equivalent level relative to 20 µPa.
    """
    recording = pf.signals.sine(
        1000, 48000, amplitude=0.1, sampling_rate=48000)
    level_spl = 94
    level_re_1pa = level_spl + 20 * np.log10(20e-6)

    factor_spl, _ = pf.level.sensitivity_correction(recording, level_spl)
    factor_1pa, offset_1pa = pf.level.sensitivity_correction(
        recording, level_re_1pa, reference_pressure=1)

    npt.assert_allclose(factor_spl, factor_1pa)
    npt.assert_allclose(offset_1pa, level_re_1pa - 20 * np.log10(0.1 / 2**.5))


def test_level_sensitivity_correction_converts_levels():
    """Both returned conversion methods must yield the calibrator level
    when applied to the calibrator recording itself.
    """
    recording = pf.signals.sine(
        1000, 48000, amplitude=[0.05, 0.3], sampling_rate=48000)
    correction_factor, level_offset = pf.level.sensitivity_correction(
        recording, 94)

    calibrated = recording * correction_factor
    level = pf.level.equivalent_continuous_level(calibrated, "Z")
    level_fs = pf.level.equivalent_continuous_level(
        recording, "Z", reference_pressure=1)

    npt.assert_allclose(level, [94, 94])
    npt.assert_allclose(level_fs + level_offset, [94, 94])


def test_level_sensitivity_correction_removes_dc():
    recording = pf.signals.sine(
        1000, 48000, amplitude=0.1, sampling_rate=48000)
    # add DC in the time domain, pyfar arithmetic uses the frequency domain
    recording_dc = pf.Signal(recording.time + 0.05, recording.sampling_rate)

    npt.assert_allclose(
        pf.level.sensitivity_correction(recording_dc),
        pf.level.sensitivity_correction(recording))


@pytest.mark.parametrize("amplitude", [
    0.1, [0.1, 0.2], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
def test_level_sensitivity_correction_output_shapes(amplitude):
    recording = pf.signals.sine(1000, 4800, amplitude=amplitude)
    for output in pf.level.sensitivity_correction(recording):
        assert output.shape == recording.cshape


def test_level_sensitivity_correction_input_does_not_change():
    recording = pf.signals.sine(1000, 4800, amplitude=0.1)
    recording = pf.Signal(recording.time + 0.05, recording.sampling_rate)
    original = recording.copy()
    pf.level.sensitivity_correction(recording)
    assert recording == original


@pytest.mark.parametrize(("kwargs", "err_type", "match"), [
    ({"signal": np.ones(10)}, TypeError, "pyfar.Signal"),
    ({"calibrator_level": "94"}, TypeError, "calibrator_level"),
    ({"calibrator_level": True}, TypeError, "calibrator_level"),
    ({"calibrator_level": np.nan}, ValueError, "calibrator_level"),
    ({"calibrator_level": np.inf}, ValueError, "calibrator_level"),
    ({"reference_pressure": None}, TypeError, "reference_pressure"),
    ({"reference_pressure": 0}, ValueError, "reference_pressure"),
    ({"reference_pressure": -1}, ValueError, "reference_pressure"),
    ({"reference_pressure": np.inf}, ValueError, "reference_pressure"),
])
def test_level_sensitivity_correction_invalid_inputs(kwargs, err_type, match):
    kwargs = {
        "signal": pf.signals.sine(1000, 4800, amplitude=0.1), **kwargs}
    with pytest.raises(err_type, match=match):
        pf.level.sensitivity_correction(**kwargs)


def test_level_sensitivity_correction_silent_channel():
    recording = pf.signals.sine(1000, 4800, amplitude=[0.1, 0])
    with pytest.raises(ValueError, match="silent"):
        pf.level.sensitivity_correction(recording)
