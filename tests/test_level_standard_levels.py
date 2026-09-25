"""
Test for the standard-conform level functions.

Note that the tests for the shared parameters of these functions are in
`test_level_common_parameters.py`, so this file ony contains tests
against known values or other tests that are specific to a single function.
"""

import pytest
import pyfar as pf
import numpy as np

# 1 Pa in dB SPL
ONE_PA = 20 * np.log10(1 / pf.constants.reference_sound_pressure)
SINE_PAPR = 10 * np.log10(2)  # peak-to-average power ratio of sine signals


def test_level_equivalent_continuous_level_known_value():
    s = pf.signals.sine(1000, 22050)
    levels = pf.level.equivalent_continuous_level(
        s, "Z", None, 2e-5)
    assert np.isclose(levels, ONE_PA - SINE_PAPR, atol=0.001)


@pytest.mark.parametrize(("duration", "level_increase"), [
    (None, 0),  # signal is one second long, which is the unit length
    (1, 0),     # unit length in the standard
    (10, 10),   # 10 s => 10x the energy => 10 dB increase
    (100, 20),  # 100 s => 100x the energy => 20 dB increase
])
def test_level_exposure_level_duration(duration, level_increase):
    s = pf.signals.sine(1000, 44100, sampling_rate=44100)
    levels = pf.level.exposure_level(s, "Z", duration)
    assert np.isclose(levels, ONE_PA - SINE_PAPR + level_increase, atol=0.001)


@pytest.mark.parametrize("duration", [-1, 0, np.int32(-1)])
def test_level_exposure_level_duration_value_errors(duration):
    s = pf.signals.sine(1000, 44100, sampling_rate=44100)
    with pytest.raises(ValueError, match="positive"):
        pf.level.exposure_level(s, "Z", duration)


@pytest.mark.parametrize("duration", ["1", np.array([1]), [1], complex(1, 0)])
def test_level_exposure_level_duration_type_errors(duration):
    s = pf.signals.sine(1000, 44100, sampling_rate=44100)
    with pytest.raises(TypeError, match="number"):
        pf.level.exposure_level(s, "Z", duration)


def test_level_sliding_equivalent_continuous_level_known_value():
    # phase prevents the first sample from being exactly zero, which would
    # cause a division by zero in the level calculation
    s = pf.signals.sine(1000, 44100, phase=0.01, sampling_rate=44100)
    levels = pf.level.sliding_equivalent_continuous_level(
        s, "Z", None, 1, False, False, 2e-5)
    # with 1 second window size, the value at 1 second should be
    # the same as the equivalent continuous level of the full signal
    assert np.isclose(levels[0][-1], ONE_PA - SINE_PAPR, atol=0.001)


def test_level_sliding_equivalent_continuous_level_shape():
    s = pf.signals.impulse(1000, sampling_rate=48000)
    levels = pf.level.sliding_equivalent_continuous_level(s, "Z")
    assert levels.shape == s.time.shape


@pytest.mark.parametrize("oversampling", [None, 4, 8])
def test_level_peak_level_known_value(oversampling):
    delay = 5432
    s = pf.signals.impulse(10000, delay)
    levels, times = pf.level.peak_level(s, "Z", oversampling, 1)
    assert np.isclose(levels[0], 0, atol=0.01)   # peak is 1, so 0 dbFS
    assert np.isclose(times[0], delay / s.sampling_rate, atol=0.001)


def test_level_peak_level_intersample_peak():
    """Test that oversampling is actually applied by checking that the
    peak level of a signal with intersample peaks is higher than the
    peak level of the same signal without oversampling.
    """
    s = pf.Signal([0, 0, 0, 1, 0.99, 0, 0, 0], sampling_rate=100)

    level_no_over, time_no_over = pf.level.peak_level(s, "Z", None)
    level_with_over, time_with_over = pf.level.peak_level(s, "Z", 2)

    assert level_with_over > level_no_over
    # peak must be at the 4th sample (0-indexed)
    assert time_no_over == 0.03
    # after oversampling, the peak is at the 7th sample, i.e. between the
    # 3rd and 4th sample at the original sampling rate
    assert time_with_over == 0.035 # at 7th oversampled sample


def test_level_time_weighted_level_replace_zeros_false():
    """Test that setting replace_zeros to False returns -inf
    and raises a warning from numpy.
    """
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        levels_no_replace = pf.level.time_weighted_level(
            s, "Z", "F", replace_zeros=False)
    assert np.all(levels_no_replace == -np.inf)


def test_level_time_weighted_level_replace_zeros_true():
    """Test that setting replace_zeros to True replaces zeros with the
    array type's epsilon to avoid -inf values and numpy warnings.
    """
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)
    levels_replace = pf.level.time_weighted_level(
        s, "Z", "F", replace_zeros=True)

    # since there are only zeros in the signal, all values must be epsilon
    expected_value = 10 * np.log10(np.finfo(s.time.dtype).eps / 2e-5**2)
    assert np.allclose(levels_replace, expected_value)


@pytest.mark.parametrize("time_weighting", ["F", "S"])
def test_level_max_time_weighted_level_known_value(time_weighting):
    """A single impulse should result in a peak level that is equal to the
    level of the impulse dampened by the exponential smoothing parameters,
    which are the sampling rate and the time constant of the time weighting.
    """
    delay = 5432
    s = pf.signals.impulse(10000, delay)
    levels, times = pf.level.maximum_time_weighted_level(
        s, "Z", time_weighting, None, 1)
    time_constant = 0.125 if time_weighting == "F" else 1
    expected_peak_energy = 1 / s.sampling_rate / time_constant
    expected_peak_level = 10 * np.log10(expected_peak_energy)
    assert np.isclose(levels[0], expected_peak_level, atol=0.01)
    assert np.isclose(times[0], delay / s.sampling_rate, atol=0.001)


@pytest.mark.parametrize("time_w", ["F", "S"])
def test_level_peak_and_max_two_sample_peak(time_w):
    """Unlike the regular peak level, when two loud samples appear
    right next to each other and the first one is only slightly louder,
    the maximum time-weighted level should be at the second sample, since
    the time-weighting will integrate most of the first sample's energy
    into the output for the second sample.
    """
    s = pf.Signal([0, 0, 0, 1, 0.99, 0, 0, 0], sampling_rate=100)
    _, times_peak = pf.level.peak_level(s, "Z", None)
    _, times_max = pf.level.maximum_time_weighted_level(s, "Z", time_w, None)
    assert np.isclose(times_peak[0], 0.03)  # at the first sample of the peak
    assert np.isclose(times_max[0], 0.04)  # at the second sample of the peak
