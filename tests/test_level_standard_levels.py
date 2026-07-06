import pytest
import pyfar as pf
import numpy as np

# helpers that make it easier to test identical parameters across
# the different standard level functions despite some having
# different other paramters or returning multiple values
FUNCTION_WRAPPERS_FREQ_WEIGHTING = [
    lambda s, w: pf.level.time_weighted_level(s, w, "F"),
    lambda s, w: pf.level.equivalent_continuous_level(s, w),
    lambda s, w: pf.level.sliding_equivalent_continuous_level(s, w),
    lambda s, w: pf.level.exposure_level(s, w),
    lambda s, w: pf.level.peak_level(s, w)[0],
    lambda s, w: pf.level.maximum_time_weighted_level(s, w, "F")[0],
]

FUNCTION_WRAPPERS_REFERENCE_PRESSURE = [
    lambda s, r: pf.level.time_weighted_level(
        s, "Z", "F", reference_pressure=r),
    lambda s, r: pf.level.equivalent_continuous_level(
        s, "Z", reference_pressure=r),
    lambda s, r: pf.level.sliding_equivalent_continuous_level(
        s, "Z", reference_pressure=r),
    lambda s, r: pf.level.exposure_level(
        s, "Z", reference_energy=r**2),
    lambda s, r: pf.level.peak_level(
        s, "Z", reference_pressure=r)[0],
    lambda s, r: pf.level.maximum_time_weighted_level(
        s, "Z", "F", reference_pressure=r)[0],
]


@pytest.mark.parametrize(("frequency", "mags_descending"), [
    (63, ["Z", "C", "A"]),
    (4000, ["A", "Z", "C"]),
    (16000, ["Z", "A", "C"]),
])
@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_FREQ_WEIGHTING)
def test_level_standard_levels_freq_weighting_level(
        frequency, mags_descending, function):
    """Test that the frequency weighting is applied correctly
    by making sure the magnitudes are in the expected order relative
    to each other. This intentionally avoids checking absolute values,
    since the weighting uses approximated filters which should be tested
    in the weighting filter function's tests.
    """
    # start with small phase to avoid the first sample being zero,
    # which would lead to -inf dB values and division by zero warnings
    s = pf.signals.sine(frequency, 22050, phase=0.01)
    mags = {
        "Z": pf.level.average_levels(function(s, "Z")),
        "C": pf.level.average_levels(function(s, "C")),
        "A": pf.level.average_levels(function(s, "A")),
    }
    assert mags[mags_descending[0]] > mags[mags_descending[1]]
    assert mags[mags_descending[1]] > mags[mags_descending[2]]

@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_FREQ_WEIGHTING)
def test_level_standard_levels_freq_weighting_errors(function):
    """Test that an invalid frequency weighting raises an error."""
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(ValueError, match="Frequency weighting"):
        function(s, "X")

@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_REFERENCE_PRESSURE)
def test_level_standard_levels_reference_pressure_relative(function):
    """Test that the reference pressure is used correctly by comparing
    the norm pressure to dbFS, which must be ~94 dB apart.
    """
    s = pf.signals.impulse(22050)
    dbSPL = pf.level.average_levels(function(s, 2e-5))
    dbFS = pf.level.average_levels(function(s, 1))
    assert np.isclose(dbSPL - dbFS, 94, atol=0.1)

@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_REFERENCE_PRESSURE)
def test_level_standard_levels_reference_pressure_errors(function):
    """Test that an invalid reference pressure raises an error."""
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(ValueError, match="Reference pressure"):
        function(s, 0)

@pytest.mark.parametrize("time_weighting", ["F", "S"])
def test_level_time_weighted_level_reference_pressure(time_weighting):
    s = pf.Signal(np.ones(44100 * 60), 44100)
    levels = pf.level.time_weighted_level(
        s, "Z", time_weighting, reference_pressure=2e-5)
    assert np.isclose(levels[0][-1], 94, atol=0.1)

def test_level_equivalent_continuous_level_reference_pressure():
    s = pf.signals.impulse(22050)
    levels = pf.level.equivalent_continuous_level(
        s, "Z", reference_pressure=2e-5)
    assert np.isclose(levels, 94, atol=0.1)
