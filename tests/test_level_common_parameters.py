"""
The standard-conform level functions have shared parameters that
can and should be tested under the exact same conditions.
All tests that apply to multiple of these functions are grouped
in this file to avoid code duplication or accidentally
testing the same parameter in different ways for different functions.

To do this, lambdas are used to wrap the level functions so they
can be called with the same number and order of parameters and the same
return type and shape, if necessary.
"""

import pyfar as pf
import numpy as np
import pytest

### signal parameter tests ###


@pytest.mark.parametrize("function", [
    lambda s: pf.level.equivalent_continuous_level(s, "Z"),
    # other level functions go here once implemented
])
def test_level_common_signal_parameter(function):
    # should not raise an error
    function(pf.signals.sine(1000, 22050))

    # all these should raise errors
    with pytest.raises(TypeError):
        function("not a signal")
    with pytest.raises(TypeError):
        function(123)
    with pytest.raises(TypeError):
        function(None)
    with pytest.raises(TypeError):
        function(np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        function(pf.TimeData(np.array([1, 2, 3]), [1, 2, 3]))
    with pytest.raises(TypeError):
        function(pf.FrequencyData(np.array([1, 2, 3]), [1, 2, 3]))


### frequency_weighting parameter tests ###

FUNCTION_WRAPPERS_FREQ_WEIGHTING = [
    lambda s, w: pf.level.equivalent_continuous_level(s, w),
    # other level functions go here once implemented
]


@pytest.mark.parametrize(("frequency", "weightings_max_to_min"), [
    (63, ["Z", "C", "A"]),
    (4000, ["A", "Z", "C"]),
    (16000, ["Z", "A", "C"]),
])
@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_FREQ_WEIGHTING)
def test_level_common_freq_weighting_relative_level(
        frequency, weightings_max_to_min, function):
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
        "Z": function(s, "Z"),
        "C": function(s, "C"),
        "A": function(s, "A"),
    }
    assert mags[weightings_max_to_min[0]] > mags[weightings_max_to_min[1]]
    assert mags[weightings_max_to_min[1]] > mags[weightings_max_to_min[2]]


@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_FREQ_WEIGHTING)
def test_level_common_freq_weighting_errors(function):
    """Test that an invalid frequency weighting raises an error."""
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(ValueError, match="Frequency weighting"):
        function(s, "X")
    with pytest.raises(ValueError, match="Frequency weighting"):
        function(s, None)


### num_octave_band_fractions parameter tests ###

FUNCTION_WRAPPERS_BAND_FRACTIONS = [
    lambda s, n: pf.level.equivalent_continuous_level(s, "Z", n),
    # other level functions go here once implemented
]


@pytest.mark.parametrize("signal", [
    pf.signals.impulse(1000, sampling_rate=48000),
    pf.signals.impulse(1000, 0, (1, 2), sampling_rate=48000),
    pf.signals.impulse(1000, 0, ((1, 2), (3, 4)), sampling_rate=48000),
])
@pytest.mark.parametrize("num_fractions", [1, 3, 6])
@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_BAND_FRACTIONS)
def test_level_common_num_octave_band_fractions_dimensions(
        signal, function, num_fractions):
    """Test that the number of octave band fractions is applied correctly
    by checking the dimensions of the output.
    """
    band_freqs, _, _ = pf.constants.fractional_octave_frequencies_exact(
        num_fractions)
    num_bands = len(band_freqs)

    full_band = function(signal, None)
    band_filtered = function(signal, num_fractions)

    # new first dimension with the same other dimensions
    assert band_filtered.shape == (num_bands, *full_band.shape)


@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_BAND_FRACTIONS)
def test_level_common_num_octave_band_fractions_errors(function):
    """Test that an invalid number of octave band fractions raises an error."""
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(TypeError, match="integer"):
        function(s, "3")
    with pytest.raises(TypeError, match="integer"):
        function(s, 2.5)
    with pytest.raises(ValueError, match="positive"):
        function(s, -1)
    with pytest.raises(ValueError, match="positive"):
        function(s, 0)


### reference_pressure parameter tests ###

FUNCTION_WRAPPERS_REFERENCE_PRESSURE = [
    lambda s, r: pf.level.equivalent_continuous_level(
        s, "Z", None, r),
    # other level functions go here once implemented
]


@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_REFERENCE_PRESSURE)
def test_level_common_reference_pressure_relative(function):
    """Test that the reference pressure is used correctly by comparing
    the norm SPL to dbFS, which must be ~94 dB apart.
    """
    s = pf.signals.impulse(22050)
    dbSPL = function(s, pf.constants.reference_sound_pressure)
    dbFS = function(s, 1)
    assert np.isclose(dbSPL - dbFS, 94, atol=0.1)


@pytest.mark.parametrize("function", FUNCTION_WRAPPERS_REFERENCE_PRESSURE)
def test_level_common_reference_pressure_errors(function):
    """Test that an invalid reference pressure raises an error."""
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(ValueError, match="Reference pressure"):
        function(s, 0)
