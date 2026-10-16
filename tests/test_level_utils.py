import pytest
import pyfar as pf
import numpy as np


@pytest.mark.parametrize("time_constant", [0.125, 1])
@pytest.mark.parametrize("amplitude", [1, [1, 1]])
def test_level_time_weighted_pressure_shape(time_constant, amplitude):
    fs = 48000
    impulse = pf.signals.impulse(fs + 1, 0, amplitude, fs)
    weighted = pf.level.time_weighted_pressure(impulse, time_constant)

    assert isinstance(weighted, pf.TimeData)
    assert impulse.time.shape == weighted.time.shape


@pytest.mark.parametrize("fs", [44100, 48000, 192000])
@pytest.mark.parametrize(("time_constant", "expected_decay"),
                         [(0.125, -34.7), (1, -4.3)])
def test_level_time_weighted_pressure_decay(time_constant, expected_decay, fs):
    """
    According to DIN EN 61672-1 §5.8.2, after an impulse the level
    should fall of at a rate of 34.7 dB/s for FAST and 4.3 dB/s for the
    SLOW time weighting.
    """
    impulse = pf.signals.impulse(2 * fs, sampling_rate=fs)

    weighted = pf.level.time_weighted_pressure(impulse, time_constant)
    levels = 10 * np.log10(weighted.time ** 2)
    diff1 = levels[0][fs] - levels[0][0]
    diff2 = levels[0][fs + 100] - levels[0][100]
    assert abs(diff1 - expected_decay) < 0.1
    assert abs(diff2 - expected_decay) < 0.1


@pytest.mark.parametrize("time_constant", [0.125, 1])
def test_level_time_weighted_pressure_same_level(time_constant):
    """The energy of a long 1kHz signal must be identical
    across F, S and eq levels (DIN EN 61672-1 §5.8.3).
    """
    fs = 48000
    duration = 60
    x = pf.signals.sine(1000, fs * duration, sampling_rate=48000)

    L_eq = 10 * np.log10(np.sum(x.time[0]**2) / x.n_samples)
    weighted = pf.level.time_weighted_pressure(x, time_constant)
    # compare last sample to the equivalent average,
    # so the exp-filter has enough time to integrate energy
    level = 10 * np.log10(weighted.time[0][-1] ** 2)
    assert abs(L_eq - level) < 0.1


@pytest.mark.parametrize(("time_constant", "err_type"), [
    (None, TypeError),
    ("F", TypeError),
    ("S", TypeError),
    (-1, ValueError),
    (0, ValueError),
    (np.nan, ValueError),
    (np.inf, ValueError),
])
def test_level_time_weighted_pressure_time_constant_error(
    time_constant, err_type,
):
    s = pf.signals.sine(1000, 22050)
    with pytest.raises(err_type, match="Time constant"):
        pf.level.time_weighted_pressure(s, time_constant)


TIMES_TWO_IN_DB = 10 * np.log10(2)


@pytest.mark.parametrize(("levels", "function", "expected"), [
    ([40, 40], pf.level.energetic_mean, 40),
    ([40, 40.0], pf.level.energetic_sum, 40 + TIMES_TWO_IN_DB),
    ([-40, -40.0], pf.level.energetic_mean, -40),
    ([-40, -40], pf.level.energetic_sum, -40 + TIMES_TWO_IN_DB),
    ([27.4, 34.6], pf.level.energetic_mean, 32.34716),
    ([27.4, 34.6], pf.level.energetic_sum, 35.35746),
])
def test_level_energetic_helpers_known_values(levels, function, expected):
    """Test that the energetic helpers return the expected values
    for known inputs.
    """
    result = function(levels)
    assert np.allclose(result, expected, atol=1e-5)


def test_level_energetic_helpers_empty_input():
    """Test that the energetic helpers return an empty array for empty input.
    """
    assert np.allclose(pf.level.energetic_mean([]), np.array([]))
    assert np.allclose(pf.level.energetic_sum([]), np.array([]))


@pytest.mark.parametrize("function", [
    pf.level.energetic_mean,
    pf.level.energetic_sum,
])
@pytest.mark.parametrize(("levels", "axis", "expected"), [
    ([40], None, 40),
    ([40], 0, 40),
    ([40], -1, 40),
])
def test_level_energetic_helpers_axis_single(function, levels, axis, expected):
    """Test that a single level input is returned unchanged regardless
    of the axis argument.
    """
    result = function(levels, axis=axis)
    assert isinstance(result, np.number)
    assert np.isclose(result, expected)


@pytest.mark.parametrize(("levels", "axis", "expected"), [
    ([40, 40], 0, 40),
    ([40, 40], -1, 40),
    ([40, 40], None, 40),
])
def test_level_energetic_helpers_axis_1d(levels, axis, expected):
    """Test that the axis argument works like in numpy.mean and numpy.sum."""
    e_mean = pf.level.energetic_mean(levels, axis=axis)
    e_sum = pf.level.energetic_sum(levels, axis=axis)

    assert isinstance(e_mean, np.number)
    assert isinstance(e_sum, np.number)
    assert np.isclose(e_mean, expected)
    assert np.isclose(e_sum, expected + TIMES_TWO_IN_DB)


@pytest.mark.parametrize(("levels", "axis", "expected"), [
    ([[40, 40], [50, 50]], 1, np.array([40, 50])),
    ([[40, 40], [50, 50]], -1, np.array([40, 50])),
    ([[40, 50], [40, 50]], 0, np.array([40, 50])),
    ([[40, 50], [40, 50]], -2, np.array([40, 50])),
])
def test_level_energetic_helpers_axis_2d(levels, axis, expected):
    """Test that the axis argument works like in numpy.mean and numpy.sum."""
    e_mean = pf.level.energetic_mean(levels, axis=axis)
    e_sum = pf.level.energetic_sum(levels, axis=axis)

    assert isinstance(e_mean, np.ndarray)
    assert isinstance(e_sum, np.ndarray)
    assert e_mean.shape == expected.shape
    assert e_sum.shape == expected.shape
    assert np.allclose(e_mean, expected)
    assert np.allclose(e_sum, expected + TIMES_TWO_IN_DB)


@pytest.mark.parametrize("input_type", [list, tuple, np.array])
@pytest.mark.parametrize("function", [
    pf.level.energetic_mean,
    pf.level.energetic_sum,
])
def test_level_energetic_helpers_input_types(input_type, function):
    """Test that the energetic helpers can handle different
    array-like input types.
    """
    levels = input_type([56, 98])
    result = function(levels)
    assert isinstance(result, np.number)


@pytest.mark.parametrize("function", [
    pf.level.energetic_mean,
    pf.level.energetic_sum,
])
def test_sum_levels_complex_raises(function):
    with pytest.raises(ValueError, match="real-valued"):
        function([1+2j, 3+4j])
