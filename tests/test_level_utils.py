import warnings
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


@pytest.mark.parametrize("function", [
    pf.level.energetic_mean,
    pf.level.energetic_sum,
])
def test_level_energetic_helpers_empty_input(function):
    """Test that the energetic helpers raise for empty input arrays."""
    with pytest.raises(ValueError, match="Levels must not be empty."):
        function([])


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


@pytest.mark.parametrize(("levels", "axis"), [
    ([[40, 40], [50, 50]], 1),
    ([[40, 40], [50, 50]], -1),
    ([[40, 50], [40, 50]], 0),
    ([[40, 50], [40, 50]], -2),
])
def test_level_energetic_helpers_axis_2d(levels, axis):
    """Test that the axis argument works like in numpy.mean and numpy.sum."""
    expected = np.array([40, 50])
    e_mean = pf.level.energetic_mean(levels, axis=axis)
    e_sum = pf.level.energetic_sum(levels, axis=axis)

    assert isinstance(e_mean, np.ndarray)
    assert isinstance(e_sum, np.ndarray)
    assert e_mean.shape == expected.shape
    assert e_sum.shape == expected.shape
    assert np.allclose(e_mean, expected)
    assert np.allclose(e_sum, expected + TIMES_TWO_IN_DB)


@pytest.mark.parametrize("axis", [(0, 1), (1, 0), (-3, -2)])
def test_level_energetic_helpers_axis_tuple(axis):
    """Test that a tuple of axes works like in numpy.mean and numpy.sum."""
    levels = [[[40, 50], [40, 50]], [[40, 50], [40, 50]]]
    expected = np.array([40, 50])
    times_four_in_db = 10 * np.log10(4)
    e_mean = pf.level.energetic_mean(levels, axis=axis)
    e_sum = pf.level.energetic_sum(levels, axis=axis)

    assert e_mean.shape == expected.shape
    assert e_sum.shape == expected.shape
    assert np.allclose(e_mean, expected)
    assert np.allclose(e_sum, expected + times_four_in_db)


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
def test_level_energetic_helpers_complex_input_raises_error(function):
    """Test that the energetic helpers raise an error
    for input arrays with complex values, since levels must be real-valued.
    """
    with pytest.raises(ValueError, match="real-valued"):
        function([1+2j, 3+4j])


DECIBEL_FUNCTIONS = [pf.level.to_decibels, pf.level.from_decibels]
QUANTITIES = ["power", "amplitude"]
P0 = pf.constants.reference_sound_pressure
SPL_OF_ONE_PASCAL = 20 * np.log10(1 / P0)


@pytest.mark.parametrize(("data", "quantity", "reference", "expected"), [
    (1, "amplitude", P0, SPL_OF_ONE_PASCAL),
    (1, "power", P0, SPL_OF_ONE_PASCAL),
    (2, "amplitude", 1, 20 * np.log10(2)),
    (2, "power", 1, 10 * np.log10(2)),
    (10, "amplitude", 1, 20),
    (10, "power", 1, 10),
    (1, "amplitude", 0.5, 20 * np.log10(2)),
    (1, "power", 0.5, 20 * np.log10(2)),
])
def test_level_decibels_known_values(data, quantity, reference, expected):
    """Test that both conversions return the expected values for known
    inputs, with the reference always given as a pressure.
    """
    levels = pf.level.to_decibels(data, quantity, reference)
    data_from_levels = pf.level.from_decibels(expected, quantity, reference)
    assert np.isclose(levels, expected)
    assert np.isclose(data_from_levels, data)


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_default_reference(quantity):
    """Test that the default reference is the reference sound pressure,
    so that 1 Pa or 1 Pa² correspond to ~94 dB SPL.
    """
    levels = pf.level.to_decibels(1, quantity)
    data = pf.level.from_decibels(SPL_OF_ONE_PASCAL, quantity)
    assert np.isclose(levels, SPL_OF_ONE_PASCAL)
    assert np.isclose(data, 1)


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("reference", [1, 0.5])
def test_level_decibels_zero_decibels_is_reference(quantity, reference):
    """Test that 0 dB is converted to the reference pressure for amplitudes
    and to the squared reference pressure for power quantities.
    """
    expected = reference if quantity == "amplitude" else reference**2
    result = pf.level.from_decibels(0, quantity, reference)
    assert np.isclose(result, expected)


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("reference", [1, P0])
@pytest.mark.parametrize("data", [
    0.3,
    [1e-3, 1, 1e3],
    [[0.1, 0.2], [3, 40]],
])
def test_level_decibels_round_trip_data(quantity, reference, data):
    """Test that from_decibels inverts to_decibels for non-negative
    real data.
    """
    levels = pf.level.to_decibels(data, quantity, reference)
    result = pf.level.from_decibels(levels, quantity, reference)
    assert np.allclose(result, data)


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("reference", [1, P0])
@pytest.mark.parametrize("levels", [
    -20.5,
    [-100, 0, 94],
    [[-3, 6], [60, 120]],
])
def test_level_decibels_round_trip_levels(quantity, reference, levels):
    """Test that to_decibels inverts from_decibels for any real levels."""
    data = pf.level.from_decibels(levels, quantity, reference)
    result = pf.level.to_decibels(data, quantity, reference)
    assert np.allclose(result, levels)


@pytest.mark.parametrize("data", [0.5, -2, 1j, 3+4j, [1e-3, 7]])
def test_level_decibels_amplitude_matches_power(data):
    """Test that converting amplitudes is equivalent to converting their
    squared absolute values as power quantities.
    """
    from_amplitude = pf.level.to_decibels(data, "amplitude")
    from_power = pf.level.to_decibels(np.abs(data)**2, "power")
    assert np.allclose(from_amplitude, from_power)


@pytest.mark.parametrize("data", [-2, 2j, 3+4j, [-1, 1j, -3-4j]])
def test_level_decibels_amplitude_ignores_sign_and_phase(data):
    """Test that negative and complex amplitudes are converted by their
    absolute value.
    """
    result = pf.level.to_decibels(data, "amplitude")
    expected = pf.level.to_decibels(np.abs(data), "amplitude")
    assert np.allclose(result, expected)


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("data", [0.1, 1, 10])
def test_level_decibels_to_decibels_reference_relative(quantity, data):
    """Test that to_decibels uses the reference pressure correctly by
    comparing the norm SPL to dBFS, which must be ~94 dB apart.
    """
    db_spl = pf.level.to_decibels(data, quantity, P0)
    db_fs = pf.level.to_decibels(data, quantity, 1)
    assert np.isclose(db_spl - db_fs, SPL_OF_ONE_PASCAL)


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("levels", [0, 60])
def test_level_decibels_from_decibels_reference_relative(quantity, levels):
    """Test that from_decibels uses the reference pressure correctly. The
    same level in dB SPL must correspond to data ~94 dB weaker than in dBFS.
    """
    from_spl = pf.level.from_decibels(levels, quantity, P0)
    from_fs = pf.level.from_decibels(levels, quantity, 1)
    ratio_in_db = pf.level.to_decibels(from_spl / from_fs, quantity, 1)
    assert np.isclose(ratio_in_db, -SPL_OF_ONE_PASCAL)


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)])
def test_level_decibels_output_shape(function, quantity, shape):
    """Test that the output has the same shape as the input."""
    result = function(np.ones(shape), quantity)
    assert isinstance(result, np.ndarray)
    assert result.shape == shape


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_scalar_output(function, quantity):
    """Test that scalar input returns a scalar."""
    result = function(2, quantity)
    assert isinstance(result, np.number)


@pytest.mark.parametrize("input_type", [list, tuple, np.array])
@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_input_types(input_type, function, quantity):
    """Test that the conversions can handle different array-like input
    types.
    """
    result = function(input_type([1, 2]), quantity)
    expected = function(np.array([1., 2.]), quantity)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_integer_input(function, quantity):
    """Test that integer input returns float output."""
    result = function(np.array([1, 2, 3]), quantity)
    assert np.issubdtype(result.dtype, np.floating)


def test_level_decibels_large_integer_amplitude():
    """Test that squaring large integer amplitudes does not overflow."""
    result = pf.level.to_decibels(2**40, "amplitude", 1)
    assert np.isclose(result, 20 * np.log10(2**40))


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("reference", [
    1, 1.0, np.float64(1), np.float32(1), np.int64(1),
])
def test_level_decibels_reference_types(function, reference):
    """Test that the reference pressure can be any real number, including
    numpy scalars.
    """
    result = function(10, "power", reference)
    assert np.isclose(result, function(10, "power", 1))


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_zeros_without_replacing(quantity):
    """Test that zeros result in -inf and a numpy warning if they are not
    replaced.
    """
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        result = pf.level.to_decibels([0, 1], quantity, 1)
    assert result[0] == -np.inf


@pytest.mark.parametrize("quantity", QUANTITIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_level_decibels_replace_zeros(quantity, dtype):
    """Test that zeros are replaced by the float epsilon without warnings,
    using the epsilon of the data type after squaring amplitudes.
    """
    reference = 0.5
    data = np.array([0, 1], dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error") # turn warnings into errors
        result = pf.level.to_decibels(
            data, quantity, reference, replace_zeros=True)

    # integers are converted to float64 before replacing zeros
    eps_dtype = np.float64 if dtype == np.int64 else dtype
    expected = 10 * np.log10(np.finfo(eps_dtype).eps / reference**2)
    assert np.isclose(result[0], expected)


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_replace_zeros_keeps_non_zeros(quantity):
    """Test that replacing zeros does not change any other values."""
    data = [1e-10, 0.5, 2]
    replaced = pf.level.to_decibels(data, quantity, replace_zeros=True)
    unchanged = pf.level.to_decibels(data, quantity, replace_zeros=False)
    assert np.array_equal(replaced, unchanged)


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("quantity", ["energy", "Power", "", None])
def test_level_decibels_invalid_quantity(function, quantity):
    """Test that invalid quantities raise an error."""
    with pytest.raises(ValueError, match="Quantity"):
        function(1, quantity)


@pytest.mark.parametrize(("data", "match"), [
    ([1j], "real-valued"),
    ([1+0j], "real-valued"),
    ([-1], "negative"),
    ([1, -1], "negative"),
])
def test_level_decibels_invalid_power_input(data, match):
    """Test that complex and negative power quantities raise an error.
    Complex data must raise even if the imaginary part is zero.
    """
    with pytest.raises(ValueError, match=match):
        pf.level.to_decibels(data, "power")


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_complex_levels(quantity):
    """Test that complex levels raise an error, since levels must be
    real-valued.
    """
    with pytest.raises(ValueError, match="real-valued"):
        pf.level.from_decibels([1+2j], quantity)


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize(("reference", "error"), [
    (0, ValueError),
    (-1, ValueError),
    (True, TypeError),
    (False, TypeError),
    ("1", TypeError),
    (1j, TypeError),
    (np.array([1., 2]), TypeError),
    (None, TypeError),
])
def test_level_decibels_invalid_reference(function, reference, error):
    """Test that invalid reference pressures raise an error."""
    with pytest.raises(error, match="Reference pressure"):
        function(1, "power", reference)


@pytest.mark.parametrize("function", DECIBEL_FUNCTIONS)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_empty_input(function, quantity):
    """Test that empty input returns an empty array. Unlike for the
    energetic helpers, this is a valid conversion.
    """
    result = function([], quantity)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_level_decibels_non_finite_values(quantity):
    """Test that infinite and nan values are passed through."""
    assert pf.level.to_decibels(np.inf, quantity) == np.inf
    assert np.isnan(pf.level.to_decibels(np.nan, quantity))
    assert pf.level.from_decibels(-np.inf, quantity) == 0
    assert pf.level.from_decibels(np.inf, quantity) == np.inf
    assert np.isnan(pf.level.from_decibels(np.nan, quantity))
