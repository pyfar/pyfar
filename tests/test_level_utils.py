import pyfar as pf
import numpy as np
import pytest


@pytest.mark.parametrize("window_size", [2, 3, 100, 101])
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("signal_data", [
    np.repeat([0, 1, -0.5], 400),
    np.array([np.linspace(0, 1, 1000), np.linspace(1, 0, 1000)]),
])
def test_moving_average_same_results(signal_data, window_size, cyclic):

    signal = pf.Signal(signal_data, 48000)
    filt = pf.Signal(np.ones(window_size) / window_size, 48000)
    mode = "cyclic" if cyclic else "cut"
    expected = pf.dsp.convolve(signal, filt, mode=mode).time

    result = pf.level._utils._moving_average(
        signal_data, window_size, -1, cyclic, False)
    result = np.atleast_2d(result)
    print(result)

    assert expected.shape == result.shape
    assert np.allclose(expected, result)


@pytest.mark.parametrize(("center_window", "expected"),
                         [(False, [0, 0, 1/3, 1/3, 1/3, 0]),
                          (True, [0, 1/3, 1/3, 1/3, 0, 0])])
def test_moving_average_center_window(center_window: bool, expected):
    data = np.array([0, 0, 1, 0, 0, 0])
    result = pf.level._utils._moving_average(data, 3, -1,
                                             False, center_window)
    print(result)
    assert np.allclose(expected, result)


# ── average_levels ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_type", [list, tuple, np.array])
def test_average_levels_input_types(input_type):
    # Two equal levels must average to themselves
    levels = input_type([60.0, 60.0])
    assert np.isclose(pf.level.average_levels(levels), 60.0)


def test_average_levels_single_value():
    assert np.isclose(pf.level.average_levels([40.0]), 40.0)


def test_average_levels_unequal():
    levels = [60, 70]
    energies = 10 ** (np.array(levels) / 10)
    expected_energy = np.mean(energies)
    expected = 10 * np.log10(expected_energy)
    assert np.isclose(pf.level.average_levels(levels), expected)


def test_average_levels_2d_axis_none():
    # mean of equal values is that value
    levels = np.full((3, 4), 50.0)
    assert np.isclose(pf.level.average_levels(levels, axis=None), 50.0)


def test_average_levels_2d_axis_last():
    # per-row average equals row level
    levels = np.array([[60.0, 60.0],
                       [70.0, 70.0]])
    result = pf.level.average_levels(levels, axis=-1)
    assert np.allclose(result, [60.0, 70.0])


def test_average_levels_2d_axis_first():
    # per-column average equals column level
    levels = np.array([[60.0, 70.0],
                       [60.0, 70.0]])
    result = pf.level.average_levels(levels, axis=0)
    assert np.allclose(result, [60.0, 70.0])


def test_average_levels_complex_raises():
    with pytest.raises(ValueError, match="real-valued"):
        pf.level.average_levels([1+2j, 3+4j])


# ── sum_levels ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_type", [list, tuple, np.array])
def test_sum_levels_input_types(input_type):
    # Summing two equal levels raises the result by 10*log10(2)
    levels = input_type([60.0, 60.0])
    expected = 60.0 + 10 * np.log10(2)
    assert np.isclose(pf.level.sum_levels(levels), expected)


def test_sum_levels_single_value():
    assert np.isclose(pf.level.sum_levels([40.0]), 40.0)


def test_sum_levels_unequal():
    energies = [0.3, 0.5]
    expected_energy = sum(energies)
    levels = 10 * np.log10(energies)
    expected = 10 * np.log10(expected_energy)
    assert np.isclose(pf.level.sum_levels(levels), expected)


def test_sum_levels_2d_axis_none():
    # Four equal values of 60 dB: sum of energies = 4 * 10^6
    levels = np.full((2, 2), 60.0)
    expected = 10 * np.log10(4 * 10 ** (60 / 10))
    assert np.isclose(pf.level.sum_levels(levels, axis=None), expected)


def test_sum_levels_2d_axis_last():
    # Each row sums two equal levels
    levels = np.array([[60.0, 60.0],
                       [70.0, 70.0]])
    expected = np.array([60.0, 70.0]) + 10 * np.log10(2)
    result = pf.level.sum_levels(levels, axis=-1)
    assert np.allclose(result, expected)


def test_sum_levels_2d_axis_first():
    levels = np.array([[60.0, 70.0],
                       [60.0, 70.0]])
    expected = np.array([60.0, 70.0]) + 10 * np.log10(2)
    result = pf.level.sum_levels(levels, axis=0)
    assert np.allclose(result, expected)


def test_sum_levels_complex_raises():
    with pytest.raises(ValueError, match="real-valued"):
        pf.level.sum_levels([1+2j, 3+4j])
