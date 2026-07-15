import pyfar as pf
import numpy as np
import pytest


@pytest.mark.parametrize("window_size", [2, 3, 100, 101])
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("signal_data", [
    np.repeat([0, 1, -0.5], 400),
    np.array([np.linspace(0, 1, 1000), np.linspace(1, 0, 1000)]),
])
def test_level_moving_average_same_results(signal_data, window_size, cyclic):
    """Test that the moving average function returns the same result as
    convolving with a rectangular window using pyfar.dsp.convolve.
    """
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


@pytest.mark.parametrize(("center_window", "window_size", "expected"),
                         [(False, 3, [0, 0, 1/3, 1/3, 1/3, 0]),
                          (True, 3, [0, 1/3, 1/3, 1/3, 0, 0]),
                          (True, 4, [0, 1/4, 1/4, 1/4, 1/4, 0]),
                          (True, 2, [0, 0, 1/2, 1/2, 0, 0]),
                          ])
def test_level_moving_average_center_window(center_window,
                                            window_size,
                                            expected):
    data = np.array([0, 0, 1, 0, 0, 0])
    result = pf.level._utils._moving_average(data, window_size, -1,
                                             False, center_window)
    print(result)
    assert np.allclose(expected, result)


@pytest.mark.parametrize(("window_size", "err", "match"), [
    (0, ValueError, "positive"),
    (-1, ValueError, "positive"),
    (2.5, TypeError, "integer"),
])
def test_level_moving_average_invalid_window_size(window_size, err, match):
    """Test that the moving average function raises errors if the
    window size is not a positive integer.
    """
    with pytest.raises(err, match=match):
        pf.level._utils._moving_average(np.array([0, 1, 2]), window_size)
