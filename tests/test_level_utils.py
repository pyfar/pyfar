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
