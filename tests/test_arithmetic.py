import numpy as np
import numpy.testing as npt
from pytest import raises

from pyfar.signal import Signal
import pyfar.arithmetic as arithmetic


# test adding two Signals
def test_add_two_signals():
    # generate and add signals
    x = Signal([1, 0, 0], 44100)
    y = arithmetic.add((x, x), 'time')

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([2, 0, 0]), atol=1e-15)


# test add Signals and number
def test_add_signal_and_number():
    # generate and add signals
    x = Signal([1, 0, 0], 44100)
    y = arithmetic.add((x, 1), 'time')

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([2, 1, 1]), atol=1e-15)


# test add number and Signal
def test_add_number_and_signal():
    # generate and add signals
    x = Signal([1, 0, 0], 44100)
    y = arithmetic.add((1, x), 'time')

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([2, 1, 1]), atol=1e-15)


def test_assert_match_for_arithmetic():
    s = Signal([1, 2, 3, 4], 44100)
    s1 = Signal([1, 2, 3, 4], 48000)
    s2 = Signal([1, 2, 3], 44100)
    s4 = Signal([1, 2, 3, 4], 44100, signal_type="power")

    # check with two signals
    arithmetic._assert_match_for_arithmetic((s, s), 'time')
    # check with one signal and one array like
    arithmetic._assert_match_for_arithmetic((s, [1, 2]), 'time')
    # check with more than two inputs
    arithmetic._assert_match_for_arithmetic((s, s, s), 'time')

    # check output
    out = arithmetic._assert_match_for_arithmetic((s, s), 'time')
    assert out[0] == 44100
    assert out[1] == 4
    assert out[2] == 'energy'
    out = arithmetic._assert_match_for_arithmetic((s, s4), 'time')
    assert out[2] == 'power'

    # check with only one argument
    with raises(TypeError):
        arithmetic._assert_match_for_arithmetic((s, s))
    # check with single input
    with raises(ValueError):
        arithmetic._assert_match_for_arithmetic(s, 'time')
    # check with invalid data type
    with raises(ValueError):
        arithmetic._assert_match_for_arithmetic((s, ['str', 'ing']), 'time')
    # check with complex data and time domain signal
    with raises(ValueError):
        arithmetic._assert_match_for_arithmetic(
            (s, np.array([1 + 1j])), 'time')
    # test signals with different sampling rates
    with raises(ValueError):
        arithmetic._assert_match_for_arithmetic((s, s1), 'time')
    # test signals with different n_samples
    with raises(ValueError):
        arithmetic._assert_match_for_arithmetic((s, s2), 'time')
