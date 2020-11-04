import numpy as np
import numpy.testing as npt
from pytest import raises

from pyfar.signal import Signal
import pyfar.arithmetic as arithmetic


# test adding two Signals
def test_add_two_signals():
    # generate test signal
    x = Signal([1, 0, 0], 44100)

    # time domain
    y = arithmetic.add((x, x), 'time')
    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)
    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([2, 0, 0]), atol=1e-15)

    # frequency domain
    y = arithmetic.add((x, x), 'freq')
    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)
    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'freq'
    npt.assert_allclose(y.freq, np.atleast_2d([2, 2]), atol=1e-15)


def test_add_three_signals():
    # generate and add signals
    x = Signal([1, 0, 0], 44100)
    y = arithmetic.add((x, x, x), 'time')

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([3, 0, 0]), atol=1e-15)


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
    s5 = Signal([1, 2, 3, 4], 44100, signal_type="power", fft_norm='amplitude')

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
    out = arithmetic._assert_match_for_arithmetic((s5, s4), 'time')
    assert out[2] == 'power'
    assert out[3] == 'amplitude'

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


def test_get_arithmetic_data_with_array():

    data_in = np.asarray(1)
    data_out = arithmetic._get_arithmetic_data(data_in, 1, None, None, None)
    npt.assert_allclose(data_in, data_out)


def test_get_arithmetic_data_with_signal():
    # all possible combinations of `domain`, `signal_type`, and `fft_norm`
    meta = [['time', 'energy', 'unitary'],
            ['freq', 'energy', 'unitary'],
            ['time', 'power', 'unitary'],
            ['freq', 'power', 'unitary'],
            ['time', 'power', 'amplitude'],
            ['freq', 'power', 'amplitude'],
            ['time', 'power', 'rms'],
            ['freq', 'power', 'rms'],
            ['time', 'power', 'power'],
            ['freq', 'power', 'power'],
            ['time', 'power', 'psd'],
            ['freq', 'power', 'psd']]

    for m_in in meta:
        # create input signal with current domain, type, and norm
        s_in = Signal([1, 0, 0], 44100, signal_type=m_in[1], fft_norm=m_in[2])
        s_in.domain = m_in[0]
        for m_out in meta:
            print(f"Testing from {', '.join(m_in)} to {', '.join(m_out)}.")
            # create output signal with current domain, type, and norm
            s_out = Signal([1, 0, 0], 44100,
                           signal_type=m_out[1], fft_norm=m_out[2])
            s_out.domain = m_out[0]

            # get output data
            data_out = arithmetic._get_arithmetic_data(
                s_in, n_samples=3, domain=m_out[0], signal_type=m_out[1],
                fft_norm=m_out[2])
            npt.assert_allclose(s_out._data, data_out, atol=1e-15)


def test_get_arithmetic_data_assertion():
    with raises(ValueError):
        arithmetic._get_arithmetic_data(
            Signal(1, 44100), 1, 'space', 'energy', 'unitary')
