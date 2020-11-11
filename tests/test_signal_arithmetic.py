import numpy as np
import numpy.testing as npt
from pytest import raises

import pyfar.signal as signal
from pyfar.signal import Signal


# test adding two Signals
def test_add_two_signals_time():
    # generate test signal
    x = Signal([1, 0, 0], 44100)

    # time domain
    y = signal.add((x, x), 'time')
    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)
    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([2, 0, 0]), atol=1e-15)


# test adding two Signals
def test_add_two_signals_freq():
    # generate test signal
    x = Signal([1, 0, 0], 44100)

    # frequency domain
    y = signal.add((x, x), 'freq')
    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)
    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'freq'
    npt.assert_allclose(y.freq, np.atleast_2d([2, 2]), atol=1e-15)


# test adding three signals
def test_add_three_signals():
    # generate and add signals
    x = Signal([1, 0, 0], 44100)
    y = signal.add((x, x, x), 'time')

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
    y = signal.add((x, 1), 'time')

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
    y = signal.add((1, x), 'time')

    # check if old signal did not change
    npt.assert_allclose(x.time, np.atleast_2d([1, 0, 0]), atol=1e-15)

    # check result
    assert isinstance(y, Signal)
    assert y.domain == 'time'
    npt.assert_allclose(y.time, np.atleast_2d([2, 1, 1]), atol=1e-15)


def test_subtraction():
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100)
    y = Signal([0, 1, 0], 44100)
    z = signal.subtract((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([1, -1, 0]), atol=1e-15)


def test_multiplication():
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100)
    y = Signal([0, 1, 0], 44100)
    z = signal.multiply((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([0, 0, 0]), atol=1e-15)


def test_division():
    # only test one case - everything else is tested below
    x = Signal([1, 0, 0], 44100)
    y = Signal([2, 2, 2], 44100)
    z = signal.divide((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([0.5, 0, 0]), atol=1e-15)


def test_power():
    # only test one case - everything else is tested below
    x = Signal([2, 1, 0], 44100)
    y = Signal([2, 2, 2], 44100)
    z = signal.power((x, y), 'time')

    # check result
    npt.assert_allclose(z.time, np.atleast_2d([4, 1, 0]), atol=1e-15)


def test_overloaded_operators():
    x = Signal([2, 1, 0], 44100, n_samples=5, domain='freq')
    y = Signal([2, 2, 2], 44100, n_samples=5, domain='freq')

    # addition
    z = x + y
    npt.assert_allclose(z.freq, np.array([4, 3, 2], ndmin=2), atol=1e-15)
    # subtraction
    z = x - y
    npt.assert_allclose(z.freq, np.array([0, -1, -2], ndmin=2), atol=1e-15)
    # multiplication
    z = x * y
    npt.assert_allclose(z.freq, np.array([4, 2, 0], ndmin=2), atol=1e-15)
    # division
    z = x / y
    npt.assert_allclose(z.freq, np.array([1, .5, 0], ndmin=2), atol=1e-15)
    # power
    z = x**y
    npt.assert_allclose(z.freq, np.array([4, 1, 0], ndmin=2), atol=1e-15)


def test_assert_match_for_arithmetic():
    s = Signal([1, 2, 3, 4], 44100)
    s1 = Signal([1, 2, 3, 4], 48000)
    s2 = Signal([1, 2, 3], 44100)
    s4 = Signal([1, 2, 3, 4], 44100, signal_type="power")
    s5 = Signal([1, 2, 3, 4], 44100, signal_type="power", fft_norm='amplitude')

    # check with two signals
    signal._assert_match_for_arithmetic((s, s), 'time')
    # check with one signal and one array like
    signal._assert_match_for_arithmetic((s, [1, 2]), 'time')
    # check with more than two inputs
    signal._assert_match_for_arithmetic((s, s, s), 'time')

    # check output
    out = signal._assert_match_for_arithmetic((s, s), 'time')
    assert out[0] == 44100
    assert out[1] == 4
    assert out[2] == 'energy'
    out = signal._assert_match_for_arithmetic((s, s4), 'time')
    assert out[2] == 'power'
    out = signal._assert_match_for_arithmetic((s5, s4), 'time')
    assert out[2] == 'power'
    assert out[3] == 'amplitude'

    # check with only one argument
    with raises(TypeError):
        signal._assert_match_for_arithmetic((s, s))
    # check with single input
    with raises(ValueError):
        signal._assert_match_for_arithmetic(s, 'time')
    # check with invalid data type
    with raises(ValueError):
        signal._assert_match_for_arithmetic((s, ['str', 'ing']), 'time')
    # check with complex data and time domain signal
    with raises(ValueError):
        signal._assert_match_for_arithmetic(
            (s, np.array([1 + 1j])), 'time')
    # test signals with different sampling rates
    with raises(ValueError):
        signal._assert_match_for_arithmetic((s, s1), 'time')
    # test signals with different n_samples
    with raises(ValueError):
        signal._assert_match_for_arithmetic((s, s2), 'time')


def test_get_arithmetic_data_with_array():

    data_in = np.asarray(1)
    data_out = signal._get_arithmetic_data(data_in, None, None)
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

    # reference signal - _get_arithmetic_data should return the data without
    # any normalization regardless of the input data
    s_ref = Signal([1, 0, 0], 44100)

    for m_in in meta:
        # create input signal with current domain, type, and norm
        s_in = Signal([1, 0, 0], 44100, signal_type=m_in[1], fft_norm=m_in[2])
        s_in.domain = m_in[0]
        for domain in ['time', 'freq']:
            print(f"Testing from {', '.join(m_in)} to {domain}.")

            # get output data
            data_out = signal._get_arithmetic_data(
                s_in, n_samples=3, domain=domain)
            if domain == 'time':
                npt.assert_allclose(s_ref.time, data_out, atol=1e-15)
            elif domain == 'freq':
                npt.assert_allclose(s_ref.freq, data_out, atol=1e-15)


def test_get_arithmetic_data_assertion():
    with raises(ValueError):
        signal._get_arithmetic_data(
            Signal(1, 44100), 1, 'space')
