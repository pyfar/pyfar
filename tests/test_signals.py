import numpy as np
import numpy.testing as npt
import pytest
from pyfar import Signal
import pyfar.dsp.filter as pff
import pyfar.signals as pfs
from pyfar.signals.signals import _get_common_shape, _match_shape


def test_sine_with_defaults():
    """Test sine signal with default parameters."""

    signal = pfs.sine(99, 441)
    sin = np.sin(np.arange(441) / 44100 * 2 * np.pi * 99)

    assert isinstance(signal, Signal)
    assert signal.comment == "f = 99 Hz"
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == "rms"
    assert signal.time.shape == (1, 441)
    npt.assert_allclose(signal.time, np.atleast_2d(sin))


def test_sine_user_parameters():
    """Test sine signal with custom amplitude, phase, and sampling rate."""
    signal = pfs.sine(99, 441, 2, np.pi/4, 48000)
    sin = np.sin(np.arange(441) / 48000 * 2 * np.pi * 99 + np.pi/4) * 2
    npt.assert_allclose(signal.time, np.atleast_2d(sin))


def test_sine_full_period():
    """Test sine signal with full period option."""
    signal = pfs.sine(99, 441, full_period=True)
    sin = np.sin(np.arange(441) / 44100 * 2 * np.pi * 100)

    assert signal.comment == "f = 100 Hz"
    npt.assert_allclose(signal.time, np.atleast_2d(sin))


def test_sine_multi_channel():
    """Test multi channel sine."""
    signal = pfs.sine([99, 50], 441)
    sin = np.concatenate(
        (np.atleast_2d(np.sin(np.arange(441) / 44100 * 2 * np.pi * 99)),
         np.atleast_2d(np.sin(np.arange(441) / 44100 * 2 * np.pi * 50))), 0)

    assert signal.comment == "f = [99 50] Hz"
    npt.assert_allclose(signal.time, sin)


def test_impulse_with_defaults():
    """Test impulse with default parameters"""
    signal = pfs.impulse(3)
    assert isinstance(signal, Signal)
    npt.assert_allclose(signal.time, np.atleast_2d([1, 0, 0]))
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == 'none'


def test_impulse_with_user_parameters():
    """Test impulse with custom delay, amplitude and sampling rate"""
    signal = pfs.impulse(3, 1, 2, 48000)
    npt.assert_allclose(signal.time, np.atleast_2d([0, 2, 0]))


def test_impulse_multi_channel():
    """Test multi channel impulse."""
    # test with array and number
    signal = pfs.impulse(3, [0, 1], 1)
    ref = np.atleast_2d([[1, 0, 0], [0, 1, 0]])
    npt.assert_allclose(signal.time, ref)
    # test with two arrays
    signal = pfs.impulse(3, [0, 1], [1, 2])
    ref = np.atleast_2d([[1, 0, 0], [0, 2, 0]])
    npt.assert_allclose(signal.time, ref)


def test_white_noise_with_defaults():
    """Test wite noise with default parameters."""
    signal = pfs.white_noise(100)

    assert isinstance(signal, Signal)
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == "rms"
    npt.assert_allclose(
        np.sqrt(np.mean(signal.time**2, axis=-1)), 1)


def test_white_noise_with_user_parameters():
    """Test wite noise with amplitude and sampling rate."""
    signal = pfs.white_noise(100, 2, 48000)

    assert signal.sampling_rate == 48000
    npt.assert_allclose(np.sqrt(np.mean(signal.time**2, axis=-1)), 2)


def test_white_noise_multi_channel():
    """Test wite noise with amplitude and sampling rate."""
    rms = [[1, 2, 3], [4, 5, 6]]
    signal = pfs.white_noise(100, rms)
    npt.assert_allclose(np.sqrt(np.mean(signal.time**2, axis=-1)), rms)


def test_white_noise_seed():
    """Test passing seeds to the random generator."""
    a = pfs.white_noise(100, seed=1)
    b = pfs.white_noise(100, seed=1)
    assert a == b

    a = pfs.white_noise(100)
    b = pfs.white_noise(100)
    with pytest.raises(AssertionError):
        assert a == b


def test_pink_noise_with_defaults():
    """
    Test only the defaults because pink noise uses the same private functions
    as white noise.
    """
    signal = pfs.pink_noise(100)

    assert isinstance(signal, Signal)
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == "rms"
    npt.assert_allclose(
        np.sqrt(np.mean(signal.time**2, axis=-1)), 1)


def test_pink_noise_rms_spectrum():
    """
    Test for constant energy across filters of constant relative band width.
    """
    # filtered pink noise
    # (use only center octaves, because the spectrum is less stochastic there)
    signal = pfs.pink_noise(5000, seed=1)
    signal = pff.fractional_octave_bands(signal, 1, freq_range=(1e3, 16e3))
    # check if stdandard deviation is less then 1%
    rms = np.atleast_1d(np.sqrt(np.mean(signal.time**2, axis=-1)))
    assert np.std(rms) < .01


def test_pink_noise_seed():
    """Test passing seeds to the random generator."""
    a = pfs.pink_noise(100, seed=1)
    b = pfs.pink_noise(100, seed=1)
    assert a == b

    a = pfs.pink_noise(100)
    b = pfs.pink_noise(100)
    with pytest.raises(AssertionError):
        assert a == b


def test_get_common_shape():
    """Test get_common_shape with all possible inputs."""
    a = 1
    b = [1, 2]
    c = [[1, 2, 3], [1, 2, 3]]

    # test with two numbers
    assert _get_common_shape(a, a) == (1, )
    # test with three numbers
    assert _get_common_shape(a, a, a) == (1, )
    # test with number and 1d data
    assert _get_common_shape(a, b) == (2, )
    # test with two 1d data entries
    assert _get_common_shape(b, b) == (2, )
    # test with number and 2d data
    assert _get_common_shape(a, c) == (2, 3)
    # test with two 2d data entries
    assert _get_common_shape(c, c) == (2, 3)
    # test not matching data
    with pytest.raises(ValueError, match="Input data must be of the same"):
        _get_common_shape(b, c)


def test_match_shape():
    """Test _match_shape with all possible inputs."""
    a = 1
    b = [[1, 2, 3], [1, 2, 3]]

    a_match, b_match = _match_shape((3, 2), a, b)
    npt.assert_allclose(np.ones((3, 2)), a_match)
    npt.assert_allclose(b, b_match)
