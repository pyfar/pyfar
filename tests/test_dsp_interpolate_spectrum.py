import pytest
from pytest import raises
import pyfar as pf
from pyfar.dsp import interpolate_spectrum


def test_init():
    """Test return objects"""
    fd = pf.FrequencyData([1, .5], [100, 200])

    # interpolation object
    interpolator = interpolate_spectrum(
        fd, "complex", ("linear", "linear", "linear"))
    assert isinstance(interpolator, interpolate_spectrum)

    # interpolation result
    signal = interpolator(8, 44100)
    assert isinstance(signal, pf.Signal)


def test_init_assertions():
    """Test if init raises assertions correctly"""
    fd = pf.FrequencyData([1, .5], [100, 200])

    # data (invalid type)
    with raises(TypeError, match="data must be"):
        interpolate_spectrum(1, "complex", ("linear", "linear", "linear"))
    # data (invalid FFT normalization)
    with raises(ValueError, match="data.fft_norm is 'rms'"):
        fd_rms = pf.FrequencyData([1, .5], [100, 200], 'rms')
        interpolate_spectrum(
            fd_rms, "complex", ("linear", "linear", "linear"))
    # data (not enough bins)
    with raises(ValueError, match="data.n_bins must be at least 2"):
        fd_short = pf.FrequencyData(1, 100)
        interpolate_spectrum(
            fd_short, "complex", ("linear", "linear", "linear"))

    # test invalid method
    with raises(ValueError, match="method is 'invalid'"):
        interpolate_spectrum(fd, "invalid", ("linear", "linear", "linear"))

    # test kind (invald type)
    with raises(ValueError, match="kind must be a tuple of length 3"):
        interpolate_spectrum(fd, "complex", "linear")
    # test kind (invalid length)
    with raises(ValueError, match="kind must be a tuple of length 3"):
        interpolate_spectrum(fd, "complex", ("linear", "linear"))
    # test kind (wrong entry)
    with raises(ValueError, match="kind contains 'wrong'"):
        interpolate_spectrum(fd, "complex", ("linear", "linear", "wrong"))

    # test fscale
    with raises(ValueError, match="fscale is 'nice'"):
        interpolate_spectrum(
            fd, "complex", ("linear", "linear", "linear"), fscale="nice")

    # test clip (wrong value of bool)
    with raises(ValueError, match="clip must be a tuple of length 2"):
        interpolate_spectrum(
            fd, "complex", ("linear", "linear", "linear"), clip=True)
    # test clip (invalid type)
    with raises(ValueError, match="clip must be a tuple of length 2"):
        interpolate_spectrum(
            fd, "complex", ("linear", "linear", "linear"), clip=1)
    # test clip (invalid length)
    with raises(ValueError, match="clip must be a tuple of length 2"):
        interpolate_spectrum(
            fd, "complex", ("linear", "linear", "linear"), clip=(1, 2, 3))

    # test not providing group_delay for method="magnitude_linear"
    with raises(ValueError, match="The group delay must be specified"):
        interpolate_spectrum(
            fd, "magnitude_linear", ("linear", "linear", "linear"))
