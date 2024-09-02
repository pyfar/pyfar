from pyfar.dsp import soft_limit_spectrum
import pyfar as pf
import pytest


def test_assertions_signal():
    """Test assertion for passing wrong type of audio object."""

    with pytest.raises(TypeError, match='input signal must be a pyfar'):
        soft_limit_spectrum(pf.TimeData([1, 2, 3], [0, 1, 3]), 0, 0)


def test_assertion_direction():
    """Test assertion for passing invalid value for `direction` parameter."""

    with pytest.raises(ValueError, match="direction is 'both'"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, 0, direction='both')


def test_assertion_knee():
    """Test assertion for passing invalid value for `knee` parameter."""

    # wrong string
    with pytest.raises(ValueError, match="knee is 'tangens'"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, knee='tangens')

    # wrong number
    with pytest.raises(ValueError, match="knee is -1"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, knee=-1)

    # wrong type
    with pytest.raises(TypeError, match="knee must be"):
        soft_limit_spectrum(pf.Signal([1, 2, 3], 1), 0, knee=(1, 1))
