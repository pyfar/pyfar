import numpy as np
import numpy.testing as npt
import pytest

from haiopy import Signal
from haiopy import Coordinates
from haiopy import Orientation


def test_signal_init(sine):
    """Test to init Signal without optional parameters."""
    signal = Signal(sine, 44100)
    assert isinstance(signal, Signal)


def test_signal_init_val(sine):
    """Test to init Signal with complete parameters."""
    coord = Coordinates(1, 1, 1)
    view = np.array([1, 0, 0])
    up = np.array([0, 1, 0])
    orient = Orientation(view, up)
    signal = Signal(sine, 44100, domain="time", signaltype="power",
                    position=coord, orientation=orient)
    assert isinstance(signal, Signal)


def test_signal_init_fsignaltype(sine):
    """Test to init Signal with invalid signaltype."""
    with pytest.raises(ValueError):
        Signal(sine, 44100, signaltype="falsetype")
        pytest.fail("Not a valid signal type ('power'/'energy')")


def test_signal_init_fcoord(sine):
    """Test to init Signal with position that is not of type Coordinates."""
    fcoord = np.array([1, 1, 1])
    with pytest.raises(TypeError):
        Signal(sine, 44100, position=fcoord)
        pytest.fail("Input value has to be coordinates object.")


@pytest.fixture
def sine():
    """Generate a sine signal with f = 440 Hz and samplingrate = 44100 Hz.

    Returns
    -------
    signal : ndarray, double
        The sine signal

    """
    amplitude = 1
    frequency = 440
    samplingrate = 44100
    num_samples = 100
    fullperiod = False

    if fullperiod:
        num_periods = np.floor(num_samples / samplingrate * frequency)
        # round to the nearest frequency resulting in a fully periodic sine signal
        # in the given time interval
        frequency = num_periods * samplingrate / num_samples
    times = np.arange(0, num_samples) / samplingrate
    signal = amplitude * np.sin(2 * np.pi * frequency * times)

    return signal
