import numpy as np
import numpy.testing as npt
import pytest
from unittest import mock

from haiopy import Signal
from haiopy import Coordinates
from haiopy import Orientation


def test_signal_init(sine):
    """Test to init Signal without optional parameters."""
    signal = Signal(sine, 44100)
    assert isinstance(signal, Signal)


def test_signal_init_val(sine):
    """Test to init Signal with complete parameters."""
    coord_mock = mock.Mock(spec_set=Coordinates())
    coord_mock.x = 1
    coord_mock.y = 1
    coord_mock.z = 1
    orient_mock = mock.Mock(spec_set=Orientation())
    orient_mock.view = np.array([1, 0, 0])
    orient_mock.up = np.array([0, 1, 0])
    signal = Signal(sine, 44100, domain="time", signaltype="power",
                    position=coord_mock, orientation=orient_mock)
    assert isinstance(signal, Signal)


def test_signal_init_false_signaltype(sine):
    """Test to init Signal with invalid signaltype."""
    with pytest.raises(ValueError):
        Signal(sine, 44100, signaltype="falsetype")
        pytest.fail("Not a valid signal type ('power'/'energy')")


def test_signal_init_false_coord(sine):
    """Test to init Signal with position that is not of type Coordinates."""
    coord_false = np.array([1, 1, 1])
    with pytest.raises(TypeError):
        Signal(sine, 44100, position=coord_false)
        pytest.fail("Input value has to be coordinates object.")


def test_signal_init_false_orientation(sine):
    """Test to init Signal with orientation that is not of type Orientation."""
    orientation_false = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(TypeError):
        Signal(sine, 44100, orientation=orientation_false)
        pytest.fail("Input value has to be coordinates object.")


def test_getter_time(sine, impulse):
    signal = Signal(sine, 44100)
    signal._data = impulse
    npt.assert_allclose(signal.time, impulse)


def test_setter_time(sine, impulse):
    signal = Signal(sine, 44100)
    signal.time = impulse
    npt.assert_allclose(impulse, signal._data)


def test_getter_freq(sine, impulse):
    signal = Signal(sine, 44100)
    signal._data = impulse
    npt.assert_allclose(signal.freq, np.fft.fft(impulse), atol=1e-15)


def test_setter_freq(sine, impulse):
    signal = Signal(sine, 44100)
    signal.freq = np.fft.rfft(impulse)
    npt.assert_allclose(impulse, signal._data, atol=1e-15)


def test_getter_samplingrate(sine):
    samplingrate = 48000
    signal = Signal(sine, 44100)
    signal._samplingrate = samplingrate
    npt.assert_allclose(signal.samplingrate, samplingrate)


def test_setter_sampligrate(sine):
    samplingrate = 48000
    signal = Signal(sine, 44100)
    signal.samplingrate = samplingrate
    npt.assert_allclose(samplingrate, signal._samplingrate)


def test_getter_signaltype(sine):
    signaltype = "energy"
    signal = Signal(sine, 44100)
    signal._signaltype = signaltype
    npt.assert_string_equal(signal.signaltype, signaltype)


def test_setter_signaltype(sine):
    signaltype = "energy"
    signal = Signal(sine, 44100)
    signal.signaltype = signaltype
    npt.assert_string_equal(signaltype, signal._signaltype)


def test_getter_position(sine):
    coord_mock = mock.Mock(spec_set=Coordinates())
    coord_mock.x = 1
    coord_mock.y = 1
    coord_mock.z = 1
    signal = Signal(sine, 44100)
    signal._position = coord_mock
    npt.assert_allclose(signal.position.x, coord_mock.x)
    npt.assert_allclose(signal.position.y, coord_mock.y)
    npt.assert_allclose(signal.position.z, coord_mock.z)


def test_setter_position(sine):
    coord_mock = mock.Mock(spec_set=Coordinates())
    coord_mock.x = 1
    coord_mock.y = 1
    coord_mock.z = 1
    signal = Signal(sine, 44100)
    signal.position = coord_mock
    npt.assert_allclose(coord_mock.x, signal._position.x)
    npt.assert_allclose(coord_mock.y, signal._position.y)
    npt.assert_allclose(coord_mock.z, signal._position.z)


def test_getter_orientation(sine):
    orient_mock = mock.Mock(spec_set=Orientation())
    orient_mock.view = np.array([1, 0, 0])
    orient_mock.up = np.array([0, 1, 0])
    signal = Signal(sine, 44100)
    signal._orientation = orient_mock
    npt.assert_allclose(signal.orientation.up, orient_mock.up)
    npt.assert_allclose(signal.orientation.view, orient_mock.view)


def test_setter_orientation(sine):
    orient_mock = mock.Mock(spec_set=Orientation())
    orient_mock.view = np.array([1, 0, 0])
    orient_mock.up = np.array([0, 1, 0])
    signal = Signal(sine, 44100)
    signal.orientation = orient_mock
    npt.assert_allclose(orient_mock.up, signal._orientation.up)
    npt.assert_allclose(orient_mock.view, signal._orientation.view)


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


@pytest.fixture
def impulse():
    """Generate an impulse, also known as the Dirac delta function

    .. math::

        s(n) =
        \\begin{cases}
        a,  & \\text{if $n$ = 0} \\newline
        0, & \\text{else}
        \\end{cases}

    Returns
    -------
    signal : ndarray, double
        The impulse signal

    """
    amplitude = 1
    num_samples = 100

    signal = np.zeros(num_samples, dtype=np.double)
    signal[0] = amplitude

    return signal
