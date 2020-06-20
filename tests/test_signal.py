from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest

from haiopy import Coordinates, Orientation, Signal
from haiopy import fft


def test_signal_init(sine):
    """Test to init Signal without optional parameters."""
    signal = Signal(sine, 44100, domain='time')
    assert isinstance(signal, Signal)


def test_signal_init_list(impulse_list):
    signal = Signal(impulse_list, 44100, domain='time')
    assert isinstance(signal, Signal)


def test_domain_getter_freq(sine):
    signal = Signal(np.array([1]), 44100)
    signal._domain = 'freq'
    assert signal.domain == 'freq'


def test_domain_getter_time(sine):
    signal = Signal(np.array([1]), 44100)
    signal._domain = 'time'
    assert signal.domain == 'time'


def test_domain_setter_error():
    signal = Signal(np.array([1]), 44100)
    with pytest.raises(ValueError, match='Incorrect domain'):
        signal.domain = 'quark'


def test_domain_setter_freq_when_freq(sine):
    signal = Signal(np.array([1]), 44100)
    domain = 'freq'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_freq_when_time(sine):
    stype = 'power'
    spec = np.atleast_2d(fft.rfft(sine, len(sine), stype))
    signal = Signal(sine, 44100, domain='time', signal_type=stype)
    domain = 'freq'
    signal.domain = domain
    assert signal.domain == domain
    npt.assert_allclose(signal._data, spec, atol=1e-14, rtol=1e-14)


def test_domain_setter_time_when_time(sine):
    signal = Signal(np.array([1]), 44100)
    domain = 'time'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_time_when_freq(sine):
    stype = 'power'
    spec = np.atleast_2d(fft.rfft(sine, len(sine), stype))
    signal = Signal(spec, 44100, domain='freq', signal_type=stype)
    signal._data = spec
    signal._n_samples = len(sine)
    domain = 'time'
    signal.domain = domain
    assert signal.domain == domain
    npt.assert_allclose(
        signal._data, np.atleast_2d(sine), atol=1e-14, rtol=1e-14)


def test_signal_init_val(sine):
    """Test to init Signal with complete parameters."""
    coord_mock = mock.Mock(spec_set=Coordinates())
    coord_mock.x = 1
    coord_mock.y = 1
    coord_mock.z = 1
    orient_mock = mock.Mock(spec_set=Orientation())
    orient_mock.view = np.array([1, 0, 0])
    orient_mock.up = np.array([0, 1, 0])
    signal = Signal(sine, 44100, domain="time", signal_type="power",
                    position=coord_mock, orientation=orient_mock)
    assert isinstance(signal, Signal)


def test_signal_init_false_signal_type(sine):
    """Test to init Signal with invalid signal type."""
    with pytest.raises(ValueError):
        Signal(sine, 44100, signal_type="falsetype")
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


def test_n_samples(impulse):
    """Test for number of samples."""
    data = impulse
    signal = Signal(data, 44100, domain='time')
    assert signal.n_samples == len(data)


def test_n_bins(sine):
    """Test for number of freq bins."""
    data = sine
    signal = Signal(data, 44100, domain='time')
    data_freq = np.fft.rfft(data)
    assert signal.n_bins == len(data_freq)


def test_times(sine):
    """Test for the time instances."""
    signal = Signal(sine, 44100, domain='time')
    times = np.atleast_1d(np.arange(0, len(sine)) / 44100)
    npt.assert_allclose(signal.times, times)


def test_getter_time(sine, impulse):
    """Test if attribute time is accessed correctly."""
    signal = Signal(sine, 44100)
    signal._domain = 'time'
    signal._data = impulse
    npt.assert_allclose(signal.time, impulse)


def test_setter_time(sine, impulse):
    """Test if attribute time is set correctly."""
    signal = Signal(sine, 44100)
    signal.time = impulse
    assert signal._domain == 'time'
    npt.assert_allclose(np.atleast_2d(impulse), signal._data)


def test_getter_freq(sine, impulse):
    """Test if attribute freq is accessed correctly."""
    signal = Signal(sine, 44100, signal_type='power')
    new_sine = sine * 2
    spec = fft.rfft(new_sine, len(new_sine), 'power')
    signal._domain = 'freq'
    signal._data = spec
    npt.assert_allclose(signal.freq, spec, atol=1e-15)


def test_setter_freq(sine, impulse):
    """Test if attribute freq is set correctly."""
    signal = Signal(sine, 44100, signal_type='energy')
    spec = fft.rfft(impulse, len(impulse), signal_type='energy')
    signal.freq = spec
    assert signal.domain == 'freq'
    npt.assert_allclose(np.atleast_2d(spec), signal._data, atol=1e-15)


def test_getter_sampling_rate(sine):
    """Test if attribute sampling rate is accessed correctly."""
    sampling_rate = 48000
    signal = Signal(sine, 44100)
    signal._sampling_rate = sampling_rate
    npt.assert_allclose(signal.sampling_rate, sampling_rate)


def test_setter_sampligrate(sine):
    """Test if attribute sampling rate is set correctly."""
    sampling_rate = 48000
    signal = Signal(sine, 44100)
    signal.sampling_rate = sampling_rate
    npt.assert_allclose(sampling_rate, signal._sampling_rate)


def test_getter_signal_type(sine):
    """Test if attribute signal type is accessed correctly."""
    signal_type = "energy"
    signal = Signal(sine, 44100)
    signal._signal_type = signal_type
    npt.assert_string_equal(signal.signal_type, signal_type)


def test_setter_signal_type(sine):
    """Test if attribute signal type is set correctly."""
    signal_type = "energy"
    signal = Signal(sine, 44100)
    signal.signal_type = signal_type
    npt.assert_string_equal(signal_type, signal._signal_type)


def test_setter_signal_type_false_type(sine):
    """Test if ValueError is raised when signal type is set incorrectly."""
    signal = Signal(sine, 44100)
    with pytest.raises(ValueError):
        signal.signal_type = "falsetype"
        pytest.fail("Not a valid signal type ('power'/'energy')")


def test_dtype(sine):
    """Test for the getter od dtype."""
    dtype = np.float64
    signal = Signal(sine, 44100, dtype=dtype)
    assert signal.dtype == dtype


def test_signal_length(sine):
    """Test for the signal length."""
    signal = Signal(sine, 44100)
    length = (1000 - 1) / 44100
    assert signal.signal_length == length


def test_getter_position(sine):
    """Test if attribute position is accessed correctly."""
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
    """Test if attribute position is set correctly."""
    coord_mock = mock.Mock(spec_set=Coordinates())
    coord_mock.x = 1
    coord_mock.y = 1
    coord_mock.z = 1
    signal = Signal(sine, 44100)
    signal.position = coord_mock
    npt.assert_allclose(coord_mock.x, signal._position.x)
    npt.assert_allclose(coord_mock.y, signal._position.y)
    npt.assert_allclose(coord_mock.z, signal._position.z)


def test_setter_position_false_type(sine):
    """Test if TypeError is raised when position is not set with Coordinates.
    """
    signal = Signal(sine, 44100)
    with pytest.raises(TypeError):
        signal.position = np.array([1, 1, 1])
        pytest.fail("Input has to be Coordinates object")


def test_getter_orientation(sine):
    """Test if attribute orientation is accessed correctly."""
    orient_mock = mock.Mock(spec_set=Orientation())
    orient_mock.view = np.array([1, 0, 0])
    orient_mock.up = np.array([0, 1, 0])
    signal = Signal(sine, 44100)
    signal._orientation = orient_mock
    npt.assert_allclose(signal.orientation.up, orient_mock.up)
    npt.assert_allclose(signal.orientation.view, orient_mock.view)


def test_setter_orientation(sine):
    """Test if attribute orientation is set correctly."""
    orient_mock = mock.Mock(spec_set=Orientation())
    orient_mock.view = np.array([1, 0, 0])
    orient_mock.up = np.array([0, 1, 0])
    signal = Signal(sine, 44100)
    signal.orientation = orient_mock
    npt.assert_allclose(orient_mock.up, signal._orientation.up)
    npt.assert_allclose(orient_mock.view, signal._orientation.view)


def test_setter_orientation_false_type(sine):
    """Test if TypeError is raised when orientation is set incorrectly."""
    signal = Signal(sine, 44100)
    with pytest.raises(TypeError):
        signal.orientation = np.array([[1, 0, 0], [0, 1, 0]])
        pytest.fail("Input has to be Orientation object")


def test_shape(sine, impulse):
    """Test the attribute shape."""
    data = np.array([sine, impulse])
    signal = Signal(data, 44100)
    assert signal.shape == (2, 1000)


def test_magic_getitem(sine, impulse):
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([sine, impulse])
    sr = 44100
    signal = Signal(data, sr)
    npt.assert_allclose(Signal(sine, sr)._data, signal[0]._data)


def test_magic_getitem_slice(sine, impulse):
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([sine, impulse])
    sr = 44100
    signal = Signal(data, sr)
    npt.assert_allclose(Signal(sine, sr)._data, signal[:1]._data)


def test_magic_getitem_allslice(sine, impulse):
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([sine, impulse])
    sr = 44100
    signal = Signal(data, sr)
    npt.assert_allclose(Signal(data, sr)._data, signal[:]._data)


def test_magic_setitem(sine, impulse):
    """Test the magic function __setitem__."""
    signal = Signal(sine, 44100)
    signal[0] = impulse
    impulse_2d = np.atleast_2d(impulse)
    npt.assert_allclose(signal._data, impulse_2d)


def test_magic_len(impulse):
    """Test the magic function __len__."""
    signal = Signal(impulse, 44100)
    assert len(signal) == 1000


@pytest.fixture
def sine():
    """Generate a sine signal with f = 440 Hz and sampling_rate = 44100 Hz.

    Returns
    -------
    signal : ndarray, double
        The sine signal

    """
    amplitude = 1
    frequency = 440
    sampling_rate = 44100
    num_samples = 1000
    fullperiod = False

    if fullperiod:
        num_periods = np.floor(num_samples / sampling_rate * frequency)
        # round to the nearest frequency resulting in a fully periodic sine
        # signal in the given time interval
        frequency = num_periods * sampling_rate / num_samples
    times = np.arange(0, num_samples) / sampling_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * times)

    return signal


def impulse_func():
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
    num_samples = 1000

    signal = np.zeros(num_samples, dtype=np.double)
    signal[0] = amplitude

    return signal


@pytest.fixture
def impulse():
    return impulse_func()


@pytest.fixture
def impulse_list():
    imp = impulse_func()

    return imp.tolist()
