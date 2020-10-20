import numpy as np
import numpy.testing as npt
import pytest
from unittest import mock
import os.path
import scipy.io.wavfile as wavfile
import sofa

from haiopy import io
from haiopy import Signal


def test_read_wav():
    """Test default without optional parameters."""
    sampling_rate = 44100
    noise = np.random.rand(1000)
    filename = "test_wav.wav"
    # Create testfile
    wavfile.write(filename, sampling_rate, noise)
    signal = io.read_wav(filename)
    os.remove(filename)
    assert isinstance(signal, Signal)


def test_write_wav(signal_mock):
    """Test default without optional parameters."""
    filename = "test_wav.wav"
    io.write_wav(signal_mock, filename)
    assert os.path.isfile(filename)
    os.remove(filename)


def test_write_wav_overwrite(signal_mock):
    """Test overwriting behavior."""
    filename = "test_wav.wav"
    io.write_wav(signal_mock, filename)
    # Call with overwrite disabled
    with pytest.raises(FileExistsError):
        io.write_wav(signal_mock, filename, overwrite=False)
    # Call with overwrite enabled
    io.write_wav(signal_mock, filename, overwrite=True)
    os.remove(filename)


def test_write_wav_nd(signal_mock_nd):
    """Test for signals of higher dimension."""
    filename = "test_wav.wav"
    io.write_wav(signal_mock_nd, filename)
    # Check for correct dimensions
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        signal_mock_nd.time,
        signal_reload.reshape(signal_mock_nd.time.shape),
        rtol=1e-10)
    os.remove(filename)


def test_read_sofa():
    dir = 'tests/test_io_data/'
    # DataTypes
    io.read_sofa((dir + 'GeneralFIR.sofa'))
    with pytest.raises(ValueError):
        io.read_sofa((dir + 'GeneralTF.sofa'))
    # Units
    # PositionType


@pytest.fixture
def signal_mock():
    """ Generate a signal mock object.
    Returns
    -------
    signal : Signal
        The noise signal
    """
    n_samples = 1000
    sampling_rate = 44100
    amplitude = 1

    # time signal:
    time = amplitude * np.random.rand(n_samples)

    # create a mock object of Signal class to test independently
    signal_object = mock.Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time[np.newaxis, :]
    signal_object.sampling_rate = sampling_rate

    return signal_object


@pytest.fixture
def signal_mock_nd():
    """ Generate a higher dimensional signal mock object.
    Returns
    -------
    signal : Signal
        The signal
    """
    n_samples = 1000
    sampling_rate = 44100
    amplitude = 1

    # time signal:
    time = amplitude * np.random.random_sample((3, 3, 3, n_samples))

    # create a mock object of Signal class to test independently
    signal_object = mock.Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time
    signal_object.sampling_rate = sampling_rate

    return signal_object
