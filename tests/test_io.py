from numpy.lib.npyio import save
from numpy.lib.ufunclike import fix
from haiopy.coordinates import Coordinates
from os import read
import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture
from mock import Mock, mock_open, patch
import os.path
from io import BytesIO
import copy
import scipy.io.wavfile as wavfile

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


@fixture
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
    signal_object = Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time[np.newaxis, :]
    signal_object.sampling_rate = sampling_rate

    return signal_object


@fixture
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
    signal_object = Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time
    signal_object.sampling_rate = sampling_rate

    return signal_object


@fixture
def filename():
    return 'data.hpy'

@fixture
def obj():
    return Coordinates([1, -1], [2, -2], [3, -3])

@fixture
def obj_dict_encoded(obj):
    obj_dict_encoded = obj.__dict__
    for key, value in obj_dict_encoded.items():
        if isinstance(value, np.ndarray):
            memfile = BytesIO()
            np.save(memfile, value)
            memfile.seek(0)
            obj_dict_encoded[key] = memfile.read().decode('latin-1')
    return obj_dict_encoded


@patch('haiopy.io.json')
@patch('haiopy.io.open', new_callable=mock_open())
def test_write_coordinates(m_open, m_json, filename, obj, obj_dict_encoded):
    # assert False
    io.write(filename, obj)

    m_open.assert_called_with(filename, 'w')

    # m_json.dump.assert_called_with(
    #     obj_dict_encoded, m_json.return_value.__enter__.return_value)
