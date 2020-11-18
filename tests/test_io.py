import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture, mark
from mock import Mock, mock_open, patch
import tempfile

import os.path
from io import BytesIO
import scipy.io.wavfile as wavfile

from pyfar import io
from pyfar import Signal
from pyfar import Coordinates

from .test_io_data.generate_test_io_data import reference_signal
from .test_io_data.generate_test_io_data import reference_coordinates


baseline_path = 'tests/test_io_data'


def test_read_wav():
    """Test default without optional parameters."""
    filename = baseline_path + "/test_wav.wav"
    signal = io.read_wav(filename)
    signal_ref, sampling_rate = reference_signal()
    assert isinstance(signal, Signal)
    npt.assert_allclose(
        signal.time,
        np.atleast_2d(signal_ref),
        rtol=1e-10)
    assert signal.sampling_rate == sampling_rate


def test_write_wav(signal_mock):
    """Test default without optional parameters."""
    with tempfile.TemporaryDirectory() as td:
        filename = os.path.join(td, 'test_wav.wav')
        io.write_wav(signal_mock, filename)
        signal_reload = wavfile.read(filename)[-1].T
        npt.assert_allclose(
            signal_mock.time,
            np.atleast_2d(signal_reload),
            rtol=1e-10)


def test_write_wav_overwrite(signal_mock):
    """Test overwriting behavior."""
    with tempfile.TemporaryDirectory() as td:
        filename = os.path.join(td, 'test_wav.wav')
        io.write_wav(signal_mock, filename)
        # Call with overwrite disabled
        with pytest.raises(FileExistsError):
            io.write_wav(signal_mock, filename, overwrite=False)
        # Call with overwrite enabled
        io.write_wav(signal_mock, filename, overwrite=True)


def test_write_wav_nd(signal_mock_nd):
    """Test for signals of higher dimension."""
    with tempfile.TemporaryDirectory() as td:
        filename = os.path.join(td, 'test_wav.wav')
        io.write_wav(signal_mock_nd, filename)
        signal_reload = wavfile.read(filename)[-1].T
        npt.assert_allclose(
            signal_mock_nd.time,
            signal_reload.reshape(signal_mock_nd.time.shape),
            rtol=1e-10)


def test_read_sofa_signal():
    """Test for sofa signal properties"""
    # Correct DataType
    filename = os.path.join(baseline_path, 'GeneralFIR.sofa')
    signal = io.read_sofa(filename)[0]
    signal_ref = reference_signal(signal.cshape)[0]
    npt.assert_allclose(
        signal.time,
        signal_ref,
        rtol=1e-10)
    # Wrong DataType
    filename = os.path.join(baseline_path, 'GeneralTF.sofa')
    with pytest.raises(ValueError):
        io.read_sofa(filename)
    # Wrong sampling rate Unit
    filename = os.path.join(baseline_path, 'GeneralFIR_unit.sofa')
    with pytest.raises(ValueError):
        io.read_sofa(filename)


def test_read_sofa_coordinates():
    """Test for sofa cooridnate properties"""
    # Correct coordinates
    filename = os.path.join(baseline_path, 'GeneralFIR.sofa')
    source_coordinates = io.read_sofa(filename)[1]
    receiver_coordinates = io.read_sofa(filename)[2]
    source_coordinates_ref = reference_coordinates()[0]
    receiver_coordinates_ref = reference_coordinates()[1]
    npt.assert_allclose(
        source_coordinates.get_cart(),
        source_coordinates_ref,
        rtol=1e-10)
    npt.assert_allclose(
        receiver_coordinates.get_cart(),
        receiver_coordinates_ref[:, :, 0],
        rtol=1e-10)
    # Wrong PositionType
    filename = os.path.join(baseline_path, 'GeneralFIR_postype.sofa')
    with pytest.raises(ValueError):
        io.read_sofa(filename)


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
def objs(obj_instance):
    return obj_instance


def obj_dict_encoded(obj):
    obj_dict_encoded = obj.__dict__
    for key, value in obj_dict_encoded.items():
        if isinstance(value, np.ndarray):
            memfile = BytesIO()
            np.save(memfile, value)
            memfile.seek(0)
            obj_dict_encoded[key] = memfile.read().decode('latin-1')
    obj_dict_encoded['type'] = type(obj).__name__
    return obj_dict_encoded


@fixture
def obj_dicts_encoded(objs):
    return [obj_dict_encoded(obj) for obj in objs]


objects = [
    [Coordinates([1, -1], [2, -2], [3, -3])],
    # Coordinates(1, 2, 3, domain='sph', convention='side'),
    # Coordinates([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], 0, 0),
    # Orientations.from_view_up([
    #     [1, 0, 0], [2, 0, 0], [-1, 0, 0]], [[0, 1, 0], [0, -2, 0], [0, 1, 0]])
]


@patch('haiopy.io.json')
@patch('haiopy.io.open', new_callable=mock_open())
@mark.parametrize('objs', objects)
def test_read_coordinates(m_open, m_json, filename, objs, obj_dicts_encoded):
    m_json.load.return_value = obj_dicts_encoded
    obj_loaded = io.read(filename)

    m_open.assert_called_with(filename, 'r')

    assert obj_loaded == objs


@patch('haiopy.io.json')
@patch('haiopy.io.open', new_callable=mock_open())
@mark.parametrize('objs', objects)
def test_write_coordinates(
        m_open, m_json, filename, objs, obj_dicts_encoded):
    # assert False
    io.write(filename, objs)

    m_open.assert_called_with(filename, 'w')

    m_json.dump.assert_called_with(
        obj_dicts_encoded, m_open.return_value.__enter__.return_value)
