from pyfar.orientations import Orientations
import numpy as np
import numpy.testing as npt
import pytest
import json
from pytest import fixture
from mock import Mock, mock_open, patch, call

import os.path
from io import BytesIO
import scipy.io.wavfile as wavfile
import sofa

from pyfar import io
from pyfar import Signal
from pyfar import Coordinates
from pyfar.spatial.spatial import SphericalVoronoi
import pyfar.dsp.classes as fo


def test_read_wav(tmpdir):
    """Test default without optional parameters."""
    # Generate test files
    filename = os.path.join(tmpdir, 'test_wav.wav')
    signal_ref, sampling_rate = reference_signal()
    wavfile.write(filename, sampling_rate, signal_ref.T)
    # Read wav
    signal = io.read_wav(filename)
    assert isinstance(signal, Signal)
    npt.assert_allclose(
        signal.time,
        np.atleast_2d(signal_ref),
        rtol=1e-10)
    assert signal.sampling_rate == sampling_rate


def test_write_wav(signal_mock, tmpdir):
    """Test default without optional parameters."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_wav(signal_mock, filename)
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        signal_mock.time,
        np.atleast_2d(signal_reload),
        rtol=1e-10)


def test_write_wav_overwrite(signal_mock, tmpdir):
    """Test overwriting behavior."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_wav(signal_mock, filename)
    # Call with overwrite disabled
    with pytest.raises(FileExistsError):
        io.write_wav(signal_mock, filename, overwrite=False)
    # Call with overwrite enabled
    io.write_wav(signal_mock, filename, overwrite=True)


def test_write_wav_nd(signal_mock_nd, tmpdir):
    """Test for signals of higher dimension."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_wav(signal_mock_nd, filename)
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        signal_mock_nd.time,
        signal_reload.reshape(signal_mock_nd.time.shape),
        rtol=1e-10)


def test_read_sofa_GeneralFIR(tmpdir):
    """Test for sofa datatype GeneralFIR"""
    sofatype = 'GeneralFIR'
    # Generate test file
    generate_sofa_file(tmpdir, sofatype)
    # Correct DataType
    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    signal = io.read_sofa(filename)[0]
    signal_ref = reference_signal(signal.cshape)[0]
    npt.assert_allclose(
        signal.time,
        signal_ref,
        rtol=1e-10)


def test_read_sofa_GeneralTF(tmpdir):
    """Test for sofa datatype GeneralTF"""
    sofatype = 'GeneralTF'
    # Generate test file
    generate_sofa_file(tmpdir, sofatype)
    # Wrong DataType
    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    with pytest.raises(ValueError):
        io.read_sofa(filename)


def test_read_sofa_coordinates(tmpdir):
    """Test for reading coordinates in sofa file"""
    sofatype = 'GeneralFIR'
    # Generate test file
    generate_sofa_file(tmpdir, sofatype)
    # Correct coordinates
    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    # Source coordinates
    source_coordinates = io.read_sofa(filename)[1]
    source_coordinates_ref = reference_coordinates()[0]
    npt.assert_allclose(
        source_coordinates.get_cart(),
        source_coordinates_ref,
        rtol=1e-10)
    # Receiver coordinates
    receiver_coordinates = io.read_sofa(filename)[2]
    receiver_coordinates_ref = reference_coordinates()[1]
    npt.assert_allclose(
        receiver_coordinates.get_cart(),
        receiver_coordinates_ref[:, :, 0],
        rtol=1e-10)


def test_read_sofa_sampling_rate_unit(tmpdir):
    """Test to verify correct sampling rate unit of sofa file"""
    sofatype = 'GeneralFIR_unit'
    # Generate test file
    generate_sofa_file(tmpdir, sofatype)
    # Wrong sampling rate Unit
    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    with pytest.raises(ValueError):
        io.read_sofa(filename)


def test_read_sofa_position_type(tmpdir):
    """Test for correct position type of sofa file"""
    sofatype = 'GeneralFIR_postype'
    # Generate test file
    generate_sofa_file(tmpdir, sofatype)
    # Wrong PositionType
    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    with pytest.raises(ValueError):
        io.read_sofa(filename)


def generate_sofa_file(filedir, sofatype):
    """ Generate the reference sofa files used for testing the read_sofa
    function.

    Parameters
    -------
    filedir : String
        Path to directory.
    """
    n_measurements = 1
    n_receivers = 2
    signal, sampling_rate = reference_signal(
        shape=(n_measurements, n_receivers))
    n_samples = signal.shape[-1]

    filename = os.path.join(filedir, (sofatype + '.sofa'))
    sofafile = sofa.Database.create(
        filename,
        sofatype.split('_')[0],
        dimensions={
            "M": n_measurements,
            "R": n_receivers,
            "N": n_samples})
    sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.Position = reference_coordinates()[0]
    sofafile.Receiver.initialize(fixed=["Position", "View", "Up"])
    sofafile.Receiver.Position = reference_coordinates()[1]
    sofafile.Emitter.initialize(fixed=["Position", "View", "Up"], count=1)

    if sofatype == 'GeneralFIR':
        sofafile.Data.Type = 'FIR'
        sofafile.Data.initialize()
        sofafile.Data.IR = signal
        sofafile.Data.SamplingRate = sampling_rate
    elif sofatype == 'GeneralTF':
        sofafile.Data.Type = 'TF'
        sofafile.Data.initialize()
        sofafile.Data.Real = signal
        sofafile.Data.Imag = signal
    elif sofatype == 'GeneralFIR_unit':
        sofafile.Data.Type = 'FIR'
        sofafile.Data.initialize()
        sofafile.Data.IR = signal
        sofafile.Data.SamplingRate = sampling_rate
        sofafile.Data.SamplingRate.Units = 'not_hertz'
    elif sofatype == 'GeneralFIR_postype':
        sofafile.Data.Type = 'FIR'
        sofafile.Data.initialize()
        sofafile.Data.IR = signal
        sofafile.Data.SamplingRate = sampling_rate
        sofafile.Source.Position.Type = 'not_type'
    sofafile.close()


def reference_signal(shape=(1,)):
    """ Generate sine of 440 Hz as numpy array.
    Returns
    -------
    sine : ndarray
        The sine signal
    sampling_rate : int
        The sampling rate
    """
    sampling_rate = 44100
    n_periods = 20
    amplitude = 1
    frequency = 440

    # time signal
    times = np.arange(0, n_periods * frequency) / sampling_rate
    sine = amplitude * np.sin(2 * np.pi * times * frequency)

    shape + (3,)
    sine = np.ones(shape + (sine.shape[-1],)) * sine

    return sine, sampling_rate


def reference_coordinates():
    """ Generate coordinate array
    Returns
    -------
    coordinates : ndarray
        The coordinates
    """
    source_coordinates = np.ones((1, 3))
    receiver_coordinates = np.ones((2, 3, 1))
    return source_coordinates, receiver_coordinates


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


# TODO: Use objects in parametrized tests?
objects = [
    [Coordinates([1, -1], [2, -2], [3, -3])],
    # Coordinates(1, 2, 3, domain='sph', convention='side'),
    # Coordinates([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], 0, 0),
    # Orientations.from_view_up([
    #     [1, 0, 0], [2, 0, 0], [-1, 0, 0]],
    #     [[0, 1, 0], [0, -2, 0], [0, 1, 0]])
]


# TODO: parametrize test with objects listed above?
# @mark.parametrize('objs', objects)
@patch('pyfar.io.zipfile.ZipFile')
@patch('pyfar.io.open', new_callable=mock_open, read_data=b'any')
def test_read(m_open, m_zipfile, filename):
    m_zipenter = m_zipfile.return_value.__enter__.return_value
    m_zipenter.namelist.return_value = [
        'my_obj/json', 'my_obj/ndarrays/_points']
    m_zipenter.read.side_effect = lambda x: mock_zipfile_read(x)
    io.read(filename)


# TODO: parametrize test with objects listed above?
# @mark.parametrize('objs', objects)
@patch('pyfar.io._encode', )
@patch('pyfar.io.zipfile.ZipFile')
@patch('pyfar.io.open', new_callable=mock_open)
def test_write(m_open, m_zipfile, m__encode, filename):
    # m_json.dumps.side_effect = lambda x: mock_json_dumps(x)
    # TODO: Should we really mock this function? Shouldn't we
    # test private functions indirectly through the public interface?
    # i.e test this function with real objects (parametrized)
    m__encode.return_value = (
        json.loads(mock_zipfile_read('json').decode('UTF-8')),
        {'obj_ndarray': mock_zipfile_read('ndarrays')})

    io.write(filename, c=Coordinates())

    m_zipfile.return_value.__enter__.return_value.writestr.assert_has_calls([
        call('c/json', mock_zipfile_read('json').decode('UTF-8')),
        call('c/ndarrays/obj_ndarray', mock_zipfile_read('ndarrays'))
    ])
    m_open.assert_called_with(filename, 'wb')
    m_open.return_value.__enter__.return_value.write.assert_called_with(b'')


def mock_zipfile_read(obj_path):
    obj_path_split = obj_path.split('/')
    if 'json' in obj_path_split:
        return b'{"type": "Coordinates"}'
    elif 'ndarrays' in obj_path_split:
        return (
            b'\x93NUMPY\x01\x00v\x00{"descr": "<f8", "fortran_order": False, '
            b'"shape": (1, 3), }                                             '
            b'             \n\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00'
            b'\x00\x00@\x00\x00\x00\x00\x00\x00\x08@')


def test_str_to_type():
    PyfarType = io.str_to_type('Coordinates')
    assert PyfarType.__module__.startswith('pyfar')
    PyfarType = io.str_to_type('Orientations')
    assert PyfarType.__module__.startswith('pyfar')
    PyfarType = io.str_to_type('SphericalVoronoi')
    assert PyfarType.__module__.startswith('pyfar')
    pass


def test_read_orientations(generate_far_file_orientations, orientations):
    """Write is already called in the `generate_orientations_file` fixture"""
    actual = io.read(generate_far_file_orientations)['orientations']
    assert isinstance(actual, Orientations)
    assert actual == orientations


def test_read_coordinates(generate_far_file_coordinates, coordinates):
    actual = io.read(generate_far_file_coordinates)['coordinates']
    assert isinstance(actual, Coordinates)
    assert actual == coordinates


def test_read_signal(generate_far_file_signal, signal):
    actual = io.read(generate_far_file_signal)['signal']
    assert isinstance(actual, Signal)
    assert actual == signal


def test_read_sphericalvoronoi(
    generate_far_file_sphericalvoronoi,
    sphericalvoronoi):
    actual = io.read(generate_far_file_sphericalvoronoi)['sphericalvoronoi']
    assert isinstance(actual, SphericalVoronoi)
    assert actual == sphericalvoronoi


def test_read_filter(
    generate_far_file_filter,
    filter):
    actual = io.read(generate_far_file_filter)['filter']
    assert isinstance(actual, fo.Filter)
    assert actual == filter


def test_read_filterIIR(
    generate_far_file_filterIIR,
    filterIIR):
    actual = io.read(generate_far_file_filterIIR)['filterIIR']
    assert isinstance(actual, fo.FilterIIR)
    assert actual == filterIIR

    