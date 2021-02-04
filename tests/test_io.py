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

from pyfar import io
from pyfar import Signal
from pyfar import Coordinates
from pyfar.spatial.spatial import SphericalVoronoi
import pyfar.dsp.classes as fo


def test_read_wav(generate_wav_file, noise):
    """Test default without optional parameters."""
    signal = io.read_wav(generate_wav_file)
    assert isinstance(signal, Signal)
    npt.assert_allclose(signal.time, noise.time)
    assert signal.sampling_rate == noise.sampling_rate


def test_write_wav(tmpdir, noise):
    """Test default without optional parameters."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_wav(noise, filename)
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        noise.time,
        np.atleast_2d(signal_reload),
        rtol=1e-10)


def test_write_wav_overwrite(noise, tmpdir):
    """Test overwriting behavior."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_wav(noise, filename)
    # Call with overwrite disabled
    with pytest.raises(FileExistsError):
        io.write_wav(noise, filename, overwrite=False)
    # Call with overwrite enabled
    io.write_wav(noise, filename, overwrite=True)


def test_write_wav_nd(noise_two_by_two_channel, tmpdir):
    """Test for signals of higher dimension."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_wav(noise_two_by_two_channel, filename)
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        signal_reload.reshape(noise_two_by_two_channel.time.shape),
        noise_two_by_two_channel.time)


def test_read_sofa_GeneralFIR(
        generate_sofa_GeneralFIR, noise_two_by_three_channel):
    """Test for sofa datatype GeneralFIR"""
    signal = io.read_sofa(generate_sofa_GeneralFIR)[0]
    npt.assert_allclose(signal.time, noise_two_by_three_channel.time)


def test_read_sofa_GeneralTF(generate_sofa_GeneralTF):
    """Test for sofa datatype GeneralTF"""
    with pytest.raises(ValueError):
        io.read_sofa(generate_sofa_GeneralTF)


def test_read_sofa_coordinates(
        generate_sofa_GeneralFIR, sofa_reference_coordinates):
    """Test for reading coordinates in sofa file"""
    _, s_coords, r_coords, = io.read_sofa(generate_sofa_GeneralFIR)
    npt.assert_allclose(
        s_coords.get_cart(), sofa_reference_coordinates[0])
    npt.assert_allclose(
        r_coords.get_cart(), sofa_reference_coordinates[1])


def test_read_sofa_sampling_rate_unit(generate_sofa_unit_error):
    """Test to verify correct sampling rate unit of sofa file"""
    with pytest.raises(ValueError):
        io.read_sofa(generate_sofa_unit_error)


def test_read_sofa_position_type_unit(generate_sofa_postype_error):
    """Test to verify correct position type of sofa file"""
    with pytest.raises(ValueError):
        io.read_sofa(generate_sofa_postype_error)

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


def test_read_filterFIR(
    generate_far_file_filterFIR,
    filterFIR):
    pass
    # actual = io.read(generate_far_file_filterFIR)['filterFIR']
    # assert isinstance(actual, fo.FilterIIR)
    # assert actual == filterFIR

@patch('pyfar.io.str_to_type')
def test_read_nested_data_struct(
        patched_str_to_type,
        generate_far_file_nested_data_struct,
        nested_data_struct,
        other_class):
    str_to_type = {
        'MyOtherClass': type(other_class),
        'NestedDataStruct': type(nested_data_struct)}
    patched_str_to_type.side_effect = str_to_type.get
    actual = io.read(generate_far_file_nested_data_struct)[
        'nested_data_struct']
    assert actual == nested_data_struct
    pass