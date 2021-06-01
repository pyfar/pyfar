from pyfar import Orientations
import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch
from pyfar.testing.stub_utils import stub_str_to_type, stub_is_pyfar_type

import os.path
import pathlib
import scipy.io.wavfile as wavfile

from pyfar import io
from pyfar import Signal
from pyfar import Coordinates
from pyfar.samplings import SphericalVoronoi
import pyfar.classes.filter as fo
from pyfar import FrequencyData, TimeData


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


def test_write_wav_pathlib(noise, tmpdir):
    """Test write functionality with filename as pathlib Path object."""
    filename = pathlib.Path(tmpdir, 'test_wav.wav')
    io.write_wav(noise, filename)
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        noise.time,
        np.atleast_2d(signal_reload),
        rtol=1e-10)


def test_write_wav_suffix(noise, tmpdir):
    """Test for .wav extension of filename."""
    filename = pathlib.Path(tmpdir, 'test_wav')
    io.write_wav(noise, filename)
    # Without suffix
    with pytest.raises(FileNotFoundError):
        wavfile.read(filename)
    # With suffix added
    filename = filename.with_suffix('.wav')
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        noise.time,
        np.atleast_2d(signal_reload),
        rtol=1e-10)


def test_write_wav_nd(noise_two_by_three_channel, tmpdir):
    """Test for signals of higher dimension."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    with pytest.warns(UserWarning, match='flattened'):
        io.write_wav(noise_two_by_three_channel, filename)
    signal_reload = wavfile.read(filename)[-1].T
    npt.assert_allclose(
        signal_reload.reshape(noise_two_by_three_channel.time.shape),
        noise_two_by_three_channel.time)


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


def test_read_sofa_position_type_spherical(
        generate_sofa_postype_spherical, sofa_reference_coordinates):
    """Test to verify correct position type of sofa file"""
    _, s_coords, r_coords = io.read_sofa(generate_sofa_postype_spherical)
    npt.assert_allclose(
        s_coords.get_sph(convention='top_elev', unit='deg'),
        sofa_reference_coordinates[0])
    npt.assert_allclose(
        r_coords.get_sph(convention='top_elev', unit='deg'),
        sofa_reference_coordinates[1])


def test_read_sofa_position_type_unit(generate_sofa_postype_error):
    """Test to verify correct position type of sofa file"""
    with pytest.raises(ValueError):
        io.read_sofa(generate_sofa_postype_error)


@patch('pyfar.io._codec._str_to_type', new=stub_str_to_type())
@patch('pyfar.io._codec._is_pyfar_type', new=stub_is_pyfar_type())
def test_write_read_flat_data(tmpdir, flat_data):
    """ Check if file can be read back after writing without explicitply
    passing the .far-extension.
    """
    filename = os.path.join(tmpdir, 'write_read_flat_data.far')
    io.write(filename, flat_data=flat_data)
    actual = io.read(filename)
    assert actual['flat_data'] == flat_data


@patch('pyfar.io._codec._str_to_type', new=stub_str_to_type())
@patch('pyfar.io._codec._is_pyfar_type', new=stub_is_pyfar_type())
def test_write_read_nested_data(nested_data, flat_data, tmpdir):
    filename = os.path.join(tmpdir, 'write_nested_flat_data.far')
    io.write(filename, nested_data=nested_data)
    actual = io.read(filename)
    assert actual['nested_data'] == nested_data


@patch('pyfar.io._codec._str_to_type', new=stub_str_to_type())
@patch('pyfar.io._codec._is_pyfar_type', new=stub_is_pyfar_type())
def test_write_read_multipleObjects(flat_data, nested_data, tmpdir):
    """ Check if file can be read back after writing without explicitply
    passing the .far-extension.
    """
    filename = os.path.join(tmpdir, 'write_read_multipleObjects.far')
    io.write(
        filename,
        any_flat_data=flat_data,
        any_nested_data=nested_data)
    actual = io.read(filename)
    assert actual['any_flat_data'] == flat_data
    assert actual['any_nested_data'] == nested_data


def test_write_anyObj_TypeError(any_obj, tmpdir):
    """ Check if a TypeError is raised when writing an arbitrary
    object.
    """
    filename = os.path.join(tmpdir, 'anyObj.far')
    with pytest.raises(TypeError):
        io.write(filename, any_obj=any_obj)


@patch('pyfar.io._codec._str_to_type', new=stub_str_to_type())
@patch('pyfar.io._codec._is_pyfar_type', new=stub_is_pyfar_type())
def test_write_NoEncode_NotImplemented(no_encode_obj, tmpdir):
    """ Check if a TypeError is raised when writing an arbitrary
    object.
    """
    filename = os.path.join(tmpdir, 'no_encode_obj.far')
    with pytest.raises(NotImplementedError):
        io.write(filename, no_encode_obj=no_encode_obj)


@patch('pyfar.io._codec._str_to_type', new=stub_str_to_type())
@patch('pyfar.io._codec._is_pyfar_type', new=stub_is_pyfar_type())
def test_write_read_FlatDataNoDecode_NotImplemented(no_decode_obj, tmpdir):
    """ Check if a NotImplementedError is raised when writing an arbitrary
    object.
    """
    filename = os.path.join(tmpdir, 'no_decode_obj.far')
    io.write(filename, no_decode_obj=no_decode_obj)
    with pytest.raises(NotImplementedError):
        io.read(filename)


@patch('pyfar.io._codec._str_to_type', new=stub_str_to_type())
@patch('pyfar.io._codec._is_pyfar_type', side_effect=[True] + 42 * [False])
def test_write_read_FlatDataNoPyfarType_TypeError(_, no_decode_obj, tmpdir):
    """ Check if a TypeError is raised when reading a .far-file that has been
    created with a different version of Pyfar in which types exist which are
    not present in the current version.

    Notes
    -----
    The `42` in `[True] + 42 * [False]` is arbitrary. What matters is, that
    `_is_pyfar_type` returns True on the first call to enable writing the
    .far-file which is not compatible with the current version in the first
    place.
    """
    filename = os.path.join(tmpdir, 'no_decode_obj.far')
    io.write(filename, no_decode_obj=no_decode_obj)
    with pytest.raises(TypeError):
        io.read(filename)


def test_write_read_orientations(orientations, tmpdir):
    """ Orientations
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'orientations.far')
    io.write(filename, orientations=orientations)
    actual = io.read(filename)['orientations']
    assert isinstance(actual, Orientations)
    assert actual == orientations


def test_write_read_coordinates(coordinates, tmpdir):
    """ Coordinates
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'coordinates.far')
    io.write(filename, coordinates=coordinates)
    actual = io.read(filename)['coordinates']
    assert isinstance(actual, Coordinates)
    assert actual == coordinates


def test_write_read_signal(sine, tmpdir):
    """ Signal
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'signal.far')
    io.write(filename, signal=sine)
    actual = io.read(filename)['signal']
    assert isinstance(actual, Signal)
    assert actual == sine


def test_write_read_timedata(time_data, tmpdir):
    """ TimeData
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'timedata.far')
    io.write(filename, timedata=time_data)
    actual = io.read(filename)['timedata']
    assert isinstance(actual, TimeData)
    assert actual == time_data


def test_write_read_frequencydata(frequency_data, tmpdir):
    """ TimeData
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'frequencydata.far')
    io.write(filename, frequencydata=frequency_data)
    actual = io.read(filename)['frequencydata']
    assert isinstance(actual, FrequencyData)
    assert actual == frequency_data


def test_write_read_sphericalvoronoi(sphericalvoronoi, tmpdir):
    """ SphericalVoronoi
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'sphericalvoronoi.far')
    io.write(filename, sphericalvoronoi=sphericalvoronoi)
    actual = io.read(filename)['sphericalvoronoi']
    assert isinstance(actual, SphericalVoronoi)
    assert actual == sphericalvoronoi


def test_write_read_filter(filter, tmpdir):
    """ Filter
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'filter.far')
    io.write(filename, filter=filter)
    actual = io.read(filename)['filter']
    assert isinstance(actual, fo.Filter)
    assert actual == filter


def test_write_filterFIR(filterFIR, tmpdir):
    """ filterFIR
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'filterIIR.far')
    io.write(filename, filterFIR=filterFIR)
    actual = io.read(filename)['filterFIR']
    assert isinstance(actual, fo.Filter)
    assert actual == filterFIR


def test_write_filterIIR(filterIIR, tmpdir):
    """ FilterIIR
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'filterIIR.far')
    io.write(filename, filterIIR=filterIIR)
    actual = io.read(filename)['filterIIR']
    assert isinstance(actual, fo.Filter)
    assert actual == filterIIR


def test_write_filterSOS(filterSOS, tmpdir):
    """ filterSOS
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'filterSOS.far')
    io.write(filename, filterSOS=filterSOS)
    actual = io.read(filename)['filterSOS']
    assert isinstance(actual, fo.Filter)
    assert actual == filterSOS


def test_write_read_numpy_ndarrays(tmpdir):
    """ Numpy ndarray
    Make sure `read` understands the bits written by `write`
    """
    matrix_2d_int = np.arange(0, 24, dtype=np.int).reshape((4, 6))
    matrix_2d_float = matrix_2d_int.astype(np.float)
    matrix_2d_complex = matrix_2d_int.astype(np.complex)

    matrix_3d_int = np.arange(0, 24, dtype=np.int).reshape((2, 3, 4))
    matrix_3d_float = matrix_3d_int.astype(np.float)
    matrix_3d_complex = matrix_3d_int.astype(np.complex)

    filename = os.path.join(tmpdir, 'ndarray.far')

    io.write(
        filename,
        matrix_2d_int=matrix_2d_int,
        matrix_2d_float=matrix_2d_float,
        matrix_2d_complex=matrix_2d_complex,
        matrix_3d_int=matrix_3d_int,
        matrix_3d_float=matrix_3d_float,
        matrix_3d_complex=matrix_3d_complex)

    actual = io.read(filename)
    assert isinstance(actual['matrix_2d_int'], np.ndarray)
    assert np.allclose(actual['matrix_2d_int'], matrix_2d_int)
    assert isinstance(actual['matrix_2d_float'], np.ndarray)
    assert np.allclose(actual['matrix_2d_float'], matrix_2d_float)
    assert isinstance(actual['matrix_2d_complex'], np.ndarray)
    assert np.allclose(actual['matrix_2d_complex'], matrix_2d_complex)
    assert isinstance(actual['matrix_3d_int'], np.ndarray)
    assert np.allclose(actual['matrix_3d_int'], matrix_3d_int)
    assert isinstance(actual['matrix_2d_float'], np.ndarray)
    assert np.allclose(actual['matrix_3d_float'], matrix_3d_float)
    assert isinstance(actual['matrix_2d_complex'], np.ndarray)
    assert np.allclose(actual['matrix_3d_complex'], matrix_3d_complex)


def test_write_read_multiplePyfarObjects(
        filter,
        filterFIR,
        filterIIR,
        filterSOS,
        coordinates,
        orientations,
        sphericalvoronoi,
        time_data,
        frequency_data,
        sine,
        tmpdir):
    """ Check if multiple different PyFar-objects can be written to disk
    and read back.
    """
    filename = os.path.join(tmpdir, 'multiplePyfarObjects.far')
    matrix_2d_int = np.arange(0, 24, dtype=np.int).reshape((4, 6))
    io.write(
        filename,
        filter=filter,
        filterFIR=filterFIR,
        filterIIR=filterIIR,
        filterSOS=filterSOS,
        coordinates=coordinates,
        orientations=orientations,
        sphericalvoronoi=sphericalvoronoi,
        timedata=time_data,
        frequencydata=frequency_data,
        signal=sine,
        matrix_2d_int=matrix_2d_int)
    actual = io.read(filename)
    assert isinstance(actual['filter'], fo.Filter)
    assert actual['filter'] == filter
    assert isinstance(actual['filterFIR'], fo.FilterFIR)
    assert actual['filterFIR'] == filterFIR
    assert isinstance(actual['filterIIR'], fo.FilterIIR)
    assert actual['filterIIR'] == filterIIR
    assert isinstance(actual['filterSOS'], fo.FilterSOS)
    assert actual['filterSOS'] == filterSOS
    assert isinstance(actual['coordinates'], Coordinates)
    assert actual['coordinates'] == coordinates
    assert isinstance(actual['orientations'], Orientations)
    assert actual['orientations'] == orientations
    assert isinstance(actual['sphericalvoronoi'], SphericalVoronoi)
    assert actual['sphericalvoronoi'] == sphericalvoronoi
    assert isinstance(actual['timedata'], TimeData)
    assert actual['timedata'] == time_data
    assert isinstance(actual['frequencydata'], FrequencyData)
    assert actual['frequencydata'] == frequency_data
    assert isinstance(actual['signal'], Signal)
    assert actual['signal'] == sine
    assert isinstance(actual['matrix_2d_int'], np.ndarray)
    assert np.allclose(actual['matrix_2d_int'], matrix_2d_int)


def test_write_read_multiplePyfarObjectsWithCompression(
        filter,
        filterFIR,
        filterIIR,
        filterSOS,
        coordinates,
        orientations,
        sphericalvoronoi,
        time_data,
        frequency_data,
        sine,
        tmpdir):
    """ Check if multiple different PyFar-objects can be written to disk
    and read back with zip compression.
    """
    filename = os.path.join(tmpdir, 'multiplePyfarObjects.far')
    matrix_2d_int = np.arange(0, 24, dtype=np.int).reshape((4, 6))
    io.write(
        filename,
        compress=True,
        filter=filter,
        filterFIR=filterFIR,
        filterIIR=filterIIR,
        filterSOS=filterSOS,
        coordinates=coordinates,
        orientations=orientations,
        sphericalvoronoi=sphericalvoronoi,
        timedata=time_data,
        frequencydata=frequency_data,
        signal=sine,
        matrix_2d_int=matrix_2d_int)
    actual = io.read(filename)
    assert isinstance(actual['filter'], fo.Filter)
    assert actual['filter'] == filter
    assert isinstance(actual['filterFIR'], fo.FilterFIR)
    assert actual['filterFIR'] == filterFIR
    assert isinstance(actual['filterIIR'], fo.FilterIIR)
    assert actual['filterIIR'] == filterIIR
    assert isinstance(actual['filterSOS'], fo.FilterSOS)
    assert actual['filterSOS'] == filterSOS
    assert isinstance(actual['coordinates'], Coordinates)
    assert actual['coordinates'] == coordinates
    assert isinstance(actual['orientations'], Orientations)
    assert actual['orientations'] == orientations
    assert isinstance(actual['sphericalvoronoi'], SphericalVoronoi)
    assert actual['sphericalvoronoi'] == sphericalvoronoi
    assert isinstance(actual['timedata'], TimeData)
    assert actual['timedata'] == time_data
    assert isinstance(actual['frequencydata'], FrequencyData)
    assert actual['frequencydata'] == frequency_data
    assert isinstance(actual['signal'], Signal)
    assert actual['signal'] == sine
    assert isinstance(actual['matrix_2d_int'], np.ndarray)
    assert np.allclose(actual['matrix_2d_int'], matrix_2d_int)
