from pyfar import Orientations
import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch
import pyfar as pf
from pyfar.testing.stub_utils import stub_str_to_type, stub_is_pyfar_type

import os.path
import pathlib
import soundfile

from pyfar import io
from pyfar import Signal
from pyfar import Coordinates
from pyfar.samplings import SphericalVoronoi
import pyfar.classes.filter as fo
from pyfar import FrequencyData, TimeData


def test_read_sofa_GeneralFIR(
        generate_sofa_GeneralFIR, noise_two_by_three_channel):
    """Test for sofa datatype GeneralFIR"""
    signal = io.read_sofa(generate_sofa_GeneralFIR)[0]
    npt.assert_allclose(signal.time, noise_two_by_three_channel.time)


def test_read_sofa_GeneralTF(
        generate_sofa_GeneralTF, noise_two_by_three_channel):
    """Test for sofa datatype GeneralTF"""
    signal = io.read_sofa(generate_sofa_GeneralTF)[0]
    npt.assert_allclose(signal.freq, noise_two_by_three_channel.freq)


def test_read_sofa_coordinates(
        generate_sofa_GeneralFIR, sofa_reference_coordinates):
    """Test for reading coordinates in sofa file"""
    _, s_coords, r_coords, = io.read_sofa(generate_sofa_GeneralFIR)
    npt.assert_allclose(
        s_coords.cart, sofa_reference_coordinates[0])
    npt.assert_allclose(
        r_coords.cart, sofa_reference_coordinates[1])


def test_read_sofa_position_type_spherical(
        generate_sofa_postype_spherical, sofa_reference_coordinates):
    """Test to verify correct position type of sofa file"""
    _, s_coords, r_coords = io.read_sofa(generate_sofa_postype_spherical)
    npt.assert_allclose(
        s_coords.azimuth * 180/np.pi, sofa_reference_coordinates[0][..., 0])
    npt.assert_allclose(
        s_coords.elevation * 180/np.pi, sofa_reference_coordinates[0][..., 1])
    npt.assert_allclose(
        s_coords.radius, sofa_reference_coordinates[0][..., 2])
    npt.assert_allclose(
        r_coords.azimuth * 180/np.pi, sofa_reference_coordinates[1][..., 0])
    npt.assert_allclose(
        r_coords.elevation * 180/np.pi, sofa_reference_coordinates[1][..., 1])
    npt.assert_allclose(
        r_coords.radius, sofa_reference_coordinates[1][..., 2])


def test_convert_sofa_assertion():
    """
    Test assertion for convert_sofa
    (everything else is tested through read_sofa)
    """

    with pytest.raises(TypeError, match="Input must be a sofar.Sofa object"):
        io.convert_sofa("test")


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


def test_write_gammatone_bands(tmpdir):
    """ dsp.filter.GammatoneBands
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'gammatone_bands.far')
    gammatone_bands = pf.dsp.filter.GammatoneBands((0, 22050))
    io.write(filename, gammatone_bands=gammatone_bands)
    actual = io.read(filename)["gammatone_bands"]
    assert isinstance(actual, pf.dsp.filter.GammatoneBands)
    assert actual == gammatone_bands


def test_write_read_numpy_ndarrays(tmpdir):
    """ Numpy ndarray
    Make sure `read` understands the bits written by `write`
    """
    matrix_2d_int = np.arange(0, 24, dtype=int).reshape((4, 6))
    matrix_2d_float = matrix_2d_int.astype(float)
    matrix_2d_complex = matrix_2d_int.astype(complex)

    matrix_3d_int = np.arange(0, 24, dtype=int).reshape((2, 3, 4))
    matrix_3d_float = matrix_3d_int.astype(float)
    matrix_3d_complex = matrix_3d_int.astype(complex)

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


def test_write_read_builtins(dict_of_builtins, tmpdir):
    """ All python builtin types except for exceptions and the types:
    filter, map, object, super, and staticmethod.

    Make sure `read` understands the bits written by `write`
    """
    any_int = 49
    any_bool = True
    filename = os.path.join(tmpdir, 'builtins.far')

    io.write(
        filename,
        any_int=any_int,
        any_bool=any_bool,
        **dict_of_builtins
    )

    actual = io.read(filename)
    assert isinstance(any_int, int)
    assert actual['any_int'] == any_int
    assert isinstance(any_bool, bool)
    assert actual['any_bool'] == any_bool
    # checks if `dict_of_builtins` is a subset of `actual`
    assert dict_of_builtins.items() <= actual.items()


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
        dict_of_builtins,
        tmpdir):
    """ Check if multiple different PyFar-objects can be written to disk
    and read back.
    """
    filename = os.path.join(tmpdir, 'multiplePyfarObjects.far')
    matrix_2d_int = np.arange(0, 24, dtype=int).reshape((4, 6))
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
        matrix_2d_int=matrix_2d_int,
        **dict_of_builtins)
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
    assert dict_of_builtins.items() <= actual.items()


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
        dict_of_builtins,
        tmpdir):
    """ Check if multiple different PyFar-objects can be written to disk
    and read back with zip compression.
    """
    filename = os.path.join(tmpdir, 'multiplePyfarObjects.far')
    matrix_2d_int = np.arange(0, 24, dtype=int).reshape((4, 6))
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
        matrix_2d_int=matrix_2d_int,
        **dict_of_builtins)
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
    assert dict_of_builtins.items() <= actual.items()


@patch('soundfile.read', return_value=(np.array([1., 2., 3.]), 1000))
def test_read_audio_defaults(read_mock):
    """Test correct call of the wrapped soundfile.read() function."""
    signal = pf.io.read_audio('test.wav')
    read_mock.assert_called_with(
        file='test.wav', dtype='float64', always_2d=True)
    assert isinstance(signal, pf.Signal)
    assert np.allclose(signal.time, np.array([1., 2., 3.]))
    assert signal.sampling_rate == 1000


@patch('soundfile.read', return_value=(np.array([1., 2., 3.]), 1000))
def test_read_audio_kwargs(read_mock):
    """Test correct call of the wrapped soundfile.read() function."""
    signal = pf.io.read_audio(
        'test.wav', dtype='float32', kwarg1='kwarg1', kwarg2='kwarg2')
    read_mock.assert_called_with(
        file='test.wav', dtype='float32', always_2d=True,
        kwarg1='kwarg1', kwarg2='kwarg2')
    assert isinstance(signal, pf.Signal)
    assert np.allclose(signal.time, np.array([1., 2., 3.]))
    assert signal.time.shape == (1, 3)
    assert signal.sampling_rate == 1000


@patch(
    'soundfile.read',
    return_value=(np.array([[1., 4.], [2., 5.], [3., 6.]]), 1000))
def test_read_audio_stereo(read_mock):
    """Test correct call of the wrapped soundfile.read() function."""
    signal = pf.io.read_audio('test.wav')
    read_mock.assert_called_with(
        file='test.wav', dtype='float64', always_2d=True)
    assert isinstance(signal, pf.Signal)
    assert np.allclose(signal.time, np.array([[1., 2., 3.], [4., 5., 6.]]))
    assert signal.time.shape == (2, 3)
    assert signal.sampling_rate == 1000


@pytest.mark.parametrize("audio_format", soundfile.available_formats().keys())
@pytest.mark.parametrize("subtype", soundfile.available_subtypes().keys())
def test_write_audio(audio_format, subtype, tmpdir, noise):
    """Test all available audio formats and subtypes."""
    if soundfile.check_format(audio_format, subtype):
        filename = os.path.join(tmpdir, 'test_file.'+audio_format)
        if audio_format == 'AIFF' and subtype == 'DWVW_12':
            # This seems to be an error in soundfile/libsndfile?
            with pytest.raises(RuntimeError):
                io.write_audio(noise, filename, subtype=subtype)
        else:
            io.write_audio(noise, filename, subtype=subtype)
        # In some cases a file 'C._t' is created
        if os.path.exists('C._t'):
            os.remove('C._t')


def test_write_audio_overwrite(noise, tmpdir):
    """Test overwriting behavior."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    io.write_audio(noise, filename)
    # Call with overwrite disabled
    with pytest.raises(FileExistsError):
        io.write_audio(noise, filename, overwrite=False)
    # Call with overwrite enabled
    io.write_audio(noise, filename, overwrite=True)


@patch('soundfile.write')
def test_write_audio_kwargs(sf_write_mock, noise):
    pf.io.write_audio(
        signal=noise, filename='test.wav', kwarg1='kwarg1', kwarg2='kwarg2')
    actual_args = sf_write_mock.call_args[1]
    assert actual_args['kwarg1'] == 'kwarg1'
    assert actual_args['kwarg2'] == 'kwarg2'


def test_write_audio_nd(noise_two_by_three_channel, tmpdir):
    """Test for signals of higher dimension."""
    filename = os.path.join(tmpdir, 'test_wav.wav')
    with pytest.warns(UserWarning, match='flattened'):
        io.write_audio(noise_two_by_three_channel, filename)
    signal_reload = soundfile.read(filename)[0].T
    npt.assert_allclose(
        signal_reload.reshape(noise_two_by_three_channel.time.shape),
        noise_two_by_three_channel.time,
        atol=1e-4)


@patch('soundfile.write')
def test_write_audio_clip(sf_write_mock):
    """Test for clipping warning."""
    signal = pf.Signal([1., 2., 3.], 44100)
    with pytest.warns(Warning, match='clipped'):
        pf.io.write_audio(
            signal=signal, filename='test.wav', subtype='PCM_16')


@pytest.mark.parametrize("audio_format", soundfile.available_formats().keys())
@pytest.mark.parametrize("subtype", soundfile.available_subtypes().keys())
def test_write_audio_read_audio(audio_format, subtype, tmpdir, noise):
    """Test all reading and writing of available audio formats and subtypes."""
    if soundfile.check_format(audio_format, subtype):
        filename = os.path.join(tmpdir, 'test_file.'+audio_format)
        # Write Audio file
        # For exceptions see tests below
        if audio_format == 'AIFF' and subtype == 'DWVW_12':
            with pytest.raises(RuntimeError):
                io.write_audio(noise, filename, subtype=subtype)
        else:
            io.write_audio(noise, filename, subtype=subtype)
        # Read Audio file
        # For exceptions see tests below
        if audio_format == 'AIFF' and 'DWVW' in subtype:
            # This seems to be an error in soundfile/libsndfile?
            with pytest.raises(RuntimeError):
                io.read_audio(filename)
        elif audio_format == 'RAW':
            if 'DWVW' in subtype:
                with pytest.raises(RuntimeError):
                    io.read_audio(
                        filename, samplerate=44100, channels=1,
                        subtype=subtype)
            else:
                # RAW files need to be read with additional parameters
                io.read_audio(
                    filename, samplerate=44100, channels=1, subtype=subtype)
        else:
            # A comparison between written and read signals is not implemented
            # due to the difference caused by the coding
            io.read_audio(filename)
    # In some cases a file 'C._t' is created
    if os.path.exists('C._t'):
        os.remove('C._t')


@pytest.mark.parametrize(
    "subtype", soundfile.available_subtypes('AIFF').keys())
def test_write_audio_read_audio_aiff(subtype, tmpdir, noise):
    """Test for errors in soundfile/libsndfile for AIFF format"""
    filename = os.path.join(tmpdir, 'test_file.aiff')
    if subtype == 'DWVW_12':
        with pytest.raises(RuntimeError):
            io.write_audio(noise, filename, subtype=subtype)
    else:
        io.write_audio(noise, filename, subtype=subtype)
    if 'DWVW' in subtype:
        with pytest.raises(RuntimeError):
            io.read_audio(filename)
    else:
        io.read_audio(filename)


@pytest.mark.parametrize("subtype", soundfile.available_subtypes('RAW').keys())
def test_write_audio_read_audio_raw(subtype, tmpdir, noise):
    """Test for errors in soundfile/libsndfile for RAW format"""
    filename = os.path.join(tmpdir, 'test_file.raw')
    io.write_audio(noise, filename, subtype=subtype)
    if 'DWVW' in subtype:
        with pytest.raises(RuntimeError):
            io.read_audio(
                filename, samplerate=44100, channels=1, subtype=subtype)
    else:
        io.read_audio(filename, samplerate=44100, channels=1, subtype=subtype)


def test_write_audio_pathlib(noise, tmpdir):
    """Test write functionality with filename as pathlib Path object."""
    filename = pathlib.Path(tmpdir, 'test_wav.wav')
    io.write_audio(noise, filename)
    signal_reload = soundfile.read(filename)[0].T
    npt.assert_allclose(
        noise.time,
        np.atleast_2d(signal_reload),
        atol=1e-4)


@patch('soundfile.available_formats')
def test_audio_formats(audio_formats_mock):
    """Test correct call of the wrapped soundfile function."""
    pf.io.audio_formats()
    audio_formats_mock.assert_called()


@patch('soundfile.available_subtypes')
def test_audio_subtypes(audio_subtypes_mock):
    """Test correct call of the wrapped soundfile function."""
    pf.io.audio_subtypes()
    audio_subtypes_mock.assert_called()


@patch('soundfile.default_subtype', return_value='bla')
def test_default_audio_subtype(default_audio_subtype_mock):
    """Test correct call of the wrapped soundfile function."""
    audio_format = 'wav'
    subtype_return = pf.io.default_audio_subtype(format=audio_format)
    assert subtype_return == 'bla'
    default_audio_subtype_mock.assert_called_with(audio_format)


def test_read_comsol_file_not_found():
    """Test error of find not found."""
    path = 'blabla.csv'
    with pytest.raises(FileNotFoundError):
        io.read_comsol(path)


def test_read_comsol_wrong_file_type():
    """Test read comsol file format."""
    filename = 'bla.bla'
    with pytest.raises(SyntaxError):
        io.read_comsol(filename)


@pytest.mark.parametrize("filename,d,n,e",  [
    ("Lp_d2_n6_e1", 2, 6, 1),
    ("Lp_d2_n6_e3", 2, 6, 3),
    ])
@pytest.mark.parametrize("postfix", [(".csv"), (".dat"), (".txt")])
def test_read_comsol_check_output_Lp(filename, d, n, e, postfix):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    out = io.read_comsol(path + postfix)
    assert out['Dimension'] == d
    assert out['Nodes'] == n
    assert out['Coordinates'].csize == n
    assert out['Expressions'] == e
    for key in out['Data']:
        assert out['Data'][key].n_bins == e
        assert out['Data'][key].cshape[0] == n


@pytest.mark.parametrize("filename,d,n,e",  [
    # ("p_t_d2_n1_e1", 2, 1, 1),
    ("p_t_d2_n1_e3", 2, 1, 3),
    ("p_t_d2_n6_e3", 2, 6, 3),
    ("p_t_d1_n6_e3", 1, 6, 3),
    ("p_t_d3_n6_e3", 3, 6, 3),
    ])
def test_read_comsol_check_output_time_data(filename, d, n, e):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    out = io.read_comsol(path + '.txt')
    assert out['Dimension'] == d
    assert out['Nodes'] == n
    assert out['Coordinates'].csize == n
    assert out['Expressions'] == e
    for key in out['Data']:
        assert out['Data'][key].n_samples == e
        assert out['Data'][key].cshape[0] == n


@pytest.mark.parametrize("filename,d,n,e",  [
    ("p_f_d2_n6_e3", 2, 6, 3),
    ("p_f_d2_n1_e3", 2, 1, 3),
    ("p_f_d2_n6_e1", 2, 6, 1),
    ])
def test_read_comsol_check_output_freq_data(filename, d, n, e):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    out = io.read_comsol(path + '.txt')
    assert out['Dimension'] == d
    assert out['Nodes'] == n
    assert out['Coordinates'].csize == n
    assert out['Expressions'] == e
    for key in out['Data']:
        assert out['Data'][key].n_bins == e
        assert out['Data'][key].cshape[0] == n


@pytest.mark.parametrize("filename",  ["Lp_d2_n6_e1", "Lp_d2_n6_e3",])
@pytest.mark.parametrize("type1,type2",
                         [(".txt", ".dat"),
                          (".txt", ".csv"),
                          (".csv", ".dat")])
def test_read_comsol_cmp_same_files_different_formats(filename, type1, type2):
    path = os.path.join(os.getcwd(), 'tests', 'test_io_data', filename)
    out_1 = io.read_comsol(path + type1)
    out_2 = io.read_comsol(path + type2)
    assert out_1['Date'] == out_2['Date']
    assert out_1['Description'] == out_2['Description']
    assert out_1['Dimension'] == out_2['Dimension']
    assert out_1['Expressions'] == out_2['Expressions']
    assert out_1['Length unit'] == out_2['Length unit']
    assert out_1['Model'] == out_2['Model']
    assert out_1['Nodes'] == out_2['Nodes']
    assert out_1['Version'] == out_2['Version']
    assert out_1['Coordinates'] == out_2['Coordinates']
