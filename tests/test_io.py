from pyfar.orientations import Orientations
import numpy as np
import numpy.testing as npt
import pytest
from mock import patch
from tests.conftest import stub_str_to_type

import os.path
import pyfar._codec as codec
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


def test__str_to_type():
    """ Test if str_to_type works properly. """
    PyfarType = codec._str_to_type('Coordinates')
    assert PyfarType.__module__.startswith('pyfar')
    PyfarType = codec._str_to_type('Orientations')
    assert PyfarType.__module__.startswith('pyfar')
    PyfarType = codec._str_to_type('SphericalVoronoi')
    assert PyfarType.__module__.startswith('pyfar')
    pass


def test__eq___dict__flat_data(flat_data):
    """ Test equality for stub. """
    actual = flat_data.copy()
    assert actual == flat_data


def test__eq___dict__nested_data(nested_data):
    """ Test equality for stub. """
    actual = nested_data.copy()
    assert actual == nested_data


@patch('pyfar._codec._str_to_type', new=stub_str_to_type())
def test_write_read_flat_data(tmpdir, flat_data):
    """ Check if file can be read back after writing without explicitply
    passing the .far-extension.
    """
    filename = os.path.join(tmpdir, 'write_read_flat_data.far')
    io.write(filename, flat_data=flat_data)
    actual = io.read(filename)
    assert actual['flat_data'] == flat_data


@patch('pyfar._codec._str_to_type', new=stub_str_to_type())
def test_write_read_nested_data(nested_data, flat_data, tmpdir):
    filename = os.path.join(tmpdir, 'write_nested_flat_data.far')
    io.write(filename, nested_data=nested_data)
    actual = io.read(filename)
    assert actual['nested_data'] == nested_data


@patch('pyfar._codec._str_to_type', new=stub_str_to_type())
def test_write_read_multipleObjects(flat_data, tmpdir):
    """ Check if file can be read back after writing without explicitply
    passing the .far-extension.
    """
    filename = os.path.join(tmpdir, 'write_read_multipleObjects.far')
    io.write(
        filename,
        obj1=flat_data,
        obj2=flat_data)
    actual = io.read(filename)
    assert actual['obj1'] == flat_data
    assert actual['obj2'] == flat_data


def test_write_anyObj_TypeError(any_obj, tmpdir):
    """ Check if a TypeError is raised when writing an arbitrary
    object.
    """
    filename = os.path.join(tmpdir, 'test_anyObj.far')
    with pytest.raises(TypeError):
        io.write(filename, any_obj=any_obj)


def test_read_orientations(orientations, tmpdir):
    """ Orientations
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'orientations.far')
    io.write(filename, orientations=orientations)
    actual = io.read(filename)['orientations']
    assert isinstance(actual, Orientations)
    assert actual == orientations


def test_read_coordinates(coordinates, tmpdir):
    """ Coordinates
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'coordinates.far')
    io.write(filename, coordinates=coordinates)
    actual = io.read(filename)['coordinates']
    assert isinstance(actual, Coordinates)
    assert actual == coordinates


def test_read_signal(signal, tmpdir):
    """ Signal
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'signal.far')
    io.write(filename, signal=signal)
    actual = io.read(filename)['signal']
    assert isinstance(actual, Signal)
    assert actual == signal


def test_read_sphericalvoronoi(sphericalvoronoi, tmpdir):
    """ SphericalVoronoi
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'sphericalvoronoi.far')
    io.write(filename, sphericalvoronoi=sphericalvoronoi)
    actual = io.read(filename)['sphericalvoronoi']
    assert isinstance(actual, SphericalVoronoi)
    assert actual == sphericalvoronoi


def test_read_filter(filter, tmpdir):
    """ Filter
    Make sure `read` understands the bits written by `write`
    """
    filename = os.path.join(tmpdir, 'filter.far')
    io.write(filename, filter=filter)
    actual = io.read(filename)['filter']
    assert isinstance(actual, fo.Filter)
    assert actual == filter


def test_write_filterIIR_TypeError(filterIIR, tmpdir):
    """ FilterIIR
    Can't be written to disk because objects store user-defined function.
    """
    filename = os.path.join(tmpdir, 'filterIIR.far')
    with pytest.raises(TypeError):
        io.write(filename, filterIIR=filterIIR)


def test_write_filterFIR_TypeError(filterFIR, tmpdir):
    """ FilterFIR
    Can't be written to disk because objects store user-defined function.
    """
    filename = os.path.join(tmpdir, 'filterIIR.far')
    with pytest.raises(TypeError):
        io.write(filename, filterFIR=filterFIR)


def test_write_filterSOS_TypeError(filterSOS, tmpdir):
    """ FilterIIR
    Can't be written to disk because objects store user-defined function.
    """
    filename = os.path.join(tmpdir, 'filterSOS.far')
    with pytest.raises(TypeError):
        io.write(filename, filterSOS=filterSOS)


def test_write_read_multiplePyfarObjects(
        filter,
        coordinates,
        orientations,
        sphericalvoronoi,
        tmpdir):
    """ Check if multiple different PyFar-objects can be written to disk
    and read back.
    """
    filename = os.path.join(tmpdir, 'multiplePyfarObjects.far')
    io.write(
        filename,
        filter=filter,
        coordinates=coordinates,
        orientations=orientations,
        sphericalvoronoi=sphericalvoronoi)
    actual = io.read(filename)
    assert isinstance(actual['filter'], fo.Filter)
    assert actual['filter'] == filter
    assert isinstance(actual['coordinates'], Coordinates)
    assert actual['coordinates'] == coordinates
    assert isinstance(actual['orientations'], Orientations)
    assert actual['orientations'] == orientations
    assert isinstance(actual['sphericalvoronoi'], SphericalVoronoi)
    assert actual['sphericalvoronoi'] == sphericalvoronoi


def test_write_read_multiplePyfarObjectsWithCompression(
        filter,
        coordinates,
        orientations,
        sphericalvoronoi,
        tmpdir):
    """ Check if multiple different PyFar-objects can be written to disk
    and read back with zip compression.
    """
    filename = os.path.join(tmpdir, 'multiplePyfarObjects.far')
    io.write(
        filename,
        compress=True,
        filter=filter,
        coordinates=coordinates,
        orientations=orientations,
        sphericalvoronoi=sphericalvoronoi)
    actual = io.read(filename)
    assert isinstance(actual['filter'], fo.Filter)
    assert actual['filter'] == filter
    assert isinstance(actual['coordinates'], Coordinates)
    assert actual['coordinates'] == coordinates
    assert isinstance(actual['orientations'], Orientations)
    assert actual['orientations'] == orientations
    assert isinstance(actual['sphericalvoronoi'], SphericalVoronoi)
    assert actual['sphericalvoronoi'] == sphericalvoronoi
