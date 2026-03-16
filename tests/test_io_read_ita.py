import pyfar as pf
import os
import pytest
import numpy as np

@pytest.mark.parametrize(('filename'), [
    ('freq_itaResult_mult_ww.ita'),
    ('dirac_itaAudio_energy.ita')])

def test_exceptions(filename):
    """
    Tests correct exception raise for the case where the number of
    coordinates and the number of weights for the coordinates do not match.
    """

    filepath = os.path.join('tests', 'test_io_data', 'read_ita', filename)
    message = "weights must have same size as self.csize"
    with pytest.raises(AssertionError, match=message):
            pf.io.read_ita(filepath)

@pytest.mark.parametrize(('filename', 'domain'), [
    ('dirac_itaAudio_old.ita', 'time'),
    ('dirac_itaAudio.ita', 'time'),
    ('freq_itaResult_mult.ita', 'freq'),
    ('time_itaResult.ita', 'time')])

def test_read_ita_domain(filename, domain):
    """Test correct reading of domain of some files."""

    filepath = os.path.join('tests', 'test_io_data', 'read_ita', filename)
    data, _, _, metadata = pf.io.read_ita(filepath)
    print(metadata)
    print("HIER FÄNGT DATA AN", data._data)
    assert metadata["domain"] == domain

@pytest.mark.parametrize(('filename', 'channelCoordinates'), [
    ('freq_itaResult_mult.ita', np.array([[1, 2, 3], [1, 2, 3],
                                        [1, 2, 3], [1, 2, 3]])),
    ('dirac_itaAudio.ita', np.array([[1, 2, 3]]))])

def test_read_ita_channel_coordinates(filename, channelCoordinates):
    """
    Test correct reading of channelCoordinates from metadata and correct
    conversion of this data to pf.Coordinates.
    """

    filepath = os.path.join('tests', 'test_io_data', 'read_ita',  filename)
    _, _, channel_coords, metadata = pf.io.read_ita(filepath)
    metadata_channel_coords_reshaped = \
        metadata["channelCoordinates"]["cart"].reshape(channel_coords.cartesian.shape)
    assert np.array_equal(metadata_channel_coords_reshaped,
                          channel_coords.cartesian)
    assert np.array_equal(channel_coords.cartesian, channelCoordinates)

@pytest.mark.parametrize(('filename', 'data_to_test', 'expected_data'), [
    ('freq_itaResult.ita', 'n_bins', 31),
    ('time_itaResult_mult.ita', 'n_samples', 44100),
    ('dirac_itaAudio.ita', 'n_bins', 22051),
    ('dirac_itaAudio.ita', 'n_samples', 44100)])

def test_read_ita_data_samples_bins(filename, data_to_test, expected_data):
    """
    Test the correct reading of the number of samples for time domain data
    and the number of bins for frequency domain data.
    """

    filepath = os.path.join('tests', 'test_io_data', 'read_ita',  filename)
    data, _, _, _ = pf.io.read_ita(filepath)
    if data_to_test == "n_bins":
        assert data.n_bins == expected_data
    elif data_to_test == "n_samples":
         assert data.n_samples == expected_data

@pytest.mark.parametrize(('filename', 'dimension'), [
    ('dirac_itaAudio.ita', (1,)),
    ('dirac_itaAudio_old.ita', (1,)),
    ('freq_itaResult.ita', (1,)),
    ('time_itaResult.ita',(1,)),
    ('dirac_itaAudio_mult.ita', (4,)),
    ('freq_itaResult_mult.ita',(4,)),
    ('time_itaResult_mult.ita', (4,))])

def test_read_ita_data_dimension(filename, dimension):
    """Test the correct reading of the number of dimensions from the data."""

    filepath = os.path.join('tests', 'test_io_data', 'read_ita', filename)
    data, _, _, _ = pf.io.read_ita(filepath)
    assert data.cshape == dimension
