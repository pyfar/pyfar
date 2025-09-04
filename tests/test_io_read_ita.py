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

    filepath = os.path.join('tests', 'test_io_data', filename)
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

    filepath = os.path.join('tests', 'test_io_data', filename)
    _, _, _, metadata = pf.io.read_ita(filepath)
    assert metadata["domain"] == domain

@pytest.mark.parametrize(('filename', 'channelCoordinates'), [
    ('freq_itaResult_mult.ita', np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])),
    ('dirac_itaAudio.ita', np.array([[1, 2, 3]]))])

def test_read_ita_channel_coordinates(filename, channelCoordinates):
    """
    Test correct reading of channelCoordinates from metadata and correct
    conversion of this data to pf.Coordinates.
    """

    filepath = os.path.join('tests', 'test_io_data', filename)
    _, _, channel_coords, metadata = pf.io.read_ita(filepath)
    metadata_channel_coords_reshaped = \
        metadata["channelCoordinates"]["cart"].reshape(channel_coords.cartesian.shape)
    assert np.array_equal(metadata_channel_coords_reshaped, channel_coords.cartesian) and \
        np.array_equal(channel_coords.cartesian, channelCoordinates)
