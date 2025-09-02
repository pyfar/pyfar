import pyfar as pf
import os
import pytest


@pytest.mark.parametrize(('filename', 'data_type'), [
    ('freq_itaResult.ita', 'signal'),
    ('freq_itaResult_mult_ww.ita', 'data'),
    ])
def test_error(filename, data_type):
    """Tests correct exception raise for example *.ita files."""
    filepath = os.path.join('tests', 'test_io_data', filename)
    if (filename == 'freq_itaResult.ita' and data_type == 'signal'):
        message = "The itaResult object can't contain a signal."
        with pytest.raises(Exception, match=message):
            pf.io.read_ita(filepath)
        pass
    elif (filename == 'freq_itaResult_mult_ww.ita'):
        message = "channelCoordinates.weights must have the same size as\
                   channelCoordinates.csize."
        with pytest.raises(Exception, match=message):
            pf.io.read_ita(filepath)
    else:
        pf.io.read_ita(filepath)


@pytest.mark.parametrize(('filename', 'data_type'), [
    ('dirac_itaAudio.ita', 'signal'),
    ('dirac_itaAudio_oldCoordiantes.ita', 'signal'),
    ('dirac_itaAudio_mult.ita', 'signal'),
    ('dirac_itaAudio_mult.ita', 'data'),
    ('dirac_itaAudio_energy.ita', 'signal'),
    ('freq_itaResult.ita', 'data'),
    ('freq_itaResult_mult.ita', 'data'),
    ('time_itaResult.ita', 'data'),
    ('time_itaResult_mult.ita', 'data')])
def test_read_ita_files(filename, data_type):
    """Tests correct exception raise for example *.ita files."""
    filepath = os.path.join('tests', 'test_io_data', filename)
    data, obj_coords, channel_coords, metadata = pf.io.read_ita(filepath)
