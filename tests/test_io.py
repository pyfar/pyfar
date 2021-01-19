import numpy as np
import numpy.testing as npt
import pytest
import os.path
import scipy.io.wavfile as wavfile

from pyfar import io
from pyfar import Signal


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


def test_read_sofa_GeneralTF(
        generate_sofa_GeneralTF, noise_two_by_three_channel):
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
