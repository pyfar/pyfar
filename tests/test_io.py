import numpy as np
import pytest
import os.path
import scipy.io.wavfile as wavfile

from haiopy import Signal
from haiopy import io


def test_read_wav():
    """Test default without optional parameters."""
    sampling_rate = 44100
    noise = np.random.rand(1000)
    filename = "test_wav.wav"
    # Create testfile
    wavfile.write(filename, sampling_rate, noise)
    signal = io.read_wav(filename)
    os.remove(filename)
    assert isinstance(signal, Signal)


def test_write_wav():
    """Test default without optional parameters."""
    sampling_rate = 44100
    noise = np.random.rand(1000)
    signal = Signal(noise, sampling_rate, domain='time')
    filename = "test_wav.wav"
    io.write_wav(signal, filename)
    assert os.path.isfile(filename)
    os.remove(filename)


def test_write_wav_overwrite():
    """Test overwriting behavior."""
    sampling_rate = 44100
    noise = np.random.rand(1000)
    signal = Signal(noise, sampling_rate, domain='time')
    filename = "test_wav.wav"
    io.write_wav(signal, filename)
    # Call with overwrite disabled
    with pytest.raises(FileExistsError):
        io.write_wav(signal, filename, overwrite=False)
    # Call with overwrite enabled
    io.write_wav(signal, filename, overwrite=True)
    os.remove(filename)
