import numpy as np
import numpy.testing as npt
import pytest

from unittest import mock
import os.path
import scipy.io.wavfile as wavfile
import sofa

from pyfar import io
from pyfar import Signal


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


def test_read_sofa(tmpdir):
    """Test for sofa signal properties"""
    td = tmpdir.strpath
    # Generate test files
    generate_test_sofas(td)
    # Correct DataType
    filename = os.path.join(td, 'GeneralFIR.sofa')
    signal = io.read_sofa(filename)[0]
    signal_ref = reference_signal(signal.cshape)[0]
    npt.assert_allclose(
            signal.time,
            signal_ref,
            rtol=1e-10)
    # Wrong DataType
    filename = os.path.join(td, 'GeneralTF.sofa')
    with pytest.raises(ValueError):
        io.read_sofa(filename)
    # Wrong sampling rate Unit
    filename = os.path.join(td, 'GeneralFIR_unit.sofa')
    with pytest.raises(ValueError):
        io.read_sofa(filename)

    # Correct coordinates
    filename = os.path.join(td, 'GeneralFIR.sofa')
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
    # Wrong PositionType
    filename = os.path.join(td, 'GeneralFIR_postype.sofa')
    with pytest.raises(ValueError):
        io.read_sofa(filename)


def generate_test_sofas(filedir):
    """ Generate the reference sofa files used for testing the read_sofa function.
    Parameters
    -------
    filedir : String
        Path to directory.
    """
    conventions = [
        'GeneralFIR',
        'GeneralTF',
        'GeneralFIR_unit',
        'GeneralFIR_postype']
    n_measurements = 1
    n_receivers = 2
    signal, sampling_rate = reference_signal(
        shape=(n_measurements, n_receivers))
    n_samples = signal.shape[-1]

    for convention in conventions:
        filename = os.path.join(filedir, (convention + '.sofa'))
        sofafile = sofa.Database.create(
                        filename,
                        convention.split('_')[0],
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

        if convention == 'GeneralFIR':
            sofafile.Data.Type = 'FIR'
            sofafile.Data.initialize()
            sofafile.Data.IR = signal
            sofafile.Data.SamplingRate = sampling_rate
        elif convention == 'GeneralTF':
            sofafile.Data.Type = 'TF'
            sofafile.Data.initialize()
            sofafile.Data.Real = signal
            sofafile.Data.Imag = signal
        elif convention == 'GeneralFIR_unit':
            sofafile.Data.Type = 'FIR'
            sofafile.Data.initialize()
            sofafile.Data.IR = signal
            sofafile.Data.SamplingRate = sampling_rate
            sofafile.Data.SamplingRate.Units = 'not_hertz'
        elif convention == 'GeneralFIR_postype':
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
    times = np.arange(0, n_periods*frequency) / sampling_rate
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
    signal_object = mock.Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time[np.newaxis, :]
    signal_object.sampling_rate = sampling_rate

    return signal_object


@pytest.fixture
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
    signal_object = mock.Mock(spec_set=Signal(time, sampling_rate))
    signal_object.time = time
    signal_object.sampling_rate = sampling_rate

    return signal_object
