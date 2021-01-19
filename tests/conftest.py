import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile

from pyfar.orientations import Orientations

import stub_utils


@pytest.fixture
def sine():
    """Sine signal stub.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def sine_rms():
    """Sine signal stub,
    RMS FFT-normalization.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def sine_odd():
    """Sine signal stub,
    odd number of samples.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 9999
    fft_norm = 'none'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def sine_odd_rms():
    """Sine signal stub,
    odd number of samples,
    RMS FFT-normalization

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 9999
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def sine_two_by_two_channel():
    """2-by-2 channel sine signal stub.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = np.array([[1, 2], [3, 4]]) * 441
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2, 2)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def impulse():
    """Delta impulse signal stub.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    """
    delay = 0
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def impulse_rms():
    """Delta impulse signal stub,
    RMS FFT-normalization.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    """
    delay = 0
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'rms'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def impulse_group_delay():
    """Delayed delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    delay = 1000
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)
    group_delay = delay * np.ones_like(freq, dtype=float)

    return signal, group_delay


@pytest.fixture
def impulse_group_delay_two_channel():
    """Delayed 2 channel delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    delay = np.atleast_1d([1000, 2000])
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)
    group_delay = delay[..., np.newaxis] * np.ones_like(freq, dtype=float)

    return signal, group_delay


@pytest.fixture
def impulse_group_delay_two_by_two_channel():
    """Delayed 2-by-2 channel delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    delay = np.array([[1000, 2000], [3000, 4000]])
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2, 2)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)
    group_delay = delay[..., np.newaxis] * np.ones_like(freq, dtype=float)

    return signal, group_delay


@pytest.fixture
def sine_plus_impulse():
    """Combined sine and delta impulse signal stub.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    delay = 100
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time_sine, freq_sine, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    time_imp, freq_imp = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = stub_utils.signal_stub(
        time_sine + time_imp, freq_sine + freq_imp, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def noise():
    """Gaussian white noise signal stub.
    The frequency spectrum is set to dummy value None.

    Returns
    -------
    signal : Signal
        Stub of noise signal
    """
    sigma = 1
    n_samples = int(1e5)
    cshape = (1,)
    sampling_rate = 44100
    fft_norm = 'rms'
    freq = None

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def noise_odd():
    """Gaussian white noise signal stub,
    odd number of samples.
    The frequency spectrum is set to dummy value None.

    Returns
    -------
    signal : Signal
        Stub of noise signal
    """
    sigma = 1
    n_samples = int(1e5-1)
    cshape = (1,)
    sampling_rate = 44100
    fft_norm = 'rms'
    freq = None

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def noise_two_by_two_channel():
    """ 2-by-2 channel gaussian white noise signal stub.
    The frequency spectrum is set to dummy value None.

    Returns
    -------
    signal : Signal
        Stub of noise signal
    """
    sigma = 1
    n_samples = int(1e5)
    cshape = (2, 2)
    sampling_rate = 44100
    fft_norm = 'rms'
    freq = None

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def noise_two_by_three_channel():
    """ 2-by-3 channel gaussian white noise signal stub.
    The frequency spectrum is created with np.fft.rfft.

    Returns
    -------
    signal : Signal
        Stub of noise signal
    """
    sigma = 1
    n_samples = int(1e5)
    cshape = (2, 3)
    sampling_rate = 44100
    fft_norm = 'none'

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    freq = np.fft.rfft(time)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def fft_lib_np(monkeypatch):
    """Set numpy.fft as fft library.
    """
    import pyfar.fft
    monkeypatch.setattr(pyfar.fft, 'fft_lib', np.fft)


@pytest.fixture
def fft_lib_pyfftw(monkeypatch):
    """Set pyfftw as fft library.
    """
    import pyfar.fft
    from pyfftw.interfaces import numpy_fft as npi_fft
    monkeypatch.setattr(pyfar.fft, 'fft_lib', npi_fft)


@pytest.fixture
def generate_wav_file(tmpdir, noise):
    """Create wav file in temporary folder.
    """
    filename = os.path.join(tmpdir, 'test_wav.wav')
    wavfile.write(filename, noise.sampling_rate, noise.time.T)
    return filename


@pytest.fixture
def sofa_reference_coordinates(noise_two_by_three_channel):
    """Define coordinates to write in reference files.
    """
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    source_coordinates = np.random.rand(n_measurements, 3)
    receiver_coordinates = np.random.rand(n_receivers, n_measurements, 3)
    return source_coordinates, receiver_coordinates


@pytest.fixture
def generate_sofa_GeneralFIR(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralFIR.
    """
    sofatype = 'GeneralFIR'
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    n_samples = noise_two_by_three_channel.n_samples
    dimensions = {"M": n_measurements, "R": n_receivers, "N": n_samples}

    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    sofafile = sofa.Database.create(filename, sofatype, dimensions=dimensions)

    sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.initialize(variances=["Position"], fixed=["View", "Up"])
    sofafile.Source.Position.set_values(sofa_reference_coordinates[0])
    sofafile.Receiver.initialize(variances=["Position"], fixed=["View", "Up"])
    r_coords = np.transpose(sofa_reference_coordinates[1], (0, 2, 1))
    sofafile.Receiver.Position.set_values(r_coords)
    sofafile.Emitter.initialize(fixed=["Position", "View", "Up"], count=1)
    sofafile.Data.Type = 'FIR'
    sofafile.Data.initialize()
    sofafile.Data.IR = noise_two_by_three_channel.time
    sofafile.Data.SamplingRate = noise_two_by_three_channel.sampling_rate

    sofafile.close()
    return filename


@pytest.fixture
def generate_sofa_GeneralTF(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralTF.
    """
    sofatype = 'GeneralTF'
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    n_bins = noise_two_by_three_channel.n_bins
    dimensions = {"M": n_measurements, "R": n_receivers, "N": n_bins}

    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    sofafile = sofa.Database.create(filename, sofatype, dimensions=dimensions)

    sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.initialize(variances=["Position"], fixed=["View", "Up"])
    sofafile.Source.Position.set_values(sofa_reference_coordinates[0])
    sofafile.Receiver.initialize(variances=["Position"], fixed=["View", "Up"])
    r_coords = np.transpose(sofa_reference_coordinates[1], (0, 2, 1))
    sofafile.Receiver.Position.set_values(r_coords)
    sofafile.Emitter.initialize(fixed=["Position", "View", "Up"], count=1)
    sofafile.Data.Type = 'TF'
    sofafile.Data.initialize()
    sofafile.Data.Real.set_values(np.real(noise_two_by_three_channel.freq))
    sofafile.Data.Imag.set_values(np.imag(noise_two_by_three_channel.freq))

    sofafile.close()
    return filename


@pytest.fixture
def generate_sofa_unit_error(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralFIR
    with incorrect sampling rate unit.
    """
    sofatype = 'GeneralFIR'
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    n_samples = noise_two_by_three_channel.n_samples
    dimensions = {"M": n_measurements, "R": n_receivers, "N": n_samples}

    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    sofafile = sofa.Database.create(filename, sofatype, dimensions=dimensions)

    sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.initialize(variances=["Position"], fixed=["View", "Up"])
    sofafile.Source.Position.set_values(sofa_reference_coordinates[0])
    sofafile.Receiver.initialize(variances=["Position"], fixed=["View", "Up"])
    r_coords = np.transpose(sofa_reference_coordinates[1], (0, 2, 1))
    sofafile.Receiver.Position.set_values(r_coords)
    sofafile.Emitter.initialize(fixed=["Position", "View", "Up"], count=1)
    sofafile.Data.Type = 'FIR'
    sofafile.Data.initialize()
    sofafile.Data.IR = noise_two_by_three_channel.time
    sofafile.Data.SamplingRate = noise_two_by_three_channel.sampling_rate
    sofafile.Data.SamplingRate.Units = 'not_hertz'

    sofafile.close()
    return filename


@pytest.fixture
def generate_sofa_postype_error(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralFIR
    with incorrect position type.
    """
    sofatype = 'GeneralFIR'
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    n_samples = noise_two_by_three_channel.n_samples
    dimensions = {"M": n_measurements, "R": n_receivers, "N": n_samples}

    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    sofafile = sofa.Database.create(filename, sofatype, dimensions=dimensions)

    sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.initialize(variances=["Position"], fixed=["View", "Up"])
    sofafile.Source.Position.set_values(sofa_reference_coordinates[0])
    sofafile.Receiver.initialize(variances=["Position"], fixed=["View", "Up"])
    r_coords = np.transpose(sofa_reference_coordinates[1], (0, 2, 1))
    sofafile.Receiver.Position.set_values(r_coords)
    sofafile.Emitter.initialize(fixed=["Position", "View", "Up"], count=1)
    sofafile.Data.Type = 'FIR'
    sofafile.Data.initialize()
    sofafile.Data.IR = noise_two_by_three_channel.time
    sofafile.Data.SamplingRate = noise_two_by_three_channel.sampling_rate
    sofafile.Source.Position.Type = 'wrong_type'

    sofafile.close()
    return filename


@pytest.fixture
def views():
    return [[1, 0, 0], [2, 0, 0], [-1, 0, 0]]


@pytest.fixture
def ups():
    return [[0, 1, 0], [0, -2, 0], [0, 1, 0]]


@pytest.fixture
def positions():
    return [[0, 0.5, 0], [0, -0.5, 0], [1, 1, 1]]


@pytest.fixture
def orientations(views, ups):
    return Orientations.from_view_up(views, ups)
