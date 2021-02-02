from pyfar.spatial.spatial import SphericalVoronoi
import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile

from pyfar.orientations import Orientations
from pyfar.coordinates import Coordinates
from pyfar.signal import Signal
import pyfar.dsp.classes as fo
import pyfar.io

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
    n_samples = int(1e5 - 1)
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



@pytest.fixture
def generate_far_file_orientations(tmpdir, orientations):
    """Create a far file in temporary folder that contains an Orientations
    object.
    """
    filename = os.path.join(tmpdir, 'test_orientations.far')
    pyfar.io.write(filename, orientations=orientations)
    return filename


@pytest.fixture
def generate_far_file_coordinates(tmpdir, coordinates):
    """Create a far file in temporary folder that contains an Coordinates
    object.
    """
    filename = os.path.join(tmpdir, 'test_coordinates.far')
    pyfar.io.write(filename, coordinates=coordinates)
    return filename


@pytest.fixture
def generate_far_file_signal(tmpdir, signal):
    """Create a far file in temporary folder that contains an Signal
    object.
    """
    filename = os.path.join(tmpdir, 'test_signal.far')
    pyfar.io.write(filename, signal=signal)
    return filename


@pytest.fixture
def generate_far_file_sphericalvoronoi(tmpdir, sphericalvoronoi):
    """Create a far file in temporary folder that contains an SphericalVoronoi
    object.
    """
    filename = os.path.join(tmpdir, 'test_sphericalvoronoi.far')
    pyfar.io.write(filename, sphericalvoronoi=sphericalvoronoi)
    return filename


@pytest.fixture
def generate_far_file_filter(tmpdir, filter):
    """Create a far file in temporary folder that contains an SphericalVoronoi
    object.
    """
    filename = os.path.join(tmpdir, 'test_filter.far')
    pyfar.io.write(filename, filter=filter)
    return filename


@pytest.fixture
def generate_far_file_filterIIR(tmpdir, filterIIR):
    """Create a far file in temporary folder that contains an SphericalVoronoi
    object.
    """
    filename = os.path.join(tmpdir, 'test_filterIIR.far')
    pyfar.io.write(filename, filterIIR=filterIIR)
    return filename


@pytest.fixture
def generate_far_file_filterFIR(tmpdir, filterFIR):
    """Create a far file in temporary folder that contains an SphericalVoronoi
    object.
    """
    filename = os.path.join(tmpdir, 'test_filterFIR.far')
    pyfar.io.write(filename, filterFIR=filterFIR)
    return filename


@pytest.fixture
def generate_far_file_nested_data_struct(tmpdir, nested_data_struct):
    filename = os.path.join(tmpdir, 'test_nested_data_struct.far')
    pyfar.io.write(filename, nested_data_struct=nested_data_struct)
    return filename


@pytest.fixture
def signal():
    # TODO: replace sine with fixture sine
    sine = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 44100))
    return Signal(sine, 44100, len(sine), domain='time')


@pytest.fixture
def sphericalvoronoi():
    dihedral = 2 * np.arcsin(np.cos(np.pi / 3) / np.sin(np.pi / 5))
    R = np.tan(np.pi / 3) * np.tan(dihedral / 2)
    rho = np.cos(np.pi / 5) / np.sin(np.pi / 10)

    theta1 = np.arccos(
        (np.cos(np.pi / 5) / np.sin(np.pi / 5)) /
        np.tan(np.pi / 3))

    a2 = 2 * np.arccos(rho / R)

    theta2 = theta1 + a2
    theta3 = np.pi - theta2
    theta4 = np.pi - theta1

    phi1 = 0
    phi2 = 2 * np.pi / 3
    phi3 = 4 * np.pi / 3

    theta = np.concatenate((
        np.tile(theta1, 3),
        np.tile(theta2, 3),
        np.tile(theta3, 3),
        np.tile(theta4, 3)))
    phi = np.tile(np.array(
            [phi1, phi2, phi3, phi1 + np.pi / 3,
             phi2 + np.pi / 3, phi3 + np.pi / 3]), 2)
    rad = np.ones(np.size(theta))

    s = Coordinates(
        phi, theta, rad,
        domain='sph', convention='top_colat')
    return SphericalVoronoi(s)


@pytest.fixture
def filter():
    coeff = np.array([[[1, 0, 0], [1, 0, 0]]])
    state = np.array([[[1, 0]]])
    return fo.Filter(coefficients=coeff, state=state, comment='my comment')


@pytest.fixture
def filterIIR():
    coeff = np.array([[1, 1 / 2, 0], [1, 0, 0]])
    return fo.FilterIIR(coeff, sampling_rate=2 * np.pi)


@pytest.fixture
def filterFIR():
    coeff = np.array([
        [1, 1 / 2, 0],
        [1, 1 / 4, 1 / 8]])
    desired = np.array([
        [[1, 1 / 2, 0], [1, 0, 0]],
        [[1, 1 / 4, 1 / 8], [1, 0, 0]]
        ])
    return fo.FilterFIR(coeff, sampling_rate=2*np.pi)


@pytest.fixture
def filterSOS():
    sos = np.array([[1, 1/2, 0, 1, 0, 0]])
    return fo.FilterSOS(sos, sampling_rate=2*np.pi)


@pytest.fixture
def nested_data_struct():
    n = 42
    comment = 'My String'
    matrix = np.arange(0, 24).reshape((2, 3, 4))
    subobj = stub_utils.MyOtherClass()
    mylist = [1, np.int32, np.arange(10), stub_utils.MyOtherClass()]
    mydict = {
        'a': 1,
        'b': np.int32,
        'c': np.arange(10),
        'd': stub_utils.MyOtherClass()}
    return stub_utils.NestedDataStruct(
        n, comment, matrix, subobj, mylist, mydict)