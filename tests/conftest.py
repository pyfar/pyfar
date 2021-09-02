import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile

from pyfar.samplings import SphericalVoronoi
from pyfar import Orientations
from pyfar import Coordinates
from pyfar import FrequencyData, TimeData
import pyfar.classes.filter as fo
import pyfar.signals

from pyfar.testing import stub_utils


@pytest.fixture
def sine_stub():
    """Sine signal stub.
    To be used in cases, when a dependence on the Signal class is prohibited,
    but a correct, fixed relation of the time signal and the spectrum is
    needed.

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
def sine_stub_odd():
    """Sine signal stub, odd number of samples
    To be used in cases, when a dependence on the Signal class is prohibited,
    but a correct, fixed relation of the time signal and the spectrum is
    needed.

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
def impulse_stub():
    """Delta impulse signal stub.
    To be used in cases, when a dependence on the Signal class is prohibited,
    but a correct, fixed relation of the time signal and the spectrum is
    needed.

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
def noise_stub():
    """Gaussian white noise signal stub.
    To be used in cases, when a dependence on the Signal class is prohibited,
    but a correct, fixed relation of the time signal and the spectrum is
    needed.

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

    time, freq = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def noise_stub_odd():
    """Gaussian white noise signal stub, odd number of samples.
    To be used in cases, when a dependence on the Signal class is prohibited,
    but a correct, fixed relation of the time signal and the spectrum is
    needed.

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

    time, freq = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = stub_utils.signal_stub(
        time, freq, sampling_rate, fft_norm)

    return signal


@pytest.fixture
def sine():
    """Sine signal.

    Returns
    -------
    signal : Signal
        Sine signal
    """
    frequency = 441
    n_samples = 10000
    sampling_rate = 44100
    amplitude = 1

    signal = pyfar.signals.sine(
        frequency, n_samples, amplitude=amplitude,
        sampling_rate=sampling_rate)

    return signal


@pytest.fixture
def sine_short():
    """Short sine signal where the first frequency is > 20 Hz.

    This is used for testing plot._line._lower_frequency_limit.

    Returns
    -------
    signal : Signal
        Sine signal
    """
    frequency = 441
    n_samples = 100
    sampling_rate = 44100
    amplitude = 1

    signal = pyfar.signals.sine(
        frequency, n_samples, amplitude=amplitude,
        sampling_rate=sampling_rate)

    return signal


@pytest.fixture
def impulse():
    """Delta impulse signal.

    Returns
    -------
    signal : Signal
        Impulse signal
    """
    n_samples = 10000
    delay = 0
    amplitude = 1
    sampling_rate = 44100

    signal = pyfar.signals.impulse(
        n_samples, delay=delay, amplitude=amplitude,
        sampling_rate=sampling_rate)

    return signal


@pytest.fixture
def impulse_group_delay():
    """Delayed delta impulse signal with analytical group delay.

    Returns
    -------
    signal : Signal
        Impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    n_samples = 10000
    delay = 0
    amplitude = 1
    sampling_rate = 44100

    signal = pyfar.signals.impulse(
        n_samples, delay=delay, amplitude=amplitude,
        sampling_rate=sampling_rate)
    group_delay = delay * np.ones_like(signal.freq, dtype=float)

    return signal, group_delay


@pytest.fixture
def impulse_group_delay_two_channel():
    """Delayed 2 channel delta impulse signal with analytical group delay.

    Returns
    -------
    signal : Signal
        Impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    n_samples = 10000
    delay = np.atleast_1d([1000, 2000])
    amplitude = np.atleast_1d([1, 1])
    sampling_rate = 44100

    signal = pyfar.signals.impulse(
        n_samples, delay=delay, amplitude=amplitude,
        sampling_rate=sampling_rate)
    group_delay = delay[..., np.newaxis] * np.ones_like(
        signal.freq, dtype=float)

    return signal, group_delay


@pytest.fixture
def impulse_group_delay_two_by_two_channel():
    """Delayed 2-by-2 channel delta impulse signal with analytical group delay.

    Returns
    -------
    signal : Signal
        Impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    n_samples = 10000
    delay = np.array([[1000, 2000], [3000, 4000]])
    amplitude = np.atleast_1d([[1, 1], [1, 1]])
    sampling_rate = 44100

    signal = pyfar.signals.impulse(
        n_samples, delay=delay, amplitude=amplitude,
        sampling_rate=sampling_rate)
    group_delay = delay[..., np.newaxis] * np.ones_like(
        signal.freq, dtype=float)

    return signal, group_delay


@pytest.fixture
def sine_plus_impulse():
    """Added sine and delta impulse signals.

    Returns
    -------
    signal : Signal
        Combined signal
    """
    frequency = 441
    delay = 100
    n_samples = 10000
    sampling_rate = 44100
    amplitude = 1

    sine_signal = pyfar.signals.sine(
        frequency, n_samples, amplitude=amplitude,
        sampling_rate=sampling_rate)
    sine_signal.fft_norm = 'none'

    impulse_signal = pyfar.signals.impulse(
        n_samples, delay=delay, amplitude=amplitude,
        sampling_rate=sampling_rate)
    signal = sine_signal + impulse_signal

    return signal


@pytest.fixture
def noise():
    """Gaussian white noise signal.

    Returns
    -------
    signal : Signal
        Noise signal
    """
    n_samples = 10000
    rms = 1
    sampling_rate = 44100
    seed = 1234

    signal = pyfar.signals.noise(
        n_samples, spectrum="white", rms=rms, sampling_rate=sampling_rate,
        seed=seed)

    return signal


@pytest.fixture
def noise_two_by_three_channel():
    """ 2-by-3 channel gaussian white noise signal.

    Returns
    -------
    signal : Signal
        Noise signal
    """
    n_samples = 10000
    rms = np.ones((2, 3))
    sampling_rate = 44100
    seed = 1234

    signal = pyfar.signals.noise(
        n_samples, spectrum="white", rms=rms, sampling_rate=sampling_rate,
        seed=seed)

    return signal


@pytest.fixture
def time_data():
    """
    TimeData object with three data points.

    Returns
    -------
    time_data TimeData
        Data
    """
    time_data = TimeData([1, 0, -1], [0, .1, .4])
    return time_data


@pytest.fixture
def frequency_data():
    """
    FrequencyData object with three data points.

    Returns
    -------
    frequency_data FrequencyData
        Data
    """
    frequency_data = FrequencyData([2, .25, .5], [100, 1000, 20000])
    return frequency_data


@pytest.fixture
def frequency_data_one_point():
    """
    FrequencyData object with one data point.

    Returns
    -------
    frequency_data FrequencyData
        Data
    """
    frequency_data = FrequencyData([2], [0])
    return frequency_data


@pytest.fixture
def fft_lib_np(monkeypatch):
    """Set numpy.fft as fft library.
    """
    import pyfar.dsp.fft
    monkeypatch.setattr(pyfar.dsp.fft, 'fft_lib', np.fft)
    return np.fft.__name__


@pytest.fixture
def fft_lib_pyfftw(monkeypatch):
    """Set pyfftw as fft library.
    """
    import pyfar.dsp.fft
    from pyfftw.interfaces import numpy_fft as npi_fft
    monkeypatch.setattr(pyfar.dsp.fft, 'fft_lib', npi_fft)
    return npi_fft.__name__


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
def generate_sofa_postype_spherical(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralFIR,
    spherical position type.
    """
    sofatype = 'GeneralFIR'
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    n_samples = noise_two_by_three_channel.n_samples
    dimensions = {"M": n_measurements, "R": n_receivers, "N": n_samples}

    filename = os.path.join(tmpdir, (sofatype + '.sofa'))
    sofafile = sofa.Database.create(filename, sofatype, dimensions=dimensions)

    sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
    sofafile.Source.initialize(
        variances=["Position"], fixed=["View", "Up"])
    sofafile.Source.Position.set_system('spherical')
    sofafile.Source.Position.set_values(sofa_reference_coordinates[0])
    sofafile.Receiver.initialize(
        variances=["Position"], fixed=["View", "Up"])
    sofafile.Receiver.Position.set_system('spherical')
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
    """ Used for the creation of Orientation objects with
    `Orientations.from_view_up`
    """
    return [[1, 0, 0], [2, 0, 0], [-1, 0, 0]]


@pytest.fixture
def ups():
    """ Used for the creation of Orientation objects with
    `Orientations.from_view_up`
    """
    return [[0, 1, 0], [0, -2, 0], [0, 1, 0]]


@pytest.fixture
def positions():
    """ Used for the visualization of Orientation objects with
    `Orientations.show`
    """
    return [[0, 0.5, 0], [0, -0.5, 0], [1, 1, 1]]


@pytest.fixture
def orientations(views, ups):
    """ Orientations object uses fixtures `views` and `ups`.
    """
    return Orientations.from_view_up(views, ups)


@pytest.fixture
def coordinates():
    """ Coordinates object.
    """
    return Coordinates([0, 1], [2, 3], [4, 5])


@pytest.fixture
def coeffs():
    return np.array([[[1, 0, 0], [1, 0, 0]]])


@pytest.fixture
def state():
    return np.array([[[1, 0]]])


@pytest.fixture
def filter(coeffs, state):
    """ Filter object.
    """
    return fo.Filter(coefficients=coeffs, state=state)


@pytest.fixture
def filterFIR():
    """ FilterFIR objectr.
    """
    coeff = np.array([
        [1, 1 / 2, 0],
        [1, 1 / 4, 1 / 8]])
    return fo.FilterFIR(coeff, sampling_rate=2*np.pi)


@pytest.fixture
def filterIIR():
    """ FilterIIR object.
    """
    coeff = np.array([[1, 1 / 2, 0], [1, 0, 0]])
    return fo.FilterIIR(coeff, sampling_rate=2 * np.pi)


@pytest.fixture
def filterSOS():
    """ FilterSOS objectr.
    """
    sos = np.array([[1, 1 / 2, 0, 1, 0, 0]])
    return fo.FilterSOS(sos, sampling_rate=2 * np.pi)


@pytest.fixture
def sphericalvoronoi():
    """ SphericalVoronoi object.
    """
    points = np.array(
        [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 0]])
    sampling = Coordinates(points[:, 0], points[:, 1], points[:, 2])
    return SphericalVoronoi(sampling)


@pytest.fixture
def any_obj():
    """ Any object acting as placeholder for non-PyFar-objects.
    """
    return stub_utils.AnyClass()


@pytest.fixture
def no_encode_obj():
    """ Any object acting as placeholder for non-PyFar-objects.
    """
    return stub_utils.NoEncodeClass()


@pytest.fixture
def no_decode_obj():
    """ Any object acting as placeholder for non-PyFar-objects.
    """
    return stub_utils.NoDecodeClass()


@pytest.fixture
def flat_data():
    """ Class being primarily used as a subclass of the nested data object.
    """
    return stub_utils.FlatData()


@pytest.fixture
def nested_data():
    """ General nested data structure primarily used to illustrate mechanism of
    `io.write` and `io.read`.
    """
    return stub_utils.NestedData.create()
