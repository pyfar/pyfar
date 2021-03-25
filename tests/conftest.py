import pytest
import numpy as np
import os.path
import sofa
import scipy.io.wavfile as wavfile

from pyfar.spatial.spatial import SphericalVoronoi
from pyfar.orientations import Orientations
from pyfar.coordinates import Coordinates
from pyfar.signal import FrequencyData, Signal, TimeData
import pyfar.dsp.classes as fo

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
    fft_norm = 'none'
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
        Sine signal
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
def sine_stub_rms():
    """Sine signal stub, RMS FFT-normalization.
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
def sine_stub_odd_rms():
    """Sine signal stub, odd number of samples, RMS FFT-normalization.
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
def impulse_stub_rms():
    """Delta impulse signal stub, RMS FFT-normalization.
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
    fft_norm = 'rms'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
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
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

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
    sampling_rate = 44100
    n_samples = 100
    fft_norm = 'none'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def sine_rms():
    """Sine signal, RMS FFT-normalization.

    Returns
    -------
    signal : Signal
        Sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def impulse():
    """Delta impulse signal.

    Returns
    -------
    signal : Signal
        Impulse signal
    """
    delay = 0
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def impulse_rms():
    """Delta impulse signal, RMS FFT-normalization.

    Returns
    -------
    signal : Signal
        Impulse signal
    """
    delay = 0
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'rms'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

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
    delay = 1000
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)
    group_delay = delay * np.ones_like(freq, dtype=float)

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
    delay = np.atleast_1d([1000, 2000])
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2,)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)
    group_delay = delay[..., np.newaxis] * np.ones_like(freq, dtype=float)

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
    delay = np.array([[1000, 2000], [3000, 4000]])
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (2, 2)

    time, freq = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)
    group_delay = delay[..., np.newaxis] * np.ones_like(freq, dtype=float)

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
    sampling_rate = 44100
    n_samples = 10000
    fft_norm = 'none'
    cshape = (1,)

    time_sine, freq_sine, frequency = stub_utils.sine_func(
        frequency, sampling_rate, n_samples, fft_norm, cshape)
    time_imp, freq_imp = stub_utils.impulse_func(
        delay, n_samples, fft_norm, cshape)
    signal = Signal(
        time_sine + time_imp, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def noise():
    """Gaussian white noise signal.

    Returns
    -------
    signal : Signal
        Noise signal
    """
    sigma = 1
    n_samples = int(1e5)
    cshape = (1,)
    sampling_rate = 44100
    fft_norm = 'rms'

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def noise_odd():
    """Gaussian white noise signal, odd number of samples.

    Returns
    -------
    signal : Signal
        Noise signal
    """
    sigma = 1
    n_samples = int(1e5 - 1)
    cshape = (1,)
    sampling_rate = 44100
    fft_norm = 'rms'

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def noise_two_by_three_channel():
    """ 2-by-3 channel gaussian white noise signal.

    Returns
    -------
    signal : Signal
        Noise signal
    """
    sigma = 1
    n_samples = int(1e5)
    cshape = (2, 3)
    sampling_rate = 44100
    fft_norm = 'rms'

    time = stub_utils.noise_func(sigma, n_samples, cshape)
    signal = Signal(time, sampling_rate, fft_norm=fft_norm)

    return signal


@pytest.fixture
def time_data_three_points():
    """
    TimeData stub with three data points.

    Returns
    -------
    time_data
        stub of pyfar TimeData class
    """
    time_data = stub_utils.time_data_stub([1, 0, -1], [0, .1, .4])
    return time_data


@pytest.fixture
def frequency_data_three_points():
    """
    FrequencyData stub with three data points.

    Returns
    -------
    frequency_data
        stub of pyfar FrequencyData class
    """
    frequency_data = stub_utils.frequency_data_stub(
        [2, .25, .5], [100, 1_000, 20_000])
    return frequency_data


@pytest.fixture
def frequency_data_one_point():
    """
    FrequencyData stub with one data points.

    Returns
    -------
    frequency_data
        stub of pyfar FrequencyData class
    """
    frequency_data = stub_utils.frequency_data_stub([2], [0])
    return frequency_data


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
def timedata():
    data = [1, 0, -1]
    times = [0, .1, .3]
    return TimeData(data, times)


@pytest.fixture
def frequencydata():
    data = [1, 0, -1]
    freqs = [0, .1, .3]
    return FrequencyData(data, freqs)


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
def filterIIR():
    """ FilterIIR object.
    """
    coeff = np.array([[1, 1 / 2, 0], [1, 0, 0]])
    return fo.FilterIIR(coeff, sampling_rate=2 * np.pi)


@pytest.fixture
def filterFIR():
    """ FilterFIR objectr.
    """
    coeff = np.array([
        [1, 1 / 2, 0],
        [1, 1 / 4, 1 / 8]])
    return fo.FilterFIR(coeff, sampling_rate=2*np.pi)


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
