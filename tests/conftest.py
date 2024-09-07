import pytest
import numpy as np
import os.path
import sofar as sf
import pyfar as pf

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
    # Amplitude Normalization
    signal.time = signal.time / np.abs(signal.time.max())

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

    # Amplitude Normalization
    signal.time = signal.time / np.abs(signal.time.max())

    return signal


@pytest.fixture
def handsome_signal():
    """
    Windows 200 Hz sine signal for testing plots

    Returns
    -------
    signal : Signal
        Windowed sine
    """

    signal = pf.signals.sine(200, 4410)
    signal = pf.dsp.time_window(signal, (1500, 2000, 3000, 3500))
    signal.fft_norm = 'none'
    return signal


@pytest.fixture
def handsome_signal_v2():
    """
    Windowed 1kHz sine signal for testing plots

    Returns
    -------
    signal : Signal
        Windowed sine
    """

    signal = pf.signals.sine(2000, 4410)
    signal = pf.dsp.time_window(signal, (500, 1000, 2000, 2500))
    signal.fft_norm = 'none'
    return signal


@pytest.fixture
def handsome_signal_2d():
    """
    45 channel signal with delayed, scaled and bell-filtered impulses
    for testing 2D plots

    Returns
    -------
    signal : Signal
        Multi channel signal
    """

    delays = np.array(np.sin(np.linspace(0, 2*np.pi, 45))*50 + 55, dtype=int)
    amplitudes = 10**(-10*(1-np.cos(np.linspace(0, 2*np.pi, 45)))/20)
    signal = pyfar.signals.impulse(2**9, delays, amplitudes)
    for idx, s in enumerate(signal):
        signal[idx] = pf.dsp.filter.bell(s, (idx+1)*200, -20, 5)

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
def sofa_reference_coordinates(noise_two_by_three_channel):
    """Define coordinates to write in reference files.
    """
    n_measurements = noise_two_by_three_channel.cshape[0]
    n_receivers = noise_two_by_three_channel.cshape[1]
    source_coordinates = np.random.rand(n_measurements, 3)
    receiver_coordinates = np.random.rand(n_receivers, 3)
    return source_coordinates, receiver_coordinates


@pytest.fixture
def generate_sofa_GeneralFIR(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralFIR."""
    filename = os.path.join(tmpdir, ('GeneralFIR' + '.sofa'))

    sofafile = sf.Sofa('GeneralFIR', True)
    sofafile.Data_IR = noise_two_by_three_channel.time
    sofafile.Data_Delay = np.zeros((1, noise_two_by_three_channel.cshape[1]))
    sofafile.Data_SamplingRate = noise_two_by_three_channel.sampling_rate
    sofafile.SourcePosition = sofa_reference_coordinates[0]
    sofafile.SourcePosition_Type = "cartesian"
    sofafile.SourcePosition_Units = "meter"
    sofafile.ReceiverPosition = sofa_reference_coordinates[1]

    sf.write_sofa(filename, sofafile)

    return filename


@pytest.fixture
def generate_sofa_GeneralTF(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralTF."""
    filename = os.path.join(tmpdir, ('GeneralTF' + '.sofa'))

    sofafile = sf.Sofa('GeneralTF', True)
    sofafile.Data_Real = np.real(noise_two_by_three_channel.freq)
    sofafile.Data_Imag = np.imag(noise_two_by_three_channel.freq)
    sofafile.N = noise_two_by_three_channel.frequencies
    sofafile.SourcePosition = sofa_reference_coordinates[0]
    sofafile.ReceiverPosition = sofa_reference_coordinates[1]

    sf.write_sofa(filename, sofafile)

    return filename


@pytest.fixture
def generate_sofa_GeneralFIR_E(tmpdir):
    """ Generate the reference sofa files of type GeneralFIR-E."""
    filename = os.path.join(tmpdir, ('GeneralFIR-E.sofa'))

    sofafile = sf.Sofa('GeneralFIR-E', True)
    sofafile.Data_IR = np.zeros((4, 2, 10, 3))
    sofafile.Data_Delay = np.zeros((4, 2, 3))

    sf.write_sofa(filename, sofafile)

    return filename


@pytest.fixture
def generate_sofa_GeneralTF_E(tmpdir):
    """ Generate the reference sofa files of type GeneralFIR-E."""
    filename = os.path.join(tmpdir, ('GeneralTF-E.sofa'))

    sofafile = sf.Sofa('GeneralTF-E', True)
    sofafile.Data_Real = np.ones((4, 2, 10, 3))
    sofafile.Data_Imag = np.zeros((4, 2, 10, 3))
    sofafile.N = np.arange(10)

    sf.write_sofa(filename, sofafile)

    return filename


@pytest.fixture
def generate_sofa_postype_spherical(
        tmpdir, noise_two_by_three_channel, sofa_reference_coordinates):
    """ Generate the reference sofa files of type GeneralFIR,
    spherical position type.
    """

    filename = os.path.join(tmpdir, ('GeneralFIR' + '.sofa'))

    sofafile = sf.Sofa('GeneralFIR', True)
    sofafile.Data_IR = noise_two_by_three_channel.time
    sofafile.Data_Delay = np.zeros((1, noise_two_by_three_channel.cshape[1]))
    sofafile.Data_SamplingRate = noise_two_by_three_channel.sampling_rate
    sofafile.SourcePosition = sofa_reference_coordinates[0]
    sofafile.ReceiverPosition = sofa_reference_coordinates[1]
    sofafile.ReceiverPosition_Type = "spherical"
    sofafile.ReceiverPosition_Units = "degree, degree, meter"

    sf.write_sofa(filename, sofafile)

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


@pytest.fixture
def dict_of_builtins():
    """ Dictionary that contains builtins with support for writing and reading.
    """
    return stub_utils.dict_of_builtins()
