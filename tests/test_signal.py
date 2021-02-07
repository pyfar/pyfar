import numpy as np
import numpy.testing as npt
import pytest

from pyfar import Signal


def test_signal_init(sine):
    """Test to init Signal without optional parameters."""
    signal = Signal(sine.time, sine.sampling_rate, domain='time')
    assert isinstance(signal, Signal)


def test_signal_init_list(sine):
    signal = Signal(sine.time.tolist(), sine.sampling_rate, domain='time')
    assert isinstance(signal, Signal)


def test_signal_init_default_parameter(sine):
    # using all defaults
    signal = Signal(sine.time, sine.sampling_rate)
    assert signal.domain == 'time'
    assert signal.fft_norm == 'none'
    assert signal.comment == 'none'
    assert signal.fft_norm == 'none'


def test_signal_comment():
    signal = Signal([1, 0, 0], 44100, comment='Bla')
    assert signal.comment == 'Bla'

    signal.comment = 'Blub'
    assert signal.comment == 'Blub'


def test_domain_getter_freq():
    signal = Signal(np.array([1]), 44100)
    signal._domain = 'freq'
    assert signal.domain == 'freq'


def test_domain_getter_time():
    signal = Signal(np.array([1]), 44100)
    signal._domain = 'time'
    assert signal.domain == 'time'


def test_domain_setter_error():
    signal = Signal(np.array([1]), 44100)
    with pytest.raises(ValueError, match='Incorrect domain'):
        signal.domain = 'quark'


def test_domain_setter_freq_when_freq():
    signal = Signal(np.array([1]), 44100)
    domain = 'freq'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_freq_when_time(sine):
    signal = Signal(
        sine.time, sine.sampling_rate, domain='time',
        fft_norm=sine.fft_norm)
    domain = 'freq'
    signal.domain = domain
    assert signal.domain == domain
    npt.assert_allclose(
        signal._data, sine.freq, rtol=1e-10, atol=1e-10)


def test_domain_setter_time_when_time():
    signal = Signal(np.array([1]), 44100)
    domain = 'time'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_time_when_freq(sine):
    signal = Signal(
        sine.freq, sine.sampling_rate, domain='freq',
        fft_norm=sine.fft_norm)
    domain = 'time'
    signal.domain = domain
    assert signal.domain == domain
    npt.assert_allclose(
        signal._data, sine.time, atol=1e-10, rtol=1e-10)


def test_signal_init_val(sine):
    """Test to init Signal with complete parameters."""
    signal = Signal(
        sine.time, sine.sampling_rate, domain='time',
        fft_norm=sine.fft_norm)
    assert isinstance(signal, Signal)


def test_n_samples(sine):
    """Test for number of samples."""
    signal = Signal(sine.time, sine.sampling_rate, domain='time')
    assert signal.n_samples == sine.n_samples


def test_n_bins(sine):
    """Test for number of freq bins."""
    signal = Signal(sine.time, sine.sampling_rate, domain='time')
    assert signal.n_bins == sine.n_bins


def test_times(sine):
    """Test for the time instances."""
    signal = Signal(sine.time, sine.sampling_rate, domain='time')
    npt.assert_allclose(signal.times, sine.times)


def test_getter_time(sine, impulse):
    """Test if attribute time is accessed correctly."""
    signal = Signal(sine.time, sine.sampling_rate)
    signal._domain = 'time'
    signal._data = impulse.time
    npt.assert_allclose(signal.time, impulse.time)


def test_setter_time(sine, impulse):
    """Test if attribute time is set correctly."""
    signal = Signal(sine.time, sine.sampling_rate)
    signal.time = impulse.time
    assert signal._domain == 'time'
    npt.assert_allclose(signal._data, impulse.time)


def test_getter_freq(sine, impulse):
    """Test if attribute freq is accessed correctly."""
    signal = Signal(sine.time, sine.sampling_rate, fft_norm='rms')
    signal._domain = 'freq'
    signal._data = impulse.freq
    npt.assert_allclose(signal.freq, impulse.freq)


def test_setter_freq(sine, impulse):
    """Test if attribute freq is set correctly."""
    signal = Signal(sine.time, sine.sampling_rate)
    signal.freq = impulse.freq
    assert signal.domain == 'freq'
    npt.assert_allclose(signal._data, impulse.freq)


def test_re_setter_freq():
    """Test the warning for estimating the number of samples from n_bins."""
    signal = Signal([1, 2, 3], 44100, domain='freq', n_samples=4)
    with pytest.warns(UserWarning):
        signal.freq = [1, 2, 3, 4]


def test_getter_sampling_rate(sine):
    """Test if attribute sampling rate is accessed correctly."""
    signal = Signal(sine, sine.sampling_rate)
    signal._sampling_rate = 1000
    assert signal.sampling_rate == 1000


def test_setter_sampligrate(sine):
    """Test if attribute sampling rate is set correctly."""
    signal = Signal(sine.time, sine.sampling_rate)
    signal.sampling_rate = 1000
    assert signal._sampling_rate == 1000


def test_getter_signal_type(sine, sine_rms):
    """Test if attribute signal type is accessed correctly."""
    signal = Signal(sine.time, sine.sampling_rate, fft_norm=sine.fft_norm)
    npt.assert_string_equal(signal.signal_type, 'energy')

    signal = Signal(
        sine_rms.time, sine_rms.sampling_rate, fft_norm=sine_rms.fft_norm)
    npt.assert_string_equal(signal.signal_type, 'power')


def test_setter_signal_type(sine):
    """Test if attribute signal type is set correctly."""
    signal = Signal(sine.time, sine.sampling_rate)
    with pytest.raises(DeprecationWarning):
        signal.signal_type = 'energy'


def test_getter_fft_norm(sine):
    signal = Signal(sine.time, sine.sampling_rate, fft_norm='psd')
    assert signal.fft_norm == 'psd'


def test_setter_fft_norm():
    spec_power_unitary = np.atleast_2d([1, 2, 1])
    spec_power_amplitude = np.atleast_2d([1/4, 2/4, 1/4])

    signal = Signal(
        spec_power_unitary, 44100, n_samples=4, domain='freq',
        fft_norm='unitary')

    # changing the fft_norm also changes the spectrum
    signal.fft_norm = 'amplitude'
    assert signal.fft_norm == 'amplitude'
    npt.assert_allclose(signal.freq, spec_power_amplitude, atol=1e-15)

    # changing the fft norm in the time domain does not change the time data
    signal.domain = 'time'
    time_power_amplitude = signal._data.copy()
    signal.fft_norm = 'unitary'
    npt.assert_allclose(signal.time, time_power_amplitude)
    npt.assert_allclose(signal.freq, spec_power_unitary)

    # setting an invalid fft_norm
    with pytest.raises(ValueError):
        signal.fft_norm = 'bullshit'


def test_dtype(sine):
    """Test for the getter od dtype."""
    dtype = np.float64
    signal = Signal(sine.time, sine.sampling_rate, dtype=dtype)
    assert signal.dtype == dtype


def test_signal_length(sine):
    """Test for the signal length."""
    signal = Signal(sine.time, sine.sampling_rate)
    assert signal.signal_length == sine.times[-1]


def test_cshape(sine_two_by_two_channel):
    """Test the attribute cshape."""
    signal = Signal(
        sine_two_by_two_channel.time, sine_two_by_two_channel.sampling_rate)
    assert signal.cshape == sine_two_by_two_channel.cshape


def test_magic_getitem(sine_two_by_two_channel):
    """Test slicing operations by the magic function __getitem__."""
    signal = Signal(
        sine_two_by_two_channel.time, sine_two_by_two_channel.sampling_rate,
        domain='time')
    npt.assert_allclose(signal[0]._data, sine_two_by_two_channel.time[0])


def test_magic_getitem_slice(sine_two_by_two_channel):
    """Test slicing operations by the magic function __getitem__."""
    signal = Signal(
        sine_two_by_two_channel.time, sine_two_by_two_channel.sampling_rate,
        domain='time')
    npt.assert_allclose(signal[:1]._data, sine_two_by_two_channel.time[:1])


def test_magic_getitem_allslice(sine_two_by_two_channel):
    """Test slicing operations by the magic function __getitem__."""
    signal = Signal(
        sine_two_by_two_channel.time, sine_two_by_two_channel.sampling_rate,
        domain='time')
    npt.assert_allclose(signal[:]._data, sine_two_by_two_channel.time[:])


def test_magic_setitem(sine, impulse):
    """Test the magic function __setitem__."""
    signal = Signal(sine.time, sine.sampling_rate)
    set_signal = Signal(impulse.time, impulse.sampling_rate)
    signal[0] = set_signal
    npt.assert_allclose(signal._data, set_signal._data)


def test_magic_setitem_wrong_sr(sine):
    """Test the magic function __setitem__."""
    signal = Signal(sine.time, sine.sampling_rate)
    set_signal = Signal(sine.time, 48000)
    with pytest.raises(ValueError, match='sampling rates do not match'):
        signal[0] = set_signal


def test_magic_setitem_wrong_norm(sine, sine_rms):
    """Test the magic function __setitem__."""
    signal = Signal(sine.time, sine.sampling_rate, fft_norm=sine.fft_norm)
    set_signal = Signal(
        sine_rms.time, sine_rms.sampling_rate, fft_norm=sine_rms.fft_norm)
    with pytest.raises(ValueError, match='FFT norms do not match'):
        signal[0] = set_signal


def test_magic_setitem_wrong_n_samples(sine):
    """Test the magic function __setitem__."""
    signal = Signal(sine.time, sine.sampling_rate)
    set_signal = Signal(sine.time[..., :-10], sine.sampling_rate)
    with pytest.raises(ValueError, match='number of samples does not match'):
        signal[0] = set_signal


def test_magic_len(sine):
    """Test the magic function __len__."""
    signal = Signal(sine.time, sine.sampling_rate)
    assert len(signal) == sine.n_samples


def test_find_nearest_time():
    sampling_rate = 100
    signal = Signal(np.zeros(100), sampling_rate)
    actual = signal.find_nearest_time(0.5)
    expected = 50
    assert actual == expected

    actual = signal.find_nearest_time([0.5, 0.75])
    expected = [50, 75]
    npt.assert_allclose(actual, expected)


def test_find_nearest_frequency():
    sampling_rate = 100
    signal = Signal(np.zeros(100*2), sampling_rate*2)
    actual = signal.find_nearest_frequency(50)
    expected = 50
    assert actual == expected

    actual = signal.find_nearest_frequency([50, 75])
    expected = [50, 75]
    npt.assert_allclose(actual, expected)


def test_reshape():

    # test reshape with tuple
    signal_in = Signal(np.random.rand(6, 256), 44100)
    signal_out = signal_in.reshape((3, 2))
    npt.assert_allclose(signal_in._data.reshape(3, 2, -1), signal_out._data)
    assert id(signal_in) != id(signal_out)

    signal_out = signal_in.reshape((3, -1))
    npt.assert_allclose(signal_in._data.reshape(3, 2, -1), signal_out._data)
    assert id(signal_in) != id(signal_out)

    # test reshape with int
    signal_in = Signal(np.random.rand(3, 2, 256), 44100)
    signal_out = signal_in.reshape(6)
    npt.assert_allclose(signal_in._data.reshape(6, -1), signal_out._data)
    assert id(signal_in) != id(signal_out)


def test_reshape_exceptions():
    signal_in = Signal(np.random.rand(6, 256), 44100)
    signal_out = signal_in.reshape((3, 2))
    npt.assert_allclose(signal_in._data.reshape(3, 2, -1), signal_out._data)
    # test assertion for non-tuple input
    with pytest.raises(ValueError):
        signal_out = signal_in.reshape([3, 2])

    # test assertion for wrong dimension
    with pytest.raises(ValueError, match='Can not reshape signal of cshape'):
        signal_out = signal_in.reshape((3, 4))


def test_flatten():

    # test 2D signal (flatten should not change anything)
    x = np.random.rand(2, 256)
    signal_in = Signal(x, 44100)
    signal_out = signal_in.flatten()

    npt.assert_allclose(signal_in._data, signal_out._data)
    assert id(signal_in) != id(signal_out)

    # test 3D signal
    x = np.random.rand(3, 2, 256)
    signal_in = Signal(x, 44100)
    signal_out = signal_in.flatten()

    npt.assert_allclose(signal_in._data.reshape((6, -1)), signal_out._data)
    assert id(signal_in) != id(signal_out)


def test___eq___equal(signal):
    sine = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 44100))
    actual = Signal(sine, 44100, len(sine), domain='time')
    assert signal == actual


def test___eq___notEqual(signal):
    sine = np.sin(2 * np.pi * 220 * np.arange(0, 1, 1 / 44100))
    actual = Signal(sine, 44100, len(sine), domain='time')
    assert not signal == actual
