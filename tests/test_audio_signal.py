import numpy as np
import numpy.testing as npt
import pytest

from pyfar import Signal
import pyfar as pf


def test_signal_init():
    """Test to init Signal without optional parameters."""
    signal = Signal(np.array([1., 2., 3.]), 44100)
    assert isinstance(signal, Signal)


def test_signal_init_list():
    signal = Signal([1, 2, 3], 44100)
    assert isinstance(signal, Signal)


def test_signal_init_default_parameter():
    # using all defaults
    signal = Signal([1, 2, 3], 44100)
    assert signal.domain == 'time'
    assert signal.fft_norm == 'none'
    assert signal.comment == 'none'
    assert signal.fft_norm == 'none'


def test_signal_init_assertions():
    """Test assertions in initialization"""

    with pytest.raises(ValueError, match="Invalid FFT normalization"):
        Signal(1, 44100, fft_norm="funky")

    with pytest.raises(ValueError, match="n_samples can not be larger"):
        Signal(1, 44100, domain="freq", n_samples=10)

    with pytest.raises(ValueError, match="Invalid domain"):
        Signal(1, 44100, domain="space")


def test_signal_init_time_dtype():
    """
    Test casting and assertions of dtype (also test time setter because
    it is called during initialization)
    """
    # integer to float casting
    signal = Signal([1, 2, 3], 44100)
    assert signal.time.dtype.kind == "f"

    # float
    signal = Signal([1., 2., 3.], 44100)
    assert signal.time.dtype.kind == "f"

    # complex
    with pytest.raises(ValueError, match="time data is complex"):
        Signal([1+1j, 2+2j, 3+3j], 44100)


def test_data_frequency_init_dtype():
    """
    Test casting and assertions of dtype (also test freq setter because
    it is called during initialization)
    """

    # integer to float casting
    signal = Signal([1, 2, 3], 44100, 4, "freq")
    assert signal.freq.dtype.kind == "c"

    # float
    signal = Signal([1., 2., 3.], 44100, 4, "freq")
    assert signal.freq.dtype.kind == "c"

    # complex
    signal = Signal([1+1j, 2+2j, 3+3j], 44100, 4, "freq")
    assert signal.freq.dtype.kind == "c"

    # object array
    with pytest.raises(ValueError, match="frequency data is"):
        Signal(["1", "2", "3"], 44100, 4, "freq")


def test_signal_comment():
    signal = Signal([1, 2, 3], 44100, comment='Bla')
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


def test_domain_setter_freq_when_time():
    signal = Signal([1, 2, 3, 4], 44100, domain='time', fft_norm='rms')
    domain = 'freq'
    signal.domain = domain
    assert signal.domain == domain
    assert signal._data.shape == signal.cshape + (signal.n_bins,)


def test_domain_setter_time_when_time():
    signal = Signal(np.array([1]), 44100)
    domain = 'time'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_time_when_freq():
    signal = Signal([1, 2, 3, 4], 44100, domain='freq', fft_norm='rms')
    domain = 'time'
    signal.domain = domain
    assert signal.domain == domain
    assert signal._data.shape == signal.cshape + (signal.n_samples,)


def test_signal_init_val():
    """Test to init Signal with complete parameters."""
    signal = Signal([1, 2, 3], 44100, domain='time', fft_norm='none')
    assert isinstance(signal, Signal)


def test_signal_init_freq():
    """Test to init Signal with spectrum."""
    signal = Signal(
        [1, 2, 3], 44100, n_samples=4, domain='freq', fft_norm='amplitude')
    assert isinstance(signal, Signal)
    npt.assert_allclose(signal.freq, np.array([[1., 2., 3.]]), atol=1e-15)
    desired = np.array([[1., 2./2, 3.]]) * 4
    npt.assert_allclose(signal._data, desired, atol=1e-15)


def test_n_samples():
    """Test for number of samples."""
    signal = Signal([1, 2, 3], 44100, domain='time')
    assert signal.n_samples == 3


def test_n_bins():
    """Test for number of freq bins."""
    signal = Signal([1, 2, 3], 44100, domain='time')
    assert signal.n_bins == 2
    signal = Signal([1, 2, 3, 4], 44100, domain='time')
    assert signal.n_bins == 3


def test_times():
    """Test for the time instances."""
    signal = Signal([1, 2, 3, 4], 2, domain='time')
    npt.assert_allclose(signal.times, [0., 0.5, 1., 1.5])


def test_getter_time():
    """Test if attribute time is accessed correctly."""
    signal = Signal([1, 2, 3], 44100, domain='time')
    signal._domain = 'time'
    signal._data = np.array([[1., 2., 3.]])
    npt.assert_allclose(signal.time, np.array([[1., 2., 3.]]))


def test_setter_time():
    """Test if attribute time is set correctly."""
    signal = Signal([1, 2, 3], 44100, domain='time')
    signal.time = np.array([1., 2., 3.])
    assert signal._domain == 'time'
    npt.assert_allclose(signal._data, np.array([[1., 2., 3.]]))


def test_getter_freq():
    """Test if attribute freq is accessed correctly."""
    signal = Signal([1, 2, 3, 4], 44100, fft_norm='amplitude')
    signal._domain = 'freq'
    signal._data = np.array([[1., 2., 3.]])
    desired = np.array([[1., 2*2, 3.]]) / signal.n_samples
    npt.assert_allclose(signal.freq, desired)


def test_setter_freq():
    """Test if attribute freq is set correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm='amplitude')
    signal.freq = np.array([[1., 2., 3.]])
    assert signal.domain == 'freq'
    desired = signal.n_samples * np.array([[1., 2./2, 3.]])
    npt.assert_allclose(signal._data, desired)
    npt.assert_allclose(signal.freq, np.array([[1., 2., 3.]]))


def test_re_setter_freq():
    """Test the warning for estimating the number of samples from n_bins."""
    signal = Signal([1, 2, 3], 44100, domain='freq', n_samples=4)
    with pytest.warns(UserWarning):
        signal.freq = [1, 2, 3, 4]


def test_getter_sampling_rate():
    """Test if attribute sampling rate is accessed correctly."""
    signal = Signal([1, 2, 3], 44100)
    signal._sampling_rate = 1000
    assert signal.sampling_rate == 1000


def test_setter_sampligrate():
    """Test if attribute sampling rate is set correctly."""
    signal = Signal([1, 2, 3], 44100)
    signal.sampling_rate = 1000
    assert signal._sampling_rate == 1000


def test_getter_signal_type():
    """Test if attribute signal type is accessed correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm='none')
    npt.assert_string_equal(signal.signal_type, 'energy')

    signal = Signal([1, 2, 3], 44100, fft_norm='rms')
    npt.assert_string_equal(signal.signal_type, 'power')


def test_getter_fft_norm():
    signal = Signal([1, 2, 3], 44100, fft_norm='psd')
    assert signal.fft_norm == 'psd'


def test_setter_fft_norm():
    spec_power_unitary = np.atleast_2d([1, 2, 1])
    spec_power_amplitude = np.atleast_2d([1/4, 2/4, 1/4])

    signal = Signal(
        spec_power_unitary, 44100, n_samples=4, domain='freq',
        fft_norm='unitary')

    # changing the fft_norm also changes the spectrum
    npt.assert_allclose(signal.freq, spec_power_unitary, atol=1e-15)
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


def test_dtype():
    """Test for converting int to float upon init."""

    signal = Signal([1, 2, 3], 44100)
    assert signal._data.dtype.kind == "f"


def test_signal_length():
    """Test for the signal length."""
    signal = Signal([1, 2, 3, 4], 2)
    assert signal.signal_length == 1.5


def test_cshape():
    """Test the attribute cshape."""
    time = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    signal = Signal(time, 44100)
    assert signal.cshape == (2, 3)


def test_magic_getitem():
    """Test slicing operations by the magic function __getitem__."""
    time = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    signal = Signal(time, 44100, domain='time')
    npt.assert_allclose(signal[0]._data, time[0])


def test_magic_getitem_slice():
    """Test slicing operations by the magic function __getitem__."""
    time = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    signal = Signal(time, 44100, domain='time')
    npt.assert_allclose(signal[:1]._data, time[:1])


def test_magic_getitem_allslice():
    """Test slicing operations by the magic function __getitem__."""
    time = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    signal = Signal(time, 44100, domain='time')
    npt.assert_allclose(signal[:]._data, time[:])


def test_magic_setitem():
    """Test the magic function __setitem__."""
    signal = Signal([1, 2, 3], 44100)
    set_signal = Signal([2, 3, 4], 44100)
    signal[0] = set_signal
    npt.assert_allclose(signal._data, set_signal._data)


def test_magic_setitem_wrong_sr():
    """Test the magic function __setitem__."""
    signal = Signal([1, 2, 3], 44100)
    set_signal = Signal([1, 2, 3], 48000)
    with pytest.raises(ValueError, match='sampling rates do not match'):
        signal[0] = set_signal


def test_magic_setitem_wrong_norm():
    """Test the magic function __setitem__."""
    signal = Signal([1, 2, 3], 44100, fft_norm='none')
    set_signal = Signal([1, 2, 3], 44100, fft_norm='rms')
    with pytest.raises(ValueError, match='FFT norms do not match'):
        signal[0] = set_signal


def test_magic_setitem_wrong_n_samples():
    """Test the magic function __setitem__."""
    signal = Signal([1, 2, 3, 4], 44100)
    set_signal = Signal([1, 2, 3], 44100)
    with pytest.raises(ValueError, match='number of samples does not match'):
        signal[0] = set_signal


@pytest.mark.parametrize("audio", (
    pf.TimeData([1, 2], [1, 2]), pf.FrequencyData([1, 2], [1, 2])))
def test_magic_setitem_wrong_type(audio):
    signal = Signal([1, 2, 3, 4], 44100)
    with pytest.raises(ValueError, match="Comparison only valid"):
        signal[0] = audio


def test_magic_len():
    """Test the magic function __len__."""
    signal = Signal([1, 2, 3], 44100)
    assert len(signal) == 3


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
    with pytest.raises(ValueError, match='Can not reshape audio object'):
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


def test___eq___equal():
    signal = Signal([1, 2, 3], 44100)
    actual = Signal([1, 2, 3], 44100)
    assert signal == actual


def test___eq___notEqual():
    time = np.arange(2*3*4).reshape((2, 3, 4))
    signal = Signal(time, 44100, domain='time')

    actual = Signal(0.5 * time, 44100, domain='time')
    assert not signal == actual
    actual = Signal(time, 2 * 44100, domain='time')
    assert not signal == actual
    comment = f'{signal.comment} A completely different thing'
    actual = Signal(time, 44100, domain='time', comment=comment)
    assert not signal == actual


def test__repr__(capfd):
    """Test string representation"""
    print(Signal([0, 1, 0], 44100))
    out, _ = capfd.readouterr()
    assert ("time domain energy Signal:\n"
            "(1,) channels with 3 samples @ 44100 Hz sampling rate "
            "and none FFT normalization") in out


def test_freq_raw():
    """Test to access unnormalized spectrum."""
    signal = Signal([1, 0, 0, 0], 44100, domain='time')
    npt.assert_allclose(signal.freq_raw, np.array([[1., 1., 1.]]))
    signal.fft_norm = 'amplitude'
    npt.assert_allclose(signal.freq, np.array([[1., 2., 1.]])/4)
    npt.assert_allclose(signal.freq_raw, np.array([[1., 1., 1.]]))


def test_setter_freq_raw():
    """Test if attribute freq_raw is set correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm='amplitude')
    signal.freq_raw = np.array([[1., 2., 3.]])
    assert signal.domain == 'freq'
    npt.assert_allclose(signal._data, np.array([[1., 2., 3.]]))


def test_setter_freq_raw_warning():
    """Test the warning for estimating the number of samples from n_bins."""
    signal = Signal([1, 2, 3], 44100, domain='freq', n_samples=4)
    with pytest.warns(UserWarning, match="Number of frequency bins changed"):
        signal.freq_raw = [1, 2, 3, 4]


def test_setter_freq_raw_dtype():
    """
    Test casting and assertions of dtype (not tested during initialization
    because that calls the `freq` setter)
    """
    signal = Signal([0, 1, 2], 44100, 4, "freq")

    # integer to float casting
    signal.freq_raw = [1, 2, 3]
    assert signal.freq_raw.dtype.kind == "c"
    npt.assert_allclose(signal.freq_raw, np.array([[1., 2., 3.]]))

    # float
    signal.freq_raw = [1., 2., 3.]
    assert signal.freq_raw.dtype.kind == "c"

    # complex
    signal.freq_raw = [1+1j, 2+2j, 3+3j]
    assert signal.freq_raw.dtype.kind == "c"

    # object array
    with pytest.raises(ValueError, match="frequency data is"):
        signal.freq_raw = ["1", "2", "3"]
