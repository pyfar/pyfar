import numpy as np
import numpy.testing as npt
import pytest

from pyfar import Signal
import pyfar as pf


@pytest.mark.parametrize("data", [
    np.array([1., 2., 3.]),
    [1, 2, 3],
    (1, 2, 3)])
def test_signal_init(data):
    """Test initializing a Signal without optional parameters."""
    signal = Signal(data, 44100)
    assert isinstance(signal, Signal)
    npt.assert_allclose(signal.time, np.atleast_2d(data))


def test_signal_init_default_parameter():
    """Test initializing a Signal with default parameters."""
    signal = Signal([1, 2, 3], 44100)
    assert signal.domain == 'time'
    assert signal.fft_norm == 'none'
    assert signal.comment == ''
    assert not signal.complex


def test_signal_init_time():
    """Test initializing a Signal with time domain data."""
    signal = Signal([1, 2, 3], 44100, domain='time', fft_norm='none')
    assert isinstance(signal, Signal)


@pytest.mark.parametrize(("data","n_samples", "domain",
                           "fft_norm", "is_complex", "desired"), [
    ([1, 2, 3], 4, "freq", "amplitude", False, np.array([[1., 2./2, 3.]])*4),
    ([1], None, "freq", 'none', False, np.array([[1]]))])
def test_signal_init_freq(data, n_samples, domain, fft_norm, is_complex,
                          desired):
    """Test initializing a Signal with spectrum."""
    if n_samples is None:
        with pytest.warns(UserWarning, match="Number of samples not given"):
            signal = Signal(data, 44100, n_samples=n_samples, domain=domain,
                            fft_norm=fft_norm, is_complex=is_complex)
    else:
        signal = Signal(data, 44100, n_samples=n_samples, domain=domain,
                        fft_norm=fft_norm, is_complex=is_complex)
    npt.assert_allclose(signal._data, desired, atol=1e-15)
    npt.assert_allclose(signal.freq, np.atleast_2d(data), atol=1e-15)


@pytest.mark.parametrize(("data","n_samples", "domain",
                           "fft_norm", "is_complex", "match"), [
    (1,None, "time", "funky", False, "Invalid FFT normalization"),
    (1, 10, "freq", "none", False, "n_samples can not be larger"),
    (1, None, "space", 'none', False, "Invalid domain"),
    (1, 10, "freq", 'none', True, "n_samples can not be larger"),
    ([1+1j, 2+2j, 3+3j], None, "time", 'none', False,
     "time data is complex, set is_complex flag or pass real-valued data.")])
def test_signal_init_assertions(data, n_samples, domain, fft_norm, is_complex,
                                 match):
    """Test assertions in initialization."""
    with pytest.raises(ValueError, match=match):
        Signal(data, 44100, n_samples=n_samples, domain=domain,
               fft_norm=fft_norm, is_complex=is_complex)


@pytest.mark.parametrize(("data", "dtype", "n_samples", "domain",
                          "is_complex"),[
    ([1, 2, 3], "f", None, "time", False),
    ([1, 2, 3],"c", None, "time", True),
    ([1., 2., 3.], "f", None, "time", False),
    ([1., 2., 3.], "c", None, "time", True),
    ([1+1j, 2+2j, 3+3j], "c", None, "time", True),
    ([1, 2, 3], "c", 4, "freq", False),
    ([1., 2., 3.], "c", 4, "freq", False),
    ([1+1j, 2+2j, 3+3j], "c", 4, "freq", False)])
def test_signal_init_dtype(data, dtype, n_samples, domain,
                           is_complex):
    """
    Test casting and assertions of dtype (also test time setter because
    it is called during initialization).
    """
    signal = Signal(data, 44100, n_samples=n_samples,
                     domain=domain, is_complex=is_complex)
    actual = signal.time if domain == "time" else signal.freq
    assert actual.dtype.kind == dtype


@pytest.mark.parametrize(("data", "n_samples", "domain", "error_type",
                           "match"), [
    (['1', '2', '3'], None, "time", TypeError, "int, uint, float, or complex"),
    (np.array([1, 2, np.nan]), None, "time", ValueError,
        "input values must be numeric"),
    (['1', '2', '3'], 4, "freq", TypeError, "int, uint, float, or complex"),
    (np.array([1, 2, np.nan]), 4, "freq",ValueError,
        "input values must be numeric")])
def test_signal_init_dtype_error(data, n_samples, domain, error_type,
                                       match):
    """Test error handling of dtype."""
    with pytest.raises(error_type, match=match):
        Signal(data, 44100, n_samples=n_samples, domain=domain)


def test_signal_comment():
    """Test the comment attribute of Signal."""
    signal = Signal([1, 2, 3], 44100, comment='Bla')
    assert signal.comment == 'Bla'
    signal.comment = 'Blub'
    assert signal.comment == 'Blub'


def test_signal_comment_error():
    """Test error handling of the comment attribute."""
    with pytest.raises(TypeError, match="comment has to be of type string."):
        Signal([1, 2, 3], 44100, comment=[1, 2, 3])


@pytest.mark.parametrize('domain', ['time', 'freq'])
def test_domain_getter_time_freq(domain):
    """Test accessing the domain attribute for both 'time' and 'freq'."""
    signal = Signal(np.array([1]), 44100)
    signal._domain = domain
    assert signal.domain == domain


def test_domain_setter_error():
    """Test error handling of the domain setter."""
    signal = Signal(np.array([1]), 44100)
    with pytest.raises(ValueError, match='Incorrect domain'):
        signal.domain = 'quark'


@pytest.mark.parametrize(('set_domain'), ['time', 'freq'])
def test_domain_setter_same_domain(set_domain):
    """Test setting the domain attribute to the same value."""
    signal = Signal(np.array([1]), 44100)
    signal._domain = set_domain
    signal.domain = set_domain
    assert signal.domain == set_domain


@pytest.mark.parametrize(('domain', 'set_domain'), [('time', 'freq'),
                                                    ('freq', 'time')])
def test_domain_setter_opposite_domain(domain, set_domain):
    """Test setting the domain attribute to the opposite value."""
    if domain == 'freq':
        with pytest.warns(UserWarning, match="Number of samples not given"):
            signal = Signal([1, 2, 3, 4], 44100, domain=domain, fft_norm='rms')
    else:
        signal = Signal([1, 2, 3, 4], 44100, domain=domain, fft_norm='rms')
    signal.domain = set_domain
    actual = signal.n_bins if signal.domain == 'freq' else signal.n_samples
    assert signal.domain == set_domain
    assert signal._data.shape == signal.cshape + (actual,)


@pytest.mark.parametrize("is_complex", [False, True])
def test_n_samples(is_complex):
    """Test correct number of samples is returned for time-domain signals."""
    signal = Signal([1, 2, 3], 44100, domain='time', is_complex=is_complex)
    assert signal.n_samples == 3


@pytest.mark.parametrize(("is_complex", "desired"), [(False, 4), (True, 3)])
def test_n_samples_estimated_from_n_bins(is_complex, desired):
    """
    Test that n_samples is estimated from n_bins when not given during
     freq-domain initialization.
    """
    with pytest.warns(UserWarning, match='Number of samples not given'):
        signal = Signal([1, 2, 3], 44100, domain='freq', is_complex=is_complex)
    assert signal.n_samples == desired


@pytest.mark.parametrize(("data", "is_complex", "desired"), [
    ([1, 2, 3], False, 2),
    ([1, 2, 3, 4], False, 3),
    ([1, 2, 3], True, 3),
    ([1, 2, 3, 4], True, 4)])
def test_n_bins(data, is_complex, desired):
    """Test for number of freq bins."""
    signal = Signal(data, 44100, domain='time', is_complex=is_complex)
    assert signal.n_bins == desired


def test_times():
    """Test for the time instances."""
    signal = Signal([1, 2, 3, 4], 2, domain='time')
    npt.assert_allclose(signal.times, [0., 0.5, 1., 1.5])


@pytest.mark.parametrize(("data", "domain", "fft_norm", "desired"), [
    ([1, 2, 3], 'time', 'none', np.array([[1., 2., 3.]])),
    ([1, 2, 3, 4], 'freq', 'amplitude', np.array([[0.25, 1., 0.75]]))])
def test_getter_time_freq(data, domain, fft_norm, desired):
    """Test if attribute time/freq is accessed correctly."""
    signal = Signal(data, 44100, domain='time', fft_norm=fft_norm)
    signal._domain = domain
    signal._data = np.array([[1., 2., 3.]])
    actual = signal.time if domain == 'time' else signal.freq
    npt.assert_allclose(actual, desired)


@pytest.mark.parametrize(("data", "domain", "fft_norm", "desired"), [
    (np.array([[1., 2., 3.]]), 'time', 'none', np.array([[1., 2., 3.]])),
    (np.array([[1., 2., 3.]]), 'freq', 'amplitude',
    4*np.array([[1., 1., 3.]])),
    (np.array([[1.]]), 'freq', 'amplitude',  1*np.array([[1.]]))])
def test_setter_time_freq(data, domain, fft_norm, desired):
    """Test if attributes time and freq are set correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm=fft_norm)
    if domain == 'freq':
        with pytest.warns(UserWarning, match="Number of samples not given"):
            signal.freq = data
    else:
        signal.time = data
    assert signal.domain == domain
    assert signal._domain == domain
    npt.assert_allclose(signal._data, desired)
    if domain == 'freq':
        npt.assert_allclose(signal.freq, data)


def test_getter_sampling_rate():
    """Test if attribute sampling rate is accessed correctly."""
    signal = Signal([1, 2, 3], 44100)
    signal._sampling_rate = 1000
    assert signal.sampling_rate == 1000


def test_setter_sampling_rate():
    """Test if attribute sampling rate is set correctly."""
    signal = Signal([1, 2, 3], 44100)
    signal.sampling_rate = 1000
    assert signal._sampling_rate == 1000


@pytest.mark.parametrize('fs', [1, [1], [[1]], [[[1]]]])
def test_sampling_rate_parsing(fs):
    Signal([0], fs)
    Signal([0], np.array(fs))


@pytest.mark.parametrize(("sampling_rate", "match"), [
    ([1, 2], "Multirate signals are not supported."),
    ('string', "Sampling rate needs to be a number.")])
def test_sampling_rate_errors(sampling_rate, match):
    with pytest.raises(ValueError, match=match):
        Signal(1, sampling_rate)


@pytest.mark.parametrize(('fft_norm', 'desired'), [
    ('none', 'energy'), ('rms', 'power')])
def test_getter_signal_type(fft_norm, desired):
    """Test if attribute signal type is accessed correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm=fft_norm)
    npt.assert_string_equal(signal.signal_type, desired)


def test_getter_fft_norm():
    """Test if attribute fft_norm is accessed correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm='psd')
    assert signal.fft_norm == 'psd'


@pytest.mark.parametrize(('fft_norm', 'desired'), [
    ('unitary', [1, 2, 1]),
    ('amplitude', [1/4, 2/4, 1/4])])
def test_setter_fft_norm_renormalizes_spectrum(fft_norm, desired):
    """Changing fft_norm re-renders `freq` but leaves the raw data alone."""
    signal = Signal([1, 2, 1], 44100, n_samples=4, domain='freq',
                    fft_norm='unitary')
    signal.fft_norm = fft_norm

    npt.assert_allclose(signal.freq_raw, np.atleast_2d([1., 1., 1.]),
                        atol=1e-15)
    npt.assert_allclose(signal.freq, np.atleast_2d(desired), atol=1e-15)


@pytest.mark.parametrize('fft_norm', ['none', 'unitary', 'amplitude',
                                      'rms', 'power', 'psd'])
def test_setter_fft_norm_keeps_time_data(fft_norm):
    """
    In the time domain, changing fft_norm changes neither data
    nor domain.
    """
    signal = Signal([1, 2, 3, 4], 44100, fft_norm='none')
    time = signal.time.copy()
    signal.fft_norm = fft_norm
    assert signal.domain == 'time'
    npt.assert_allclose(signal.time, time)


@pytest.mark.parametrize(('fft_norm', "match"), [
    ('invalid', 'Invalid FFT normalization. Has to be none, unitary,'
    ' amplitude'),
    ('rms', "'rms', 'power', and 'psd' FFT normalization is not valid for"
    " complex time signals"),
    ('power', "'rms', 'power', and 'psd' FFT normalization is not valid for"
    " complex time signals"),
    ('psd', "'rms', 'power', and 'psd' FFT normalization is not valid for"
    " complex time signals")])
def test_setter_fft_norm_error(fft_norm, match):
    """Test that setting an invalid fft_norm raises a ValueError."""
    signal = Signal([1, 2, 3], 44100, fft_norm='none',  is_complex=True)
    with pytest.raises(ValueError, match=match):
        signal.fft_norm = fft_norm


@pytest.mark.parametrize(("is_complex", "expected"), [(False, 2), (True, 3)])
def test_fft_selection(is_complex, expected):
    """Test if appropriate FFT is computed."""
    signal = Signal([1, 2, 3], 44100, is_complex=is_complex)
    assert signal.freq.shape[1] == expected


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


def test_cdim():
    """Test the attribute cdim."""
    time = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    signal = Signal(time, 44100)
    assert signal.cdim == 2


@pytest.mark.parametrize(("index", "is_complex"), [
    (0, False),
    (0, True),
    (slice(None, 1), False),
    (slice(None), False)])
def test_magic_getitem(index, is_complex):
    """Test slicing operations by the magic function __getitem__."""
    dtype = complex if is_complex else None
    time = np.arange(2 * 3 * 4, dtype=dtype).reshape((2, 3, 4))
    signal = Signal(time, 44100, domain='time', is_complex=is_complex)
    npt.assert_allclose(signal[index]._data, time[index])
    if is_complex:
        assert signal[index].complex


def test_magic_getitem_ellipsis():
    """Test slicing operations by the magic function __getitem__."""
    signal = pf.Signal([[[1, 1, 1], [2, 2, 2]]], 44100)
    npt.assert_allclose(signal[..., 0].time, np.atleast_2d([1, 1, 1]))
    assert signal[..., 0].time.shape == (1, 3)


@pytest.mark.parametrize('domain', ['time', 'freq'])
def test_magic_getitem_error(domain):
    """
    Test if indexing that would return a subset of the samples or frequency
    bins raises a key error.
    """
    signal = pf.Signal([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], 1)
    signal.domain = domain
    # manually indexing too many dimensions
    with pytest.raises(IndexError, match='Indexed dimensions must not exceed'):
        signal[0, 1]
    # indexing too many dimensions with ellipsis operator
    with pytest.raises(IndexError, match='Indexed dimensions must not exceed'):
        signal[0, 0, ..., 1]


def test_magic_setitem():
    """Test the magic function __setitem__."""
    signal = Signal([1, 2, 3], 44100)
    set_signal = Signal([2, 3, 4], 44100)
    signal[0] = set_signal
    npt.assert_allclose(signal._data, set_signal._data)


@pytest.mark.parametrize(("set_signal", "match"), [
    (Signal([1, 2, 3], 48000), 'sampling rates do not match'),
    (Signal([1, 2, 3], 44100, fft_norm='rms'), 'FFT norms do not match'),
    (Signal([1, 2, 3, 4], 44100), 'number of samples does not match')])
def test_magic_setitem_errors(set_signal, match):
    """Test the error handling of the magic function __setitem__."""
    signal = Signal([1, 2, 3], 44100)
    with pytest.raises(ValueError, match=match):
        signal[0] = set_signal


@pytest.mark.parametrize("audio", [
    pf.TimeData([1, 2], [1, 2]), pf.FrequencyData([1, 2], [1, 2])])
def test_magic_setitem_wrong_type(audio):
    """Test the error handling of __setitem__ for wrong input type."""
    signal = Signal([1, 2, 3, 4], 44100)
    with pytest.raises(ValueError, match="Comparison only valid"):
        signal[0] = audio


@pytest.mark.parametrize(("time", "expected"), [
    (0.5, 50),
    ([0.5, 0.75], [50, 75])])
def test_find_nearest_time(time, expected):
    """Test finding the nearest sample index for a given time instance."""
    sampling_rate = 100
    signal = Signal(np.zeros(100), sampling_rate)
    actual = signal.find_nearest_time(time)
    npt.assert_allclose(actual, expected)


@pytest.mark.parametrize(("frequency", "expected"), [
    (50, 50),
    ([50, 75], [50, 75])])
def test_find_nearest_frequency(frequency, expected):
    """Test finding the nearest bin index for a given frequency."""
    sampling_rate = 100
    signal = Signal(np.zeros(100*2), sampling_rate*2)
    actual = signal.find_nearest_frequency(frequency)
    npt.assert_allclose(actual, expected)


@pytest.mark.parametrize(("input_shape", "reshape_arg", "expected_shape"),[
    ((6, 256), (3, 2), (3, 2, -1)),
    ((6, 256), (3, -1), (3, 2, -1)),
    ((3, 2, 256), 6, (6, -1))])
def test_reshape(input_shape, reshape_arg, expected_shape):
    """Test Signal.reshape with tuple and int arguments."""
    rng = np.random.default_rng()
    x = rng.random(input_shape)
    signal_in = Signal(x, 44100)
    signal_out = signal_in.reshape(reshape_arg)
    npt.assert_allclose(
        signal_in._data.reshape(expected_shape), signal_out._data)
    assert id(signal_in) != id(signal_out)


@pytest.mark.parametrize(("new_shape", "match"),[
    ([3, 2], 'newshape must be an integer or tuple'),
    ((3, 4), 'Cannot reshape audio object')])
def test_reshape_exceptions(new_shape, match):
    """Test error handling of Signal.reshape with invalid arguments."""
    rng = np.random.default_rng()
    x = rng.random((6, 256))
    signal_in = Signal(x, 44100)
    with pytest.raises(ValueError, match=match):
        signal_in.reshape(new_shape)


def test_transpose():
    """Test the default behavior of Signal.transpose."""
    rng = np.random.default_rng()
    x = rng.random((6, 2, 5, 256))
    signal_in = Signal(x, 44100)
    signal_out = signal_in.transpose()
    npt.assert_allclose(signal_in.T._data, signal_out._data)
    npt.assert_allclose(
        signal_in._data.transpose(2, 1, 0, 3), signal_out._data)


@pytest.mark.parametrize('taxis', [(2, 0, 1), (-1, 0, -2)])
def test_transpose_args(taxis):
    """Test Signal.transpose with an explicit axis order."""
    rng = np.random.default_rng()
    x = rng.random((6, 2, 5, 256))
    signal_in = Signal(x, 44100)
    signal_out = signal_in.transpose(taxis)
    npt.assert_allclose(
        signal_in._data.transpose(2, 0, 1, 3), signal_out._data)
    signal_out = signal_in.transpose(*taxis)
    npt.assert_allclose(
        signal_in._data.transpose(2, 0, 1, 3), signal_out._data)


@pytest.mark.parametrize(("input_shape", "expected_shape"),[
    ((2, 256), (2, -1)),
    ((3, 2, 256), (6, -1))])
def test_flatten(input_shape, expected_shape):
    """Test Signal.flatten for 2D and 3D input shapes."""
    rng = np.random.default_rng()
    x = rng.random(input_shape)
    signal_in = Signal(x, 44100)
    signal_out = signal_in.flatten()
    npt.assert_allclose(signal_in._data.reshape(expected_shape),
                        signal_out._data)
    assert id(signal_in) != id(signal_out)


def test___eq___equal():
    """Test the eq magic function for equal signals."""
    signal = Signal([1, 2, 3], 44100)
    actual = Signal([1, 2, 3], 44100)
    assert signal == actual


@pytest.mark.parametrize("signal", [
    Signal([1., 2., 3.], 44100, domain='freq', n_samples=4),
    Signal([0.5, 1., 1.5], 44100,  domain='time'),
    Signal([1., 2., 3.], 2*44100,  domain='time'),
    Signal([1., 2., 3.], 44100, domain='time',
           comment='A completely different thing')])
def test___eq___notEqual(signal):
    """Test the eq magic function for non-equal signals."""
    actual = Signal([1., 2., 3.], 44100, domain='time')
    assert not actual == signal


def test__repr__(capfd):
    """Test string representation."""
    print(Signal([0, 1, 0], 44100))
    out, _ = capfd.readouterr()
    assert ("time domain energy Signal:\n"
            "(1,) channels with 3 samples @ 44100 Hz sampling rate "
            "and none FFT normalization") in out


def test_freq_raw():
    """Test accessing the freq_raw attribute."""
    signal = Signal([1, 0, 0, 0], 44100, domain='time')
    npt.assert_allclose(signal.freq_raw, np.array([[1., 1., 1.]]))
    signal.fft_norm = 'amplitude'
    npt.assert_allclose(signal.freq, np.array([[1., 2., 1.]])/4)
    npt.assert_allclose(signal.freq_raw, np.array([[1., 1., 1.]]))


@pytest.mark.parametrize(("data", "n_samples"), [(np.array([[1., 2., 3.]]), 4),
                                                 (np.array([[1.]]), None)])
def test_setter_freq_raw(data, n_samples):
    """Test if attribute freq_raw is set correctly."""
    signal = Signal([1, 2, 3], 44100, fft_norm='amplitude',
                    n_samples=n_samples)
    with pytest.warns(UserWarning, match="Number of samples not given"):
        signal.freq_raw = data
    assert signal.domain == 'freq'
    npt.assert_allclose(signal._data, data)


@pytest.mark.parametrize(("data"), [([1, 2, 3]), ([1.0, 2.0, 3.0]),
                                    ([1+1j, 2+2j, 3+3j])])
def test_setter_freq_raw_dtype(data):
    """
    Test casting and assertions of dtype (not tested during initialization
    because that calls the `freq` setter).
    """
    signal = Signal([0, 1, 2], 44100, 4, "freq")
    signal.freq_raw = data
    npt.assert_allclose(signal.freq_raw, np.atleast_2d(data))
    assert signal.freq_raw.dtype.kind == "c"


def test_setter_freq_raw_dtype_error():
    """Test error handling of the freq_raw setter for invalid input dtype."""
    signal = Signal([0, 1, 2], 44100, 4, "freq")
    with pytest.raises(TypeError, match="int, uint, float, or complex"):
        signal.freq_raw = ["1", "2", "3"]


@pytest.mark.parametrize(("data", "n_samples", "set_domain"), [
    ([0 + 1j, 1 + 1j, 2 + 2j], 4, "time"),
    ([0 + 1j, 1 + 1j, 2 + 2j], 4, "freq"),
    ([0 + 1j, 1 + 1j, 2 + 2j, 3 + 3j], 4, "time"),
    ([0 + 1j, 1 + 1j, 2 + 2j, 3 + 3j], 4, "freq"),
    ([0 + 1j, 1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], 5, "time"),
    ([0 + 1j, 1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], 5, "freq")])
def test_setter_complex_assert(data, n_samples, set_domain):
    """Test setting complex flag of time and frequency domain signals."""
    if set_domain == "time":
        match = ("Signal has complex-valued time data is_complex"
        " flag cannot be `False`.")
    else:
        match = ("Signals frequency spectrum is not conjugate symmetric,"
        " is_complex flag cannot be `False`.")

    signal = Signal(data, 44100, n_samples, "time",
                    is_complex=True)
    signal.domain = set_domain
    with pytest.raises(ValueError, match=match):
        signal.complex = False


@pytest.mark.parametrize(("data", "n_samples", "set_domain", "is_complex",
                          "set_complex", "desired_dtype", "desired_n_bins"), [
    ([0, 1, 2, 3], 4, "time", False, True, "c", 4),
    ([0, 1, 2, 3], 4, "time", True, False, "f", 3),
    ([0, 1, 2, 3], 4, "freq", False, True, "c", 4),
    ([0, 1, 2, 3], 4, "freq", True, False, "f", 3),
    ([0, 1, 2, 3, 4], 5, "time", False, True, "c", 5),
    ([0, 1, 2, 3, 4], 5, "time", True, False, "f", 3),
    ([0, 1, 2, 3, 4], 5, "freq", False, True, "c", 5),
    ([0, 1, 2, 3, 4], 5, "freq", True, False, "f", 3)])
def test_setter_complex_even_odd(data, n_samples, set_domain, is_complex,
                             set_complex, desired_dtype, desired_n_bins):
    """Test setting complex flag of time and frequency domain signals
    with even and odd number of samples.
    """
    signal = Signal(data, 44100, n_samples, "time", is_complex=is_complex)
    signal.domain = set_domain
    signal.complex = set_complex
    assert signal.time.dtype.kind == desired_dtype
    assert signal.freq.shape[1] == desired_n_bins


@pytest.mark.parametrize(("data", "is_complex", "desired"), [
    ([0, 1, 2], False, np.array([0, 16000])),
    ([0, 1, 2, 4], False, np.array([0, 12000, 24000])),
    ([0, 1, 2], True, np.array([-16000, 0, 16000])),
    ([0, 1, 2, 4], True, np.array([-24000, -12000, 0, 12000])),
    ])
def test_frequencies(data, is_complex, desired):
    """
    Test computing the discrete frequencies of the rfft/fft.
    """
    signal = Signal(data, 48000, is_complex=is_complex)
    npt.assert_allclose(signal.frequencies, desired)
