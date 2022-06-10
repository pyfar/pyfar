import numpy as np
import numpy.testing as npt
import pytest
import os
from pyfar import Signal
import pyfar.dsp.filter as pff
import pyfar.signals as pfs
from pyfar.signals.deterministic import _match_shape


def test_sine_with_defaults():
    """Test sine signal with default parameters."""

    signal = pfs.sine(99, 441)
    sin = np.sin(np.arange(441) / 44100 * 2 * np.pi * 99)

    assert isinstance(signal, Signal)
    assert signal.comment == ("Sine signal (f = [99] Hz, amplitude = [1], "
                              "phase = [0] rad)")
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == "rms"
    assert signal.time.shape == (1, 441)
    npt.assert_allclose(signal.time, np.atleast_2d(sin))


def test_sine_user_parameters():
    """Test sine signal with custom amplitude, phase, and sampling rate."""
    signal = pfs.sine(99, 441, 2, np.pi/4, 48000)
    sin = np.sin(np.arange(441) / 48000 * 2 * np.pi * 99 + np.pi/4) * 2
    npt.assert_allclose(signal.time, np.atleast_2d(sin))


def test_sine_full_period():
    """Test sine signal with full period option."""
    signal = pfs.sine(99, 441, full_period=True)
    sin = np.sin(np.arange(441) / 44100 * 2 * np.pi * 100)

    assert signal.comment == ("Sine signal (f = [100] Hz, amplitude = [1], "
                              "phase = [0] rad)")
    npt.assert_allclose(signal.time, np.atleast_2d(sin))


def test_sine_multi_channel():
    """Test multi channel sine."""
    signal = pfs.sine([99, 50], 441)
    sin = np.concatenate(
        (np.atleast_2d(np.sin(np.arange(441) / 44100 * 2 * np.pi * 99)),
         np.atleast_2d(np.sin(np.arange(441) / 44100 * 2 * np.pi * 50))), 0)

    assert signal.comment == ("Sine signal (f = [99 50] Hz, "
                              "amplitude = [1 1], "
                              "phase = [0 0] rad)")
    npt.assert_allclose(signal.time, sin)


def test_sine_float():
    """Test sine signal with float number of samples."""
    signal = pfs.sine(100, 441.8)
    assert signal.n_samples == 441


def test_sine_assertions():
    """Test assertions for sine."""
    with pytest.raises(ValueError, match="The frequency must be"):
        pfs.sine(40000, 100)
    with pytest.raises(ValueError, match="The parameters frequency"):
        pfs.sine(100, 100, [1, 2], [1, 2, 3])


def test_impulse_with_defaults():
    """Test impulse with default parameters"""
    signal = pfs.impulse(3)
    assert isinstance(signal, Signal)
    npt.assert_allclose(signal.time, np.atleast_2d([1, 0, 0]))
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == 'none'
    assert signal.comment == ("Impulse signal (delay = [0] samples, "
                              "amplitude = [1])")


def test_impulse_with_user_parameters():
    """Test impulse with custom delay, amplitude and sampling rate"""
    signal = pfs.impulse(3, 1, 2, 48000)
    npt.assert_allclose(signal.time, np.atleast_2d([0, 2, 0]))
    assert signal.sampling_rate == 48000


def test_impulse_multi_channel():
    """Test multi channel impulse."""
    # test with array and number
    signal = pfs.impulse(3, [0, 1], 1)
    ref = np.atleast_2d([[1, 0, 0], [0, 1, 0]])
    npt.assert_allclose(signal.time, ref)
    # test with two arrays
    signal = pfs.impulse(3, [0, 1], [1, 2])
    ref = np.atleast_2d([[1, 0, 0], [0, 2, 0]])
    npt.assert_allclose(signal.time, ref)


def test_impulse_float():
    """Test impulse signal with float number of samples."""
    signal = pfs.impulse(441.8)
    assert signal.n_samples == 441


def test_impulse_assertions():
    """Test assertions for impulse functions"""
    with pytest.raises(ValueError, match="The parameters delay"):
        pfs.impulse(10, [1, 2], [1, 2, 3])


def test_noise_with_defaults():
    """Test noise with default parameters."""
    signal = pfs.noise(100)

    assert isinstance(signal, Signal)
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == "rms"
    assert signal.comment == "white noise signal (rms = [1])"
    npt.assert_allclose(np.sqrt(np.mean(signal.time**2, axis=-1)), 1)


def test_white_with_user_parameters():
    """Test wite noise with rms and sampling rate."""
    signal = pfs.noise(100, "pink", rms=2, sampling_rate=48000)

    assert signal.sampling_rate == 48000
    assert signal.comment == "pink noise signal (rms = [2])"
    npt.assert_allclose(np.sqrt(np.mean(signal.time**2, axis=-1)), 2)


def test_noise_multi_channel():
    """Test multi channel noise."""
    rms = [[1, 2, 3], [4, 5, 6]]
    signal = pfs.noise(100, rms=rms)
    npt.assert_allclose(np.sqrt(np.mean(signal.time**2, axis=-1)), rms)


def test_noise_seed():
    """Test passing seeds to the random generator."""
    a = pfs.noise(100, seed=1)
    b = pfs.noise(100, seed=1)
    assert a == b

    a = pfs.noise(100)
    b = pfs.noise(100)
    with pytest.raises(AssertionError):
        assert a == b


def test_white_float():
    """Test noise signal with float number of samples."""
    signal = pfs.noise(441.8)
    assert signal.n_samples == 441


def test_noise_rms_pink_spectrum():
    """
    Test for constant energy across filters of constant relative band width
    for pink noise.
    """
    # filtered pink noise
    # (use only center octaves, because the spectrum is less stochastic there)
    signal = pfs.noise(5000, "pink", seed=1)
    signal = pff.fractional_octave_bands(signal, 1, freq_range=(1e3, 16e3))
    # check if stdandard deviation is less then 1%
    rms = np.atleast_1d(np.sqrt(np.mean(signal.time**2, axis=-1)))
    assert np.std(rms) < .01


def test_noise_assertion():
    """Test noise with invalid spectrum."""
    with pytest.raises(ValueError, match="spectrum is 'brown'"):
        pfs.noise(200, "brown")


def test_pulsed_noise_with_defaults():
    """Test pulsed noise signal generation with default values."""
    signal = pfs.pulsed_noise(n_pulse=200, n_pause=100)

    assert isinstance(signal, Signal)
    assert signal.sampling_rate == 44100
    assert signal.fft_norm == "rms"
    assert signal.n_samples == 5 * 200 + 4 * 100
    assert signal.cshape == (1, )
    assert np.all(signal.time[..., 200:300] == 0)
    assert np.all(signal.time[..., 500:600] == 0)
    assert np.all(signal.time[..., 800:900] == 0)
    assert np.all(signal.time[..., 1100:1200] == 0)
    assert signal.comment == ("frozen pink pulsed noise signal (rms = 1, "
                              "5 repetitions, 200 samples pulse duration, "
                              "100 samples pauses, and 90 samples fades.")


def test_pulsed_noise_fade_spectrum_and_seed():
    """
    Test pulsed noise signal generation with custom n_fade, spectrum, and seed.
    """
    # pink noise with 50 samples fade
    signal = pfs.pulsed_noise(
        n_pulse=200, n_pause=100, n_fade=50, spectrum="pink", seed=1)

    noise = pfs.noise(200, "pink", seed=1).time
    fade = np.sin(np.linspace(0, np.pi / 2, 50))**2
    noise[..., 0:50] *= fade
    noise[..., -50:] *= fade[::-1]

    npt.assert_allclose(signal.time[..., 0:200], noise)

    # white noise without fade
    signal = pfs.pulsed_noise(
        n_pulse=200, n_pause=100, n_fade=0, spectrum="white", seed=1)
    npt.assert_allclose(
        signal.time[..., 0:200], pfs.noise(200, "white", seed=1).time)


def test_pulsed_noise_repetitions():
    """Test pulsed noise signal generation with custom repetitions."""
    signal = pfs.pulsed_noise(n_pulse=200, n_pause=100, repetitions=6)
    assert signal.n_samples == 6 * 200 + 5 * 100


def test_pulsed_noise_rms():
    """Test pulsed noise signal generation with custom rms."""
    signal = pfs.pulsed_noise(n_pulse=200, n_pause=100, n_fade=0, rms=2)
    npt.assert_allclose(
        np.sqrt(np.mean(signal.time[..., 0:200]**2, axis=-1)),
        np.atleast_1d(2))


def test_pulsed_noise_freeze():
    """Test pulsed noise signal generation with frozen option."""
    signal = pfs.pulsed_noise(200, 100, 50, frozen=True)
    npt.assert_allclose(signal.time[..., 0:200], signal.time[..., 300:500])

    signal = pfs.pulsed_noise(200, 100, 50, frozen=False)
    with pytest.raises(AssertionError):
        npt.assert_allclose(signal.time[..., 0:200], signal.time[..., 300:500])


def test_pulsed_noise_sampling_rate():
    """Test pulsed noise signal generation with custom sampling_rate."""
    signal = pfs.pulsed_noise(200, 100, sampling_rate=48000)
    assert signal.sampling_rate == 48000


def test_pulsed_noise_float():
    """Test pulsed noise signal with float number of samples."""
    signal = pfs.pulsed_noise(200.8, 100.8, 50.8)
    assert signal.n_samples == 5 * 200 + 4 * 100


def test_pulsed_noise_assertions():
    """Test assertions for pulsed noise."""
    with pytest.raises(ValueError, match="n_fade too large."):
        pfs.pulsed_noise(100, 100)

    with pytest.raises(ValueError, match="spectrum is 'brown'"):
        pfs.pulsed_noise(200, 100, spectrum="brown")


def test_linear_sweep_time_against_reference():
    """Test linear sweep against manually verified reference."""
    sweep = pfs.linear_sweep_time(2**10, [1e3, 20e3])
    reference = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "references",
        "signals.linear_sweep_time.csv"))

    npt.assert_allclose(sweep.time, np.atleast_2d(reference))
    assert sweep.cshape == (1, )
    assert sweep.n_samples == 2**10
    assert sweep.sampling_rate == 44100
    assert sweep.fft_norm == "none"
    assert sweep.comment == ("linear sweep between 1000.0 and 20000.0 Hz with "
                             "90 samples squared cosine fade-out.")


def test_linear_sweep_time_amplitude_sampling_rate():
    """Test linear sweep with custom amplitude and sampling rate."""
    sweep = pfs.linear_sweep_time(
        2**10, [1e3, 20e3], amplitude=2, sampling_rate=48000)

    assert sweep.sampling_rate == 48000
    npt.assert_allclose(
        np.max(np.abs(sweep.time)), np.array(2), rtol=1e-6, atol=1e-6)


def test_linear_sweep_time_float():
    """Test linear sweep with float number of samples."""
    sweep = pfs.linear_sweep_time(100.6, [1e3, 20e3])
    assert sweep.n_samples == 100


def test_linear_sweep_time_assertions():
    """Test assertions for linear sweep."""
    with pytest.raises(ValueError, match="The sweep must be longer"):
        pfs.linear_sweep_time(50, [1, 2])
    with pytest.raises(ValueError, match="frequency_range must be an array"):
        pfs.linear_sweep_time(100, 1)
    with pytest.raises(ValueError, match="Upper frequency limit"):
        pfs.linear_sweep_time(100, [1, 40e3])


def test_exponential_sweep_time_against_reference():
    """Test exponential sweep against manually verified reference."""
    sweep = pfs.exponential_sweep_time(2**10, [1e3, 20e3])
    reference = np.loadtxt(os.path.join(os.path.dirname(__file__),
                           "references", "signals.exponential_sweep_time.csv"))

    npt.assert_allclose(sweep.time, np.atleast_2d(reference))
    assert sweep.cshape == (1, )
    assert sweep.n_samples == 2**10
    assert sweep.sampling_rate == 44100
    assert sweep.fft_norm == "none"
    assert sweep.comment == ("exponential sweep between 1000.0 and 20000.0 Hz "
                             "with 90 samples squared cosine fade-out.")


def test_exponential_sweep_time_amplitude_sampling_rate():
    """Test exponential sweep with custom amplitude and sampling rate."""
    sweep = pfs.exponential_sweep_time(
        2**10, [1e3, 20e3], amplitude=2, sampling_rate=48000)

    assert sweep.sampling_rate == 48000
    npt.assert_allclose(
        np.max(np.abs(sweep.time)), np.array(2), rtol=1e-6, atol=1e-6)


def test_exponential_sweep_time_rate():
    """Test exponential sweep with sweep rate."""
    sweep = pfs.exponential_sweep_time(None, [1e3, 2e3], sweep_rate=10)

    # duration in seconds
    T = 1 / 10 / np.log(2) * np.log(2e3 / 1e3)
    # duration in samples
    n_samples = np.round(T * 44100)
    # only test the length because it is the only thing that changes
    assert sweep.n_samples == n_samples


def test_exponential_sweep_time_assertion():
    with pytest.raises(ValueError, match="The exponential sweep can not"):
        pfs.exponential_sweep_time(2**10, [0, 20e3])


def test_match_shape():
    """Test _match_shape with all possible inputs."""
    a = 1
    b = [[1, 2, 3], [1, 2, 3]]

    cshape, (a_match, b_match) = _match_shape(a, b)
    assert cshape == (2, 3)
    npt.assert_allclose(np.ones(cshape), a_match)
    npt.assert_allclose(b, b_match)
