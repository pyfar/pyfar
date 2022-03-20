import pytest
from pytest import raises
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import pyfar as pf
from pyfar.dsp import (InterpolateSpectrum,
                       fractional_delay_sinc,
                       resample_sinc)


def test_fractional_delay_sinc_assertions():
    """Test if the assertions are raised correctly"""

    # wrong audio data type
    with raises(TypeError, match="Input data has to be of type pyfar.Signal"):
        fractional_delay_sinc(pf.FrequencyData(1, 1), .5)

    # wrong values for order and side_lobe_suppression
    with raises(ValueError, match="The order must be > 0"):
        fractional_delay_sinc(pf.Signal([1, 0, 0], 44100), .5, 0)
    with raises(ValueError, match="The side lobe suppression must be > 0"):
        fractional_delay_sinc(pf.Signal([1, 0, 0], 44100), .5, 2, 0)

    # filter length exceeds signal length
    with raises(ValueError, match="The order is 30 but must not exceed 2"):
        fractional_delay_sinc(pf.Signal([1, 0, 0], 44100), .5)

    # wrong mode
    with raises(ValueError, match="The mode is 'full' but must be 'cut'"):
        fractional_delay_sinc(pf.Signal([1, 0, 0], 44100), .5, 2, mode="full")


@pytest.mark.parametrize("mode", ["cut", "cyclic"])
@pytest.mark.parametrize("delays_impulse, fractional_delays", [
    # single channel signals and delays
    # (positive/negative with fractions <0.5 and >0.5)
    (64, 10.4), (64, 10.6), (64, -10.4), (64, -10.6),
    # special case for testing a special line
    (64, -60),
    # multi channel signal with single channel delays
    ([64, 32], 10.4), ([[64, 32], [48, 16]], 10.4),
    # multi channel signals with multi channel delays
    ([64, 32], [10.4, 5.4]), ([[64, 32], [48, 16]], [10.4, 5.4])
])
def test_fractional_delay_sinc_channels(
        mode, delays_impulse, fractional_delays):
    """
    Test fractional delay with different combinations of single/multi-channel
    signals and delays and the two modes "cut" and "cyclic"
    """

    # generate input and delay signal
    signal = pf.signals.impulse(128, delays_impulse)
    delayed = fractional_delay_sinc(signal, fractional_delays, mode=mode)

    # frequency up to which group delay is tested
    f_id = delayed.find_nearest_frequency(19e3)

    # get and broadcast actual and target group delays
    group_delays = pf.dsp.group_delay(delayed)[..., :f_id]
    group_delays = np.broadcast_to(group_delays, signal.cshape + (f_id, ))

    target_delays = np.atleast_1d(
        np.array(delays_impulse) + np.array(fractional_delays))
    target_delays = np.broadcast_to(target_delays[..., np.newaxis],
                                    signal.cshape + (f_id, ))

    # check equality
    npt.assert_allclose(group_delays[..., :f_id], target_delays, atol=.05)


@pytest.mark.parametrize("order", [2, 3])
def test_fractional_delay_order(order):
    """Test if the order parameter behaves as intended"""

    signal = pf.signals.impulse(32, 16)
    delayed = pf.dsp.fractional_delay_sinc(signal, 0.5, order)

    # number of non-zero samples must equal filter_length = order+1
    assert np.count_nonzero(np.abs(delayed.time) > 1e-14) == order + 1


@pytest.mark.parametrize("delay", [20.1, -20.1])
def test_fractional_delay_mode_cut(delay):
    """Test the mode cut"""

    signal = pf.signals.impulse(16, 8)
    delayed = fractional_delay_sinc(signal, delay, 2)

    # if the delay is too large, the signal is shifted out of the sampled range
    # and only zeros remain
    npt.assert_array_equal(delayed.time, np.zeros_like(delayed.time))


@pytest.mark.parametrize("delay", [30.4, -30.4])
def test_fractional_delay_mode_cyclic(delay):
    """Test the mode delay"""

    signal = pf.signals.impulse(32, 16)
    delayed = fractional_delay_sinc(signal, delay, mode="cyclic")

    # if the delay is too large, it is cyclicly shifted
    group_delay = pf.dsp.group_delay(delayed)[0]
    npt.assert_allclose(group_delay, (16+delay) % 32, atol=.05)


def test_resample_sinc_assertions():
    """Test if the assertions are raised correctly"""

    # wrong audio data type
    with raises(TypeError, match="Input data has to be of type pyfar.Signal"):
        resample_sinc(pf.FrequencyData(1, 1), 22e3)

    # too many processes
    # wrong audio data type
    with raises(ValueError, match="processes is 1000000"):
        resample_sinc(pf.Signal(1, 1), 22e3, 1000000)


def test_resample_sinc_single_channel():
    """
    The resempling itself is tested by resampy. Here we only test mutability
    and the meta data of the output signal.
    """

    signal = pf.signals.impulse(256, 128)
    signal_resampled = resample_sinc(signal, 48e3)

    # check if input remained unchanged
    npt.assert_equal(signal.time, pf.signals.impulse(256, 128).time)

    # test target sampling rate and n_samples
    assert signal_resampled.sampling_rate == 48e3
    assert signal_resampled.n_samples > signal.n_samples


@pytest.mark.parametrize("processes", [1, 2, None])
def test_resample_sinc(processes):
    """
    The resempling itself is tested by resampy. Here we only test for different
    array shapes and process modes.
    """

    # resample single channel signals for reference
    ref_60 = resample_sinc(pf.signals.impulse(128, 60), 48e3).time
    ref_62 = resample_sinc(pf.signals.impulse(128, 62), 48e3).time
    ref_64 = resample_sinc(pf.signals.impulse(128, 64), 48e3).time
    ref_66 = resample_sinc(pf.signals.impulse(128, 66), 48e3).time

    # resample two channel signal with flat cshape
    signal = pf.signals.impulse(128, [60, 62])
    signal_resampled = resample_sinc(signal, 48e3, processes)

    assert signal_resampled.cshape == signal.cshape

    npt.assert_allclose(ref_60.flatten(), signal_resampled.time[0])
    npt.assert_allclose(ref_62.flatten(), signal_resampled.time[1])

    # resample two-by-two channel signal
    signal = pf.signals.impulse(128, [[60, 62], [64, 66]])
    signal_resampled = resample_sinc(signal, 48e3, processes)

    assert signal_resampled.cshape == signal.cshape

    npt.assert_allclose(ref_60.flatten(), signal_resampled.time[0, 0])
    npt.assert_allclose(ref_62.flatten(), signal_resampled.time[0, 1])
    npt.assert_allclose(ref_64.flatten(), signal_resampled.time[1, 0])
    npt.assert_allclose(ref_66.flatten(), signal_resampled.time[1, 1])


def test_interpolate_spectrum_init():
    """Test return objects"""
    fd = pf.FrequencyData([1, .5], [100, 200])

    # interpolation object
    interpolator = InterpolateSpectrum(
        fd, "complex", ("linear", "linear", "linear"))
    assert isinstance(interpolator, InterpolateSpectrum)

    # interpolation result
    signal = interpolator(8, 44100)
    assert isinstance(signal, pf.Signal)


def test_interpolate_spectrum_init_assertions():
    """Test if init raises assertions correctly"""
    fd = pf.FrequencyData([1, .5], [100, 200])

    # data (invalid type)
    with raises(TypeError, match="data must be"):
        InterpolateSpectrum(1, "complex", ("linear", "linear", "linear"))
    # data (not enough bins)
    with raises(ValueError, match="data.n_bins must be at least 2"):
        fd_short = pf.FrequencyData(1, 100)
        InterpolateSpectrum(
            fd_short, "complex", ("linear", "linear", "linear"))

    # test invalid method
    with raises(ValueError, match="method is 'invalid'"):
        InterpolateSpectrum(fd, "invalid", ("linear", "linear", "linear"))

    # test kind (invald type)
    with raises(ValueError, match="kind must be a tuple of length 3"):
        InterpolateSpectrum(fd, "complex", "linear")
    # test kind (invalid length)
    with raises(ValueError, match="kind must be a tuple of length 3"):
        InterpolateSpectrum(fd, "complex", ("linear", "linear"))
    # test kind (wrong entry)
    with raises(ValueError, match="kind contains 'wrong'"):
        InterpolateSpectrum(fd, "complex", ("linear", "linear", "wrong"))

    # test fscale
    with raises(ValueError, match="fscale is 'nice'"):
        InterpolateSpectrum(
            fd, "complex", ("linear", "linear", "linear"), fscale="nice")

    # test clip (wrong value of bool)
    with raises(ValueError, match="clip must be a tuple of length 2"):
        InterpolateSpectrum(
            fd, "complex", ("linear", "linear", "linear"), clip=True)
    # test clip (invalid type)
    with raises(ValueError, match="clip must be a tuple of length 2"):
        InterpolateSpectrum(
            fd, "complex", ("linear", "linear", "linear"), clip=1)
    # test clip (invalid length)
    with raises(ValueError, match="clip must be a tuple of length 2"):
        InterpolateSpectrum(
            fd, "complex", ("linear", "linear", "linear"), clip=(1, 2, 3))


@pytest.mark.parametrize(
    "method, freq_in, frequencies, n_samples, sampling_rate, freq_out",
    [
     ("complex", [1+2j, 2+1j], [1, 2], 12, 6,
      [0+3j, 0.5+2.5j, 1+2j, 1.5+1.5j, 2+1j, 2.5+0.5j, 3+0j]),

     ("magnitude_phase",
      # magnitude increases with 1 per Hz, phase with pi per Hz
      [np.linspace(1, 2, 3) * np.exp(-1j * np.linspace(np.pi, np.pi*2, 3))],
      [1, 1.5, 2], 24, 6,
      # freq_out be means of magnitude and unwrapped phase response
      [np.linspace(0, 3, 13), np.linspace(0, 3*np.pi, 13)]),

     ("magnitude", [1, 2], [1, 2], 12, 6,
      [0, .5, 1, 1.5, 2, 2.5, 3])
    ])
def test_interpolate_spectrum_interpolation(
        method, freq_in, frequencies, freq_out, n_samples, sampling_rate):
    """
    Test the if the interpolated spectrum matches the reference across methods.
    """

    # create test data
    data = pf.FrequencyData(freq_in, frequencies)
    interpolator = InterpolateSpectrum(
        data, method, ("linear", "linear", "linear"))
    signal = interpolator(n_samples, sampling_rate)

    # check output depending on method
    if method == "magnitude_phase":
        # test magnitude and unwrapped phase response
        npt.assert_allclose(np.abs(signal.freq), np.atleast_2d(freq_out[0]))
        npt.assert_allclose(pf.dsp.phase(signal, unwrap=True),
                            np.atleast_2d(freq_out[1]))
    else:
        # test complex spectrum
        npt.assert_allclose(signal.freq, np.atleast_2d(freq_out))


def test_interpolate_spectrum_clip():
    """Test if clipping the magnitude data works."""

    data = pf.FrequencyData([1, 2], [1, 2])
    # interpolate with and without clipping
    interpolator = InterpolateSpectrum(
        data, "magnitude", ("linear", "linear", "linear"))
    signal_no_clip = interpolator(6, 6)

    interpolator = InterpolateSpectrum(
        data, "magnitude", ("linear", "linear", "linear"), clip=(1, 2))
    signal_clip = interpolator(6, 6)

    assert np.any(np.abs(signal_no_clip.freq) < 1) and \
           np.any(np.abs(signal_no_clip.freq) > 2)
    assert np.all(np.abs(signal_clip.freq) >= 1) and \
           np.all(np.abs(signal_clip.freq) <= 2)


def test_interpolate_spectrum_fscale():
    """
    Test frequency vectors for linear and logarithmic frequency interpolation.
    """

    # test parametres and data
    f_in_lin = [0, 10, 20]
    f_in_log = np.log([10, 10, 20])
    n_samples = 10
    sampling_rate = 40
    f_query_lin = pf.dsp.fft.rfftfreq(n_samples, sampling_rate)
    f_query_log = f_query_lin.copy()
    f_query_log[0] = f_query_log[1]
    f_query_log = np.log(f_query_log)
    data = pf.FrequencyData([1, 1, 1], f_in_lin)

    # generate interpolator with linear frequency
    interpolator_lin = InterpolateSpectrum(
        data, "magnitude", ("linear", "linear", "linear"), fscale="linear")
    _ = interpolator_lin(n_samples, sampling_rate)
    # generate interpolator with logarithmic frequency
    interpolator_log = InterpolateSpectrum(
        data, "magnitude", ("linear", "linear", "linear"), fscale="log")
    _ = interpolator_log(n_samples, sampling_rate)

    # test frequency vectors
    npt.assert_allclose(interpolator_lin._f_in, f_in_lin)
    npt.assert_allclose(interpolator_lin._f_query, f_query_lin)

    npt.assert_allclose(interpolator_log._f_in, f_in_log)
    npt.assert_allclose(interpolator_log._f_query, f_query_log)


def test_interpolate_spectrum_show():
    """Test plotting the results.

    This only tests if the code finishes without errors. Because the plot is
    an informal plot for inspection, we don't test specifics of the figure and
    axes for speed up the testing."""

    data = pf.FrequencyData([1, 2], [1, 2])
    interpolator = InterpolateSpectrum(
        data, "magnitude", ("linear", "linear", "linear"))
    _ = interpolator(10, 10, show=True)

    plt.close()
