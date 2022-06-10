import pytest
from pytest import raises
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import os
import pyfar as pf
from pyfar.dsp import (InterpolateSpectrum,
                       smooth_fractional_octave,
                       fractional_time_shift)


def test_smooth_fractional_octave_assertions():
    """Test if the assertions are raised correctly"""

    # wrong audio data type
    with raises(TypeError, match="Input signal has to be of type"):
        smooth_fractional_octave(pf.FrequencyData(1, 1), .5)

    # wrong value for mode
    with raises(ValueError, match="mode is 'smooth' but must be"):
        smooth_fractional_octave(pf.Signal(1, 1), 1, "smooth")

    # smoothing width too small
    with raises(ValueError, match="The smoothing width"):
        smooth_fractional_octave(pf.Signal([1, 0], 1), 1)


@pytest.mark.parametrize("mode", (
    "magnitude_zerophase", "magnitude_phase", "magnitude", "complex"))
def test_smooth_fractional_octave_mode(mode):
    """
    Test return signal for different smoothing modes against saved references
    """

    # load input data
    input = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            "dsp.smooth_fractional_octave_input.csv"))
    input = pf.Signal(input, 44100)

    # smooth
    output, _ = smooth_fractional_octave(input, 1, mode)

    # compare to reference
    reference = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            f"dsp.smooth_fractional_octave_{mode}.csv"))
    npt.assert_allclose(output.time.flatten(), reference)


@pytest.mark.parametrize("num_fractions", (1, 5))
def test_smooth_fractional_octave_num_fractions(num_fractions):
    """
    Test return signal for different smoothing widths against saved references
    """

    # load input data
    signal = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            "dsp.smooth_fractional_octave_input.csv"))
    signal = pf.Signal(signal, 44100)

    # smooth
    smoothed, _ = smooth_fractional_octave(signal, num_fractions)

    # compare to reference
    reference = np.loadtxt(os.path.join(
            os.path.dirname(__file__), "references",
            f"dsp.smooth_fractional_octave_{num_fractions}.csv"))
    npt.assert_allclose(smoothed.time.flatten(), reference)


def test_smooth_fractional_octave_window_parameter():
    """
    Test the returned window paramters. Only the types are tested. Testing
    values would require implementing the same code as contained in the
    function
    """

    _, window_paraeter = smooth_fractional_octave(pf.signals.impulse(64), 1)

    assert len(window_paraeter) == 2
    assert isinstance(window_paraeter[0], int)
    assert isinstance(window_paraeter[1], float)


@pytest.mark.parametrize("amplitudes", (
    1,                   # single channel signal
    [1, .9, .8, .7],     # flat multi-channel signal.
    [[1, .9], [.8, .7]]  # 2D multi-channel signal
))
def test_smooth_fractional_octave_input_signal_shape(amplitudes):
    """
    - Test for different shapes of the input signal
    - Test if padding is correct (if it would not be the output spectrum
      would be shifted
    """

    # manually path a window for smoothing (undocumented feature for testing)
    # The window does not smooth and thus the return spectrum should be
    # identical to the input spectrum (appart from interpolation errors during
    # step 1 and 3)
    window = np.array([0, 0, 0, 1, 0, 0, 0])

    signal = pf.signals.impulse(64, amplitude=amplitudes)
    signal = pf.dsp.filter.bell(signal, 4e3, 10, 3)
    smoothed, _ = smooth_fractional_octave(signal, 1, window=window)
    npt.assert_allclose(np.abs(smoothed.freq), np.abs(signal.freq), atol=.02)


def test_fractional_time_shift_assertions():
    """Test if the assertions are raised correctly"""

    # wrong audio data type
    with raises(TypeError, match="Input data has to be of type pyfar.Signal"):
        fractional_time_shift(pf.FrequencyData(1, 1), .5)

    # wrong values for order and side_lobe_suppression
    with raises(ValueError, match="The order must be > 0"):
        fractional_time_shift(pf.Signal([1, 0, 0], 44100), .5, order=0)
    with raises(ValueError, match="The side lobe suppression must be > 0"):
        fractional_time_shift(pf.Signal([1, 0, 0], 44100), .5, "samples", 2, 0)

    # filter length exceeds signal length
    with raises(ValueError, match="The order is 30 but must not exceed 2"):
        fractional_time_shift(pf.Signal([1, 0, 0], 44100), .5)

    # wrong unit
    with raises(ValueError, match="Unit is 'meter' but has to be"):
        fractional_time_shift(pf.signals.impulse(64), 1, 'meter')

    # wrong mode
    with raises(ValueError, match="The mode is 'full' but must be 'linear'"):
        fractional_time_shift(pf.Signal([1, 0, 0], 44100), .5, 2, mode="full")


@pytest.mark.parametrize("mode", ["linear", "cyclic"])
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
def test_fractional_time_shift_channels(
        mode, delays_impulse, fractional_delays):
    """
    Test fractional delay with different combinations of single/multi-channel
    signals and delays and the two modes "linear" and "cyclic"
    """

    # generate input and delay signal
    signal = pf.signals.impulse(128, delays_impulse)
    delayed = fractional_time_shift(signal, fractional_delays, mode=mode)

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


def test_fractional_time_shift_unit():
    """Test passing shift in different units"""

    impulse = pf.signals.impulse(128, 64)
    delayed_samples = fractional_time_shift(impulse, 1, 'samples')
    delayed_seconds = fractional_time_shift(impulse, 1/44100, 's')

    npt.assert_almost_equal(delayed_samples.time, delayed_seconds.time)


@pytest.mark.parametrize("order", [2, 3])
def test_fractional_delay_order(order):
    """Test if the order parameter behaves as intended"""

    signal = pf.signals.impulse(32, 16)
    delayed = pf.dsp.fractional_time_shift(signal, 0.5, order=order)

    # number of non-zero samples must equal filter_length = order+1
    assert np.count_nonzero(np.abs(delayed.time) > 1e-14) == order + 1


@pytest.mark.parametrize("delay", [30.4, -30.4])
def test_fractional_delay_mode_cyclic(delay):
    """Test the mode delay"""

    signal = pf.signals.impulse(32, 16)
    delayed = fractional_time_shift(signal, delay, mode="cyclic")

    # if the delay is too large, it is cyclicly shifted
    group_delay = pf.dsp.group_delay(delayed)[0]
    npt.assert_allclose(group_delay, (16+delay) % 32, atol=.05)


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
