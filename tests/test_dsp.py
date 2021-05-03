import numpy as np
import numpy.testing as npt
import pytest
from pyfar import Signal

from pyfar import dsp


def test_phase_rad(sine_plus_impulse):
    """Test the function returning the phase of a signal in radians."""
    phase = dsp.phase(sine_plus_impulse, deg=False, unwrap=False)
    truth = np.angle(sine_plus_impulse.freq)
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_phase_deg(sine_plus_impulse):
    """Test the function returning the phase of a signal in degrees."""
    phase = dsp.phase(sine_plus_impulse, deg=True, unwrap=False)
    truth = np.degrees(np.angle(sine_plus_impulse.freq))
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_phase_unwrap(sine_plus_impulse):
    """Test the function returning the unwrapped phase of a signal."""
    phase = dsp.phase(sine_plus_impulse, deg=False, unwrap=True)
    truth = np.unwrap(np.angle(sine_plus_impulse.freq))
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_phase_deg_unwrap(sine_plus_impulse):
    """Test the function returning the unwrapped phase of a signal in deg."""
    phase = dsp.phase(sine_plus_impulse, deg=True, unwrap=True)
    truth = np.degrees(np.unwrap(np.angle(sine_plus_impulse.freq)))
    npt.assert_allclose(phase, truth, rtol=1e-10)


def test_group_delay_single_channel(impulse_group_delay):
    """Test the function returning the group delay of a signal,
    single channel."""
    signal = impulse_group_delay[0]

    with pytest.raises(ValueError, match="Invalid method"):
        dsp.group_delay(signal, method='invalid')

    with pytest.raises(ValueError, match="not supported"):
        dsp.group_delay(signal, method='fft', frequencies=[1, 2, 3])

    grp = dsp.group_delay(signal, method='scipy')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)

    grp = dsp.group_delay(signal, method='fft')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)

    grp = dsp.group_delay(
        signal, method='fft')
    assert grp.shape == (signal.n_bins, )
    npt.assert_allclose(grp, impulse_group_delay[1].flatten(), rtol=1e-10)


def test_group_delay_two_channel(impulse_group_delay_two_channel):
    """Test the function returning the group delay of a signal,
    two channels."""
    signal = impulse_group_delay_two_channel[0]
    grp = dsp.group_delay(signal, method='scipy')
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(grp, impulse_group_delay_two_channel[1], rtol=1e-10)

    grp = dsp.group_delay(signal, method='fft')
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(grp, impulse_group_delay_two_channel[1], rtol=1e-10)


def test_group_delay_two_by_two_channel(
        impulse_group_delay_two_by_two_channel):
    """Test the function returning the group delay of a signal,
    2-by-2 channels."""
    signal = impulse_group_delay_two_by_two_channel[0]
    grp = dsp.group_delay(signal)
    assert grp.shape == (signal.cshape + (signal.n_bins,))
    npt.assert_allclose(
        grp, impulse_group_delay_two_by_two_channel[1], rtol=1e-10)


def test_group_delay_custom_frequencies(impulse_group_delay):
    """Test the function returning the group delay of a signal,
    called for specific frequencies."""
    signal = impulse_group_delay[0]
    # Single frequency, of type int
    frequency = 1000
    frequency_idx = np.abs(signal.frequencies-frequency).argmin()
    grp = dsp.group_delay(signal, frequency, method='scipy')
    assert grp.shape == ()
    npt.assert_allclose(grp, impulse_group_delay[1][0, frequency_idx])
    # Multiple frequencies
    frequency = np.array([1000, 2000])
    frequency_idx = np.abs(
        signal.frequencies-frequency[..., np.newaxis]).argmin(axis=-1)
    grp = dsp.group_delay(signal, frequency, method='scipy')
    assert grp.shape == (2,)
    npt.assert_allclose(grp, impulse_group_delay[1][0, frequency_idx])


def test_normalization_time_max_max_value():
    """Test the function along time, max, max & value path."""
    signal = Signal([[1, 2, 1], [1, 4, 1]], 44100)
    truth = Signal([[0.25, 0.5, 0.25], [0.25, 1., 0.25]], 44100)
    answer = dsp.normalize(signal, normalize='time', normalize_to='max',
                           channel_handling='max')
    assert answer == truth


def test_normalization_magnitude_mean_min_freqrange():
    """Test the function along magnitude, mean, min & value path."""
    signal = Signal([[1, 4, 1], [1, 10, 1]], 44100, n_samples=4, domain='freq')
    truth = Signal([[2.5, 10, 2.5], [2.5, 25, 2.5]], 44100, n_samples=4,
                   domain='freq')
    answer = dsp.normalize(signal, normalize='magnitude', normalize_to='mean',
                           channel_handling='min', value=10)
    assert answer == truth


def test_average_complex():
    """Test the function in complex domain"""
    signal = Signal([[1, 4, 1], [1, 10, 1]], 44100, n_samples=4, domain='freq')
    truth = Signal([1, 7, 1], 44100, n_samples=4, domain='freq')
    answer = dsp.average(signal, average_mode='complex')
    assert answer == truth


def test_xfade(impulse):
    first = np.ones(5001)
    idx_1 = 500
    second = np.ones(5001)*2
    idx_2 = 1000

    res = dsp.dsp._cross_fade(first, second, [idx_1, idx_2])
    np.testing.assert_array_almost_equal(first[:idx_1], res[:idx_1])
    np.testing.assert_array_almost_equal(second[idx_2:], res[idx_2:])

    idx_1 = 501
    idx_2 = 1000
    res = dsp.dsp._cross_fade(first, second, [idx_1, idx_2])
    np.testing.assert_array_almost_equal(first[:idx_1], res[:idx_1])
    np.testing.assert_array_almost_equal(second[idx_2:], res[idx_2:])


def test_regu_inversion(impulse):

    with pytest.raises(
            ValueError, match='needs to be of type pyfar.Signal'):
        dsp.regularized_spectrum_inversion('error', (1, 2))

    with pytest.raises(
            ValueError, match='lower and upper limits'):
        dsp.regularized_spectrum_inversion(impulse, (2))

    impulse.freq = impulse.freq*2
    impulse.time = impulse.time*2

    res = dsp.regularized_spectrum_inversion(impulse, [200, 10e3])

    ind = impulse.find_nearest_frequency([200, 10e3])
    npt.assert_allclose(
        res.freq[:, ind[0]:ind[1]],
        np.ones((1, ind[1]-ind[0]), dtype=complex)*0.5)

    npt.assert_allclose(res.freq[:, 0], [0.25])
    npt.assert_allclose(res.freq[:, -1], [0.25])
