import numpy as np
import numpy.testing as npt
import pytest

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
