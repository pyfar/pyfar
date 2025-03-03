import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt
import re


def test_regularization_errors(impulse):
    """Test errors."""
    with pytest.raises(RuntimeError,
                       match=re.escape("Regularization objects must be created"
                                       " using one of the 'from_()' "
                                       "classmethods.")):
        assert pf.dsp.RegularizedSpectrumInversion()

    with pytest.raises(ValueError,
                       match="The frequency range needs to specify lower and"
                        " upper limits."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            impulse, (200,))

    with pytest.raises(ValueError,
                       match="Beta must be a scalar or 'energy', 'mean' or "
                             "'max'."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            impulse, (200, 15e3), beta='bla')

    with pytest.raises(ValueError,
                       match="Target function must be a pyfar.Signal object."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            impulse, (200, 20e3), target=1)

    with pytest.raises(ValueError,
                       match="The number of samples in the signal and the "
                       "target function differs but must be equal."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            impulse, (200, 20e3), target=pf.signals.impulse(12))

    with pytest.raises(ValueError,
                       match="The sampling rate of the signal and the "
                       "target function differs but must be equal."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            impulse, (200, 20e3),
            target=pf.signals.impulse(impulse.n_samples, sampling_rate=48e3))

    with pytest.raises(ValueError,
                       match="Regularization must be a pyfar.Signal"
                       " object."):
        assert pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(
            impulse, 1)

    with pytest.raises(ValueError,
                       match="The number of samples in the signal and the "
                       "regularization function differs but must be equal."):
        assert pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(
            impulse, pf.signals.noise(12))

    with pytest.raises(ValueError,
                       match="The sampling rate of the signal and the "
                       "regularization function differs but must be equal."):
        assert pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(
            impulse, pf.signals.noise(impulse.n_samples, sampling_rate=48e3))


@pytest.mark.parametrize(("beta", "expected"), [(1, 0.5), (0.5, 2/3), (0, 1)])
def test_regularization_frequency_range(impulse, beta, expected):
    """Test Regularization from a frequency range using different beta
    values.
    """
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(impulse,
        (200, 10e3), beta=beta)
    inv = Regu.invert

    idx = inv.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(inv.freq[:, idx[0]:idx[1]], 1)
    npt.assert_allclose(inv.freq[:, 0], expected)
    npt.assert_allclose(inv.freq[:, -1], expected)


@pytest.mark.parametrize(("norm", "expected"), [(1, 0.4),
                                                ('max', 1/3)])
def test_regularization_normalization(impulse, norm, expected):
    """Test normalization parameter of Regularization."""
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(impulse*2,
        (200, 10e3), beta=norm)
    inv = Regu.invert

    idx = inv.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(inv.freq[:, idx[0]:idx[1]], 0.5)
    npt.assert_allclose(inv.freq[:, 0], expected)
    npt.assert_allclose(inv.freq[:, -1], expected)


def test_regularization_within(impulse):
    """Test regularization_within parameter."""
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(impulse,
        (200, 10e3), regularization_within=.5)
    inv = Regu.invert

    idx = inv.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(inv.freq[:, idx[0]:idx[1]], 0.8)


def test_regularization_beta_mean_max(impulse):
    """Test normalization factor of caclulated during inversion."""
    impulse = impulse*2
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(impulse,
        (200, 10e3))

    Regu.beta = 'energy'
    assert Regu.beta_value == \
        pf.dsp.energy(impulse) / pf.dsp.energy(Regu.regularization)
    assert Regu.beta == 'energy'

    Regu.beta = 'max'
    assert Regu.beta_value == \
        np.max(np.abs(impulse.freq)) / np.max(np.abs(Regu.regularization.freq))
    assert Regu.beta == 'max'

    Regu.beta = 'mean'
    assert Regu.beta_value == \
        np.mean(np.abs(impulse.freq)) / \
            np.mean(np.abs(Regu.regularization.freq))
    assert Regu.beta == 'mean'


def test_regularization_target(impulse):
    """Test Regularization from a frequency range using a bandpass filter
    target function.
    """
    bp = pf.dsp.filter.butterworth(None, 4, (20, 15.5e3), 'bandpass',
                                   impulse.sampling_rate)
    target = bp.process(impulse)

    ReguTarget = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
        impulse, (20, 15.5e3), target=target)

    inv = ReguTarget.invert
    idx = inv.find_nearest_frequency([20, 15.5e3])

    npt.assert_allclose(np.abs(inv.freq[0, idx]), np.abs(target.freq[0, idx]))


def test_regularization_from_magnitude_spectrum(noise):
    """Test Regularization from an arbitrary signal."""
    Regu = pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(
        pf.signals.impulse(noise.n_samples), noise)

    npt.assert_allclose(Regu.regularization.freq, np.abs(noise.freq))
