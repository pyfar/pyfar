import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


def test_regularization_errors(impulse):
    """Test errors"""
    with pytest.raises(RuntimeError,
                       match="Regularization objects must be created using a"
                       " classmethod."):
        assert pf.dsp.RegularizedSpectrumInversion()

    with pytest.raises(ValueError,
                       match="The frequency range needs to specify lower and"
                        " upper limits."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range((200,))

    with pytest.raises(ValueError,
                       match="Target function must be a pyfar.Signal object."):
        assert pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
            (200, 20e3)).invert(impulse, target=1)

    with pytest.raises(ValueError,
                       match="Regularization must be a pyfar.Signal"
                       " object."):
        assert pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(1)


@pytest.mark.parametrize(("beta", "expected"), [(1, 0.5), (0.5, 2/3), (0, 1)])
def test_regularization_frequency_range(impulse, beta, expected):
    """Test Regularization from a frequency range using different beta
    values."""
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
        (200, 10e3))
    inv = Regu.invert(impulse, beta=beta)

    idx = inv.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(inv.freq[:, idx[0]:idx[1]], 1)
    npt.assert_allclose(inv.freq[:, 0], expected)
    npt.assert_allclose(inv.freq[:, -1], expected)


@pytest.mark.parametrize(("norm", "expected"), [(None, 0.4),
                                                ('max', 0.25)])
def test_regularization_normalization(impulse, norm, expected):
    """Test normalization parameter of Regularization."""
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
        (200, 10e3))
    inv = Regu.invert(impulse*2, normalize_regularization=norm)

    idx = inv.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(inv.freq[:, idx[0]:idx[1]], 0.5)
    npt.assert_allclose(inv.freq[:, 0], expected)
    npt.assert_allclose(inv.freq[:, -1], expected)


def test_regularization_normalization_factor(impulse):
    """Test normalization factor of caclulated during inversion."""
    impulse = impulse*2
    Regu = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
        (200, 10e3))
    regu = Regu.regularization(impulse)

    assert Regu.normalization_factor(impulse, 'max') == \
        np.max(np.abs(impulse.freq)) / np.max(np.abs(regu.freq))

    assert Regu.normalization_factor(impulse, 'mean') == \
        np.mean(np.abs(impulse.freq)) / np.mean(np.abs(regu.freq))


def test_regularization_target(impulse):
    """Test Regularization from a frequency range using a bandpass filter
    target function."""
    bp = pf.dsp.filter.butterworth(None, 4, (20, 15.5e3), 'bandpass',
                                   impulse.sampling_rate)
    target = bp.process(impulse)

    ReguTarget = pf.dsp.RegularizedSpectrumInversion.from_frequency_range(
        (20, 15.5e3))

    inv = ReguTarget.invert(impulse,  target=target, beta=1,
                            normalize_regularization=None)
    idx = inv.find_nearest_frequency([20, 15.5e3])

    npt.assert_allclose(np.abs(inv.freq[0, idx]), np.abs(target.freq[0, idx]))


def test_regularization_from_magnitude_spectrum(noise):
    """Test Regularization from an arbitrary signal."""
    Regu = pf.dsp.RegularizedSpectrumInversion.from_magnitude_spectrum(noise)
    regu = Regu.regularization(pf.signals.impulse(noise.n_samples))

    npt.assert_allclose(regu.freq, np.abs(noise.freq))
