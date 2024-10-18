import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


def test_regularization_errors():
    """Test errors"""
    with pytest.raises(RuntimeError,
                       match="Regularization objects must be created using a"
                       " classmethod."):
        assert pf.dsp.Regularization()

    with pytest.raises(ValueError,
                       match="The frequency range needs to specify lower and"
                        " upper limits."):
        assert pf.dsp.Regularization.from_frequency_range((200,))

    with pytest.raises(ValueError,
                       match="Target function must be a pyfar.Signal object."):
        assert pf.dsp.Regularization.from_frequency_range((200, 20e3),
                                                          target=1)

    with pytest.raises(ValueError,
                       match="Regularization must be a pyfar.Signal"
                       " object."):
        assert pf.dsp.Regularization.from_signal(1)


@pytest.mark.parametrize(("beta", "expected"), [(1, 0.5), (0.5, 2/3), (0, 1)])
def test_regularization_frequency_range(impulse, beta, expected):
    """Test Regularization from a frequency range using different beta
    values."""
    Regu = pf.dsp.Regularization.from_frequency_range((200, 10e3), beta=beta)
    inv = Regu.invert(impulse)

    idx = inv.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(inv.freq[:, idx[0]:idx[1]], 1)
    npt.assert_allclose(inv.freq[:, 0], expected)
    npt.assert_allclose(inv.freq[:, -1], expected)


def test_regularization_compare_to_dsp_function(impulse):
    """Compare result to dsp.regularized_spectrum_inversion."""
    res_dsp = pf.dsp.regularized_spectrum_inversion(impulse * 2, [200, 10e3])
    Regu = pf.dsp.Regularization.from_frequency_range((200, 10e3))
    res_regu = Regu.invert(impulse * 2)

    npt.assert_allclose(res_dsp.freq, res_regu.freq)


def test_regularization_target(impulse):
    """Test Regularization from a frequency range using a bandpass filter
    target function."""
    bp = pf.dsp.filter.butterworth(None, 4, (20, 15.5e3), 'bandpass',
                                   impulse.sampling_rate)
    target = bp.process(impulse)

    ReguTarget = pf.dsp.Regularization.from_frequency_range((20, 15.5e3),
                                                        target=target,
                                                        beta=1)

    inv = ReguTarget.invert(impulse)
    idx = inv.find_nearest_frequency([20, 15.5e3])

    npt.assert_allclose(np.abs(inv.freq[0, idx]), np.abs(target.freq[0, idx]))


def test_regularization_from_signal(noise):
    """Test Regularization from an arbitrary signal."""
    Regu = pf.dsp.Regularization.from_signal(noise)
    regu = Regu.get_regularization(pf.signals.impulse(noise.n_samples))

    npt.assert_allclose(regu.freq, np.abs(noise.freq))
