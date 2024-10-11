import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt

# test errors
def test_regularization_errors():
    with pytest.raises(RuntimeError,
                       match="Regularization objects must be created using a"
                       " classmethod."):
        assert pf.Regularization()

    with pytest.raises(ValueError,
                       match="The frequency range needs to specify lower and"
                        " upper limits."):
        assert pf.Regularization.from_frequency_range((200,))

    with pytest.raises(ValueError,
                       match="Target function must be a pyfar.Signal object."):
        assert pf.Regularization.from_frequency_range((200, 20e3), target=1)

    with pytest.raises(ValueError,
                       match="Regularization must be a pyfar.Signal object."):
        assert pf.Regularization.from_signal(1)


# test frequency_range regularization
def test_regularization_frequency_range(impulse):
    Regu = pf.Regularization.from_frequency_range((200, 10e3))
    regu = Regu.get_regularization(impulse)

    idx = regu.find_nearest_frequency([200, 10e3])

    npt.assert_allclose(regu.freq[:, idx[0]:idx[1]], 0)
    npt.assert_allclose(regu.freq[:, 0], 1)
    npt.assert_allclose(regu.freq[:, -1], 1)


# test beta
def test_regularization_beta(impulse):
    Regu = pf.Regularization.from_frequency_range((200, 20e3), beta=0)
    inv = Regu.invert(impulse)

    npt.assert_allclose(inv.freq, 1/impulse.freq)

# test target function

# test inversion
