import pyfar
import numpy as np
import pytest


def test_pad_zeros():
    num_zeros = 100
    n_samples = 1024

    test_signal = pyfar.signals.impulse(
        n_samples, delay=0, amplitude=np.ones((2, 3)), sampling_rate=44100)

    with pytest.raises(ValueError, match="Unknown padding mode"):
        pyfar.dsp.pad_zeros(test_signal, 1, mode='invalid')

    padded = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='before')
    assert test_signal.cshape == padded.cshape
    assert test_signal.n_samples + num_zeros == padded.n_samples

    desired = pyfar.signals.impulse(
        n_samples + num_zeros,
        delay=num_zeros, amplitude=np.ones((2, 3)), sampling_rate=44100)

    np.testing.assert_allclose(padded.time, desired.time)

    padded = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='after')
    assert test_signal.cshape == padded.cshape
    assert test_signal.n_samples + num_zeros == padded.n_samples

    desired = pyfar.signals.impulse(
        n_samples + num_zeros,
        delay=0, amplitude=np.ones((2, 3)), sampling_rate=44100)

    np.testing.assert_allclose(padded.time, desired.time)

    test_signal.time = np.ones_like(test_signal.time)
    padded = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='center')
    assert test_signal.cshape == padded.cshape
    assert test_signal.n_samples + num_zeros == padded.n_samples

    desired = np.concatenate(
        (
            np.ones((2, 3, int(1024/2))),
            np.zeros((2, 3, num_zeros)),
            np.ones((2, 3, int(1024/2)))),
        axis=-1)

    np.testing.assert_allclose(padded.time, desired)
