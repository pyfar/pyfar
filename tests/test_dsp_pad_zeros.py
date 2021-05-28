import pyfar
import numpy as np
import pytest


def test_pad_zeros():
    num_zeros = 2
    n_samples = 8

    test_signal = pyfar.signals.impulse(
        n_samples, delay=0, amplitude=np.ones((2, 3)), sampling_rate=44100)

    # test error raising for invalid modes
    with pytest.raises(ValueError, match="Unknown padding mode"):
        pyfar.dsp.pad_zeros(test_signal, 1, mode='invalid')

    # test padding before start of the signal
    padded = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='before')
    # check of dimensions are maintained
    assert test_signal.cshape == padded.cshape
    # check if final number of samples after padding is correct
    assert test_signal.n_samples + num_zeros == padded.n_samples

    desired = pyfar.signals.impulse(
        n_samples + num_zeros,
        delay=num_zeros, amplitude=np.ones((2, 3)), sampling_rate=44100)

    np.testing.assert_allclose(padded.time, desired.time)

    # test padding after end of the signal
    padded = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='after')
    # check of dimensions are maintained
    assert test_signal.cshape == padded.cshape
    # check if final number of samples after padding is correct
    assert test_signal.n_samples + num_zeros == padded.n_samples

    desired = pyfar.signals.impulse(
        n_samples + num_zeros,
        delay=0, amplitude=np.ones((2, 3)), sampling_rate=44100)

    np.testing.assert_allclose(padded.time, desired.time)

    # test padding at the center of the signal
    test_signal.time = np.ones_like(test_signal.time)
    padded = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='center')
    # check of dimensions are maintained
    assert test_signal.cshape == padded.cshape
    # check if final number of samples after padding is correct
    assert test_signal.n_samples + num_zeros == padded.n_samples

    desired = np.concatenate(
        (
            np.ones((2, 3, int(n_samples/2))),
            np.zeros((2, 3, num_zeros)),
            np.ones((2, 3, int(n_samples/2)))),
        axis=-1)

    np.testing.assert_allclose(padded.time, desired)
