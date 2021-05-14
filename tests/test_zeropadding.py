import pyfar
import numpy as np


def test_zeropadding():
    num_zeros = 100
    n_samples = 1024

    test_signal = pyfar.signals.impulse(
        n_samples, delay=0, amplitude=np.ones((2, 3)), sampling_rate=44100)

    padded_before = pyfar.dsp.pad_zeros(test_signal, num_zeros, mode='before')
    assert test_signal.cshape == padded_before.cshape
    assert test_signal.n_samples + num_zeros == padded_before.n_samples

    desired = pyfar.signals.impulse(
        n_samples + num_zeros,
        delay=num_zeros, amplitude=np.ones((2, 3)), sampling_rate=44100)

    np.testing.assert_allclose(padded_before, desired)

    # padded_back = pyfar.dsp.pad_zeros(test_signal, num_zeros, 'after')
    # assert test_signal.cshape == padded_back.cshape
    # assert test_signal.n_samples + num_zeros == padded_back.n_samples
    # for i in range(padded_back.cshape[0]):
    #     for j in range(padded_back.cshape[1]):
    #         for sample in range(padded_back.n_samples):
    #             if sample < test_signal.n_samples:
    #                 assert padded_back[i][j][sample] == 1
    #             else:
    #                 assert padded_back[i][j][sample] == 0
