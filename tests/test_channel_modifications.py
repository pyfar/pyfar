
import pyfar
import numpy as np
import numpy.testing as npt


def test_concatenate():
    sr = 48e3
    n_samples = 512

    data_first = np.random.randn(n_samples)
    data_second = np.random.randn(n_samples)

    first = pyfar.Signal(
        data_first, sr)

    second = pyfar.Signal(
        data_second, sr)

    res = pyfar.concatenate(first, second, axis=0)
    ideal = np.concatenate(
        (np.atleast_2d(data_first), np.atleast_2d(data_second)), axis=0)

    npt.assert_allclose(res._data, ideal)


def test_concatenate_multichannel():
    sr = 48e3
    n_samples = 512

    data_first = np.random.randn(n_samples)
    data_second = np.random.randn(2, n_samples)

    first = pyfar.Signal(
        data_first, sr)

    second = pyfar.Signal(
        data_second, sr)

    res = pyfar.concatenate(first, second, axis=0)
    ideal = np.concatenate(
        (np.atleast_2d(data_first), np.atleast_2d(data_second)), axis=0)

    npt.assert_allclose(res._data, ideal)

