import pyfar as pf
import pytest
from pytest import raises
import numpy.testing as npt
import numpy as np


@pytest.mark.parametrize('operation,channel_handling,truth', (
                        ['max', 'max', [0.2, 1.0]],
                        ['max', 'each', [1.0, 1.0]],
                        ['max', 'min', [1.0, 5.0]],
                        ['max', 'mean', [0.333, 1.666]],
                        ['mean', 'max', [0.6, 3.0]],
                        ['mean', 'each', [3.0, 3.0]],
                        ['mean', 'min', [3.0, 15.0]],
                        ['mean', 'mean', [1.0, 5.0]],
                        ['rms', 'max', [np.sqrt(3.0)/5, np.sqrt(3.0)]],
                        ['rms', 'each', [np.sqrt(3.0), np.sqrt(3.0)]],
                        ['rms', 'min', [np.sqrt(3.0), np.sqrt(3.0)*5]],
                        ['rms', 'mean', [np.sqrt(3.0)*0.3334,
                                         np.sqrt(3.0)*1.6667]]
                        ))
def test_parametrized_normalization(operation, channel_handling, truth):
    """Parametrized test for all combinations of operation and channel_handling
    parameters using an impulse.
    """
    signal = pf.signals.impulse(3, amplitude=[1, 5])
    answer = pf.dsp.normalize(signal, domain='time', operation=operation,
                              channel_handling=channel_handling)
    npt.assert_allclose(answer.time[..., 0], truth, rtol=1e-02)


@pytest.mark.parametrize('domain,truth', (
                        ['time', [0.25, 1.0, 0.25]],
                        ['freq', [0.25, 1.0, 0.25]]
                        ))
def test_parametrized_domains_normalization(domain, truth):
    """Parametrized test for normalization in time and frequency domain."""
    signal = pf.Signal([1, 4, 1], 44100)
    answer = pf.dsp.normalize(signal, domain=domain)
    if domain == 'time':
        npt.assert_allclose(abs(answer.time[0]), truth)
    else:
        npt.assert_allclose(abs(answer.freq[0]), truth)


def test_dB_normalization():
    # Test normalizatin in dB
    signal = pf.Signal([1, 2, 3], 44100)
    answer = pf.dsp.normalize(signal, dB=True)
    truth = [[1/3, 2/3, 1]]
    npt.assert_allclose(answer.time, truth)


def test_power_normalization():
    # Test normalizatin in dB
    signal = pf.Signal([1, 2, 3], 44100)
    answer = pf.dsp.normalize(signal, power=True)
    truth = [[1/9, 2/9, 3/9]]
    npt.assert_allclose(answer.time, truth)


def test_normalization_magnitude_mean_min_freqrange():
    """Test the function along frequency domain, mean, min & target path."""
    signal = pf.Signal([[1, 4, 1], [1, 10, 1]], 44100, n_samples=4,
                       domain='freq')
    truth = pf.Signal([[2.5, 10, 2.5], [2.5, 25, 2.5]], 44100, n_samples=4,
                      domain='freq')
    answer = pf.dsp.normalize(signal, domain='freq',
                              operation='mean', channel_handling='min',
                              target=10)
    assert answer == truth


def test_value_cshape_broadcasting():
    """Test broadcasting of value with signal.cshape = (3,) to
    signal.cshape = (2,3)."""
    signal = pf.Signal([[[1, 2, 1], [1, 4, 1], [1, 2, 1]],
                        [[1, 2, 1], [1, 4, 1], [1, 2, 1]]], 44100)
    answer = pf.dsp.normalize(signal, domain='time', target=[1, 2, 3])
    assert signal.cshape == answer.cshape


def test_value_return():
    """Test the parameter return_values = True, which returns the values_norm
    data."""
    n_samples, amplitude = 3., 1.
    signal = pf.signals.impulse(n_samples, amplitude=amplitude)
    _, values_norm = pf.dsp.normalize(signal, return_reference=True,
                                      operation='mean')
    assert values_norm == amplitude / n_samples


def test_error_raises():
    """Test normalize function errors"""
    with raises(TypeError, match=("Input data has to be of type 'Signal', "
                                  "'TimeData' or 'FrequencyData'.")):
        pf.dsp.normalize([0, 1, 0])

    with raises(ValueError, match=("domain is 'time' and signal is type "
                                   "'<class "
                                   "'pyfar.classes.audio.FrequencyData'>'")):
        pf.dsp.normalize(pf.FrequencyData([1, 1, 1], [100, 200, 300]),
                         domain='time')

    with raises(ValueError, match=("domain is 'freq' and signal is type "
                                   "'<class "
                                   "'pyfar.classes.audio.TimeData'>'")):
        pf.dsp.normalize(pf.TimeData([1, 1, 1], [1, 2, 3]), domain='freq')

    with raises(ValueError, match=("domain must be 'time' or 'freq'.")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), domain='invalid_domain')

    with raises(ValueError, match=("operation must be 'max', 'mean' or")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         operation='invalid_operation')

    with raises(ValueError, match=("channel_handling must be 'each', ")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         channel_handling='invalid')
    with pytest.warns(UserWarning,
                      match="power and dB are both 'True'."):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), power=True, dB=True)
