import pyfar as pf
import pytest
from pytest import raises
import numpy.testing as npt
import numpy as np


@pytest.mark.parametrize('normalize,channel_handling,truth', (
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
def test_parametrized_normalization(normalize, channel_handling, truth):
    """Parametrized test for all combinations of normalize and channel_handling
    parameters using an impulse.
    """
    signal = pf.signals.impulse(3, amplitude=[1, 5])
    answer = pf.dsp.normalize(signal, mode='time', normalize=normalize,
                              channel_handling=channel_handling)
    npt.assert_allclose(answer.time[..., 0], truth, rtol=1e-02)


@pytest.mark.parametrize('mode,truth', (
                        ['magnitude', [0.25, 1.0, 0.25]],
                        ['log_magnitude', [0.25, 1.0, 0.25]],
                        ['power', [1/16, 4/16, 1/16]]
                        ))
def test_parametrized_modes_normalization(mode, truth):
    """Parametrized test for normalization with all modi."""
    signal = pf.FrequencyData([1, 4, 1], [1, 2, 3])
    answer = pf.dsp.normalize(signal, mode=mode)
    npt.assert_allclose(abs(answer.freq[0]), truth)


def test_normalization_magnitude_mean_min_freqrange():
    """Test the function along magnitude, mean, min & value path."""
    signal = pf.Signal([[1, 4, 1], [1, 10, 1]], 44100, n_samples=4,
                       domain='freq')
    truth = pf.Signal([[2.5, 10, 2.5], [2.5, 25, 2.5]], 44100, n_samples=4,
                      domain='freq')
    answer = pf.dsp.normalize(signal, mode='magnitude',
                              normalize='mean', channel_handling='min',
                              value=10)
    assert answer == truth


def test_value_cshape_broadcasting():
    """Test broadcasting of value with signal.cshape = (3,) to 
    signal.cshape = (2,3)."""
    signal = pf.Signal([[[1, 2, 1], [1, 4, 1], [1, 2, 1]],
                        [[1, 2, 1], [1, 4, 1], [1, 2, 1]]], 44100)
    answer = pf.dsp.normalize(signal, mode='time', value=[1, 2, 3])
    assert signal.cshape == answer.cshape


def test_error_raises():
    """Test normalize function errors"""
    with raises(TypeError, match=("Input data has to be of type 'Signal', "
                                  "'TimeData' or 'FrequencyData'.")):
        pf.dsp.normalize([0, 1, 0])

    with raises(ValueError, match=("mode is 'time' and signal is type "
                                   "'<class "
                                   "'pyfar.classes.audio.FrequencyData'>'")):
        pf.dsp.normalize(pf.FrequencyData([1, 1, 1], [100, 200, 300]),
                         mode='time')

    with raises(ValueError, match=("mode is 'power' and signal is type "
                                   "'<class "
                                   "'pyfar.classes.audio.TimeData'>'")):
        pf.dsp.normalize(pf.TimeData([1, 1, 1], [1, 2, 3]), mode='power')

    with raises(ValueError, match=("mode must be 'time', 'magnitude', ")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), mode='invalid_mode')

    with raises(ValueError, match=("normalize must be 'max', 'mean' or")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         normalize='invalid_normalize')

    with raises(ValueError, match=("channel_handling must be 'each', ")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         channel_handling='invalid')
