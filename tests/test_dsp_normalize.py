import pyfar as pf
from pytest import raises


def test_normalization_time_max_max_value():
    """Test the function along time, max, max & value path."""
    signal = pf.Signal([[1, 2, 1], [1, 4, 1]], 44100)
    truth = pf.Signal([[0.25, 0.5, 0.25], [0.25, 1., 0.25]], 44100)
    answer = pf.dsp.normalize(signal, mode='time', normalize='max',
                              channel_handling='max')
    assert answer == truth


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
