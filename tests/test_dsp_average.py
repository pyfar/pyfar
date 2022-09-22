import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize('signal, mode, answer', (
    [pf.Signal([[1, 2, 3], [4, 5, 6]], 44100), 'time', [2.5, 3.5, 4.5]],
    [pf.FrequencyData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100), 'complex',
     [2.5, 3.5, 4.5]],
    [pf.FrequencyData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100),
     'magnitude_zerophase', [2.5, 3.5, 4.5]],
    [pf.FrequencyData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100),
     'magnitude_phase', [2.5, 3.5, 4.5]],
    [pf.FrequencyData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100),
     'power', np.sqrt([0.5**2+2**2, 1**2+2.5**2, 1.5**2+3**2])],
    [pf.FrequencyData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100),
     'log_magnitude_zerophase', [1, 2.5, 4.5]]
    ))
def test_averaging(signal, mode, answer):
    """
    Parametrized test for averaging data in all modi.
    """
    ave_sig = pf.dsp.average(signal, mode)
    if mode == 'time':
        npt.assert_equal(ave_sig.time[0], answer)
    else:
        npt.assert_almost_equal(ave_sig.freq[0], answer, decimal=15)


@pytest.mark.parametrize('axis, answer', (
    [(0, 1), [[(1+3+5+7)/4, (2+4+6+8)/4]]],
    [1, [[(1+3)/2, (2+4)/2], [(5+7)/2, (6+8)/2]]]
    ))
def test_axis_averaging(axis, answer):
    """
    Parametrized test for averaging along axis
    """
    signal = pf.Signal(np.arange(1, 9).reshape(2, 2, 2), 44100)
    ave_sig = pf.dsp.average(signal, axis=axis)
    npt.assert_equal(ave_sig.time, answer)


def test_weighted_averaging():
    """Tests averaging Signal with weighted channels """
    signal = pf.Signal([[1, 2, 3], [4, 5, 6]], 44100)
    ave_sig = pf.dsp.average(signal, weights=(0.8, 0.2))
    answer = [[1*0.8+4*0.2, 2*0.8+5*0.2, 3*0.8+6*0.2]]
    npt.assert_almost_equal(ave_sig.time, answer, decimal=15)


def test_keepdims_parameters():
    """Test keepdims parameter"""
    signal = pf.Signal(np.arange(1, 9).reshape(2, 2, 2), 44100)
    ave1 = pf.dsp.average(signal)
    ave2 = pf.dsp.average(signal, keepdims=True)
    assert len(signal.cshape) != len(ave1.cshape)
    assert len(signal.cshape) == len(ave2.cshape)


def test_error_raises():
    # test wrong signal type
    with pytest.raises(TypeError, match='Input data has to be of type'):
        pf.dsp.average([1, 2, 3])
    # test wrong mode for signal types
    signal = pf.FrequencyData([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError,
                       match="mode is 'time' and signal is type"):
        pf.dsp.average(signal, 'time')
    signal = pf.TimeData([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError,
                       match="mode is 'complex' and signal is type"):
        pf.dsp.average(signal, 'complex')
    # test wrong axis input
    signal = pf.Signal(np.ones((2, 3, 4)), 44100)
    with pytest.raises(ValueError,
                       match="The maximum of axis needs to be smaller"):
        pf.dsp.average(signal, axis=(0, 3))
    # test invalid mode input
    with pytest.raises(ValueError,
                       match="mode must be 'time', 'complex',"):
        pf.dsp.average(signal, mode='invalid_mode')
    with pytest.warns(UserWarning,
                      match="Averaging one dimensional axis"):
        pf.dsp.average(pf.Signal(np.zeros((5, 2, 1, 1)), 44100), axis=(1, 2))
