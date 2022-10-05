import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize('signal, mode, answer', (
    [pf.Signal([[1, 2, 3], [4, 5, 6]], 44100),
     'linear', [2.5, 3.5, 4.5]],
    [pf.TimeData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100),
     'linear', [2.5, 3.5, 4.5]],
    [pf.signals.impulse(128, [0, 2], [1, 3]),
     'magnitude_zerophase', np.zeros(65)+2],
    [pf.signals.impulse(128, [0, 2], [1, 3]),
     'magnitude_phase', pf.signals.impulse(128, 1, 2).freq[0]],
    [pf.FrequencyData([[1, 2, 3], [4, 5, 6]], [1, 2, 3], 44100),
     'power', np.sqrt([(1+16)/2, (4+25)/2, (9+36)/2])],
    [pf.FrequencyData([[0.01, 0.1, ], [1, 10]], [1, 2], 44100),
     'log_magnitude_zerophase', 10**(np.array([(-40+0)/2, (-20+20)/2])/20)]
    ))
def test_averaging(signal, mode, answer):
    """
    Parametrized test for averaging data in all modi.
    """
    ave_sig = pf.dsp.average(signal, mode)
    if mode == 'linear':
        npt.assert_equal(ave_sig.time[0], answer)
    else:
        npt.assert_almost_equal(ave_sig.freq[0], answer, decimal=15)


@pytest.mark.parametrize('caxis, answer', (
    [(0, 2), [[(1+2+5+6)/4, (3+4+7+8)/4]]],
    [1, [[(1+3)/2, (2+4)/2], [(5+7)/2, (6+8)/2]]]
    ))
def test_caxis_averaging(caxis, answer):
    """
    Parametrized test for averaging along caxis
    """
    signal = pf.Signal(np.arange(1, 9).reshape(2, 2, 2), 44100)
    ave_sig = pf.dsp.average(signal, caxis=caxis)
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
    signal = pf.TimeData([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError,
                       match="mode is 'magnitude_phase' and signal is type"):
        pf.dsp.average(signal, 'magnitude_phase')
    # test wrong caxis input
    signal = pf.Signal(np.ones((2, 3, 4)), 44100)
    with pytest.raises(ValueError,
                       match="The maximum of caxis needs to be smaller"):
        pf.dsp.average(signal, caxis=(0, 3))
    # test invalid mode input
    with pytest.raises(ValueError,
                       match="mode must be 'linear', 'magnitude_zerophase',"):
        pf.dsp.average(signal, mode='invalid_mode')
    with pytest.warns(UserWarning,
                      match="Averaging one dimensional caxis"):
        pf.dsp.average(pf.Signal(np.zeros((5, 2, 1, 1)), 44100), caxis=(1, 2))
