import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


def test_averaging_time():
    signal = pf.Signal([[1, 2, 3], [4, 5, 6]], 44100)
    ave_sig = pf.dsp.average(signal)
    answer = [(1+4)/2, (2+5)/2, (3+6)/2]
    npt.assert_equal(ave_sig.time[0], answer)


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
