import pyfar as pf
import numpy as np
from pytest import raises
import numpy.testing as npt


def test_decibel_Signal():
    test_signal = [0.01, 0.1, 1, 10, 100]
    test_Signal = pf.Signal(test_signal, 44100)
    # Test right calculation
    # and default values log_prefix = 20 & log_reference = 1
    npt.assert_equal(pf.dsp.decibel(test_Signal, domain='time'),
                     [[-40, -20, 0, 20, 40]])
    npt.assert_equal(pf.dsp.decibel(test_Signal, domain='freq'),
                     20*np.log10(np.abs(test_Signal.freq)))
    npt.assert_equal(pf.dsp.decibel(test_Signal, domain='freq_raw'),
                     20*np.log10(np.abs(test_Signal.freq_raw)))
    # Testing overwrite log_prefix Parameter
    npt.assert_almost_equal(pf.dsp.decibel(test_Signal, domain='time',
                            log_prefix=15),
                            [[-30, -15, 0, 15, 30]], decimal=10)
    test_Signal.fft_norm = 'power'
    npt.assert_almost_equal(pf.dsp.decibel(test_Signal, domain='time'),
                            [[-20, -10, 0, 10, 20]], decimal=10)
    test_Signal.fft_norm = 'none'
    # Test using eps if data == 0
    npt.assert_equal(pf.dsp.decibel(pf.Signal([0], 44100), domain='time'),
                     20*np.log10(np.abs(np.finfo(float).eps)))
    # Test invalid domain
    with raises(ValueError, match=("Domain is 'invalid domain', but has to be "
                                   "'time', 'freq', or 'freq_raw'.")):
        pf.dsp.decibel(test_Signal, domain='invalid domain')


def test_decibel_TimeData():
    test_signal = [0.01, 0.1, 1, 10, 100]
    test_TimeData = pf.TimeData(test_signal, np.arange(len(test_signal)))
    # Test right calculation
    npt.assert_equal(pf.dsp.decibel(test_TimeData, domain='time'),
                     [[-40, -20, 0, 20, 40]])
    # Test wrong domain input
    with raises(ValueError, match="Domain is 'freq' and signal is type"):
        pf.dsp.decibel(test_TimeData, domain='freq')


def test_decibel_FrequencyData():
    test_signal = [0.01, 0.1, 1, 10, 100]
    test_FreqData = pf.FrequencyData(test_signal, [1, 10, 100, 1000, 10000])
    # Test right calculation
    npt.assert_equal(pf.dsp.decibel(test_FreqData, domain='freq'),
                     [[-40, -20, 0, 20, 40]])
    # Test wrong domain input
    with raises(ValueError, match="Domain is 'time' and signal is type"):
        pf.dsp.decibel(test_FreqData, domain='time')


def test_return_prefix():
    test_signal = pf.Signal([0, 1, 0], sampling_rate=44100, fft_norm='power')
    dB_sig, prefix = pf.dsp.decibel(test_signal, return_prefix=True)
    assert prefix == 10
