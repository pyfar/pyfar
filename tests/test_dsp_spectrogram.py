import pyfar as pf
from pyfar.dsp import spectrogram
from pytest import raises
import numpy as np
import numpy.testing as npt


def test_assertions():
    """Test assertions due to wrong input data"""

    with raises(TypeError, match="Input data has to be of type: Signal."):
        spectrogram([1, 2, 3])

    with raises(ValueError, match="window_length exceeds signal length"):
        spectrogram(pf.Signal(1, 44100))


def test_return_values():
    """Test return values of the spectrogram with default parameters"""
    # test signal and spectrogram
    signal = pf.signals.sine(256, 2*1024, sampling_rate=1024)
    signal.fft_norm = 'amplitude'
    freqs, times, spectro = pf.dsp.spectrogram(signal,  window='rect')

    # check frequencies and times
    npt.assert_allclose(freqs, np.arange(513))
    npt.assert_allclose(times, [0, 512/1024, 1])

    # check middle slice
    assert np.all(spectro[:256, 1] < -200)
    npt.assert_allclose(spectro[256, 1], 0, atol=1e-13)
    assert np.all(spectro[257:, 1] < -200)


def test_dB_False():
    """Test values of the spectrogram without dB calculation"""
    # test signal and spectrogram
    signal = pf.signals.sine(256, 2*1024, sampling_rate=1024)
    signal.fft_norm = 'amplitude'
    _, _, spectro = pf.dsp.spectrogram(signal,  window='rect', dB=False)

    # check middle slice
    npt.assert_allclose(spectro[:256, 1], 0, atol=1e-13)
    npt.assert_allclose(spectro[256, 1], 1, atol=1e-13)
    npt.assert_allclose(spectro[257:, 1], 0, atol=1e-13)
