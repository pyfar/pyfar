import pyfar as pf
from pyfar.dsp import spectrogram
import pytest
from pytest import raises
import numpy as np
import numpy.testing as npt


def test_assertions(sine):
    """Test assertions due to wrong input data"""

    with raises(TypeError, match="Input data has to be of type: Signal."):
        spectrogram([1, 2, 3])

    with raises(ValueError, match="window_length exceeds signal length"):
        spectrogram(pf.Signal(1, 44100))

    with raises(TypeError, match="The normalize parameter"):
        spectrogram(sine, normalize=1)


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
    npt.assert_allclose(spectro[:256, 1], 0, atol=1e-13)
    npt.assert_allclose(spectro[256, 1], 1, atol=1e-13)
    npt.assert_allclose(spectro[257:, 1], 0, atol=1e-13)


@pytest.mark.parametrize('window,value', [
    ('rect', [0, 1, 0]),         # rect window does not spread energy
    ('hann', [.5, 1, .5])])      # hann window spreads energy
def test_window(window, value):
    """Test return values of the spectrogram with default parameters"""
    # test signal and spectrogram
    signal = pf.signals.sine(256, 2*1024, sampling_rate=1024)
    signal.fft_norm = 'amplitude'
    _, _, spectro = pf.dsp.spectrogram(signal, window=window)

    # check middle slice
    npt.assert_allclose(spectro[255:258, 1], value, atol=1e-13)


def test_normalize(sine):
    """Test normalize parameter"""
    sine.fft_norm = 'amplitude'
    assert pf.dsp.spectrogram(sine)[-1].max() < 1
    assert pf.dsp.spectrogram(sine, normalize=False)[-1].max() > 1
