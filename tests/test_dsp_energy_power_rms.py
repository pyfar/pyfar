import pyfar as pf
import numpy as np
import pytest
import numpy.testing as npt


@pytest.mark.parametrize('freq, amplitude', ([1, 1],
                                             [44100/20, 2],
                                             [44100/50, 3]))
def test_sinewave(freq, amplitude):
    """
    Test the energy, power and rms of different full period Sinewaves.
    """
    n_samples = 44100
    signal = pf.signals.sine(freq, n_samples, amplitude)
    energy = pf.dsp.energy(signal)
    answer_e = n_samples/2 * amplitude**2
    power = pf.dsp.power(signal)
    answer_p = amplitude**2 / 2
    rms = pf.dsp.rms(signal)
    answer_r = np.sqrt(1/2) * amplitude
    npt.assert_almost_equal(energy, answer_e)
    npt.assert_almost_equal(power, answer_p)
    npt.assert_almost_equal(rms, answer_r)


def test_multichannel_signals():
    # Test returned values for multichannel signals
    data = np.ones((2, 3, 3, 100))
    signal = pf.Signal(data, 44100)
    assert np.all(pf.dsp.energy(signal) == np.ones((2, 3, 3))*100)
    assert np.all(pf.dsp.power(signal) == np.ones((2, 3, 3)))
    assert np.all(pf.dsp.rms(signal) == np.sqrt(np.ones((2, 3, 3))))
