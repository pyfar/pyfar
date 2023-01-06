import pyfar as pf
import pytest
from pytest import raises
import numpy.testing as npt
import numpy as np


@pytest.mark.parametrize('reference_method,channel_handling,truth', (
    ['max', 'max', [0.2, 1.0]],
    ['max', 'individual', [1.0, 1.0]],
    ['max', 'min', [1.0, 5.0]],
    ['max', 'mean', [1/3, 5/3]],
    ['mean', 'max', [0.6, 3.0]],
    ['mean', 'individual', [3.0, 3.0]],
    ['mean', 'min', [3.0, 15.0]],
    ['mean', 'mean', [1.0, 5.0]],
    ['rms', 'max', [1*np.sqrt(3/5**2), 5*np.sqrt(3/5**2)]],
    ['rms', 'individual', [1*np.sqrt(3/1**2), 5*np.sqrt(3/5**2)]],
    ['rms', 'min', [1*np.sqrt(3/1**2), 5*np.sqrt(3/1**2)]],
    ['rms', 'mean', [2/(np.sqrt(1/3)+np.sqrt(25/3)),
                     2*5/(np.sqrt(1/3)+np.sqrt(25/3))]],
    ['power', 'max', [1*3/5**2, 5*3/5**2]],
    ['power', 'individual', [1*3/1**2, 5*3/5**2]],
    ['power', 'min', [1*3/1**2, 5*3/1**2]],
    ['power', 'mean', [1*3/((1**2+5**2)/2), 5*3/((1**2+5**2)/2)]],
    ['energy', 'max', [1/5**2, 5/5**2]],
    ['energy', 'individual', [1/1**2, 5/5**2]],
    ['energy', 'min', [1/1**2, 5/1**2]],
    ['energy', 'mean', [1/((1**2+5**2)/2), 5/((1**2+5**2)/2)]]))
def test_normalization(reference_method, channel_handling, truth):
    """Parametrized test for all combinations of reference_method and
    channel_handling parameters using an impulse.
    """
    signal = pf.signals.impulse(3, amplitude=[1, 5])
    answer = pf.dsp.normalize(signal, domain='time',
                              reference_method=reference_method,
                              channel_handling=channel_handling)
    npt.assert_allclose(answer.time[..., 0], truth, rtol=1e-14)


def test_domains_normalization():
    """Test for normalization in time and frequency domain."""
    signal = pf.signals.noise(128, seed=7)
    time = pf.dsp.normalize(signal, domain="time")
    freq = pf.dsp.normalize(signal, domain="freq")

    npt.assert_allclose(np.max(np.abs(time.time)), 1)
    assert np.max(np.abs(time.freq)) != 1

    assert np.max(np.abs(freq.time)) != 1
    npt.assert_allclose(np.max(np.abs(freq.freq)), 1)


@pytest.mark.parametrize('unit, limit1, limit2', (
                         [None, (0, 1000), (1000, 2000)],
                         ['s', (0, 0.5), (0.5, 1)]))
def test_time_limiting(unit, limit1, limit2):
    # Test for normalization with setting limits in samples(None) and seconds.
    signal = np.append(np.ones(1000), np.zeros(1000)+0.1)
    signal = pf.Signal(signal, 2000)
    sig_norm1 = pf.dsp.normalize(signal, domain='time', target=2,
                                 limits=limit1, unit=unit)
    sig_norm2 = pf.dsp.normalize(signal, domain='time', target=2,
                                 limits=limit2, unit=unit)
    npt.assert_allclose(np.max(sig_norm1.time), np.min(sig_norm2.time))
    npt.assert_allclose(10*np.max(sig_norm1.time), np.max(sig_norm2.time))


@pytest.mark.parametrize('unit, limit1, limit2', (
                         [None, (20, 25), (5, 10)],
                         ['Hz', (400, 600), (100, 200)]))
def test_frequency_limiting(unit, limit1, limit2):
    """Test for normalization with setting limits in bins(None) and hertz."""
    signal = pf.signals.sine(500, 2000)
    sig_norm1 = pf.dsp.normalize(signal, domain='freq',
                                 limits=limit1, unit=unit)
    sig_norm2 = pf.dsp.normalize(signal, domain='freq',
                                 limits=limit2, unit=unit)
    assert np.max(sig_norm1.time) < np.max(sig_norm2.time)


def test_value_cshape_broadcasting():
    """Test broadcasting of target with shape (3,) to signal.cshape = (2,3)."""
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
                                      reference_method='mean')
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

    with raises(ValueError, match=("reference_method must be 'max', 'mean',")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         reference_method='invalid_reference_method')

    with raises(ValueError, match=("channel_handling must be 'individual', ")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         channel_handling='invalid')
    with raises(ValueError, match=("limits must be an array like")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), limits=2)
    with raises(ValueError, match=("limits must be")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), limits=(100, 200),
                         reference_method='energy')
    with raises(ValueError, match=("'Hz' is an invalid unit")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), domain='time', unit='Hz')
    with raises(ValueError, match=("Upper and lower limit are identical")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), limits=(1, 1))
