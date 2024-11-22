import pyfar as pf
import pytest
import numpy.testing as npt
import numpy as np


@pytest.mark.parametrize(('reference_method', 'channel_handling', 'truth'), [
    ('max', 'max', [0.2, 1.0]),
    ('max', 'individual', [1.0, 1.0]),
    ('max', 'min', [1.0, 5.0]),
    ('max', 'mean', [1/3, 5/3]),
    ('mean', 'max', [0.6, 3.0]),
    ('mean', 'individual', [3.0, 3.0]),
    ('mean', 'min', [3.0, 15.0]),
    ('mean', 'mean', [1.0, 5.0])])
def test_normalization_max_mean(reference_method, channel_handling, truth):
    """
    Parametrized test for all combinations of 'max' and 'mean'
    reference_method and channel_handling parameters using impulse signals.
    """
    signal = pf.signals.impulse(3, amplitude=[1, 5])
    answer = pf.dsp.normalize(signal, domain='time',
                              reference_method=reference_method,
                              channel_handling=channel_handling)
    npt.assert_allclose(answer.time[..., 0], truth, rtol=1e-14)


@pytest.mark.parametrize(('reference_method', 'reference_function'), [
    ('energy', pf.dsp.energy), ('power', pf.dsp.power), ('rms', pf.dsp.rms)])
@pytest.mark.parametrize('target', [.5, 1, 2])
@pytest.mark.parametrize(('channel_handling', 'assert_function'), [
    ('max', np.max),
    ('individual', np.array),
    ('min', np.min),
    ('mean', np.mean)])
def test_normalization_energy_power_rms(
    reference_method, reference_function, target, channel_handling,
    assert_function):
    """
    Parametrized test for all combinations of 'energy', 'power' and 'rms'
    reference_method and channel_handling parameters using impulse signals.
    """
    signal = pf.signals.impulse(3, amplitude=[1, 5])
    normalized = pf.dsp.normalize(signal, domain='time',
                              reference_method=reference_method,
                              channel_handling=channel_handling,
                              target=target)
    value = assert_function(reference_function(normalized))
    npt.assert_almost_equal(value, target * np.ones(value.shape), 10)


@pytest.mark.parametrize('is_complex', [False, True])
def test_domains_normalization(is_complex):
    """Test for normalization in time and frequency domain."""
    signal = pf.signals.noise(128, seed=7)
    signal.fft_norm = "none"
    signal.complex = is_complex
    time = pf.dsp.normalize(signal, domain="time")
    freq = pf.dsp.normalize(signal, domain="freq")

    npt.assert_allclose(np.max(np.abs(time.time)), 1)
    assert np.max(np.abs(time.freq)) != 1

    assert np.max(np.abs(freq.time)) != 1
    npt.assert_allclose(np.max(np.abs(freq.freq)), 1)


def test_auto_domain_normalization():
    """Test for normalization with auto domain."""
    signal = pf.Signal([1, 3, 0], 10)
    time_data = pf.TimeData([1, 3, 0], [0, 1, 4])
    freq_data = pf.FrequencyData([2, 0, 1.5], [10, 100, 1000])

    assert pf.dsp.normalize(signal) == pf.dsp.normalize(signal, domain='time')
    assert pf.dsp.normalize(time_data) == pf.dsp.normalize(time_data,
                                                           domain='time')
    assert pf.dsp.normalize(freq_data) == pf.dsp.normalize(freq_data,
                                                           domain='freq')


@pytest.mark.parametrize(('unit', 'limit1', 'limit2'), [
                         (None, (0, 1000), (1000, 2000)),
                         ('s', (0, 0.5), (0.5, 1))])
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


@pytest.mark.parametrize(('unit', 'limit1', 'limit2'), [
                         (None, (20, 25), (5, 10)),
                         ('Hz', (400, 600), (100, 200))])
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
    data.
    """
    n_samples, amplitude = 3., 1.
    signal = pf.signals.impulse(n_samples, amplitude=amplitude)
    _, values_norm = pf.dsp.normalize(signal, return_reference=True,
                                      reference_method='mean')
    assert values_norm == amplitude / n_samples


@pytest.mark.parametrize('data', [
                        pf.TimeData([1, np.nan, 2], [1, 2, 3]),
                        pf.FrequencyData([1, np.nan, 2], [1, 2, 3])])
def test_nan_value_normalization(data):
    # Test normalization with data including NaNs.
    if data.domain == 'time':
        norm_prop = pf.dsp.normalize(data, nan_policy='propagate')
        npt.assert_equal(norm_prop.time[0], [np.nan, np.nan, np.nan])
        norm_omit = pf.dsp.normalize(data, nan_policy='omit')
        npt.assert_equal(norm_omit.time[0], [0.5, np.nan, 1.0])
    else:
        norm_prop = pf.dsp.normalize(data, domain='freq',
                                     nan_policy='propagate')
        npt.assert_equal(norm_prop.freq[0], [np.nan, np.nan, np.nan])
        norm_omit = pf.dsp.normalize(data, domain='freq', nan_policy='omit')
        npt.assert_equal(norm_omit.freq[0], [0.5, np.nan, 1.0])


def test_error_raises():
    """Test normalize function errors."""
    with pytest.raises(
        TypeError, match=("Input data has to be of type 'Signal', "
                          "'TimeData' or 'FrequencyData'.")):
        pf.dsp.normalize([0, 1, 0])

    with pytest.raises(
            ValueError, match=("domain is 'time' and signal is type "
                               "'<class "
                               "'pyfar.classes.audio.FrequencyData'>'")):
        pf.dsp.normalize(pf.FrequencyData([1, 1, 1], [100, 200, 300]),
                         domain='time')

    with pytest.raises(
            ValueError, match=("domain is 'freq' and signal is type "
                               "'<class "
                               "'pyfar.classes.audio.TimeData'>'")):
        pf.dsp.normalize(pf.TimeData([1, 1, 1], [1, 2, 3]), domain='freq')

    with pytest.raises(ValueError, match=(
            "domain must be 'time', 'freq' or 'auto' "
            "but is 'invalid_domain'.")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), domain='invalid_domain')

    with pytest.raises(
            ValueError, match=("reference_method must be 'max', 'mean',")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         reference_method='invalid_reference_method')

    with pytest.raises(
            ValueError, match=("channel_handling must be 'individual', ")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100),
                         channel_handling='invalid')
    with pytest.raises(ValueError, match=("limits must be an array like")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), limits=2)
    with pytest.raises(ValueError, match=("limits must be")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), limits=(100, 200),
                         reference_method='energy')
    with pytest.raises(ValueError, match=("'Hz' is an invalid unit")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), domain='time', unit='Hz')
    with pytest.raises(
            ValueError, match=("Upper and lower limit are identical")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), limits=(1, 1))
    with pytest.raises(
            ValueError, match=("nan_policy has to be 'propagate',")):
        pf.dsp.normalize(pf.Signal([0, 1, 0], 44100), nan_policy='invalid')
    with pytest.raises(ValueError, match=("The signal includes NaNs.")):
        pf.dsp.normalize(pf.TimeData([0, np.nan, 0], [0, 1, 3]),
                         nan_policy='raise')


@pytest.mark.parametrize('reference_method', ['energy', 'power', 'rms'])
@pytest.mark.parametrize('input_signal', [pf.TimeData([1, 1, 1],
                                                      [0, 1, 2],
                                                      is_complex=True),
                                          pf.Signal([1, 0, 1],
                                                    sampling_rate=48000,
                                                    is_complex=True)])
def test_invalid_modes_complex(reference_method, input_signal):
    """Parametrized test for all combinations of reference_method and
    channel_handling parameters using an impulse.
    """
    with pytest.raises(
            ValueError, match=("'energy', 'power', and 'rms' reference "
            "method is not implemented for complex "
            "time signals.")):
        pf.dsp.normalize(input_signal, reference_method=reference_method)
