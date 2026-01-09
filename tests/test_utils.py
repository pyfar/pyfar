import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize("signal", [
    pf.Signal([1, 2, 3], 44100),
    pf.TimeData([1, 2, 3], [1, 2, 4]),
    pf.FrequencyData([1, 2, 3], [1, 2, 4])])
def test_broadcast_cshape(signal):
    """Test broadcasting for all audio classes."""

    broadcasted = pf.utils.broadcast_cshape(signal, (2, 3))
    assert signal.cshape == (1, )
    assert broadcasted.cshape == (2, 3)
    assert isinstance(broadcasted, type(signal))


def test_broadcast_cshape_assertions():
    """Test assertions."""

    # invalid input type
    with pytest.raises(TypeError, match="Input data must be a pyfar audio"):
        pf.utils.broadcast_cshape([1, 2, 3], (1, ))


@pytest.mark.parametrize(("cshape", "reference"), [
    (None, (2,)), ((2, 2), (2, 2))])
def test_broadcast_cshapes(cshape, reference):
    """Test broadcasting multiple signals with all audio classes."""

    signals = (pf.signals.impulse(5, [0, 1]),
               pf.TimeData([1, 2, 3], [1, 2, 4]),
               pf.FrequencyData([1, 2], [2, 3]))
    cshapes = [signal.cshape for signal in signals]

    broadcasted = pf.utils.broadcast_cshapes(signals, cshape)
    for signal, broadcast, cshape in zip(signals, broadcasted, cshapes):
        assert signal.cshape == cshape
        assert broadcast.cshape == reference
        assert isinstance(broadcast, type(signal))


def test_broadcast_cshapes_assertions():
    """Test assertions."""

    # invalid input type
    with pytest.raises(TypeError, match="All input data must be pyfar"):
        pf.utils.broadcast_cshapes([1, 2, 3], (1, ))


@pytest.mark.parametrize("signal", [
    pf.Signal([1, 2, 3], 44100),
    pf.TimeData([1, 2, 3], [1, 2, 4]),
    pf.FrequencyData([1, 2, 3], [1, 2, 4])])
def test_broadcast_cdim(signal):
    """Test broadcast cdim for all audio classes."""

    broadcasted = pf.utils.broadcast_cdim(signal, 2)

    assert signal.cshape == (1, )
    assert broadcasted.cshape == (1, 1)
    assert isinstance(broadcasted, type(signal))


def test_broadcast_cdim_assertions():
    """Test assertions."""

    # invalid input type
    with pytest.raises(TypeError, match="Input data must be a pyfar audio"):
        pf.utils.broadcast_cdim([1, 2, 3], 2)

    # invalid cdim
    with pytest.raises(ValueError, match="Can not broadcast:"):
        pf.utils.broadcast_cdim(pf.Signal(1, 44100), 0)


@pytest.mark.parametrize(("cdim", "reference"), [
    (None, 2), (3, 3)])
def test_broadcast_cdims(cdim, reference):
    """Test broadcasting multiple signals with all audio classes."""

    signals = (pf.signals.impulse(5, [[0, 1], [2, 3]]),
               pf.TimeData([1, 2, 3], [1, 2, 4]))

    cdims = [len(signal.cshape) for signal in signals]

    broadcasted = pf.utils.broadcast_cdims(signals, cdim)
    for signal, broadcast, cdim in zip(signals, broadcasted, cdims):
        assert len(signal.cshape) == cdim
        assert len(broadcast.cshape) == reference
        assert isinstance(broadcast, type(signal))


def test_broadcast_cdims_assertions():

    # invalid input type
    with pytest.raises(TypeError, match="All input data must be pyfar"):
        pf.utils.broadcast_cdims([1, 2, 3])


@pytest.mark.parametrize("data_second", [
    (np.array([2, -2, 2, -2])),
    (np.array([[2, -2, 2, -2], [3, -3, 3, -3]]))])
@pytest.mark.parametrize("domains", [
    ('time', 'time'), ('time', 'freq'), ('freq', 'freq')])
def test_concatenate_channels(data_second, domains):

    # Generate signals for concatenation
    # (use rms normalization for being able to test behavior in freq. domain)
    sr = 48e3
    data_first = np.array([1, -1, 1, -1])
    signals = (pf.Signal(data_first, sr, fft_norm='rms'),
               pf.Signal(data_second, sr, fft_norm='rms'))

    # force signal domain
    for signal, domain in zip(signals, domains):
        signal.domain = domain

    res = pf.utils.concatenate_channels(signals, caxis=0)
    ideal = np.concatenate(
        (np.atleast_2d(data_first), np.atleast_2d(data_second)), axis=0)

    npt.assert_allclose(res._data, ideal)


def test_broadcasting_in_concatenate_channels():
    # Test broadcasting into largest dimension and shape, while ignoring caxis.
    sr = 48e3
    n_samples = 512
    signals = (pf.Signal(np.ones((1, 2, 3) + (n_samples, )), sr),
               pf.Signal(np.ones((2,) + (n_samples, )), sr),
               pf.Signal(np.ones((1, 1, 2) + (n_samples, )), sr))
    conc = pf.utils.concatenate_channels(signals, caxis=-1, broadcasting=True)
    assert conc.cshape == (1, 2, 7)


@pytest.mark.parametrize("signals", [
    (pf.Signal([1, 2, 3], 44100), pf.Signal([4, 5, 6], 44100)),
    (pf.FrequencyData([1, 2, 3], [1, 2, 3]),
     pf.FrequencyData([4, 5, 6], [1, 2, 3])),
    (pf.TimeData([1, 2, 3], [1, 2, 3]), pf.TimeData([4, 5, 6], [1, 2, 3]))])
def test_pyfar_object_types(signals):
    # Test concatenation for all pyfar object types
    conc = pf.utils.concatenate_channels(signals)
    ideal = [[1, 2, 3], [4, 5, 6]]
    npt.assert_equal(conc._data, ideal)


def test_concatenate_channels_complex_signals():
    # Test concanation for complex signals
    s = pf.Signal([1, 2, 3], 44100)
    s_complex = pf.Signal([1, 2, 3], 44100, is_complex=True)

    s_conc = pf.utils.concatenate_channels((s, s_complex))
    assert s_conc.complex
    desired = [[1 + 0j, 2 + 0j, 3 + 0j], [1 + 0j, 2 + 0j, 3 + 0j]]
    npt.assert_equal(s_conc._data, desired)

    s_conc = pf.utils.concatenate_channels((s_complex, s))
    assert s_conc.complex
    desired = [[1 + 0j, 2 + 0j, 3 + 0j], [1 + 0j, 2 + 0j, 3 + 0j]]
    npt.assert_equal(s_conc._data, desired)


def test_concatenate_assertions():
    """Test assertions."""
    with pytest.raises(TypeError, match="All input data must be"):
        pf.utils.concatenate_channels(([1, 2], [3, 4]))
    signals = (pf.Signal(np.ones((1, 2, 512)), 44100),
               pf.Signal(np.ones((1, 1, 512)), 44100))
    with pytest.raises(TypeError, match="'broadcasting' needs to be"):
        pf.utils.concatenate_channels(signals, caxis=-1, broadcasting=1)
    signals = (pf.Signal([1, 2, 3], 44100),
               pf.TimeData([1, 2, 3], [1, 2, 3]))
    with pytest.raises(ValueError, match="Comparison only valid against"):
        pf.utils.concatenate_channels(signals)


def test_concatenate_bins_frequencydata():
    """Test concatenate_bins function with FrequencyData objects"""
    signals = (
        pf.FrequencyData([1, 2, 3], [1, 2, 3]),
        pf.FrequencyData([4, 5, 6], [4, 5, 6]))
    merged = pf.utils.concatenate_bins(signals)
    assert isinstance(merged, pf.FrequencyData)
    npt.assert_array_equal(merged.frequencies, np.arange(1, 7))


def test_concatenate_bins_frequencydata_with_sort():
    """Test concatenate_bins function with FrequencyData objects"""
    signals = (
        pf.FrequencyData([1, 3, 5], [1, 3, 5]),
        pf.FrequencyData([2, 4, 6], [2, 4, 6]))
    merged = pf.utils.concatenate_bins(signals)
    assert isinstance(merged, pf.FrequencyData)
    npt.assert_array_equal(merged.frequencies, np.arange(1, 7))
    npt.assert_array_equal(merged.freq, np.arange(1, 7).reshape((1, 6)))


def test_concatenate_bins_multidim():
    """Test concatenate_bins function with multidimensional input"""
    signals = (
        pf.FrequencyData(np.array([[1, 2, 3], [4, 5, 6]]), [1, 2, 3]),
        pf.FrequencyData(np.array([[7, 8, 9], [10, 11, 12]]), [4, 5, 6]))
    merged = pf.utils.concatenate_bins(signals)
    assert isinstance(merged, pf.FrequencyData)
    npt.assert_array_equal(merged.frequencies, np.arange(1, 7))
    npt.assert_array_equal(merged.freq, np.array(
        [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]))


def test_concatenate_bins_multidim_with_sort():
    """Test concatenate_bins function with multidimensional
    input and unsorted frequencies"""
    signals = (
        pf.FrequencyData(np.array([[1, 3, 5], [7, 9, 11]]), [1, 3, 5]),
        pf.FrequencyData(np.array([[2, 4, 6], [8, 10, 12]]), [2, 4, 6]))
    merged = pf.utils.concatenate_bins(signals)
    assert isinstance(merged, pf.FrequencyData)
    npt.assert_array_equal(merged.frequencies, np.arange(1, 7))
    npt.assert_array_equal(merged.freq, np.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))


def test_concatenate_bins_assertions():
    """Test assertions"""
    # invalid input type
    with pytest.raises(
            TypeError,
            match="All input data must be of type pf.FrequencyData."):
        pf.utils.concatenate_bins([1, 2, 3])


def test_input_type():
    # Create some FrequencyData objects for testing
    signal1 = pf.FrequencyData(np.random.rand(10), np.arange(10))
    signal3 = "not a FrequencyData object"

    # Test that a TypeError is raised if the input is not a tuple
    with pytest.raises(TypeError):
        pf.utils.concatenate_bins(signal1)

    # Test that a TypeError is raised if the input is not a
    # tuple of FrequencyData objects
    with pytest.raises(TypeError):
        pf.utils.concatenate_bins((signal1, signal3))
