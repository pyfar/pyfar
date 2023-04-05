import pyfar as pf
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize("signal", [
    pf.Signal([1, 2, 3], 44100),
    pf.TimeData([1, 2, 3], [1, 2, 4]),
    pf.FrequencyData([1, 2, 3], [1, 2, 4])])
def test_broadcast_cshape(signal):
    """Test broadcasting for all audio classes"""

    broadcasted = pf.utils.broadcast_cshape(signal, (2, 3))
    assert signal.cshape == (1, )
    assert broadcasted.cshape == (2, 3)
    assert isinstance(broadcasted, type(signal))


def test_broadcast_cshape_assertions():
    """Test assertions"""

    # invalid input type
    with pytest.raises(TypeError, match="Input data must be a pyfar audio"):
        pf.utils.broadcast_cshape([1, 2, 3], (1, ))


@pytest.mark.parametrize("cshape,reference", [
    (None, (2,)), ((2, 2), (2, 2))])
def test_broadcast_cshapes(cshape, reference):
    """Test broadcasting multiple signals with all audio classes"""

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
    """Test assertions"""

    # invalid input type
    with pytest.raises(TypeError, match="All input data must be pyfar"):
        pf.utils.broadcast_cshapes([1, 2, 3], (1, ))


@pytest.mark.parametrize("signal", [
    pf.Signal([1, 2, 3], 44100),
    pf.TimeData([1, 2, 3], [1, 2, 4]),
    pf.FrequencyData([1, 2, 3], [1, 2, 4])])
def test_broadcast_cdim(signal):
    """Test broadcast cdim for all audio classes"""

    broadcasted = pf.utils.broadcast_cdim(signal, 2)

    assert signal.cshape == (1, )
    assert broadcasted.cshape == (1, 1)
    assert isinstance(broadcasted, type(signal))


def test_broadcast_cdim_assertions():
    """Test assertions"""

    # invalid input type
    with pytest.raises(TypeError, match="Input data must be a pyfar audio"):
        pf.utils.broadcast_cdim([1, 2, 3], 2)

    # invalid cdim
    with pytest.raises(ValueError, match="Can not broadcast:"):
        pf.utils.broadcast_cdim(pf.Signal(1, 44100), 0)


@pytest.mark.parametrize("cdim,reference", [
    (None, 2), (3, 3)])
def test_broadcast_cdims(cdim, reference):
    """Test broadcasting multiple signals with all audio classes"""

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
    (np.random.randn(512)),
    (np.random.randn(2, 512))])
def test_concatenate(data_second):
    # Parametrized concatenation test for single- and multichannel signal
    sr = 48e3

    # Get random data with 512 samples
    data_first = np.random.randn(512)
    signals = (pf.Signal(data_first, sr), pf.Signal(data_second, sr))

    res = pf.utils.concatenate(signals, caxis=0)
    ideal = np.concatenate(
        (np.atleast_2d(data_first), np.atleast_2d(data_second)), axis=0)

    npt.assert_allclose(res._data, ideal)


def test_broadcasting_in_concatenate():
    # Test broadcasting into largest dimension and shape, while ignoring caxis.
    sr = 48e3
    n_samples = 512
    signals = (pf.Signal(np.ones((1, 2, 3) + (n_samples, )), sr),
               pf.Signal(np.ones((2,) + (n_samples, )), sr),
               pf.Signal(np.ones((1, 1, 2) + (n_samples, )), sr))
    conc = pf.utils.concatenate(signals, caxis=-1, broadcasting=True)
    assert conc.cshape == (1, 2, 7)


def test_concatenate_comments():
    # Test for comment concatenation
    sr = 48e3
    n_samples = 512
    signals = (pf.Signal(np.ones((1, 2) + (n_samples, )), sr, comment="one"),
               pf.Signal(np.ones((1, 2) + (n_samples, )), sr),
               pf.Signal(np.ones((1, 2) + (n_samples, )), sr, comment="three"))
    conc = pf.utils.concatenate(signals, caxis=-1, comment="Conc Signal")
    assert conc.comment == "Conc Signal\n"\
                           "Signals concatenated in caxis=-1.\n"\
                           "Channel 1-2: one\n"\
                           "Channel 5-6: three\n"


def test_concatenate_assertions():
    """Test assertions"""
    with pytest.raises(TypeError, match="All input data must be"):
        pf.utils.concatenate(([1, 2], [3, 4]))
    signals = (pf.Signal(np.ones((1, 2, 512)), 44100),
               pf.Signal(np.ones((1, 1, 512)), 44100))
    with pytest.raises(TypeError, match="'comment' needs to be a string."):
        pf.utils.concatenate(signals, caxis=-1, comment=1)
    with pytest.raises(TypeError, match="'broadcasting' needs to be"):
        pf.utils.concatenate(signals, caxis=-1, broadcasting=1)
