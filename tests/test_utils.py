import pyfar as pf
import pytest


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
    with pytest.raises(TypeError, match="All input data must be pyfar"):
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
    with pytest.raises(TypeError, match="All input data must be pyfar"):
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
