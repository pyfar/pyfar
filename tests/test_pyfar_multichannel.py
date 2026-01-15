from pyfar.classes._pyfar_multichannel import _PyfarMultichannel
import numpy as np
import pytest
from unittest.mock import patch
import numpy.testing as npt

@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_cshape():
    """Test the cshape property."""
    data = np.array([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.cshape == (2,2)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_cdim():
    """Test the cdim property."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.cdim == (1)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_csize():
    """Test the csize property."""
    data = np.array([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.csize == 4


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
@pytest.mark.parametrize('cshape', [(4, 1), (1, 4)])
def test_reshape(cshape):
    """Test the reshape method."""
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    instance = _PyfarMultichannel()
    instance._data = data
    reshaped = instance.reshape(cshape)
    assert reshaped.cshape == cshape


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
@pytest.mark.parametrize('cshape', [(5, 1), (2, 3)])
def test_reshape_value_error(cshape):
    """Test whether ValueError is raised for the reshape method."""
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    instance = _PyfarMultichannel()
    instance._data = data
    with pytest.raises(ValueError, match='reshape an object of cshape'):
        instance.reshape(cshape)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_flatten():
    """Test the flatten method."""
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    instance = _PyfarMultichannel()
    instance._data = data
    flatted = instance.flatten()
    assert flatted.cshape == (4,)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
@pytest.mark.parametrize('cshape', [(3, 3), (6, 3), (7, 5, 3)])
def test_broadcast_cshape(cshape):
    """Test the broadcast_cshape method."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    instance = _PyfarMultichannel()
    instance._data = data
    broadcasted = instance.broadcast_cshape(cshape)
    assert broadcasted.cshape == cshape


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
@pytest.mark.parametrize('cdim', [1, 2, 5, 9])
def test_broadcast_cdim(cdim):
    """Test the broadcast_cdim method."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    instance = _PyfarMultichannel()
    instance._data = data
    broadcasted = instance.broadcast_cdim(cdim)
    assert broadcasted.cdim == cdim


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_broadcast_cdim_value_error():
    """Test whether ValueError is raised for the broadcast_csize method."""
    data = np.array([[[[1, 2, 3]]]])
    instance = _PyfarMultichannel()
    instance._data = data
    with pytest.raises(ValueError, match='channel dimensions exceeds'):
        instance.broadcast_cdim(2)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_transpose():
    rng = np.random.default_rng()
    data = rng.random((6, 2, 5, 256))
    instance1 = _PyfarMultichannel()
    instance1._data = data
    instance2 = instance1.transpose()
    npt.assert_allclose(instance1.T._data, instance2._data)
    npt.assert_allclose(instance1._data.transpose(2, 1, 0, 3), instance2._data)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
@pytest.mark.parametrize('taxis', [(2, 0, 1), (-1, 0, -2)])
def test_transpose_positional_arguments(taxis):
    rng = np.random.default_rng()
    data = rng.random((6, 2, 5, 256))
    instance1 = _PyfarMultichannel()
    instance1._data = data
    instance2 = instance1.transpose(taxis)
    npt.assert_allclose(instance1._data.transpose(2, 0, 1, 3), instance2._data)
    instance2 = instance1.transpose(*taxis)
    npt.assert_allclose(instance1._data.transpose(2, 0, 1, 3), instance2._data)


@patch.multiple(_PyfarMultichannel, __abstractmethods__=set())
def test_transpose_errors():
    rng = np.random.default_rng()
    data = rng.random((10, 10, 256))
    instance = _PyfarMultichannel()
    instance._data = data
    with pytest.raises(ValueError, match="Number of axes must match the cdim"):
        instance.transpose(1, 0, 2, 3)
    with pytest.raises(ValueError, match="Axes must be a rearrangement of cdim"):
        instance.transpose(2, 1)
