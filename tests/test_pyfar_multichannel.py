from pyfar.classes._pyfar_multichannel import _PyfarMultichannel
import numpy as np
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def _disable_abstract_methods():
    """Automatically disable abstract methods."""
    with patch.multiple(_PyfarMultichannel,__abstractmethods__=set()):
        yield

def test_cshape():
    """Test the cshape property."""
    data = np.array([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.cshape == (2,2)

def test_cdim():
    """Test the cdim property."""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.cdim == (1)

def test_csize():
    """Test the csize property."""
    data = np.array([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.csize == 4

@pytest.mark.parametrize('cshape', [(4, 1), (1, 4)])
def test_reshape(cshape):
    """Test the reshape method."""
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    instance = _PyfarMultichannel()
    instance._data = data
    reshaped = instance.reshape(cshape)
    assert reshaped.cshape == cshape

@pytest.mark.parametrize('cshape', [(5, 1), (2, 3)])
def test_reshape_value_error(cshape):
    """Test whether ValueError is raised for the reshape method."""
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    instance = _PyfarMultichannel()
    instance._data = data
    with pytest.raises(ValueError, match='reshape an object of cshape'):
        instance.reshape(cshape)

def test_flatten():
    """Test the flatten method."""
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    instance = _PyfarMultichannel()
    instance._data = data
    flatted = instance.flatten()
    assert flatted.cshape == (4,)

@pytest.mark.parametrize('cshape', [(3, 3), (6, 3), (7, 5, 3)])
def test_broadcast_cshape(cshape):
    """Test the broadcast_cshape method."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    instance = _PyfarMultichannel()
    instance._data = data
    broadcasted = instance.broadcast_cshape(cshape)
    assert broadcasted.cshape == cshape

@pytest.mark.parametrize('cdim', [1, 2, 5, 9])
def test_broadcast_cdim(cdim):
    """Test the broadcast_csize method."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    instance = _PyfarMultichannel()
    instance._data = data
    broadcasted = instance.broadcast_cdim(cdim)
    assert broadcasted.cdim == cdim

def test_broadcast_cdim_value_error():
    """Test whether ValueError is raised for the broadcast_csize method."""
    data = np.array([[[[1, 2, 3]]]])
    instance = _PyfarMultichannel()
    instance._data = data
    with pytest.raises(ValueError, match='channel dimensions exceeds'):
        instance.broadcast_cdim(2)
