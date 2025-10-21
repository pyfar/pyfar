from pyfar.classes._pyfar_multichannel import _PyfarMultichannel
import numpy as np
import pytest
from copy import deepcopy
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def _disable_abstract_methods():
    """Automatically disable abstract methods and provide fake copy()
    for tests.
    """
    with patch.multiple(_PyfarMultichannel,__abstractmethods__=set()):
            with patch.object(_PyfarMultichannel, 'copy', side_effect=lambda self: deepcopy(self)):
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
    """Test the cshape property."""
    data = np.array([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    instance = _PyfarMultichannel()
    instance._data = data
    assert instance.csize == 4

@pytest.mark.parametrize('cshape', [(3, 3), (6, 3), (7, 5, 3)])

def test_broadcast_cshape(cshape):
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    instance = _PyfarMultichannel()
    instance._data = data
    broadcasted = instance.broadcast_cshape(cshape)
    assert broadcasted.cshape == cshape

@pytest.mark.parametrize('cdim', [2, 5, 9])

def test_broadcast_cdim(cdim):
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    instance = _PyfarMultichannel()
    instance._data = data
    print(len(instance.cshape))
    broadcasted = instance.broadcast_cdim(cdim)
    print(broadcasted._data)
