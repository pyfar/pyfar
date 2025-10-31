from pyfar.classes._pyfar_base import _PyfarBase
from unittest.mock import patch
import pytest

@pytest.fixture(autouse=True)
def _disable_abstract_methods():
    """Automatically disable abstract methods for tests."""
    with patch.multiple(_PyfarBase,__abstractmethods__=set()):
            yield

def test_comment():
    """Test the comment property."""
    instance =_PyfarBase("Dummy #1")
    assert instance.comment == "Dummy #1"

def test_comment_setter():
    """Test the comment.setter."""
    instance = _PyfarBase()
    assert instance.comment == ''
    instance.comment = "Dummy #2"
    assert instance.comment == "Dummy #2"

def test_copy():
    """Test the copy() function."""
    instance = _PyfarBase("Dummy #3")
    instance_copy = instance.copy()
    assert instance == instance_copy

def test_encode():
    """Test the _encode() function."""
    instance = _PyfarBase("Dummy #4")
    instance_encoded = instance._encode()
    assert instance_encoded == {'_comment': 'Dummy #4'}

def test_eq():
    """Test the __eq__() function."""
    instance_1 = _PyfarBase("Dummy #5")
    instance_2 = _PyfarBase("Dummy #5")
    instance_3 = _PyfarBase("Dummy #6")
    assert instance_1 == instance_2
    assert instance_1 != instance_3
