from pyfar.classes._pyfar_base import _PyfarBase

class Dummy(_PyfarBase):
    """
    Minimal dummy class for testing _PyfarBase.
    """

    def _decode(self):
        pass

def test_comment():
    """Test the comment property."""

    instance = Dummy("Dummy #1")
    assert instance.comment == "Dummy #1"

def test_comment_setter():
    """Test the comment.setter."""

    instance = Dummy()
    instance.comment = "Dummy #2"
    assert instance.comment == "Dummy #2"

def test_copy():
    """Test the copy() function."""

    instance = Dummy("Dummy #3")
    instance_copy = instance.copy()
    assert instance == instance_copy

def test_encode():
    """Test the _encode() function."""

    instance = Dummy("Dummy #4")
    instance_encoded = instance._encode()
    assert instance_encoded == {'_comment': 'Dummy #4'}

def test_eq():
    """Test the __eq__() function."""

    instance_1 = Dummy("Dummy #5")
    instance_2 = Dummy("Dummy #5")
    instance_3 = Dummy("Dummy #6")
    assert instance_1 == instance_2
    assert instance_1 != instance_3
