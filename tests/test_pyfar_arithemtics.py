from pyfar.classes._PyfarArithmetics import _PyfarArithmetics
import numpy as np
import pytest
from unittest.mock import patch

@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
@pytest.mark.parametrize('method', ['add', 'radd'])
def test_add_radd(method):
    """
    Test the add and radd methods with a _PyfarArithmetics instance as other.
    """
    data = np.ones((2, 3, 4))
    allowed_classes = (_PyfarArithmetics,)
    instance1, instance2 = _PyfarArithmetics(), _PyfarArithmetics()

    instance1._data = data
    instance2._data = data

    instance1._allowed_datatypes_add_sub = allowed_classes
    instance2._allowed_datatypes_add_sub = allowed_classes

    result = getattr(instance1, method)(instance2)

    assert (result._data == (np.ones((2, 3, 4)) + np.ones((2, 3, 4)))).all()


@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
@pytest.mark.parametrize('method', ['add', 'radd'])
def test_add_radd_with_array(method):
    """Test the add and radd methods with an array as other."""
    data = np.ones((2, 3, 4))
    allowed_classes = (np.ndarray,)
    instance1 = _PyfarArithmetics()
    instance2 = np.ones((2, 3, 4))

    instance1._data = data
    instance1._allowed_datatypes_add_sub = allowed_classes

    result = getattr(instance1, method)(instance2)

    assert (result._data == (np.ones((2, 3, 4)) + np.ones((2, 3, 4)))).all()


@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
@pytest.mark.parametrize('method', ['sub', 'rsub'])
def test_sub_rsub(method):
    """
    Test the sub and rsub methods with a _PyfarArithmetics instance as other.
    """
    data = np.ones((2, 3, 4))
    allowed_classes = (_PyfarArithmetics,)
    instance1, instance2 = _PyfarArithmetics(), _PyfarArithmetics()

    instance1._data = data
    instance2._data = data

    instance1._allowed_datatypes_add_sub = allowed_classes
    instance2._allowed_datatypes_add_sub = allowed_classes

    result =  getattr(instance1, method)(instance2)
    assert (result._data == (np.ones((2, 3, 4)) - np.ones((2, 3, 4)))).all()


@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
@pytest.mark.parametrize('method', ['sub', 'rsub'])
def test_sub_rsub_with_array(method):
    """Test the sub and rsub methods with an array as other."""
    data = np.ones((2, 3, 4))
    allowed_classes = (np.ndarray,)
    instance1 = _PyfarArithmetics()
    instance2 = np.ones((2, 3, 4))

    instance1._data = data
    instance1._allowed_datatypes_add_sub = allowed_classes

    result = getattr(instance1, method)(instance2)

    assert (result._data == (np.ones((2, 3, 4)) - np.ones((2, 3, 4)))).all()


@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
def test_sub_rsub_difference():
    """Test the difference between the sub and rsub methods."""
    data = np.array([[1, -2], [3, 4]])

    allowed_classes = (np.ndarray,)
    instance1 = _PyfarArithmetics()
    instance2 = np.array([[-1, 0], [-4, 1]])

    instance1._data = data
    instance1._allowed_datatypes_add_sub = allowed_classes

    result1 = instance1.sub(instance2)
    result2 = instance1.rsub(instance2)

    assert (result1._data == -result2._data).all()


@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
@pytest.mark.parametrize('method', ['add', 'radd', 'sub', 'rsub'])
def test_additive_methods_type_error(method):
    """
    Test the methods with an instance of an invalid datatype, which should
    cause a type error.
    """
    data = np.ones((2, 3, 4))
    allowed_classes = (int, float, _PyfarArithmetics)
    instance1 = _PyfarArithmetics()
    instance2 = "cat"

    instance1._data = data
    instance1._allowed_datatypes_add_sub = allowed_classes

    with pytest.raises(TypeError,  match="Incompatible object for additive"):
        getattr(instance1, method)(instance2)


@patch.multiple(_PyfarArithmetics, __abstractmethods__=set())
@pytest.mark.parametrize('method', ['add', 'radd', 'sub', 'rsub'])
def test_additive_methods_value_error(method):
    """
    Test the methods with data of different shapes, which should cause
    an value error.
    """
    data = np.ones((2, 3, 4))
    allowed_classes = (np.ndarray)
    instance1 = _PyfarArithmetics()
    instance2 =  np.ones((2, 4))

    instance1._data = data
    instance1._allowed_datatypes_add_sub = allowed_classes

    with pytest.raises(ValueError, match="Incompatible shapes for additive"):
        getattr(instance1, method)(instance2)
