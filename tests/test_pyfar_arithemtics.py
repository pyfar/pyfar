from pyfar.classes._PyfarArithmetics import _PyfarArithmetics
import numpy as np
import pytest
from unittest.mock import patch
import operator

def _extract_data(self, other, mode=None, operation=None):
    """Provide an implementation for the @abstractmethod _extract_data()."""
    del mode, operation
    result = self.copy()
    a = self._data
    if hasattr(other, "_data"):
        b = other._data
    else:
        b = other
    return (a, b, result)


@pytest.fixture()
def _fixture_arithmetics():
    """Patch the abstract methods of _PyfarArithmetics."""
    with patch.multiple(_PyfarArithmetics, __abstractmethods__=set()), \
         patch.object(_PyfarArithmetics, "_extract_data", _extract_data):
        yield


@pytest.fixture()
def instances(_fixture_arithmetics):
    """Create instances of _PyfarArithmetics for testing."""
    data = np.ones((2, 3, 4))
    a, b = _PyfarArithmetics(), _PyfarArithmetics()
    a._data = data
    b._data = 3 * data
    return a, b


@pytest.mark.parametrize("swap", [False, True])
def test_add(_fixture_arithmetics,  instances, swap):
    """Test the add method with a _PyfarArithmetics instance as other."""
    instance1, instance2 = instances
    if swap:
        result = instance1.add(instance2)
    else:
        result = instance2.add(instance1)

    assert (result._data == (np.ones((2, 3, 4)) + 3*np.ones((2, 3, 4)))).all()


@pytest.mark.parametrize("swap", [False, True])
def test_subtract(_fixture_arithmetics, instances, swap):
    """Test the subtract method with a _PyfarArithmetics instance as other."""
    instance1, instance2 = instances
    if swap:
        result = instance1.subtract(instance2)
        assert (result._data == (
            np.ones((2, 3, 4)) - 3*np.ones((2, 3, 4)))).all()
    else:
        result = instance2.subtract(instance1)
        assert (result._data == (
            3*np.ones((2, 3, 4)) - np.ones((2, 3, 4)))).all()


@pytest.mark.parametrize("swap", [False, True])
def test_multiply(_fixture_arithmetics, instances, swap):
    """Test the multiply method with a _PyfarArithmetics instance as other."""
    instance1, instance2 = instances
    if swap:
        result = instance1.multiply(instance2)
    else:
        result = instance2.multiply(instance1)

    assert (result._data == (np.ones((2, 3, 4)) * 3*np.ones((2, 3, 4)))).all()


@pytest.mark.parametrize("swap", [False, True])
def test_divide(_fixture_arithmetics, instances, swap):
    """Test the divide method with a _PyfarArithmetics instance as other."""
    instance1, instance2 = instances
    if swap:
        result = instance1.divide(instance2)
        assert (result._data == (
            np.ones((2, 3, 4)) / 3*np.ones((2, 3, 4)))).all()
    else:
        result = instance2.divide(instance1)
        assert (result._data == (
            3*np.ones((2, 3, 4)) / np.ones((2, 3, 4)))).all()


@pytest.mark.parametrize(("method_name", "op"), [
    ("add", operator.add),
    ("subtract", operator.sub),
    ("multiply", operator.mul),
    ("divide", operator.truediv),
    ("power", operator.pow),
])
def test_dunder_methods(_fixture_arithmetics, instances, method_name, op):
    """Test the dunder methods with a _PyfarArithmetics instance as other."""
    instance1, instance2  = instances
    assert op(instance1, instance2) == getattr(
        instance1, method_name)(instance2)


@pytest.mark.parametrize(("method_name", "other", "expected"), [
    ("add", 3, np.array(np.ones((2, 3, 4)) + 3)),
    ("subtract", 3, np.array(np.ones((2, 3, 4)) - 3)),
    ("multiply", 3, np.array(np.ones((2, 3, 4)) * 3)),
    ("divide", 3, np.array(np.ones((2, 3, 4)) / 3)),
    ("power", 3, np.array(np.ones((2, 3, 4)) ** 3)),
    ("add", 3*np.ones((2, 3, 4)),
        np.array(np.ones((2, 3, 4)) + 3*np.ones((2, 3, 4)))),
    ("subtract", 3*np.ones((2, 3, 4)),
        np.array(np.ones((2, 3, 4)) - 3*np.ones((2, 3, 4)))),
    ("multiply", 3*np.ones((2, 3, 4)),
        np.array(np.ones((2, 3, 4)) * 3*np.ones((2, 3, 4)))),
    ("divide", 3*np.ones((2, 3, 4)),
        np.array(np.ones((2, 3, 4)) / 3*np.ones((2, 3, 4)))),
    ("power", 3*np.ones((2, 3, 4)),
        np.array(np.ones((2, 3, 4)) ** 3*np.ones((2, 3, 4)))),
])
def test_all_methods_with_array_and_int(_fixture_arithmetics, instances,
                                   method_name, other, expected):
    """Test the dunder methods with a numpy array and an int as other."""
    instance1, _  = instances
    instance2 = other
    actual = getattr(instance1, method_name)(instance2)
    assert (actual._data == expected).all()


@pytest.mark.parametrize(("method_name", "op"), [
    ("add", operator.add),
    ("subtract", operator.sub),
    ("multiply", operator.mul),
    ("divide", operator.truediv),
    ("power", operator.pow),
])
def test_dunder_methods_with_array(_fixture_arithmetics, instances,
                                   method_name, op):
    """Test the dunder methods with a numpy array as other."""
    instance1, _  = instances
    instance2 = np.ones((2, 3, 4))
    assert op(instance1, instance2) == getattr(
        instance1, method_name)(instance2)


@pytest.mark.parametrize(("expected", "op"), [
    (np.array(np.ones((2, 3, 4)) + 3*np.ones((2, 3, 4))),
                  operator.add),
    ((np.ones((2, 3, 4)) - 3*np.ones((2, 3, 4))),
                  operator.sub),
    ((np.ones((2, 3, 4)) * 3*np.ones((2, 3, 4))),
                  operator.mul),
    ((np.ones((2, 3, 4)) / 3*np.ones((2, 3, 4))),
                  operator.truediv),
    ((np.ones((2, 3, 4)) ** 3*np.ones((2, 3, 4))),
                  operator.pow),
])
def test_reflected_dunder_methods_with_array(_fixture_arithmetics, instances,
                                              op, expected):
    """Test the reflected dunder methods with a numpy array as other."""
    _, instance2= instances
    instance1 = np.ones((2, 3, 4))
    actual = op(instance1, instance2)
    assert (actual._data == expected).all()

