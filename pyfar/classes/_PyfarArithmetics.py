"""
The following documents the abstract class for pyfar classes,
which defines arithmetic operations.
"""

from abc import abstractmethod
from pyfar.classes._pyfar_base import _PyfarBase


class _PyfarArithmetics(_PyfarBase):
    """
    Internal abstract base class for pyfar classes.

    This class defines the basic arithmetic operations (addition, subtraction,
    multiplication, division and exponentiation) and their right-hand
    alternatives.
    """

    # Set to "None" so that _PyfarArithmetics instances are not wrapped
    # in numpy arrays
    __array_ufunc__ = None

    @abstractmethod
    def _extract_data(self, other, mode, operation: str):
        pass


    def add(self, *others, mode=None):
        """Addition of two objects (self + other)."""
        result = self.copy()
        for other in others:
            a, b, result = result._extract_data(other, mode, "_add")
            result._data = a + b
        return result


    def subtract(self, *others, mode=None):
        """Subtraction of two objects (self - other)."""
        result = self.copy()
        for other in others:
            a, b, result = result._extract_data(other, mode, "_subtract")
            result._data = a - b
        return result


    def multiply(self, *others, mode=None):
        """Multiplication of two objects (self * other)."""
        result = self.copy()
        for other in others:
            a, b, result = result._extract_data(other, mode, "_multiply")
            result._data = a * b
        return result


    def divide(self, *others, mode=None):
        """Division of two objects (self / other)."""
        result = self.copy()
        for other in others:
            a, b, result = result._extract_data(other, mode, "_divide")
            result._data = a / b
        return result


    def power(self, *others, mode=None):
        """Exponentiation.
        Raise the object to the power of other (self ** other).
        """
        result = self.copy()
        for other in others:
            a, b, result = result._extract_data(other, mode, "_power")
            result._data = a**b
        return result


    def __add__(self, data):
        """Return self + data."""
        return self.add(data)


    def __radd__(self, data):
        """Return data + self."""
        return self.add(data)


    def __sub__(self, data):
        """Return self - data."""
        return self.subtract(data)


    def __rsub__(self, data):
        """Return data - self."""
        return (self.subtract(data))*(-1)


    def __mul__(self, data):
        """Return self * data."""
        return self.multiply(data)


    def __rmul__(self, data):
        """Return data * self."""
        return self.multiply(data)


    def __truediv__(self, data):
        """Return self / data."""
        return self.divide(data)


    def __rtruediv__(self, data):
        """Return data / self."""
        return (self.divide(data))**(-1)


    def __pow__(self, data):
        """Return self ** data."""
        return self.power(data)


    def __rpow__(self, data):
        """Return data ** self."""
        result = self.copy()
        result._data = data**(self._data)
        return result
