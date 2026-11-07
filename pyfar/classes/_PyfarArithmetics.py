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
    alternatives, which must be implemented by other classes.
    """

    #specifies which datatypes are allowed for arithmetic operations.
    _allowed_datatypes_add_sub = None
    _allowed_datatypes_div_mul = None


    def _check_data_add_sub(self, other):
        if not isinstance(other, self._allowed_datatypes_add_sub):
            raise TypeError("Incompatible object for additive operations")

        if hasattr(other, "_data"):
            data = other._data
        else:
            data = other

        if self._data.shape != data.shape:
            raise ValueError("Incompatible shapes for additive operations")
        return data

    def add(self, other):
        """Addition of two objects (self + other)."""
        data = self._check_data_add_sub(other)

        result = self.copy()
        result._data = self._data + data

        return result

    def radd(self, other):
        """Reflected addition of two objects (other + self)."""

        data = self._check_data_add_sub(other)

        result = self.copy()
        result._data = data + self._data

        return result

    def sub(self, other):
        """Subtraction of two objects (self - other)."""
        data = self._check_data_add_sub(other)

        result = self.copy()
        result._data = self._data - data

        return result

    def rsub(self, other):
        """Reflected subtraction of two objects (other - self)."""
        data = self._check_data_add_sub(other)

        result = self.copy()
        result._data = data - self._data

        return result

    @abstractmethod
    def mul(self, other):
        """Multiplication of two objects (self * other)."""
        pass

    @abstractmethod
    def rmul(self, other):
        """Reflected multiplication of two objects (other * self)."""
        pass

    @abstractmethod
    def truediv(self, other):
        """Division of two objects (self / other)."""
        pass

    @abstractmethod
    def rtruediv(self, other):
        """Reflected division of two objects (other / self)."""
        pass

    @abstractmethod
    def pow(self, other):
        """Exponentiation.
        Raise the object to the power of other (self ** other).
        """
        pass

    @abstractmethod
    def rpow(self, other):
        """
        Reflected exponentiation.
        Raise other to the power of the object (other ** self).
        """
        pass

    @abstractmethod
    def _assert_match_for_arithmetics(self, other):
        """
        Check that two objects are compatible for arithmetic
        operations.
        """
