"""
The following documents the abstract class for pyfar classes,
which defines arithmetic operations.
"""

from abc import ABC, abstractmethod

class _PyfarArithmetics(ABC):
    """
    Internal abstract base class for pyfar classes.

    This class defines the basic arithmetic operations (addition, subtraction,
    multiplication, division and exponentiation) and their right-hand
    alternatives, which must be implemented by other classes.
    """

    @abstractmethod
    def __add__(self, data):
        """Addition of two objects (`self + data`)."""
        pass

    @abstractmethod
    def __radd__(self, data):
        """Reflected addition of two objects (`data + self`)."""
        pass

    @abstractmethod
    def __sub__(self, data):
        """Subtraction of two objects (`self - data`)."""
        pass

    @abstractmethod
    def __rsub__(self, data):
        """Reflected subtraction of two objects (`data - self`)."""
        pass

    @abstractmethod
    def __mul__(self, data):
        """Multiplication of two objects (`self * data`)."""
        pass

    @abstractmethod
    def __rmul__(self, data):
        """Reflected multiplication of two objects (`data * self`)."""
        pass

    @abstractmethod
    def __truediv__(self, data):
        """Division of two objects (`self / data`)."""
        pass

    @abstractmethod
    def __rtruediv__(self, data):
        """Reflected division of two objects (`data / self`)."""
        pass

    @abstractmethod
    def __pow__(self, data):
        """Exponentiation.
        Raise the object to the power of `data` (`self ** data`).
        """
        pass

    @abstractmethod
    def __rpow__(self, data):
        """
        Reflected exponentiation.
        Raise `data` to the power of the object (`data ** self`).
        """
        pass

    @abstractmethod
    def _assert_match_for_arithmetics(self, data):
        """
        Check that two objects are compatible for arithmetic
        operations.
        """
