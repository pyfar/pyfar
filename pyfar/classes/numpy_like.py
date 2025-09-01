"""
The following documents the NumPy-like abstract class.
"""

from abc import ABC, abstractmethod

class _NumPyLike(ABC):
    """
    Internal abstract base class providing NumPy-like functionality.

    This class defines properties and methods inspired by NumPy's interface.
    Subclasses are expected to implement these methods to support array-like
    operations on custom data types.
    """

    def __init__(self, comment=""):
            self._comment = comment

    @abstractmethod
    def reshape(self, newshape):
        """
        Return reshaped copy of the object.

        Parameters
        ----------
        newshape : int, tuple
            New `cshape` of the object.
        """

        pass

    @abstractmethod
    def flatten(self):
        """Return a copy of the object collapsed into one dimension."""

        pass
    @abstractmethod
    def transpose(self):
        """Transpose the objects data and return copy of this object."""

        pass

    @abstractmethod
    def broadcast_cshape(self):
        """Broadcast an object to a certain channel shape (`cshape`)."""

        pass

    @abstractmethod
    def broadcast_cdim(self):
        """Broadcast an object to a certain channel dimension (`cdim`)."""

        pass

    @abstractmethod
    # new method
    def abs(self):
        """Calculate the absolute value of the object element-wise."""

        pass

    @abstractmethod
    # new method
    def sum(self, axis=None):
        """
        Calculate the sum of elements.

        Parameters
        ----------
        axis : int, optional
            If ``axis`` is ``None``, the sum of all elements in the the array
            stored in the object is returned as a scalar. If ``axis`` is
            specified, the sum is computed over the given axis or axes,
            preserving the other dimensions. The default is ``None``.
        """

        pass

    @abstractmethod
    # new method
    def resize(self):
        """Change the shape and size of the array stored in the object."""

        pass

    @abstractmethod
    # new method
    def swap_caxes(self, axis1, axis2):
        """
        Swap two axes of the array stored in the object.

        Parameters
        ----------
        axis1 : int
            First axis to swap.
        axis2 : int
            Second axis to swap.
        """

        pass
