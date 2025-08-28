"""
The following documents the pyfar base class.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

class _PyfarBase(ABC):
    """
    Abstract base class for Audio and Coordinates classes in pyfar.

    This class defines the interface that Audio and Coordinates classes
    must follow.
    """

    def __init__(self, comment=""):
            self._comment = comment

    @property
    def comment(self):
        """Get comment."""

        return self._comment

    @property
    @abstractmethod
    def cshape(self):
        """
        Subclasses must define a channel shape.

        The channel shape gives the shape of the data excluding the last
        dimension.
        """
        pass

    @property
    @abstractmethod
    def cdim(self):
        """
        Subclasses must define a channel dimension.

        The channel dimension (`cdim`) gives the number of dimensions of the
        audio data excluding the last dimension.
        """
        pass

    @property
    @abstractmethod
    def csize(self):
        """
        Subclasses must define a channel size.

        The channel size gives the number of points stored in the object.
        """
        pass

    def copy(self):
        """Return a deep copy of the object."""

        return deepcopy(self)

    @abstractmethod
    def _encode(self):
        """Return a dictionary for the encoding."""
        pass

    @abstractmethod
    def _decode(self):
        """Decode an object based on its respective `_encode` counterpart."""
        pass

    @abstractmethod
    def __eq__(self, other):
        """Check for equality of two objects."""
        pass



