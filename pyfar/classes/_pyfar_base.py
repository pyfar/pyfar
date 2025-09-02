"""
The following documents the pyfar base class.
"""

from abc import ABC, abstractmethod
from copy import deepcopy

class _PyfarBase(ABC):
    """
    Abstract base class for classes in pyfar.

    This class defines the interface that pyfar classes
    must follow.
    """

    def __init__(self, comment=""):
            self._comment = comment

    @property
    def comment(self):
        """Get comment."""

        return self._comment

    @comment.setter
    def comment(self, comment):
        """Set comment."""
        if not isinstance(comment, str):
            raise TypeError("Comment has to be of type string.")
        else:
            self._comment = comment

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



