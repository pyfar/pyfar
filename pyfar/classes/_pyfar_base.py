"""
The following documents the pyfar base class.

It serves the purpose to implement the requirements for all pyfar classes.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import deepdiff

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

    def _encode(self):
        """Return a dictionary for the encoding."""
        return self.copy().__dict__

    def __eq__(self, other):
        """Check for equality of two objects."""
        return not deepdiff.DeepDiff(self.__dict__, other.__dict__)

    @abstractmethod
    def _decode(self):
        """Decode an object based on its respective `_encode` counterpart."""
        pass
