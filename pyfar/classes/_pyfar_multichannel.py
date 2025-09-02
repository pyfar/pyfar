"""
The following documents the NumPy-like abstract class for multichannel
pyfar classes.
"""

from abc import ABC, abstractmethod

class _PyfarMultichannel(ABC):
    """
    Internal abstract base class providing NumPy-like functionality for
    multichannel pyfar classes.

    This class defines properties and methods inspired by NumPy's interface.
    Subclasses are expected to implement these methods to support array-like
    operations on custom pyfar classes.
    """

    def __init__(self, comment=""):
            self._comment = comment

    @property
    @abstractmethod
    def cshape(self):
        """
        Return channel shape.
        """
        pass

    @property
    @abstractmethod
    def cdim(self):
        """
        Return channel dimension.

        The channel dimension (`cdim`) is the length of the channel
        shape (`cshape`) (e.g. ``self.cshape = (2, 3)``; ``self.cdim = 2``).
        """
        return len(self.cshape)

    @property
    @abstractmethod
    def csize(self):
        """
        Return channel size.

        The channel size is the total number of channels
        (e.g. ``self.cshape = (2, 3)``; ``self.csize = 6``)
        """
        pass

    @abstractmethod
    def reshape(self, newshape):
        """
        Return reshaped copy of the object.

        Parameters
        ----------
        newshape : int, tuple
            New `cshape` of the object.

        Returns
        -------
        reshaped : _PyfarMultichannel
            reshaped copy of the audio object.
        """

        pass

    @abstractmethod
    def flatten(self):
        """Return a copy of the object collapsed into one channel dimension."""

        pass
    @abstractmethod
    def transpose(self):
        """Returns a copy of the object with channel axes transposed."""

        pass

    @abstractmethod
    def broadcast_cshape(self, cshape):
        """
        Broadcast the object to a certain channel shape (`cshape`).

        Parameters
        ----------
        cshape : tuple
            The cshape to which the object is broadcasted.

        Returns
        -------
        object : _NumpyLike
            Broadcasted copy of the object.
        """

        pass

    @abstractmethod
    def broadcast_cdim(self, cdim):
        """
        Broadcast an copy of the object wth a certain channel dimension
        (`cdim`).

        Parameters
        ----------
        cdim : int
            The cdim to which the object is broadcasted.

        Returns
        -------
        object : _PyfarMultichannel
            Broadcasted copy of the object.
        """

        pass
