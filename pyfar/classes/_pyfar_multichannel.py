"""
The following documents the abstract class for multichannel pyfar classes,
which implements functionality similar to NumPy arrays.
"""

from abc import ABC, abstractmethod
import numpy as np

class _PyfarMultichannel(ABC):
    """
    Internal abstract base class for multichannel pyfar classes.

    This class defines properties and methods inspired by NumPy arrays.
    Subclasses are expected to implement these methods on custom
    pyfar classes.
    """

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
    def n_channels(self):
        """
        Return the total number of channels.
        """
        return np.prod(self.cshape)

    @property
    def T(self):
        """Shorthand for `self.transpose()`."""
        return self.transpose()

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
            Reshaped copy of the object.
        """

        pass

    def flatten(self):
        """Return a copy of the object collapsed into one channel dimension."""

        return self.reshape(self.n_channels)

    @abstractmethod
    def transpose(self, caxes=None):
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
        object : _PyfarMultichannel
            Broadcasted copy of the object.
        """

        pass

    def broadcast_cdim(self, cdim):
        """
        Broadcast a copy of the object with a certain channel dimension
        (`cdim`) by prepending dimensions to the channel shape (`cshape`).

        Parameters
        ----------
        cdim : int
            The cdim to which the object is broadcasted.

        Returns
        -------
        object : _PyfarMultichannel
            Broadcasted copy of the object.
        """
        if len(self.cshape) > cdim:
            raise ValueError(
            "Can not broadcast: Current channel dimensions exceeds `cdim`.")
        newshape = tuple(np.ones(cdim-self.n_channels, dtype=int))+self.cshape

        return self.copy().reshape(newshape)

    @abstractmethod
    def __getitem__(self, index):
        """
        Get a value of the object at the specified index.
        """
        pass

    @abstractmethod
    def __setitem__(self, index, value):
        """
        Set the value of the object at the specified index.
        """
        pass

