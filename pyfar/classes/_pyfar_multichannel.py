"""
The following documents the abstract class for multichannel pyfar classes,
which implements functionality similar to NumPy arrays.
"""

from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from math import prod
from pyfar.classes._pyfar_base import _PyfarBase

class _PyfarMultichannel(_PyfarBase, ABC):
    """
    Internal abstract base class for multichannel pyfar classes.

    This class defines properties and methods inspired by NumPy arrays.
    Subclasses are expected to implement these methods on custom
    pyfar classes.
    """

    @property
    def cshape(self):
        """
        Return channel shape.
        """
        return self._data.shape[:-1]

    @property
    def cdim(self):
        """
        Return channel dimension.

        The channel dimension (`cdim`) is the length of the channel
        shape (`cshape`) (e.g. ``self.cshape = (2, 3)``; ``self.cdim = 2``).
        """
        return len(self.cshape)

    @property
    def csize(self):
        """
        Return the channel size, i.e., total number of channels.
        """
        return prod(self.cshape)

    @property
    def T(self):
        """Shorthand for `self.transpose()`."""
        return self.transpose()

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

        # Check input
        if not isinstance(newshape, int) and not isinstance(newshape, tuple):
            raise ValueError("newshape must be an integer or tuple")

        if isinstance(newshape, int):
            newshape = (newshape, )

        # reshape
        reshaped = deepcopy(self)
        length_last_dimension = reshaped._data.shape[-1]
        try:
            reshaped._data = reshaped._data.reshape(
                newshape + (length_last_dimension, ))
        except ValueError as e:
            if np.prod(newshape) != np.prod(self.cshape):
                raise ValueError(
                    (f"Cannot reshape an object of cshape "
                     f"{self.cshape} to {newshape}")) from e

        return reshaped

    def flatten(self):
        """Return a copy of the object collapsed into one channel dimension."""
        return self.reshape(self.csize)

    def transpose(self, *caxes):
        """Return a copy of the object with channel axes transposed.

        Parameters
        ----------
        caxes : iterable of int or None
            Define how the axes are ordered in the transposed object.
            If not specified, reverses the order of ``self.caxes``.
        """

        if not caxes:
            caxes = tuple(range(len(self.cshape)))[::-1]
        elif len(caxes) == 1 and isinstance(caxes[0], (tuple, list)):
            caxes = tuple(caxes[0])
        print(caxes)
        caxes = tuple(a + self.cdim if a < 0 else a for a in caxes)
        if len(caxes) != self.cdim:
            raise ValueError("Number of axes must match the cdim of the object")
        if sorted(caxes) != list(range(self.cdim)):
            raise ValueError("Axes must be a rearrangement of cdim")

        result = deepcopy(self)
        result._data = result._data.transpose(*caxes, self.cdim)

        return result

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

        signal = self.copy()
        signal._data = np.broadcast_to(
            signal._data, cshape + (signal._data.shape[-1], ))
        return signal

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
        if self.cdim > cdim:
            raise ValueError(
            "Can not broadcast: Current channel dimensions exceeds `cdim`.")
        signal = self.copy()
        while signal.cdim < cdim:
            signal._data = signal._data[None, ...]
        return signal

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

