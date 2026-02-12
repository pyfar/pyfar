"""This module contains the Rotation class."""
from scipy.spatial.transform import Rotation as scRotation
import numpy as np
import warnings

import pyfar as pf
if np.__version__ < '2.0.0':
    from numpy import VisibleDeprecationWarning
else:
    from numpy.exceptions import VisibleDeprecationWarning

# this warning needs to be caught and appears if numpy array are generated
# from nested lists containing lists of unequal lengths,
#  e.g., [[1, 0, 0], [1, 0]]
warnings.filterwarnings("error", category=VisibleDeprecationWarning)


class Rotation():
    """
    Pyfar Rotation class.
    """
    def __init__(self):
        raise RuntimeError("Rotation objects must be created using one of "
                           "the `from_...` methods")

    def __repr__(self):
        """String representation of Rotation object."""
        num_orientations = \
            1 if self._quat.ndim == 1 else self._quat.shape[0]

        _repr = f"Pyfar Rotations object with {num_orientations} rotations."
        return _repr

    # Factory Methods
    @classmethod
    def from_davenport(cls, axes, order, angles, degrees=False):
        """"""
        instance = cls.__new__(cls)
        instance._store_rotation(scRotation.from_davenport(
            axes, order, angles, degrees=False))

        return instance

    @classmethod
    def from_euler(cls, seq, angles, degrees=False):
        """"""
        instance = cls.__new__(cls)
        instance._store_rotation(scRotation.from_euler(
            seq, angles, degrees=False))

        return instance

    @classmethod
    def from_matrix(cls, matrix):
        """"""
        instance = cls.__new__(cls)
        instance._store_rotation(scRotation.from_matrix(matrix))

        return instance

    @classmethod
    def from_quat(cls, quat):
        """"""
        instance = cls.__new__(cls)
        instance._store_rotation(scRotation.from_quat(quat))
        return instance


    @classmethod
    def _from_scipy_rotation(cls, sc_rotation):
        """"""
        instance = cls.__new__(cls)
        instance._store_rotation(sc_rotation)
        return instance

    # Instance methods
    def as_davenport(self):
        """"""
        return self._rot.as_davenport()

    def as_euler(self):
        """"""
        return self._rot.as_euler()

    def as_quat(self):
        """"""
        return self._quat

    def as_matrix(self):
        """"""
        return self._rot.as_matrix()

    def mean(self, weights=None, axis=None):
        """"""
        return self._from_scipy_rotation(self._rot.mean(weights, axis))

    # Private methods
    def _store_rotation(self, sc_rotation: scRotation):
        """
        Stores scipy rotation and raw quaternion data.

        This is an auxiliary private function to save some codelines.

        Parameters
        ----------
        sc_rotatiton : scipy.Rotation
            Scipy Rotation object
        """
        self._rot = sc_rotation
        self._quat = sc_rotation.as_quat()
