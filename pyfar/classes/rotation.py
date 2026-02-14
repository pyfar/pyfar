"""This module contains the Rotation class."""
from scipy.spatial.transform import Rotation as scRotation
import numpy as np
import warnings

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
        quat = self._rot.as_quat()
        num_orientations = \
            1 if quat.ndim == 1 else quat.shape[0]

        repr = f"Pyfar Rotations object with {num_orientations} rotations."
        return repr

    def __getitem__(self, idx):
        self._rot = self._rot[idx]
        return self

    def __iter__(self):
        raise NotImplementedError

    def __setitem__(self):
        raise NotImplementedError('Setting an item is disabled for pyfar'
        'Rotations. If you want to modify the Rotation, use an array'
        'representation like `as_quat()` or `as_matrix()` and create a new '
        'object.')

    def __mul__(self, other):
        """"""
        if isinstance(other, Rotation):
            other = scRotation.from_quat(other.as_quat())
        if isinstance(other, scRotation):
            self._rot *= other
            return self

    def __rmul__(self, other):
        """"""
        if isinstance(other, Rotation):
            other = scRotation.from_quat(other.as_quat())
        if isinstance(other, scRotation):
            self._rot *= other
            return self

    def __pow__(self, n):
        """Compose orientation with itself n times."""
        self._rot = self._rot.__pow__(n)
        return self

    def __eq__(self, other):
        """Check for equality of two objects."""
        return np.array_equal(self.as_quat(), other.as_quat())

    # from-... methods
    @classmethod
    def from_davenport(cls, axes, order, angles, degrees=False):
        """"""
        instance = cls.__new__(cls)
        instance._rot = scRotation.from_davenport(
            axes, order, angles, degrees=degrees)
        return instance

    @classmethod
    def from_euler(cls, seq, angles, degrees=False):
        """"""
        instance = cls.__new__(cls)
        instance._rot = scRotation.from_euler(
            seq, angles, degrees=degrees)
        return instance

    @classmethod
    def from_matrix(cls, matrix):
        """"""
        instance = cls.__new__(cls)
        instance._rot = scRotation.from_matrix(matrix)
        return instance

    @classmethod
    def from_mrp(cls, mrp):
        instance = cls.__new__(cls)
        instance._rot = scRotation.from_mrp(mrp)
        return instance

    @classmethod
    def from_quat(cls, quat):
        """"""
        instance = cls.__new__(cls)
        instance._rot = scRotation.from_quat(quat)
        return instance

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):
        """"""
        instance = cls.__new__(cls)
        instance._rot = scRotation.from_rotvec(rotvec, degrees)
        return instance

    @classmethod
    def from_view_up(cls, views, ups):
        """Initialize Orientations from a view an up vector.

        Orientations are internally stored as quaternions for better spherical
        linear interpolation (SLERP) and spherical harmonics operations.
        More intuitionally, they can be expressed as view and up vectors
        which cannot be collinear. In this case, they are restricted to be
        perpendicular to minimize rounding errors.

        Parameters
        ----------
        views : array_like, shape (N, 3) or (3,), Coordinates
            A single vector or a stack of vectors, giving the look-direction of
            an object in three-dimensional space, e.g. from a listener, or the
            acoustic axis of a loudspeaker, or the direction of a main lobe.
            Views can also be passed as a Coordinates object.

        ups : array_like, shape (N, 3) or (3,), Coordinates
            A single vector or a stack of vectors, giving the up-direction of
            an object, which is usually the up-direction in world-space. Views
            can also be passed as a Coordinates object.

        Returns
        -------
        orientations : Orientations
            Object containing the orientations represented by quaternions.
        """
        instance = cls.__new__(cls)

        # init views and up
        try:
            views = np.atleast_2d(views).astype(np.float64)
            ups = np.atleast_2d(ups).astype(np.float64)
        except VisibleDeprecationWarning as exc:
            raise ValueError(
                "Expected `views` and `ups` to have shape (N, 3)") from exc

        # check views and ups
        if (views.ndim > 2 or views.shape[-1] != 3 or
                ups.ndim > 2 or ups.shape[-1] != 3):
            raise ValueError(f"Expected `views` and `ups` to have shape (N, 3)"
                             f" or (3,), got {views.shape}")
        if views.shape == ups.shape:
            pass
        elif views.shape[0] > 1 and ups.shape[0] == 1:
            ups = np.repeat(ups, views.shape[0], axis=0)
        elif ups.shape[0] > 1 and views.shape[0] == 1:
            views = np.repeat(views, ups.shape[0], axis=0)
        else:
            raise ValueError("Expected 1:1, 1:N or N:1 `views` and `ups` "
                             f"not M:N, got {views.shape} and {ups.shape}")

        if not (np.all(np.linalg.norm(views, axis=1)) and
                np.all(np.linalg.norm(ups, axis=1))):
            raise ValueError("View and Up Vectors must have a length.")
        if not np.allclose(0, np.einsum('ij,kj->k', views, ups)):
            raise ValueError("View and Up vectors must be perpendicular.")

        # Assuming that the direction of the cross product is defined
        # by the right-hand rule
        rights = np.cross(views, ups)

        # In a standard Cartesian right-handed coordinate system,
        # these vectors are defined as [x, y, z] = [view, left, up], where
        # left is the same vector as -rights
        rotation_matrix = np.asarray([views, -rights, ups])
        rotation_matrix = np.swapaxes(rotation_matrix, 0, 1)

        return instance.from_matrix(rotation_matrix)

    # other class methods
    @classmethod
    def align_vectors(a, b, weights=None, return_sensitivity=False):
        raise NotImplementedError

    @classmethod
    def concatenate(cls, rotation):
        raise NotImplementedError

    @classmethod
    def identity(cls, num=None, *, shape=None):
        raise NotImplementedError

    @classmethod
    def random(cls, num=None, rng=None, *, shape=None):
        raise NotImplementedError

    # private class methods
    @classmethod
    def _decode(cls, obj_dict):
        """Decode object based on its respective `_encode` counterpart."""
        return cls.from_quat(obj_dict['quat'])

    @classmethod
    def _from_scipy_rotation(cls, sc_rotation):
        """"""
        instance = cls.__new__(cls)
        instance._rot = sc_rotation
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
        return self._rot.as_quat()

    def as_matrix(self):
        """"""
        return self._rot.as_matrix()

    def as_view_up_right(self):
        raise NotImplementedError

    def copy(self):
        """Return a deep copy of the Orientations object."""
        return self.from_quat(self.as_quat())

    def inv(self):
        raise NotImplementedError

    def mean(self, weights=None, axis=None):
        """"""
        return self._from_scipy_rotation(self._rot.mean(weights, axis))

    def reduce(self, left=None, right=None, return_indices=False):
        raise NotImplementedError

    def show(self, positions=None,
             show_views=True, show_ups=True, show_rights=True, **kwargs):
        raise NotImplementedError

    # private instance methods

