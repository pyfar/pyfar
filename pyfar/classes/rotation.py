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
    This class for Rotation in the three-dimensional space,
    is largely based on :py:class:`scipy:scipy.spatial.transform.Rotation` and
    wraps all functionality that
    :py:class:`scipy:scipy.spatial.transform.Rotation` provides.
    In addition the pyfar Rotation class adds the creation from perpendicular
    view and up vectors through :py:func:`~from_view_up`, the representation
    as view / up in :py:func:`~as_view_up` and a convenient plot
    function :py:func:`~show`.

    An orientation can be visualized with the triple of view, up and right
    vectors and it is tied to the object's local coordinate system.
    Alternatively the object's orientation can be illustrated with help of the
    right hand: Thumb (view), forefinger (up) and middle finger (right).

    Examples
    --------
    >>> from pyfar import Orientations
    >>> views = [[1, 0, 0], [2, 0, 0]]
    >>> ups = [[0, 1, 0], [0, -2, 0]]
    >>> orientations = Orientations.from_view_up(views, ups)

    Visualize orientations at certain positions:

    >>> positions = [[0, 0.5, 0], [0, -0.5, 0]]
    >>> orientations.show(positions)

    Rotate first element of orientations:

    >>> from scipy.spatial.transform import Rotation
    >>> rot_x45 = Rotation.from_euler('x', 45, degrees=True)
    >>> orientations[1] = orientations[1] * rot_x45
    >>> orientations.show(positions)

    To create `Orientations` objects use ``from_...`` methods.


    .. note::
        Class uses scipy.spatial.transform.Rotation for internal storage of
        rotations. Methods from scipy.spatial.transform.Rotation are wrapped
        in pyfar Rotation.
    """

    def __init__(self):
        raise RuntimeError("Rotation objects must be created using one of "
                           "the `from_...` methods")

    def __repr__(self):
        """String representation of Rotation object."""
        repr_string = \
              f"Pyfar Rotations object with {self.n_rotations} rotations."
        return repr_string

    def __iter__(self):
        """Iterate over rotations."""
        if self.as_quat().ndim == 1:
            raise TypeError("Single rotation is not iterable.")

        for i in range(self.n_rotations):
            rot = self._rot[i]
            yield self._from_scipy_rotation(rot)

    def __getitem__(self, idx):
        """"""
        return self._from_scipy_rotation(self._rot[idx])

    def __setitem__(self, *args):
        """"""
        raise NotImplementedError('Setting an item is disabled for pyfar '
        'Rotations. If you want to modify the Rotation, use an array '
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
        rot = self._rot.__pow__(n)
        return self._from_scipy_rotation(rot)

    def __eq__(self, other):
        """Check for equality of two objects."""
        return np.array_equal(self.as_quat(), other.as_quat())

    # properties
    @property
    def n_rotations(self):
        """"""
        quat = self._rot.as_quat()
        n_rotations = \
            1 if quat.ndim == 1 else quat.shape[0]
        return n_rotations

    # from-... methods
    @classmethod
    def from_davenport(cls, axes, order, angles, degrees=False):
        """"""
        rot = scRotation.from_davenport(
            axes, order, angles, degrees=degrees)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_euler(cls, seq, angles, degrees=False):
        """"""
        rot = scRotation.from_euler(
            seq, angles, degrees=degrees)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_matrix(cls, matrix):
        """"""
        rot = scRotation.from_matrix(matrix)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_mrp(cls, mrp):
        """"""
        rot = scRotation.from_mrp(mrp)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_quat(cls, quat):
        """"""
        rot = scRotation.from_quat(quat)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def from_rotvec(cls, rotvec, degrees=False):
        """"""
        rot = scRotation.from_rotvec(rotvec, degrees)
        return cls._from_scipy_rotation(rot)

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

        return cls.from_matrix(rotation_matrix)

    # other class methods
    @classmethod
    def align_vectors(cls, a, b, weights=None, return_sensitivity=False):
        """"""
        result = scRotation.align_vectors(a, b, weights, return_sensitivity)

        return (cls._from_scipy_rotation(result[0]), *result[1:])

    @classmethod
    def concatenate(cls, rotations):
        """"""
        if np.asarray(rotations).shape[0] == 1:
            return rotations[0].copy()

        rotations = [rotation._rot for rotation in rotations]

        rot = scRotation.concatenate(rotations)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def identity(cls, num=None, *, shape=None):
        """"""
        rot  = scRotation.identity(num, shape=shape)
        return cls._from_scipy_rotation(rot)

    @classmethod
    def random(cls, num=None, rng=None, *, shape=None):
        """"""
        rot = scRotation.random(num, rng, shape=shape)
        return cls._from_scipy_rotation(rot)

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

    def as_matrix(self):
        """"""
        return self._rot.as_matrix()

    def as_mrp(self):
        """"""
        return self._rot.as_mrp()

    def as_quat(self):
        """"""
        return self._rot.as_quat()

    def as_rotvec(self):
        """"""
        return self._rot.as_rotvec()

    def as_view_up(self):
        """"""
        vector_triple = self.as_matrix()
        views, lefts, ups = np.split(vector_triple, 3, axis=-2)

        # In a standard Cartesian right-handed coordinate system,
        # standard basis is defined as [x, y, z] = [view, left, up], where
        # left is the same vector as -rights
        vector_triple = np.concatenate((views, ups, -lefts), axis=-2)

        views = np.squeeze(views, axis=-2)
        ups   = np.squeeze(ups, axis=-2)

        return views, ups

    def copy(self):
        """Return a deep copy of the Orientations object."""
        return self.from_quat(self.as_quat())

    def inv(self):
        """"""
        self._rot = self._rot.inv()
        return self

    def mean(self, weights=None, axis=None):
        """"""
        return self._from_scipy_rotation(self._rot.mean(weights, axis))

    def reduce(self, left=None, right=None, return_indices=False):
        """"""
        if return_indices:
            rot, left_idx, right_idx = \
                self._rot.reduce(left, right, return_indices)
            return self._from_scipy_rotation(rot), left_idx, right_idx

        rot = self._rot.reduce(left, right, return_indices)
        return self._from_scipy_rotation(rot)

    def show(self, positions=None,
             show_views=True, show_ups=True, show_rights=True, **kwargs):
        """"""
        if positions is None:
            positions = np.zeros((self.as_quat().shape[0], 3))
        positions = np.atleast_2d(positions).astype(np.float64)
        if positions.shape[0] != self.as_quat().shape[0]:
            raise ValueError("If provided, there must be the same number"
                             "of positions as orientations.")

        # Create view, up and right vectors from Rotation object
        views, ups = self.as_view_up()
        rights = np.cross(views, ups)

        kwargs.pop('color', None)

        ax = None
        if show_views:
            ax = pf.plot.quiver(
                positions, views, color=pf.plot.color('r'), **kwargs)
        if show_ups:
            ax = pf.plot.quiver(
                positions, ups, ax=ax, color=pf.plot.color('g'), **kwargs)
        if show_rights:
            ax = pf.plot.quiver(
                positions, rights, ax=ax, color=pf.plot.color('b'), **kwargs)
